"""Exercise the real finite sender with filesystem receipts and Slurm wire output.

Primary owner of submission lifecycle, ambiguity, finite budgets and restart
behavior. Only external Slurm and runtime admission/reconciliation are replaced;
no fake decides admission, writes submission receipts or counts allocations.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_gemma_v4 as runtime
from analysis.scripts import horeka_gemma_v4_sender as sender


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Cluster:
    """External scheduler: native parsable rows, acknowledgements and job endings."""
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.now = 1800000000
        self.jobs, self.accounting, self.submissions, self.reconciled = {}, {}, [], []
        self.sleeps, self.commands = [], []
        self.submit_action = None
        self.end_action = None
        self.ticks = 0
        self.storage_error = None
        self.reservations = 0

    def queue_row(self, job, comment='', name='unrelated-qwen', state='PENDING'):
        self.jobs[job] = {'job': job, 'comment': comment, 'name': name, 'state': state}

    def command(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == 'squeue':
            return SimpleNamespace(returncode=0, stdout='\n'.join('|'.join([job, r['state'], r['comment'], r['name']]) for job, r in self.jobs.items()), stderr='')
        if command[0] == 'sacct':
            # --parsable2 has no trailing separator. Both live and terminal
            # allocation records can be present in accounting.
            return SimpleNamespace(returncode=0, stdout='\n'.join('|'.join([job, r['state'], r['comment'], r['name'], '600', '32', '1', 'cpu=32,gres/gpu=4,node=1', '2027-01-15T08:00:00', '2027-01-15T08:10:00', '0:0']) for job, r in self.accounting.items()), stderr='')
        assert command[0] == 'sbatch'
        shard = next(s for s in self.plan['shards'] if str(Path(s['directory']) / 'run.sh') == command[-1])
        comment = next(a.split('=', 1)[1] for a in command if a.startswith('--comment='))
        markers = list((Path(shard['directory']) / 'submissions').glob('*/intent.json'))
        intent = next(json.loads(p.read_text()) for p in markers if json.loads(p.read_text())['comment'] == comment)
        assert intent['command'] == command  # durable intent must precede the wire request
        assert {'--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=32', '--gres=gpu:4', '--mem=0', '--exclusive', '--time=05:00:00', '--no-requeue', '--partition=accelerated'} <= set(command)
        job = str(1000 + len(self.submissions))
        self.submissions.append({'job': job, 'shard': shard, 'comment': comment, 'at': self.now})
        if self.submit_action:
            answer = self.submit_action(self, self.submissions[-1])
            if answer is not None:
                return answer
        self.queue_row(job, comment, 'geodml-gemma-v4-bout')
        return SimpleNamespace(returncode=0, stdout=job+';horeka\n', stderr='')

    def result(self, shard, status='finished', *, done=1, pending=0, failures=0):
        output = Path(shard['directory']) / 'results'
        write(output / 'reports/writer/summary.json', {'status': status, 'states': {'done': done, 'pending': pending}, 'inference_failures': failures, 'counts': {'cells_complete': done}})
        write(output / 'reports/latest.json', {'directory': 'writer'})

    def end(self, job, state='COMPLETED'):
        row = self.jobs.pop(job)
        self.accounting[job] = {**row, 'state': state}

    def sleep(self, seconds):
        assert 0 < seconds <= 600
        self.sleeps.append(seconds)
        self.now += seconds
        self.ticks += 1
        if self.end_action:
            self.end_action(self)
            return
        for s in self.submissions:
            if s['job'] in self.jobs:
                self.result(s['shard'])
                self.end(s['job'])

    def reconcile(self, results):
        owned = [s for s in self.submissions if Path(s['shard']['directory']) / 'results' == results]
        assert owned and owned[-1]['job'] not in self.jobs
        assert self.accounting[owned[-1]['job']]['state'] in sender.TERMINAL
        self.reconciled.append(owned[-1]['job'])
        latest = results / 'reports/latest.json'
        if not latest.exists():
            return {'status': 'not_started', 'states': {}, 'counts': {}, 'inference_failures': 0}
        return json.loads((results / 'reports/writer/summary.json').read_text())


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    def make(count=1, maximum=2, deadline=3600):
        root = tmp_path / 'run'
        shards = []
        for index in range(count):
            directory = root / f'shard-{index:04d}'
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text('{}')
            (directory / 'run.sh').write_text('#!/bin/bash\n')
            shards.append({'id': str(index), 'directory': str(directory), 'cells': 2, 'max_allocations': maximum,
                           'config_sha256': hashlib.sha256(b'{}').hexdigest()})
        plan = {'format_version': 'gemma-v4-bouts-v1', 'git_commit': 'a'*40, 'repository': str(tmp_path),
                'workspace': str(tmp_path / 'workspace'), 'account': 'test', 'walltime': '05:00:00',
                'job_name': 'geodml-gemma-v4-bout', 'partition': 'accelerated', 'max_inflight': 200,
                'poll_seconds': 600, 'deadline_epoch': 1800000000+deadline, 'maximum_allocations': count*maximum,
                'shards': shards}
        write(root / 'plan.json', plan)
        c = Cluster(root, plan)
        monkeypatch.setattr(sender.subprocess, 'run', c.command)
        monkeypatch.setattr(sender.time, 'time', lambda: c.now)
        monkeypatch.setattr(sender.time, 'sleep', c.sleep)
        monkeypatch.setattr(runtime, 'checked_plan', lambda root: json.loads((root / 'plan.json').read_text()))
        def storage(*args):
            if c.storage_error:
                raise ValueError(c.storage_error)
            return {'safe_to_admit': True}
        monkeypatch.setattr(runtime, 'storage', storage)
        def reserve(*args):
            c.reservations += 1
        monkeypatch.setattr(runtime, 'reserve', reserve)
        monkeypatch.setattr(runtime, 'reconcile_output', c.reconcile)
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        return c
    return make


def run(c):
    return sender.main(['send', '--output', str(c.root)])


def summary(c):
    return json.loads((c.root / 'sender/summary.json').read_text())


def test_first_pass_fills_200_plan_slots_without_gating_on_foreign_queue(cluster):
    c = cluster(count=201, maximum=1)
    for index in range(250):
        c.queue_row(str(index+1), state='RUNNING' if index < 20 else 'PENDING')
    assert run(c) == 0
    assert len(c.submissions) == 201
    assert sum(s['at'] == 1800000000 for s in c.submissions) == 200
    assert c.submissions[-1]['at'] == 1800000600
    assert len(c.jobs) == 250  # foreign jobs preserved
    assert summary(c)['all_user_queued_or_active'] == 250
    assert summary(c)['allocations_attempted'] == 201
    assert c.sleeps == [600, 600]
    assert c.reservations == 1


@pytest.mark.parametrize('lost_reply', ['timeout', 'process_interrupted'])
def test_lost_acknowledgement_recovers_job_by_comment_without_duplicate(cluster, lost_reply):
    c = cluster()
    def lose(c, s):
        c.queue_row(s['job'], s['comment'], 'geodml-gemma-v4-bout')
        if lost_reply == 'timeout':
            raise subprocess.TimeoutExpired('sbatch', 120)
        raise KeyboardInterrupt
    c.submit_action = lose
    code = run(c)
    if lost_reply == 'process_interrupted':
        assert code == 130
        c.submit_action = None
        code = run(c)
    assert code == 0
    assert len(c.submissions) == 1
    receipt = json.loads(next(c.root.glob('shard-*/submissions/*/receipt.json')).read_text())
    assert receipt['job_id'] == '1000' and receipt['recovered_by_comment'] is True
    assert c.reconciled == ['1000']


def test_unresolved_submission_spends_budget_and_waits_only_until_finite_deadline(cluster):
    c = cluster(deadline=1200)
    c.submit_action = lambda c, s: SimpleNamespace(returncode=1, stdout='', stderr='connection lost')
    assert run(c) == 2
    assert len(c.submissions) == 1
    assert summary(c)['state'] == 'expired'
    assert summary(c)['ambiguous_or_missing_allocations'] == 1
    assert c.sleeps == [600, 600]


def test_scheduler_submit_limit_pauses_then_retries_without_spending_allocation_budget(cluster):
    c = cluster(count=2, maximum=1)
    def reject_once(c, s):
        if len(c.submissions) == 1:
            return SimpleNamespace(returncode=1, stdout='', stderr='sbatch: error: Batch job submission failed: QOSMaxSubmitJobPerUserLimit')
    c.submit_action = reject_once
    assert run(c) == 0
    assert [s['at'] for s in c.submissions] == [1800000000, 1800000600, 1800000600]
    assert summary(c)['allocations_attempted'] == 2
    assert len(list(c.root.glob('shard-*/submissions/*/intent.json'))) == 3


def test_generic_qos_policy_refusal_stops_with_reason_instead_of_retrying_for_days(cluster):
    c = cluster(count=2, maximum=1, deadline=1800)
    c.submit_action = lambda c, s: SimpleNamespace(returncode=1, stdout='',
        stderr='sbatch: error: Batch job submission failed: Job violates accounting/QOS policy')
    assert run(c) == 2
    assert len(c.submissions) == 1
    assert summary(c)['state'] == 'blocked'
    assert 'accounting/QOS policy' in summary(c)['reason']
    assert c.sleeps == []


def test_resume_waits_for_live_job_to_end_then_reconciles_and_keeps_terminal_failures(cluster):
    c = cluster(maximum=3)
    def endings(c):
        s = c.submissions[-1]
        if c.ticks == 1:
            c.result(s['shard'], 'incomplete', done=1, pending=1, failures=1)
            c.jobs[s['job']]['state'] = 'RUNNING'
            c.accounting[s['job']] = {**c.jobs[s['job']], 'state': 'TIMEOUT'}  # stale terminal accounting
        elif c.ticks == 2:
            c.end(s['job'], 'TIMEOUT')
        else:
            c.result(s['shard'], 'finished_with_failures', done=2, failures=1)
            c.end(s['job'])
    c.end_action = endings
    assert run(c) == 2
    assert [s['at'] for s in c.submissions] == [1800000000, 1800001200]
    assert c.reconciled == ['1000', '1001']
    assert summary(c)['state'] == 'finished_with_failures'
    assert summary(c)['shards'][0]['progress']['inference_failures'] == 1
    # Restarting an exhausted run never retries its terminal task failure.
    assert run(c) == 2
    assert len(c.submissions) == 2


@pytest.mark.parametrize('outcome,expected', [('no_results', 'blocked'), ('pending_after_budget', 'budget_exhausted')])
def test_failed_startup_and_finite_budget_stop_without_replacement_loop(cluster, outcome, expected):
    c = cluster(maximum=1)
    def end(c):
        s = c.submissions[-1]
        if outcome == 'pending_after_budget':
            c.result(s['shard'], 'incomplete', done=1, pending=1)
        c.end(s['job'], 'FAILED' if outcome == 'no_results' else 'TIMEOUT')
    c.end_action = end
    assert run(c) == 2
    assert len(c.submissions) == 1
    assert summary(c)['state'] == expected
    assert summary(c)['shards'][0]['state'] == expected


def test_storage_failure_preserves_live_allocations_and_submits_nothing(cluster):
    c = cluster()
    c.queue_row('45', state='RUNNING')
    c.storage_error = 'stale quota evidence'
    with pytest.raises(ValueError, match='stale quota'):
        run(c)
    assert not c.submissions and '45' in c.jobs
    assert summary(c)['state'] == 'blocked'
    assert not any(command[0] in {'scancel', 'scontrol'} for command in c.commands)


def test_workspace_sender_lock_prevents_a_second_controller(cluster):
    c = cluster()
    path = Path(c.plan['workspace']) / 'control/gemma-v4-sender.lock'
    path.parent.mkdir(parents=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='another Gemma sender'):
            run(c)
    assert not c.submissions


def test_expired_sender_does_not_allocate_and_status_is_read_only(cluster):
    c = cluster(deadline=0)
    assert run(c) == 2
    assert summary(c)['state'] == 'expired'
    assert not c.submissions and not c.reservations
    files = {p.relative_to(c.root): p.read_bytes() for p in c.root.rglob('*') if p.is_file()}
    assert sender.main(['status', '--output', str(c.root)]) == 0
    assert files == {p.relative_to(c.root): p.read_bytes() for p in c.root.rglob('*') if p.is_file()}


def test_conflicting_comment_matches_stop_before_any_new_submission(cluster):
    c = cluster()
    def interrupt(c, s):
        c.queue_row(s['job'], s['comment'], 'geodml-gemma-v4-bout')
        raise KeyboardInterrupt
    c.submit_action = interrupt
    assert run(c) == 130
    c.queue_row('2000', c.submissions[0]['comment'], 'geodml-gemma-v4-bout')
    with pytest.raises(ValueError, match='conflicting allocations'):
        run(c)
    assert len(c.submissions) == 1
    assert set(c.jobs) == {'1000', '2000'}


def test_restart_cannot_expand_the_frozen_sender_deadline(cluster):
    c = cluster(deadline=0)
    assert run(c) == 2
    plan = json.loads((c.root / 'plan.json').read_text())
    plan['deadline_epoch'] += 3600
    write(c.root / 'plan.json', plan)
    with pytest.raises(ValueError, match='frozen sender plan changed'):
        run(c)
    assert not c.submissions


def test_deadline_expiring_during_reservation_prevents_submission(cluster, monkeypatch):
    c = cluster(deadline=600)
    monkeypatch.setattr(runtime, 'reserve', lambda *args: setattr(c, 'now', c.now+700))
    assert run(c) == 2
    assert summary(c)['state'] == 'expired'
    assert not c.submissions and not c.sleeps


def test_empty_eligible_queue_preserves_input_blockage_and_does_not_claim_corpus_completion(cluster):
    c = cluster(count=0)
    c.plan.update(input_counts={'ready_cells': 0, 'blocked_cells': 12},
                  input_inventory=[{'dataset': 'frozen', 'rows': 12}],
                  excluded_diagnostics={'cells': 20})
    write(c.root / 'plan.json', c.plan)
    assert run(c) == 0
    result = summary(c)
    assert result['state'] == 'finished' and result['completion_scope'] == 'eligible_frozen_queue_only'
    assert result['input_counts'] == {'ready_cells': 0, 'blocked_cells': 12}
    assert result['input_inventory'] == [{'dataset': 'frozen', 'rows': 12}]
    assert result['excluded_diagnostics'] == {'cells': 20}
    assert result['scientific_result'] is False
    assert result['semantic_acceptance'] == result['full_corpus_completion'] == 'not_established'
    assert not c.submissions


def test_finished_job_without_accounted_comment_is_reconciled_and_resumed(cluster):
    # HoreKa sacct returns an empty Comment for finished jobs; the receipt's job ID still binds it.
    c = cluster(maximum=2)
    def end(c):
        s = c.submissions[-1]
        if len(c.submissions) == 1:
            c.result(s['shard'], 'incomplete', done=1, pending=1)
            c.end(s['job'], 'FAILED')
            c.accounting[s['job']]['comment'] = ''
        else:
            c.result(s['shard'], 'finished', done=2)
            c.end(s['job'])
            c.accounting[s['job']]['comment'] = ''
    c.end_action = end
    assert run(c) == 0
    assert c.reconciled == ['1000', '1001']
    assert len(c.submissions) == 2
    assert summary(c)['shards'][0]['state'] == 'exhausted'


@pytest.mark.parametrize('where', ['accounting', 'queue'])
def test_known_job_with_a_different_comment_still_stops(cluster, where):
    c = cluster()
    def end(c):
        s = c.submissions[-1]
        if where == 'queue':
            c.jobs[s['job']]['comment'] = 'geodml-gemma-v4:' + 'f' * 32
        else:
            c.end(s['job'], 'FAILED')
            c.accounting[s['job']]['comment'] = 'geodml-gemma-v4:' + 'f' * 32
    c.end_action = end
    with pytest.raises(ValueError, match='no longer matches its submission comment'):
        run(c)



def test_small_map_recovery_uses_its_own_lock_and_a_large_one_is_refused(cluster):
    c = cluster()
    main = Path(c.plan['workspace']) / 'control/gemma-v4-sender.lock'
    main.parent.mkdir(parents=True)
    plan = json.loads((c.root / 'plan.json').read_text())
    plan['recovery_of'] = 'gemma-v4-main'
    write(c.root / 'plan.json', plan)
    with main.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the main sender is running
        assert run(c) == 0  # the recovery still sends and finishes
    assert c.submissions
    plan['shards'] = [dict(plan['shards'][0], id=str(i)) for i in range(sender.RECOVERY_MAX_SHARDS + 1)]
    with pytest.raises(ValueError, match="at most"):
        sender.lock_name(plan)
