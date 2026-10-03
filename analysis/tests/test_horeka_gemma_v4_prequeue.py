"""First-wave lifecycle tests at Slurm's boundary, using native sender receipts.

The existing sender tests cannot cover jobs accepted before a frozen plan exists.
These protect no-GPU-before-validation, adoption without duplicate submission,
ambiguous reply recovery, and cancellation limited to surplus held slots.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_gemma_v4 as runtime
from analysis.scripts import horeka_gemma_v4_prequeue as prequeue
from analysis.scripts import horeka_gemma_v4_sender as sender
from analysis.tests.test_horeka_gemma_v4_sender import cluster, write


@pytest.fixture
def wave(cluster, monkeypatch):
    def make(count=200):
        c = cluster(count=count, maximum=1)
        c.plan['workspace'] = str(c.root.parent)
        write(c.root / 'plan.json', c.plan)
        c.holds, c.released, c.cancelled = set(), [], []
        c.loss = False
        c.reject_after = None
        c.release_loss = False
        c.prep_failure = False
        c.prep_job = '999'
        c.queue_row('999', name='geodml-gemma-v4-prepare')
        c.queue_row('998', name='unrelated-qwen', state='RUNNING')
        spec = {'root': str(c.root), 'workspace': c.plan['workspace'], 'account': 'test',
                'git_commit': 'a'*40, 'created_at_epoch': c.now,
                'preparation_deadline_epoch': c.now + 7200}
        write(c.root / 'preparation.json', spec)
        write(c.root / 'preparation-submission.json', {'stdout': '999\n', 'returncode': 0})
        monkeypatch.setattr(runtime, 'clean_pin', lambda: 'a'*40)
        monkeypatch.setattr(runtime, 'REPO', Path(c.plan['repository']))
        original_command = c.command
        def wire(cmd, **kwargs):
            if cmd[0] == 'sbatch' and '--hold' in cmd:
                if c.reject_after is not None and len(c.submissions) >= c.reject_after:
                    return SimpleNamespace(returncode=1, stdout='', stderr='QOSMaxSubmitJobPerUserLimit')
                assert '--dependency=afterok:999' in cmd
                assert '--time=05:00:00' in cmd and '--gres=gpu:4' in cmd
                folder = Path(cmd[-1]).parent
                intents = list(folder.glob('submissions/*/intent.json'))
                intent = json.loads(intents[-1].read_text())
                assert intent['command'] == cmd  # durable before external acceptance
                job = str(1000 + len(c.submissions))
                c.submissions.append({'job': job, 'slot': folder, 'at': c.now, 'comment': intent['comment'],
                                      'shard': {'directory': str(folder)}})
                c.queue_row(job, intent['comment'], 'geodml-gemma-v4-bout')
                c.holds.add(job)
                if c.loss:
                    c.loss = False
                    raise KeyboardInterrupt
                return SimpleNamespace(returncode=0, stdout=job+'\n', stderr='')
            if cmd[:3] == ['scontrol', 'show', 'job']:
                job = cmd[3]
                r = c.jobs[job]
                reason = 'JobHeldUser' if job in c.holds else 'Priority'
                row = {'JobId': job, 'Account': 'test', 'Partition': 'accelerated',
                       'UserId': f'user({os.getuid()})', 'Comment': r['comment'], 'JobName': r['name'],
                       'TimeLimit': '05:00:00', 'NumNodes': '1-1', 'NumCPUs': '32',
                       'OverSubscribe': 'NO', 'TresPerNode': 'gres/gpu:4',
                       'JobState': r['state'], 'Reason': reason}
                return SimpleNamespace(returncode=0, stdout=' '.join(f'{k}={v}' for k,v in row.items()), stderr='')
            if cmd[:2] == ['scontrol', 'release']:
                job = cmd[2]
                assert job in c.holds and c.reservations > 0
                assert c.accounting['999']['state'] == 'COMPLETED'
                attempt = next(p for p in c.root.glob('shard-*/submissions/*/receipt.json')
                               if json.loads(p.read_text())['job_id'] == job)
                shard = next(s for s in c.plan['shards'] if Path(s['directory']) == attempt.parents[2])
                # Exercise real worker ownership validation using the adopted receipt.
                runtime.bind_submission(c.root, shard, job, c.jobs[job]['comment'])
                submission = next(s for s in c.submissions if s['job'] == job)
                submission['shard'] = shard
                assert (submission['slot'] / 'assigned.sh').exists()
                c.holds.remove(job)
                c.released.append(job)
                if c.release_loss:
                    c.release_loss = False
                    raise KeyboardInterrupt
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            if cmd[0] == 'scancel':
                job = cmd[1]
                assert job in c.holds and job not in ('998', '999')
                c.cancelled.append(job)
                c.holds.remove(job)
                c.end(job, 'CANCELLED')
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            return original_command(cmd, **kwargs)
        monkeypatch.setattr(sender.subprocess, 'run', wire)
        def tick(c):
            if '999' in c.jobs:
                assert not c.released
                c.end('999', 'FAILED' if c.prep_failure else 'COMPLETED')
                return
            for s in c.submissions:
                if s['job'] in c.jobs and s['job'] not in c.holds:
                    c.result(s['shard'])
                    c.end(s['job'])
        c.end_action = tick
        return c
    return make


def launch(c):
    return prequeue.run(c.root, Path(c.plan['repository']), 'b'*40, runtime, sender)


def test_200_held_jobs_are_adopted_then_refill_sender_observes_without_resubmitting(wave):
    c = wave()
    assert launch(c) == 0
    assert len(c.submissions) == len(c.released) == 200
    assert {s['at'] for s in c.submissions} == {1800000000}
    assert not c.cancelled and not c.holds
    assert c.sleeps == [600, 600]
    assert set(c.jobs) == {'998'}
    summary = sender.read(c.root / 'sender/summary.json')
    assert summary['allocations_attempted'] == summary['maximum_allocations'] == 200
    assert summary['state'] == 'finished'


@pytest.mark.parametrize('failure', ['submit_reply', 'release_reply'])
def test_restart_reconciles_accepted_jobs_and_partial_release(wave, failure):
    c = wave(2)
    if failure == 'submit_reply':
        c.loss = True
    else:
        c.release_loss = True
    with pytest.raises(KeyboardInterrupt):
        launch(c)
    assert launch(c) == 0
    assert len(c.submissions) == 200
    assert len(c.released) == 2
    assert len(set(c.released)) == 2
    assert len(c.cancelled) == 198
    assert set(c.jobs) == {'998'}


def test_preparation_failure_retires_only_held_first_wave(wave):
    c = wave()
    c.prep_failure = True
    with pytest.raises(ValueError, match='preparation failed'):
        launch(c)
    assert len(c.cancelled) == 200
    assert not c.released
    assert set(c.jobs) == {'998'}
    assert not list(c.root.glob('shard-*/submissions/*/intent.json'))


def test_capacity_rejection_keeps_accepted_slots_and_normal_sender_fills_missing_shards(wave):
    c = wave(3)
    c.reject_after = 2
    assert launch(c) == 0
    assert len(c.submissions) == 3
    assert len(c.released) == 2
    assert sender.read(c.root / 'sender/summary.json')['allocations_attempted'] == 3


def test_storage_failure_prevents_first_submission(wave):
    c = wave()
    c.storage_error = 'quota stale'
    with pytest.raises(ValueError, match='quota stale'):
        launch(c)
    assert not c.submissions


def test_failed_release_admission_leaves_all_jobs_held(wave):
    c = wave(2)
    old_tick = c.end_action
    def tick(c):
        old_tick(c)
        c.storage_error = 'quota exceeded'
    c.end_action = tick
    with pytest.raises(ValueError, match='quota exceeded'):
        launch(c)
    assert len(c.holds) == 200
    assert not c.released and not c.cancelled


def test_unknown_submission_acknowledgement_never_submits_duplicate(wave):
    c = wave(2)
    c.loss = True
    with pytest.raises(KeyboardInterrupt):
        launch(c)
    c.jobs.pop('1000')  # neither squeue nor sacct can establish its identity
    with pytest.raises(ValueError, match='ambiguous submission'):
        launch(c)
    assert len(c.submissions) == 1
