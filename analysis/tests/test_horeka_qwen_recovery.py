"""Recovery operates on real saved records; scheduler and Hub stay external."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import horeka_qwen_recovery as recovery
from analysis.scripts import horeka_qwen_bouts as bouts


def fixture(tmp_path, *, extra=0):
    dataset, source, root = tmp_path / 'dataset', tmp_path / 'original', tmp_path / 'recovery'
    initialize_dataset(dataset, population_id='frozen', acceptance_policy_id='v2')
    writer = FinalDatasetWriter(dataset, writer_id='registration')
    writer.append('keyword_memberships', {'prompt_id': 'p', 'primary_keyword_id': 'k',
                                         'primary_priority_rank': 1}, transaction_id='membership')
    ids = {}
    for name in ['saved', 'missing', 'remote', 'terminal', 'uncommitted', 'active'] + [f'extra-{n}' for n in range(extra)]:
        prompt = 'p'
        if name.startswith('extra-'):
            n = int(name.split('-')[1])
            prompt = f'extra-p{n // 12:04d}'
            if n % 12 == 0:
                writer.append('keyword_memberships', {'prompt_id': prompt, 'primary_keyword_id': 'k',
                    'primary_priority_rank': 1}, transaction_id=prompt)
        ident = ClaimIdentity(task_id=name, model_id='qwen-model', model_revision='fixed', protocol='fixed',
                              request_sha256=hashlib.sha256(name.encode()).hexdigest())
        ids[name] = ident
        writer.append('task_definitions', {'task_id': name, 'prompt_id': prompt, 'model': 'qwen38',
            'claim_identity': asdict(ident), 'runnable_task': {'cell_id': name}}, transaction_id=name)
    writer.seal()
    ledger = StripedTaskLedger(dataset / 'control/task-ledger')
    saved = ledger.claim(ids['saved'], owner_id='horeka-bout0001-job42').claim
    response = FinalDatasetWriter(dataset, writer_id=saved.owner_id)
    ref = response.append('generations', {'answer': 'preserve this'}, transaction_id=saved.fingerprint)
    response.close()
    ledger.transition(saved, state='result_saved', record_references=[ref])
    claim = ledger.claim(ids['terminal'], owner_id='horeka-bout0001-job42').claim
    ledger.transition(claim, state='terminal_failed')
    ledger.claim(ids['uncommitted'], owner_id='horeka-bout0001-job42')
    ledger.claim(ids['active'], owner_id='unknown-owner')
    plan = {'completed_before_plan': [], 'deferred': {identity_fingerprint(i): 'awaiting_calibration' for i in ids.values()}}
    recovery.save(tmp_path / 'plan.json', plan)
    bout = source / 'bouts/bout-0001'
    bout.mkdir(parents=True)
    recovery.save(bout / 'submission.json', {'returncode': 0, 'stdout': '42\n', 'stderr': ''})
    recovery.save(source / 'division.json', {'bouts': [{'number': 1}]})
    recovery.save(bout / 'attempts/job42/bout-result.json', {'job_id': '42', 'status': 'complete',
        'returncode': 0, 'elapsed_seconds': 1260, 'direct_dataset': {'committed_count': 30, 'reused_count': 0}})
    context = {'dataset_root': str(dataset), 'source_division': str(source), 'plan': str(tmp_path / 'plan.json')}
    remote = {'revision': 'fixed', 'states': {identity_fingerprint(ids['remote']): 'completed'}, 'hf_registered_qwen': []}
    snapshot = {'complete': True, 'jobs': [], 'owners': [
        {'owner_id': 'horeka-bout0001-job42', 'job_id': '42', 'state': 'COMPLETED'}]}
    (root / 'preparation').mkdir(parents=True)
    return root, context, remote, snapshot, ledger, ids


def test_recovery_preserves_saved_answers_and_freezes_only_missing_eligible_cells(tmp_path):
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path)
    original = (Path(context['source_division']) / 'division.json').read_bytes()
    report = recovery.freeze_report(context, root / 'preparation', remote, snapshot)
    assert report['registered_qwen'] == 6
    assert (report['verified_completed'], report['terminal_failed'], report['eligible_missing'], report['blocked_or_unverified']) == (2, 1, 2, 1)
    assert report['complete'] is False
    assert ledger.inspect(ids['saved'])['state'] == 'completed'
    assert ledger.inspect(ids['uncommitted'])['state'] == 'checkpointed'
    assert ledger.inspect(ids['active'])['state'] == 'claimed'
    cells = [json.loads(line)['cell_id'] for path in (root / 'division').glob('bouts/*/cells.jsonl') for line in path.read_text().splitlines()]
    assert set(cells) == {'missing', 'uncommitted'}
    ready = recovery.read(root / 'preparation/ready.json')
    assert ready['maximum_allocations'] == 1
    assert bouts.walltime_seconds(ready['sizing']['walltime']) <= 3600
    assert (Path(context['source_division']) / 'division.json').read_bytes() == original


def test_hf_registered_missing_work_cannot_enter_a_legacy_recovery_division(tmp_path):
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path)
    remote['hf_registered_qwen'] = [identity_fingerprint(ids['missing'])]
    with pytest.raises(ValueError, match='HF hour plans'):
        recovery.freeze_report(context, root / 'preparation', remote, snapshot)
    assert not (root / 'division').exists()


@pytest.mark.parametrize('live,known', [(True, True), (False, False)])
def test_original_live_or_unknown_job_blocks_recovery_before_claim_changes(tmp_path, live, known):
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path)
    if live:
        snapshot['jobs'] = [{'job_id': '42', 'state': 'RUNNING'}]
    if not known:
        snapshot['owners'] = []
    with pytest.raises(ValueError, match='live or lacks terminal'):
        recovery.freeze_report(context, root / 'preparation', remote, snapshot)
    assert ledger.inspect(ids['saved'])['state'] == 'result_saved'
    assert ledger.inspect(ids['uncommitted'])['state'] == 'claimed'


@pytest.mark.parametrize('change', ['five_active', 'pending', 'recent_start', 'storage'])
def test_admission_refuses_capacity_start_gap_and_storage_failures(change):
    now = int(time.time())
    snapshot = {'complete': True, 'cluster': 'horeka', 'captured_at_epoch': now, 'jobs': [], 'owners': []}
    health = {'cluster': 'horeka', 'captured_at_epoch': now, 'safe_to_admit': True, 'quota_verified': True}
    if change == 'five_active':
        snapshot['jobs'] = [{'job_id': str(n), 'state': 'RUNNING', 'start_epoch': now - 700} for n in range(5)]
    elif change == 'pending':
        snapshot['jobs'] = [{'job_id': '12', 'state': 'PENDING', 'held': False}]
    elif change == 'recent_start':
        snapshot['owners'] = [{'job_id': '12', 'state': 'COMPLETED', 'start_epoch': now - 30}]
    else:
        health['quota_verified'] = False
    with pytest.raises(ValueError):
        recovery.admit({}, snapshot, health, 'next')


def test_ambiguous_submission_prevents_subsequent_jobs(tmp_path):
    bout = tmp_path / 'division/bouts/bout-0001'
    bout.mkdir(parents=True)
    (bout / 'SUBMISSION_ATTEMPTED').write_text('durable intent')
    with pytest.raises(ValueError, match='ambiguous'):
        recovery.confirmed_submissions(tmp_path, {'jobs': [], 'owners': []})
    recovery.save(bout / 'submission.json', {'returncode': 0, 'stdout': '42\n'})
    with pytest.raises(ValueError, match='confirmed accounting'):
        recovery.confirmed_submissions(tmp_path, {'jobs': [], 'owners': []})
    assert recovery.confirmed_submissions(tmp_path, {'jobs': [{'job_id': '42'}], 'owners': []}) == ['42']


def test_completed_recovery_report_is_distinct_from_terminal_failure(tmp_path):
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path)
    remote['states'] = {identity_fingerprint(i): 'completed' for name, i in ids.items() if name != 'terminal'}
    report = recovery.freeze_report(context, root / 'preparation', remote, snapshot, final=True)
    assert report['verified_completed'] == 5 and report['terminal_failed'] == 1
    assert report['eligible_missing'] == 0 and report['complete'] is False


@pytest.mark.parametrize('ceiling,allocations', [(3600, 8), (18000, 2)])
def test_authorized_walltime_sizes_a_finite_division_from_verified_remaining_cells(tmp_path, ceiling, allocations):
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path, extra=600)
    context['gpu_walltime_seconds'] = ceiling
    recovery.freeze_report(context, root / 'preparation', remote, snapshot)
    ready = recovery.read(root / 'preparation/ready.json')
    assert ready['coverage']['eligible_missing'] == 602
    assert ready['maximum_allocations'] == allocations
    assert bouts.walltime_seconds(ready['sizing']['walltime']) == ceiling
    assert ready['totals']['gpu_hours'] == allocations * ceiling / 3600 * 4
    records = [recovery.read(p) for p in (root / 'division/bouts').glob('*/bout.json')]
    primary = [fp for row in records for fp in row['primary_fingerprints']]
    assert len(primary) == len(set(primary)) == 602
    assert identity_fingerprint(ids['saved']) not in primary


def sender_fixture(tmp_path, monkeypatch, *, submit_result=None, extra=0, walltime=18000):
    """Real controller, division, receipts and CPU audits; external Slurm/Hub/model files only."""
    from analysis.scripts import manage_agentic_hours, prepare_horeka_qwen, prepare_shared_hour_inputs
    root, context, remote, snapshot, ledger, ids = fixture(tmp_path, extra=extra)
    workspace = tmp_path / 'workspace'
    runtime = workspace / 'environment/qwen-runtime/bin/python'
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    manifests = Path(context['dataset_root']) / 'artifacts/shared-preparations'
    recovery.save(manifests / 'manifest.json', {'models': {'qwen38': {'reference_runtime': {}}}})
    context.update(format_version=recovery.FORMAT, workspace=str(workspace), account='test-account',
                   git_commit='a' * 40, source_division_sha256=recovery.file_hash(Path(context['source_division']) / 'division.json'),
                   plan_sha256=recovery.digest(recovery.read(context['plan'])), gpu_walltime_seconds=walltime,
                   since='2026-09-25', authorization='test finite five-hour sweep')
    recovery.save(root / 'recovery.json', context)
    clock = SimpleNamespace(now=1800000000)
    monkeypatch.setattr(recovery.time, 'time', lambda: clock.now)
    snapshot.update(cluster='horeka', captured_at_epoch=clock.now)
    def capture(**kwargs):
        snapshot['captured_at_epoch'] = clock.now
        return snapshot
    monkeypatch.setattr(recovery, 'capture', capture)
    monkeypatch.setattr(recovery.subprocess, 'check_output',
                        lambda command, **kw: 'a' * 40 if 'rev-parse' in command else '')
    def hub(context, directory):
        recovery.save(directory / 'hub-snapshot.json', remote)
        return remote
    monkeypatch.setattr(recovery, 'remote_snapshot', hub)
    def health(*args):
        return {'cluster': 'horeka', 'captured_at_epoch': clock.now, 'safe_to_admit': True, 'quota_verified': True}
    monkeypatch.setattr(recovery, 'storage', health)
    monkeypatch.setattr(prepare_horeka_qwen, 'capture_quota', lambda *a: {})
    monkeypatch.setattr(manage_agentic_hours, 'health', health)
    monkeypatch.setattr(prepare_horeka_qwen, 'snapshot', lambda *a: workspace)
    monkeypatch.setattr(prepare_shared_hour_inputs, 'verify_inputs',
                        lambda *a, **kw: {'files': {'SEARCH_AGENTIC_PROFILE': 'frozen.json'}})
    calls = []
    def sbatch(command, **kwargs):
        assert command[0] == 'sbatch'
        calls.append((clock.now, command))
        if submit_result is not None:
            return submit_result
        job = str(1000 + len(calls))
        snapshot['jobs'].append({'job_id': job, 'state': 'PENDING', 'held': False,
            'job_name': next(s.split('=', 1)[1] for s in command if s.startswith('--job-name='))})
        return SimpleNamespace(returncode=0, stdout=job + '\n', stderr='')
    monkeypatch.setattr(recovery.subprocess, 'run', sbatch)
    def sleep(seconds):
        assert 0 < seconds <= 60
        clock.now += seconds
        for row in list(snapshot['jobs']):
            row.update(state='RUNNING', start_epoch=clock.now)
            if 'recovery-' in row['job_name']:
                phase = row['job_name'].removeprefix('geodml-qwen-recovery-')
                monkeypatch.setenv('SLURM_JOB_ID', row['job_id'])
                try:
                    recovery.main(['cpu', '--output', str(root), '--phase', phase])
                finally:
                    monkeypatch.delenv('SLURM_JOB_ID')
                state = 'COMPLETED'
            else:
                state = 'TIMEOUT'  # Finite sender must audit the remaining work, never retry this allocation.
            snapshot['jobs'].remove(row)
            snapshot['owners'].append({**row, 'state': state, 'owner_id': 'job' + row['job_id'] + '-worker0'})
    monkeypatch.setattr(recovery.time, 'sleep', sleep)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    return root, context, snapshot, clock, calls


def test_sender_runs_prepare_gpu_and_final_audit_once_then_reports_remaining_cells(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch)
    assert recovery.send(root) == 2
    assert len(calls) == 3
    assert [next(s for s in command if s.startswith('--job-name=')) for _, command in calls] == [
        '--job-name=geodml-qwen-recovery-preparation', '--job-name=geodml-qwen-bout-0001',
        '--job-name=geodml-qwen-recovery-final']
    assert all(calls[n + 1][0] - calls[n][0] >= 660 for n in (0, 1))
    report = recovery.read(root / 'final/coverage.json')
    assert report['eligible_missing'] == 2 and report['verified_completed'] == 2
    assert recovery.read(root / 'sender-status.json')['state'] == 'finished'
    assert recovery.send(root) == 2
    assert len(calls) == 3  # Restart preserves all three allocations, including the timed-out GPU job.


@pytest.mark.parametrize('returncode,stdout', [(0, ''), (1, '')])
def test_sender_stops_on_ambiguous_or_refused_submission_and_never_retries(tmp_path, monkeypatch, returncode, stdout):
    result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr='submission unavailable')
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch, submit_result=result)
    with pytest.raises(ValueError, match='CPU submission not confirmed'):
        recovery.send(root)
    with pytest.raises(ValueError, match='failed or is ambiguous'):
        recovery.send(root)
    assert len(calls) == 1
    assert (root / 'preparation/SUBMISSION_ATTEMPTED').exists()
    assert recovery.read(root / 'sender-status.json')['state'] == 'blocked'


def test_sender_deadline_is_durable_and_never_extended_on_restart(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch)
    snapshot['jobs'] = [{'job_id': '42', 'state': 'RUNNING'}]
    recovery.save(root / 'sender.json', {'deadline_epoch': clock.now + 120, 'poll_seconds': 60})
    monkeypatch.setattr(recovery.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    assert recovery.send(root) == 3
    assert recovery.send(root) == 3
    assert not calls
    assert snapshot['jobs'] == [{'job_id': '42', 'state': 'RUNNING'}]
    assert recovery.read(root / 'sender-status.json')['state'] == 'expired'


def test_prior_frozen_recovery_is_preserved_and_blocks_overlapping_sweep(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch)
    previous = tmp_path / 'previous'
    recovery.save(previous / 'recovery.json', {'dataset_root': context['dataset_root']})
    recovery.save(previous / 'preparation/ready.json', {'maximum_allocations': 1})
    before = (previous / 'preparation/ready.json').read_bytes()
    context['previous_recovery'] = str(previous)
    recovery.save(root / 'recovery.json', context)
    with pytest.raises(ValueError, match='previous recovery still has unsubmitted work'):
        recovery.send(root)
    assert not calls
    assert (previous / 'preparation/ready.json').read_bytes() == before


def test_sender_refuses_changing_a_frozen_walltime_or_waiting_through_storage_failure(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch)
    before = (root / 'recovery.json').read_bytes()
    with pytest.raises(ValueError, match='wall-time is frozen'):
        recovery.main(['send', '--output', str(root), '--walltime', '01:00:00'])
    assert (root / 'recovery.json').read_bytes() == before
    monkeypatch.setattr(recovery, 'storage', lambda *a: {'cluster': 'horeka', 'captured_at_epoch': clock.now,
                                                       'safe_to_admit': False, 'quota_verified': True})
    with pytest.raises(ValueError, match='storage admission blocked'):
        recovery.send(root)
    assert not calls
    assert recovery.read(root / 'sender-status.json')['state'] == 'blocked'


def test_restarted_sender_waits_for_the_cpu_audit_operator_lock(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch)
    recovery.save(root / 'sender.json', {'deadline_epoch': clock.now + 120, 'poll_seconds': 60})
    monkeypatch.setattr(recovery.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    with recovery.locked(root):
        assert recovery.main(['send', '--output', str(root), '--walltime', '05:00:00']) == 3
    assert not calls
    assert recovery.read(root / 'sender-status.json')['state'] == 'expired'


def test_all_at_once_override_submits_every_frozen_bout_without_cap_or_start_gap(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch, extra=600, walltime=3600)
    monkeypatch.setattr(recovery, 'admit', lambda context, snapshot, health, attempt:
                        {'admission': 'admitted'} if 'bout' not in attempt and not attempt[-1].isdigit()
                        else pytest.fail('GPU bouts must not use the capped admission under the override'))
    recovery.save(root / recovery.ALL_AT_ONCE, {'authorization': 'test'})
    assert recovery.send(root) == 2
    names = [next(s for s in command if s.startswith('--job-name=')) for _, command in calls]
    gpu = [(at, name) for (at, _), name in zip(calls, names) if 'bout-' in name]
    assert [name for _, name in gpu] == [f'--job-name=geodml-qwen-bout-{n:04d}' for n in range(1, 9)]
    assert len({at for at, _ in gpu}) == 1  # one sender step, no start gap, despite pending jobs
    assert names[0].endswith('recovery-preparation') and names[-1].endswith('recovery-final')
    assert recovery.send(root) == 2 and len(calls) == len(names)  # restart never resubmits


def test_without_override_only_one_bout_is_admitted_per_step(tmp_path, monkeypatch):
    root, context, snapshot, clock, calls = sender_fixture(tmp_path, monkeypatch, extra=600, walltime=3600)
    assert recovery.send(root) == 2
    starts = [at for at, command in calls if any('bout-' in s for s in command if s.startswith('--job-name='))]
    assert len(starts) == 8 and all(b - a >= 600 for a, b in zip(starts, starts[1:]))


def test_controller_upgrade_accepts_only_descendant_controller_only_changes(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / 'repo'
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
    repo.mkdir()
    git('init', '-q')
    git('config', 'user.email', 't@example.com')
    git('config', 'user.name', 't')
    for path in (*recovery.CONTROLLER_FILES, 'analysis/scripts/horeka_qwen_bouts.py'):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text('v1\n')
    git('add', '-A')
    git('commit', '-qm', 'pinned')
    old = git('rev-parse', 'HEAD')
    (repo / recovery.CONTROLLER_FILES[0]).write_text('v2\n')
    git('commit', '-qam', 'controller only')
    controller = git('rev-parse', 'HEAD')
    (repo / 'analysis/scripts/horeka_qwen_bouts.py').write_text('v2\n')
    git('commit', '-qam', 'executor change')
    executor = git('rev-parse', 'HEAD')
    monkeypatch.setattr(recovery, 'REPO', repo)
    record = recovery.controller_upgrade(old, controller)
    assert record['changed_files'] == [recovery.CONTROLLER_FILES[0]]
    with pytest.raises(ValueError, match='may only change'):
        recovery.controller_upgrade(old, executor)
    with pytest.raises(ValueError, match='may only change'):
        recovery.controller_upgrade(controller, old)  # not a descendant
    root = tmp_path / 'run'
    root.mkdir()
    assert not recovery.upgraded_controller(root, old, controller)  # nothing recorded
    recovery.save(root / recovery.ALL_AT_ONCE, {'controller_upgrade': record})
    assert recovery.upgraded_controller(root, old, controller)
    assert not recovery.upgraded_controller(root, old, executor)
