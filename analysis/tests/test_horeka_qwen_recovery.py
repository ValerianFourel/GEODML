"""Recovery operates on real saved records; scheduler and Hub stay external."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import pytest

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import horeka_qwen_recovery as recovery
from analysis.scripts import horeka_qwen_bouts as bouts


def fixture(tmp_path):
    dataset, source, root = tmp_path / 'dataset', tmp_path / 'original', tmp_path / 'recovery'
    initialize_dataset(dataset, population_id='frozen', acceptance_policy_id='v2')
    writer = FinalDatasetWriter(dataset, writer_id='registration')
    writer.append('keyword_memberships', {'prompt_id': 'p', 'primary_keyword_id': 'k',
                                         'primary_priority_rank': 1}, transaction_id='membership')
    ids = {}
    for name in ['saved', 'missing', 'remote', 'terminal', 'uncommitted', 'active']:
        ident = ClaimIdentity(task_id=name, model_id='qwen-model', model_revision='fixed', protocol='fixed',
                              request_sha256=hashlib.sha256(name.encode()).hexdigest())
        ids[name] = ident
        writer.append('task_definitions', {'task_id': name, 'prompt_id': 'p', 'model': 'qwen38',
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
