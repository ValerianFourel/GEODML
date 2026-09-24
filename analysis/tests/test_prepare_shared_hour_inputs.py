"""Freeze registered model populations without creating new scientific work."""
import json
import time
from dataclasses import replace

import pytest

from analysis.interpretability.pipeline.agentic_dataset import initialize_dataset
from analysis.interpretability.pipeline.agentic_hours import inventory
from analysis.scripts import bootstrap_shared_hours as bootstrap
from analysis.scripts.prepare_shared_hour_inputs import stage
from analysis.scripts.register_agentic_dataset_tasks import register_generator_tasks
from analysis.scripts.search_vllm_stage import build_profile
from analysis.tests.test_agentic_hours import complete
from analysis.tests.test_register_agentic_dataset_tasks import (
    _generator_inputs,
    _priority,
)


def test_pending_legacy_job_can_publish_tracker_without_reconciliation(tmp_path, monkeypatch, capsys):
    from analysis.tests.test_agentic_hours import MemoryHub
    root, _, _ = fixture(tmp_path)
    hub = MemoryHub()
    jobs = [{'job_id': '1995245', 'state': 'PENDING'}]
    monkeypatch.setattr(bootstrap, 'capture', lambda **kw: {
        'complete': True, 'jobs': jobs, 'captured_at_epoch': time.time()})
    monkeypatch.setattr(bootstrap, 'HubStore', lambda *a: hub)
    def forbidden(*a, **kw):
        pytest.fail('tracker must not reconcile, stage inputs or plan work')
    for name in ('reconcile', 'stage', 'replan'):
        monkeypatch.setattr(bootstrap, name, forbidden)
    args = ['--source', str(root), '--output', str(tmp_path / 'out'),
            '--since', '2026-09-01', '--stripes', '4', '--publish-tracker']
    for _ in range(2):
        assert bootstrap.main(args) == 0
        assert json.loads(capsys.readouterr().out)['runnable_hours'] == 0
        report = json.loads(hub.read('coordination/progress.json', hub.head()))
        assert report['cells'] == {'awaiting_reconciliation': 48}
        assert report['scheduler_snapshot']['jobs'] == jobs
        assert report['runnable'] is False
        state = json.loads(hub.read('coordination/hours.json', hub.head()))
        assert state['hours'] == {} and state['current_plan'] is None
    hub.commit(hub.head(), {'coordination/hours.json': json.dumps({**state, 'current_plan': 'existing'}).encode()}, 'plan')
    with pytest.raises(ValueError, match='already in use'):
        bootstrap.main(args)


def test_tracker_rejects_incomplete_or_stale_scheduler(tmp_path):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.tests.test_agentic_hours import MemoryHub
    hub = MemoryHub()
    for snapshot in ({'cluster': 'jupiter', 'complete': False, 'captured_at_epoch': time.time()},
                     {'cluster': 'jupiter', 'complete': True, 'captured_at_epoch': 0}):
        with pytest.raises(ValueError, match='fresh complete'):
            bootstrap.publish_tracker(tmp_path, Exchange(hub, tmp_path / 'journal'), snapshot)
    assert hub.head() == '0'


def fixture(tmp_path):
    root = tmp_path / 'source'
    initialize_dataset(root, population_id='population', acceptance_policy_id='v2')
    inputs = _generator_inputs(tmp_path, model_id='Qwen/model', revision='a'*40)
    priority = _priority(tmp_path)
    configs = {}
    for model, model_id in [('qwen38', 'Qwen/model'), ('llama4', 'meta-llama/model')]:
        selected = replace(inputs, model_id=model_id)
        register_generator_tasks(dataset_root=root, model_slug=model, inputs=selected,
                                 keyword_priority_path=priority, writer_id='register-' + model)
        profile = build_profile(stage='qwen-generator', model_id=model_id, model_revision='a'*40,
            vllm_executable='/fixture/vllm', vllm_version='0.28.0', vllm_help='',
            visible_gpus=[{'index': i, 'uuid': f'GPU-{i}', 'name': 'GH200', 'memory_total_mib': 97871} for i in range(4)],
            cuda_visible_devices='0,1,2,3', expected_gpu_name_pattern='GH200', max_model_len=4096, request_concurrency=1)
        path = tmp_path / (model + '.json')
        path.write_text(json.dumps(profile))
        configs[model] = {'runtime_environment': {
            'SEARCH_AGENTIC_PROFILE': str(path), 'SEARCH_AGENTIC_DDG_SNAPSHOT': str(inputs.search_snapshots['duckduckgo']),
            'SEARCH_AGENTIC_SEARXNG_SNAPSHOT': str(inputs.search_snapshots['searxng']),
            'SEARCH_AGENTIC_PROMPTS_JSONL': str(inputs.prompts_jsonl),
            'SEARCH_AGENTIC_SELECTION_RECORDS_JSONL': str(inputs.selection_records_jsonl)}}
    return root, configs, priority


def test_multimodel_freeze_preserves_tasks_results_and_original_files(tmp_path):
    root, configs, priority = fixture(tmp_path)
    tasks, _, _ = inventory(root, stripes=4)
    complete(root, tasks[0], 'done')
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    scheduler = {'cluster': 'jupiter', 'complete': True, 'jobs': [], 'captured_at_epoch': int(time.time())}
    target = tmp_path / 'mirror'
    result = stage(root, target, configs, priority, scheduler, stripes=4)
    copied, done, blocked = inventory(target, stripes=4)
    assert {t['fingerprint'] for t in copied} == {t['fingerprint'] for t in tasks}
    assert len(done) == 1 and not blocked
    assert result['missing_models'] == ['nemotron']
    assert stage(root, target, configs, priority, scheduler, stripes=4) == result
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='missing frozen input'):
        stage(root, tmp_path / 'missing', {'qwen38': configs['qwen38']}, priority, scheduler, stripes=4)
    with pytest.raises(ValueError, match='legacy allocations'):
        stage(root, tmp_path / 'active', configs, priority, {**scheduler, 'jobs': [{'job_id': '1995245'}]}, stripes=4)
    assert not (tmp_path / 'active').exists()


def test_bootstrap_inspection_does_not_publish_or_change_scientific_files(tmp_path, monkeypatch, capsys):
    root, _, _ = fixture(tmp_path)
    # The existing ledger reader initializes advisory lock files on first read.
    inventory(root, stripes=4)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(bootstrap, 'capture', lambda **kw: {'complete': True, 'jobs': [], 'owners': [],
                                                          'captured_at_epoch': int(time.time())})
    monkeypatch.setattr(bootstrap, 'HubStore', lambda *a: pytest.fail('inspection must not publish'))
    assert bootstrap.main(['--source', str(root), '--output', str(tmp_path / 'bootstrap'),
                           '--since', '2026-09-24', '--stripes', '4']) == 0
    assert json.loads(capsys.readouterr().out)['verified_completed'] == 0
    assert not (tmp_path / 'bootstrap').exists()
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_bootstrap_publishes_verified_inputs_and_uncalibrated_tracker(tmp_path, monkeypatch, capsys):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.scripts import manage_agentic_hours
    from analysis.tests.test_agentic_hours import MemoryHub
    root, configs, priority = fixture(tmp_path)
    config_path = tmp_path / 'inputs.json'
    config_path.write_text(json.dumps(configs))
    quota = tmp_path / 'quota.json'
    quota.write_text('{}')
    hub = MemoryHub()
    monkeypatch.setattr(bootstrap, 'capture', lambda **kw: {'complete': True, 'jobs': [], 'owners': [],
                                                          'captured_at_epoch': int(time.time())})
    monkeypatch.setattr(bootstrap, 'HubStore', lambda *a: hub)
    monkeypatch.setattr(bootstrap.subprocess, 'check_output',
                        lambda argv, **kw: 'c'*40 if 'rev-parse' in argv else '')
    monkeypatch.setattr(manage_agentic_hours, 'health', lambda *a: {'safe_to_admit': True})
    output = tmp_path / 'bootstrap'
    argv = ['--source', str(root), '--output', str(output), '--model-inputs', str(config_path),
            '--keyword-priority', str(priority), '--quota-evidence', str(quota),
            '--since', '2026-09-24', '--stripes', '4', '--publish']
    assert bootstrap.main(argv) == 0
    capsys.readouterr()
    exchange = Exchange(hub, tmp_path / 'journal')
    revision, state = exchange.snapshot()
    assert state['current_plan'] and state['hours'] == {}
    report = json.loads(hub.read('coordination/progress.json', revision))
    assert report['cells'] == {'awaiting_calibration': 48}
    assert report['missing_models'] == ['nemotron']
    publication = json.loads((output / 'publication.json').read_bytes())
    assert exchange.manifest(publication['input_bundle'])['metadata']['kind'] == 'frozen-inputs'
    old_revision = hub.head()
    with pytest.raises(ValueError, match='already exists'):
        bootstrap.main(argv)
    assert hub.head() == old_revision


def test_shared_inputs_support_pinned_qwen_environment_preparation(tmp_path):
    from analysis.scripts.prepare_shared_hour_inputs import verify_inputs
    root, configs, priority = fixture(tmp_path)
    snapshot = {'cluster': 'jupiter', 'complete': True, 'jobs': [], 'captured_at_epoch': int(time.time())}
    target = tmp_path / 'mirror'
    stage(root, target, configs, priority, snapshot, stripes=4)
    manifest = next((target / 'artifacts/shared-preparations').glob('*.json'))
    report = verify_inputs(target, manifest, model='qwen38')
    assert report['reference_vllm_version'] == '0.28.0'
    assert report['gpu_validation'] == 'not_performed'
    from pathlib import Path
    Path(report['files']['SEARCH_AGENTIC_DDG_SNAPSHOT']).write_text('changed search')
    with pytest.raises(ValueError, match='checksum/path'):
        verify_inputs(target, manifest, model='qwen38')


def test_full_dispatch_publication_is_metadata_only_and_idempotent(tmp_path, monkeypatch):
    import gzip

    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.tests.test_agentic_hours import MemoryHub
    root, _, _ = fixture(tmp_path)
    hub = MemoryHub()
    snapshot = {'cluster': 'jupiter', 'complete': True, 'jobs': [], 'owners': [],
                'captured_at_epoch': time.time()}
    monkeypatch.setattr(bootstrap, 'reconcile', lambda *a, **kw: {'actions': [], 'blocked': []})
    exchange = Exchange(hub, tmp_path / 'journal')
    result = bootstrap.publish_dispatch(root, exchange, snapshot, stripes=4)
    doc = json.loads(gzip.decompress(hub.read(result['inventory_path'], hub.head())))
    assert len(doc['tasks']) == 48
    assert len({t['fingerprint'] for t in doc['tasks']}) == 48
    assert doc['runnable'] is False and doc['hour_packages'] is None
    assert all(t['preferred_cluster'] == ('horeka' if t['model'] == 'qwen38' else 'jupiter') for t in doc['tasks'])
    assert all('axis_bin' in t for t in doc['tasks'])
    assert all(p.startswith('coordination/dispatch') for p in hub.versions[-1])
    assert bootstrap.publish_dispatch(root, exchange, snapshot, stripes=4)['published_revision'] == result['published_revision']
    with pytest.raises(ValueError, match='legacy allocations'):
        bootstrap.publish_dispatch(root, exchange, {**snapshot, 'jobs': [{'job_id': '1'}]}, stripes=4)
