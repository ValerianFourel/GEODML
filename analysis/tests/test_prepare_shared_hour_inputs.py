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
