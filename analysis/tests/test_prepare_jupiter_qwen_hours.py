import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline.agentic_hours import (
    claim_hours,
    empty_registry,
    install_plan,
    inventory,
)
from analysis.scripts import prepare_jupiter_qwen_hours as helper
from analysis.tests.test_agentic_hours import calibration, data, plan


def test_packages_use_same_builder_and_preserve_owners(tmp_path):
    root = data(tmp_path / 'dataset')
    previous = plan(root)
    previous['calibration']['llama4'] = {'configurations': {'llama-config': {'evidence': 'keep'}}}
    original = copy.deepcopy(previous)
    state = install_plan(empty_registry(), plan(root), cluster='jupiter')
    key = next(iter(state['hours']))
    state = claim_hours(state, hour_ids=[key], cluster='horeka', attempt_id='running',
                        supported_models=['qwen38'])
    proposed = helper.prepare_plan(previous, state, inventory(root, stripes=4),
                                   'config-qwen', calibration()['qwen38'],
                                   json.loads((root / 'contract.json').read_bytes()), 'b' * 40)
    assert previous == original
    assert proposed['calibration']['llama4'] == previous['calibration']['llama4']
    assert [r['keyword_id'] for r in proposed['packages']] == ['beta']
    assert proposed['packages'][0]['reference_seconds'] == 3600
    assert install_plan(state, proposed, cluster='jupiter')['hours'][key] == state['hours'][key]
    with pytest.raises(ValueError, match='configuration absent'):
        helper.prepare_plan(previous, state, inventory(root, stripes=4), 'wrong', {}, {}, 'b' * 40)


def reference(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    profile = save('profile.json', {'model': {'model_id': helper.MODELS[0][0],
                                            'model_revision': helper.MODELS[0][1]},
                                    'serving': {'tensor_parallel_size': 4}})
    save('preparation.json', {'dataset_root': '/original',
        'summary': {'configuration_sha256': 'a' * 64},
        'files': {str(profile): {'sha256': hashlib.sha256(profile.read_bytes()).hexdigest()}}})
    save('runtime.json', {'SEARCH_AGENTIC_PROFILE': str(profile)})
    save('boundary.json', {'status': 'verified', 'slurm_job_id': '123'})
    save('allocation.json', {'slurm': {'TimeLimit': '01:00:00', 'NumNodes': '1'}})
    save('results/attempts/job123/run_manifest.json', {
        'status': 'checkpointed', 'failed_cell_ids': [], 'completed_count': 100,
        'direct_dataset': {'root': '/original', 'committed_count': 100, 'reused_count': 0}})


def test_measured_cost_and_reject_a100_or_failed_job(tmp_path, monkeypatch):
    reference(tmp_path)
    monkeypatch.setattr(helper.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='123|COMPLETED|gres/gpu:gh200=4,node=1|jupiter|\n'))
    configuration, timing = helper.measured_calibration(tmp_path)
    assert configuration == 'a' * 64
    assert timing['seconds_per_task'] == 36
    assert timing['startup_seconds'] == 0
    assert timing['drain_seconds'] == 300
    for row in ('123|COMPLETED|gres/gpu:a100=4|horeka|',
                '123|FAILED|gres/gpu:gh200=4|jupiter|'):
        monkeypatch.setattr(helper.subprocess, 'run', lambda *a, row=row, **k: SimpleNamespace(stdout=row))
        with pytest.raises(ValueError, match='scheduler evidence'):
            helper.measured_calibration(tmp_path)


def test_missing_or_changed_measurement_fails_before_scheduler(tmp_path, monkeypatch):
    reference(tmp_path)
    monkeypatch.setattr(helper.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected scheduler call'))
    path = tmp_path / 'results/attempts/job123/run_manifest.json'
    record = json.loads(path.read_bytes())
    record['direct_dataset']['reused_count'] = 100
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='clean one-hour'):
        helper.measured_calibration(tmp_path)


def test_completed_cells_are_excluded_and_other_configurations_deferred(tmp_path):
    root = data(tmp_path / 'dataset')
    previous = plan(root)
    local = inventory(root, stripes=4)
    done = {local[0][0]['fingerprint']}
    tasks = copy.deepcopy(local[0])
    tasks[-1]['configuration_sha256'] = 'unmeasured'
    proposed = helper.prepare_plan(previous, empty_registry(), (tasks, done, set()),
                                   'config-qwen', calibration()['qwen38'],
                                   json.loads((root / 'contract.json').read_bytes()), 'b' * 40)
    assigned = [fp for row in proposed['packages'] for fp in row['task_fingerprints']]
    assert len(assigned) == len(set(assigned)) == 2
    assert not done.intersection(assigned)
    assert proposed['deferred'][tasks[-1]['fingerprint']] == 'awaiting_calibration'
