"""Compatibility preparation cannot become an unapproved production allocation."""
import pytest

from analysis.scripts import horeka_qwen_probe as probe
from analysis.tests.test_search_vllm_stage import gpu_inventory, profile


def test_probe_selects_one_missing_prompt_in_keyword_order():
    rows = {}
    for i, (priority, prompt, status) in enumerate([(2, 'a', 'awaiting_calibration'),
                                                  (1, 'b', 'awaiting_calibration'),
                                                  (1, 'b', 'awaiting_calibration'),
                                                  (0, 'c', 'blocked_or_failed')]):
        rows[str(i)] = {'model': 'qwen38', 'priority_rank': priority, 'keyword_id': str(priority),
                        'prompt_id': prompt, 'configuration_sha256': 'config', 'task_id': str(i),
                        'runnable_task': {'cell_id': str(i)}}
    plan = {'tasks': rows, 'completed_before_plan': ['2'],
            'deferred': {'0': 'awaiting_calibration', '1': 'awaiting_calibration', '3': 'blocked_or_failed'}}
    assert probe.choose_probe(plan) == [{'cell_id': '1'}]


def test_local_profile_preserves_frozen_science():
    frozen = profile()
    gpus = [{**g, 'name': 'NVIDIA A100 40GB'} for g in gpu_inventory()]
    local = probe.local_profile(frozen, '/horeka/bin/vllm', '0.28.0',
                                 '--language-model-only', gpus, '0,1,2,3')
    assert local['serving'] == frozen['serving']
    assert local['features'] == frozen['features']
    assert local['model'] == frozen['model']
    assert local['visible_gpu_assignment']['expected_gpu_name_pattern'] == 'A100'
    with pytest.raises(ValueError, match='differs'):
        probe.local_profile(frozen, '/bin/vllm', '0.29.0', '', gpus, '0,1,2,3')


def test_compute_boundary_failure_precedes_config_read_and_model_loading(tmp_path, monkeypatch):
    from analysis.scripts import verify_inference_allocation
    def fail(cluster):
        assert cluster == 'horeka'
        raise ValueError('allocation unverified')
    monkeypatch.setattr(verify_inference_allocation, 'verify', fail)
    with pytest.raises(ValueError, match='allocation unverified'):
        probe.execute(tmp_path / 'does-not-exist.json')


def test_submit_once_records_one_hour_budget_and_refuses_repeat(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from analysis.interpretability.pipeline import agentic_hour_runtime, agentic_hours
    from analysis.scripts import (
        capture_agentic_scheduler_snapshot,
        manage_agentic_hours,
        prepare_horeka_qwen,
        prepare_shared_hour_inputs,
    )
    workspace = tmp_path / 'workspace'
    runtime = workspace / 'environment/qwen-runtime/bin/python'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('test runtime')
    root = workspace / 'dataset'
    manifests = root / 'artifacts/shared-preparations'
    manifests.mkdir(parents=True)
    (manifests / 'manifest.json').write_text('{"models":{"qwen38":{"reference_runtime":{}}}}')
    plan = tmp_path / 'plan.json'
    plan.write_text('{"tasks":{},"completed_before_plan":[]}')
    cross = workspace / 'models' / ('models--' + prepare_horeka_qwen.MODELS[1][0].replace('/', '--')) / 'snapshots' / prepare_horeka_qwen.MODELS[1][1]
    cross.mkdir(parents=True)
    monkeypatch.setattr(prepare_shared_hour_inputs, 'verify_inputs', lambda *a, **kw: {'files': {'SEARCH_AGENTIC_PROFILE': 'frozen.json'}})
    monkeypatch.setattr(prepare_horeka_qwen, 'capture_quota', lambda *a: {})
    monkeypatch.setattr(manage_agentic_hours, 'health', lambda *a: {})
    monkeypatch.setattr(capture_agentic_scheduler_snapshot, 'capture', lambda **kw: {})
    monkeypatch.setattr(agentic_hour_runtime, 'admission', lambda *a, **kw: {})
    monkeypatch.setattr(agentic_hour_runtime, 'validate_reservation', lambda *a: None)
    monkeypatch.setattr(agentic_hours, 'verify_plan', lambda *a: None)
    monkeypatch.setattr(probe, 'choose_probe', lambda *a: [{'cell_id': 'missing-cell'}])
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda command, **kw: 'a'*40 if 'rev-parse' in command else '')
    submissions = []
    def submit(command, **kw):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    monkeypatch.setattr(probe.subprocess, 'run', submit)
    args = SimpleNamespace(workspace=workspace, dataset=root, plan=plan, output=tmp_path / 'attempt',
                           account='project', partition='accelerated', reservation=None,
                           since='2026-09-01', approval='User approved one hour', approved_walltime='01:00:00')
    probe.prepare(args)
    assert len(submissions) == 1
    command = submissions[0]
    assert all(flag in command for flag in ['--exclusive', '--time=01:00:00', '--gres=gpu:4', '--no-requeue'])
    assert probe.read(args.output / 'config.json')['approval']['maximum_gpu_hours'] == 4
    with pytest.raises(ValueError, match='already exists'):
        probe.prepare(args)
    assert len(submissions) == 1
