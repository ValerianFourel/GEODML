"""Compatibility preparation cannot become an unapproved production allocation."""
import pytest

from analysis.scripts import horeka_qwen_probe as probe
from analysis.tests.test_search_vllm_stage import gpu_inventory, profile


def test_probe_resolves_real_uncalibrated_plan_against_dataset(tmp_path):
    from analysis.interpretability.pipeline.agentic_hours import (
        build_plan,
        empty_registry,
        inventory,
    )
    from analysis.tests.test_agentic_hours import complete
    from analysis.tests.test_prepare_shared_hour_inputs import fixture
    root, _, _ = fixture(tmp_path)
    tasks, done, blocked = inventory(root, stripes=4)
    qwen = sorted((r for r in tasks if r['model'] == 'qwen38'),
                  key=lambda r: (r['priority_rank'], r['keyword_id'], r['prompt_id'], r['task_id']))
    plan = build_plan(tasks=tasks, calibration={}, registry=empty_registry(), contract={},
                      completed=done, blocked=blocked, source_commit='a'*40,
                      input_bundle='frozen-inputs', allow_uncalibrated=True)
    assert plan['tasks'] == {} and plan['packages'] == []
    assert len(plan['deferred']) == 48
    # Completion after planning must also be excluded.
    complete(root, qwen[0], 'new-completion')
    expected = [r['runnable_task'] for r in qwen[1:] if r['prompt_id'] == qwen[0]['prompt_id']]
    assert len(expected) == 11
    assert probe.choose_probe(plan, root, stripes=4) == expected
    for row in qwen:
        plan['deferred'][row['fingerprint']] = 'blocked_or_failed'
    with pytest.raises(ValueError, match='no missing Qwen'):
        probe.choose_probe(plan, root, stripes=4)


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
