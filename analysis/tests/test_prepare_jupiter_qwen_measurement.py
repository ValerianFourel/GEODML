import json

import pytest

from analysis.scripts import prepare_jupiter_llama as first
from analysis.scripts import prepare_jupiter_qwen_measurement as helper
from analysis.tests.test_prepare_jupiter_llama import registered, submission_fixture


def test_qwen_freeze_reuses_inventory_and_excludes_llama(tmp_path, monkeypatch):
    root, _ = registered(tmp_path)
    local = first.inventory(root, stripes=8)
    monkeypatch.setattr(first, 'inventory', lambda *a, **k: pytest.fail('second inventory'))
    out = tmp_path / 'qwen'
    result = first.freeze_backlog(root, out, model='qwen38', local_inventory=local)
    assert result['eligible'] == 24
    manifest = first.read(out / 'wave/run_manifest.json')
    assert manifest['model'] == 'qwen38'
    selected = [json.loads(line)['cell_id'] for line in (out / 'wave/backlog.jsonl').read_text().splitlines()]
    assert set(selected) == {t['task_id'] for t in local[0] if t['model'] == 'qwen38'}


def test_approved_three_hour_qwen_member_submits_once(tmp_path, monkeypatch):
    args, root, output = submission_fixture(tmp_path, monkeypatch)
    record = first.verify_prepared(output)
    record.update(model='qwen38', approved_walltime='03:00:00', estimate=helper.ESTIMATE)
    args.approved_walltime = '03:00:00'
    args.approval = 'Five Qwen and five Llama three-hour allocations, 30 node-hours total'
    calls = []
    def command(argv):
        calls.append(argv)
        if argv[0] == 'sbatch':
            assert '--time=03:00:00' in argv
            assert '--hold' in argv and '--exclusive' in argv
            assert first.read(output / 'submission.json')['approval']['maximum_gpu_hours'] == 12
            return '1234'
        assert argv == ['scontrol', 'release', '1234']
        return ''
    monkeypatch.setattr(first, 'command', command)
    report = first.submit(args)
    assert report['maximum_gpu_hours'] == 12
    assert len(calls) == 2
    with pytest.raises(ValueError, match='already requested'):
        first.submit(args)
    assert (root / 'control/qwen38-first-allocation.json').is_file()


def test_qwen_bootstrap_refuses_shared_packages(tmp_path, monkeypatch):
    args, _, output = submission_fixture(tmp_path, monkeypatch)
    first.verify_prepared(output).update(model='qwen38', approved_walltime='03:00:00')
    args.approved_walltime = '03:00:00'
    monkeypatch.setattr(first.HubStore, 'read', lambda *a: b'{"hours":{"h":{"model":"qwen38"}}}')
    with pytest.raises(ValueError, match='coordinated dispatcher'):
        first.submit(args)


def test_three_hour_measurement_sizes_one_hour_packages(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from analysis.scripts import prepare_jupiter_qwen_hours as hours
    from analysis.tests.test_prepare_jupiter_qwen_hours import reference
    reference(tmp_path)
    prior = first.read(tmp_path / 'preparation.json')
    prior['approved_walltime'] = '03:00:00'
    (tmp_path / 'preparation.json').write_text(json.dumps(prior))
    (tmp_path / 'allocation.json').write_text(json.dumps({'slurm': {'TimeLimit': '03:00:00', 'NumNodes': '1'}}))
    monkeypatch.setattr(hours.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='123|COMPLETED|gres/gpu:gh200=4|jupiter|'))
    _, timing = hours.measured_calibration(tmp_path)
    assert timing['seconds_per_task'] == 108
    assert timing['evidence']['allocation_seconds'] == 10800
    assert timing['drain_seconds'] == 300
