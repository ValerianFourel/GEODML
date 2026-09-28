"""Pre-divided HoreKa Qwen bouts: fixed work per bout, explicit approval, no repeats."""
import json
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_qwen_bouts as bouts


def row(n, prompt, *, config='c1', model='qwen38', rank=1, keyword='k1'):
    return {'model': model, 'fingerprint': f'f{n}', 'priority_rank': rank, 'keyword_id': keyword,
            'prompt_id': prompt, 'task_id': f't{n:04d}', 'configuration_sha256': config,
            'runnable_task': {'cell_id': f'cell-{n}'}}


def test_remaining_rows_exclude_every_known_completion_and_keep_batch_order():
    tasks = [row(3, 'p2', rank=2), row(1, 'p1'), row(2, 'p1'), row(4, 'p3'), row(5, 'p3'),
             row(6, 'p4'), row(7, 'p4', model='llama4'), row(8, 'p5')]
    plan = {'completed_before_plan': ['f4'],
            'deferred': {f'f{n}': 'awaiting_calibration' for n in (1, 2, 3, 4, 5, 6, 7)} | {'f8': 'blocked_or_failed'}}
    rows = bouts.remaining_rows(plan, tasks, completed={'f5'}, blocked={'f6'}, published={'f2'})
    assert [r['fingerprint'] for r in rows] == ['f1', 'f3']


def test_capacity_uses_measured_rate_after_startup_and_margins():
    # 5 h minus 6 min startup, 5 min admission stop and 2 min cleanup at 52.5 s per cell.
    assert bouts.capacity('05:00:00', 52.5, 360) == 328
    with pytest.raises(ValueError, match='no useful'):
        bouts.capacity('00:10:00', 52.5, 360)


def test_split_keeps_prompts_whole_one_configuration_per_bout_and_spills_next_head():
    rows = [row(n, f'p{n // 4}') for n in range(0, 20)]            # five prompts of four cells
    rows += [row(n, f'q{n // 4}', config='c2') for n in range(20, 28)]  # new configuration
    result = bouts.split(rows, primary_cells=8, overbook=1.25)
    assert [len(b['primary']) for b in result] == [8, 8, 4, 8]
    for b in result:
        prompts = [r['prompt_id'] for r in b['primary']]
        assert all(prompts.count(p) == 4 for p in set(prompts))
        assert len({r['configuration_sha256'] for r in b['primary']}) == 1
    assert [r['fingerprint'] for r in result[0]['spill']] == ['f8', 'f9']
    assert result[2]['spill'] == []        # never spill across configurations
    assert result[3]['spill'] == []        # last bout
    primaries = [r['fingerprint'] for b in result for r in b['primary']]
    assert sorted(primaries) == sorted(r['fingerprint'] for r in rows) and len(set(primaries)) == len(rows)
    assert bouts.split(rows, primary_cells=8, overbook=1.25) == result
    with pytest.raises(ValueError, match='no remaining'):
        bouts.split([], primary_cells=8, overbook=1.25)


def _division(tmp_path, count=3):
    root = tmp_path / 'dataset'
    manifests = root / 'artifacts/shared-preparations'
    manifests.mkdir(parents=True)
    (manifests / 'manifest.json').write_text(json.dumps({'models': {'qwen38': {'reference_runtime': {
        'SEARCH_AGENTIC_PROMPT_COUNT': '26008', 'SEARCH_AGENTIC_PROMPT_SELECTION_SEED': '20260912',
        'SEARCH_AGENTIC_REQUEST_CONCURRENCY': '1', 'SEARCH_AGENTIC_CELL_CONCURRENCY': '12'}}}}))
    (root / 'contract.json').write_text('{}')
    division = tmp_path / 'division'
    for n in range(1, count + 1):
        directory = division / 'bouts' / f'bout-{n:04d}'
        directory.mkdir(parents=True)
        (directory / 'bout.json').write_text(json.dumps({'number': n, 'cell_count': 410}))
        (directory / 'cells.jsonl').write_text('{"cell_id":"x"}\n')
    (division / 'division.json').write_text(json.dumps({
        'format_version': bouts.DIVISION_VERSION, 'dataset_root': str(root),
        'sizing': {'walltime': '05:00:00', 'seconds_per_cell': 52.5, 'primary_cells': 328},
        'bouts': [{'number': n} for n in range(1, count + 1)]}))
    return division, root


def _submit_fixture(tmp_path, monkeypatch, count=3):
    from analysis.interpretability.pipeline import agentic_hour_runtime
    from analysis.scripts import manage_agentic_hours, prepare_horeka_qwen, prepare_shared_hour_inputs
    division, root = _division(tmp_path, count)
    workspace = tmp_path / 'workspace'
    runtime = workspace / 'environment/qwen-runtime/bin/python'
    runtime.parent.mkdir(parents=True)
    runtime.write_text('runtime')
    cross = tmp_path / 'bge'
    cross.mkdir()
    monkeypatch.setattr(prepare_shared_hour_inputs, 'verify_inputs',
                        lambda *a, **kw: {'files': {'SEARCH_AGENTIC_PROFILE': 'frozen.json'}})
    monkeypatch.setattr(prepare_horeka_qwen, 'capture_quota', lambda *a: {})
    monkeypatch.setattr(prepare_horeka_qwen, 'snapshot', lambda *a: cross)
    monkeypatch.setattr(manage_agentic_hours, 'health', lambda *a: {'safe_to_admit': True, 'quota_verified': True})
    reservations = []
    monkeypatch.setattr(agentic_hour_runtime, 'validate_reservation', lambda request, site: reservations.append(site))
    monkeypatch.setattr(bouts.subprocess, 'check_output', lambda command, **kw: 'a' * 40 if 'rev-parse' in command else '')
    submissions = []
    def run(command, **kw):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{1000 + len(submissions)}\n', stderr='')
    monkeypatch.setattr(bouts.subprocess, 'run', run)
    args = SimpleNamespace(division=division, workspace=workspace, first=1, count=count, account='project',
                           partition='accelerated', reservation='casualnet', approved_walltime='05:00:00',
                           approved_count=30, approval='Valerian approved 30 x 5 h in casualnet', dry_run=True)
    return args, submissions, reservations, division


def test_submit_dry_run_writes_independent_bouts_without_sbatch(tmp_path, monkeypatch):
    args, submissions, reservations, division = _submit_fixture(tmp_path, monkeypatch)
    bouts.submit(args)
    assert submissions == []
    assert reservations[0]['reservation'] == 'casualnet'
    for n in (1, 2, 3):
        directory = division / 'bouts' / f'bout-{n:04d}'
        assert not (directory / 'SUBMISSION_ATTEMPTED').exists()
        config = json.loads((directory / 'config.json').read_text())
        assert config['job_name'] == f'geodml-qwen-bout-{n:04d}' and config['walltime'] == '05:00:00'
        assert config['approval']['maximum_gpu_hours'] == 20 and config['approval']['approved_allocations'] == 30
        command = json.loads((directory / 'submission-command.json').read_text())
        for flag in ('--time=05:00:00', '--exclusive', '--gres=gpu:4', '--nodes=1', '--no-requeue',
                     '--reservation=casualnet', f'--job-name=geodml-qwen-bout-{n:04d}'):
            assert flag in command
        assert command[-1] == str(directory / 'config.json')


def test_submit_sends_each_bout_once_and_refuses_repeats_and_overreach(tmp_path, monkeypatch):
    args, submissions, _, division = _submit_fixture(tmp_path, monkeypatch)
    args.dry_run = False
    bouts.submit(args)
    assert len(submissions) == 3
    receipt = json.loads((division / 'submissions/submitted-0001-0003.json').read_text())
    assert [r['job_id'] for r in receipt] == ['1001', '1002', '1003']
    assert all((division / 'bouts' / f'bout-{n:04d}' / 'SUBMISSION_ATTEMPTED').exists() for n in (1, 2, 3))
    with pytest.raises(ValueError, match='already submitted'):
        bouts.submit(args)
    assert len(submissions) == 3
    args.count, args.approved_count = 3, 2
    with pytest.raises(ValueError, match='exceeds the approved'):
        bouts.submit(args)
    args.approved_count, args.approved_walltime = 30, '01:00:00'
    with pytest.raises(ValueError, match='wall-time differs'):
        bouts.submit(args)


def test_submit_refuses_a_dataset_without_contract(tmp_path, monkeypatch):
    args, submissions, _, division = _submit_fixture(tmp_path, monkeypatch)
    (tmp_path / 'dataset/contract.json').unlink()
    with pytest.raises(ValueError, match='contract.json'):
        bouts.submit(args)
    assert submissions == []


def test_boundary_failure_precedes_config_read(tmp_path, monkeypatch):
    from analysis.scripts import verify_inference_allocation
    def fail(cluster):
        assert cluster == 'horeka'
        raise ValueError('allocation unverified')
    monkeypatch.setattr(verify_inference_allocation, 'verify', fail)
    with pytest.raises(ValueError, match='allocation unverified'):
        bouts.execute(tmp_path / 'missing.json')


def test_controller_writes_to_the_shared_ledger_with_a_unique_writer(tmp_path):
    config = {'repository': '/repo', 'bout_dir': str(tmp_path), 'dataset_root': '/data',
              'cross_encoder': '/bge', 'cross_revision': 'r',
              'inputs': {'SEARCH_AGENTIC_DDG_SNAPSHOT': 'd', 'SEARCH_AGENTIC_SEARXNG_SNAPSHOT': 's',
                         'SEARCH_AGENTIC_PROMPTS_JSONL': 'p', 'SEARCH_AGENTIC_SELECTION_RECORDS_JSONL': 'r'},
              'settings': {'SEARCH_AGENTIC_PROMPT_COUNT': '26008', 'SEARCH_AGENTIC_PROMPT_SELECTION_SEED': '1',
                           'SEARCH_AGENTIC_REQUEST_CONCURRENCY': '1', 'SEARCH_AGENTIC_CELL_CONCURRENCY': '12'}}
    profile = {'serving': {'public_base_url': 'http://127.0.0.1:8010/v1'},
               'model': {'model_id': 'Qwen/Qwen3.8-27B', 'model_revision': 'rev'}}
    command = bouts.controller_command(config, profile=profile, output=tmp_path / 'out', writer_id='horeka-bout0001-job7')
    joined = ' '.join(command)
    assert '--dataset-root /data --dataset-writer-id horeka-bout0001-job7' in joined
    assert f'--cell-ids-jsonl {tmp_path}/cells.jsonl' in joined
    assert '--request-concurrency 1 --cell-concurrency 12' in joined
    assert '--production-conditions' in command and '--disable-thinking' in command


def test_compiler_environment_prefers_system_gcc(monkeypatch):
    monkeypatch.setattr(bouts.shutil, 'which', lambda name: f'/usr/bin/{name}')
    assert bouts.compiler_environment() == {'CC': '/usr/bin/gcc', 'CXX': '/usr/bin/g++',
                                            'CUDAHOSTCXX': '/usr/bin/g++', 'NVCC_CCBIN': '/usr/bin/g++'}
    monkeypatch.setattr(bouts.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError, match='gcc'):
        bouts.compiler_environment()


def test_divide_writes_every_remaining_cell_once_and_refuses_overwrite(tmp_path, monkeypatch):
    from analysis.interpretability.pipeline import agentic_hour_sync, agentic_hours
    from analysis.scripts import publish_qwen_results
    tasks = [row(n, f'p{n // 4}') for n in range(0, 40)]
    plan = {'completed_before_plan': ['f0'], 'deferred': {f'f{n}': 'awaiting_calibration' for n in range(40)}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(agentic_hours, 'inventory', lambda *a, **kw: (tasks, {'f1'}, set()))
    class Store:
        def head(self):
            return 'rev123'
    monkeypatch.setattr(agentic_hour_sync, 'HubStore', lambda repo: Store())
    monkeypatch.setattr(agentic_hour_sync, 'Exchange', lambda store, journal: SimpleNamespace(store=store))
    monkeypatch.setattr(publish_qwen_results, 'read_index', lambda store, revision: {'bundles': []})
    monkeypatch.setattr(publish_qwen_results, 'published_fingerprints', lambda exchange, index: {'f2', 'f3'})
    args = SimpleNamespace(dataset=tmp_path / 'dataset', plan=tmp_path / 'plan.json', output=tmp_path / 'division',
                           repo_id='repo', walltime='05:00:00', seconds_per_cell=52.5, startup_seconds=360,
                           overbook=1.25, measurement='job 5167484', stripes=4)
    bouts.divide(args)
    division = json.loads((tmp_path / 'division/division.json').read_text())
    assert division['counts'] == {'registered_qwen': 40, 'remaining': 36, 'published_excluded': 2, 'bouts': 1}
    assert division['hub_revision'] == 'rev123' and division['sizing']['primary_cells'] == 328
    record = json.loads((tmp_path / 'division/bouts/bout-0001/bout.json').read_text())
    assert record['primary_cells'] == 36 and record['spill_cells'] == 0
    assert len((tmp_path / 'division/bouts/bout-0001/cells.jsonl').read_text().splitlines()) == 36
    with pytest.raises(ValueError, match='already exists'):
        bouts.divide(args)


def test_submit_frees_only_bouts_whose_sbatch_was_refused(tmp_path, monkeypatch):
    args, submissions, _, division = _submit_fixture(tmp_path, monkeypatch, count=4)
    args.dry_run = False
    def limited(command, **kw):
        submissions.append(command)
        if len(submissions) == 3:
            return SimpleNamespace(returncode=1, stdout='', stderr='sbatch: error: AssocGrpSubmitJobsLimit')
        return SimpleNamespace(returncode=0, stdout=f'{1000 + len(submissions)}\n', stderr='')
    monkeypatch.setattr(bouts.subprocess, 'run', limited)
    with pytest.raises(RuntimeError, match='bout 3: sbatch failed'):
        bouts.submit(args)
    first = json.loads((division / 'submissions/submitted-0001-0004.json').read_text())
    assert [(r['bout'], r['returncode']) for r in first] == [(1, 0), (2, 0), (3, 1)]
    third = division / 'bouts/bout-0003'
    assert bouts.refused_submission(third) and not bouts.refused_submission(division / 'bouts/bout-0002')
    args.first, args.count = 2, 3
    with pytest.raises(ValueError, match='already submitted'):
        bouts.submit(args)                      # bout 2 has a real job
    args.first, args.count = 3, 2
    bouts.submit(args)                          # bout 3 was refused: sent again, bout 4 for the first time
    assert len(submissions) == 5
    assert (third / 'SUBMISSION_ATTEMPTED').exists()
    assert json.loads((third / 'submission.json').read_text())['stdout'].strip() == '1004'
    assert len(list(third.glob('submission-refused-*.json'))) == 1 and len(list(third.glob('SUBMISSION_REFUSED-*'))) == 1
    second = json.loads((division / 'submissions/submitted-0003-0004.json').read_text())
    assert [r['job_id'] for r in second] == ['1004', '1005']
    assert json.loads((division / 'submissions/submitted-0001-0004.json').read_text()) == first
    with pytest.raises(ValueError, match='already submitted'):
        bouts.submit(args)


def test_submit_without_reservation_targets_the_whole_partition(tmp_path, monkeypatch):
    args, _, reservations, division = _submit_fixture(tmp_path, monkeypatch)
    args.reservation = None
    bouts.submit(args)
    command = json.loads((division / 'bouts/bout-0001/submission-command.json').read_text())
    assert '--partition=accelerated' in command and not any(c.startswith('--reservation') for c in command)
    assert reservations[0]['reservation'] is None
