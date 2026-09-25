from copy import deepcopy
from types import SimpleNamespace

import pytest

from analysis.scripts import dispatch_threehour_wave as wave
from analysis.tests.test_dispatch_llama_five import fixture


def args(tmp_path):
    return SimpleNamespace(output=tmp_path, approval='explicit ten-job three-hour wave',
        existing_qwen_job='42', maximum_concurrent=10, allow_stale_quota=True,
        since='2026-09-01', allowed_jobs=['42'], replace_cancelled_qwen=False, qwen_reference=tmp_path / 'reference')


def test_capacity_counts_existing_qwen_and_rejects_unrelated_allocations():
    snapshot = {'complete': True, 'captured_at_epoch': 100, 'jobs': [{'job_id': '42'}]}
    wave.gate(snapshot, ['42'], 4, 5, now=101)
    with pytest.raises(ValueError):
        wave.gate(snapshot, ['42'], 5, 5, now=101)
    with pytest.raises(ValueError):
        wave.gate(snapshot, [], 4, 10, now=101)
    with pytest.raises(ValueError):
        wave.gate(snapshot, ['42'], 4, 10, now=300)


def test_llama_groups_preserve_priority_and_exclude_owned_completed_and_failed():
    state = {'hours': {}}
    for i in range(8):
        state['hours'][str(i)] = {'model': 'llama4', 'owner': None, 'status': 'partial',
            'task_fingerprints': [f'{i}-{n}' for n in range(9002)], 'completed': [f'{i}-0'],
            'failed': [f'{i}-1'], 'priority_rank': i}
    state['hours']['0']['owner'] = {'attempt': 'prior'}
    state['hours']['1']['status'] = 'superseded'
    state['hours']['2']['model'] = 'qwen38'
    assert wave.groups(state) == [['3'], ['4'], ['5'], ['6'], ['7']]


@pytest.mark.parametrize('broken', [None, 'overlap', 'duration', 'storage', 'prior_active', 'six'])
def test_admission_approves_exactly_five_with_no_faked_quota(tmp_path, monkeypatch, broken):
    state, attempts, snapshot, storage = fixture()
    options = args(tmp_path)
    monkeypatch.setattr(wave.time, 'time', lambda: 1001)
    for a in attempts:
        a['request']['approval'] = wave.approval(options, 'measured estimate')
    if broken == 'overlap':
        state['hours']['wave-1']['task_fingerprints'] = ['cell-0']
    if broken == 'duration':
        attempts[0]['request']['approval']['walltime_seconds'] = 3600
    if broken == 'storage':
        storage['safe_to_admit'] = False
    if broken == 'prior_active':
        state['admission']['jupiter'] = {'finite_wave': {'attempt_ids': ['old']}}
    if broken == 'six':
        attempts.append(deepcopy(attempts[0]))
    before = deepcopy(state)
    if broken:
        with pytest.raises(ValueError):
            wave.admit(state, attempts, snapshot, storage, options)
    else:
        result = wave.admit(state, attempts, snapshot, storage, options)
        admission = result['admission']['jupiter']
        assert len(admission['tickets']) == 5
        assert admission['finite_wave']['maximum_gpu_hours'] == 60
        assert all(t['quota_verified'] is False for t in admission['tickets'].values())
    assert state == before


@pytest.mark.parametrize('uncertain', [False, True])
def test_four_qwen_submissions_are_held_then_released_once(tmp_path, monkeypatch, uncertain):
    options = args(tmp_path)
    wave.save(options.qwen_reference / 'preparation.json', {'dataset_root': str(tmp_path)})
    jobs = [{'job_id': '42', 'state': 'RUNNING', 'held': False}]
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()), 'jobs': jobs})
    commands, releases = [], []
    def command(argv):
        if argv[0] == 'sbatch':
            assert (tmp_path / 'submission-intent.json').exists()
            commands.append(argv)
            if uncertain and len(commands) == 2:
                raise OSError('uncertain response')
            job = str(100 + len(commands))
            jobs.append({'job_id': job, 'state': 'PENDING', 'held': True})
            return job
        assert len(commands) == 4
        releases.append(argv)
        return ''
    monkeypatch.setattr(wave.first, 'command', command)
    entries = [(tmp_path / str(i), ['sbatch', '--hold'], wave.approval(options, 'estimate')) for i in range(4)]
    if uncertain:
        with pytest.raises(OSError):
            wave.submit(options, entries, tmp_path, ['42'])
        assert releases == []
    else:
        wave.submit(options, entries, tmp_path, ['42'])
        assert [r[-1] for r in releases] == ['101', '102', '103', '104']
        assert wave.read(tmp_path / 'submitted.json')['maximum_gpu_hours'] == 48
    with pytest.raises(ValueError, match='already attempted'):
        wave.submit(options, entries, tmp_path, ['42'])


def test_qwen_reuses_frozen_queue_and_ledger_without_touching_member_one(tmp_path, monkeypatch):
    options = args(tmp_path / 'new')
    options.output.mkdir()
    options.qwen_reference = tmp_path / 'reference'
    options.environment_file = tmp_path / 'env.sh'
    options.account, options.partition = 'project', 'booster'
    runtime = {'GEODML_DATASET_ROOT': str(tmp_path / 'dataset'), 'ACL_ARR_VENV': '/venv',
               'GEODML_WAVE_ROOT': str(options.qwen_reference / 'wave')}
    wave.save(options.qwen_reference / 'runtime.json', runtime)
    reference = {'dataset_root': runtime['GEODML_DATASET_ROOT'], 'files': {}, 'dataset_files': {}, 'external_files': {}}
    monkeypatch.setattr(wave, 'verify_reference', lambda p: (reference, {'job_id': '42'}))
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()),
        'jobs': [{'job_id': '42', 'state': 'RUNNING'}], 'owners': []})
    captured = []
    monkeypatch.setattr(wave, 'submit', lambda *a: captured.append(a))
    exchange = SimpleNamespace(snapshot=lambda: ('revision', {'hours': {}}))
    wave.qwen(options, 'a' * 40, exchange)
    assert len(captured[0][1]) == 4
    for directory, command, approved in captured[0][1]:
        result = wave.read(directory / 'runtime.json')
        assert result['GEODML_DATASET_ROOT'] == runtime['GEODML_DATASET_ROOT']
        assert result['GEODML_WAVE_ROOT'] == runtime['GEODML_WAVE_ROOT']
        assert result['GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY'] == '1'
        assert '--time=03:00:00' in command and '--exclusive' in command
        assert approved['maximum_gpu_hours'] == 12
    assert wave.read(options.qwen_reference / 'runtime.json') == runtime


def test_saved_qwen_reference_rejects_changed_frozen_input(tmp_path):
    frozen = tmp_path / 'backlog.jsonl'
    frozen.write_text('frozen\n')
    wave.save(tmp_path / 'preparation.json', {
        'model': 'qwen38', 'approved_walltime': '03:00:00',
        'files': {str(frozen): wave.first.identity(frozen)}, 'dataset_files': {}, 'external_files': {}})
    wave.save(tmp_path / 'submission.json', {'status': 'submitted', 'job_id': '42'})
    assert wave.verify_reference(tmp_path)[1]['job_id'] == '42'
    frozen.write_text('changed\n')
    with pytest.raises(ValueError, match='Frozen Qwen reference changed'):
        wave.verify_reference(tmp_path)


def test_llama_never_reassigns_an_unconfirmed_previous_allocation(tmp_path, monkeypatch):
    options = args(tmp_path)
    options.llama_site = tmp_path / 'site.json'
    path = tmp_path / 'old/attempt.json'
    wave.save(path, {'writer_id': 'old-writer'})
    wave.save(options.llama_site, {'dataset_root': str(tmp_path), 'attempts': [str(path)]})
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()), 'jobs': []})
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    monkeypatch.setattr(wave, 'scheduler', lambda *a: {'owners': []})
    exchange = SimpleNamespace(snapshot=lambda: ('revision', {'hours': {}}))
    with pytest.raises(ValueError, match='not confirmed terminal'):
        wave.llama(options, 'a' * 40, exchange)
    assert not (tmp_path / 'submission-intent.json').exists()


@pytest.mark.parametrize('state,blocked', [('CANCELLED', False), ('CANCELLED', True), ('RUNNING', False), ('FAILED', False), (None, False)])
def test_explicit_cancelled_replacement_preserves_results_and_creates_exactly_five(tmp_path, monkeypatch, state, blocked):
    options = args(tmp_path / 'new')
    options.output.mkdir()
    options.replace_cancelled_qwen = True
    options.qwen_reference = tmp_path / 'reference'
    options.environment_file = tmp_path / 'env.sh'
    options.account, options.partition = 'project', 'booster'
    source = {'GEODML_DATASET_ROOT': str(tmp_path / 'dataset'), 'ACL_ARR_VENV': '/venv',
              'GEODML_WAVE_ROOT': str(options.qwen_reference / 'wave')}
    wave.save(options.qwen_reference / 'runtime.json', source)
    reference = {'dataset_root': source['GEODML_DATASET_ROOT'], 'files': {}, 'dataset_files': {}, 'external_files': {}}
    monkeypatch.setattr(wave, 'verify_reference', lambda p: (reference, {'job_id': '42'}))
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    snapshot = {'complete': True, 'captured_at_epoch': int(wave.time.time()),
                'jobs': [{'job_id': '42', 'state': state}] if state == 'RUNNING' else [],
                'owners': [{'job_id': '42', 'state': state}] if state and state != 'RUNNING' else []}
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: snapshot)
    monkeypatch.setattr(wave.first, 'command', lambda command: '42|CANCELLED by 123|0:0|35|gres/gpu=4')
    reconciled, submitted = [], []
    def reconcile(root, **kw):
        reconciled.append((root, kw))
        return {'blocked': ['bad result'] if blocked else [], 'actions': ['preserved checkpoint']}
    monkeypatch.setattr(wave, 'reconcile', reconcile)
    monkeypatch.setattr(wave, 'submit', lambda *a: submitted.append(a))
    exchange = SimpleNamespace(snapshot=lambda: ('revision', {'hours': {}}))
    if state == 'CANCELLED' and not blocked:
        wave.qwen(options, 'a' * 40, exchange)
        assert len(reconciled) == 1 and reconciled[0][1]['apply'] is True
        entries = submitted[0][1]
        assert [d.name for d, _, _ in entries] == ['qwen-1', 'qwen-2', 'qwen-3', 'qwen-4', 'qwen-5']
        assert sum(a['maximum_gpu_hours'] for _, _, a in entries) == 60
        assert submitted[0][3] == []
        saved = wave.read(options.output / 'cancelled-reference.json')
        assert saved['new_wave_gpu_hours'] == 120 and '35' in saved['accounting']
        assert wave.read(options.qwen_reference / 'runtime.json') == source
    else:
        with pytest.raises(ValueError):
            wave.qwen(options, 'a' * 40, exchange)
        assert submitted == []
        if state != 'CANCELLED':
            assert reconciled == []
