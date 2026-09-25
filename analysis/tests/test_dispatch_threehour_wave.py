from copy import deepcopy
from pathlib import Path
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


def test_localize_reference_binds_frozen_files_inside_the_member_directory(tmp_path):
    reference_dir = tmp_path / 'reference'
    (reference_dir / 'nested').mkdir(parents=True)
    profile = reference_dir / 'profile.json'
    profile.write_text('{"profile": true}')
    nested = reference_dir / 'nested' / 'backlog.jsonl'
    nested.write_text('cell\n')
    reference = {'files': {str(profile): wave.first.identity(profile),
                           str(nested): wave.first.identity(nested)}}
    member = tmp_path / 'qwen-1'
    member.mkdir()
    localized = wave.localize_reference(member, reference, reference_dir)
    assert set(localized) == {str(member / 'inputs' / 'profile.json'),
                              str(member / 'inputs' / 'nested' / 'backlog.jsonl')}
    from analysis.interpretability.pipeline.agentic_verification_cache import VerificationCache
    with VerificationCache(member) as verification:
        for path, expected in localized.items():
            assert verification.file(Path(path), expected)
        # The exact wave failure: member-one paths are outside the member cache root.
        with pytest.raises(ValueError, match='not in the subpath'):
            verification.file(profile, reference['files'][str(profile)])
    # Re-running is idempotent, and tampering fails closed.
    assert wave.localize_reference(member, reference, reference_dir) == localized
    profile.write_text('{"profile": false}')
    with pytest.raises(ValueError):
        wave.localize_reference(tmp_path / 'qwen-2', reference, reference_dir)
    outside = tmp_path / 'elsewhere.json'
    outside.write_text('x')
    with pytest.raises(ValueError, match='outside member one'):
        wave.localize_reference(tmp_path / 'qwen-3',
                                {'files': {str(outside): wave.first.identity(outside)}}, reference_dir)


def test_qwen_staging_copies_frozen_reference_into_each_member(tmp_path, monkeypatch):
    options = args(tmp_path / 'new')
    options.output.mkdir()
    options.qwen_reference = tmp_path / 'reference'
    options.qwen_reference.mkdir()
    profile = options.qwen_reference / 'profile.json'
    profile.write_text('{"p": 1}')
    options.environment_file = tmp_path / 'env.sh'
    options.account, options.partition = 'project', 'booster'
    runtime = {'GEODML_DATASET_ROOT': str(tmp_path / 'dataset'), 'ACL_ARR_VENV': '/venv'}
    wave.save(options.qwen_reference / 'runtime.json', runtime)
    reference = {'dataset_root': runtime['GEODML_DATASET_ROOT'],
                 'files': {str(profile): wave.first.identity(profile)},
                 'dataset_files': {}, 'external_files': {}}
    monkeypatch.setattr(wave, 'verify_reference', lambda p: (reference, {'job_id': '42'}))
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()),
        'jobs': [{'job_id': '42', 'state': 'RUNNING'}], 'owners': []})
    captured = []
    monkeypatch.setattr(wave, 'submit', lambda *a, **kw: captured.append(a))
    exchange = SimpleNamespace(snapshot=lambda: ('revision', {'hours': {}}))
    wave.qwen(options, 'a' * 40, exchange)
    for directory, _, _ in captured[0][1]:
        record = wave.read(directory / 'preparation.json')
        assert str(profile) not in record['files']
        local = directory / 'inputs' / 'profile.json'
        assert record['files'][str(local)] == wave.first.identity(profile)
        assert wave.first.identity(local) == wave.first.identity(profile)


def relaunch_env(tmp_path, monkeypatch, states, live_members=(), live_wave_jobs=('201', '202')):
    reference = tmp_path / 'reference'
    reference.mkdir()
    profile = reference / 'profile.json'
    profile.write_text('{"p": 1}')
    wave.save(reference / 'preparation.json', {
        'model': 'qwen38', 'approved_walltime': '03:00:00',
        'dataset_root': str(tmp_path / 'dataset'),
        'files': {str(profile): wave.first.identity(profile)},
        'dataset_files': {}, 'external_files': {}})
    wave.save(reference / 'submission.json', {'status': 'submitted', 'job_id': '42'})
    wave.save(reference / 'runtime.json', {'GEODML_DATASET_ROOT': str(tmp_path / 'dataset'),
                                           'ACL_ARR_VENV': '/venv'})
    output = tmp_path / 'wave'
    output.mkdir()
    wave.save(reference / 'expansion-qwen.json', {'output': str(output), 'git_commit': 'b' * 40})
    (output / 'environment.sh').write_text('source env\n')
    jobs = {n: str(100 + n) for n in range(1, 6)}
    wave.save(output / 'submitted.json', {'job_ids': list(jobs.values()), 'maximum_gpu_hours': 60})
    for n, job in jobs.items():
        d = output / f'qwen-{n}'
        wave.save(d / 'submission.json', {'status': 'submitted', 'job_id': job,
                                          'command': ['sbatch'], 'approval': {'walltime': '03:00:00'}})
        wave.save(d / 'preparation.json', {'files': {str(profile): wave.first.identity(profile)}})
        wave.save(d / 'runtime.json', {'GEODML_EXECUTION_COMMIT': 'b' * 40})
        (d / 'run.sh').write_text('old\n')
    live_wave = tmp_path / 'llama-wave'
    live_wave.mkdir()
    wave.save(live_wave / 'submitted.json', {'job_ids': list(live_wave_jobs)})
    snapshot_jobs = [{'job_id': j, 'state': 'RUNNING', 'held': False} for j in live_wave_jobs]
    snapshot_jobs += [{'job_id': jobs[n], 'state': 'PENDING', 'held': False} for n in live_members]
    new_jobs = []
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()),
        'jobs': snapshot_jobs + new_jobs, 'owners': []})
    released, commands = [], []
    next_id = [300]
    def command(argv):
        if argv[0] == 'sbatch':
            commands.append(argv)
            job = str(next_id[0])
            next_id[0] += 1
            new_jobs.append({'job_id': job, 'state': 'PENDING', 'held': True})
            return job
        if argv[0] == 'scontrol':
            released.append(argv[-1])
            return ''
        if argv[0] == 'sacct' and argv[-1] == '--format=State':
            state = states.get(argv[argv.index('-j') + 1])
            return '' if state is None else state
        if argv[0] == 'sacct':
            return 'acct-row'
        raise AssertionError(argv)
    monkeypatch.setattr(wave.first, 'command', command)
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    options = SimpleNamespace(output=output, qwen_reference=reference, since='2026-09-01',
        account='project', partition='booster', approval='fresh relaunch approval',
        maximum_concurrent=10, allow_stale_quota=True, quota=None, existing_qwen_job=None,
        replace_cancelled_qwen=False, live_wave=[live_wave / 'submitted.json'], allowed_jobs=[],
        simultaneous_starts=True)
    options.new_jobs = new_jobs
    return options, jobs, profile, commands, released


def test_relaunch_resubmits_every_failed_member_with_append_only_receipts(tmp_path, monkeypatch):
    options, jobs, profile, commands, released = relaunch_env(
        tmp_path, monkeypatch, {str(100 + n): 'FAILED' for n in range(1, 6)})
    wave.relaunch_qwen(options, 'a' * 40)
    assert released == ['300', '301', '302', '303', '304']
    assert (options.output / 'relaunch-intent-1.json').exists()
    receipt = wave.read(options.output / 'relaunch-1.json')
    assert receipt['job_ids'] == released and receipt['maximum_gpu_hours'] == 60
    assert set(receipt['failed_jobs']) == set(jobs.values())
    assert receipt['approval'] == 'fresh relaunch approval' and receipt['git_commit'] == 'a' * 40
    names = [' '.join(c) for c in commands]
    for n, old in jobs.items():
        d = options.output / f'qwen-{n}'
        assert any(f'--job-name=geodml-qwen-threehour-{n}-r1' in name for name in names)
        assert '--time=03:00:00' in names[int(n) - 1] and '--hold' in names[int(n) - 1]
        assert wave.read(d / f'submission-failed-{old}.json')['job_id'] == old
        assert (d / f'preparation-failed-{old}.json').exists()
        fresh = wave.read(d / 'submission.json')
        assert fresh['status'] == 'submitted' and fresh['job_id'] not in jobs.values()
        record = wave.read(d / 'preparation.json')
        assert record['wave_member'] == f'qwen-{n}-of-5-relaunch-1' and record['relaunch_of'] == old
        assert record['git_commit'] == 'a' * 40
        local = d / 'inputs' / 'profile.json'
        assert record['files'][str(local)]['sha256'] == wave.first.identity(profile)['sha256']
        assert str(profile) not in record['files']
        assert wave.read(d / 'runtime.json')['GEODML_EXECUTION_COMMIT'] == 'a' * 40


def test_relaunch_never_touches_live_completed_or_unknown_members(tmp_path, monkeypatch):
    states = {'101': 'FAILED', '103': 'COMPLETED', '104': 'FAILED', '105': 'CANCELLED'}
    options, jobs, profile, commands, released = relaunch_env(
        tmp_path, monkeypatch, states, live_members=(2,))
    wave.relaunch_qwen(options, 'a' * 40)
    assert released == ['300', '301']
    receipt = wave.read(options.output / 'relaunch-1.json')
    assert receipt['job_ids'] == released and receipt['maximum_gpu_hours'] == 24
    assert set(receipt['failed_jobs']) == {'101', '104'}
    assert receipt['member_states']['102'] == 'PENDING'
    for n in ('2', '3', '5'):
        d = options.output / f'qwen-{n}'
        assert wave.read(d / 'submission.json')['job_id'] == jobs[int(n)]
        assert not list(d.glob('submission-failed-*'))
        assert wave.read(d / 'preparation.json') == {'files': {str(profile): wave.first.identity(profile)}}


def test_relaunch_without_failures_is_a_clean_noop(tmp_path, monkeypatch):
    options, jobs, profile, commands, released = relaunch_env(
        tmp_path, monkeypatch, {str(100 + n): 'COMPLETED' for n in range(1, 6)})
    assert wave.relaunch_qwen(options, 'a' * 40)['status'] == 'no_failed_members'
    assert commands == [] and released == []
    assert not (options.output / 'relaunch-intent-1.json').exists()
    assert not (options.output / 'relaunch-1.json').exists()


def test_relaunch_stops_without_scheduler_evidence_and_keeps_receipts(tmp_path, monkeypatch):
    options, jobs, profile, commands, released = relaunch_env(tmp_path, monkeypatch, {'101': None})
    with pytest.raises(ValueError, match='No scheduler evidence'):
        wave.relaunch_qwen(options, 'a' * 40)
    assert commands == []
    assert wave.read(options.output / 'qwen-1' / 'submission.json')['job_id'] == '101'


def test_second_round_accepts_relaunched_receipts_and_only_the_failed_member(tmp_path, monkeypatch):
    states = {str(100 + n): 'FAILED' for n in range(1, 6)}
    options, jobs, profile, commands, released = relaunch_env(tmp_path, monkeypatch, states)
    wave.relaunch_qwen(options, 'a' * 40)
    assert released == ['300', '301', '302', '303', '304']
    # Round two: only job 304 (qwen-5) died; its siblings stay live in squeue.
    options.new_jobs[:] = [row for row in options.new_jobs if row['job_id'] != '304']
    states['304'] = 'FAILED'
    before = {n: wave.read(options.output / f'qwen-{n}' / 'submission.json')['job_id']
              for n in range(1, 6)}
    assert before == {1: '300', 2: '301', 3: '302', 4: '303', 5: '304'}
    wave.relaunch_qwen(options, 'a' * 40)
    assert released == ['300', '301', '302', '303', '304', '305']
    receipt = wave.read(options.output / 'relaunch-2.json')
    assert receipt['job_ids'] == ['305'] and receipt['maximum_gpu_hours'] == 12
    assert set(receipt['failed_jobs']) == {'304'}
    assert receipt['member_states'] == {'300': 'PENDING', '301': 'PENDING', '302': 'PENDING',
                                        '303': 'PENDING', '304': 'FAILED'}
    d5 = options.output / 'qwen-5'
    assert wave.read(d5 / 'submission-failed-304.json')['job_id'] == '304'
    assert wave.read(d5 / 'submission.json')['job_id'] == '305'
    assert wave.read(d5 / 'preparation.json')['wave_member'] == 'qwen-5-of-5-relaunch-2'
    for n in range(1, 5):
        d = options.output / f'qwen-{n}'
        assert wave.read(d / 'submission.json')['job_id'] == str(299 + n)
        assert not list(d.glob('submission-failed-3*'))
    assert any('--job-name=geodml-qwen-threehour-5-r2' in ' '.join(c) for c in commands)


def test_relaunch_guards_rounds_and_wave_ownership(tmp_path, monkeypatch):
    states = {str(100 + n): 'FAILED' for n in range(1, 6)}
    options, jobs, profile, commands, released = relaunch_env(tmp_path, monkeypatch, states)
    wave.save(options.output / 'relaunch-intent-1.json', {'members': []})
    with pytest.raises(ValueError, match='already attempted'):
        wave.relaunch_qwen(options, 'a' * 40)
    (options.output / 'relaunch-intent-1.json').unlink()
    wave.save(options.output / 'relaunch-1.json', {'job_ids': []})
    wave.relaunch_qwen(options, 'a' * 40)
    assert (options.output / 'relaunch-intent-2.json').exists()
    assert wave.read(options.output / 'relaunch-2.json')['job_ids'] == released
    assert all('--job-name=geodml-qwen-threehour-' in ' '.join(c) and '-r2' in ' '.join(c)
               for c in commands)
    wave.save(options.qwen_reference / 'expansion-qwen.json',
              {'output': str(tmp_path / 'elsewhere'), 'git_commit': 'b' * 40})
    with pytest.raises(ValueError, match='originally dispatched'):
        wave.relaunch_qwen(options, 'a' * 40)


def qwen_continuation_env(tmp_path, monkeypatch, live_jobs, members=5, maximum=10):
    options = args(tmp_path / 'round2')
    options.output.mkdir()
    options.qwen_reference = tmp_path / 'reference'
    options.environment_file = tmp_path / 'env.sh'
    options.account, options.partition = 'project', 'booster'
    options.round, options.members = 2, members
    options.maximum_concurrent = maximum
    options.allowed_jobs = list(live_jobs)
    runtime = {'GEODML_DATASET_ROOT': str(tmp_path / 'dataset'), 'ACL_ARR_VENV': '/venv'}
    wave.save(options.qwen_reference / 'runtime.json', runtime)
    reference = {'dataset_root': runtime['GEODML_DATASET_ROOT'], 'files': {},
                 'dataset_files': {}, 'external_files': {}}
    monkeypatch.setattr(wave, 'verify_reference', lambda p: (reference, {'job_id': '42'}))
    monkeypatch.setattr(wave, 'health', lambda *a: {})
    monkeypatch.setattr(wave.first, 'current_scheduler', lambda *a: {
        'complete': True, 'captured_at_epoch': int(wave.time.time()),
        'jobs': [{'job_id': j, 'state': 'RUNNING'} for j in live_jobs], 'owners': []})
    exchange = SimpleNamespace(snapshot=lambda: ('revision', {'hours': {}}))
    return options, exchange


def test_qwen_continuation_round_stages_a_fresh_approved_wave(tmp_path, monkeypatch):
    options, exchange = qwen_continuation_env(tmp_path, monkeypatch, ['201', '202'])
    captured = []
    monkeypatch.setattr(wave, 'submit', lambda *a: captured.append(a))
    wave.qwen(options, 'a' * 40, exchange)
    assert captured[0][3] == ['201', '202']
    entries = captured[0][1]
    assert [d.name for d, _, _ in entries] == [f'qwen-{n}' for n in range(1, 6)]
    for number, (directory, command, approved) in enumerate(entries, 1):
        assert f'--job-name=geodml-qwen-threehour-{number}-c2' in command
        assert '--time=03:00:00' in command and '--exclusive' in command
        assert approved['maximum_gpu_hours'] == 12
        preparation = wave.read(directory / 'preparation.json')
        assert preparation['wave_member'] == f'qwen-{number}-of-5-round-2'
        assert 'continuation round 2' in preparation['estimate']


def test_qwen_continuation_respects_the_concurrency_cap(tmp_path, monkeypatch):
    live = [str(200 + n) for n in range(1, 7)]
    options, exchange = qwen_continuation_env(tmp_path, monkeypatch, live)
    with pytest.raises(ValueError):
        wave.qwen(options, 'a' * 40, exchange)
    assert not list((tmp_path / 'round2').glob('qwen-*'))


def test_llama_groups_and_admission_scale_to_the_approved_member_count(tmp_path):
    state = {'hours': {}}
    for i in range(12):
        state['hours'][str(i)] = {'model': 'llama4', 'owner': None, 'status': 'available',
            'task_fingerprints': [f'{i}-{n}' for n in range(9000)], 'completed': [],
            'failed': [], 'priority_rank': i}
    assert wave.groups(state, 9) == [[str(i)] for i in range(9)]
    with pytest.raises(ValueError, match='Fewer than 13 eligible'):
        wave.groups(state, 13)
    fixture_state, attempts, snapshot, storage = fixture()
    options = args(tmp_path)
    options.members = 9
    for a in attempts:
        a['request']['approval'] = wave.approval(options, 'measured estimate')
    with pytest.raises(ValueError, match='Exactly 9 distinct'):
        wave.admit(fixture_state, attempts, snapshot, storage, options)


def test_guard_names_are_round_scoped():
    assert wave.guard_name('llama', 1) == 'expansion-llama.json'
    assert wave.guard_name('qwen', 1) == 'expansion-qwen.json'
    assert wave.guard_name('llama', 2) == 'expansion-llama-r2.json'
    assert wave.guard_name('qwen', 3) == 'expansion-qwen-r3.json'
