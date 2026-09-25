from copy import deepcopy

import pytest

from analysis.interpretability.pipeline.agentic_hour_runtime import admission
from analysis.scripts import dispatch_llama_five as dispatcher
from analysis.scripts.dispatch_llama_five import admit_five


def fixture():
    state = {'admission': {}, 'hours': {}}
    attempts = []
    for i in range(5):
        name = f'wave-{i}'
        owner = {'cluster': 'jupiter', 'attempt_id': name, 'generation': 1}
        state['hours'][name] = {'owner': owner, 'task_fingerprints': [f'cell-{i}']}
        request = {'attempt_id': name, 'cluster': 'jupiter', 'mode': 'batch',
                   'git_commit': 'a' * 40, 'hour_ids': [name],
                   'approval': {'status': 'approved', 'evidence': 'five concurrent jobs',
                                'estimate': 'measured prior run', 'walltime_seconds': 3600,
                                'maximum_gpu_hours': 4,
                                'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'}}}
        attempts.append({'attempt_id': name, 'request': request, 'model': 'llama4',
                         'cluster_profile': {'cluster': 'jupiter'}, 'owners': {name: owner},
                         'admission_ticket': None})
    snapshot = {'cluster': 'jupiter', 'complete': True, 'jobs': [], 'owners': [], 'captured_at_epoch': 1000}
    storage = {'cluster': 'jupiter', 'safe_to_admit': True, 'quota_verified': False, 'captured_at_epoch': 1000}
    return state, attempts, snapshot, storage


def test_exact_five_are_admitted_atomically_without_faking_quota():
    state, attempts, snapshot, storage = fixture()
    value = admit_five(state, attempts, snapshot, storage, now=1001, evidence='explicit exception')
    assert state['admission'] == {}
    tickets = value['admission']['jupiter']['tickets']
    assert sorted(tickets) == ['wave-0', 'wave-1', 'wave-2', 'wave-3', 'wave-4']
    assert all(t['quota_verified'] is False for t in tickets.values())
    assert value['admission']['jupiter']['finite_wave']['maximum_gpu_hours'] == 20
    with pytest.raises(ValueError, match='Prior admission'):
        admit_five(value, attempts, snapshot, storage, now=1001, evidence='explicit exception')


@pytest.mark.parametrize('case', ['six', 'live', 'stale', 'storage', 'overlap', 'approval', 'duration', 'owner'])
def test_wave_fails_closed(case):
    state, attempts, snapshot, storage = fixture()
    evidence = 'explicit exception'
    if case == 'six':
        attempts.append(deepcopy(attempts[0]))
    elif case == 'live':
        snapshot['jobs'] = [{'job_id': '42', 'state': 'RUNNING'}]
    elif case == 'stale':
        snapshot['captured_at_epoch'] = 0
    elif case == 'storage':
        storage['safe_to_admit'] = False
    elif case == 'overlap':
        state['hours']['wave-1']['task_fingerprints'] = ['cell-0']
    elif case == 'approval':
        evidence = ''
    elif case == 'duration':
        attempts[0]['request']['approval']['walltime_seconds'] = 7200
    elif case == 'owner':
        state['hours']['wave-0']['owner'] = None
    with pytest.raises(ValueError):
        admit_five(state, attempts, snapshot, storage, now=1001, evidence=evidence)
    assert state['admission'] == {}


def test_regular_admission_cannot_race_the_five_job_wave():
    state, attempts, snapshot, storage = fixture()
    state = admit_five(state, attempts, snapshot, storage, now=1001, evidence='explicit exception')
    storage['quota_verified'] = True
    request = deepcopy(attempts[0]['request'])
    request['attempt_id'] = 'sixth'
    with pytest.raises(ValueError, match='concurrent wave'):
        admission(state, request, snapshot, storage, now=1001)


@pytest.mark.parametrize('uncertain', [False, True])
def test_submit_once_records_intent_before_slurm_and_holds_until_all_five(tmp_path, monkeypatch, uncertain):
    import json
    import time
    _, attempts, _, _ = fixture()
    for i, a in enumerate(attempts):
        a['request']['attempt_dir'] = str(tmp_path / f'attempt-{i}')
        a['cluster_profile'].update(account='project', partition='booster')
    monkeypatch.setattr(dispatcher, 'idle', lambda since: {})
    monkeypatch.setattr(dispatcher, 'storage_health', lambda path: {'safe_to_admit': True})
    commands, releases = [], []
    def submit(command, **kwargs):
        assert (tmp_path / 'submission-intent.json').is_file()
        assert '--hold' in command and '--no-requeue' in command and '--time=01:00:00' in command
        assert '--gres=gpu:4' in command and '--exclusive' in command
        commands.append(command)
        if uncertain and len(commands) == 3:
            raise OSError('connection lost after possible submission')
        return str(100 + len(commands))
    monkeypatch.setattr(dispatcher.subprocess, 'check_output', submit)
    monkeypatch.setattr(dispatcher, 'current_scheduler', lambda since: {
        'complete': True, 'captured_at_epoch': int(time.time()),
        'jobs': [{'job_id': str(n), 'state': 'PENDING', 'held': True} for n in range(101, 106)]})
    def release(command, **kwargs):
        assert len(commands) == 5
        releases.append(command)
    monkeypatch.setattr(dispatcher.subprocess, 'run', release)
    if uncertain:
        with pytest.raises(OSError):
            dispatcher.submit_held(tmp_path, attempts, since='2026-09-01', wave_id='wave', dataset=tmp_path)
        assert releases == []
    else:
        dispatcher.submit_held(tmp_path, attempts, since='2026-09-01', wave_id='wave', dataset=tmp_path)
        assert releases == [['scontrol', 'release', str(n)] for n in range(101, 106)]
        assert json.loads((tmp_path / 'submitted.json').read_text())['job_ids'] == ['101', '102', '103', '104', '105']
    before = len(commands)
    with pytest.raises(ValueError, match='unsubmitted'):
        dispatcher.submit_held(tmp_path, attempts, since='2026-09-01', wave_id='wave', dataset=tmp_path)
    assert len(commands) == before
