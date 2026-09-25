"""Reservation checks use site scheduler evidence, not assumed access."""
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline import agentic_hour_runtime as runtime


def test_horeka_reservation_requires_active_matching_access(monkeypatch):
    now = time.time()
    row = {'ReservationName': 'casualnet', 'State': 'ACTIVE', 'Flags': 'SPEC_NODES',
           'PartitionName': 'accelerated', 'Accounts': 'project', 'Users': '(null)',
           'StartTime': datetime.fromtimestamp(now-100, timezone.utc).isoformat(),
           'EndTime': datetime.fromtimestamp(now+7200, timezone.utc).isoformat()}
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=' '.join(f'{k}={v}' for k,v in row.items()))
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    request = {'approval': {'walltime_seconds': 3600}}
    profile = {'cluster': 'horeka', 'reservation': 'casualnet', 'partition': 'accelerated', 'account': 'project'}
    runtime.validate_reservation(request, profile)
    assert calls == [['scontrol', '-o', 'show', 'reservation', 'casualnet']]
    row['Accounts'] = 'other'
    with pytest.raises(ValueError, match='access'):
        runtime.validate_reservation(request, profile)
    row['Accounts'] = 'project'
    row['EndTime'] = datetime.fromtimestamp(now+10, timezone.utc).isoformat()
    with pytest.raises(ValueError, match='validity'):
        runtime.validate_reservation(request, profile)
    row['State'] = 'INACTIVE'
    with pytest.raises(ValueError, match='not active'):
        runtime.validate_reservation(request, profile)


def test_unconfigured_reservation_does_not_query_slurm(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected scheduler query'))
    runtime.validate_reservation({}, {'cluster': 'jupiter'})


def test_three_hour_interactive_allocation_requires_matching_explicit_extension(tmp_path):
    request = {'attempt_id': 'qwen-interactive', 'cluster': 'horeka', 'mode': 'interactive',
               'git_commit': 'a' * 40, 'hour_ids': ['hour-3', 'hour-4', 'hour-5'],
               'approval': {'status': 'approved', 'evidence': 'Three hours requested',
                            'estimate': 'One node, four A100; package throughput unknown',
                            'walltime_seconds': 10800, 'maximum_gpu_hours': 12,
                            'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'}}}
    profile = {'cluster': 'horeka', 'account': 'project', 'partition': 'accelerated'}
    with pytest.raises(ValueError, match='extended-walltime'):
        runtime.allocation_command(request, profile, tmp_path)
    request['approval']['extended_walltime_approval'] = {
        'walltime_seconds': 10800, 'evidence': 'Explicit three-hour interactive approval'}
    command = runtime.allocation_command(request, profile, tmp_path)
    assert command[0] == 'salloc'
    assert '--time=03:00:00' in command
    assert '--gres=gpu:4' in command
    request['approval']['maximum_gpu_hours'] = 4
    with pytest.raises(ValueError, match='GPU-hour'):
        runtime.allocation_command(request, profile, tmp_path)
    request['approval']['maximum_gpu_hours'] = 12
    request['approval']['extended_walltime_approval']['walltime_seconds'] = 7200
    with pytest.raises(ValueError, match='extended-walltime'):
        runtime.allocation_command(request, profile, tmp_path)
