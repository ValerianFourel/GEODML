from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline import agentic_hour_sync as sync


class HTTPFailure(Exception):
    def __init__(self, status, headers=None):
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def clock(monkeypatch):
    elapsed = [0]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds
    monkeypatch.setattr(sync.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(sync.time, 'sleep', sleep)
    return elapsed, sleeps


@pytest.mark.parametrize('headers,expected', [({}, 3660), ({'Retry-After': '45'}, 45)])
def test_waits_then_retries_only_commit(monkeypatch, headers, expected):
    elapsed, sleeps = clock(monkeypatch)
    calls = []
    def commit():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPFailure(429, headers)
        return 'committed'
    assert sync.commit_with_cooldown(commit) == 'committed'
    assert len(calls) == 2
    assert elapsed[0] == expected
    assert max(sleeps) <= 30


def test_second_rate_limit_stops_and_permission_failure_never_waits(monkeypatch):
    elapsed, _ = clock(monkeypatch)
    calls = []
    def commit():
        calls.append(1)
        raise HTTPFailure(429, {'Retry-After': '1'})
    with pytest.raises(HTTPFailure):
        sync.commit_with_cooldown(commit)
    assert len(calls) == 2 and elapsed[0] == 1
    def forbidden():
        raise HTTPFailure(403)
    with pytest.raises(HTTPFailure):
        sync.commit_with_cooldown(forbidden)
    assert elapsed[0] == 1


def test_longer_server_cooldown_is_not_shortened(monkeypatch):
    elapsed, _ = clock(monkeypatch)
    def commit():
        raise HTTPFailure(429, {'Retry-After': '7200'})
    with pytest.raises(HTTPFailure):
        sync.commit_with_cooldown(commit)
    assert elapsed[0] == 0
