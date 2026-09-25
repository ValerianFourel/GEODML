import pytest

from analysis.interpretability.pipeline import agentic_hour_sync as sync
from analysis.tests.test_agentic_hours import MemoryHub


def test_ninety_files_use_three_data_commits_and_one_manifest(tmp_path):
    root = tmp_path / 'data'
    (root / 'reports').mkdir(parents=True)
    names = []
    for i in range(90):
        name = f'reports/{i:03d}.json'
        (root / name).write_text(f'{{"value": {i}}}')
        names.append(name)
    hub = MemoryHub()
    exchange = sync.Exchange(hub, tmp_path / 'journal')
    bundle = exchange.upload(root, names, outcomes={}, metadata={})
    assert len(hub.versions) - 1 == 4
    assert len(exchange.manifest(bundle)['files']) == 90
    assert exchange.upload(root, names, outcomes={}, metadata={}) == bundle
    assert len(hub.versions) - 1 == 4


def test_lost_batch_response_resumes_without_early_manifest(tmp_path):
    root = tmp_path / 'data'
    (root / 'reports').mkdir(parents=True)
    for i in range(40):
        (root / 'reports' / str(i)).write_text(f'payload {i}')
    hub = MemoryHub()
    exchange = sync.Exchange(hub, tmp_path / 'journal')
    hub.lose_response = True
    with pytest.raises(ConnectionError):
        exchange.upload(root, [f'reports/{i}' for i in range(40)], outcomes={}, metadata={})
    assert len(hub.versions[-1]) == 32
    assert not any(name.startswith('exchange/bundles/') for name in hub.versions[-1])
    bundle = exchange.upload(root, [f'reports/{i}' for i in range(40)], outcomes={}, metadata={})
    assert len(exchange.manifest(bundle)['files']) == 40
    assert len(hub.versions) - 1 == 3


def test_byte_limit_and_oversized_file_keep_manifest_last(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, 'UPLOAD_BATCH_BYTES', 10)
    root = tmp_path / 'data'
    (root / 'reports').mkdir(parents=True)
    payloads = [b'a' * 6, b'b' * 6, b'c' * 20, b'd']
    names = [f'reports/{i}' for i in range(4)]
    for name, raw in zip(names, payloads, strict=True):
        (root / name).write_bytes(raw)
    hub = MemoryHub()
    exchange = sync.Exchange(hub, tmp_path / 'journal')
    bundle = exchange.upload(root, names, outcomes={}, metadata={})
    for before, after in zip(hub.versions, hub.versions[1:]):
        added = {k: v for k, v in after.items() if k not in before and k.startswith('exchange/objects/')}
        assert len(added) <= 1 or sum(map(len, added.values())) <= 10
    assert not any(k.startswith('exchange/bundles/') for k in hub.versions[-2])
    assert exchange.manifest(bundle)['files']['reports/2']['bytes'] == 20
