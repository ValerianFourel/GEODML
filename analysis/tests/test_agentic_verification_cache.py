"""Repeated preparations reuse bytes but always observe current task state."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_verification_cache import (
    VerificationCache,
)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def test_file_reuse_force_and_same_size_mutation_with_restored_mtime(tmp_path, monkeypatch):
    path = tmp_path / 'input'
    path.write_bytes(b'original')
    expected = identity(b'original')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, expected)
        assert cache.checked == 1
    original_open = Path.open
    reads = []
    def track(p, *args, **kwargs):
        if p == path:
            reads.append(p)
        return original_open(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', track)
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, expected)
        assert cache.reused == 1 and reads == []
    with VerificationCache(tmp_path, force=True) as cache:
        assert cache.file(path, expected)
        assert cache.checked == 1 and len(reads) == 1
    before = path.stat()
    path.write_bytes(b'corrupt!')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with VerificationCache(tmp_path) as cache:
        assert not cache.file(path, expected)
        assert cache.checked == 1


def test_changed_expectation_missing_symlink_and_invalid_receipt(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'original')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
        receipt = cache.path
    with VerificationCache(tmp_path) as cache:
        assert not cache.file(path, identity(b'different'))
    receipt.write_text('broken receipt')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
        assert cache.checked == 1
    moved = tmp_path / 'moved'
    path.rename(moved)
    with VerificationCache(tmp_path) as cache, pytest.raises(FileNotFoundError):
        cache.file(path, identity(b'original'))
    path.symlink_to(moved)
    with VerificationCache(tmp_path) as cache, pytest.raises(ValueError, match='symlink'):
        cache.file(path, identity(b'original'))


def test_verification_race_never_produces_receipt(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'original')
    def racing_check():
        path.write_bytes(b'changed')
        return True
    with pytest.raises(ValueError, match='changed during'), VerificationCache(tmp_path) as cache:
        cache.check('test', [path], racing_check)
    assert not cache.path.exists()


def test_inventory_reuses_payloads_but_refreshes_completions_and_claims(tmp_path, monkeypatch):
    from analysis.interpretability.pipeline import agentic_dataset
    from analysis.interpretability.pipeline.agentic_hours import inventory
    from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
    from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
    from analysis.tests.test_agentic_hours import complete, data
    root = data(tmp_path / 'dataset')
    tasks, _, _ = inventory(root, stripes=4)
    complete(root, tasks[0], 'first')
    original = agentic_dataset.verify_record_reference
    checked = []
    def track(*args, **kwargs):
        if kwargs.get('verification') is None:
            checked.append(args[1]['record_id'])
        return original(*args, **kwargs)
    monkeypatch.setattr(agentic_dataset, 'verify_record_reference', track)
    _, done, blocked = inventory(root, stripes=4, reuse_verified=True)
    assert done == {tasks[0]['fingerprint']} and blocked == set()
    assert len(checked) == 1
    checked.clear()
    complete(root, tasks[1], 'second')
    ledger = StripedTaskLedger(root / 'control/task-ledger', stripe_count=4)
    ledger.claim(ClaimIdentity(**tasks[2]['claim_identity']), owner_id='live')
    _, done, blocked = inventory(root, stripes=4, reuse_verified=True)
    assert done == {tasks[0]['fingerprint'], tasks[1]['fingerprint']}
    assert blocked == {tasks[2]['fingerprint']}
    assert len(checked) == 1  # Only the new result needed verification.
    checked.clear()
    inventory(root, stripes=4, reuse_verified=True)
    assert checked == []
    # A changed result cannot stay completed merely because an old receipt exists.
    result = next((root / 'data/generations').glob('part-first-*.jsonl'))
    result.write_bytes(result.read_bytes().replace(b'synthetic', b'corrupted'))
    _, done, blocked = inventory(root, stripes=4, reuse_verified=True)
    assert done == {tasks[1]['fingerprint']}
    assert blocked == {tasks[0]['fingerprint'], tasks[2]['fingerprint']}


def test_download_reuses_verified_payloads_and_rechecks_remote_on_request(tmp_path, monkeypatch):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.tests.test_agentic_hours import MemoryHub
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / 'journal')
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    (source / 'README.md').write_bytes(b'input data')
    bundle = exchange.upload(source, ['README.md'], outcomes={}, metadata={'kind': 'frozen-inputs'})
    exchange.download(bundle, destination, stripes=4)
    original = hub.read
    def metadata_only(name, revision):
        assert not name.startswith('exchange/objects/')
        return original(name, revision)
    monkeypatch.setattr(hub, 'read', metadata_only)
    original_open = Path.open
    def no_payload_read(p, *args, **kwargs):
        assert p != destination / 'README.md'
        return original_open(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_payload_read)
    exchange.download(bundle, destination, stripes=4)
    monkeypatch.setattr(Path, 'open', original_open)
    monkeypatch.setattr(hub, 'read', original)
    exchange.download(bundle, destination, stripes=4, verify_remote=True)
    (destination / 'README.md').write_bytes(b'bad input!')
    with pytest.raises(ValueError, match='corrupt'):
        exchange.download(bundle, destination, stripes=4)


def test_corrupted_receipt_content_cannot_forge_verification(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'original')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
    value = json.loads(cache.path.read_bytes())
    value['body']['root'] = 'a different dataset'
    cache.path.write_text(json.dumps(value))
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
        assert cache.checked == 1


def test_failed_forced_audit_invalidates_prior_receipt(tmp_path):
    path = tmp_path / 'input'
    path.write_bytes(b'original')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
    with pytest.raises(ValueError, match='suspected corruption'), VerificationCache(tmp_path, force=True):
        raise ValueError('suspected corruption')
    with VerificationCache(tmp_path) as cache:
        assert cache.file(path, identity(b'original'))
        assert cache.checked == 1


def test_frozen_inputs_reuse_search_bytes_and_force_audit_reads_them(tmp_path, monkeypatch):
    import time

    from analysis.scripts.prepare_shared_hour_inputs import stage, verify_inputs
    from analysis.tests.test_prepare_shared_hour_inputs import fixture
    source, configs, priority = fixture(tmp_path)
    root = tmp_path / 'mirror'
    stage(source, root, configs, priority, {
        'cluster': 'jupiter', 'complete': True, 'jobs': [], 'captured_at_epoch': int(time.time())}, stripes=4)
    manifest = next((root / 'artifacts/shared-preparations').glob('*.json'))
    report = verify_inputs(root, manifest, model='qwen38')
    search = Path(report['files']['SEARCH_AGENTIC_DDG_SNAPSHOT'])
    original = Path.open
    reads = []
    def track(path, *args, **kwargs):
        if path == search:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', track)
    assert verify_inputs(root, manifest, model='qwen38') == report
    assert reads == []
    assert verify_inputs(root, manifest, model='qwen38', full_audit=True) == report
    assert reads == [search]
