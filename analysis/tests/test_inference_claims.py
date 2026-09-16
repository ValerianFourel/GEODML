"""Shared task ownership survives concurrent workers and process termination."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
from dataclasses import asdict, replace

import pytest

from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
)


def _identity():
    return ClaimIdentity(
        task_id="task-1", model_id="test/model", model_revision="a" * 40,
        protocol="judge-v1", request_sha256="b" * 64,
    )


def _validate(outcome):
    if outcome != {"answer": "validated"}:
        raise ValueError("invalid answer")


def _worker(root, identity, messages, release, commit):
    store = InferenceClaimStore(root)
    with store.try_claim(identity, validate=_validate) as claim:
        if claim.status == "owned" and commit:
            claim.commit({"answer": "validated"})
            messages.put("committed")
        else:
            messages.put(claim.status)
        if claim.status != "busy" and not release.wait(15):
            raise RuntimeError("test parent did not release worker")


def test_owned_claim_commits_once_then_other_store_reuses(tmp_path):
    root = tmp_path / "shared"
    first = InferenceClaimStore(root)
    with first.try_claim(_identity(), validate=_validate) as claim:
        assert claim.status == "owned"
        assert claim.outcome is None
        with InferenceClaimStore(root).try_claim(asdict(_identity())) as competing:
            assert competing.status == "busy"
            with pytest.raises(RuntimeError):
                competing.commit({"answer": "validated"})
        claim.commit({"answer": "validated"})
        with pytest.raises(RuntimeError):
            claim.commit({"answer": "validated"})
        # Even a committed result stays locked until its owner leaves the call.
        with first.try_claim(_identity()) as competing:
            assert competing.status == "busy"
    with InferenceClaimStore(root).try_claim(_identity(), validate=_validate) as reused:
        assert reused.status == "completed"
        assert reused.outcome == {"answer": "validated"}
        assert reused.fingerprint == claim.fingerprint
        with pytest.raises(RuntimeError):
            reused.commit({"answer": "validated"})
    with pytest.raises(RuntimeError):
        claim.commit({"answer": "validated"})


def test_uncommitted_exception_releases_claim_for_retry(tmp_path):
    store = InferenceClaimStore(tmp_path)
    with pytest.raises(ValueError, match="failed inference"), store.try_claim(_identity()) as claim:
        assert claim.status == "owned"
        raise ValueError("failed inference")
    with store.try_claim(_identity()) as retry:
        assert retry.status == "owned"


def test_terminal_failure_is_durable_and_not_executable_again(tmp_path):
    store = InferenceClaimStore(tmp_path)
    failure = {"error": "bounded attempts exhausted", "attempts": 3}
    def validate_failure(value):
        if value != failure:
            raise ValueError("invalid terminal failure")
    with store.try_claim(_identity(), validate=_validate, validate_failure=validate_failure) as claim:
        with pytest.raises(ValueError, match="invalid terminal failure"):
            claim.fail({"error": "wrong"})
        claim.fail(failure)
        assert claim.status == "failed"
        assert claim.outcome is None
        assert claim.failure == failure
        with store.try_claim(_identity()) as competing:
            assert competing.status == "busy"
        with pytest.raises(RuntimeError):
            claim.commit({"answer": "validated"})
    with InferenceClaimStore(tmp_path).try_claim(
        _identity(), validate=_validate, validate_failure=validate_failure,
    ) as later:
        assert later.status == "failed"
        assert later.failure == failure
        assert later.outcome is None
        with pytest.raises(RuntimeError):
            later.fail(failure)


def test_terminal_failure_corruption_is_not_retried(tmp_path):
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity()) as claim:
        claim.fail({"error": "bounded failure"})
    path, = tmp_path.rglob("*.failed.json")
    saved = json.loads(path.read_text())
    saved["outcome"]["error"] = "tampered"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="envelope"), store.try_claim(_identity()):
        pytest.fail("corrupt failure became executable")


def test_failure_and_success_records_for_same_identity_fail_closed(tmp_path):
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity()) as claim:
        claim.commit({"answer": "validated"})
    success, = tmp_path.rglob("*.json")
    success.with_suffix(".failed.json").write_text("{}")
    with pytest.raises(ValueError, match="both success and failure"), store.try_claim(_identity()):
        pytest.fail("contradictory records were accepted")


@pytest.mark.parametrize("field,value", [
    ("task_id", "task-2"), ("model_id", "another/model"),
    ("model_revision", "c" * 40), ("protocol", "judge-v2"),
    ("request_sha256", "c" * 64),
])
def test_every_scientific_identity_field_separates_claims(tmp_path, field, value):
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity()) as original:
        original.commit({"answer": "validated"})
    with store.try_claim(replace(_identity(), **{field: value})) as changed:
        assert changed.status == "owned"
        assert changed.fingerprint != original.fingerprint


@pytest.mark.parametrize("extra", ["worker_count", "output_dir"])
def test_worker_layout_cannot_enter_claim_identity(tmp_path, extra):
    identity = {**asdict(_identity()), extra: "different-worker-layout"}
    with pytest.raises(ValueError, match="identity"), InferenceClaimStore(tmp_path).try_claim(identity):
        pass


def test_validation_runs_before_commit_and_after_reuse(tmp_path):
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity(), validate=_validate) as claim:
        with pytest.raises(ValueError, match="invalid answer"):
            claim.commit({"answer": "incorrect"})
        claim.commit({"answer": "validated"})
    def reject(_):
        raise ValueError("caller rejects stored outcome")
    with pytest.raises(ValueError, match="caller rejects"), store.try_claim(_identity(), validate=reject):
        pass
    with store.try_claim(_identity(), validate=_validate) as claim:
        assert claim.status == "completed"


@pytest.mark.parametrize("corruption", ["json", "duplicate_json", "identity", "identity_hash", "outcome", "format"])
def test_corrupt_or_mismatched_record_fails_closed(tmp_path, corruption):
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity()) as claim:
        claim.commit({"answer": "validated"})
    path, = tmp_path.rglob("*.json")
    record = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("broken json")
    elif corruption == "duplicate_json":
        text = path.read_text()
        path.write_text(text.replace('"format_version":', '"format_version":"invalid","format_version":', 1))
    else:
        if corruption == "identity":
            record["identity"]["model_id"] = "another/model"
            raw = json.dumps(record["identity"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            record["identity_sha256"] = hashlib.sha256(raw).hexdigest()
        elif corruption == "identity_hash":
            record["identity_sha256"] = "c" * 64
        elif corruption == "outcome":
            record["outcome"]["answer"] = "tampered"
        else:
            record["format_version"] = "unsupported"
        path.write_text(json.dumps(record))
    original = path.read_bytes()
    with pytest.raises(ValueError), store.try_claim(_identity()):
        pytest.fail("corrupt record was treated as executable work")
    assert path.read_bytes() == original


def test_commit_syncs_file_before_rename_and_directory_after_rename(tmp_path, monkeypatch):
    import analysis.interpretability.pipeline.inference_claims as claims

    original_fsync = os.fsync
    original_replace = os.replace
    events = []
    def sync(descriptor):
        events.append("directory_fsync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file_fsync")
        original_fsync(descriptor)
    def rename(source, destination):
        events.append("rename")
        original_replace(source, destination)
    store = InferenceClaimStore(tmp_path)
    with store.try_claim(_identity()) as claim:
        monkeypatch.setattr(claims.os, "fsync", sync)
        monkeypatch.setattr(claims.os, "replace", rename)
        claim.commit({"answer": "validated"})
        assert events == ["file_fsync", "rename", "directory_fsync"]
    with store.try_claim(_identity(), validate=_validate) as reused:
        assert reused.status == "completed"
        assert reused.outcome == {"answer": "validated"}


def test_failed_file_sync_does_not_publish_or_allow_a_second_commit(tmp_path, monkeypatch):
    import analysis.interpretability.pipeline.inference_claims as claims

    store = InferenceClaimStore(tmp_path)
    original_fsync = os.fsync
    def fail(_):
        raise OSError("disk write failed")
    with store.try_claim(_identity()) as claim:
        monkeypatch.setattr(claims.os, "fsync", fail)
        with pytest.raises(OSError, match="disk write failed"):
            claim.commit({"answer": "validated"})
        with pytest.raises(RuntimeError):
            claim.commit({"answer": "validated"})
        monkeypatch.setattr(claims.os, "fsync", original_fsync)
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    with store.try_claim(_identity()) as retry:
        assert retry.status == "owned"


def test_completed_claim_survives_local_journal_failure(tmp_path):
    store = InferenceClaimStore(tmp_path)
    with pytest.raises(OSError, match="local journal failed"), store.try_claim(_identity()) as claim:
        claim.commit({"answer": "validated"})
        raise OSError("local journal failed")
    with store.try_claim(_identity(), validate=_validate) as retry:
        assert retry.status == "completed"
        assert retry.outcome == {"answer": "validated"}


def test_four_processes_cannot_own_one_task_at_the_same_time(tmp_path):
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    release = context.Event()
    workers = [context.Process(target=_worker, args=(str(tmp_path), asdict(_identity()), messages, release, False))
               for _ in range(4)]
    try:
        for worker in workers:
            worker.start()
        statuses = [messages.get(timeout=15) for _ in workers]
        assert sorted(statuses) == ["busy", "busy", "busy", "owned"]
        # An ancient lock-file timestamp never authorizes stealing a live lock.
        for path in tmp_path.rglob("*.lock"):
            os.utime(path, (1, 1))
        with InferenceClaimStore(tmp_path).try_claim(_identity()) as claim:
            assert claim.status == "busy"
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        messages.close()
    assert all(worker.exitcode == 0 for worker in workers)


@pytest.mark.parametrize("committed", [False, True])
def test_process_death_releases_lock_and_preserves_committed_outcome(tmp_path, committed):
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    release = context.Event()
    worker = context.Process(target=_worker, args=(str(tmp_path), asdict(_identity()), messages, release, committed))
    worker.start()
    try:
        assert messages.get(timeout=15) == ("committed" if committed else "owned")
        worker.kill()
        worker.join(timeout=10)
        assert not worker.is_alive()
        with InferenceClaimStore(tmp_path).try_claim(_identity(), validate=_validate) as claim:
            assert claim.status == ("completed" if committed else "owned")
            assert claim.outcome == ({"answer": "validated"} if committed else None)
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)
        messages.close()
