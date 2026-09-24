"""Direct final-format shard and durable transaction contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    JsonlShardWriter,
    finalize_orphaned_shard,
    initialize_dataset,
    recover_inprogress_writer,
    verify_record_reference,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity


def _identity(task="task-1"):
    return ClaimIdentity(
        task_id=task,
        model_id="model",
        model_revision="revision",
        protocol="protocol",
        request_sha256=hashlib.sha256(task.encode()).hexdigest(),
    )


def test_dataset_contract_and_sealed_shard_are_already_publication_format(tmp_path: Path):
    root = tmp_path / "dataset"
    contract = initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    assert initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2") == contract
    assert (root / "README.md").is_file()
    schema = json.loads((root / "schemas/records-v1.json").read_text())
    assert "transport_attempts" in schema["tables"]
    assert "judgments" in schema["tables"]
    with JsonlShardWriter(root, table="generations", writer_id="job123-worker0") as writer:
        reference = writer.append(
            {"answer": "text", "ranking": ["https://example.org"]},
            transaction_id="transaction-1",
        )
        manifest = writer.seal()
    shard = root / manifest["path"]
    row = json.loads(shard.read_text())
    assert manifest["rows"] == 1
    assert manifest["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert row["record_id"] == reference.record_id
    assert not list(root.rglob("*.inprogress"))


def test_contract_marker_fails_closed_when_initialization_files_are_missing(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    (root / "README.md").unlink()
    with pytest.raises(ValueError, match="incomplete initialization"):
        initialize_dataset(
            root, population_id="population", acceptance_policy_id="pilot-v2"
        )


def test_dataset_rejects_contract_changes_duplicates_secrets_and_partial_active_shards(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    with pytest.raises(ValueError, match="different immutable contract"):
        initialize_dataset(root, population_id="other", acceptance_policy_id="pilot-v2")
    writer = JsonlShardWriter(root, table="calls", writer_id="writer")
    reference = writer.append({"request": "ok"}, transaction_id="tx")
    with pytest.raises(ValueError, match="duplicate record"):
        writer.append({"request": "ok"}, transaction_id="tx", record_id=reference.record_id)
    with pytest.raises(ValueError, match="credential-shaped"):
        writer.append({"token": "hf_abcdefghijklmnopqrstuvwxyz123456"}, transaction_id="secret")
    writer.close()
    with writer.active_path.open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError, match="incomplete active shard"):
        JsonlShardWriter(root, table="calls", writer_id="writer")


def test_task_completion_follows_saved_records_and_prevents_duplicate_inference(tmp_path):
    ledger = StripedTaskLedger(tmp_path / "ledger", stripe_count=8)
    identity = _identity()
    first = ledger.claim(identity, owner_id="job-1")
    assert first.status == "owned"
    assert ledger.claim(identity, owner_id="job-2").status == "busy"
    ledger.transition(first.claim, state="running")
    ledger.transition(
        first.claim,
        state="result_saved",
        record_references=[{"table": "generations", "record_id": "record-1"}],
    )
    with pytest.raises(ValueError, match="record references"):
        ledger.transition(first.claim, state="completed")
    ledger.transition(
        first.claim,
        state="completed",
        record_references=[{"table": "generations", "record_id": "record-1"}],
    )
    completed = ledger.claim(identity, owner_id="job-2")
    assert completed.status == "completed"
    assert completed.claim is None


def test_stale_claim_requires_scheduler_confirmation_and_fences_old_owner(tmp_path):
    ledger = StripedTaskLedger(tmp_path / "ledger", stripe_count=8)
    identity = _identity()
    old = ledger.claim(identity, owner_id="job-old").claim
    with pytest.raises(ValueError, match="confirmed terminal"):
        ledger.release_stale(
            identity,
            expected_token=old.token,
            scheduler_confirmation={"owner_terminal": False},
        )
    ledger.release_stale(
        identity,
        expected_token=old.token,
        scheduler_confirmation={"owner_terminal": True, "scheduler_state": "FAILED"},
    )
    new = ledger.claim(identity, owner_id="job-new").claim
    assert new.generation == old.generation + 1
    with pytest.raises(RuntimeError, match="no longer current"):
        ledger.transition(old, state="running")


def test_partial_ledger_event_fails_closed(tmp_path):
    ledger = StripedTaskLedger(tmp_path / "ledger", stripe_count=1)
    identity = _identity()
    ledger.claim(identity, owner_id="job")
    event_path = next((tmp_path / "ledger/events").glob("*.jsonl"))
    with event_path.open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError, match="incomplete event"):
        ledger.inspect(identity)


def test_final_writer_rolls_over_and_resumes_active_final_format_shards(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    writer = FinalDatasetWriter(root, writer_id="job-1", maximum_shard_bytes=220)
    first = writer.append(
        "generations", {"answer": "a" * 30}, transaction_id="task-1"
    )
    second = writer.append(
        "generations", {"answer": "b" * 30}, transaction_id="task-2"
    )
    writer.close()
    resumed = FinalDatasetWriter(root, writer_id="job-1", maximum_shard_bytes=220)
    third = resumed.append(
        "generations", {"answer": "c" * 30}, transaction_id="task-3"
    )
    manifests = resumed.seal()
    assert first["shard_sequence"] <= second["shard_sequence"] <= third["shard_sequence"]
    assert sum(item["rows"] for item in manifests) >= 1
    assert not list(root.rglob("*.inprogress"))
    assert len(list((root / "data/generations").glob("*.jsonl"))) >= 2


def test_orphaned_sealed_shard_gets_checksum_manifest_after_restart(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    writer = JsonlShardWriter(root, table="calls", writer_id="job")
    writer.append({"request": "one"}, transaction_id="task")
    writer.stream.close()
    writer.active_path.replace(writer.sealed_path)
    manifest = finalize_orphaned_shard(
        root, table="calls", writer_id="job", shard_sequence=0
    )
    assert manifest["rows"] == 1
    assert writer.manifest_path.is_file()
    assert finalize_orphaned_shard(
        root, table="calls", writer_id="job", shard_sequence=0
    ) == manifest


def test_transport_audit_callback_requires_and_preserves_transaction_context(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    writer = FinalDatasetWriter(root, writer_id="job")
    with pytest.raises(ValueError, match="transaction ID"):
        writer.audit_callback({"task_context": {}, "event": "start"})
    writer.audit_callback({
        "task_context": {"transaction_id": "task-1", "task_id": "cell-1"},
        "event": "end",
        "raw_output": "saved model output",
    })
    manifest = writer.seal()[0]
    saved = json.loads((root / manifest["path"]).read_text())
    assert saved["transaction_id"] == "task-1"
    assert saved["row"]["raw_output"] == "saved model output"


def test_terminal_writer_recovery_seals_valid_records_without_rewriting(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    writer = FinalDatasetWriter(root, writer_id="job-42-worker0")
    reference = writer.append(
        "generations", {"answer": "durable"}, transaction_id="task-1"
    )
    writer.close()
    manifests = recover_inprogress_writer(root, writer_id="job-42-worker0")
    assert len(manifests) == 1
    assert verify_record_reference(root, reference)
    assert recover_inprogress_writer(root, writer_id="job-42-worker0") == []


def test_terminal_writer_recovery_fails_closed_on_partial_tail(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="pilot-v2")
    writer = JsonlShardWriter(root, table="generations", writer_id="job-7-worker0")
    writer.append({"answer": "durable"}, transaction_id="task-1")
    writer.close()
    with writer.active_path.open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError, match="incomplete active shard tail"):
        recover_inprogress_writer(root, writer_id="job-7-worker0")
    assert writer.active_path.exists()
