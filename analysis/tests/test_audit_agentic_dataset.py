"""Incremental audits read final-format shards and the bounded task ledger."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.audit_agentic_dataset import build_audit


def _identity(task_id: str, model: str) -> ClaimIdentity:
    return ClaimIdentity(
        task_id=task_id,
        model_id=model,
        model_revision="revision-1",
        protocol="experiment-v2",
        request_sha256=(task_id.encode().hex() + "0" * 64)[:64],
    )


def _dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    initialize_dataset(
        root, population_id="population-26009", acceptance_policy_id="experiment-v2"
    )
    identities = {
        "q-complete": _identity("q-complete", "qwen38"),
        "q-active": _identity("q-active", "qwen38"),
        "q-eligible": _identity("q-eligible", "qwen38"),
        "q-failed": _identity("q-failed", "qwen38"),
        "l-complete": _identity("l-complete", "llama4"),
        "n-ready": _identity("n-ready", "nemotron"),
    }
    memberships = [
        {"prompt_id": "prompt-a", "keyword_ids": ["alpha"],
         "primary_keyword_id": "alpha", "primary_priority_rank": 0},
        {"prompt_id": "prompt-b", "keyword_ids": ["beta"],
         "primary_keyword_id": "beta", "primary_priority_rank": 1},
    ]
    tasks = [
        {"task_id": "q-complete", "prompt_id": "prompt-a", "model": "qwen38",
         "stage": "generation", "claim_identity": asdict(identities["q-complete"])},
        {"task_id": "q-active", "prompt_id": "prompt-a", "model": "qwen38",
         "stage": "generation", "claim_identity": asdict(identities["q-active"])},
        {"task_id": "q-eligible", "prompt_id": "prompt-b", "model": "qwen38",
         "stage": "generation", "claim_identity": asdict(identities["q-eligible"])},
        {"task_id": "q-failed", "prompt_id": "prompt-b", "model": "qwen38",
         "stage": "generation", "claim_identity": asdict(identities["q-failed"])},
        {"task_id": "l-complete", "prompt_id": "prompt-b", "model": "llama4",
         "stage": "generation", "claim_identity": asdict(identities["l-complete"])},
        {"task_id": "n-ready", "prompt_id": "prompt-b", "model": "nemotron",
         "stage": "bulk_judging",
         "dependency_fingerprints": [identity_fingerprint(identities["l-complete"])],
         "claim_identity": asdict(identities["n-ready"])},
    ]
    writer = FinalDatasetWriter(root, writer_id="import")
    for number, row in enumerate(memberships):
        writer.append("keyword_memberships", row, transaction_id=f"membership-{number}")
    for row in tasks:
        writer.append("task_definitions", row, transaction_id=row["task_id"])
    writer.seal()
    results = FinalDatasetWriter(root, writer_id="results")
    q_reference = results.append(
        "generations", {"answer": "q"}, transaction_id="q-complete"
    )
    l_reference = results.append(
        "generations", {"answer": "l"}, transaction_id="l-complete"
    )
    results.seal()
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    complete = ledger.claim(identities["q-complete"], owner_id="job-1").claim
    ledger.transition(complete, state="running")
    ledger.transition(
        complete,
        state="completed",
        record_references=[q_reference],
    )
    active = ledger.claim(identities["q-active"], owner_id="job-2").claim
    ledger.transition(active, state="running")
    failed = ledger.claim(identities["q-failed"], owner_id="job-3").claim
    ledger.transition(failed, state="terminal_failed", detail={"error": "invalid"})
    llama = ledger.claim(identities["l-complete"], owner_id="job-4").claim
    ledger.transition(
        llama,
        state="completed",
        record_references=[l_reference],
    )
    return root


def test_incremental_audit_counts_each_task_once_and_is_idempotent(tmp_path):
    root = _dataset(tmp_path)
    first = build_audit(root, model="qwen38", stripe_count=8)
    second = build_audit(root, model="qwen38", stripe_count=8)
    assert first == second
    assert first["task_count"] == 4
    assert first["completed"] == 1
    assert first["active"] == 1
    assert first["blocked"] == 1
    assert first["eligible_remaining"] == 1
    assert [row["keyword_id"] for row in first["keywords"]] == ["alpha", "beta"]
    assert [row["priority_rank"] for row in first["keywords"]] == [0, 1]
    assert len(first["unresolved_failures"]) == 1


def test_cross_model_dependency_can_make_nemotron_task_eligible(tmp_path):
    root = _dataset(tmp_path)
    audit = build_audit(root, model="nemotron", stripe_count=8)
    assert audit["task_count"] == 1
    assert audit["eligible_remaining"] == 1
    assert audit["blocked"] == 0


def test_incomplete_scheduler_snapshot_fails_closed(tmp_path):
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match="marked complete"):
        build_audit(
            root,
            model="qwen38",
            stripe_count=8,
            scheduler_snapshot={"complete": False, "jobs": []},
        )


def test_audit_summarizes_measured_throughput_by_keyword_and_method(tmp_path):
    root = _dataset(tmp_path)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/throughput.json").write_text(json.dumps({
        "measurements": [
            {
                "model": "qwen38",
                "keyword_id": "alpha",
                "method": "Parallel-Expansion-v1",
                "completed_tasks": 2,
                "work_seconds": 20,
            },
            {
                "model": "qwen38",
                "keyword_id": "alpha",
                "method": "Reactive-Snippet-Loop-v1",
                "completed_tasks": 2,
                "work_seconds": 40,
            },
            {
                "model": "llama4",
                "keyword_id": "alpha",
                "method": "Parallel-Expansion-v1",
                "completed_tasks": 1,
                "work_seconds": 99,
            },
        ]
    }))
    audit = build_audit(root, model="qwen38", stripe_count=8)
    assert audit["throughput"]["aggregate"] == {
        "measurement_count": 2,
        "minimum_seconds_per_task": 10,
        "median_seconds_per_task": 15,
        "maximum_seconds_per_task": 20,
    }
    assert len(audit["throughput"]["groups"]) == 2
    assert audit["keywords"][1]["estimated_remaining_seconds"] == 15
