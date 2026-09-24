"""Terminal-owner recovery preserves saved responses and releases safe retries."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.reconcile_agentic_dataset import reconcile


def _identity(task_id):
    return ClaimIdentity(
        task_id=task_id,
        model_id="model",
        model_revision="revision",
        protocol="protocol",
        request_sha256=hashlib.sha256(task_id.encode()).hexdigest(),
    )


def _register(root, *identities):
    writer = FinalDatasetWriter(root, writer_id="registration")
    for identity in identities:
        writer.append(
            "task_definitions",
            {
                "task_id": identity.task_id,
                "prompt_id": "prompt",
                "model": "qwen38",
                "claim_identity": asdict(identity),
            },
            transaction_id=identity.task_id,
        )
    writer.seal()


def test_reconcile_completes_saved_response_and_releases_uncommitted_claim(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    saved, uncommitted = _identity("saved"), _identity("uncommitted")
    _register(root, saved, uncommitted)
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    saved_claim = ledger.claim(saved, owner_id="job42-worker0").claim
    ledger.transition(saved_claim, state="running")
    writer = FinalDatasetWriter(root, writer_id="job42-worker0")
    reference = writer.append(
        "generations", {"answer": "kept"}, transaction_id=saved_claim.fingerprint
    )
    writer.close()
    ledger.transition(saved_claim, state="result_saved", record_references=[reference])
    retry_claim = ledger.claim(uncommitted, owner_id="job43-worker0").claim
    ledger.transition(retry_claim, state="running")
    snapshot = {
        "complete": True,
        "owners": [
            {"owner_id": "job42-worker0", "job_id": "42", "state": "FAILED"},
            {"owner_id": "job43-worker0", "job_id": "43", "state": "TIMEOUT"},
        ],
    }
    dry = reconcile(
        root, scheduler_snapshot=snapshot, stripe_count=8, apply=False
    )
    assert dry["applied"] is False
    assert len(dry["actions"]) == 1
    assert dry["blocked"][0]["reason"] == (
        "saved_response_references_not_sealed_or_invalid"
    )
    applied = reconcile(
        root, scheduler_snapshot=snapshot, stripe_count=8, apply=True
    )
    assert {row["action"] for row in applied["actions"]} == {
        "complete_saved_result", "release_uncommitted_claim"
    }
    assert ledger.inspect(saved)["state"] == "completed"
    assert ledger.inspect(uncommitted)["state"] == "checkpointed"
    assert ledger.claim(saved, owner_id="new").status == "completed"
    assert ledger.claim(uncommitted, owner_id="new").status == "owned"


def test_reconcile_does_not_touch_live_owner(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    identity = _identity("live")
    _register(root, identity)
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    ledger.claim(identity, owner_id="job44-worker0")
    result = reconcile(
        root,
        scheduler_snapshot={
            "complete": True,
            "owners": [{
                "owner_id": "job44-worker0", "job_id": "44", "state": "RUNNING"
            }],
        },
        stripe_count=8,
        apply=True,
    )
    assert result["actions"] == []
    assert ledger.inspect(identity)["state"] == "claimed"
