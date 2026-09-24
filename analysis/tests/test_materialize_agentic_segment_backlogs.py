"""Keyword-ordered immutable queues for homogeneous segments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_segment_plan import build_segment_plan
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.materialize_agentic_segment_backlogs import materialize


def _identity(task_id: str) -> ClaimIdentity:
    return ClaimIdentity(
        task_id=task_id,
        model_id="qwen38",
        model_revision="revision",
        protocol="experiment-v2",
        request_sha256=hashlib.sha256(task_id.encode()).hexdigest(),
    )


def test_materializer_orders_batch_forward_and_interactive_reverse(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    writer = FinalDatasetWriter(root, writer_id="bootstrap")
    memberships = [
        {"prompt_id": "p-alpha", "primary_keyword_id": "alpha",
         "keyword_ids": ["alpha"], "primary_priority_rank": 0},
        {"prompt_id": "p-beta", "primary_keyword_id": "beta",
         "keyword_ids": ["beta"], "primary_priority_rank": 1},
    ]
    for row in memberships:
        writer.append("keyword_memberships", row, transaction_id=row["prompt_id"])
    identities = {task: _identity(task) for task in ("alpha-a", "alpha-b", "beta-a")}
    for task_id, prompt_id in (
        ("alpha-a", "p-alpha"),
        ("alpha-b", "p-alpha"),
        ("beta-a", "p-beta"),
    ):
        writer.append(
            "task_definitions",
            {
                "task_id": task_id,
                "prompt_id": prompt_id,
                "model": "qwen38",
                "stage": "generation",
                "claim_identity": asdict(identities[task_id]),
                "runnable_task": {"cell_id": task_id},
            },
            transaction_id=task_id,
        )
    writer.seal()
    results = FinalDatasetWriter(root, writer_id="results")
    completed_reference = results.append(
        "generations", {"answer": "saved"}, transaction_id="alpha-a"
    )
    results.seal()
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    completed = ledger.claim(identities["alpha-a"], owner_id="old-job").claim
    ledger.transition(
        completed,
        state="completed",
        record_references=[completed_reference],
    )
    keyword_rows = [
        {"keyword_id": "alpha", "priority_rank": 0, "completed": 1,
         "active": 0, "blocked": 0, "eligible_remaining": 1,
         "estimated_remaining_seconds": 10, "task_set_ref": "alpha-tasks"},
        {"keyword_id": "beta", "priority_rank": 1, "completed": 0,
         "active": 0, "blocked": 0, "eligible_remaining": 1,
         "estimated_remaining_seconds": 10, "task_set_ref": "beta-tasks"},
    ]
    plan = build_segment_plan(
        audit_id="audit",
        ledger_sequence=2,
        acceptance_policy_id="v2",
        model="qwen38",
        keyword_rows=keyword_rows,
        segment_count=2,
        interactive_segment_count=1,
        approval_status="approved",
    ).to_dict()
    result = materialize(
        dataset_root=root, plan=plan, output=tmp_path / "queues", stripe_count=8
    )
    assert result["eligible_task_count"] == 2
    first, second = result["waves"]
    with open(first["backlog"]["path"]) as stream:
        forward = [json.loads(line)["cell_id"] for line in stream]
    with open(second["backlog"]["path"]) as stream:
        reverse = [json.loads(line)["cell_id"] for line in stream]
    assert forward == ["alpha-b"]
    assert reverse == ["beta-a"]
    assert [wave["primary_keyword_id"] for wave in result["waves"]] == [
        "alpha", "beta"
    ]
    assert all(wave["backlog"]["task_count"] == 1 for wave in result["waves"])


def test_batch_only_wave_keeps_ordered_spillover_keywords(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    writer = FinalDatasetWriter(root, writer_id="bootstrap")
    identities = {}
    for rank, keyword in enumerate(("alpha", "beta")):
        prompt_id = f"p-{keyword}"
        writer.append("keyword_memberships", {
            "prompt_id": prompt_id, "primary_keyword_id": keyword,
            "keyword_ids": [keyword], "primary_priority_rank": rank,
        }, transaction_id=prompt_id)
        identity = _identity(keyword)
        identities[keyword] = identity
        writer.append("task_definitions", {
            "task_id": keyword, "prompt_id": prompt_id, "model": "qwen38",
            "stage": "generation", "claim_identity": asdict(identity),
            "runnable_task": {"cell_id": keyword},
        }, transaction_id=keyword)
    writer.seal()
    rows = [{
        "keyword_id": keyword, "priority_rank": rank, "completed": 0,
        "active": 0, "blocked": 0, "eligible_remaining": 1,
        "estimated_remaining_seconds": 10, "task_set_ref": f"{keyword}-tasks",
    } for rank, keyword in enumerate(("alpha", "beta"))]
    plan = build_segment_plan(
        audit_id="audit", ledger_sequence=0, acceptance_policy_id="v2",
        model="qwen38", keyword_rows=rows, segment_count=2,
        approval_status="approved",
    ).to_dict()
    result = materialize(
        dataset_root=root, plan=plan, output=tmp_path / "queues", stripe_count=8
    )
    for wave in result["waves"]:
        with open(wave["backlog"]["path"]) as stream:
            assert [json.loads(line)["cell_id"] for line in stream] == [
                "alpha", "beta"
            ]


def test_materializer_gives_meeting_keyword_to_batch_workers(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    writer = FinalDatasetWriter(root, writer_id="bootstrap")
    writer.append(
        "keyword_memberships",
        {"prompt_id": "p-only", "primary_keyword_id": "only",
         "keyword_ids": ["only"], "primary_priority_rank": 0},
        transaction_id="p-only",
    )
    identity = _identity("only-task")
    writer.append(
        "task_definitions",
        {
            "task_id": "only-task",
            "prompt_id": "p-only",
            "model": "qwen38",
            "stage": "generation",
            "claim_identity": asdict(identity),
            "runnable_task": {"cell_id": "only-task"},
        },
        transaction_id="only-task",
    )
    writer.seal()
    plan = build_segment_plan(
        audit_id="audit",
        ledger_sequence=0,
        acceptance_policy_id="v2",
        model="qwen38",
        keyword_rows=[{
            "keyword_id": "only", "priority_rank": 0, "completed": 0,
            "active": 0, "blocked": 0, "eligible_remaining": 1,
            "estimated_remaining_seconds": 10, "task_set_ref": "only-tasks",
        }],
        segment_count=2,
        interactive_segment_count=1,
        approval_status="approved",
    ).to_dict()
    result = materialize(
        dataset_root=root, plan=plan, output=tmp_path / "queues", stripe_count=8
    )
    assert result["waves"][0]["backlog"]["task_count"] == 1
    assert result["waves"][0]["primary_keyword_id"] == "only"
    assert result["waves"][1]["backlog"]["task_count"] == 0
    assert result["waves"][1]["primary_keyword_id"] is None
    assert result["waves"][1]["queue_exhausted"] is True
