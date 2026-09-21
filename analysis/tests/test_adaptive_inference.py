"""Adaptive scheduling never weakens task identity or completion barriers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from analysis.interpretability.pipeline.adaptive_inference import (
    Progress,
    Queue,
    audit_registries,
    choose_queue,
    select_sweep_concurrency,
    stratified_sweep_sample,
    validate_queues,
)
from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
)


def identity(task: str, model: str = "model/llama") -> ClaimIdentity:
    return ClaimIdentity(
        task_id=task,
        model_id=model,
        model_revision="a" * 40,
        protocol="protocol-v1",
        request_sha256="b" * 64,
    )


def queues(tmp_path):
    generation = Queue(
        "original-llama",
        "llama",
        0,
        tmp_path / "llama",
        tuple(identity(f"generation-{i}") for i in range(2)),
    )
    judge = Queue(
        "original-judge",
        "nemotron",
        0,
        tmp_path / "judge",
        tuple(identity(f"judge-{i}", "model/nemotron") for i in range(2)),
        requires=("original-llama",),
    )
    overflow = Queue(
        "paired-qwen",
        "qwen",
        10,
        tmp_path / "qwen",
        (identity("overflow", "model/qwen"),),
    )
    return generation, judge, overflow


def test_completion_barrier_and_minimum_model_residence(tmp_path):
    generation, judge, overflow = queues(tmp_path)
    all_queues = (generation, judge, overflow)
    validate_queues(all_queues)
    progress = {
        generation.name: Progress(2, completed=1, missing=1),
        judge.name: Progress(2, missing=2),
        overflow.name: Progress(1, missing=1),
    }
    assert choose_queue(all_queues, progress).name == generation.name
    progress[generation.name] = Progress(2, completed=2)
    assert (
        choose_queue(
            all_queues, progress, current=overflow.name, residence_seconds=299
        ).name
        == overflow.name
    )
    assert (
        choose_queue(
            all_queues, progress, current=overflow.name, residence_seconds=300
        ).name
        == judge.name
    )


def test_failed_or_busy_generation_never_opens_judge_gate(tmp_path):
    generation, judge, _ = queues(tmp_path)
    for progress in (
        Progress(2, completed=1, failed=1),
        Progress(2, completed=1, busy=1),
    ):
        selected = choose_queue(
            (generation, judge),
            {
                generation.name: progress,
                judge.name: Progress(2, missing=2),
            },
        )
        assert selected is None


def test_duplicate_scientific_task_cannot_be_split_across_registries(tmp_path):
    task = identity("same")
    left = Queue("left", "llama", 0, tmp_path / "left", (task,))
    right = Queue("right", "llama", 0, tmp_path / "right", (task,))
    with pytest.raises(ValueError, match="separate claim roots"):
        validate_queues((left, right))


def test_registry_audit_unions_only_exact_identities_and_surfaces_conflict(tmp_path):
    task = identity("same")
    left, right = tmp_path / "left", tmp_path / "right"
    with InferenceClaimStore(left).try_claim(task) as claim:
        claim.commit({"answer": "one"})
    with InferenceClaimStore(right).try_claim(task) as claim:
        claim.commit({"answer": "one"})
    report = audit_registries((left, right), (task,))
    assert report["completed"] == 1
    assert report["duplicate_success_records"] == 1
    assert report["conflicts"] == []
    changed = replace(task, task_id="changed")
    with InferenceClaimStore(left).try_claim(changed) as claim:
        claim.commit({"answer": "one"})
    with InferenceClaimStore(right).try_claim(changed) as claim:
        claim.commit({"answer": "two"})
    assert audit_registries((left, right), (changed,))["conflicts"] == ["changed"]


def test_sweep_sample_is_deterministic_stratified_and_disjoint():
    counts = {f"task-{i:03d}": i + 1 for i in range(100)}
    first = stratified_sweep_sample(counts)
    assert first == stratified_sweep_sample(dict(reversed(list(counts.items()))))
    assert len(first["measured"]) == 64
    assert len(first["warmup"]) == 4
    assert not set(first["measured"]) & set(first["warmup"])
    quartiles = [
        sum(int(task[-3:]) in range(q * 25, (q + 1) * 25) for task in first["measured"])
        for q in range(4)
    ]
    assert quartiles == [16, 16, 16, 16]


def test_sweep_selects_fastest_complete_candidate_and_falls_back_to_four():
    shared = {
        "sample_sha256": "a" * 64,
        "profile_sha256": "b" * 64,
        "status": "complete",
        "succeeded": 64,
        "failed": 0,
        "warmup_succeeded": 4,
    }
    results = [
        {**shared, "concurrency": value, "measurement_seconds": seconds}
        for value, seconds in ((4, 40), (8, 20), (16, 20), (32, 80))
    ]
    assert (
        select_sweep_concurrency(
            results, sample_sha256="a" * 64, profile_sha256="b" * 64
        )["concurrency"]
        == 8
    )
    incomplete = [{**row, "status": "checkpointed"} for row in results]
    assert (
        select_sweep_concurrency(
            incomplete, sample_sha256="a" * 64, profile_sha256="b" * 64
        )["concurrency"]
        == 4
    )
