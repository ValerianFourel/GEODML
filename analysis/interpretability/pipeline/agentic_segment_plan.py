"""Deterministic one-hour plans for homogeneous GEODML inference waves.

This module plans work only.  It does not call Slurm or mutate task state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

FORMAT_VERSION = "geodml-agentic-segment-plan-v1"
SUPPORTED_MODELS = frozenset({"qwen38", "llama4", "nemotron"})
SUPPORTED_STAGES = {
    "qwen38": "generation",
    "llama4": "generation",
    "nemotron": "bulk_judging",
}
MAX_WALLTIME_SECONDS = 3600
DEFAULT_MAX_CONCURRENT_ALLOCATIONS = 5
DEFAULT_START_GAP_SECONDS = 600
DEFAULT_DRAIN_SECONDS = 300


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


@dataclass(frozen=True)
class KeywordProgress:
    keyword_id: str
    priority_rank: int
    completed: int
    active: int
    blocked: int
    eligible_remaining: int
    estimated_remaining_seconds: float | None
    task_set_ref: str


@dataclass(frozen=True)
class Segment:
    segment_id: str
    ordinal: int
    mode: str
    direction: str
    model: str
    stage: str
    walltime_seconds: int
    drain_seconds: int
    keyword_ids: tuple[str, ...]
    primary_task_set_ref: str
    spillover_task_set_ref: str
    spillover_policy: str
    approval_status: str
    nodes: int
    gpus: int
    cpus: int
    memory: str
    proposed_admission_order: int


@dataclass(frozen=True)
class SegmentPlan:
    plan_id: str
    format_version: str
    audit_id: str
    ledger_sequence: int
    acceptance_policy_id: str
    model: str
    stage: str
    priority_policy: str
    keyword_priority: tuple[KeywordProgress, ...]
    segment_count: int
    interactive_segment_count: int
    walltime_seconds: int
    drain_seconds: int
    max_concurrent_allocations: int
    start_gap_seconds: int
    maximum_node_hours: float
    maximum_gpu_hours: float
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "keyword_priority": [asdict(row) for row in self.keyword_priority],
            "segments": [asdict(row) for row in self.segments],
        }


def build_keyword_priority(rows: Sequence[Mapping[str, Any]]) -> tuple[KeywordProgress, ...]:
    """Freeze configured keyword priority without inventing memberships."""

    progress: list[KeywordProgress] = []
    seen: set[str] = set()
    seen_ranks: set[int] = set()
    for number, raw in enumerate(rows, 1):
        keyword_id = raw.get("keyword_id")
        if not isinstance(keyword_id, str) or not keyword_id.strip():
            raise ValueError(f"keyword row {number} lacks a keyword_id")
        if keyword_id in seen:
            raise ValueError(f"duplicate keyword_id: {keyword_id}")
        seen.add(keyword_id)
        priority_rank = _integer(raw.get("priority_rank"), "priority_rank")
        if priority_rank in seen_ranks:
            raise ValueError(f"duplicate keyword priority_rank: {priority_rank}")
        seen_ranks.add(priority_rank)
        estimate = raw.get("estimated_remaining_seconds")
        if estimate is not None and (
            isinstance(estimate, bool)
            or not isinstance(estimate, (int, float))
            or estimate < 0
        ):
            raise ValueError("estimated_remaining_seconds must be non-negative or null")
        task_set_ref = raw.get("task_set_ref")
        if not isinstance(task_set_ref, str) or not task_set_ref:
            raise ValueError(f"keyword {keyword_id} lacks a stable task_set_ref")
        progress.append(
            KeywordProgress(
                keyword_id=keyword_id,
                priority_rank=priority_rank,
                completed=_integer(raw.get("completed"), "completed"),
                active=_integer(raw.get("active"), "active"),
                blocked=_integer(raw.get("blocked"), "blocked"),
                eligible_remaining=_integer(
                    raw.get("eligible_remaining"), "eligible_remaining"
                ),
                estimated_remaining_seconds=(
                    None if estimate is None else float(estimate)
                ),
                task_set_ref=task_set_ref,
            )
        )

    def priority(row: KeywordProgress) -> tuple[object, ...]:
        actionable = row.eligible_remaining > 0
        estimated = row.estimated_remaining_seconds is not None
        return (
            row.priority_rank,
            not actionable,
            not estimated,
            row.estimated_remaining_seconds if estimated else row.eligible_remaining,
            row.eligible_remaining,
            row.keyword_id,
        )

    return tuple(sorted(progress, key=priority))


def build_segment_plan(
    *,
    audit_id: str,
    ledger_sequence: int,
    acceptance_policy_id: str,
    model: str,
    keyword_rows: Sequence[Mapping[str, Any]],
    segment_count: int,
    interactive_segment_count: int = 0,
    walltime_seconds: int = MAX_WALLTIME_SECONDS,
    drain_seconds: int = DEFAULT_DRAIN_SECONDS,
    max_concurrent_allocations: int = DEFAULT_MAX_CONCURRENT_ALLOCATIONS,
    start_gap_seconds: int = DEFAULT_START_GAP_SECONDS,
    approval_status: str = "proposed",
    nodes: int = 1,
    gpus: int = 4,
    cpus: int = 32,
    memory: str = "all",
) -> SegmentPlan:
    """Build a finite single-model wave with forward and reverse workers."""

    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {model}")
    segment_count = _integer(segment_count, "segment_count", minimum=1)
    interactive_segment_count = _integer(
        interactive_segment_count, "interactive_segment_count"
    )
    if interactive_segment_count > segment_count:
        raise ValueError("interactive_segment_count exceeds segment_count")
    walltime_seconds = _integer(walltime_seconds, "walltime_seconds", minimum=1)
    if walltime_seconds > MAX_WALLTIME_SECONDS:
        raise ValueError("segment walltime exceeds the default one-hour limit")
    drain_seconds = _integer(drain_seconds, "drain_seconds")
    if drain_seconds >= walltime_seconds:
        raise ValueError("drain_seconds must fit within the segment walltime")
    max_concurrent_allocations = _integer(
        max_concurrent_allocations, "max_concurrent_allocations", minimum=1
    )
    if max_concurrent_allocations > DEFAULT_MAX_CONCURRENT_ALLOCATIONS:
        raise ValueError("concurrency exceeds the default five-allocation limit")
    start_gap_seconds = _integer(start_gap_seconds, "start_gap_seconds", minimum=600)
    nodes = _integer(nodes, "nodes", minimum=1)
    gpus = _integer(gpus, "gpus", minimum=1)
    cpus = _integer(cpus, "cpus", minimum=1)
    if approval_status not in {"proposed", "approved"}:
        raise ValueError("approval_status must be proposed or approved")
    if not all(
        isinstance(value, str) and value
        for value in (audit_id, acceptance_policy_id, memory)
    ):
        raise ValueError("audit, acceptance policy, and memory must be non-empty")
    ledger_sequence = _integer(ledger_sequence, "ledger_sequence")
    keyword_priority = build_keyword_priority(keyword_rows)
    if not keyword_priority:
        raise ValueError("a segment plan requires at least one keyword")
    approved_backlog = "task-set-" + _digest(
        [row.task_set_ref for row in keyword_priority]
    )[:24]
    forward = tuple(row.keyword_id for row in keyword_priority)
    reverse = tuple(reversed(forward))
    batch_count = segment_count - interactive_segment_count
    identity = {
        "format_version": FORMAT_VERSION,
        "audit_id": audit_id,
        "ledger_sequence": ledger_sequence,
        "acceptance_policy_id": acceptance_policy_id,
        "model": model,
        "stage": SUPPORTED_STAGES[model],
        "keyword_priority": [asdict(row) for row in keyword_priority],
        "segment_count": segment_count,
        "interactive_segment_count": interactive_segment_count,
        "walltime_seconds": walltime_seconds,
        "drain_seconds": drain_seconds,
        "max_concurrent_allocations": max_concurrent_allocations,
        "start_gap_seconds": start_gap_seconds,
        "resources": {"nodes": nodes, "gpus": gpus, "cpus": cpus, "memory": memory},
        "approval_status": approval_status,
    }
    plan_id = "segment-plan-" + _digest(identity)[:24]
    segments: list[Segment] = []
    for ordinal in range(segment_count):
        mode = "batch" if ordinal < batch_count else "interactive"
        direction = "forward" if mode == "batch" else "reverse"
        segment_identity = {"plan_id": plan_id, "ordinal": ordinal, "mode": mode}
        segments.append(
            Segment(
                segment_id="segment-" + _digest(segment_identity)[:24],
                ordinal=ordinal,
                mode=mode,
                direction=direction,
                model=model,
                stage=SUPPORTED_STAGES[model],
                walltime_seconds=walltime_seconds,
                drain_seconds=drain_seconds,
                keyword_ids=forward if direction == "forward" else reverse,
                primary_task_set_ref=approved_backlog,
                spillover_task_set_ref=approved_backlog,
                spillover_policy="post-audit-replan-only-v1",
                approval_status=approval_status,
                nodes=nodes,
                gpus=gpus,
                cpus=cpus,
                memory=memory,
                proposed_admission_order=ordinal,
            )
        )
    return SegmentPlan(
        plan_id=plan_id,
        format_version=FORMAT_VERSION,
        audit_id=audit_id,
        ledger_sequence=ledger_sequence,
        acceptance_policy_id=acceptance_policy_id,
        model=model,
        stage=SUPPORTED_STAGES[model],
        priority_policy="configured-rank-then-progress-v1",
        keyword_priority=keyword_priority,
        segment_count=segment_count,
        interactive_segment_count=interactive_segment_count,
        walltime_seconds=walltime_seconds,
        drain_seconds=drain_seconds,
        max_concurrent_allocations=max_concurrent_allocations,
        start_gap_seconds=start_gap_seconds,
        maximum_node_hours=segment_count * nodes * walltime_seconds / 3600,
        maximum_gpu_hours=segment_count * gpus * walltime_seconds / 3600,
        segments=tuple(segments),
    )
