"""Admission decisions for approved GEODML Slurm allocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

RESOURCE_HOLDING_STATES = frozenset(
    {"CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT"}
)
PENDING_STATES = frozenset({"PENDING", "REQUEUED"})


@dataclass(frozen=True)
class AdmissionDecision:
    action: str
    reason: str
    segment_id: str | None
    job_id: str | None
    active_allocations: int
    released_pending_allocations: int
    latest_observed_start_epoch: int | None
    next_start_not_before_epoch: int | None


def _job_id(job: Mapping[str, Any]) -> str:
    value = job.get("job_id")
    if not isinstance(value, (str, int)) or not str(value):
        raise ValueError("scheduler job lacks a job_id")
    return str(value)


def _state(job: Mapping[str, Any]) -> str:
    value = job.get("state")
    if not isinstance(value, str) or not value:
        raise ValueError(f"job {_job_id(job)} lacks a state")
    return value.upper()


def choose_release(
    *,
    plan: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    now_epoch: int,
    controller_state: Mapping[str, Any] | None = None,
    completed_segment_ids: Sequence[str] = (),
) -> AdmissionDecision:
    """Choose at most one held approved job to release.

    ``jobs`` must include every live GEODML experiment allocation owned by the
    user, not just allocations from the current wave.
    """

    if type(now_epoch) is not int or now_epoch < 0:
        raise ValueError("now_epoch must be a non-negative integer")
    if any(not isinstance(item, str) or not item for item in completed_segment_ids):
        raise ValueError("completed segment IDs must be non-empty strings")
    completed_segments = set(completed_segment_ids)
    model = plan.get("model")
    segments = plan.get("segments")
    if not isinstance(model, str) or not isinstance(segments, list) or not segments:
        raise ValueError("plan lacks a homogeneous model and segments")
    if {segment.get("model") for segment in segments} != {model}:
        raise ValueError("plan contains mixed model roles")
    limit = plan.get("max_concurrent_allocations")
    gap = plan.get("start_gap_seconds")
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("invalid allocation concurrency limit")
    if type(gap) is not int or gap < 600:
        raise ValueError("invalid observed-start spacing")

    by_segment: dict[str, Mapping[str, Any]] = {}
    active = 0
    released_pending = 0
    observed_starts: list[int] = []
    for job in jobs:
        state = _state(job)
        segment_id = job.get("segment_id")
        if isinstance(segment_id, str) and segment_id:
            if segment_id in by_segment:
                raise ValueError(f"multiple live jobs claim segment {segment_id}")
            by_segment[segment_id] = job
        if state in RESOURCE_HOLDING_STATES:
            active += 1
        if state in PENDING_STATES and not bool(job.get("held", False)):
            released_pending += 1
        start = job.get("start_epoch")
        if type(start) is int and start > 0 and state in RESOURCE_HOLDING_STATES:
            observed_starts.append(start)
    state_start = None if controller_state is None else controller_state.get(
        "latest_observed_start_epoch"
    )
    if state_start is not None:
        if type(state_start) is not int or state_start < 0:
            raise ValueError("controller state has an invalid observed start")
        observed_starts.append(state_start)
    latest_start = max(observed_starts, default=None)
    next_start = None if latest_start is None else latest_start + gap

    def decision(action: str, reason: str, segment_id=None, job_id=None):
        return AdmissionDecision(
            action=action,
            reason=reason,
            segment_id=segment_id,
            job_id=job_id,
            active_allocations=active,
            released_pending_allocations=released_pending,
            latest_observed_start_epoch=latest_start,
            next_start_not_before_epoch=next_start,
        )

    if active >= limit:
        return decision("wait", "concurrency_limit")
    unconfirmed_job_id = (
        None
        if controller_state is None
        else controller_state.get("unconfirmed_release_job_id")
    )
    if unconfirmed_job_id is not None:
        if not isinstance(unconfirmed_job_id, str) or not unconfirmed_job_id:
            raise ValueError("controller state has an invalid unconfirmed release")
        released = next(
            (job for job in jobs if _job_id(job) == unconfirmed_job_id), None
        )
        if released is None:
            return decision("wait", "released_job_missing_unconfirmed")
        released_start = released.get("start_epoch")
        if type(released_start) is not int or released_start <= 0:
            return decision("wait", "released_job_start_unconfirmed")
    if released_pending:
        return decision("wait", "released_job_start_unconfirmed")
    if next_start is not None and now_epoch < next_start:
        return decision("wait", "observed_start_gap")

    interactive_reservation = (
        None
        if controller_state is None
        else controller_state.get("interactive_reservation_segment_id")
    )
    if interactive_reservation is not None:
        if not isinstance(interactive_reservation, str) or not interactive_reservation:
            raise ValueError("controller state has an invalid interactive reservation")
        if interactive_reservation not in by_segment:
            return decision("wait", "interactive_reservation_unconfirmed")

    pending_interactive = [
        segment
        for segment in segments
        if segment.get("mode") == "interactive"
        and segment.get("approval_status") == "approved"
        and segment.get("segment_id") not in completed_segments
        and segment.get("segment_id") not in by_segment
    ]
    if pending_interactive:
        segment = min(
            pending_interactive, key=lambda row: row["proposed_admission_order"]
        )
        return decision(
            "reserve_interactive",
            "interactive_allocation_ready",
            segment["segment_id"],
        )

    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for segment in sorted(segments, key=lambda row: row["proposed_admission_order"]):
        if segment.get("approval_status") != "approved":
            continue
        segment_id = segment.get("segment_id")
        job = by_segment.get(segment_id)
        if job is None or _state(job) not in PENDING_STATES or not job.get("held"):
            continue
        if job.get("model") not in {None, model}:
            raise ValueError(f"job {_job_id(job)} has a different model")
        candidates.append((segment, job))
    if not candidates:
        return decision("wait", "no_approved_held_segment")

    segment, job = candidates[0]
    return decision("release", "eligible", segment["segment_id"], _job_id(job))
