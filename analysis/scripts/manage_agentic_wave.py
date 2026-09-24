#!/usr/bin/env python3
"""Evaluate or apply one safe release for a finite GEODML wave."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_admission import choose_release

TERMINAL_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
})


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scheduler-snapshot", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path)
    parser.add_argument("--storage-snapshot", type=Path)
    parser.add_argument("--now-epoch", type=int)
    parser.add_argument("--max-scheduler-snapshot-age-seconds", type=int, default=120)
    parser.add_argument("--max-storage-snapshot-age-seconds", type=int, default=300)
    parser.add_argument(
        "--apply-release", "--apply-admission", dest="apply_release", action="store_true"
    )
    return parser


def _require_fresh_snapshot(
    value: dict[str, Any], *, now: int, maximum_age: int, label: str
) -> None:
    captured = value.get("captured_at_epoch")
    if type(maximum_age) is not int or maximum_age < 0:
        raise ValueError(f"{label} maximum age must be non-negative")
    if type(captured) is not int or not 0 <= now - captured <= maximum_age:
        raise ValueError(f"{label} snapshot is missing, stale, or from the future")


def _reconcile_state(
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    completed_segment_ids: list[str],
) -> dict[str, Any]:
    """Resolve one prior release only from a complete scheduler snapshot."""

    updated = dict(state)
    reserved_segment = updated.get("interactive_reservation_segment_id")
    if reserved_segment is not None:
        if not isinstance(reserved_segment, str) or not reserved_segment:
            raise ValueError("controller state has an invalid interactive reservation")
        if reserved_segment in completed_segment_ids or any(
            job.get("segment_id") == reserved_segment for job in jobs
        ):
            updated.pop("interactive_reservation_segment_id", None)
            updated.pop("interactive_reservation_epoch", None)
    unconfirmed = updated.get("unconfirmed_release_job_id")
    if unconfirmed is None:
        return updated
    matches = [job for job in jobs if str(job.get("job_id")) == unconfirmed]
    if len(matches) > 1:
        raise ValueError("scheduler snapshot duplicates the unconfirmed release")
    if not matches:
        return updated
    job = matches[0]
    start = job.get("start_epoch")
    if type(start) is int and start > 0:
        prior = updated.get("latest_observed_start_epoch")
        updated["latest_observed_start_epoch"] = (
            start if type(prior) is not int else max(prior, start)
        )
        updated.pop("unconfirmed_release_job_id", None)
        return updated
    if str(job.get("state", "")).upper() in TERMINAL_STATES:
        updated.pop("unconfirmed_release_job_id", None)
    return updated


def main() -> None:
    args = _parser().parse_args()
    plan = _read(args.plan)
    snapshot = _read(args.scheduler_snapshot)
    jobs = snapshot.get("jobs")
    if not isinstance(jobs, list) or snapshot.get("complete") is not True:
        raise ValueError("scheduler snapshot must be complete and contain every live GEODML job")
    completed_segment_ids = snapshot.get("completed_segment_ids", [])
    if not isinstance(completed_segment_ids, list):
        raise TypeError("scheduler snapshot completed_segment_ids must be a list")
    state_path = None if args.controller_root is None else args.controller_root / "controller-state.json"
    state = _read(state_path) if state_path and state_path.exists() else {}
    state = _reconcile_state(
        state, jobs, completed_segment_ids=completed_segment_ids
    )
    now = int(time.time()) if args.now_epoch is None else args.now_epoch
    decision = choose_release(
        plan=plan,
        jobs=jobs,
        now_epoch=now,
        controller_state=state,
        completed_segment_ids=completed_segment_ids,
    )
    result = asdict(decision)
    storage = _read(args.storage_snapshot) if args.storage_snapshot else None
    if storage is not None:
        if storage.get("format_version") != "geodml-agentic-storage-snapshot-v1":
            raise ValueError("unsupported storage snapshot")
        if storage.get("safe_to_admit") is not True:
            result.update(
                action="wait",
                reason="storage_admission_blocked",
                segment_id=None,
                job_id=None,
                storage_reasons=storage.get("reasons", []),
            )
    if not args.apply_release:
        print(json.dumps({"dry_run": True, "decision": result}, indent=2, sort_keys=True))
        return
    if args.controller_root is None:
        raise ValueError("--apply-release requires --controller-root")
    if storage is None:
        raise ValueError("--apply-release requires a fresh --storage-snapshot")
    _require_fresh_snapshot(
        snapshot,
        now=now,
        maximum_age=args.max_scheduler_snapshot_age_seconds,
        label="scheduler",
    )
    _require_fresh_snapshot(
        storage,
        now=now,
        maximum_age=args.max_storage_snapshot_age_seconds,
        label="storage",
    )
    args.controller_root.mkdir(parents=True, exist_ok=True)
    with (args.controller_root / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another admission controller is active") from error
        # Re-evaluate under the controller lock. The caller must supply a fresh,
        # complete snapshot for every tick.
        state = _read(state_path) if state_path.exists() else {}
        state = _reconcile_state(
            state, jobs, completed_segment_ids=completed_segment_ids
        )
        decision = choose_release(
            plan=plan,
            jobs=jobs,
            now_epoch=now,
            controller_state=state,
            completed_segment_ids=completed_segment_ids,
        )
        result = asdict(decision)
        if storage.get("safe_to_admit") is not True:
            result.update(
                action="wait",
                reason="storage_admission_blocked",
                segment_id=None,
                job_id=None,
                storage_reasons=storage.get("reasons", []),
            )
            _atomic(state_path, state)
            print(json.dumps({"dry_run": False, "decision": result}, indent=2, sort_keys=True))
            return
        if decision.latest_observed_start_epoch is not None:
            state["latest_observed_start_epoch"] = decision.latest_observed_start_epoch
        if decision.action == "reserve_interactive":
            reservation = {
                "format_version": "geodml-agentic-interactive-reservation-v1",
                "status": "reserved",
                "plan_id": plan["plan_id"],
                "segment_id": decision.segment_id,
                "reserved_at_epoch": now,
                "scheduler_snapshot_id": snapshot.get("snapshot_id"),
            }
            reservation_path = (
                args.controller_root
                / f"interactive-reservation-{decision.segment_id}.json"
            )
            if reservation_path.exists():
                raise FileExistsError(
                    "interactive reservation already exists; reconcile it before retrying"
                )
            _atomic(reservation_path, reservation)
            _atomic(
                state_path,
                {
                    **state,
                    "interactive_reservation_segment_id": decision.segment_id,
                    "interactive_reservation_epoch": now,
                },
            )
            print(
                json.dumps(
                    {"dry_run": False, "decision": result}, indent=2, sort_keys=True
                )
            )
            return
        if decision.action != "release":
            _atomic(state_path, state)
            print(json.dumps({"dry_run": False, "decision": result}, indent=2, sort_keys=True))
            return
        intent = {
            "format_version": "geodml-agentic-admission-intent-v1",
            "status": "release_requested",
            "plan_id": plan["plan_id"],
            "segment_id": decision.segment_id,
            "job_id": decision.job_id,
            "requested_at_epoch": now,
            "scheduler_snapshot_id": snapshot.get("snapshot_id"),
        }
        intent_path = args.controller_root / f"release-{decision.segment_id}.json"
        if intent_path.exists():
            raise FileExistsError("release intent already exists; reconcile Slurm before retrying")
        _atomic(intent_path, intent)
        completed = subprocess.run(
            ["scontrol", "release", decision.job_id],
            text=True,
            capture_output=True,
            check=False,
        )
        intent.update(
            status="accepted" if completed.returncode == 0 else "uncertain",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        _atomic(intent_path, intent)
        if completed.returncode != 0:
            raise RuntimeError("scontrol release failed; intent is uncertain and must be reconciled")
        _atomic(
            state_path,
            {
                **state,
                "last_release_job_id": decision.job_id,
                "last_release_segment_id": decision.segment_id,
                "last_release_epoch": now,
                "unconfirmed_release_job_id": decision.job_id,
            },
        )
        print(json.dumps({"dry_run": False, "decision": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
