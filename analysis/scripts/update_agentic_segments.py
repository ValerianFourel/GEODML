#!/usr/bin/env python3
"""Create a dry-run one-hour segment plan from one incremental audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_segment_plan import (
    build_segment_plan,
)

FROZEN_SUBMISSION_STATES = frozenset(
    {"submission_requested", "accepted", "pending", "configuring", "running"}
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def build_plan(audit: dict[str, Any], wave: dict[str, Any]) -> dict[str, Any]:
    required_audit = {"audit_id", "ledger_sequence", "acceptance_policy_id", "keywords"}
    if not required_audit.issubset(audit):
        raise ValueError("audit is missing required planning fields")
    plan = build_segment_plan(
        audit_id=audit["audit_id"],
        ledger_sequence=audit["ledger_sequence"],
        acceptance_policy_id=audit["acceptance_policy_id"],
        model=wave["model"],
        keyword_rows=audit["keywords"],
        segment_count=wave["segment_count"],
        interactive_segment_count=wave.get("interactive_segment_count", 0),
        walltime_seconds=wave.get("walltime_seconds", 3600),
        drain_seconds=wave.get("drain_seconds", 300),
        max_concurrent_allocations=wave.get("max_concurrent_allocations", 5),
        start_gap_seconds=wave.get("start_gap_seconds", 600),
        approval_status=wave.get("approval_status", "proposed"),
        nodes=wave.get("nodes", 1),
        gpus=wave.get("gpus", 4),
        cpus=wave.get("cpus", 32),
        memory=wave.get("memory", "all"),
    ).to_dict()
    throughput = audit.get("throughput", {})
    plan["throughput"] = throughput
    plan["active_allocations"] = audit.get("active_allocations", [])
    plan["unresolved_failures"] = audit.get("unresolved_failures", [])
    drain_seconds = plan["drain_seconds"]
    plan["dependencies"] = {
        "blocked_tasks": sum(row.get("blocked", 0) for row in plan["keyword_priority"]),
        "policy": (
            "Nemotron tasks become eligible only after their exact generator "
            "dependency is validated as completed."
            if plan["model"] == "nemotron"
            else "Generator tasks have no cross-model completion dependency."
        ),
    }
    aggregate = throughput.get("aggregate", {}) if isinstance(throughput, dict) else {}
    minimum = aggregate.get("minimum_seconds_per_task")
    median = aggregate.get("median_seconds_per_task")
    maximum = aggregate.get("maximum_seconds_per_task")
    available = plan["walltime_seconds"] - drain_seconds
    remaining = sum(
        row.get("eligible_remaining", 0) for row in plan["keyword_priority"]
    )
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in (minimum, median, maximum)
    ):
        capacity = {
            "conservative": min(remaining, math.floor(available / maximum)),
            "expected": min(remaining, math.floor(available / median)),
            "optimistic": min(remaining, math.floor(available / minimum)),
        }
        runtime = {
            "optimistic_hours": remaining * minimum / 3600,
            "expected_hours": remaining * median / 3600,
            "conservative_hours": remaining * maximum / 3600,
        }
        plan["remaining_resource_estimate"] = {
            "eligible_tasks": remaining,
            "allocation_hours": runtime,
            "gpu_hours": {
                key: value * plan["segments"][0]["gpus"]
                for key, value in runtime.items()
            },
            "assumption": (
                "Observed seconds per completed task are allocation-wall-clock "
                "rates; ranges use the observed minimum and maximum."
            ),
        }
    else:
        capacity = {"conservative": None, "expected": None, "optimistic": None}
        plan["remaining_resource_estimate"] = {
            "eligible_tasks": remaining,
            "allocation_hours": None,
            "gpu_hours": None,
            "assumption": "No valid measured throughput is available.",
        }
    for segment in plan["segments"]:
        segment["estimated_capacity_tasks"] = capacity
        segment["estimated_runtime_range_seconds"] = {
            "work": available,
            "drain": drain_seconds,
            "total": plan["walltime_seconds"],
        }
    return plan


def _frozen_assignments(previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    if previous is None:
        return []
    return [
        row
        for row in previous.get("segments", [])
        if row.get("submission_status") in FROZEN_SUBMISSION_STATES
    ]


def _diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {
            "previous_plan_id": None,
            "current_plan_id": current["plan_id"],
            "changed": True,
            "reason": "initial_plan",
        }
    return {
        "previous_plan_id": previous.get("plan_id"),
        "current_plan_id": current["plan_id"],
        "changed": previous.get("plan_id") != current["plan_id"],
        "reason": (
            "audit_or_wave_changed"
            if previous.get("plan_id") != current["plan_id"]
            else "identical_inputs"
        ),
    }


def _csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def write_plan(
    output: Path,
    *,
    plan: dict[str, Any],
    difference: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite plan directory: {output}")
    output.mkdir(parents=True)
    _atomic_text(
        output / "plan.json",
        json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_text(
        output / "diff.json",
        json.dumps(difference, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        output / "segments.csv",
        _csv(
            plan["segments"],
            [
                "segment_id", "ordinal", "mode", "direction", "model", "stage",
                "walltime_seconds", "approval_status", "nodes", "gpus", "cpus",
                "memory", "proposed_admission_order", "estimated_capacity_tasks",
                "estimated_runtime_range_seconds",
            ],
        ),
    )
    _atomic_text(
        output / "keyword-coverage.csv",
        _csv(
            plan["keyword_priority"],
            [
                "keyword_id", "completed", "active", "blocked",
                "priority_rank", "eligible_remaining", "estimated_remaining_seconds",
                "task_set_ref",
            ],
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--wave", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path)
    parser.add_argument("--write-plan", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit, wave = _json(args.audit), _json(args.wave)
    previous = _json(args.previous_plan) if args.previous_plan else None
    plan = build_plan(audit, wave)
    plan["frozen_assignments"] = _frozen_assignments(previous)
    difference = _diff(previous, plan)
    if args.write_plan:
        write_plan(args.write_plan, plan=plan, difference=difference)
        print(f"PLAN_DIRECTORY={args.write_plan.resolve()}")
    else:
        print(json.dumps({"plan": plan, "diff": difference}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
