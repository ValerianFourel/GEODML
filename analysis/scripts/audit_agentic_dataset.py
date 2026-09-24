#!/usr/bin/env python3
"""Audit one model from sealed final-format shards and the compact task ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_dataset import verify_record_reference
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity

ACTIVE_STATES = frozenset({"claimed", "running", "result_saved"})
SUPPORTED_MODELS = frozenset({"qwen38", "llama4", "nemotron"})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sealed_rows(root: Path, table: str) -> Iterable[dict[str, Any]]:
    directory = root / "data" / table
    if not directory.is_dir():
        raise ValueError(f"missing required dataset table: {table}")
    manifests = sorted(directory.glob("*.manifest.json"))
    if not manifests:
        raise ValueError(f"dataset table has no sealed shards: {table}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != "geodml-jsonl-shard-v1":
            raise ValueError(f"unsupported shard manifest: {manifest_path}")
        shard = root / manifest["path"]
        raw = shard.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
            raise ValueError(f"shard checksum mismatch: {shard}")
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) != manifest.get("rows"):
            raise ValueError(f"shard row count mismatch: {shard}")
        for number, line in enumerate(lines, 1):
            envelope = json.loads(line)
            row = envelope.get("row")
            if not isinstance(row, dict):
                raise TypeError(f"shard row is not an object: {shard}:{number}")
            yield row


def _memberships(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_prompt: dict[str, dict[str, Any]] = {}
    ranks: dict[str, int] = {}
    for row in _sealed_rows(root, "keyword_memberships"):
        prompt_id = row.get("prompt_id")
        keyword_ids = row.get("keyword_ids")
        primary = row.get("primary_keyword_id")
        rank = row.get("primary_priority_rank")
        if (
            not isinstance(prompt_id, str)
            or not isinstance(keyword_ids, list)
            or not keyword_ids
            or any(not isinstance(item, str) or not item for item in keyword_ids)
            or not isinstance(primary, str)
            or primary not in keyword_ids
            or type(rank) is not int
            or rank < 0
        ):
            raise ValueError("invalid keyword membership row")
        if prompt_id in by_prompt:
            raise ValueError(f"duplicate keyword membership for prompt: {prompt_id}")
        prior = ranks.setdefault(primary, rank)
        if prior != rank:
            raise ValueError(f"keyword priority rank changed within dataset: {primary}")
        by_prompt[prompt_id] = row
    if len(set(ranks.values())) != len(ranks):
        raise ValueError("keyword priority ranks must be unique")
    return by_prompt, ranks


def _rate_summary(seconds_per_task: list[float]) -> dict[str, Any]:
    if not seconds_per_task:
        return {
            "measurement_count": 0,
            "minimum_seconds_per_task": None,
            "median_seconds_per_task": None,
            "maximum_seconds_per_task": None,
        }
    return {
        "measurement_count": len(seconds_per_task),
        "minimum_seconds_per_task": min(seconds_per_task),
        "median_seconds_per_task": statistics.median(seconds_per_task),
        "maximum_seconds_per_task": max(seconds_per_task),
    }


def _throughput(root: Path, model: str) -> dict[str, Any]:
    path = root / "reports" / "throughput.json"
    if not path.exists():
        return {"measurements": [], "aggregate": _rate_summary([]), "groups": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("measurements") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise TypeError("throughput report lacks measurements")
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, dict)
        and row.get("model") == model
        and type(row.get("completed_tasks")) is int
        and row["completed_tasks"] > 0
        and isinstance(row.get("work_seconds"), (int, float))
        and not isinstance(row.get("work_seconds"), bool)
        and row["work_seconds"] > 0
    ]
    seconds_per_task = [
        row["work_seconds"] / row["completed_tasks"] for row in selected
    ]
    grouped: dict[tuple[str | None, str | None], list[float]] = {}
    for row, rate in zip(selected, seconds_per_task, strict=True):
        keyword_id = row.get("keyword_id")
        method = row.get("method")
        if keyword_id is not None and not isinstance(keyword_id, str):
            raise TypeError("throughput keyword_id must be a string or null")
        if method is not None and not isinstance(method, str):
            raise TypeError("throughput method must be a string or null")
        grouped.setdefault((keyword_id, method), []).append(rate)
    groups = [
        {
            "keyword_id": keyword_id,
            "method": method,
            **_rate_summary(rates),
        }
        for (keyword_id, method), rates in sorted(
            grouped.items(), key=lambda item: ((item[0][0] or ""), (item[0][1] or ""))
        )
    ]
    return {
        "measurements": selected,
        "aggregate": _rate_summary(seconds_per_task),
        "groups": groups,
    }


def build_audit(
    root: Path,
    *,
    model: str,
    stripe_count: int = 256,
    scheduler_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {model}")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    if contract.get("format_version") != "geodml-incremental-dataset-v1":
        raise ValueError("unsupported incremental dataset contract")
    memberships, ranks = _memberships(root)
    all_tasks: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for row in _sealed_rows(root, "task_definitions"):
        task_id = row.get("task_id")
        prompt_id = row.get("prompt_id")
        identity_value = row.get("claim_identity")
        if (
            not isinstance(task_id, str)
            or not isinstance(prompt_id, str)
            or not isinstance(identity_value, dict)
        ):
            raise TypeError("task definition lacks stable identities")
        if prompt_id not in memberships:
            raise ValueError(f"task prompt lacks keyword membership: {prompt_id}")
        identity = ClaimIdentity(**identity_value)
        if identity.task_id != task_id:
            raise ValueError("task ID differs from claim identity")
        fingerprint = identity_fingerprint(identity)
        if fingerprint in fingerprints:
            raise ValueError("duplicate validated task identity")
        fingerprints.add(fingerprint)
        prepared = {
            **row,
            "fingerprint": fingerprint,
            "primary_keyword_id": memberships[prompt_id]["primary_keyword_id"],
        }
        all_tasks.append(prepared)
    tasks = [task for task in all_tasks if task.get("model") == model]
    if not tasks:
        raise ValueError(f"no task definitions found for model: {model}")
    ledger = StripedTaskLedger(root / "control" / "task-ledger", stripe_count=stripe_count)
    snapshot = ledger.snapshot()
    latest = snapshot["latest"]
    completed_fingerprints = set()
    for task in all_tasks:
        fingerprint = task["fingerprint"]
        event = latest.get(fingerprint, {})
        if event.get("state") != "completed":
            continue
        references = event.get("record_references", [])
        if references and all(
            isinstance(reference, dict)
            and verify_record_reference(root, reference)
            for reference in references
        ):
            completed_fingerprints.add(fingerprint)
    groups: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for keyword_id, rank in ranks.items():
        groups[keyword_id] = {
            "keyword_id": keyword_id,
            "priority_rank": rank,
            "completed": 0,
            "active": 0,
            "blocked": 0,
            "eligible_remaining": 0,
            "task_ids": [],
        }
    for task in tasks:
        group = groups[task["primary_keyword_id"]]
        group["task_ids"].append(task["fingerprint"])
        event = latest.get(task["fingerprint"])
        state = None if event is None else event.get("state")
        dependencies = task.get("dependency_fingerprints", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise ValueError("task dependency fingerprints must be a list of strings")
        blocked_reason = task.get("blocked_reason")
        if state == "completed" and task["fingerprint"] in completed_fingerprints:
            group["completed"] += 1
        elif state == "completed":
            group["blocked"] += 1
            failures.append({
                "task_id": task["task_id"],
                "reason": "completed_record_reference_invalid",
            })
        elif state in ACTIVE_STATES:
            group["active"] += 1
        elif state == "terminal_failed":
            group["blocked"] += 1
            failures.append({"task_id": task["task_id"], "reason": "terminal_failed"})
        elif blocked_reason:
            group["blocked"] += 1
            failures.append({"task_id": task["task_id"], "reason": blocked_reason})
        elif any(
            dependency not in completed_fingerprints for dependency in dependencies
        ):
            group["blocked"] += 1
        else:
            group["eligible_remaining"] += 1
    throughput = _throughput(root, model)
    seconds_per_task = throughput["aggregate"]["median_seconds_per_task"]
    keyword_rows: list[dict[str, Any]] = []
    for group in sorted(groups.values(), key=lambda value: value["priority_rank"]):
        task_set_ref = "task-set-" + hashlib.sha256(
            _canonical({
                "model": model,
                "keyword_id": group["keyword_id"],
                "task_fingerprints": sorted(group.pop("task_ids")),
            })
        ).hexdigest()[:24]
        group["task_set_ref"] = task_set_ref
        group["estimated_remaining_seconds"] = (
            None
            if seconds_per_task is None
            else round(group["eligible_remaining"] * seconds_per_task, 3)
        )
        keyword_rows.append(group)
    active_allocations = []
    if scheduler_snapshot is not None:
        if scheduler_snapshot.get("complete") is not True:
            raise ValueError("scheduler snapshot must be marked complete")
        jobs = scheduler_snapshot.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("scheduler snapshot lacks jobs")
        active_allocations = [
            row for row in jobs
            if isinstance(row, dict)
            and row.get("model") == model
            and row.get("state") in {
                "PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT"
            }
        ]
    identity = {
        "contract_sha256": hashlib.sha256((root / "contract.json").read_bytes()).hexdigest(),
        "model": model,
        "ledger_event_count": snapshot["event_count"],
        "keywords": keyword_rows,
        "active_allocations": active_allocations,
        "throughput": throughput,
    }
    return {
        "format_version": "geodml-agentic-incremental-audit-v1",
        "audit_id": "audit-" + hashlib.sha256(_canonical(identity)).hexdigest()[:24],
        "ledger_sequence": snapshot["event_count"],
        "population_id": contract["population_id"],
        "acceptance_policy_id": contract["acceptance_policy_id"],
        "model": model,
        "task_count": len(tasks),
        "completed": sum(row["completed"] for row in keyword_rows),
        "active": sum(row["active"] for row in keyword_rows),
        "blocked": sum(row["blocked"] for row in keyword_rows),
        "eligible_remaining": sum(row["eligible_remaining"] for row in keyword_rows),
        "keywords": keyword_rows,
        "throughput": throughput,
        "active_allocations": active_allocations,
        "unresolved_failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), required=True)
    parser.add_argument("--stripe-count", type=int, default=256)
    parser.add_argument("--scheduler-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    scheduler = (
        json.loads(args.scheduler_snapshot.read_text(encoding="utf-8"))
        if args.scheduler_snapshot
        else None
    )
    audit = build_audit(
        args.dataset_root,
        model=args.model,
        stripe_count=args.stripe_count,
        scheduler_snapshot=scheduler,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"AUDIT={args.output.resolve()}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
