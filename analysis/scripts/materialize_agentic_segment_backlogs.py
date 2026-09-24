#!/usr/bin/env python3
"""Materialize immutable, keyword-ordered queues for an approved segment plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.agentic_dataset import verify_record_reference
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity

PLAN_FORMAT = "geodml-agentic-segment-plan-v1"
WAVE_FORMAT = "geodml-inference-wave-v2"
ACTIVE_STATES = frozenset({"claimed", "running", "result_saved"})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _sealed_rows(root: Path, table: str) -> Iterable[dict[str, Any]]:
    table_root = root / "data" / table
    manifests = sorted(table_root.glob("*.manifest.json"))
    if not manifests:
        raise ValueError(f"dataset has no sealed {table} table")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard = root / manifest["path"]
        raw = shard.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
            raise ValueError(f"shard checksum mismatch: {shard}")
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) != manifest.get("rows"):
            raise ValueError(f"shard row count mismatch: {shard}")
        for line in lines:
            envelope = json.loads(line)
            row = envelope.get("row")
            if not isinstance(row, dict):
                raise TypeError(f"non-object dataset row in {shard}")
            yield row


def materialize(
    *,
    dataset_root: Path,
    plan: Mapping[str, Any],
    output: Path,
    stripe_count: int = 256,
) -> dict[str, Any]:
    if plan.get("format_version") != PLAN_FORMAT:
        raise ValueError("unsupported segment plan format")
    if any(row.get("approval_status") != "approved" for row in plan["segments"]):
        raise ValueError("all materialized segments must be explicitly approved")
    model = plan.get("model")
    memberships = {
        row["prompt_id"]: row["primary_keyword_id"]
        for row in _sealed_rows(dataset_root, "keyword_memberships")
    }
    task_rows: list[dict[str, Any]] = []
    identities: dict[str, ClaimIdentity] = {}
    for row in _sealed_rows(dataset_root, "task_definitions"):
        task_id = row.get("task_id")
        prompt_id = row.get("prompt_id")
        runnable = row.get("runnable_task")
        identity_value = row.get("claim_identity")
        if (
            not isinstance(task_id, str)
            or not isinstance(prompt_id, str)
            or prompt_id not in memberships
            or not isinstance(identity_value, dict)
        ):
            raise ValueError("task definition lacks membership or stable identity")
        identity = ClaimIdentity(**identity_value)
        fingerprint = identity_fingerprint(identity)
        if identity.task_id != task_id or fingerprint in identities:
            raise ValueError("task definitions have duplicate or mismatched identities")
        identities[fingerprint] = identity
        if row.get("model") != model:
            continue
        if not isinstance(runnable, dict):
            raise TypeError("selected task definition lacks a runnable task")
        task_rows.append(
            {
                **row,
                "fingerprint": fingerprint,
                "primary_keyword_id": memberships[prompt_id],
                "runnable_task": runnable,
            }
        )
    if not task_rows:
        raise ValueError(f"dataset has no runnable task definitions for {model}")
    ledger = StripedTaskLedger(
        dataset_root / "control" / "task-ledger", stripe_count=stripe_count
    )
    snapshot = ledger.snapshot()
    latest = snapshot["latest"]
    completed_ids = set()
    invalid_completed: set[str] = set()
    for fingerprint in identities:
        event = latest.get(fingerprint, {})
        if event.get("state") != "completed":
            continue
        references = event.get("record_references", [])
        if references and all(
            isinstance(reference, dict)
            and verify_record_reference(dataset_root, reference)
            for reference in references
        ):
            completed_ids.add(fingerprint)
        else:
            invalid_completed.add(fingerprint)
    eligible: list[dict[str, Any]] = []
    blocked = 0
    for row in task_rows:
        state = latest.get(row["fingerprint"], {}).get("state")
        dependencies = row.get("dependency_fingerprints", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise ValueError("task dependencies must be identity fingerprints")
        if state == "completed":
            if row["fingerprint"] in invalid_completed:
                blocked += 1
            continue
        if state in ACTIVE_STATES or state == "terminal_failed":
            continue
        if row.get("blocked_reason") or any(
            dependency not in completed_ids for dependency in dependencies
        ):
            blocked += 1
            continue
        eligible.append(row)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite segment queues: {output}")
    output.mkdir(parents=True)
    waves: list[dict[str, Any]] = []
    has_batch_segments = any(
        segment.get("mode") == "batch" for segment in plan["segments"]
    )
    has_interactive_segments = any(
        segment.get("mode") == "interactive" for segment in plan["segments"]
    )
    ranked_keywords = [
        row["keyword_id"]
        for row in sorted(
            plan["keyword_priority"], key=lambda item: item["priority_rank"]
        )
        if any(task["primary_keyword_id"] == row["keyword_id"] for task in eligible)
    ]
    split = (len(ranked_keywords) + 1) // 2
    batch_owned = set(
        ranked_keywords[:split] if has_interactive_segments else ranked_keywords
    )
    interactive_owned = set(
        ranked_keywords[split:] if has_batch_segments else ranked_keywords
    )
    batch_keyword_id: str | None = None
    for segment in plan["segments"]:
        keyword_ids = segment.get("keyword_ids")
        if (
            not isinstance(keyword_ids, (list, tuple))
            or len(set(keyword_ids)) != len(keyword_ids)
        ):
            raise ValueError("segment keyword order is invalid")
        order = {keyword_id: position for position, keyword_id in enumerate(keyword_ids)}
        if any(row["primary_keyword_id"] not in order for row in eligible):
            raise ValueError("segment keyword order does not cover eligible tasks")
        primary_keyword_id = next(
            (
                keyword_id
                for keyword_id in keyword_ids
                if any(row["primary_keyword_id"] == keyword_id for row in eligible)
            ),
            None,
        )
        if segment.get("mode") == "batch":
            if batch_keyword_id is None:
                batch_keyword_id = primary_keyword_id
            elif batch_keyword_id != primary_keyword_id:
                raise ValueError("batch segments disagree on their front keyword")
        elif segment.get("mode") == "interactive":
            if has_batch_segments and primary_keyword_id == batch_keyword_id:
                primary_keyword_id = None
        else:
            raise ValueError("segment mode must be batch or interactive")
        # Freeze the whole approved eligible backlog in traversal order.  The
        # durable ledger makes concurrent workers divide the front keyword and
        # then continue to later keywords if it finishes during the allocation.
        # Interactive traversal omits the batch-owned front keyword so opposite
        # directions cannot collide before a post-audit replan.
        selected = sorted(
            (
                row
                for row in eligible
                if row["primary_keyword_id"] in (
                    batch_owned
                    if segment.get("mode") == "batch"
                    else interactive_owned
                )
            ),
            key=lambda row: (
                order[row["primary_keyword_id"]],
                row["task_id"],
                row["fingerprint"],
            ),
        )
        wave_root = output / "segments" / segment["segment_id"] / "wave"
        backlog_path = wave_root / "backlog.jsonl"
        backlog = b"".join(
            _canonical({
                **row["runnable_task"],
                "geodml_keyword_id": row["primary_keyword_id"],
                "geodml_task_fingerprint": row["fingerprint"],
            }) + b"\n"
            for row in selected
        )
        _atomic(backlog_path, backlog)
        backlog_identity = {
            "path": str(backlog_path.resolve()),
            "sha256": hashlib.sha256(backlog).hexdigest(),
            "task_count": len(selected),
        }
        wave = {
            "format_version": WAVE_FORMAT,
            "status": "planned",
            "dispatch_mode": "backlog",
            "segment_id": segment["segment_id"],
            "segment_plan_id": plan["plan_id"],
            "model": model,
            "direction": segment["direction"],
            "keyword_ids": keyword_ids,
            "primary_keyword_id": primary_keyword_id,
            "ledger_event_count": snapshot["event_count"],
            "eligible_task_count": len(selected),
            "queue_exhausted": not selected,
            "blocked_task_count": blocked,
            "backlog": backlog_identity,
        }
        _atomic(wave_root / "run_manifest.json", json.dumps(
            wave, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n")
        waves.append({
            "segment_id": segment["segment_id"],
            "wave_root": str(wave_root.resolve()),
            "primary_keyword_id": primary_keyword_id,
            "queue_exhausted": not selected,
            "backlog": backlog_identity,
        })
    result = {
        "format_version": "geodml-agentic-segment-backlogs-v1",
        "plan_id": plan["plan_id"],
        "model": model,
        "ledger_event_count": snapshot["event_count"],
        "eligible_task_count": len(eligible),
        "blocked_task_count": blocked,
        "waves": waves,
    }
    _atomic(output / "materialization.json", json.dumps(
        result, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stripe-count", type=int, default=256)
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = materialize(
        dataset_root=args.dataset_root,
        plan=plan,
        output=args.output,
        stripe_count=args.stripe_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
