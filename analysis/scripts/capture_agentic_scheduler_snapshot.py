#!/usr/bin/env python3
"""Capture current and terminal GEODML allocations for admission and recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

COMMENT = re.compile(
    r"geodml-v2:(segment-plan-[0-9a-f]{24}):(segment-[0-9a-f]{24}):"
    r"(qwen38|llama4|nemotron):(batch|interactive)"
)
ACTIVE_STATES = frozenset({"CONFIGURING", "RUNNING", "COMPLETING", "STAGE_OUT"})
TERMINAL_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
})


def _epoch(value: str) -> int | None:
    if value in {"", "N/A", "Unknown", "None"}:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _comment(value: str) -> dict[str, str]:
    hour = re.fullmatch(r"geodml-hours:(jupiter|horeka):([A-Za-z0-9][A-Za-z0-9_.-]{0,159})", value)
    if hour:
        return {"cluster": hour[1], "attempt_id": hour[2]}
    match = COMMENT.fullmatch(value)
    if match is None:
        return {}
    plan_id, segment_id, model, mode = match.groups()
    return {
        "plan_id": plan_id,
        "segment_id": segment_id,
        "model": model,
        "mode": mode,
    }


def _run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"scheduler query failed ({command[0]}): {result.stderr.strip()}"
        )
    return result.stdout


def capture(
    *,
    plan: dict[str, Any],
    since: str,
    runner: Callable[[list[str]], str] = _run,
    include_job_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Capture all live geodml-* jobs plus this plan's terminal history."""

    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", since) is None:
        raise ValueError("since must use YYYY-MM-DD")
    plan_id = plan.get("plan_id")
    if not isinstance(plan_id, str):
        raise TypeError("plan lacks a plan_id")
    live_raw = runner([
        "squeue", "--noheader", "--me",
        "--format=%i|%j|%T|%S|%r|%k",
    ])
    jobs: list[dict[str, Any]] = []
    for number, line in enumerate(live_raw.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 5)
        if len(fields) != 6:
            raise ValueError(f"unexpected squeue row {number}")
        job_id, name, state, start, reason, comment = (item.strip() for item in fields)
        if not name.startswith("geodml-") and job_id not in include_job_ids:
            continue
        state = state.upper()
        metadata = _comment(comment)
        jobs.append({
            "job_id": job_id,
            "job_name": name,
            "state": state,
            "held": state == "PENDING" and reason == "JobHeldUser",
            "start_epoch": _epoch(start) if state in ACTIVE_STATES else None,
            "reason": reason,
            "comment": comment,
            **metadata,
        })
    history_raw = runner([
        "sacct", "-X", "--noheader", "--parsable2", f"--starttime={since}",
        "--format=JobIDRaw,JobName%200,State,Start,Comment%512",
    ])
    completed_segment_ids: set[str] = set()
    owners: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(history_raw.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 5)
        if len(fields) < 5:
            raise ValueError(f"unexpected sacct row {number}")
        job_id, name, state, start, comment = (item.strip() for item in fields[:5])
        if not re.fullmatch(r"[0-9]+", job_id) or (not name.startswith("geodml-") and job_id not in include_job_ids):
            continue
        state = state.split()[0].rstrip("+").upper()
        metadata = _comment(comment)
        if state in TERMINAL_STATES:
            if metadata.get("plan_id") == plan_id:
                completed_segment_ids.add(metadata["segment_id"])
            owner_ids = [f"job{job_id}-worker0", f"judge-{job_id}-0"]
            if metadata.get("attempt_id"):
                owner_ids.append(f"{metadata['cluster']}-{metadata['attempt_id']}")
            for owner_id in owner_ids:
                owners[owner_id] = {
                    "owner_id": owner_id,
                    "job_id": job_id,
                    "state": state,
                    "start_epoch": _epoch(start),
                    **metadata,
                }
    now = int(time.time())
    identity = {
        "plan_id": plan_id,
        "jobs": jobs,
        "completed_segment_ids": sorted(completed_segment_ids),
        "owners": sorted(owners.values(), key=lambda row: row["owner_id"]),
        "since": since,
    }
    return {
        "format_version": "geodml-agentic-scheduler-snapshot-v1",
        "snapshot_id": "scheduler-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24],
        "captured_at_epoch": now,
        "complete": True,
        **identity,
    }


def _write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scheduler snapshot: {path}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = capture(plan=plan, since=args.since)
    if args.output:
        _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
