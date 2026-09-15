"""Print a read-only JSON snapshot of the 500-prompt study and its judge pilot."""

from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS = ("qwen38", "llama4")
SHARDS = 4
CELLS_PER_SHARD = 1500
PILOT_ITEMS = 24


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError(f"cannot inspect {path}: {error}") from error
    return True


def _directory(path: Path) -> bool:
    if not _present(path):
        return False
    try:
        if not stat.S_ISDIR(path.stat().st_mode):
            raise ValueError(f"expected a directory: {path}")
    except OSError as error:
        raise ValueError(f"cannot inspect {path}: {error}") from error
    return True


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error


def _object(text: str, location: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as error:
        raise ValueError(f"invalid JSON at {location}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {location}")  # noqa: TRY004
    return value


def _manifest(path: Path) -> dict[str, Any]:
    if not _present(path):
        return {}
    value = _object(_read(path), str(path))
    for field in ("completed_count", "remaining_count"):
        count = value.get(field)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"invalid {field} at {path}")
    if "status" in value and not isinstance(value["status"], str):
        raise ValueError(f"invalid status at {path}")
    failed = value.get("failed_cell_ids", [])
    if not isinstance(failed, list) or any(
        not isinstance(item, str) or not item for item in failed
    ):
        raise ValueError(f"invalid failed_cell_ids at {path}")
    return value


def _result_ids(root: Path) -> set[str]:
    if not _directory(root):
        return set()
    try:
        paths = sorted(root.iterdir())
    except OSError as error:
        raise ValueError(f"cannot list {root}: {error}") from error
    ids: set[str] = set()
    for path in paths:
        if path.suffix != ".json":
            continue
        row = _object(_read(path), str(path))
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id != path.stem:
            raise ValueError(f"cell_id does not match result filename: {path}")
        ids.add(cell_id)
    return ids


def _model(root: Path, model: str) -> dict[str, Any]:
    shards = []
    all_ids: set[str] = set()
    for index in range(SHARDS):
        shard = root / "models" / model / f"shard-{index}"
        exists = _directory(shard)
        manifest = _manifest(shard / "run_manifest.json")
        ids = _result_ids(shard / "results")
        duplicate = ids & all_ids
        if duplicate:
            raise ValueError(f"duplicate cells across {model} shards: {sorted(duplicate)[:3]}")
        all_ids.update(ids)
        shards.append({
            "shard": index,
            "completed": len(ids),
            "expected": CELLS_PER_SHARD,
            "status": manifest.get("status", "missing_manifest" if exists else "not_started"),
            "manifest_completed": manifest.get("completed_count"),
            "failed_cells": len(set(manifest.get("failed_cell_ids", []))),
            "artifact_overflow": max(0, len(ids) - CELLS_PER_SHARD),
        })
    return {"model": model, "expected": SHARDS * CELLS_PER_SHARD,
            "completed": len(all_ids), "shards": shards}


def _pilot(root: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "expected": PILOT_ITEMS, "completed": 0, "status": "not_inspected",
        "inspected": root is not None, "manifest_completed": None,
        "artifact_overflow": 0, "duplicate_rows": 0,
    }
    if root is None:
        return summary
    _directory(root)
    output = root / "nemotron"
    _directory(output)
    manifest = _manifest(output / "run_manifest.json")
    summary.update(status=manifest.get("status", "no_manifest"),
                   manifest_completed=manifest.get("completed_count"))
    path = output / "outcomes.jsonl"
    if not _present(path):
        return summary
    outcomes: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(_read(path).splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{number}"
        try:
            row = _object(line, location)
        except ValueError as error:
            raise ValueError(f"{error}. Retry after the writer finishes its current append; "
                             "if this persists, inspect the journal.") from error
        task_id, parsed = row.get("judge_task_id"), row.get("parsed_output")
        if not isinstance(task_id, str) or not task_id or not isinstance(parsed, dict):
            raise ValueError(f"invalid judge_task_id or parsed_output at {location}")
        if task_id in outcomes:
            if outcomes[task_id] != parsed:
                raise ValueError(f"conflicting judgments for {task_id} at {location}")
            summary["duplicate_rows"] += 1
        outcomes[task_id] = parsed
    summary.update(completed=len(outcomes), artifact_overflow=max(0, len(outcomes) - PILOT_ITEMS))
    return summary


def _jobs(job_ids: list[str]) -> tuple[list[dict[str, str]], str | None]:
    if not job_ids:
        return [], None
    if any(not re.fullmatch(r"[1-9][0-9]*", job_id) for job_id in job_ids):
        raise ValueError("--job must be a positive numeric allocation ID")
    command = ["sacct", "--jobs=" + ",".join(job_ids), "--parsable2", "--noheader",
               "--allocations", "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,ExitCode"]
    try:
        response = subprocess.run(command, capture_output=True, text=True,
                                  check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"Scheduler status unavailable: {error}"
    fields = ("job_id", "name", "state", "elapsed", "time_limit", "exit_code")
    jobs = []
    for line in response.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != len(fields) or parts[0] not in job_ids:
            return [], f"Unexpected sacct row; scheduler status unknown: {line!r}"
        jobs.append(dict(zip(fields, parts)))
    missing = set(job_ids) - {job["job_id"] for job in jobs}
    return jobs, (f"sacct returned no record for: {', '.join(sorted(missing))}; "
                  "their status is unknown" if missing else None)


def collect_snapshot(
    run_root: Path, *, pilot_root: Path | None = None, jobs: list[str] | None = None,
) -> dict[str, Any]:
    """Read artifact counts, saved manifests, and optional Slurm accounting."""
    if not _directory(run_root):
        raise ValueError(f"run root does not exist; verify --run-root: {run_root}")
    job_ids = list(dict.fromkeys(jobs or []))
    job_rows, job_error = _jobs(job_ids)
    snapshot = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_root": str(run_root),
        "models": [_model(run_root, model) for model in MODELS],
        "pilot": _pilot(pilot_root),
        "jobs": job_rows,
        "requested_job_ids": job_ids,
        "source_note": (
            "Read-only snapshot of the four-shard 500-prompt study. Counts are JSON artifacts, "
            "not quality scores or scientific validation. Saved manifest status is advisory, "
            "not live scheduler state. Files are read sequentially while writers may be active. "
            "No full-population reuse eligibility or external result roots are inferred."
        ),
    }
    if job_error:
        snapshot["job_error"] = job_error
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--pilot-root", type=Path,
                        help="Pilot root containing nemotron/outcomes.jsonl; omitted means not inspected.")
    parser.add_argument("--job", action="append", default=[], help="Allocation ID to query with sacct.")
    args = parser.parse_args(argv)
    try:
        snapshot = collect_snapshot(args.run_root, pilot_root=args.pilot_root, jobs=args.job)
    except ValueError as error:
        print(f"ROADMAP_SNAPSHOT_ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(snapshot, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
