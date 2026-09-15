"""Read saved new-cohort cells and Slurm accounting without submitting work."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def collect_progress(root: Path, *, scheduler: bool = True) -> dict:
    if not root.is_dir():
        raise ValueError(f"Trial directory not found: {root}")
    tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_text().splitlines()
             if line.strip()]
    expected = {row["cell_id"]: row for row in tasks}
    if len(expected) != len(tasks) or len(tasks) != 1440:
        raise ValueError("Expected 1,440 unique new-cohort cells")
    models = []
    cutoff = time.time() - 900
    for slug in ("qwen38", "llama4"):
        model_root = root / "models" / slug
        output = model_root / "outputs" / "worker-00000"
        completed = set()
        prompts = Counter()
        factors = Counter()
        recent = 0
        result_root = output / "results"
        try:
            paths = sorted(result_root.iterdir())
        except FileNotFoundError:
            paths = []
        for path in paths:
            if path.suffix != ".json":
                continue
            row = json.loads(path.read_text())
            cell_id = row["cell_id"]
            if cell_id not in expected or cell_id != path.stem:
                raise ValueError(f"Unexpected result identity: {path}")
            if any(row.get(key) != value for key, value in expected[cell_id].items()):
                raise ValueError(f"Result differs from frozen task: {path}")
            completed.add(cell_id)
            prompts[row["prompt_id"]] += 1
            factors["|".join(row[key] for key in ("method", "engine", "condition"))] += 1
            recent += path.stat().st_mtime >= cutoff
        path = output / "run_manifest.json"
        manifest = json.loads(path.read_text()) if path.is_file() else {}
        job_file = model_root / "job-id.txt"
        job = job_file.read_text().strip() if job_file.is_file() else None
        if job is not None and not re.fullmatch(r"[1-9][0-9]*", job):
            raise ValueError(f"Invalid saved job ID: {job_file}")
        models.append({
            "model": slug, "expected": len(expected), "completed": len(completed),
            "remaining": len(expected) - len(completed),
            "percent": round(100 * len(completed) / len(expected), 2),
            "saved_status": manifest.get("status", "no_manifest_yet"),
            "failed_cells": len(set(manifest.get("failed_cell_ids", [])) - completed),
            "complete_prompt_groups": sum(count == 12 for count in prompts.values()),
            "prompts_with_results": len(prompts), "completed_by_factor": dict(factors),
            "recent_cells_per_minute": round(recent / 15, 3),
            "recent_window_minutes": 15,
            "job_id": job, "output": str(output),
        })
    ids = [row["job_id"] for row in models if row["job_id"]]
    jobs = []
    error = None
    if scheduler and ids:
        try:
            response = subprocess.run(
                ["sacct", "--jobs=" + ",".join(ids), "--allocations", "--parsable2",
                 "--noheader", "--format=JobID,State,Elapsed,Timelimit,ExitCode"],
                text=True, capture_output=True, check=True, timeout=15,
            )
            for line in response.stdout.splitlines():
                if not line.strip():
                    continue
                fields = line.split("|")
                if len(fields) != 5 or not any(
                    fields[0] == job or fields[0] == job + "_0" for job in ids
                ):
                    raise ValueError(f"Unexpected accounting row: {line}")
                jobs.append(dict(zip(
                    ("job_id", "state", "elapsed", "time_limit", "exit_code"), fields,
                )))
            missing = [job for job in ids if not any(
                row["job_id"] in (job, job + "_0") for row in jobs
            )]
            if missing:
                error = "No accounting record yet for " + ", ".join(missing)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            error = str(exc)
    total = sum(row["completed"] for row in models)
    return {
        "format_version": "agentic-paired-trial-progress-v1",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_root": str(root), "prompt_count": 120, "models": models,
        "completed": total, "expected": 2880,
        "percent": round(100 * total / 2880, 2), "jobs": jobs,
        "scheduler_checked": scheduler, "job_error": error,
        "note": "Saved artifacts, not scientific validation. A one-hour TIMEOUT can retain "
                "usable cells. Active calls can be interrupted. Recent rate becomes zero "
                "after jobs stop; it is not the benchmark's overall rate. This trial does "
                "not fill missing cells in the earlier 500-prompt study.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--json", action="store_true", dest="json_only")
    parser.add_argument("--no-scheduler", action="store_true")
    args = parser.parse_args()
    try:
        value = collect_progress(args.run_root, scheduler=not args.no_scheduler)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"TRIAL_STATUS_ERROR: {error}") from error
    if not args.json_only:
        print(f"RUN_ROOT={value['run_root']}")
        for row in value["models"]:
            print(f"MODEL={row['model']} saved={row['completed']}/{row['expected']} "
                  f"percent={row['percent']:.2f}% missing={row['remaining']} "
                  f"complete_prompts={row['complete_prompt_groups']}/120 "
                  f"manifest={row['saved_status']} job={row['job_id']}")
        print(value["note"])
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
