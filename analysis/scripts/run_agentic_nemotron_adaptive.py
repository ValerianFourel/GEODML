#!/usr/bin/env python3
"""Run one approved Nemotron sweep slot, then drain the shared judge backlog."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.inference_wave import _atomic_json
from analysis.scripts.manage_agentic_nemotron_sweep import prepare, select
from analysis.scripts.search_vllm_stage import load_profile


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runner(
    *,
    tasks: Path,
    judge_plan: Path,
    output: Path,
    concurrency: int,
    base_url: str,
    model_id: str,
    revision: str,
    pilot_only: bool,
    claim_root: Path | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
    role_end: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "analysis/scripts/run_acl_arr_vllm.py"),
        "agentic-judge",
        "--tasks",
        str(tasks),
        "--judge-manifest",
        str(judge_plan),
        "--judge-role",
        "bulk",
        "--output-dir",
        str(output),
        "--base-url",
        base_url,
        "--server-model-name",
        model_id,
        "--server-model-revision",
        revision,
        "--max-concurrency",
        str(concurrency),
        "--max-output-tokens",
        "2048",
        "--max-attempts",
        "3",
        "--request-timeout",
        "120",
        "--scheduler",
        "rolling",
        "--resume",
        "--disable-thinking",
    ]
    if pilot_only:
        command.append("--pilot-only")
    if claim_root is not None:
        command.extend(
            (
                "--claim-root",
                str(claim_root),
                "--worker-index",
                str(worker_index),
                "--worker-count",
                str(worker_count),
                "--dispatch-mode",
                "backlog",
            )
        )
    environment = dict(os.environ)
    if role_end is not None:
        environment["GEODML_ROLE_END_TIME"] = str(role_end)
    else:
        environment.pop("GEODML_ROLE_END_TIME", None)
    return subprocess.run(command, text=True, check=False, env=environment)


def _benchmark_result(
    root: Path,
    *,
    concurrency: int,
    sample_sha256: str,
    profile_sha256: str,
    measurement_seconds: float,
    telemetry: Path,
) -> dict[str, Any]:
    warm_manifest = json.loads((root / "warmup/run_manifest.json").read_text())
    measured_manifest = json.loads((root / "measured/run_manifest.json").read_text())
    outcomes = _rows(root / "measured/outcomes.jsonl")
    failures = _rows(root / "measured/failures.jsonl")
    warm_outcomes = _rows(root / "warmup/outcomes.jsonl")
    usage: dict[str, float] = {}
    for row in outcomes:
        for key, value in row.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    result = {
        "format_version": "agentic-nemotron-concurrency-result-v1",
        "concurrency": concurrency,
        "sample_sha256": sample_sha256,
        "profile_sha256": profile_sha256,
        "status": "complete"
        if (
            warm_manifest.get("status") == measured_manifest.get("status") == "complete"
            and len(warm_outcomes) == 4
            and len(outcomes) == 64
            and not failures
        )
        else "invalid",
        "warmup_succeeded": len(warm_outcomes),
        "succeeded": len(outcomes),
        "failed": len(failures),
        "measurement_seconds": measurement_seconds,
        "usage": usage,
        "latency_seconds": {
            "mean": sum(row["duration_seconds"] for row in outcomes) / len(outcomes)
            if outcomes
            else None,
            "maximum": max((row["duration_seconds"] for row in outcomes), default=None),
        },
        "gpu_telemetry": str(telemetry.resolve()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(root / "result.json", result)
    return result


def _ensure_sweep(
    *,
    sweep_root: Path,
    judge_plan_root: Path,
    tokenizer_snapshot: Path,
) -> dict[str, Any]:
    sweep_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = sweep_root.parent / ".sweep-prepare.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = sweep_root / "run_manifest.json"
        if not manifest_path.is_file():
            prepare(judge_plan_root, tokenizer_snapshot, sweep_root)
        return json.loads(manifest_path.read_text())


def _wait_for_selection(
    *,
    sweep_root: Path,
    profile_sha256: str,
    role_end: float,
) -> dict[str, Any]:
    destination = sweep_root / "selection.json"
    while not destination.is_file():
        complete = all(
            (sweep_root / f"concurrency-{value}/result.json").is_file()
            for value in (4, 8, 16, 32)
        )
        if complete or time.time() >= role_end - 120:
            with (sweep_root / ".selection.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not destination.is_file():
                    select(sweep_root, profile_sha256)
            break
        time.sleep(15)
    return json.loads(destination.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    args = parser.parse_args()
    if args.worker_count != 5 or not 0 <= args.worker_index < 5:
        raise SystemExit("adaptive judge requires worker index 0..4 of exactly five")
    plan = json.loads(args.plan.read_text())
    judge = plan["judge"]
    plan_root = Path(judge["plan_root"])
    judge_manifest = plan_root / "run_manifest.json"
    if not judge_manifest.is_file():
        raise SystemExit("strict generation barrier is not open: judge plan is absent")
    profile = load_profile(args.profile)
    expected = judge["bulk_model"]
    if profile["model"] != {
        "model_id": expected["model_id"],
        "model_revision": expected["model_revision"],
    }:
        raise SystemExit("Nemotron profile differs from the adaptive plan")
    sweep_root = args.plan.parent / "nemotron-sweep"
    sweep = _ensure_sweep(
        sweep_root=sweep_root,
        judge_plan_root=plan_root,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )
    role_end = min(
        float(os.environ["SLURM_JOB_END_TIME"]),
        time.time() + float(sweep["hard_cap_seconds"]),
    )
    selected = 4
    if args.worker_index < 4:
        concurrency = int(sweep["worker_assignment"][str(args.worker_index)])
        candidate = sweep_root / f"concurrency-{concurrency}"
        warm = _runner(
            tasks=Path(sweep["warmup_tasks"]["path"]),
            judge_plan=judge_manifest,
            output=candidate / "warmup",
            concurrency=concurrency,
            base_url=args.base_url,
            model_id=expected["model_id"],
            revision=expected["model_revision"],
            pilot_only=True,
            role_end=role_end,
        )
        started = time.monotonic()
        measured = _runner(
            tasks=Path(sweep["measured_tasks"]["path"]),
            judge_plan=judge_manifest,
            output=candidate / "measured",
            concurrency=concurrency,
            base_url=args.base_url,
            model_id=expected["model_id"],
            revision=expected["model_revision"],
            pilot_only=True,
            role_end=role_end,
        )
        elapsed = time.monotonic() - started
        if (candidate / "warmup/run_manifest.json").is_file() and (
            candidate / "measured/run_manifest.json"
        ).is_file():
            _benchmark_result(
                candidate,
                concurrency=concurrency,
                sample_sha256=sweep["sample_sha256"],
                profile_sha256=profile["profile_sha256"],
                measurement_seconds=elapsed,
                telemetry=args.telemetry,
            )
        print(
            f"SWEEP_SLOT={concurrency} WARMUP_STATUS={warm.returncode} "
            f"MEASURED_STATUS={measured.returncode}",
            flush=True,
        )
        decision = _wait_for_selection(
            sweep_root=sweep_root,
            profile_sha256=profile["profile_sha256"],
            role_end=role_end,
        )
        selected = int(decision["concurrency"])
    output = args.plan.parent / f"nemotron-workers/worker-{args.worker_index:05d}"
    while True:
        result = _runner(
            tasks=plan_root / "bulk_tasks.jsonl",
            judge_plan=judge_manifest,
            output=output,
            concurrency=selected,
            base_url=args.base_url,
            model_id=expected["model_id"],
            revision=expected["model_revision"],
            pilot_only=False,
            claim_root=Path(judge["claim_root"]),
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
        manifest_path = output / "run_manifest.json"
        if result.returncode or not manifest_path.is_file():
            return result.returncode or 2
        runtime = json.loads(manifest_path.read_text())
        if runtime.get("status") == "complete" and runtime.get("remaining_count") == 0:
            return 0
        if runtime.get("stop_reason") != "shared_claims_busy":
            return 0 if runtime.get("stop_reason") == "allocation_deadline" else 2
        if time.time() >= float(os.environ["SLURM_JOB_END_TIME"]) - 180:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
