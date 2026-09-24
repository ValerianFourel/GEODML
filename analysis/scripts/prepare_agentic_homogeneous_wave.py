#!/usr/bin/env python3
"""Prepare or explicitly submit one homogeneous finite wave as held Slurm jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_FORMAT = "geodml-agentic-segment-plan-v1"
SUBMISSION_FORMAT = "geodml-agentic-held-wave-v1"
SAFE_EXPORT = re.compile(r"[A-Za-z0-9_./:-]+")
ENVIRONMENT_KEY = re.compile(r"[A-Z][A-Z0-9_]*")
PROTECTED_ENVIRONMENT = frozenset({
    "GEODML_SEGMENT_PLAN", "GEODML_SEGMENT_ID", "GEODML_MODEL_SLUG",
    "GEODML_SEGMENT_ORDINAL",
    "GEODML_EXECUTION_REPOSITORY", "GEODML_EXECUTION_COMMIT",
    "GEODML_APPROVED_WALLTIME", "GEODML_DATASET_ROOT", "GEODML_WAVE_ROOT",
    "GEODML_WAVE_OUTPUT_ROOT", "GEODML_WORKER_INDEX", "GEODML_WORKER_COUNT",
    "GEODML_ALLOCATION_ESTIMATE",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _time(seconds: int) -> str:
    if type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("every held job must have an approved walltime of at most one hour")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _safe(value: str, name: str) -> str:
    if SAFE_EXPORT.fullmatch(value) is None:
        raise ValueError(f"{name} contains characters unsafe for Slurm export")
    return value


def build_submission(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    execution_repository: Path,
    runner: Path,
    account: str,
    partition: str,
    output_root: Path,
    dataset_root: Path,
    materialization_root: Path,
    runtime_environment: Mapping[str, str],
    execution_commit: str,
    allocation_estimate: str,
) -> dict[str, Any]:
    """Build independent held jobs; actual start spacing is handled after submission."""

    if plan.get("format_version") != PLAN_FORMAT:
        raise ValueError("unsupported segment plan format")
    if plan.get("approval_status") not in {None, "approved"}:
        raise ValueError("wave plan is not approved")
    segments = plan.get("segments")
    if not isinstance(segments, list) or len(segments) != plan.get("segment_count"):
        raise ValueError("segment plan has inconsistent segment rows")
    model = plan.get("model")
    if model not in {"qwen38", "llama4", "nemotron"}:
        raise ValueError("unsupported wave model")
    if {segment.get("model") for segment in segments} != {model}:
        raise ValueError("a wave must contain exactly one model role")
    if plan.get("max_concurrent_allocations") != 5:
        raise ValueError("held wave must use the five-allocation admission limit")
    if plan.get("start_gap_seconds", 0) < 600:
        raise ValueError("held wave must keep at least ten minutes between observed starts")
    if re.fullmatch(r"[0-9a-f]{40}", execution_commit) is None:
        raise ValueError("execution commit must be a full lowercase Git SHA")
    for path, label in (
        (plan_path, "plan path"),
        (execution_repository, "execution repository"),
        (runner, "runner"),
        (output_root, "output root"),
        (dataset_root, "dataset root"),
        (materialization_root, "materialization root"),
    ):
        _safe(str(path.resolve()), label)
    for value, label in (
        (str(plan["plan_id"]), "plan ID"),
        (str(model), "model"),
        (account, "account"),
        (partition, "partition"),
    ):
        _safe(value, label)
    if not allocation_estimate.strip():
        raise ValueError("allocation estimate is required")
    runtime_environment = {**runtime_environment, "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY": "1"}
    required_environment = {
        "ACL_ARR_VENV", "GEODML_CACHE_ROOT", "GEODML_WORKER_LAUNCHER",
    }
    if model in {"qwen38", "llama4"}:
        required_environment.update({
            "SEARCH_AGENTIC_PROFILE", "SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT",
            "SEARCH_AGENTIC_CROSS_ENCODER_REVISION", "SEARCH_AGENTIC_DDG_SNAPSHOT",
            "SEARCH_AGENTIC_SEARXNG_SNAPSHOT", "SEARCH_AGENTIC_PROMPTS_JSONL",
            "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL", "SEARCH_AGENTIC_PROMPT_COUNT",
        })
    else:
        required_environment.update({
            "GEODML_JUDGE_MANIFEST", "GEODML_JUDGE_ROLE", "GEODML_JUDGE_PROFILE",
        })
    missing_environment = sorted(required_environment - set(runtime_environment))
    if missing_environment:
        raise ValueError(
            "runtime environment lacks required keys: " + ", ".join(missing_environment)
        )
    if PROTECTED_ENVIRONMENT.intersection(runtime_environment):
        raise ValueError("runtime environment attempts to override protected wave state")
    for key, value in runtime_environment.items():
        if ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid runtime environment key: {key}")
        _safe(value, f"runtime environment value {key}")
    materialization = json.loads(
        (materialization_root / "materialization.json").read_text(encoding="utf-8")
    )
    if (
        materialization.get("plan_id") != plan["plan_id"]
        or materialization.get("model") != model
    ):
        raise ValueError("segment queues do not match the approved plan")
    wave_rows = {
        row["segment_id"]: row
        for row in materialization.get("waves", [])
    }
    if set(wave_rows) != {row["segment_id"] for row in segments}:
        raise ValueError("segment queue materialization is incomplete")
    walltime = _time(plan["walltime_seconds"])
    estimate_reference = "sha256:" + hashlib.sha256(
        allocation_estimate.encode("utf-8")
    ).hexdigest()
    jobs: list[dict[str, Any]] = []
    interactive: list[dict[str, Any]] = []
    exhausted: list[dict[str, Any]] = []
    for segment in segments:
        if segment.get("approval_status") != "approved":
            raise ValueError("every segment must be explicitly approved")
        if segment.get("walltime_seconds") != plan["walltime_seconds"]:
            raise ValueError("segment walltime differs from its wave")
        segment_id = _safe(str(segment["segment_id"]), "segment ID")
        wave_row = wave_rows[segment_id]
        backlog = wave_row.get("backlog")
        if not isinstance(backlog, dict) or type(backlog.get("task_count")) is not int:
            raise ValueError(f"segment materialization lacks task count: {segment_id}")
        if backlog["task_count"] < 0:
            raise ValueError(f"segment materialization has invalid task count: {segment_id}")
        wave_root = Path(wave_row["wave_root"]).resolve()
        if not (wave_root / "run_manifest.json").is_file():
            raise ValueError(f"segment wave is missing: {segment_id}")
        if backlog["task_count"] == 0:
            exhausted.append({
                "segment_id": segment_id,
                "mode": segment.get("mode"),
                "reason": "no_eligible_tasks_for_keyword_direction",
            })
            continue
        logs = output_root / "logs"
        derived_environment = {
            "GEODML_SEGMENT_PLAN": str(plan_path.resolve()),
            "GEODML_SEGMENT_ID": segment_id,
            "GEODML_SEGMENT_ORDINAL": str(segment["ordinal"]),
            "GEODML_MODEL_SLUG": model,
            "GEODML_EXECUTION_REPOSITORY": str(execution_repository.resolve()),
            "GEODML_EXECUTION_COMMIT": execution_commit,
            "GEODML_APPROVED_WALLTIME": walltime,
            "GEODML_ALLOCATION_ESTIMATE": estimate_reference,
            "GEODML_DATASET_ROOT": str(dataset_root.resolve()),
            "GEODML_WAVE_ROOT": str(wave_root),
            "GEODML_WAVE_OUTPUT_ROOT": str((output_root / segment_id).resolve()),
            "GEODML_WAVE_LOG_ROOT": str(logs.resolve()),
            # Every allocation sees the complete immutable keyword queue. The
            # durable ledger, rather than modulo sharding, distributes missing
            # tasks between staggered allocations. This lets later allocations
            # finish work left by an earlier one-hour segment.
            "GEODML_WORKER_INDEX": "0",
            "GEODML_WORKER_COUNT": "1",
        }
        merged_environment = {**runtime_environment, **derived_environment}
        for key, value in merged_environment.items():
            _safe(value, f"effective environment value {key}")
        memory = segment.get("memory")
        memory_argument = "--mem=0" if memory == "all" else f"--mem={memory}"
        if segment.get("mode") == "interactive":
            comment = (
                f"geodml-v2:{plan['plan_id']}:{segment_id}:{model}:interactive"
            )
            interactive.append({
                **dict(segment),
                "wave_root": str(wave_root),
                "backlog": backlog,
                "primary_keyword_id": wave_row.get("primary_keyword_id"),
                "slurm_comment": comment,
                "environment": dict(sorted(merged_environment.items())),
                "allocation_command": [
                    "salloc",
                    f"--account={account}",
                    f"--partition={partition}",
                    f"--nodes={segment['nodes']}",
                    "--ntasks=1",
                    f"--cpus-per-task={segment['cpus']}",
                    memory_argument,
                    f"--gres=gpu:{segment['gpus']}",
                    "--exclusive",
                    f"--time={walltime}",
                    f"--job-name=geodml-{model}-interactive-{segment['ordinal']:03d}",
                    f"--comment={comment}",
                ],
                "step_command": [
                    "srun", "--jobid=${SLURM_JOB_ID}", "--nodes=1", "--ntasks=1",
                    f"--cpus-per-task={segment['cpus']}", "--cpu-bind=none",
                    str(runner.resolve()),
                ],
            })
            continue
        if segment.get("mode") != "batch":
            raise ValueError("unsupported segment execution mode")
        export = ",".join((
            "ALL",
            *(f"{key}={value}" for key, value in sorted(merged_environment.items())),
        ))
        command = [
            "sbatch",
            "--parsable",
            "--hold",
            "--no-requeue",
            f"--account={account}",
            f"--partition={partition}",
            f"--nodes={segment['nodes']}",
            "--ntasks=1",
            f"--cpus-per-task={segment['cpus']}",
            memory_argument,
            f"--gres=gpu:{segment['gpus']}",
            "--exclusive",
            f"--time={walltime}",
            f"--job-name=geodml-{model}-{segment['ordinal']:03d}",
            (
                f"--comment=geodml-v2:{plan['plan_id']}:{segment_id}:"
                f"{model}:batch"
            ),
            f"--chdir={execution_repository.resolve()}",
            f"--output={logs}/slurm-%j.out",
            f"--error={logs}/slurm-%j.err",
            f"--export={export}",
            str(runner.resolve()),
        ]
        jobs.append({
            "segment_id": segment_id,
            "ordinal": segment["ordinal"],
            "model": model,
            "status": "prepared",
            "command": command,
        })
    from analysis.scripts.verify_inference_allocation import cluster_profile
    identity = {
        "execution_boundary_profile": cluster_profile("jupiter"),
        "format_version": SUBMISSION_FORMAT,
        "plan_id": plan["plan_id"],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "execution_commit": execution_commit,
        "account": account,
        "partition": partition,
        "runner": str(runner.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "materialization_root": str(materialization_root.resolve()),
        "runtime_environment": dict(sorted(runtime_environment.items())),
        "allocation_estimate": allocation_estimate,
        "jobs": jobs,
        "interactive_segments": interactive,
        "exhausted_segments": exhausted,
    }
    return {
        **identity,
        "submission_id": "held-wave-" + hashlib.sha256(_canonical(identity)).hexdigest()[:24],
        "status": "prepared",
        "maximum_concurrent_allocations": 5,
        "minimum_observed_start_gap_seconds": plan["start_gap_seconds"],
        "batch_job_count": len(jobs),
        "interactive_segment_count": len(interactive),
        "exhausted_segment_count": len(exhausted),
        "maximum_gpu_hours": plan["maximum_gpu_hours"],
    }


def submit_held(bundle: dict[str, Any], *, output_root: Path) -> dict[str, Any]:
    """Submit approved jobs held; this never releases or resubmits them."""

    receipt = output_root / "submission-receipt.json"
    intent = output_root / "submission-intent.json"
    if intent.exists() or receipt.exists():
        raise FileExistsError("submission state exists; reconcile Slurm instead of resubmitting")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    requested = {
        "format_version": SUBMISSION_FORMAT,
        "submission_id": bundle["submission_id"],
        "status": "submission_requested",
        "requested_at": _now(),
        "jobs": bundle["jobs"],
    }
    _atomic(intent, requested)
    submitted: list[dict[str, Any]] = []
    try:
        for job in bundle["jobs"]:
            result = subprocess.run(
                job["command"], text=True, capture_output=True, check=False, timeout=60
            )
            match = re.fullmatch(
                r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", result.stdout.strip()
            )
            if result.returncode != 0 or match is None:
                raise RuntimeError(
                    f"held submission became uncertain for {job['segment_id']}; inspect Slurm"
                )
            submitted.append({
                "segment_id": job["segment_id"],
                "job_id": match.group(1),
                "cluster": match.group(2),
                "status": "held",
            })
            _atomic(intent, {**requested, "status": "submitting", "submitted": submitted})
    except BaseException as error:
        _atomic(intent, {
            **requested,
            "status": "submission_uncertain",
            "submitted": submitted,
            "error_type": type(error).__name__,
        })
        raise
    completed = {
        "format_version": SUBMISSION_FORMAT,
        "submission_id": bundle["submission_id"],
        "status": "held",
        "submitted_at": _now(),
        "jobs": submitted,
    }
    _atomic(receipt, completed)
    _atomic(intent, {**requested, "status": "held", "submitted": submitted})
    return completed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-repository", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--runtime-environment", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--allocation-estimate", required=True)
    parser.add_argument("--submit-held", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    runtime_environment = json.loads(
        args.runtime_environment.read_text(encoding="utf-8")
    )
    if not isinstance(runtime_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in runtime_environment.items()
    ):
        raise TypeError("runtime environment must be a JSON object of strings")
    bundle = build_submission(
        plan,
        plan_path=args.plan,
        execution_repository=args.execution_repository,
        runner=args.runner,
        account=args.account,
        partition=args.partition,
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        materialization_root=args.materialization_root,
        runtime_environment=runtime_environment,
        execution_commit=args.execution_commit,
        allocation_estimate=args.allocation_estimate,
    )
    if not args.submit_held:
        print(json.dumps({"dry_run": True, "submission": bundle}, indent=2, sort_keys=True))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("refusing to submit from a non-empty wave output root")
    _atomic(args.output_root / "submission-plan.json", bundle)
    receipt = submit_held(bundle, output_root=args.output_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
