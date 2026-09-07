#!/usr/bin/env python3
"""Recover one pilot rerank shard into new, non-confirmatory artifacts."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.run_acl_arr_vllm import (  # noqa: E402
    VllmChatClient, _append, _atomic_json, _now, _prepare_primary,
    _primary_context, _read_jsonl, _sha256,
)
from analysis.scripts.acl_arr_recovery_decode import PROTOCOL, recover_one  # noqa: E402

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_REVISION = "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"


def load_source(run_root, expected_plan_commit, *, expected_counts=(384, 319, 65)):
    """Validate original artifacts before any output creation or GPU startup."""
    root = Path(run_root).resolve()
    plan_path = root / "plan/run_manifest.json"
    manifest = json.loads(plan_path.read_text())
    if manifest["source_git_commit"] != expected_plan_commit:
        raise ValueError("original plan commit does not match expected commit")
    models = [m for m in manifest["models"] if m["model_id"] == MODEL_ID
              and m["model_revision"] == MODEL_REVISION]
    if len(models) != 1:
        raise ValueError("expected exactly one pinned Llama model")
    config_id = models[0]["configuration_id"]
    tasks_path = root / "plan/tasks" / config_id / "rerank.jsonl"
    plan, tasks, model, pipeline = _primary_context(plan_path, tasks_path)
    if pipeline != "rerank" or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("expected unique rerank tasks")
    shard = root / "results" / config_id / "rerank"
    shard_manifest_path = shard / "run_manifest.json"
    shard_manifest = json.loads(shard_manifest_path.read_text())
    runtime = json.loads((root / "pilot-runtime-manifest.json").read_text())
    if runtime.get("scientific_result") is not False or runtime.get("git_commit_sha") != expected_plan_commit:
        raise ValueError("original runtime must be a pilot at the expected plan commit")
    if (shard_manifest.get("tasks", {}).get("sha256") != _sha256(tasks_path)
            or Path(shard_manifest.get("source_manifest", "")).resolve() != plan_path):
        raise ValueError("original shard task hash or source manifest does not match the plan")
    for key, value in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                       ("pipeline", "rerank"), ("fake_backend", False)):
        if shard_manifest.get(key) != value:
            raise ValueError(f"original shard identity mismatch: {key}")
    source_files = [plan_path, tasks_path, shard_manifest_path,
                    shard / "outcomes.jsonl", shard / "failures.jsonl",
                    root / "pilot-runtime-manifest.json", root / "model-snapshots.json"]
    hashes = {str(p): _sha256(p) for p in source_files}
    for name in ("outcomes", "failures"):
        if shard_manifest.get(name + "_sha256") != hashes[str(shard / (name + ".jsonl"))]:
            raise ValueError(f"original {name} hash differs from shard manifest")
    locks = json.loads((root / "model-snapshots.json").read_text())["models"]
    matches = [r for r in locks if r["model_id"] == MODEL_ID and r["revision"] == MODEL_REVISION]
    if len(matches) != 1 or not Path(matches[0]["snapshot"]).is_dir():
        raise ValueError("pinned local model snapshot is missing or ambiguous")
    prompts = {p.prompt_id: p for p in plan.prompts}
    assignments = {a.assignment_id: a for a in plan.assignments}
    documents = {d.candidate_set_id: d for d in plan.document_sets}
    prepared = {}
    for task in tasks:
        assignment = assignments[task.assignment_id]
        prepared[task.task_id] = _prepare_primary(task, prompt=prompts[task.prompt_id],
            assignment=assignment, document_set=documents[assignment.candidate_set_id], model=model)
    outcomes = _read_jsonl(shard / "outcomes.jsonl")
    completed = set()
    for row in outcomes:
        task_id = row.get("task_id")
        if task_id not in prepared or task_id in completed:
            raise ValueError("original outcomes contain unknown or duplicate task IDs")
        item = prepared[task_id]
        if any(row.get(k) != v for k, v in item["base"].items()):
            raise ValueError("original outcome identity differs from frozen task")
        raw = row["raw_output"]
        if hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_output_sha256"):
            raise ValueError("original raw output hash mismatch")
        if item["validator"](raw) != row.get("parsed_output"):
            raise ValueError("original parsed output differs from validated raw output")
        completed.add(task_id)
    pending = [t.task_id for t in tasks if t.task_id not in completed]
    failure_ids = {r["task_id"] for r in _read_jsonl(shard / "failures.jsonl")}
    if failure_ids - set(prepared) or set(pending) - failure_ids:
        raise ValueError("unresolved tasks do not match recorded original failures")
    counts = (len(tasks), len(completed), len(pending))
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"unexpected source counts {counts}; expected {expected_counts}")
    return {"root": root, "plan": plan, "model": model, "tasks_path": tasks_path,
            "prepared": prepared, "outcomes": outcomes, "completed": completed,
            "pending": pending, "shard": shard, "hashes": hashes,
            "snapshot": str(Path(matches[0]["snapshot"]).resolve())}


def verify_unchanged(source):
    for path, digest in source["hashes"].items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"original artifact changed during recovery: {path}")


async def run_recovery(source, output, *, client, fake, recovery_commit,
                       max_concurrency, max_model_len, approved_walltime, allocation_estimate):
    if max_concurrency < 1 or max_concurrency > 8 or max_model_len != 40960:
        raise ValueError("this recovery requires context 40960 and concurrency 1 through 8")
    if approved_walltime != "00:30:00" or not allocation_estimate.strip():
        raise ValueError("this recovery requires the approved 30-minute budget and estimate")
    verify_unchanged(source)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = "acl-arr-pilot-recovery-" + uuid.uuid4().hex
    originals = output / "original-records"
    originals.mkdir()
    shutil.copyfile(source["shard"] / "run_manifest.json", originals / "shard-run-manifest.json")
    shutil.copyfile(source["root"] / "pilot-runtime-manifest.json", originals / "pilot-runtime-manifest.json")
    shutil.copyfile(source["shard"] / "failures.jsonl", originals / "failures.jsonl")
    outcomes_path = output / "outcomes.jsonl"
    shutil.copyfile(source["shard"] / "outcomes.jsonl", outcomes_path)
    # Preserve the original byte prefix even if the last JSONL record lacks LF.
    with outcomes_path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        needs_newline = f.read(1) != b"\n"
    if needs_newline:
        with outcomes_path.open("ab") as f:
            f.write(b"\n")
    completed = set(source["completed"])
    manifest_path = output / "recovery_manifest.json"
    manifest = {
        "format_version": "acl-arr-pilot-recovery-v1", "run_id": run_id,
        "status": "running", "scientific_result": False, "eligible_for_analysis": False,
        "fake_backend": fake, "decoding_protocol": PROTOCOL,
        "plan_source_git_commit": source["plan"].source_git_commit,
        "recovery_git_commit": recovery_commit, "original_artifacts_sha256": source["hashes"],
        "source_run_root": str(source["root"]), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "snapshot": source["snapshot"],
        "original_success_count": len(completed), "task_count": len(source["prepared"]),
        "pending_at_start": source["pending"], "completed_count": len(completed),
        "remaining_count": len(source["pending"]), "recovered_count": 0,
        "failed_this_invocation": 0, "started_at": _now(), "finished_at": None,
        "approved_walltime": approved_walltime, "allocation_estimate": allocation_estimate,
        "max_model_len": max_model_len, "max_concurrency": max_concurrency,
        "maximum_http_attempts_per_call": 1, "request_timeout_seconds": 120,
        "maximum_decode_attempts_per_task": max(i["schema"]["properties"]["ranked_document_ids"]["maxItems"]
                                                 for i in source["prepared"].values()),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_step_id": os.getenv("SLURM_STEP_ID"),
        "resources": {"nodes": 1, "gpus": 4, "requested_cpus": 32, "memory": "512G"},
        "gpu_hour_cap": 2, "outputs": {"outcomes": str(outcomes_path),
            "failures": str(output / "failures.jsonl"), "attempts": str(output / "attempts.jsonl")},
    }
    _atomic_json(manifest_path, manifest)
    semaphore = asyncio.Semaphore(max_concurrency)
    with outcomes_path.open("a", buffering=1) as outcomes, \
            (output / "failures.jsonl").open("x", buffering=1) as failures, \
            (output / "attempts.jsonl").open("x", buffering=1) as attempts:
        def emit_attempt(record):
            item = source["prepared"][record["task_id"]]
            request = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                       "rendered_prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                       "seed": item["seed"], "temperature": item["temperature"],
                       "max_tokens": item["max_tokens"]}
            _append(attempts, {**record, "run_id": run_id, "decoding_protocol": PROTOCOL,
                              "request": request})
            attempts.flush()
            os.fsync(attempts.fileno())

        async def one(task_id):
            item = source["prepared"][task_id]
            if fake:
                return {"ok": True, "base": item["base"], "raw_output": item["fake_output"],
                        "parsed_output": item["validator"](item["fake_output"]), "usage": {},
                        "started_at": _now(), "finished_at": _now(), "decode_attempt_count": 0}
            async with semaphore:
                return await recover_one(item, client=client, emit_attempt=emit_attempt)

        try:
            for future in asyncio.as_completed([one(t) for t in source["pending"]]):
                result = await future
                row = {**result["base"], "run_id": run_id, "fake_backend": fake,
                       "scientific_result": False, "eligible_for_analysis": False,
                       "decoding_protocol": PROTOCOL, "recovery_git_commit": recovery_commit,
                       **{k: v for k, v in result.items() if k not in {"base", "ok"}}}
                if result["ok"]:
                    row["raw_output_sha256"] = hashlib.sha256(row["raw_output"].encode()).hexdigest()
                    _append(outcomes, row)
                    outcomes.flush()
                    os.fsync(outcomes.fileno())
                    completed.add(row["task_id"])
                    manifest["recovered_count"] += 1
                else:
                    _append(failures, row)
                    failures.flush()
                    os.fsync(failures.fileno())
                    manifest["failed_this_invocation"] += 1
                manifest.update(completed_count=len(completed),
                                remaining_count=len(source["prepared"]) - len(completed))
                _atomic_json(manifest_path, manifest)
                print(f"RECOVERY_COMPLETED={len(completed)}/{len(source['prepared'])} "
                      f"RECOVERED={manifest['recovered_count']} FAILED={manifest['failed_this_invocation']}", flush=True)
            verify_unchanged(source)
            manifest["status"] = "complete" if not manifest["remaining_count"] else "complete_with_failures"
        except BaseException as exc:
            manifest["status"] = "interrupted_or_error"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest["finished_at"] = _now()
            manifest["outcomes_sha256"] = _sha256(outcomes_path)
            _atomic_json(manifest_path, manifest)
    print("RECOVERY_MANIFEST=" + str(manifest_path), flush=True)
    print("RECOVERY_STATUS=" + manifest["status"], flush=True)
    return 0 if manifest["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-plan-commit", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--approved-walltime", required=True)
    parser.add_argument("--allocation-estimate", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.approved_walltime != "00:30:00" or not args.allocation_estimate.strip():
        parser.error("the 30-minute recovery approval and estimate are required")
    if args.max_model_len != 40960 or not 1 <= args.max_concurrency <= 8:
        parser.error("context must be 40960 and concurrency must be 1 through 8")
    source = load_source(args.run_root, args.expected_plan_commit)
    if args.output_dir.exists():
        raise FileExistsError(f"recovery output already exists: {args.output_dir}")
    print("RECOVERY_PREFLIGHT=PASS total=384 saved=319 pending=65", flush=True)
    if args.preflight_only:
        return 0
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("real recovery requires an existing approved Slurm allocation")
    commit = subprocess.check_output(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True).strip()

    async def execute():
        async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=MODEL_ID,
                                  timeout_seconds=120, maximum_attempts=1) as client:
            return await run_recovery(source, args.output_dir, client=client, fake=False,
                recovery_commit=commit, max_concurrency=args.max_concurrency,
                max_model_len=args.max_model_len, approved_walltime=args.approved_walltime,
                allocation_estimate=args.allocation_estimate)
    return asyncio.run(execute())


if __name__ == "__main__":
    raise SystemExit(main())
