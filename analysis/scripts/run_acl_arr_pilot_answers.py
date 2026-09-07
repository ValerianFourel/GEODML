#!/usr/bin/env python3
"""Run one isolated, budgeted Llama answer pilot with durable checkpoints."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.run_acl_arr_vllm import (  # noqa: E402
    VllmChatClient, _append, _atomic_json, _now, _prepare_primary, _primary_context, _read_jsonl, _sha256,
)
from analysis.scripts.recover_acl_arr_llama_rerank import load_source, MODEL_ID, MODEL_REVISION  # noqa: E402


def verify_hashes(hashes):
    for path, digest in hashes.items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"source artifact changed: {path}")


def _validate_saved(row, prepared):
    task_id = row.get("task_id")
    if task_id not in prepared:
        raise ValueError("saved answer references unknown task")
    item = prepared[task_id]
    if any(row.get(k) != v for k, v in item["base"].items()):
        raise ValueError("saved answer identity does not match frozen task")
    if row.get("scientific_result") is not False or row.get("eligible_for_analysis") is not False:
        raise ValueError("saved answer must remain pilot-only")
    raw = row["raw_output"]
    if hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_output_sha256"):
        raise ValueError("saved answer raw hash mismatch")
    if item["validator"](raw) != row.get("parsed_output"):
        raise ValueError("saved answer validation mismatch")
    return task_id


async def run_answers(items, output, *, client, source_hashes, metadata, stop_at,
                      clock=time.monotonic, max_concurrency=8, resume_from=None):
    """Stop dispatch at the deadline, drain active calls, and retain partial work."""
    if not 1 <= max_concurrency <= 8:
        raise ValueError("answer concurrency must be between one and eight")
    prepared = {i["base"]["task_id"]: i for i in items}
    if len(prepared) != len(items) or any(i["base"]["pipeline"] != "answer" for i in items):
        raise ValueError("expected unique answer tasks")
    hashes = dict(source_hashes)
    completed = set()
    if resume_from is not None:
        previous = Path(resume_from).resolve()
        previous_manifest = previous / "answer_manifest.json"
        previous_outcomes = previous / "outcomes.jsonl"
        record = json.loads(previous_manifest.read_text())
        for key in ("plan_source_git_commit", "tasks_sha256", "model_id", "model_revision"):
            if record.get(key) != metadata.get(key):
                raise ValueError(f"resume identity mismatch: {key}")
        if record.get("scientific_result") is not False or record.get("eligible_for_analysis") is not False:
            raise ValueError("resume manifest must be pilot-only")
        rows = _read_jsonl(previous_outcomes) if previous_outcomes.stat().st_size else []
        for row in rows:
            task_id = _validate_saved(row, prepared)
            if task_id in completed:
                raise ValueError("duplicate saved answer task ID")
            completed.add(task_id)
        hashes.update({str(p): _sha256(p) for p in (previous_manifest, previous_outcomes)})
    verify_hashes(hashes)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    outcome_path = output / "outcomes.jsonl"
    if resume_from is not None:
        shutil.copyfile(previous_outcomes, outcome_path)
        shutil.copyfile(previous_manifest, output / "previous-answer-manifest.json")
        if outcome_path.stat().st_size:
            with outcome_path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
            if needs_newline:
                with outcome_path.open("ab") as f:
                    f.write(b"\n")
    pending = deque(i for i in items if i["base"]["task_id"] not in completed)
    initial_completed = len(completed)
    manifest = {**metadata, "format_version": "acl-arr-pilot-answer-step-v1",
                "run_id": "acl-arr-answer-" + uuid.uuid4().hex,
                "status": "running", "scientific_result": False, "eligible_for_analysis": False,
                "fake_backend": False, "pipeline": "answer", "source_artifacts_sha256": hashes,
                "task_count": len(items), "already_completed_count": initial_completed,
                "completed_count": len(completed), "remaining_count": len(pending),
                "attempted_this_invocation": 0, "failed_this_invocation": 0,
                "unattempted_count": len(pending), "started_at": _now(), "finished_at": None,
                "max_concurrency": max_concurrency, "request_timeout_seconds": 300,
                "maximum_http_attempts": 1, "answer_protocol": "unchanged-primary-answer-v1"}
    manifest_path = output / "answer_manifest.json"
    _atomic_json(manifest_path, manifest)
    with outcome_path.open("a", buffering=1) as outcomes, \
            (output / "failures.jsonl").open("x", buffering=1) as failures, \
            (output / "attempts.jsonl").open("x", buffering=1) as attempts:
        def persist(stream, row):
            _append(stream, row)
            stream.flush()
            os.fsync(stream.fileno())

        async def worker():
            while pending and clock() < stop_at:
                item = pending.popleft()
                row = {**item["base"], "run_id": manifest["run_id"],
                       "scientific_result": False, "eligible_for_analysis": False,
                       "started_at": _now(), "raw_output": None, "usage": {},
                       "request": {k: item[k] for k in ("seed", "temperature", "max_tokens", "schema")},
                       "rendered_prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest()}
                try:
                    raw, usage = await asyncio.wait_for(client.complete(**{
                        k: item[k] for k in ("prompt", "seed", "temperature", "max_tokens", "schema", "schema_name")}), timeout=300)
                    row.update(raw_output=raw, usage=dict(usage),
                               raw_output_sha256=hashlib.sha256(raw.encode()).hexdigest())
                    row["parsed_output"] = item["validator"](raw)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["finished_at"] = _now()
                persist(attempts, row)
                manifest["attempted_this_invocation"] += 1
                if "error" in row:
                    persist(failures, row)
                    manifest["failed_this_invocation"] += 1
                else:
                    persist(outcomes, row)
                    completed.add(row["task_id"])
                manifest.update(completed_count=len(completed), remaining_count=len(items) - len(completed),
                                unattempted_count=len(pending))
                _atomic_json(manifest_path, manifest)
                print(f"ANSWER_COMPLETED={len(completed)}/{len(items)} "
                      f"FAILED={manifest['failed_this_invocation']} UNSTARTED={len(pending)}", flush=True)

        workers = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
        try:
            await asyncio.gather(*workers)
            verify_hashes(hashes)
            manifest["status"] = ("complete" if not manifest["remaining_count"] else
                                  "checkpointed" if pending else "complete_with_failures")
        except BaseException as exc:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            manifest.update(status="interrupted_or_error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            manifest.update(finished_at=_now(), outcomes_sha256=_sha256(outcome_path))
            _atomic_json(manifest_path, manifest)
    print("ANSWER_MANIFEST=" + str(manifest_path), flush=True)
    print("ANSWER_STATUS=" + manifest["status"], flush=True)
    return {"complete": 0, "checkpointed": 3, "complete_with_failures": 2}[manifest["status"]]


def load_answers(root, recovery_results, *, expected_count=384):
    source = load_source(root, "f9c35e499b708a4c812b49a3f550e097306ce25d")
    recovery_results = Path(recovery_results).resolve()
    recovery_manifest = recovery_results / "recovery_manifest.json"
    recovery_outcomes = recovery_results / "outcomes.jsonl"
    record = json.loads(recovery_manifest.read_text())
    if (record.get("status") != "complete" or record.get("completed_count") != expected_count
            or record.get("remaining_count") != 0 or record.get("scientific_result") is not False
            or record.get("eligible_for_analysis") is not False
            or record.get("fake_backend") is not False
            or record.get("original_artifacts_sha256") != source["hashes"]
            or record.get("outcomes_sha256") != _sha256(recovery_outcomes)):
        raise ValueError("completed rerank recovery does not match original pilot")
    rows = _read_jsonl(recovery_outcomes)
    if len(rows) != expected_count or {r["task_id"] for r in rows} != set(source["prepared"]):
        raise ValueError("rerank recovery coverage is incomplete")
    for row in rows:
        item = source["prepared"][row["task_id"]]
        if any(row.get(k) != v for k, v in item["base"].items()):
            raise ValueError("rerank recovery task identity mismatch")
        if item["validator"](row["raw_output"]) != row["parsed_output"]:
            raise ValueError("rerank recovery output validation mismatch")
    tasks = source["root"] / "plan/tasks" / source["model"].configuration_id / "answer.jsonl"
    plan, answer_tasks, model, pipeline = _primary_context(source["root"] / "plan/run_manifest.json", tasks)
    if pipeline != "answer" or len(answer_tasks) != expected_count or model.model_revision != MODEL_REVISION:
        raise ValueError("expected 384 pinned Llama answer tasks")
    prompts = {p.prompt_id: p for p in plan.prompts}
    assignments = {a.assignment_id: a for a in plan.assignments}
    documents = {d.candidate_set_id: d for d in plan.document_sets}
    prepared = []
    for task in answer_tasks:
        assignment = assignments[task.assignment_id]
        prepared.append(_prepare_primary(task, prompt=prompts[task.prompt_id], assignment=assignment,
            document_set=documents[assignment.candidate_set_id], model=model))
    hashes = {**source["hashes"], **{str(p): _sha256(p) for p in (tasks, recovery_manifest, recovery_outcomes)}}
    return prepared, hashes, {"plan_source_git_commit": plan.source_git_commit,
        "tasks_sha256": _sha256(tasks), "tasks_path": str(tasks), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "rerank_recovery_results": str(recovery_results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--rerank-recovery-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-submit-epoch", type=float, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--approved-walltime", required=True)
    parser.add_argument("--allocation-estimate", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.approved_walltime != "01:00:00" or not args.allocation_estimate.strip():
        parser.error("the approved one-hour step and estimate are required")
    items, hashes, metadata = load_answers(args.run_root, args.rerank_recovery_results)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    print("ANSWER_PREFLIGHT=PASS tasks=384; original and recovered reranks preserved", flush=True)
    if args.preflight_only:
        return 0
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SLURM_STEP_ID"):
        parser.error("run inside an approved existing Slurm job step")
    seconds = args.stop_submit_epoch - time.time()
    if seconds > 3000 or seconds <= 0:
        parser.error("submission deadline must be within the remaining 50-minute dispatch window")
    deadline = time.monotonic() + seconds
    commit = subprocess.check_output(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True).strip()
    metadata.update(execution_git_commit=commit, approved_walltime=args.approved_walltime,
        allocation_estimate=args.allocation_estimate, max_model_len=41984,
        stop_submit_epoch=args.stop_submit_epoch, drain_window_seconds=600,
        slurm_job_id=os.getenv("SLURM_JOB_ID"), slurm_step_id=os.getenv("SLURM_STEP_ID"),
        resources={"nodes": 1, "gpus": 4, "requested_cpus": 32, "memory": "512G"}, gpu_hour_cap=4)

    async def execute():
        async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=MODEL_ID,
                                  timeout_seconds=300, maximum_attempts=1) as client:
            return await run_answers(items, args.output_dir, client=client, source_hashes=hashes,
                metadata=metadata, stop_at=deadline, resume_from=args.resume_from)
    return asyncio.run(execute())


if __name__ == "__main__":
    raise SystemExit(main())
