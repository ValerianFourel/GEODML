#!/usr/bin/env python3
"""Freeze twenty new SI-v3 cells, or print saved judgments. Never allocate or infer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_cells import iter_cells
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows, verify_record_reference
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.interpretability.pipeline.source_importance import PROTOCOL
from analysis.scripts import horeka_nemotron as runner
from analysis.scripts.prepare_horeka_qwen import immutable_json
from analysis.scripts.prepare_source_importance_tasks import build_cell
from analysis.scripts.replay_source_importance_judge import frozen_cells, indexed, rows
from analysis.scripts.try_source_importance_judge import pick

SEED = 2026093011


def validate_selection(bundle, excluded, job):
    frozen_cells(bundle)
    cells = bundle["cells"]
    prompts = {c["prompt_id"] for c in cells}
    if (len(cells) != 20 or len(prompts) != 20 or prompts & excluded
            or sorted(c["model"] for c in cells) != ["llama4"] * 10 + ["qwen38"] * 10
            or bundle.get("existing_job_id") != job or bundle.get("protocol") != PROTOCOL):
        raise ValueError("expected twenty distinct fresh prompts, ten per model, bound to this allocation")


def sample_refs(root, model, excluded_prompts, *, deadline):
    """Choose verified candidates; audit unavailable references before freezing inputs."""
    tasks = {}
    for n, row in enumerate(iter_sealed_rows(root, "task_definitions", required=True)):
        if n % 50000 == 0:
            if time.time() >= deadline:
                raise ValueError("insufficient remaining time; no inference started")
            print(f"SELECT {model}: scanned {n} task definitions", flush=True)
        if (row.get("model") == model and row.get("stage", "generation") == "generation"
                and row.get("prompt_id") not in excluded_prompts):
            tasks[identity_fingerprint(ClaimIdentity(**row["claim_identity"]))] = row
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=256).snapshot()["latest"]
    candidates, seen_prompts = [], set()
    for fingerprint in sorted(tasks, key=lambda f: hashlib.sha256(f"{SEED}:{f}".encode()).hexdigest()):
        task, event = tasks[fingerprint], latest.get(fingerprint, {})
        prompt = task.get("prompt_id")
        if event.get("state") != "completed" or not prompt or prompt in seen_prompts:
            continue
        refs = {r["table"]: r for r in event.get("record_references", [])}
        if not {"generations", "traces"} <= refs.keys():
            continue
        seen_prompts.add(prompt)
        candidates.append({"fingerprint": fingerprint, "model": model,
            **{k: task.get(k) for k in ("prompt_id", "method", "engine", "condition")},
            "generation_ref": refs["generations"], "trace_ref": refs["traces"]})
    chosen, rejected = [], []
    while candidates and len(chosen) < 10:
        batch = pick(candidates, count=10 - len(chosen), seed=SEED)
        examined = {r["fingerprint"] for r in batch}
        candidates = [r for r in candidates if r["fingerprint"] not in examined]
        for candidate in batch:
            if time.time() >= deadline:
                raise ValueError("insufficient remaining time; no inference started")
            failed = [candidate[k] for k in ("generation_ref", "trace_ref")
                      if not verify_record_reference(root, candidate[k])]
            if failed:
                event = {"model": model, "prompt_id": candidate["prompt_id"],
                         "fingerprint": candidate["fingerprint"], "reason": "unverified_record_reference",
                         "failed_references": failed}
                rejected.append(event)
                print("EXCLUDED_CANDIDATE " + json.dumps(event), flush=True)
            else:
                chosen.append(candidate)
    if len(chosen) != 10:
        raise ValueError(f"need ten distinct verified fresh prompts for {model}; found {len(chosen)}, rejected {len(rejected)}")
    print(f"SELECT {model}: ten verified candidates; excluded {len(rejected)} unverified candidates", flush=True)
    return chosen, rejected


def freeze(args):
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify("horeka")
    fields = dict(t.split("=", 1) for t in subprocess.check_output(
        ["scontrol", "show", "job", args.existing_job_id, "-o"], text=True).split() if "=" in t)
    runner.check_bound_allocation({"existing_job_id": args.existing_job_id, "minimum_remaining_seconds": 1200},
                                  os.environ.get("SLURM_JOB_ID"), fields)
    if fields.get("JobName") != "geodml-gemma-si-replay" or fields.get("TimeLimit") != "01:00:00":
        raise ValueError("use only the already approved Gemma allocation")
    exclusion_bytes = args.exclude_inputs.read_bytes()
    previous = json.loads(exclusion_bytes)
    frozen_cells(previous)
    excluded = {c["prompt_id"] for c in previous["cells"]}
    roots = {"qwen38": args.qwen.resolve(), "llama4": args.llama.resolve()}
    selection = {"seed": SEED, "per_model": 10, "datasets": {k: str(v) for k, v in roots.items()},
        "excluded_inputs_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
        "excluded_prompt_ids": sorted(excluded), "distinct_prompts": True,
        "eligibility": "verified-generation-and-trace-v1"}
    if args.output.exists():
        bundle = json.loads(args.output.read_bytes())
        if bundle.get("selection") != selection:
            raise ValueError("existing frozen selection conflicts; preserve it")
        validate_selection(bundle, excluded, args.existing_job_id)
        print(f"REUSING FROZEN INPUTS {args.output}", flush=True)
        return 0
    deadline = datetime.fromisoformat(fields["EndTime"]).timestamp() - 1200
    cells, tasks, rejected = [], {}, []
    for model, root in roots.items():
        if not (root / "contract.json").is_file():
            raise ValueError(f"dataset contract missing: {root}")
        chosen, failures = sample_refs(root, model, excluded, deadline=deadline)
        rejected.extend(failures)
        for cell in iter_cells(root, chosen):
            if time.time() >= deadline:
                raise ValueError("insufficient remaining time; no inference started")
            record, records = build_cell(cell, max_tokens=640, j1_max_tokens=64, sensitivity_fraction=0.0)
            if record.get("status") != "ok" or any("judge_task_id" not in s for s in record["sources"]):
                raise ValueError(f"selected cell is not fully assessable: {record['cell_id']}; no silent replacement")
            if record["prompt_id"] in excluded:
                raise ValueError("selected prompt overlaps earlier or newly selected cells")
            excluded.add(record["prompt_id"])
            cells.append(record)
            for task in records:
                tid = task["judge_task_id"]
                if tid in tasks and tasks[tid] != task:
                    raise ValueError("conflicting shared task")
                tasks[tid] = task
    bundle = {"protocol": PROTOCOL, "selection": selection, "cells": cells, "tasks": list(tasks.values()),
              "nemotron": {}, "boundary": boundary, "existing_job_id": args.existing_job_id,
              "rejected_candidates": rejected}
    validate_selection(bundle, set(selection["excluded_prompt_ids"]), args.existing_job_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(args.output, bundle)
    print(json.dumps({"frozen_cells": len(cells), "tasks": len(tasks), "inputs": str(args.output),
                      "seed": SEED, "new_allocation": False}), flush=True)
    return 0


def review(args):
    """Print complete request/answer/source inputs and raw outputs, joined by task ID."""
    cells = rows(args.trial / "pass1/cells.jsonl")
    tasks = indexed(rows(args.trial / "pass1/tasks.jsonl"))
    results = {p: indexed(rows(args.trial / p / "results.jsonl")) for p in ("pass1", "pass2")}
    selected = cells[args.start:args.start + args.count]
    if not selected:
        raise ValueError("no cells in the requested range")
    for cell in selected:
        sources = [{**s, "input": tasks[s["judge_task_id"]],
                    "judgments": {p: records.get(s["judge_task_id"]) for p, records in results.items()}}
                   for s in cell["sources"]]
        print(json.dumps({"cell": cell, "sources": sources, "request_and_full_answer": tasks[cell["j1_task_id"]],
            "j1": {p: records.get(cell["j1_task_id"]) for p, records in results.items()}}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("freeze")
    for name in ("qwen", "llama", "exclude-inputs", "output"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--existing-job-id", required=True)
    r = commands.add_parser("review")
    r.add_argument("--trial", type=Path, required=True)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "review" and (args.start < 0 or args.count < 1):
        parser.error("start must be nonnegative and count must be positive")
    return freeze(args) if args.command == "freeze" else review(args)


if __name__ == "__main__":
    raise SystemExit(main())
