#!/usr/bin/env python3
"""Prepare/check a finite Gemma SI-v4 development test; never allocate resources."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import closing
import copy
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.inference_budget import AllocationBudget
from analysis.scripts import horeka_gemma_si as gemma
from analysis.scripts import horeka_nemotron as stage
from analysis.scripts import run_source_importance_judge as judge
from analysis.scripts.prepare_horeka_qwen import capture_quota
from analysis.scripts.prepare_si_v4_diagnostics import freeze as freeze_diagnostics
from analysis.scripts.replay_source_importance_judge import frozen_cells

JOB_NAME = "geodml-gemma-si-v4"
SELECTION_FORMAT = "si-v4-manual-selection-v1"


def terminal_choice(prompt):
    with open("/dev/tty", "r") as reader, open("/dev/tty", "w") as writer:
        writer.write(prompt)
        writer.flush()
        return reader.readline().strip()


def saved_pool(source_run):
    """Read the small saved development pool, independent of its failed attempts."""
    path = Path(source_run).resolve() / "config.json"
    config = stage.read(path)
    pool = {}
    for filename, expected in config["source_files"].items():
        if judge.file_hash(Path(filename)) != expected:
            raise ValueError(f"saved source bundle changed: {filename}")
        for cell, tasks in frozen_cells(stage.read(Path(filename))):
            if cell["cell_id"] in pool:
                raise ValueError("duplicate cell in the saved selection pool")
            pool[cell["cell_id"]] = (cell, tasks)
    if not pool:
        raise ValueError("saved selection pool is empty")
    return pool, {"source_config": str(path), "source_config_sha256": judge.file_hash(path),
                  "source_files": config["source_files"], "population_cells": len(pool)}


def selection_bundle(pool, selection):
    ids = selection["selected_cell_ids"]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(selection["sampled_cell_ids"]):
        raise ValueError("choose distinct cells from the displayed sample")
    cells, tasks = [], {}
    for cell_id in ids:
        cell, records = pool[cell_id]
        cells.append(cell)
        for task_id, record in records.items():
            if task_id in tasks and tasks[task_id] != record:
                raise ValueError("conflicting shared source task")
            tasks[task_id] = record
    bundle = {"protocol": v3.PROTOCOL, "scientific_result": False, "selection": selection,
              "cells": cells, "tasks": [tasks[key] for key in sorted(tasks)]}
    frozen_cells(bundle)
    return bundle


def select(args):
    if args.output.exists():
        raise FileExistsError("selection already exists; preserve it and use a new selection directory")
    pool, provenance = saved_pool(args.source_run)
    if args.count < 1:
        raise ValueError("candidate count must be positive")
    sampled = random.Random(args.seed).sample(sorted(pool), min(args.count, len(pool)))
    print(f"SAVED_DEVELOPMENT_POOL={len(pool)} CANDIDATES={len(sampled)} SEED={args.seed}", flush=True)
    for number, cell_id in enumerate(sampled, 1):
        cell, tasks = pool[cell_id]
        request = tasks[cell["j1_task_id"]]
        print(json.dumps({"number": number, "cell_id": cell_id, "model": cell.get("model"),
            "prompt_id": cell.get("prompt_id"), "sources": len(cell["sources"]),
            "request": request["request"], "answer_preview": request["answer"][:300]},
            ensure_ascii=False), flush=True)
    choice = args.take
    if choice is None:
        choice = terminal_choice("Choose numbers separated by commas, or all. Empty cancels: ")
    if not choice.strip():
        raise ValueError("selection cancelled; no Gemma input or allocation was created")
    try:
        numbers = list(range(1, len(sampled) + 1)) if choice.strip().lower() == "all" else [
            int(value.strip()) for value in choice.split(",")]
    except ValueError:
        raise ValueError("enter comma-separated numbers from the displayed sample, or all") from None
    if len(numbers) != len(set(numbers)) or any(number < 1 or number > len(sampled) for number in numbers):
        raise ValueError("chosen numbers must be distinct and within the displayed sample")
    selection = {"format_version": SELECTION_FORMAT, **provenance, "seed": args.seed,
                 "sampled_cell_ids": sampled, "selected_cell_ids": [sampled[n - 1] for n in numbers],
                 "purpose": "user-selected development examples; not a confirmatory sample"}
    bundle = selection_bundle(pool, selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(judge.canonical(bundle) + "\n")
    print(json.dumps({"selected_inputs": str(args.output.resolve()), "cells": len(numbers),
                      "sha256": judge.file_hash(args.output), "inference_started": False}, indent=2))
    return 0


def verified_selection(path):
    bundle = stage.read(path)
    selection = bundle.get("selection", {})
    if selection.get("format_version") != SELECTION_FORMAT:
        raise ValueError("selected inputs require a recorded manual selection")
    pool, provenance = saved_pool(Path(selection["source_config"]).parent)
    if any(selection.get(key) != value for key, value in provenance.items()):
        raise ValueError("selection source provenance changed")
    sampled = selection["sampled_cell_ids"]
    if sampled != random.Random(selection["seed"]).sample(sorted(pool), len(sampled)):
        raise ValueError("recorded candidate sample differs from its seed")
    if bundle != selection_bundle(pool, selection):
        raise ValueError("selected cell or task content changed after selection")
    return bundle


def fresh(args):
    """Choose inputs now; prepare against a live allocation and print its exact run command."""
    workspace = args.workspace.resolve()
    reviews = workspace / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="gemma-selected-", dir=reviews))
    print("FRESH_DIRECTORY", session, flush=True)
    selected = session / "selected-inputs.json"
    try:
        select(SimpleNamespace(source_run=args.source_run, output=selected,
                               seed=args.seed, count=args.count, take=args.take))
        queue = subprocess.check_output(["squeue", "--me", "--array", "--noheader",
                                         "--format=%i|%j|%T"], text=True, timeout=30)
        jobs = []
        for line in queue.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                raise ValueError("unexpected scheduler response; selection is saved")
            if parts[1:] == [JOB_NAME, "RUNNING"]:
                jobs.append(parts[0])
        jobs = sorted(set(jobs))
        print("RUNNING_GEMMA_ALLOCATIONS", ", ".join(jobs) or "none", flush=True)
        if not jobs:
            print("SELECTION_SAVED_NO_RUNNING_ALLOCATION", selected, flush=True)
            return 2
        job = args.existing_job_id or (jobs[0] if len(jobs) == 1 else
                                      terminal_choice("Existing running Gemma job ID to use: "))
        if job not in jobs:
            raise ValueError("choose an existing running Gemma job from the displayed list")
        fields = dict(t.split("=", 1) for t in subprocess.check_output(
            ["scontrol", "show", "job", job, "-o"], text=True, timeout=30).split() if "=" in t)
        prepare(SimpleNamespace(workspace=workspace, output=session / "run", bundle=None,
            selected_inputs=selected, existing_job_id=job, account=fields["Account"],
            walltime=fields["TimeLimit"], approval=(
                "Valerian requested one SI-v4 pass over explicitly selected saved development cells "
                f"inside existing job {job}; no new allocation or extension.")))
        command = "\n".join([
            "(", "  set -euo pipefail", "  source " + shlex.quote(str(workspace / "geodml-nemotron-env.sh")),
            f'  if [ "${{SLURM_JOB_ID:-}}" != {shlex.quote(job)} ]; then',
            "    printf '%s\\n' " + shlex.quote("Use the existing compute shell for job " + job),
            "    exit 1", "  fi", "  export PYTHONDONTWRITEBYTECODE=1", "  set +e",
            "  bash " + shlex.quote(str(session / "run/run.sh")) +
                " 2>&1 | tee -a " + shlex.quote(str(session / f"console-job{job}.log")),
            '  GEMMA_EXIT=${PIPESTATUS[0]}', "  printf 'GEMMA_EXIT=%s\\n' \"$GEMMA_EXIT\"",
            '  exit "$GEMMA_EXIT"', ")", ""])
        atomic(session / "compute-command.sh", command.encode())
        print("PREPARED_SELECTED_RUN", session / "run", flush=True)
        print("PASTE_IN_EXISTING_COMPUTE_SHELL\n" + command, flush=True)
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        atomic(session / "preparation-error.json", judge.canonical({
            "error_type": type(error).__name__, "error": str(error), "inference_started": False}).encode())
        print("FRESH_PREPARATION_STOPPED", str(error), flush=True)
        print("PRESERVED_DIRECTORY", session, flush=True)
        return 2


def run_picked(args):
    config_path = args.config.resolve()
    config = stage.read(config_path)
    if config.get("workload_mode") != "selected-v4-cells" or not config.get("existing_job_id"):
        raise ValueError("run-picked requires a prepared selected-cell run bound to an existing job")
    job = str(config["existing_job_id"])
    fields = dict(t.split("=", 1) for t in subprocess.check_output(
        ["scontrol", "show", "job", job, "-o"], text=True, timeout=30).split() if "=" in t)
    stage.check_bound_allocation(config, os.environ.get("SLURM_JOB_ID"), fields)
    workspace = Path(config["workspace"])
    # One selected runner per allocation, including across separate preparations.
    # Keep the inode: closing releases this advisory lock; unlinking would race.
    with (workspace / "reviews" / f".gemma-job{job}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another selected Gemma run is active in this allocation") from None
        task_ids = {row["judge_task_id"] for row in judge.rows(config_path.parent / "v4-inputs/tasks.jsonl.gz")}
        for pattern in (f"gemma*/attempts/job{job}", f"gemma*/run/attempts/job{job}"):
            for attempt in (workspace / "reviews").glob(pattern):
                if attempt.resolve() == config_path.parent / "attempts" / f"job{job}":
                    raise ValueError(f"this selected run was already attempted; inspect {attempt}")
                result = attempt / "trial-result.json"
                if not result.is_file() or stage.read(result).get("status") not in ("completed", "failed", "deadline"):
                    raise ValueError(f"another attempt has no terminal receipt; inspect {attempt}")
                for index in attempt.glob("trial/*/control/index.sqlite"):
                    with closing(sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
                        prior = db.execute("SELECT id FROM tasks WHERE state IN ('running','saved','done','blocked')")
                        if any(task_id in task_ids for (task_id,) in prior):
                            raise ValueError(f"selected tasks already have work to reconcile; inspect {index}")
        quota = config_path.parent / "startup-quota.json"
        atomic(quota, judge.canonical(capture_quota(workspace, config["account"])).encode())
        storage = judge.check_storage(workspace, quota, config_path.parent)
        if not storage["safe_to_admit"] or not storage["quota_verified"]:
            raise ValueError("fresh storage/quota evidence does not permit startup")
        return stage.execute(config_path)


def freeze_saved(bundles, output, protocol):
    """Bridge saved SI-v3 inputs without re-recovering answers or fetching sources."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tasks, cells, seen = {}, [], set()
    for bundle in bundles:
        for old_cell, old_tasks in frozen_cells(bundle):
            cell = copy.deepcopy(old_cell)
            if cell["cell_id"] in seen:
                raise ValueError("duplicate development cell")
            seen.add(cell["cell_id"])
            j1 = old_tasks[cell["j1_task_id"]]
            masked, spans = v3.mask_answer_citations(j1["answer"], cell["presented"])
            cell.pop("stored_answer_sensitivity", None)
            cell.update(protocol=protocol, judged_answer=j1["answer"],
                        judged_answer_sha256=hashlib.sha256(j1["answer"].encode()).hexdigest(),
                        masked_answer_sha256=hashlib.sha256(masked.encode()).hexdigest(),
                        answer_mask_spans=spans,
                        provenance={"status": "provenance_unresolved", "reasons": ["saved_judge_inputs_only"],
                                    "attribution_target": "supplied_record_not_verified_webpage"})
            tasks[j1["judge_task_id"]] = j1
            if protocol == v4.PROTOCOL:
                mapper = v4.prepare_map_task(request=j1["request"], answer=masked)["record"]
                tasks[mapper["judge_task_id"]] = mapper
                cell["map_task_id"] = mapper["judge_task_id"]
            for source in cell["sources"]:
                old = old_tasks[source.pop("judge_task_id")]
                if old["request"] != j1["request"] or old["masked_answer"] != masked:
                    raise ValueError("saved source and J1 inputs disagree")
                # Older SI-v3 cell records omitted this hash. Their validated
                # task records still contain the exact judge-visible source.
                source_hash = judge._digest({"title": old["source_title"], "text": old["source_text"]})
                if "source_sha256" in source and source["source_sha256"] != source_hash:
                    raise ValueError("saved source hash does not match its frozen task")
                source["source_sha256"] = source_hash
                if protocol == v4.PROTOCOL:
                    new = v4.source_dependency(mapper["judge_task_id"], old["source_title"], old["source_text"])
                    source.update(dependency_id=new["judge_task_id"], status="awaiting_map")
                else:
                    new = v3.task_record(v3._source_item(request=old["request"], masked_answer=masked,
                        title=old["source_title"], text=old["source_text"], max_tokens=4096))
                    source.update(judge_task_id=new["judge_task_id"])
                tasks[new["judge_task_id"]] = new
            cells.append(cell)
    for name, values in (("cells", cells), ("tasks", tasks.values())):
        with gzip.open(output / f"{name}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in values:
                stream.write(judge.canonical(row) + "\n")
    counts = dict(Counter(t["task"] for t in tasks.values()))
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": protocol,
                "max_tokens": 4096, "map_max_tokens": 4096, "preprocessing": v4.PREPROCESSING_VERSION,
                "cells": len(cells), "unique_tasks": counts, "scientific_result": False,
                "files": {name: judge.file_hash(output / name) for name in ("cells.jsonl.gz", "tasks.jsonl.gz")}}
    atomic(output / "manifest.json", judge.canonical(manifest).encode())
    return manifest


def schema_check(inputs):
    import xgrammar
    for record in judge.rows(inputs / "tasks.jsonl.gz"):
        if record["task"] == "source_dependency":
            continue
        item = v4.item_from_record(record) if record["protocol"] == v4.PROTOCOL else v3.item_from_record(record)
        xgrammar.Grammar.from_json_schema(json.dumps(item["schema"]))
    # Stage B enums depend on generated maps, but the JSON grammar is fixed.
    xgrammar.Grammar.from_json_schema(json.dumps(v4.source_schema(["a1w1"], ["c1"], ["text1"])))


def prepare(args):
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("clean committed checkout required")
    if not args.approval.strip():
        raise ValueError("explicit wall-time approval required")
    walltime = getattr(args, "walltime", "01:00:00")
    if walltime not in ("01:00:00", "03:00:00"):
        raise ValueError("supported development wall-times are one or three hours")
    hours = int(walltime.split(":")[0])
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("preparation exists; preserve it and use its saved run.sh")
    selected_path = getattr(args, "selected_inputs", None)
    bound = getattr(args, "existing_job_id", None)
    selected = verified_selection(selected_path) if selected_path else None
    allocation = None
    if selected is not None:
        if not bound:
            raise ValueError("selected-cell preparation needs an existing Gemma allocation ID")
        allocation = dict(t.split("=", 1) for t in subprocess.check_output(
            ["scontrol", "show", "job", str(bound), "-o"], text=True, timeout=30).split() if "=" in t)
        stage.check_bound_allocation({"existing_job_id": str(bound), "minimum_remaining_seconds": 1200},
                                     str(bound), allocation)
        if (allocation.get("JobName") != JOB_NAME or allocation.get("TimeLimit") != walltime
                or allocation.get("Account") != args.account
                or not allocation.get("UserId", "").endswith(f"({os.getuid()})")):
            raise ValueError("existing allocation name, owner, account or wall-time differs")
    elif bound:
        raise ValueError("existing-job selection mode requires --selected-inputs")
    verified = stage.read(gemma.prep(workspace) / "models-verified.json")
    if verified.get("status") != "verified" or [(m["repo_id"], m["revision"]) for m in verified["models"]] != list(gemma.MODELS):
        raise ValueError("verified pinned Gemma receipt required; no automatic download")
    snapshot = Path(verified["models"][0]["snapshot"])
    if not snapshot.is_dir():
        raise ValueError("verified snapshot missing")
    versions = gemma.runtime_check()
    config = stage.read(repo / "analysis/config/si_v4_gemma.template.json")
    expected = {**config["runtime_versions"], "vllm": config["serving_version"].removeprefix("vllm-")}
    if any(versions[k] != v for k, v in expected.items()):
        raise ValueError(f"runtime changed: {versions}; review before freezing")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != config["chat_template_sha256"]:
        raise ValueError("local pinned chat template differs; review before allocating")
    out.mkdir(parents=True)
    bundle_paths = [selected_path] if selected is not None else args.bundle
    bundles = [selected] if selected is not None else [stage.read(p) for p in bundle_paths]
    protocols = (("v4-inputs", v4.PROTOCOL),) if selected is not None else (
        ("v3-inputs", v3.PROTOCOL), ("v4-inputs", v4.PROTOCOL))
    inventories = {name: freeze_saved(bundles, out / name, protocol) for name, protocol in
                   protocols}
    if selected is None and inventories["v4-inputs"]["cells"] != 40:
        raise ValueError("this development test expects the forty previously reviewed cells")
    if selected is None:
        inventories["constructed-inputs"] = freeze_diagnostics(out / "constructed-inputs", "development")
    for name in inventories:
        schema_check(out / name)
        if selected is not None:
            with tempfile.TemporaryDirectory(prefix="input-check-", dir=out) as temporary:
                coordinator = judge.Coordinator(out / name, Path(temporary) / "results", {"gpu_count": 4})
                coordinator.close()
    config["tokenizer_path"] = str(snapshot)
    judge.load_config(repo / "analysis/config/si_v4_gemma.template.json")
    atomic(out / "judge-config.json", judge.canonical(config).encode())
    run_config = {"repository": str(repo), "git_commit": pin, "workspace": str(workspace),
        "model_id": gemma.MODEL, "model_revision": gemma.REVISION, "serving": stage.SERVING,
        "trial": "si-v4-development", "judge_protocol": v4.PROTOCOL, "cache_prefix": "gemma4-si-v4",
        "development_config": str(out / "config.json"), "job_name": JOB_NAME,
        "walltime": walltime, "approval": args.approval, "account": args.account,
        "runtime": versions, "scientific_result": False, "inventories": inventories,
        "source_files": {str(p.resolve()): judge.file_hash(p) for p in bundle_paths},
        "judge_config_sha256": judge.file_hash(out / "judge-config.json"),
        "estimate": {"minutes": [20, 50], "nodes": 1, "gpus": 4, "cpus_requested": 32,
                     "memory": "whole A100 node", "node_hours_max": hours, "gpu_hours_max": 4 * hours,
                     "basis": "historical Gemma wrapper 8.9-10.9 minutes; v4 throughput unmeasured"}}
    if selected is not None:
        run_config.update(workload_mode="selected-v4-cells", selection=selected["selection"],
                          existing_job_id=str(bound), minimum_remaining_seconds=1200,
                          cache_prefix="gemma4-si-selected-" + hashlib.sha256(str(out).encode()).hexdigest()[:12],
                          allocation_at_preparation={key: allocation.get(key) for key in
                              ("JobId", "JobState", "Partition", "NodeList", "TimeLimit", "EndTime")})
        run_config["estimate"].update(new_allocation=False, additional_allocation_hours=0,
                                     scope="one SI-v4 pass over explicitly selected development cells")
    atomic(out / "config.json", judge.canonical(run_config).encode())
    launcher = gemma.run_script(repo, out)
    if selected is not None:
        command = [sys.executable, str(repo / "analysis/scripts/horeka_si_v4.py"),
                   "run-picked", "--config", str(out / "config.json")]
        launcher = ("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n").encode()
    atomic(out / "run.sh", launcher)
    print(json.dumps({"prepared": str(out), "inventories": inventories, "allocation_submitted": False}, indent=2))
    return 0


def admission(snapshot, now, *, existing_queue_exception=False):
    from analysis.scripts.capture_agentic_scheduler_snapshot import ACTIVE_STATES
    if snapshot.get("complete") is not True or not 0 <= now - snapshot["captured_at_epoch"] <= 120:
        raise ValueError("fresh complete scheduler evidence required")
    live = snapshot["jobs"]
    if not existing_queue_exception and sum(r["state"] in ACTIVE_STATES for r in live) >= 5:
        raise ValueError("five allocations already active; wait on the login host")
    if not existing_queue_exception and any(r["state"] == "PENDING" for r in live):
        raise ValueError("pending allocations prevent a reliable start-gap check; leave them unchanged")
    if any(r["job_name"] == JOB_NAME for r in live):
        raise ValueError("SI-v4 allocation already exists; do not request another")
    starts = [r["start_epoch"] for r in live + snapshot.get("owners", []) if isinstance(r.get("start_epoch"), int)]
    wait = 600 - (now - max(starts, default=0))
    if wait > 0:
        raise ValueError(f"wait at least {wait} seconds on login host, then repeat admission")


def check(args):
    from analysis.scripts.capture_agentic_scheduler_snapshot import capture
    out = args.output.resolve()
    config = stage.read(out / "config.json")
    if config["job_name"] != JOB_NAME or config["walltime"] not in ("01:00:00", "03:00:00") or not config["approval"]:
        raise ValueError("not an approved SI-v4 preparation")
    exception = getattr(args, "approved_existing_queue_exception", False)
    legacy_scope = (
        out == Path(config["workspace"]).resolve() / "reviews/gemma-si-v4-development-20261001"
        and config.get("walltime") == "01:00:00"
        and config.get("git_commit") == "9aa5d90e0b310f3eea587a60787727787ab92757"
    )
    three_hour_scope = (
        out == Path(config["workspace"]).resolve() / "reviews/gemma-si-v4-development-20261002-3h"
        and config.get("walltime") == "03:00:00"
    )
    if exception and (
        not (legacy_scope or three_hour_scope)
        or config.get("model_id") != gemma.MODEL or config.get("model_revision") != gemma.REVISION
        or config.get("trial") != "si-v4-development"
    ):
        raise ValueError("queue exception applies only to the approved dated Gemma SI-v4 preparations")
    if (out / "ALLOCATION_ATTEMPTED").exists() or any((out / "attempts").glob("job*")):
        raise ValueError("allocation already attempted; inspect existing work before any new allocation")
    live = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%i"], text=True)
    since = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    history = subprocess.check_output(["sacct", "-X", "--noheader", "--parsable2", "--starttime=" + since,
                                       "--format=JobIDRaw"], text=True)
    ids = [s.strip().split("|")[0] for s in (live + "\n" + history).splitlines() if s.strip()]
    snapshot = capture(plan={"plan_id": "si-v4-development"}, since=since, include_job_ids=ids)
    admission(snapshot, int(time.time()), existing_queue_exception=exception)
    queue = subprocess.check_output(["squeue", "--account=" + config["account"], "--array", "--noheader", "--format=%i"], text=True)
    if len(set(queue.split())) >= 295:
        raise ValueError("account queue full; wait without changing jobs")
    subprocess.run(["sinfo", "--partition=accelerated", "--format=%P %a %l %G"], check=True)
    quota = out / "quota.json"
    atomic(quota, judge.canonical(capture_quota(Path(config["workspace"]), config["account"])).encode())
    health = judge.check_storage(config["workspace"], quota)
    if not health["safe_to_admit"] or not health["quota_verified"]:
        raise ValueError("storage admission failed")
    atomic(out / "admission.json", judge.canonical({"scheduler": snapshot, "storage": health,
                                                   "config_sha256": judge.file_hash(out / "config.json"),
                                                   "admission_helper_sha256": judge.file_hash(Path(__file__)),
                                                   "approved_exception": (
                                                       f"Valerian approved one {config['walltime']} SI-v4 allocation alongside the existing queue; "
                                                       "waive five-active and no-pending guards only for this prepared run. "
                                                       "Keep quota, account queue limit and ten-minute observed-start spacing. "
                                                       "Do not cancel or modify other jobs."
                                                       if exception else None),
                                                   "checked_at_epoch": int(time.time())}).encode())
    print("CHECK PASSED. Run the separately approved salloc once, promptly; recheck if delayed.")
    return 0


async def review(args):
    config = stage.read(args.config)
    root = args.config.parent
    if args.server_model_name != config["model_id"]:
        raise ValueError("serving model differs from frozen configuration")
    if judge.file_hash(root / "judge-config.json") != config["judge_config_sha256"]:
        raise ValueError("judge configuration changed")
    for name, manifest in config["inventories"].items():
        if stage.read(root / name / "manifest.json") != manifest:
            raise ValueError("input manifest changed")
    mode = config.get("workload_mode", "legacy-development")
    if mode not in ("legacy-development", "selected-v4-cells"):
        raise ValueError("unknown SI-v4 workload mode")
    # Preserve the legacy comparison; explicitly selected examples get one v4 pass.
    queue = [("v4-pass1", "v4-inputs")] if mode == "selected-v4-cells" else [
        ("v4-pass1", "v4-inputs"), ("v3-bridge", "v3-inputs"),
        ("constructed", "constructed-inputs"), ("v4-pass2", "v4-inputs"), ("v4-pass3", "v4-inputs")]
    args.output.mkdir(parents=True, exist_ok=False)
    quota = args.output / "quota.json"
    stop = asyncio.Event()

    async def refresh():
        while not stop.is_set():
            try:
                value = await asyncio.to_thread(capture_quota, Path(config["workspace"]), config["account"])
                atomic(quota, judge.canonical(value).encode())
            except Exception as exc:
                atomic(args.output / "quota-error.json", judge.canonical({"error": str(exc), "time": time.time()}).encode())
                # Invalidate prior evidence immediately; admissions fail closed.
                atomic(quota, b"{}")
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=120)
            except asyncio.TimeoutError:
                pass

    atomic(quota, judge.canonical(await asyncio.to_thread(capture_quota, Path(config["workspace"]), config["account"])).encode())
    refresher = asyncio.create_task(refresh())
    outcomes = []
    try:
        for name, inputs in queue:
            if not AllocationBudget.from_environment(require=True).can_start():
                break
            settings = stage.read(root / "judge-config.json")
            settings["repetition_id"] = name
            settings_path = args.output / f"{name}.json"
            atomic(settings_path, judge.canonical(settings).encode())
            code = await judge.run(SimpleNamespace(config=settings_path, inputs=root / inputs,
                output=args.output / name, workspace=Path(config["workspace"]), quota_evidence=quota,
                base_url=args.base_url, recovery_evidence=None, fixed_maps=None))
            outcomes.append({"run": name, "exit_code": code})
            print("V4_RUN " + json.dumps(outcomes[-1]), flush=True)
            reports = sorted((args.output / name / "reports").glob("*/summary.json"))
            if reports and stage.read(reports[-1])["status"] == "incomplete":
                break
    finally:
        stop.set()
        await refresher
        atomic(args.output / "summary.json", judge.canonical({"runs": outcomes, "planned_runs": len(queue),
               "scientific_result": False, "complete": len(outcomes) == len(queue),
               "status": "finished" if len(outcomes) == len(queue) and all(r["exit_code"] == 0 for r in outcomes)
                         else "partial_or_failed"}).encode())
    return 0 if len(outcomes) == len(queue) and all(r["exit_code"] == 0 for r in outcomes) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("select", help="sample saved development cells, then choose which to judge")
    s.add_argument("--source-run", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--seed", type=int, default=20261002)
    s.add_argument("--count", type=int, default=20, help="maximum candidate cells to display")
    s.add_argument("--take", help="displayed numbers separated by commas, or all; otherwise ask in the terminal")
    f = commands.add_parser("fresh", help="choose cells and prepare a new run inside an existing allocation")
    f.add_argument("--workspace", type=Path, required=True)
    f.add_argument("--source-run", type=Path, required=True)
    f.add_argument("--seed", type=int, default=20261002)
    f.add_argument("--count", type=int, default=20)
    f.add_argument("--take", help="displayed numbers or all; otherwise ask in the terminal")
    f.add_argument("--existing-job-id", help="choose this job from the user's running Gemma allocations")
    e = commands.add_parser("run-picked", help="check existing ownership and storage, then run selected cells")
    e.add_argument("--config", type=Path, required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path, action="append")
    inputs.add_argument("--selected-inputs", type=Path, help="verified output of the select command")
    p.add_argument("--existing-job-id", help="bind selected-cell execution to an already running Gemma allocation")
    p.add_argument("--account", required=True)
    p.add_argument("--approval", required=True)
    p.add_argument("--walltime", choices=("01:00:00", "03:00:00"), default="01:00:00")
    c = commands.add_parser("check")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--approved-existing-queue-exception", action="store_true",
                   help="record Valerian's scoped exception for the approved dated one- or three-hour run")
    r = commands.add_parser("review")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--server-model-name", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(review(args)) if args.command == "review" else {
        "select": select, "fresh": fresh, "run-picked": run_picked,
        "prepare": prepare, "check": check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
