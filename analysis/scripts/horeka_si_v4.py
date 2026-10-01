#!/usr/bin/env python3
"""Prepare/check a finite Gemma SI-v4 development test; never allocate resources."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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
    bundles = [stage.read(p) for p in args.bundle]
    inventories = {name: freeze_saved(bundles, out / name, protocol) for name, protocol in
                   (("v3-inputs", v3.PROTOCOL), ("v4-inputs", v4.PROTOCOL))}
    if inventories["v4-inputs"]["cells"] != 40:
        raise ValueError("this development test expects the forty previously reviewed cells")
    inventories["constructed-inputs"] = freeze_diagnostics(out / "constructed-inputs", "development")
    for name in inventories:
        schema_check(out / name)
    config["tokenizer_path"] = str(snapshot)
    judge.load_config(repo / "analysis/config/si_v4_gemma.template.json")
    atomic(out / "judge-config.json", judge.canonical(config).encode())
    run_config = {"repository": str(repo), "git_commit": pin, "workspace": str(workspace),
        "model_id": gemma.MODEL, "model_revision": gemma.REVISION, "serving": stage.SERVING,
        "trial": "si-v4-development", "judge_protocol": v4.PROTOCOL, "cache_prefix": "gemma4-si-v4",
        "development_config": str(out / "config.json"), "job_name": JOB_NAME,
        "walltime": walltime, "approval": args.approval, "account": args.account,
        "runtime": versions, "scientific_result": False, "inventories": inventories,
        "source_files": {str(p.resolve()): judge.file_hash(p) for p in args.bundle},
        "judge_config_sha256": judge.file_hash(out / "judge-config.json"),
        "estimate": {"minutes": [20, 50], "nodes": 1, "gpus": 4, "cpus_requested": 32,
                     "memory": "whole A100 node", "node_hours_max": hours, "gpu_hours_max": 4 * hours,
                     "basis": "historical Gemma wrapper 8.9-10.9 minutes; v4 throughput unmeasured"}}
    atomic(out / "config.json", judge.canonical(run_config).encode())
    atomic(out / "run.sh", gemma.run_script(repo, out))
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
    # Fixed, finite development queue: no fresh acceptance cases consumed here.
    queue = [("v4-pass1", "v4-inputs"), ("v3-bridge", "v3-inputs"),
             ("constructed", "constructed-inputs"), ("v4-pass2", "v4-inputs"), ("v4-pass3", "v4-inputs")]
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
    p = commands.add_parser("prepare")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, action="append", required=True)
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
    return asyncio.run(review(args)) if args.command == "review" else {"prepare": prepare, "check": check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
