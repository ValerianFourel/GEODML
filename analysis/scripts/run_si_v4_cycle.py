#!/usr/bin/env python3
"""Finite SI-v4 evaluation plans and resumable queues; uses the existing HoreKa server boundary."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.inference_budget import AllocationBudget
from analysis.scripts import run_source_importance_judge as judge

FORMAT = "si-v4-evaluation-queue-v1"
BUDGET = {"gpu_hours_total": 32, "gpu_hours_by_phase": {"development": 16, "fresh": 16},
          "gpus": 4, "cpus": 32, "memory": "node default", "maximum_segment_seconds": 3600,
          "maximum_segments_per_phase": 4}


def read(path):
    return json.loads(Path(path).read_text())


def latest_report(output):
    pointer = Path(output) / "reports/latest.json"
    if not pointer.exists():
        return None
    name = read(pointer)["directory"]
    if not isinstance(name, str) or not re.fullmatch(r"si-[a-f0-9]+", name):
        raise ValueError("invalid report pointer")
    report = Path(output) / "reports" / name
    if not (report / "summary.json").is_file():
        raise ValueError("report pointer has no summary")
    return report


def plan(args):
    """Freeze all declared inputs; selection is independent of candidate results."""
    from analysis.scripts.prepare_si_v4_evaluation import load_freeze, select_repeats
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest, cells, _ = load_freeze(args.inputs)
    expected = 40 if args.phase == "development" else 120
    if len(cells) != expected:
        raise ValueError(f"{args.phase} needs exactly {expected} cells")
    if manifest["protocol"] != "agentic-source-importance-v4":
        raise ValueError("candidate requires v4 frozen inputs")
    if args.phase == "fresh" and not args.development_report:
        raise ValueError("fresh evaluation needs a completed passing development report")
    selection_path = getattr(args, "selection", None)
    if args.phase == "fresh" and not selection_path:
        raise ValueError("fresh evaluation requires its paired selection receipt")
    if args.development_report:
        report = read(args.development_report)
        if report.get("phase") != "development" or report.get("development_gate_passed") is not True:
            raise ValueError("development quality gate has not passed")
    # Create only after cohort/gate checks; partial preparation is retained on failure.
    out.mkdir(parents=True)
    inventory = {}
    def install(label, source):
        original, _, _ = load_freeze(source)
        shutil.copytree(source, out / label)
        inventory[label] = original
        return label
    main = install("candidate-inputs", args.inputs)
    controls = install("constructed-inputs", args.constructed)
    if inventory[controls].get("split") != args.phase.replace("fresh", "heldout"):
        raise ValueError("constructed split differs from phase")
    bridge = install("cost-inputs", args.cost_inputs)
    if inventory[bridge]["protocol"] != "agentic-source-importance-v3":
        raise ValueError("matched cost reference requires v3 bridge inputs")
    # A bridge with different original answers or evidence is not comparable.
    _, bridge_cells, _ = load_freeze(out / bridge)
    left, right = ({c["cell_id"]: c for c in values} for values in (cells, bridge_cells))
    if left.keys() != right.keys():
        raise ValueError("cost reference cohort differs")
    from analysis.scripts.compare_source_importance_runs import comparable
    for cid in left:
        comparable(left[cid], right[cid])
    repeat = select_repeats(out / main, out / "repeat-selection")
    repeat_inputs = "repeat-selection/inputs"
    inventory[repeat_inputs] = read(out / repeat_inputs / "manifest.json")
    queue = [{"name": "candidate-e2e-1", "inputs": main},
             {"name": "constructed", "inputs": controls}]
    if args.order_inputs:
        queue.append({"name": "list-order", "inputs": install("order-inputs", args.order_inputs)})
    if args.role_inputs or args.role_maps:
        if not (args.role_inputs and args.role_maps):
            raise ValueError("role controls need both inputs and fixed maps")
        install("role-inputs", args.role_inputs)
        shutil.copyfile(args.role_maps, out / "role-fixed-maps.jsonl")
        queue.append({"name": "role-controls", "inputs": "role-inputs", "fixed_maps": "role-fixed-maps.jsonl",
                      "fixed_maps_sha256": judge.file_hash(out / "role-fixed-maps.jsonl")})
    if args.phase == "development" and not (args.order_inputs and args.role_inputs):
        raise ValueError("development requires order and role controls")
    queue += [{"name": "cost-reference", "inputs": bridge}]
    queue += [{"name": f"candidate-e2e-{i}", "inputs": repeat_inputs} for i in (2, 3)]
    queue += [{"name": f"candidate-fixed-{i}", "inputs": repeat_inputs,
               "fixed_map_from": "candidate-e2e-1"} for i in (1, 2, 3)]
    value = {"format_version": FORMAT, "phase": args.phase, "budget": BUDGET,
             "inventories": inventory, "queue": queue,
             "repeat_cell_ids": repeat["selected_cell_ids"],
             "repeat_selection": repeat,
             "fresh_selection": read(selection_path) if selection_path else None,
             "development_report": read(args.development_report) if args.development_report else None,
             "development_report_sha256": judge.file_hash(args.development_report) if args.development_report else None,
             "scientific_result": False}
    atomic(out / "evaluation-plan.json", judge.canonical(value).encode())
    validate_plan(out / "evaluation-plan.json")
    print(json.dumps({"plan": str(out / "evaluation-plan.json"), "queue": queue, "budget": BUDGET}, indent=2))
    return 0


def prepare_development(args):
    """Prepare the known development cohort using checksummed saved bundles."""
    from analysis.scripts import horeka_si_v4 as pilot
    from analysis.scripts import prepare_si_v4_diagnostics as diagnostics
    from analysis.scripts.prepare_si_v4_evaluation import prepare_packets
    from analysis.interpretability.pipeline import source_importance_v4 as v4, source_importance as v3
    _, provenance = pilot.saved_pool(args.source_run)
    bundles = [read(p) for p in provenance["source_files"]]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate, bridge = out / "candidate-inputs", out / "cost-inputs"
    pilot.freeze_saved(bundles, candidate, v4.PROTOCOL)
    pilot.freeze_saved(bundles, bridge, v3.PROTOCOL)
    diagnostics.freeze(out / "constructed-inputs", "development", task_version=v4.TASK_VERSION)
    diagnostics.freeze_supplements(out / "supplements", candidate, args.list_cell_id, read(args.list_spec))
    atomic(out / "source-provenance.json", judge.canonical(provenance).encode())
    prepare_packets(candidate, out / "reference-packets/grade")
    prepare_packets(out / "supplements/role-inputs", out / "reference-packets/role-map", phase="map",
                    candidate=out / "supplements/role-fixed-maps.jsonl")
    return plan(SimpleNamespace(output=out / "queue", inputs=candidate, cost_inputs=bridge,
        constructed=out / "constructed-inputs", phase="development", development_report=None,
        order_inputs=out / "supplements/order-inputs", role_inputs=out / "supplements/role-inputs",
        role_maps=out / "supplements/role-fixed-maps.jsonl"))


def validate_plan(path):
    path = Path(path)
    value = read(path)
    if value.get("format_version") != FORMAT or value.get("phase") not in ("development", "fresh") or value.get("budget") != BUDGET:
        raise ValueError("unrecognized evaluation plan or budget")
    seen = set()
    for row in value["queue"]:
        name = row["name"]
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) or name in seen:
            raise ValueError("duplicate or unsafe evaluation run name")
        if row.get("fixed_map_from") and row["fixed_map_from"] not in seen:
            raise ValueError("fixed map must come from an earlier run")
        seen.add(name)
        if row["inputs"] not in value["inventories"]:
            raise ValueError("queue references missing input inventory")
        if row.get("fixed_maps"):
            fixed = path.parent / row["fixed_maps"]
            if not fixed.resolve().is_relative_to(path.parent.resolve()) or judge.file_hash(fixed) != row["fixed_maps_sha256"]:
                raise ValueError("fixed control map changed")
    if not seen or len(seen) > 12:
        raise ValueError("evaluation queue must be finite")
    for name, manifest in value["inventories"].items():
        folder = path.parent / name
        if not folder.resolve().is_relative_to(path.parent.resolve()) or read(folder / "manifest.json") != manifest:
            raise ValueError("evaluation input manifest changed")
        for filename, digest in manifest["files"].items():
            if not (folder / filename).resolve().is_relative_to(folder.resolve()) or judge.file_hash(folder / filename) != digest:
                raise ValueError("evaluation input file changed")
        if manifest["protocol"] == "agentic-source-importance-v4":
            from analysis.interpretability.pipeline import source_importance_v4 as v4
            if any(task.get("task_version") != v4.TASK_VERSION for task in judge.rows(folder / "tasks.jsonl.gz")
                   if task["task"] == "answer_map"):
                raise ValueError("revised evaluation requires r2 map tasks")
    return value


def fixed_subset(report, inputs, output):
    wanted = {t["judge_task_id"] for t in judge.rows(Path(inputs) / "tasks.jsonl.gz") if t["task"] == "answer_map"}
    selected = [r for r in judge.rows(Path(report) / "maps.jsonl") if r["judge_task_id"] in wanted]
    if {r["judge_task_id"] for r in selected} != wanted or len(selected) != len(wanted):
        raise ValueError("first execution has not recorded every repeated map; no regeneration permitted")
    content = "".join(judge.canonical(r) + "\n" for r in selected).encode()
    if Path(output).exists() and Path(output).read_bytes() != content:
        raise ValueError("frozen repeat maps changed")
    if not Path(output).exists():
        atomic(Path(output), content)
    return Path(output)


async def review(args, config):
    """Resume persistent run outputs across finite allocations; retain attempt receipts."""
    from analysis.scripts.horeka_si_v4 import capture_quota
    root = args.config.parent
    plan_path = root / "evaluation-plan.json"
    if judge.file_hash(plan_path) != config["evaluation_plan_sha256"]:
        raise ValueError("evaluation plan changed")
    spec = validate_plan(plan_path)
    results = root / "evaluation-results"
    results.mkdir(exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=False)
    quota = args.output / "quota.json"
    stop = asyncio.Event()
    outcomes = []
    with (results / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        async def refresh():
            while not stop.is_set():
                try:
                    value = await asyncio.to_thread(capture_quota, Path(config["workspace"]), config["account"])
                    atomic(quota, judge.canonical(value).encode())
                except Exception as exc:
                    atomic(quota, b"{}")
                    atomic(args.output / "quota-error.json", judge.canonical({"error": str(exc)}).encode())
                    return
                try:
                    await asyncio.wait_for(stop.wait(), timeout=120)
                except asyncio.TimeoutError:
                    pass
        atomic(quota, judge.canonical(await asyncio.to_thread(capture_quota, Path(config["workspace"]), config["account"])).encode())
        refresher = asyncio.create_task(refresh())
        try:
            for row in spec["queue"]:
                name, inputs = row["name"], root / row["inputs"]
                saved = latest_report(results / name)
                if saved and read(saved / "summary.json")["status"] != "incomplete":
                    settings = read(root / "judge-config.json")
                    settings.update(repetition_id=spec["phase"] + "-" + name, code_revision=config["git_commit"])
                    if read(saved / "summary.json")["configuration"] != settings:
                        raise ValueError("completed execution configuration changed")
                    fixed = root / row["fixed_maps"] if row.get("fixed_maps") else (
                        results / (name + "-maps.jsonl") if row.get("fixed_map_from") else None)
                    verified = judge.Coordinator(inputs, results / name, settings, fixed_maps=fixed)
                    try:
                        if verified.db.execute("SELECT 1 FROM tasks WHERE state IN ('pending','running','waiting','saved') LIMIT 1").fetchone():
                            raise ValueError("completed report disagrees with saved task states")
                    finally:
                        verified.close()
                    outcomes.append({"run": name, "report": str(saved), "reused": True})
                    continue
                if not AllocationBudget.from_environment(require=True).can_start():
                    break
                fixed = root / row["fixed_maps"] if row.get("fixed_maps") else None
                if row.get("fixed_map_from"):
                    first = latest_report(results / row["fixed_map_from"])
                    if first is None or read(first / "summary.json")["status"] == "incomplete":
                        break
                    fixed = fixed_subset(first, inputs, results / (name + "-maps.jsonl"))
                settings = read(root / "judge-config.json")
                settings["repetition_id"] = spec["phase"] + "-" + name
                settings_path = results / (name + ".json")
                content = judge.canonical(settings).encode()
                if settings_path.exists() and settings_path.read_bytes() != content:
                    raise ValueError("resumption configuration changed")
                atomic(settings_path, content)
                # Recovery evidence is created only from actual terminal-owner accounting by reconcile().
                recovery = results / name / "recovery-evidence.json"
                code = await judge.run(SimpleNamespace(config=settings_path, inputs=inputs, output=results / name,
                    workspace=Path(config["workspace"]), quota_evidence=quota, base_url=args.base_url,
                    recovery_evidence=recovery if recovery.exists() else None, fixed_maps=fixed))
                saved = latest_report(results / name)
                outcomes.append({"run": name, "exit_code": code, "report": str(saved) if saved else None, "reused": False})
                print("V4_RUN " + json.dumps(outcomes[-1]), flush=True)
                if saved is None or read(saved / "summary.json")["status"] == "incomplete":
                    break
        finally:
            stop.set()
            await refresher
            complete = len(outcomes) == len(spec["queue"]) and all(
                row["report"] and read(Path(row["report"]) / "summary.json").get("status")
                in ("finished", "finished_with_failures") for row in outcomes)
            summary = {"runs": outcomes, "planned_runs": len(spec["queue"]), "complete": complete,
                       "status": "finished" if complete else "incomplete", "scientific_result": False,
                       "semantic_acceptance": "not_established"}
            atomic(args.output / "summary.json", judge.canonical(summary).encode())
            atomic(results / "queue-summary.json", judge.canonical(summary).encode())
    return 0 if complete else 2


def reconcile(root):
    """Recover only writers whose specific Slurm allocations are demonstrably terminal."""
    import sqlite3
    from contextlib import closing
    from analysis.scripts.capture_agentic_scheduler_snapshot import TERMINAL_STATES
    live = set(subprocess.check_output(["squeue", "--me", "--array", "--noheader", "--format=%i"], text=True, timeout=30).split())
    for database in Path(root).glob("evaluation-results/*/control/index.sqlite"):
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            owners = db.execute("SELECT DISTINCT writer_id,job_id FROM tasks WHERE state IN ('running','saved')").fetchall()
        evidence = []
        for writer, job in owners:
            if not job or job in live or not str(job).isdecimal():
                raise ValueError("previous writer has a live or unknown allocation; preserve it")
            raw = subprocess.check_output(["sacct", "-X", "-j", str(job), "--noheader", "--parsable2", "--format=JobIDRaw,State"], text=True, timeout=30)
            states = [s.split("|")[1].strip().split()[0].rstrip("+") for s in raw.splitlines() if s.split("|")[0].strip() == str(job)]
            if len(states) != 1 or states[0] not in TERMINAL_STATES:
                raise ValueError("previous allocation is not confirmed terminal")
            evidence.append({"writer_id": writer, "slurm_job_id": job, "owner_terminal": True,
                             "evidence": raw, "captured_at_epoch": time.time()})
        if evidence:
            atomic(database.parents[1] / "recovery-evidence.json", judge.canonical(evidence).encode())


@contextmanager
def budget_lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        value = read(path) if path.exists() else {"budget": BUDGET, "phases": {}, "submissions": []}
        if value.get("budget") != BUDGET:
            raise ValueError("cycle budget changed")
        yield value


def register_phase(config_path):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    path = Path(config["cycle_budget_path"])
    phase = config["evaluation_phase"]
    if phase not in BUDGET["gpu_hours_by_phase"]:
        raise ValueError("unknown evaluation phase")
    registration = {"config": str(config_path), "sha256": judge.file_hash(config_path)}
    with budget_lock(path) as budget:
        if phase in budget["phases"] and budget["phases"][phase] != registration:
            raise ValueError("this cycle already has a different candidate for the phase")
        if phase == "fresh":
            prior = budget["phases"].get("development")
            if not prior or judge.file_hash(prior["config"]) != prior["sha256"]:
                raise ValueError("fresh evaluation requires the original registered development configuration")
            development = read(prior["config"])
            if development["git_commit"] != config["git_commit"] or development["judge_config_sha256"] != config["judge_config_sha256"]:
                raise ValueError("candidate code/model settings changed after development; revise and evaluate development first")
        budget["phases"][phase] = registration
        atomic(path, judge.canonical(budget).encode())


def checked_config(config_path, budget):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    phase = config.get("evaluation_phase")
    if (config.get("workload_mode") != "evaluation-v4" or config.get("budget") != BUDGET
            or budget["phases"].get(phase) != {"config": str(config_path), "sha256": judge.file_hash(config_path)}):
        raise ValueError("evaluation configuration differs from its registered budget")
    if config["walltime"] not in ("00:30:00", "00:45:00", "01:00:00"):
        raise ValueError("invalid evaluation walltime")
    if judge.file_hash(config_path.parent / "judge-config.json") != config["judge_config_sha256"]:
        raise ValueError("judge configuration changed")
    if judge.file_hash(config_path.parent / "evaluation-plan.json") != config["evaluation_plan_sha256"]:
        raise ValueError("evaluation plan changed")
    validate_plan(config_path.parent / "evaluation-plan.json")
    return config


def account_existing_job(job, config_path, config, budget):
    """Charge a user-created terminal allocation before admitting its replacement."""
    registered = [row for row in budget["submissions"] if row.get("job_id") == job]
    if registered:
        if (len(registered) != 1 or registered[0].get("phase") != config["evaluation_phase"]
                or registered[0].get("config_sha256") != judge.file_hash(config_path)):
            raise ValueError("existing allocation is registered to a different cycle configuration")
        return
    from analysis.scripts.capture_agentic_scheduler_snapshot import TERMINAL_STATES
    if not job.isdecimal():
        raise ValueError("existing allocation ID must be numeric")
    raw = subprocess.check_output(["sacct", "-X", "-j", job, "--noheader", "--parsable2",
        "--format=JobIDRaw,JobName%100,Account%100,User%100,State%40,TimelimitRaw,ElapsedRaw,AllocTRES%1000"],
        text=True, timeout=30)
    rows = [line.split("|") for line in raw.splitlines() if line.split("|")[0].strip() == job]
    if len(rows) != 1:
        raise ValueError("existing allocation accounting is unavailable; preserve its budget")
    _, name, account, user, state, limit, elapsed, tres, *_ = (part.strip() for part in rows[0])
    allocation = dict(part.split("=", 1) for part in tres.split(",") if "=" in part)
    minutes = sum(int(part) * factor for part, factor in zip(config["walltime"].split(":"), (60, 1, 1 / 60)))
    if (name != config["job_name"] or account != config["account"] or user != getpass.getuser()
            or state.split()[0].rstrip("+") not in TERMINAL_STATES or limit != str(int(minutes))
            or not elapsed.isdecimal() or allocation.get("gres/gpu") != "4" or allocation.get("node") != "1"):
        raise ValueError("existing allocation does not match the terminal four-GPU cycle job")
    budget["submissions"].append({"phase": config["evaluation_phase"],
        "submission_id": "existing-allocation-" + job, "job_id": job,
        "config_sha256": judge.file_hash(config_path), "walltime": config["walltime"], "gpus": 4,
        "mode": "accounted-user-created-allocation", "registered_at_epoch": time.time(),
        "accounting_evidence": raw})


def resume_estimate(config_path, config):
    """Report saved progress and measured warm time without summing overlapping requests."""
    spec = read(config_path.parent / "evaluation-plan.json")
    progress, rates, remaining = [], [], 0
    for row in spec["queue"]:
        saved = latest_report(config_path.parent / "evaluation-results" / row["name"])
        summary = read(saved / "summary.json") if saved else {}
        cells = sum(1 for _ in judge.rows(config_path.parent / row["inputs"] / "cells.jsonl.gz"))
        complete = summary.get("counts", {}).get("cells_complete", 0)
        terminal = summary.get("status") in ("finished", "finished_with_failures")
        remaining += 0 if terminal else max(0, cells - complete)
        warm = summary.get("timing", {}).get("si", {}).get("node_hours", 0) * 60
        if warm > 0 and complete > 0:
            rates.append(warm / complete)
        progress.append({"run": row["name"], "status": summary.get("status", "unstarted"),
                         "cells": cells, "counts": summary.get("counts", {}), "warm_minutes": warm})
    rates = rates or [5.87 / 20, 1.30]
    return {**config["estimate"], "saved_progress": progress,
        "remaining_cell_executions_upper_bound": remaining,
        "remaining_warm_minutes_range": [remaining * min(rates), remaining * max(rates)],
        "startup_minutes_range": [8.9, 10.9], "drain_cleanup_minutes": 5,
        "rate_basis": "saved warm timing per completed cell; historical 20-cell/retry timings if unavailable",
        "caveat": "controls, fixed-map passes and failures differ in cost; this range is provisional",
        "cheaper_alternative": "a shorter segment avoids reserved time but pays the same 9-11 minute startup; use the registered resumable walltime"}


def submit(args):
    """Submit one finite segment after fresh admission, never chain or requeue."""
    from datetime import date, timedelta
    from analysis.scripts import horeka_si_v4 as pilot
    from analysis.scripts.capture_agentic_scheduler_snapshot import capture, TERMINAL_STATES
    interactive = getattr(args, "interactive", False)
    if getattr(args, "account_existing_job", []) and not interactive:
        raise ValueError("account-existing-job requires the interactive replacement path")
    if interactive and os.environ.get("SLURM_JOB_ID"):
        raise ValueError("request the new allocation from a separate login shell; preserve the existing allocation")
    config_path = args.config.resolve()
    ledger = Path(read(config_path)["cycle_budget_path"])
    with budget_lock(ledger) as budget:
        config = checked_config(config_path, budget)
        if interactive:
            for job in getattr(args, "account_existing_job", []):
                account_existing_job(job, config_path, config, budget)
            atomic(ledger, judge.canonical(budget).encode())
        phase = config["evaluation_phase"]
        attempted = [r for r in budget["submissions"] if r["phase"] == phase]
        if len(attempted) >= BUDGET["maximum_segments_per_phase"]:
            raise ValueError("finite phase allocation budget exhausted")
        if any(not r.get("job_id") for r in budget["submissions"]):
            raise ValueError("a submission has unresolved scheduler ownership; reconcile it before another submission")
        queue_summary = config_path.parent / "evaluation-results/queue-summary.json"
        if queue_summary.exists() and read(queue_summary).get("complete") is True:
            raise ValueError("evaluation queue already exhausted; no allocation needed")
        snapshot = capture(plan={"plan_id": "si-v4-r2"}, since=str(date.today() - timedelta(days=7)),
                           include_job_ids=[r["job_id"] for r in budget["submissions"]],
                           include_all_jobs=interactive)
        pilot.admission(snapshot, int(time.time()))
        # A queued or starting segment may have no task writer yet. Its durable
        # submission still owns the cycle until accounting confirms termination.
        live = {row["job_id"] for row in snapshot["jobs"]}
        terminal = {row["job_id"] for row in snapshot.get("owners", []) if row["state"] in TERMINAL_STATES}
        if any(row["job_id"] in live or row["job_id"] not in terminal for row in budget["submissions"]):
            raise ValueError("previous cycle allocation is live or not confirmed terminal; preserve it")
        reconcile(config_path.parent)
        estimate = config["estimate"]
        if interactive:
            usage_path = config_path.parent / ("resources-before-" + uuid.uuid4().hex + ".json")
            usage = allocation_usage(budget["submissions"])
            atomic(usage_path, judge.canonical(usage).encode())
            hours = sum(int(part) * factor for part, factor in zip(config["walltime"].split(":"), (1, 1 / 60, 1 / 3600))) * 4
            if (not usage["complete"] or usage["gpu_hours_total"] + hours > BUDGET["gpu_hours_total"]
                    or usage["gpu_hours_by_phase"][phase] + hours > BUDGET["gpu_hours_by_phase"][phase]):
                raise ValueError("actual resource accounting is incomplete or the remaining cycle budget is insufficient")
            estimate = {**resume_estimate(config_path, config), "accounting": str(usage_path),
                        "gpu_hours_used_by_phase": usage["gpu_hours_by_phase"]}
        quota = config_path.parent / "submission-quota.json"
        atomic(quota, judge.canonical(pilot.capture_quota(Path(config["workspace"]), config["account"])).encode())
        health = judge.check_storage(config["workspace"], quota)
        if not health["safe_to_admit"] or not health["quota_verified"]:
            raise ValueError("storage admission failed")
        queue = subprocess.check_output(["squeue", "--account=" + config["account"], "--array", "--noheader", "--format=%i"], text=True, timeout=30)
        if len(set(queue.split())) >= 295:
            raise ValueError("account queue full")
        subprocess.run(["sinfo", "--partition=accelerated", "--format=%P %a %l %G"], check=True)
        root = Path(config["repository"])
        if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != config["git_commit"]:
            raise ValueError("execution commit differs")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
            raise ValueError("execution checkout is dirty")
        intent = {"phase": phase, "submission_id": uuid.uuid4().hex, "config_sha256": judge.file_hash(config_path),
                  "submitted_at_epoch": time.time(), "walltime": config["walltime"], "gpus": 4,
                  "estimate": estimate, "scheduler_before": snapshot, "storage_before": health}
        if interactive:
            intent.update(mode="salloc-no-shell", controller=str(Path(__file__).resolve()),
                          controller_sha256=judge.file_hash(__file__))
        budget["submissions"].append(intent)
        atomic(ledger, judge.canonical(budget).encode())
        command = (["salloc", "--no-shell", "--immediate=30"] if interactive else ["sbatch", "--parsable"]) + [
                   "--partition=accelerated", "--account=" + config["account"],
                   "--job-name=" + pilot.JOB_NAME, "--nodes=1", "--gres=gpu:4", "--cpus-per-task=32", "--exclusive",
                   "--time=" + config["walltime"], "--comment=si-v4-r2:" + intent["submission_id"]]
        command += ["--ntasks=1"] if interactive else [
            "--output=" + str(config_path.parent / "slurm-%j.out"), str(config_path.parent / "run.sh")]
        print(json.dumps({"estimate": estimate, "command": command, "remaining_phase_segments": 4 - len(attempted)}, indent=2), flush=True)
        # A crash/timeout leaves the durable intent unresolved; never guess that submission failed.
        if interactive:
            # Slurm bounds queue waiting. Never kill salloc during allocation/prolog.
            result = subprocess.run(command, text=True, capture_output=True, env={**os.environ, "LC_ALL": "C"})
            intent["scheduler_receipt"] = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
            matches = re.findall(r"^salloc: Granted job allocation (\d+)\s*$", result.stderr, re.MULTILINE)
            job = matches[0] if len(matches) == 1 else ""
            if job:
                intent["job_id"] = job
            atomic(ledger, judge.canonical(budget).encode())
            print(result.stdout + result.stderr, end="", flush=True)
            if result.returncode:
                raise RuntimeError("salloc did not finish successfully; saved receipt retained, inspect scheduler before retrying")
        else:
            result = subprocess.run(command, text=True, capture_output=True, timeout=60, check=True)
            job = result.stdout.strip().split(";")[0]
        if not job.isdecimal():
            raise ValueError("ambiguous submission receipt; preserve intent and inspect scheduler")
        intent["job_id"] = job
        atomic(ledger, judge.canonical(budget).encode())
        print("SUBMITTED_JOB", job, flush=True)
    if interactive:
        # run-segment takes the same budget lock. Release it before starting the step.
        command = ["srun", "--jobid=" + job, "--nodes=1", "--ntasks=1", "--cpus-per-task=32",
                   "--gres=gpu:4", "--unbuffered", "bash", str(config_path.parent / "run.sh")]
        print("RUNNING_PINNED_SEGMENT " + json.dumps(command), flush=True)
        result = subprocess.run(command, env={**os.environ, "PYTHONPATH": config["repository"]})
        print(f"SEGMENT_EXIT_CODE={result.returncode} ALLOCATION_PRESERVED={job}", flush=True)
        return result.returncode
    return 0


def run_segment(args):
    from analysis.scripts import horeka_nemotron as stage
    config_path = args.config.resolve()
    ledger = Path(read(config_path)["cycle_budget_path"])
    with budget_lock(ledger, blocking=True) as budget:
        config = checked_config(config_path, budget)
        rows = [r for r in budget["submissions"] if r.get("job_id") == os.environ.get("SLURM_JOB_ID")
                and r["phase"] == config["evaluation_phase"] and r["config_sha256"] == judge.file_hash(config_path)]
        if len(rows) != 1:
            raise ValueError("allocation is not an identified member of this finite cycle")
    return stage.execute(config_path)


def allocation_usage(submissions):
    from analysis.scripts.capture_agentic_scheduler_snapshot import TERMINAL_STATES
    totals = {"development": 0.0, "fresh": 0.0}
    records, complete = [], True
    for row in submissions:
        job = row.get("job_id")
        if not job:
            complete = False
            records.append({"submission_id": row["submission_id"], "status": "unresolved_submission"})
            continue
        raw = subprocess.check_output(["sacct", "-X", "-j", job, "--noheader", "--parsable2",
                                       "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"], text=True, timeout=30)
        matches = [s.split("|") for s in raw.splitlines() if s.split("|")[0].strip() == job]
        if len(matches) != 1:
            complete = False
            records.append({"job_id": job, "status": "accounting_unavailable", "raw": raw})
            continue
        _, state, elapsed, tres, *_ = matches[0]
        state = state.strip().split()[0].rstrip("+")
        allocation = dict(part.split("=", 1) for part in tres.split(",") if "=" in part)
        if state not in TERMINAL_STATES:
            complete = False
        gpu = allocation.get("gres/gpu")
        if not elapsed.strip().isdecimal() or (int(elapsed) > 0 and gpu != "4"):
            complete = False
            records.append({"job_id": job, "status": "resources_unresolved", "raw": raw})
            continue
        hours = int(elapsed) * int(gpu or 0) / 3600
        totals[row["phase"]] += hours
        records.append({"job_id": job, "phase": row["phase"], "state": state, "gpu_hours": hours, "raw": raw})
    return {"gpu_hours_total": sum(totals.values()), "gpu_hours_by_phase": totals, "complete": complete,
            "allocations": records, "captured_at_epoch": time.time()}


def resource_usage(args):
    with budget_lock(args.budget) as budget:
        value = {**allocation_usage(budget["submissions"]), "budget_sha256": judge.file_hash(args.budget)}
    with args.output.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))
    return 0 if value["complete"] else 2


def export(args):
    """Export the bounded evaluation evidence, without serving caches or credentials."""
    import tarfile
    config = read(args.config)
    root = args.config.resolve().parent
    if config.get("workload_mode") != "evaluation-v4":
        raise ValueError("evaluation preparation required")
    results = root / "evaluation-results"
    if not results.is_dir():
        raise ValueError("no evaluation results exist")
    with (results / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = [root / "config.json", root / "evaluation-plan.json", root / "judge-config.json", results]
        entries += [root / name for name in config["inventories"]]
        entries += list((root / "attempts").glob("job*/trial-result.json"))
        entries += list((root / "attempts").glob("job*/execution.json"))
        entries += list(root.glob("slurm-*.out"))
        if args.output.resolve().is_relative_to(results.resolve()):
            raise ValueError("archive cannot be inside result data")
        with args.output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            def include(info):
                if info.issym() or info.islnk():
                    raise ValueError("evidence export refuses redirected files")
                return info
            for entry in entries:
                archive.add(entry, arcname=str(entry.relative_to(root)), filter=include)
    print(json.dumps({"archive": str(args.output.resolve()), "sha256": judge.file_hash(args.output),
                      "scientific_acceptance": "not_established_by_export"}, indent=2))
    return 0


def assess(args):
    from analysis.scripts import compare_source_importance_runs as evaluator
    from analysis.scripts.prepare_si_v4_evaluation import summarize_supplements
    root = args.config.resolve().parent
    spec = validate_plan(root / "evaluation-plan.json")
    results = root / "evaluation-results"
    candidate = latest_report(results / "candidate-e2e-1")
    if candidate is None:
        raise ValueError("no candidate execution report exists; do not invent scores")
    controls_report = latest_report(results / "constructed")
    controls = evaluator.summarize_controls(root / "constructed-inputs", controls_report) if controls_report else None
    supplements = None
    if spec["phase"] == "development":
        supplements = summarize_supplements(root / "order-inputs", latest_report(results / "list-order"),
            root / "role-inputs", latest_report(results / "role-controls"),
            references=args.supplement_references, reference_packets=args.supplement_packets)
    else:
        # The fresh phase keeps the passing development controls and frozen judge
        # settings. The evaluator re-reads their saved evidence before acceptance.
        supplements = (spec.get("development_report") or {}).get("gates", {}).get(
            "supplementary_controls", {}).get("observed")
    repeats = {"end_to_end": [latest_report(results / f"candidate-e2e-{i}") for i in (1, 2, 3)],
               "fixed_map": [latest_report(results / f"candidate-fixed-{i}") for i in (1, 2, 3)]}
    repeats = {key: [p for p in paths if p is not None] for key, paths in repeats.items()}
    report = evaluator.evaluate(candidate, inputs=root / "candidate-inputs", references=args.references,
        reference_packets=args.reference_packets, repeat_cohort=spec["repeat_selection"], repeats=repeats,
        cost_baseline=latest_report(results / "cost-reference"), phase=spec["phase"], resource_usage=args.resource_usage,
        controls=controls, supplements=supplements, selection=spec.get("fresh_selection"))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    evaluator.write_absolute_html(args.output.with_suffix(".html"), report)
    print(json.dumps({"decision": report["decision"], "report": str(args.output),
                      "html": str(args.output.with_suffix('.html'))}, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    f = commands.add_parser("plan")
    for name in ("inputs", "constructed", "cost-inputs", "output"):
        f.add_argument("--" + name, type=Path, required=True)
    f.add_argument("--phase", choices=("development", "fresh"), required=True)
    for name in ("order-inputs", "role-inputs", "role-maps", "development-report", "selection"):
        f.add_argument("--" + name, type=Path)
    development = commands.add_parser("prepare-development")
    for name in ("source-run", "output", "list-spec"):
        development.add_argument("--" + name, type=Path, required=True)
    development.add_argument("--list-cell-id", required=True)
    r = commands.add_parser("reconcile")
    r.add_argument("--root", type=Path, required=True)
    for name in ("submit", "run-segment"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "submit":
            command.add_argument("--interactive", action="store_true", help="allocate with salloc and run one pinned segment through srun; preserve allocation")
            command.add_argument("--account-existing-job", action="append", default=[],
                                 help="charge a prior terminal user-created allocation before its interactive replacement")
    usage = commands.add_parser("resource-usage")
    usage.add_argument("--budget", type=Path, required=True)
    usage.add_argument("--output", type=Path, required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--config", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    assessment = commands.add_parser("assess")
    for name in ("config", "output"):
        assessment.add_argument("--" + name, type=Path, required=True)
    for name in ("references", "reference-packets", "supplement-references", "supplement-packets", "resource-usage"):
        assessment.add_argument("--" + name, type=Path)
    args = p.parse_args(argv)
    if args.command == "plan":
        return plan(args)
    if args.command in ("submit", "run-segment", "resource-usage", "prepare-development", "export", "assess"):
        return {"submit": submit, "run-segment": run_segment, "resource-usage": resource_usage,
                "prepare-development": prepare_development, "export": export, "assess": assess}[args.command](args)
    reconcile(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
