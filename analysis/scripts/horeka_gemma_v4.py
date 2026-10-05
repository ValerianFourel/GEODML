#!/usr/bin/env python3
"""Prepare finite five-hour Gemma SI-v4 bouts on verified saved generator cells.

The login-side start command submits one GPU preparation allocation, or with
--prepare-on-cpu one four-hour CPU-only allocation, or with --prepare-on-login
freezes inputs on the login host (each an explicit operator authorization), then
hands the immutable plan to the finite sender. With --reuse-frozen, a completed
freeze of the same inputs (e.g. left by a timed-out preparation) is verified and
adopted, and the CPU allocation only partitions and plans. Inference reuses the existing HoreKa
serving boundary and SI coordinator. No model downloads or scientific repairs.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts import horeka_gemma_si as gemma, horeka_nemotron as stage
from analysis.scripts import run_source_importance_judge as judge
from analysis.scripts.prepare_horeka_qwen import capture_quota
from analysis.scripts.capture_agentic_scheduler_snapshot import TERMINAL_STATES

REPO = Path(__file__).resolve().parents[2]
FORMAT = "gemma-v4-bouts-v1"
REGISTRY = "coordination/si-v4-gemma/registry.json"
JOB_NAME = "geodml-gemma-v4-bout"
PREP_JOB = "geodml-gemma-v4-prepare"
# Measured 2026-10-04: the single-threaded llama freeze builds ~1,834 cells/min
# after an 8-minute scan, about three hours for ~312k cells. Valerian approved
# this one four-hour CPU allocation explicitly.
CPU_PREP_WALLTIME = "04:00:00"
# Partition and plan only, from a verified earlier freeze; within the default limit.
CPU_REUSE_WALLTIME = "01:00:00"
WALLTIME = "05:00:00"
SECONDS_PER_CELL = 20.39868898486273
USEFUL_SECONDS = 18000 - 654 - 300


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    atomic(Path(path), judge.canonical(value).encode())


def checkout_pin(repository):
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("use a clean committed checkout")
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def clean_pin():
    return checkout_pin(REPO)


def checked_plan(root):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    # Bouts always run from the plan's pinned checkout. A newer clean, committed
    # sender may supervise them; it then verifies that pinned checkout, not itself.
    execution = Path(plan.get("repository") or "relative")
    if not execution.is_absolute():
        raise ValueError("plan or execution pin differs from the approved five-hour run")
    supervisor = clean_pin()
    pinned = supervisor if execution == REPO else checkout_pin(execution)
    if (plan.get("format_version") != FORMAT or plan.get("git_commit") != pinned
            or plan.get("root") != str(root)
            or plan.get("walltime") != WALLTIME or plan.get("max_inflight") != 200
            or plan.get("poll_seconds") != 600 or plan.get("job_name") != JOB_NAME
            or plan.get("partition") != "accelerated"):
        raise ValueError("plan or execution pin differs from the approved five-hour run")
    if not root.is_relative_to(Path(plan["workspace"]).resolve()):
        raise ValueError("plan is outside its quota-checked workspace")
    seen = set()
    for shard in plan["shards"]:
        directory = Path(shard["directory"])
        if shard["id"] in seen or directory.resolve() != root / "shards" / shard["id"]:
            raise ValueError("duplicate or redirected shard")
        seen.add(shard["id"])
        if shard["cells"] <= 0 or shard["max_allocations"] != max(1, math.ceil(shard["cells"] * SECONDS_PER_CELL / USEFUL_SECONDS)):
            raise ValueError("shard budget differs from the frozen estimate")
        for name, digest in shard["files"].items():
            if name not in ("config.json", "judge-config.json", "run.sh", "inputs/manifest.json") or judge.file_hash(directory / name) != digest:
                raise ValueError("shard configuration changed")
    if plan["maximum_allocations"] != sum(s["max_allocations"] for s in plan["shards"]):
        raise ValueError("total budget differs from its finite shards")
    if judge.file_hash(root / "claims.jsonl.gz") != plan["claims_sha256"]:
        raise ValueError("reservation identities changed")
    return plan


def storage(plan, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    quota = directory / "quota.json"
    save(quota, capture_quota(Path(plan["workspace"]), plan["account"]))
    result = judge.check_storage(plan["workspace"], quota, directory)
    if not result.get("safe_to_admit") or not result.get("quota_verified"):
        raise ValueError("storage/quota blocks new work: " + ", ".join(result.get("reasons", [])))
    return result


def reserve(plan, root):
    """Reserve exact SI map identities through a private, compare-and-set HF update.

    This registry is separate from generator/shared-hour ownership. It never
    expires claims or marks judgments complete from scheduler state alone.
    """
    from analysis.interpretability.pipeline.agentic_hour_sync import HubStore, ConflictError
    root = Path(root)
    store = HubStore(plan["repo_id"])
    own = (root / "claims.jsonl.gz").read_bytes()
    identities = {r["map_task_id"] for r in judge.rows(root / "claims.jsonl.gz")}
    claim_path = f"coordination/si-v4-gemma/plans/{plan['plan_id']}/claims.jsonl.gz"
    entry = {"plan_id": plan["plan_id"], "cluster": "horeka", "root": str(root.resolve()),
             "git_commit": plan["git_commit"], "claims": claim_path,
             "claims_sha256": plan["claims_sha256"], "state": "reserved"}
    for _ in range(4):
        revision = store.head()
        raw = store.read(REGISTRY, revision)
        registry = json.loads(raw) if raw is not None else {"format_version": FORMAT, "plans": {}}
        if registry.get("format_version") != FORMAT:
            raise ValueError("unknown SI ownership registry")
        for other_id, other in registry["plans"].items():
            if other_id == plan["plan_id"]:
                if other != entry or store.read(claim_path, revision) != own:
                    raise ValueError("existing SI reservation changed")
                continue
            other_raw = store.read(other["claims"], revision)
            if other_raw is None or hashlib.sha256(other_raw).hexdigest() != other["claims_sha256"]:
                raise ValueError("unverified SI ownership; no new claims")
            for line in gzip.decompress(other_raw).splitlines():
                if json.loads(line)["map_task_id"] in identities:
                    raise ValueError(f"SI work already reserved by {other_id}; preserve its ownership")
        if plan["plan_id"] in registry["plans"]:
            save(root / "reservation.json", {**entry, "revision": revision})
            return
        registry["plans"][plan["plan_id"]] = entry
        try:
            committed = store.commit(revision, {REGISTRY: judge.canonical(registry).encode(),
                claim_path: own, f"coordination/si-v4-gemma/plans/{plan['plan_id']}/plan.json": (root / "plan.json").read_bytes()},
                "Reserve finite Gemma SI-v4 work")
        except ConflictError:
            continue
        save(root / "reservation.json", {**entry, "revision": committed})
        return
    raise ValueError("SI ownership registry kept changing; no submission made")


def reconcile_output(results):
    """Recover ended owners with Slurm evidence, then verify sealed cached results."""
    results = Path(results)
    database = results / "control/index.sqlite"
    if not database.exists():
        return {"status": "not_started", "states": {}, "counts": {}, "inference_failures": 0}
    live = set(subprocess.check_output(["squeue", "--me", "--array", "--noheader", "--format=%i"], text=True, timeout=30).split())
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        owners = db.execute("SELECT DISTINCT writer_id,job_id FROM tasks WHERE state IN ('running','saved')").fetchall()
    evidence = []
    for writer, job in owners:
        if not str(job).isdecimal() or job in live:
            raise ValueError("previous writer has a live or unknown allocation")
        raw = subprocess.check_output(["sacct", "-X", "-j", str(job), "--noheader", "--parsable2", "--format=JobIDRaw,State"], text=True, timeout=30)
        states = [r.split("|")[1].strip().split()[0].rstrip("+") for r in raw.splitlines() if r.split("|")[0].strip() == str(job)]
        if len(states) != 1 or states[0] not in TERMINAL_STATES:
            raise ValueError("previous writer not confirmed terminal")
        evidence.append({"writer_id": writer, "slurm_job_id": job, "owner_terminal": True,
                         "evidence": raw, "captured_at_epoch": time.time()})
    receipt = results / "recovery-evidence.json"
    save(receipt, evidence)
    config = judge.load_config(results.parent / "judge-config.json")
    config["code_revision"] = read(results.parent / "config.json")["git_commit"]
    coordinator = judge.Coordinator(results.parent / "inputs", results, config, recovery_evidence=receipt)
    try:
        report = coordinator.report()
        pending = {r[0] for r in coordinator.db.execute(
            "SELECT id FROM tasks WHERE state IN ('pending','waiting','running','saved')")}
        remaining = 0
        for cell in judge.rows(results.parent / "inputs/cells.jsonl.gz"):
            ids = {cell.get("map_task_id"), *(s.get("dependency_id") for s in cell.get("sources", []))}
            remaining += bool(ids & pending)
        complete = report["counts"].get("cells_complete", 0)
        warm_seconds = report.get("timing", {}).get("si", {}).get("node_hours", 0) * 3600
        rate = warm_seconds / complete if complete and warm_seconds > 0 else SECONDS_PER_CELL
        report["remaining_estimate"] = {"cells_with_pending_tasks": remaining,
            "seconds_per_cell": rate, "basis": "saved shard warm throughput" if complete else "original Gemma measurements",
            "warm_remaining_node_hours": remaining * rate / 3600,
            "next_bout_expected_minutes": min(300, 15.9 + remaining * rate / 60) if remaining else 0,
            "next_bout_walltime_cap": WALLTIME, "known_terminal_failures_not_retried": True}
        save(results / "remaining-estimate.json", report["remaining_estimate"])
        return report
    finally:
        coordinator.close()


async def review(args, config):
    """Keep one server loaded for one shard; use the same durable output on resume."""
    root = Path(args.config).parent
    args.output.mkdir(parents=True, exist_ok=False)
    quota = args.output / "quota.json"
    stop = asyncio.Event()

    async def refresh():
        while not stop.is_set():
            try:
                await asyncio.to_thread(storage, config, args.output)
            except Exception as error:
                atomic(quota, b"{}")
                save(args.output / "quota-error.json", {"error": str(error)})
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=120)
            except asyncio.TimeoutError:
                pass

    await asyncio.to_thread(storage, config, args.output)
    refresher = asyncio.create_task(refresh())
    result = root / "results"
    recovery = result / "recovery-evidence.json"
    try:
        code = await judge.run(SimpleNamespace(inputs=root / "inputs", output=result,
            config=root / "judge-config.json", workspace=Path(config["workspace"]),
            quota_evidence=quota, base_url=args.base_url, fixed_maps=None,
            recovery_evidence=recovery if recovery.exists() else None))
        from analysis.scripts.run_si_v4_cycle import latest_report
        summary = read(latest_report(result) / "summary.json")
        save(args.output / "summary.json", summary)
        return code
    finally:
        stop.set()
        await refresher


def bind_submission(root, shard, job, comment):
    """Bind one authorized durable submission to exactly one actual Slurm job."""
    if not isinstance(comment, str) or not re.fullmatch(r"geodml-gemma-v4:[0-9a-f]{32}", comment):
        raise ValueError("allocation lacks a unique Gemma submission comment")
    candidates = []
    consumed = 0
    for path in sorted((Path(shard["directory"]) / "submissions").glob("attempt-*/intent.json")):
        intent = read(path)
        receipt_path = path.parent / "receipt.json"
        receipt = read(receipt_path) if receipt_path.exists() else {}
        if receipt.get("disposition") in ("capacity_refused", "policy_refused"):
            continue
        consumed += 1
        if intent.get("comment") == comment:
            candidates.append((path, intent, receipt))
    if consumed > shard["max_allocations"] or len(candidates) != 1:
        raise ValueError("job is outside the finite recorded submission budget")
    path, intent, receipt = candidates[0]
    if (intent.get("shard_id") != shard["id"]
            or intent.get("plan_sha256") != judge.file_hash(Path(root) / "plan.json")
            or intent.get("git_commit") != read(Path(root) / "plan.json")["git_commit"]
            or intent.get("config_sha256") != shard["config_sha256"]
            or receipt.get("job_id", job) != job):
        raise ValueError("allocation differs from its frozen submission intent")
    # The intent is durable before sbatch. A fast worker may arrive before the
    # caller saves its receipt; the unique scheduler comment binds that window.
    with (path.parent / "worker-binding.json").open("x") as stream:
        json.dump({"job_id": job, "comment": comment, "started_at_epoch": time.time()}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def run_shard(args):
    config = read(args.config)
    root = Path(config["bulk_root"])
    plan = checked_plan(root)
    reservation = read(root / "reservation.json")
    if reservation.get("plan_id") != plan["plan_id"] or reservation.get("claims_sha256") != plan["claims_sha256"]:
        raise ValueError("missing verified HF reservation")
    shard = next((s for s in plan["shards"] if Path(s["directory"]) / "config.json" == args.config.resolve()), None)
    if shard is None:
        raise ValueError("shard not in the finite plan")
    fields = dict(p.split("=", 1) for p in subprocess.check_output(
        ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True, timeout=30).split() if "=" in p)
    if (fields.get("Account") != plan["account"] or fields.get("Partition") != "accelerated"
            or not fields.get("UserId", "").endswith(f"({os.getuid()})")):
        raise ValueError("allocation owner, account or partition differs")
    with (args.config.parent / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_submission(root, shard, os.environ["SLURM_JOB_ID"], fields.get("Comment"))
        storage(plan, args.config.parent / "startup")
        settings = read(args.config.parent / "judge-config.json")
        expected = {**settings["runtime_versions"], "vllm": settings["serving_version"].removeprefix("vllm-")}
        versions = gemma.runtime_check()
        if any(versions[key] != value for key, value in expected.items()):
            raise ValueError("runtime changed since preparation; refusing model startup")
        for filename, digest in config["inventories"]["inputs"]["files"].items():
            if filename not in ("cells.jsonl.gz", "tasks.jsonl.gz") or judge.file_hash(args.config.parent / "inputs" / filename) != digest:
                raise ValueError("frozen shard changed; refusing model startup")
        return stage.execute(args.config)


def exclusions(spec, root):
    """Record the explicitly named earlier diagnostic cohorts without rerunning them."""
    cells, maps = set(), set()
    sources = []
    for folder in spec["exclude_inputs"]:
        folder = Path(folder)
        manifest = read(folder / "manifest.json")
        if manifest["protocol"] != v4.PROTOCOL:
            raise ValueError("exclusions must be earlier SI-v4 inputs")
        for name in ("cells.jsonl.gz", "tasks.jsonl.gz"):
            if judge.file_hash(folder / name) != manifest["files"][name]:
                raise ValueError("earlier diagnostic input changed")
        for cell in judge.rows(folder / "cells.jsonl.gz"):
            cells.add(cell["fingerprint"])
            if cell.get("map_task_id"):
                maps.add(cell["map_task_id"])
        sources.append({"inputs": str(folder), "manifest_sha256": judge.file_hash(folder / "manifest.json")})
    path = root / "previously-attempted-cells.txt"
    atomic(path, ("".join(fp + "\n" for fp in sorted(cells))).encode())
    atomic(root / "previously-attempted-maps.txt", ("".join(tid + "\n" for tid in sorted(maps))).encode())
    save(root / "excluded-diagnostics.json", {"cells": len(cells), "maps": len(maps), "sources": sources,
        "reason": "preserve earlier diagnostic attempts including failures; no acceptance implied"})
    return path


def adopt_frozen(reuse, frozen, spec, root, excluded):
    """Adopt a completed freeze of exactly these inputs instead of freezing again."""
    source = Path(reuse["path"])
    if judge.file_hash(source / "manifest.json") != reuse["manifest_sha256"]:
        raise ValueError("reused freeze manifest changed since start")
    manifest = read(source / "manifest.json")
    for name in ("tasks.jsonl.gz", "cells.jsonl.gz"):
        if judge.file_hash(source / name) != manifest["files"][name]:
            raise ValueError(f"reused freeze file changed: {name}")
    # Same settings as the freeze command in prepare(); partition rechecks them.
    if (manifest.get("protocol") != v4.PROTOCOL or manifest.get("max_tokens") != 4096
            or manifest.get("map_max_tokens") != 4096
            or manifest.get("truncation_sensitivity", {}).get("fraction") != 0
            or manifest.get("limit") is not None or manifest.get("cell_fingerprints_sha256")):
        raise ValueError("reused freeze used different settings")
    if [f"{i['dataset_root']}:{i['model']}" for i in manifest["inputs"]] != spec["sources"]:
        raise ValueError("reused freeze covers different generator datasets")
    if (manifest.get("excluded_cells", {}).get("sha256") != judge.file_hash(excluded)
            or manifest.get("prior_map_tasks", {}).get("sha256")
            != judge.file_hash(root / "previously-attempted-maps.txt")):
        raise ValueError("reused freeze excluded different earlier diagnostics")
    if frozen.exists():
        if judge.file_hash(frozen / "manifest.json") != reuse["manifest_sha256"]:
            raise ValueError("frozen inputs differ from the reused freeze")
        return
    partial = frozen.with_name("frozen.reuse-partial")
    if partial.exists():
        shutil.rmtree(partial)  # only ever a copy of the verified source
    partial.mkdir()
    for name in ("manifest.json", "tasks.jsonl.gz", "cells.jsonl.gz"):
        shutil.copy2(source / name, partial / name)
    partial.rename(frozen)
    save(root / "frozen-reuse.json", {**reuse, "freeze_git_commit": manifest.get("git_commit"),
                                      "counts": manifest.get("counts"), "adopted_at_epoch": time.time()})


def prepare(args):
    """Freeze saved inputs; login preparation is an explicit operator exception."""
    from analysis.scripts.prepare_source_importance_tasks import main as freeze
    from analysis.scripts.partition_si_v4_tasks import partition
    from analysis.scripts.verify_inference_allocation import verify
    root = args.output.resolve()
    spec = read(root / "preparation.json")
    login = getattr(args, "command", "prepare") == "prepare-login"
    execution_repo = REPO
    helper_pin = clean_pin()
    if login:
        if os.environ.get("SLURM_JOB_ID"):
            raise ValueError("login preparation must run outside an allocation")
        # A fresh run chose login preparation at start; otherwise this is a
        # takeover of a retired GPU preparation job.
        if spec.get("preparation_execution") != "login":
            takeover = read(root / "prequeue/login-takeover.json")
            retired = read(root / "prequeue/login-preparation-retired.json")
            if (takeover.get("helper_pin") != helper_pin or retired.get("state") != "CANCELLED"
                    or retired.get("job_id") != takeover.get("preparation_job")):
                raise ValueError("verified retirement of GPU preparation is required")
        execution_repo = args.inference_repository.resolve(strict=True)
        pin = subprocess.check_output(["git", "-C", str(execution_repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(execution_repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
        if dirty or pin != spec["git_commit"]:
            raise ValueError("original inference checkout changed")
        boundary = {"execution": "login", "authorization": "explicit user request to prepare on login shell",
                    "preparation_code_revision": helper_pin, "inference_code_revision": pin,
                    "hostname": os.uname().nodename, "started_at_epoch": time.time(), "gpu_used": False}
    else:
        if spec["git_commit"] != helper_pin:
            raise ValueError("preparation checkout changed")
        if spec.get("preparation_execution") == "cpu":
            if not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("CPU preparation must run inside its Slurm allocation")
            # No model or inference runs here, so no exclusive GPU boundary is needed.
            boundary = {"execution": "cpu_slurm", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                        "authorization": "explicit user approval of one four-hour CPU preparation allocation",
                        "preparation_code_revision": helper_pin, "hostname": os.uname().nodename,
                        "started_at_epoch": time.time(), "gpu_used": False}
        else:
            boundary = verify("horeka")
    save(root / "preparation-boundary.json", boundary)
    storage(spec, root / "preparation-storage")
    frozen = root / "frozen"
    excluded = exclusions(spec, root)
    command = ["--output", str(frozen), "--protocol", "si-v4", "--max-tokens", "4096",
               "--map-max-tokens", "4096", "--truncation-sensitivity-fraction", "0",
               "--exclude-cell-fingerprints", str(excluded),
               "--prior-map-task-ids", str(root / "previously-attempted-maps.txt")]
    for source in spec["sources"]:
        command += ["--source", source]
    if spec.get("preparation_execution") == "cpu" and os.environ.get("TMPDIR"):
        command += ["--index-directory", os.environ["TMPDIR"]]  # node-local scratch, not the shared workspace
    if spec.get("reuse_frozen"):
        adopt_frozen(spec["reuse_frozen"], frozen, spec, root, excluded)
    else:
        freeze(command)
    storage(spec, root / "preparation-storage")
    shards = partition(frozen, root / "shards",
                       index_directory=os.environ.get("TMPDIR") if spec.get("preparation_execution") == "cpu" else None)
    settings = read(root / "judge-config.json")
    plan_id = "gemma-v4-" + hashlib.sha256((judge.file_hash(frozen / "manifest.json") + judge.file_hash(root / "judge-config.json")).encode()).hexdigest()[:24]
    claims = set()
    for shard in shards:
        folder = Path(shard["directory"])
        for cell in judge.rows(folder / "inputs/cells.jsonl.gz"):
            claims.add(cell["map_task_id"])
        save(folder / "judge-config.json", settings)
        manifest = read(folder / "inputs/manifest.json")
        config = {"repository": str(execution_repo), "git_commit": spec["git_commit"], "workspace": spec["workspace"],
            "account": spec["account"], "approval": spec["authorization"], "job_name": JOB_NAME,
            "walltime": WALLTIME, "trial": "si-v4-development", "workload_mode": "gemma-v4-bulk",
            "model_id": gemma.MODEL, "model_revision": gemma.REVISION, "serving": stage.SERVING,
            "judge_protocol": v4.PROTOCOL, "bulk_root": str(root), "plan_id": plan_id,
            "development_config": str(folder / "config.json"), "inventories": {"inputs": manifest},
            "judge_config_sha256": judge.file_hash(folder / "judge-config.json"),
            "cache_prefix": "gemma4-v4-bulk", "structured_outputs_config": settings["structured_outputs_config"],
            "scientific_result": False, "semantic_acceptance": "not_established"}
        save(folder / "config.json", config)
        run = [sys.executable, str(execution_repo / "analysis/scripts/horeka_gemma_v4.py"), "run-shard", "--config", str(folder / "config.json")]
        script = '#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1\n'
        script += 'exec srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --unbuffered ' + shlex.join(run) + '\n'
        atomic(folder / "run.sh", script.encode())
        shard["max_allocations"] = max(1, math.ceil(shard["cells"] * SECONDS_PER_CELL / USEFUL_SECONDS))
        shard["files"] = {name: judge.file_hash(folder / name) for name in
                          ("config.json", "judge-config.json", "run.sh", "inputs/manifest.json")}
        shard["config_sha256"] = shard["files"]["config.json"]
    with gzip.GzipFile(filename=str(root / "claims.jsonl.gz"), mode="wb", mtime=0) as stream:
        for task_id in sorted(claims):
            stream.write((judge.canonical({"map_task_id": task_id}) + "\n").encode())
    plan = {"format_version": FORMAT, "plan_id": plan_id, "root": str(root),
        "repository": str(execution_repo), "git_commit": spec["git_commit"], "workspace": spec["workspace"],
        "account": spec["account"], "repo_id": spec["repo_id"], "authorization": spec["authorization"],
        "created_at_epoch": int(time.time()), "deadline_epoch": int(time.time()) + 30 * 86400,
        "partition": "accelerated", "walltime": WALLTIME, "job_name": JOB_NAME,
        "max_inflight": 200, "poll_seconds": 600, "maximum_allocations": sum(s["max_allocations"] for s in shards),
        "shards": shards, "claims_sha256": judge.file_hash(root / "claims.jsonl.gz"),
        "input_manifest_sha256": judge.file_hash(frozen / "manifest.json"),
        "input_counts": read(frozen / "manifest.json")["counts"],
        "input_inventory": read(frozen / "manifest.json")["inputs"],
        "excluded_diagnostics": read(root / "excluded-diagnostics.json"),
        "cells": sum(s["cells"] for s in shards), "scientific_result": False,
        "semantic_acceptance": "not_established", "preparation_allocations": 1,
        "cost_assumptions": {"seconds_per_completed_cell": SECONDS_PER_CELL,
            "startup_seconds": 654, "drain_seconds": 300, "expected_average_running_nodes": 15,
            "queue_wait_excluded": True, "no_automatic_budget_expansion": True}}
    plan["maximum_node_hours"] = 1 + 5 * plan["maximum_allocations"]
    plan["maximum_gpu_hours"] = 4 * plan["maximum_node_hours"]
    if login or spec.get("preparation_execution") == "cpu":
        plan["preparation_execution"] = {**boundary, "finished_at_epoch": time.time()}
    if spec.get("reuse_frozen"):
        plan["reused_frozen"] = spec["reuse_frozen"]
    save(root / "plan.json", plan)
    print(json.dumps({"prepared_cells": plan["cells"], "maximum_allocations": plan["maximum_allocations"],
                      "maximum_node_hours": plan["maximum_node_hours"], "allocation_submitted": False}), flush=True)
    return 0


def start(args):
    """Login-side finite preparation then sending; safe to restart the same tmux command."""
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run this sender in a separate login shell; preserve the live allocation")
    root, workspace = args.output.resolve(), args.workspace.resolve(strict=True)
    sources = []
    for source in args.source:
        directory, sep, model = source.rpartition(":")
        if not sep or model not in ("qwen38", "llama4"):
            raise ValueError("source must be DATASET_ROOT:qwen38 or DATASET_ROOT:llama4")
        sources.append(f"{Path(directory).resolve()}:{model}")
    if len(sources) != len(set(sources)):
        raise ValueError("duplicate generator dataset source")
    excluded_inputs = [str(p.resolve()) for p in args.exclude_inputs]
    login = bool(getattr(args, "prepare_on_login", False))
    cpu = bool(getattr(args, "prepare_on_cpu", False))
    if login and cpu:
        raise ValueError("choose one preparation mode")
    mode = "login" if login else "cpu" if cpu else "gpu"
    reuse_path = getattr(args, "reuse_frozen", None)
    if reuse_path is not None and not cpu:
        raise ValueError("--reuse-frozen requires --prepare-on-cpu")
    reuse_path = str(Path(reuse_path).resolve(strict=True)) if reuse_path is not None else None
    if not root.is_relative_to(workspace):
        raise ValueError("output must be inside the workspace")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin = clean_pin()
        if not (root / "preparation.json").exists():
            from analysis.interpretability.pipeline.agentic_hour_sync import HubStore
            HubStore(args.repo_id)  # Fail before allocating if cached private-repository access is unavailable.
            receipt = read(gemma.prep(workspace) / "models-verified.json")
            if receipt.get("status") != "verified" or [(m["repo_id"], m["revision"]) for m in receipt["models"]] != list(gemma.MODELS):
                raise ValueError("cached pinned Gemma model verification required; no automatic download")
            settings = judge.load_config(REPO / "analysis/config/si_v4_gemma_full_pass.template.json")
            settings["tokenizer_path"] = str(Path(receipt["models"][0]["snapshot"]).resolve(strict=True))
            versions = gemma.runtime_check()
            expected = {**settings["runtime_versions"], "vllm": settings["serving_version"].removeprefix("vllm-")}
            if any(versions[k] != value for k, value in expected.items()):
                raise ValueError("Gemma runtime changed")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(settings["tokenizer_path"], local_files_only=True, trust_remote_code=False)
            if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != settings["chat_template_sha256"]:
                raise ValueError("cached chat template changed")
            for source in args.source:
                Path(source.rpartition(":")[0]).resolve(strict=True)
            for path in args.exclude_inputs:
                path.resolve(strict=True)
            spec = {"git_commit": pin, "workspace": str(workspace), "root": str(root),
                "account": args.account, "repo_id": args.repo_id, "sources": sources,
                "exclude_inputs": excluded_inputs,
                "authorization": "Valerian requested the saved dataset judged by Gemma v4 in five-hour bouts, up to 200 queued ASAP, checking every ten minutes; one finite frozen pass.",
                "created_at_epoch": int(time.time()), "preparation_deadline_epoch": int(time.time()) + 7 * 86400}
            if mode != "gpu":
                spec["preparation_execution"] = mode
            if reuse_path:
                spec["reuse_frozen"] = {"path": reuse_path,
                                        "manifest_sha256": judge.file_hash(Path(reuse_path) / "manifest.json")}
            save(root / "judge-config.json", settings)
            save(root / "preparation.json", spec)
        spec = read(root / "preparation.json")
        if (spec["git_commit"] != pin or spec["workspace"] != str(workspace) or spec["account"] != args.account
                or spec["repo_id"] != args.repo_id or spec["sources"] != sources
                or spec["exclude_inputs"] != excluded_inputs
                or spec.get("preparation_execution", "gpu") != mode
                or (spec.get("reuse_frozen") or {}).get("path") != reuse_path):
            raise ValueError("saved preparation differs; preserve its pinned command")
        if not (root / "plan.json").exists():
            storage(spec, root / "admission")
            if login:
                marker = root / "LOGIN_PREPARATION_ATTEMPTED"
                if marker.exists():
                    raise ValueError("interrupted login preparation preserved; no automatic repeat")
                atomic(marker, b"one login preparation; never automatically repeated\n")
                print(json.dumps({"state": "preparing_on_login", "maximum_seconds": 3600, "time": time.time()}), flush=True)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
                subprocess.run(["nice", "-n", "10", sys.executable, "-u", str(Path(__file__)), "prepare-login",
                    "--output", str(root), "--inference-repository", str(REPO)], check=True, timeout=3600, env=env)
                checked_plan(root)
            else:
                submitted = root / "preparation-submission.json"
                if not submitted.exists():
                    command = [sys.executable, str(Path(__file__)), "prepare", "--output", str(root)]
                    if cpu:
                        script = ('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 '
                                  'MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false\n')
                        script += 'exec ' + shlex.join(command[:1] + ["-u"] + command[1:]) + '\n'
                        resources = ["--partition=cpuonly", "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
                                     "--mem=32G",
                                     f"--time={CPU_REUSE_WALLTIME if reuse_path else CPU_PREP_WALLTIME}"]
                    else:
                        script = '#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexport GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1\n'
                        script += 'exec srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --unbuffered ' + shlex.join(command) + '\n'
                        resources = ["--partition=accelerated", "--nodes=1", "--ntasks=1", "--gres=gpu:4",
                                     "--cpus-per-task=32", "--mem=0", "--exclusive", "--time=01:00:00"]
                    atomic(root / "prepare.sh", script.encode())
                    marker = root / "PREPARATION_SUBMISSION_ATTEMPTED"
                    if marker.exists():
                        raise ValueError("preparation submission has no receipt; inspect scheduler before retrying")
                    atomic(marker, b"one preparation allocation; never automatically resubmit\n")
                    command = ["sbatch", "--parsable", "--no-requeue", f"--account={args.account}", *resources,
                        f"--job-name={PREP_JOB}", f"--output={root}/preparation-%j.log", str(root / "prepare.sh")]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    save(submitted, {"command": command, "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
                result = read(submitted)
                job = result["stdout"].strip().split(";")[0]
                if result["returncode"] or not job.isdecimal():
                    raise ValueError("preparation submission failed; see preparation-submission.json")
                while time.time() < spec["preparation_deadline_epoch"]:
                    queue = subprocess.check_output(["squeue", "--noheader", "--me", "--array", "--format=%i|%T"], text=True, timeout=30)
                    live = next((r.split("|", 1)[1].strip() for r in queue.splitlines()
                                 if r.split("|", 1)[0].strip() == job), "")
                    if not live:
                        raw = subprocess.check_output(["sacct", "-X", "-j", job, "--noheader", "--parsable2", "--format=JobIDRaw,State"], text=True, timeout=30)
                        states = [r.split("|")[1].strip().split()[0].rstrip("+") for r in raw.splitlines() if r.split("|")[0].strip() == job]
                        if states and states[0] in TERMINAL_STATES:
                            if states[0] != "COMPLETED" or not (root / "plan.json").exists():
                                raise ValueError("preparation ended without a verified plan; inspect its log")
                            break
                    print(json.dumps({"preparation_job": job, "state": live or "awaiting_accounting", "time": time.time()}), flush=True)
                    time.sleep(600)
                else:
                    raise ValueError("preparation sender expired; all allocations preserved")
    from analysis.scripts.horeka_gemma_v4_sender import send
    return send(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--exclude-inputs", type=Path, action="append", default=[])
    p.add_argument("--account", required=True)
    p.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    p.add_argument("--prepare-on-login", action="store_true",
                   help="explicit operator authorization: freeze inputs on the login host instead of a GPU job")
    p.add_argument("--prepare-on-cpu", action="store_true",
                   help="explicit operator authorization: freeze inputs in one four-hour cpuonly allocation")
    p.add_argument("--reuse-frozen", type=Path,
                   help="with --prepare-on-cpu: adopt this verified freeze of the same inputs; one-hour allocation")
    p = commands.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("prepare-login")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inference-repository", type=Path, required=True)
    p = commands.add_parser("run-shard")
    p.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"start": start, "prepare": prepare, "prepare-login": prepare, "run-shard": run_shard}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
