#!/usr/bin/env python3
"""Finite, durable HoreKa Gemma SI-v4 sender for the explicitly approved plan.

Run on a login host, normally inside tmux. Never cancels, extends or requeues a
job. Ambiguous submissions consume budget and cannot be submitted again. The
runtime owns frozen-input validation, storage admission and task reconciliation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TERMINAL = frozenset({"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
                      "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT", "REVOKED"})
EXHAUSTED = frozenset({"finished", "finished_with_failures"})


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Publish fsynced JSON, including its directory entry, before external work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stamp():
    return datetime.fromtimestamp(time.time(), timezone.utc).isoformat()


def emit(value):
    print(json.dumps({"timestamp_utc": stamp(), **value}, sort_keys=True), flush=True)


class SenderBusy(ValueError):
    """Another finite Gemma sender in this workspace holds the lock."""


GLOBAL_MAX_GEMMA_BOUTS = 200  # Valerian's ceiling for Gemma bouts queued or running, across all runs
STARTUP_FAILURE_SECONDS = 300
STARTUP_FAILURE_STATES = frozenset({"FAILED", "NODE_FAIL", "BOOT_FAIL"})


def lock_name(plan) -> str:
    """One sender per run; runs share the global bout ceiling through submission_lock."""
    plan_id = plan.get("plan_id")
    if plan_id is None:
        return "gemma-v4-sender.lock"  # plans written before per-run locks
    if not re.fullmatch(r"gemma-v4-[0-9a-f]{24}", str(plan_id)):
        raise ValueError("invalid plan_id for the sender lock")
    return f"gemma-v4-sender-{plan_id}.lock"


@contextmanager
def sender_lock(workspace, name="gemma-v4-sender.lock"):
    path = Path(workspace) / "control" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SenderBusy(f"another Gemma sender holds {path}; preserve it") from error
        yield


@contextmanager
def submission_lock(workspace):
    """Held only while counting the queue and submitting, so two runs never overshoot the ceiling."""
    path = Path(workspace) / "control/gemma-v4-submit.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def gemma_bouts_in_queue() -> int:
    """Gemma bouts of every run, queued or running, from a fresh squeue."""
    rows = query(["squeue", "--noheader", "--me", "--array", "--format=%i|%T|%k|%j"]).splitlines()
    return sum(1 for row in rows if row.strip() and row.rsplit("|", 1)[-1].strip() == "geodml-gemma-v4-bout")


def validate(plan):
    if (plan.get("format_version") != "gemma-v4-bouts-v1"
            or plan.get("walltime") != "05:00:00"
            or plan.get("partition") != "accelerated"
            or plan.get("job_name") != "geodml-gemma-v4-bout"):
        raise ValueError("sender requires the frozen five-hour A100 Gemma plan")
    for name, ceiling in (("max_inflight", 200), ("poll_seconds", 600)):
        value = plan.get(name)
        if type(value) is not int or value != ceiling:
            raise ValueError(f"the authorized {name} is {ceiling}")
    maximum = plan.get("maximum_allocations")
    if type(maximum) is not int or maximum < 0:
        raise ValueError("maximum_allocations must be finite and nonnegative")
    deadline = plan.get("deadline_epoch")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("a finite sender deadline is required")
    ids, directories = set(), set()
    for shard in plan["shards"]:
        if not isinstance(shard["id"], str) or not shard["id"] or shard["id"] in ids:
            raise ValueError("shard IDs must be unique nonempty strings")
        directory = Path(shard["directory"])
        if not directory.is_absolute() or str(directory.resolve()) in directories:
            raise ValueError("shard directories must be unique absolute paths")
        if type(shard["max_allocations"]) is not int or shard["max_allocations"] < 1:
            raise ValueError("each shard needs a finite allocation limit")
        ids.add(shard["id"])
        directories.add(str(directory.resolve()))
    if maximum > sum(s["max_allocations"] for s in plan["shards"]):
        raise ValueError("overall allocation budget exceeds the shard budgets")


def state_file(root, plan):
    path = root / "sender/state.json"
    fingerprint = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    if path.exists():
        state = read(path)
        if state["plan_sha256"] != fingerprint:
            raise ValueError("frozen sender plan changed; no budget expansion or restart")
    else:
        state = {"format_version": "gemma-v4-sender-v1", "plan_sha256": fingerprint,
                 "created_at_epoch": time.time(), "deadline_epoch": plan["deadline_epoch"],
                 "maximum_allocations": plan["maximum_allocations"]}
        save(path, state)
    return state


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(f"{command[0]} query failed: {result.stderr.strip()}")
    return result.stdout


def snapshot(since):
    # Read live state last: an allocation still in squeue always defeats a stale
    # terminal sacct record. --array expands pending arrays for the user-wide cap.
    history = query(["sacct", "-X", "--noheader", "--parsable2", f"--starttime={since}",
                     "--format=JobIDRaw,State,Comment%256,JobName%256,ElapsedRaw,AllocCPUS,AllocNodes,AllocTRES%256,Start,End,ExitCode"])
    live = query(["squeue", "--noheader", "--me", "--array", "--format=%i|%T|%k|%j"])
    accounting, jobs = {}, {}
    for line in history.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split("|")]
        if len(fields) == 12 and not fields[-1]:
            fields.pop()
        if len(fields) != 11:
            raise ValueError("unexpected sacct allocation row")
        job, state, comment, name, elapsed, cpus, nodes, tres, start, end, code = fields
        if not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", job):
            raise ValueError(f"unclassified sacct allocation ID: {job}")
        row = {"job_id": job, "state": state.split()[0].rstrip("+"), "comment": comment,
               "job_name": name, "elapsed_seconds": int(elapsed) if elapsed.isdigit() else None,
               "allocated_cpus": int(cpus) if cpus.isdigit() else None,
               "allocated_nodes": int(nodes) if nodes.isdigit() else None,
               "allocated_tres": tres, "start": start, "end": end, "exit_code": code}
        if job in accounting and accounting[job] != row:
            raise ValueError(f"conflicting sacct records for {job}")
        accounting[job] = row
    for line in live.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split("|")]
        if len(fields) != 4 or not re.fullmatch(r"\d+(?:_\d+)?(?:\+\d+)?", fields[0]):
            raise ValueError("unexpected squeue allocation row; capacity is unknown")
        job, state, comment, name = fields
        if job in jobs:
            raise ValueError(f"duplicate squeue allocation {job}")
        jobs[job] = {"job_id": job, "state": state, "comment": comment, "job_name": name}
    return {"captured_at_epoch": time.time(), "complete": True,
            "jobs": jobs, "accounting": accounting}


def attempts(shard):
    directory = Path(shard["directory"]) / "submissions"
    if not directory.exists():
        return []
    found = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or not re.fullmatch(r"attempt-\d{4}", path.name):
            raise ValueError(f"unexpected submission artifact: {path}")
        intent = read(path / "intent.json")
        if intent["shard_id"] != shard["id"]:
            raise ValueError("submission marker belongs to another shard")
        found.append((path, intent))
    return found


def report(shard):
    results = Path(shard["directory"]) / "results"
    latest = results / "reports/latest.json"
    if not latest.exists():
        return {"status": "not_started", "states": {}, "counts": {}}
    pointer = read(latest)
    writer = pointer.get("directory")
    if not isinstance(writer, str) or Path(writer).name != writer or writer in (".", ".."):
        raise ValueError("invalid latest judge report pointer")
    return read(results / "reports" / writer / "summary.json")


def completed(progress):
    states = progress.get("states", {})
    return states.get("done", 0) + states.get("blocked", 0)


def identify(path, intent, observed, *, persist):
    """Recover the one job with this durable comment, including lost sbatch replies."""
    receipt_path = path / "receipt.json"
    receipt = read(receipt_path) if receipt_path.exists() else {}
    confirmed = receipt.get("job_id")
    matches = {job for group in (observed["accounting"], observed["jobs"])
               for job, row in group.items() if row["comment"] == intent["comment"]}
    if len(matches) > 1 or confirmed and matches and matches != {confirmed}:
        raise ValueError(f"submission comment has conflicting allocations: {intent['comment']}")
    if matches:
        job = next(iter(matches))
        row = observed["jobs"].get(job, observed["accounting"].get(job))
        if row["job_name"] != "geodml-gemma-v4-bout":
            raise ValueError("recovered allocation has an unexpected job name")
        if not confirmed and persist:
            save(receipt_path, {**receipt, "job_id": job, "recovered_by_comment": True,
                                "recovered_at_epoch": time.time()})
        return job
    if confirmed:
        # Known IDs cannot be silently reassigned when comments disappear. HoreKa
        # accounting does not store job comments, so a finished job's sacct row
        # has an empty one; only a different comment, or a live mismatch, is a conflict.
        live = observed["jobs"].get(confirmed)
        row = live or observed["accounting"].get(confirmed)
        if row is not None and row["comment"] != intent["comment"] and (live or row["comment"]):
            raise ValueError("known allocation no longer matches its submission comment")
        if row is not None and row["job_name"] != "geodml-gemma-v4-bout":
            raise ValueError("known allocation has an unexpected job name")
    return confirmed


def input_scope(plan):
    return {"input_counts": plan.get("input_counts", {}),
            "input_inventory": plan.get("input_inventory", []),
            "excluded_diagnostics": plan.get("excluded_diagnostics", {}),
            "scientific_result": False, "semantic_acceptance": "not_established",
            "completion_scope": "eligible_frozen_queue_only",
            "full_corpus_completion": "not_established"}


def inspect(plan, observed, *, runtime=None, persist=False):
    shards, allocation_rows, tracked, unresolved = [], {}, set(), 0
    total_attempts = 0
    for shard in plan["shards"]:
        history = attempts(shard)
        allocated = 0
        progress = report(shard)
        last_job, last_path, last_intent = None, None, None
        live_history = False
        unresolved_history = False
        for path, intent in history:
            job = identify(path, intent, observed, persist=persist)
            receipt_path = path / "receipt.json"
            receipt = read(receipt_path) if receipt_path.exists() else {}
            if not job and receipt.get("disposition") == "policy_refused":
                raise ValueError("recorded Slurm policy refusal requires inspection: " + receipt["stderr"])
            if not job and receipt.get("disposition") == "capacity_refused":
                continue
            allocated += 1
            if job:
                tracked.add(job)
                if job in observed["accounting"]:
                    allocation_rows[job] = observed["accounting"][job]
                live_history |= job in observed["jobs"]
            if not job or job not in observed["jobs"] and job not in observed["accounting"]:
                unresolved += 1
            if not job or job not in observed["jobs"] and observed["accounting"].get(job, {}).get("state") not in TERMINAL:
                unresolved_history = True
            last_job, last_path, last_intent = job, path, intent
        total_attempts += allocated
        if allocated > shard["max_allocations"]:
            raise ValueError("saved submissions exceed a shard's frozen budget")
        item = {"id": shard["id"], "cells": shard["cells"], "allocations": allocated,
                "submission_attempts": len(history),
                "max_allocations": shard["max_allocations"], "last_job_id": last_job,
                "progress": progress}
        row = observed["accounting"].get(last_job, {})
        if live_history:
            item.update(state="live", reason="preserving current allocation")
        elif allocated and unresolved_history:
            item.update(state="awaiting_accounting", reason="submission or terminal ownership is unresolved; no resubmission")
        elif allocated:
            recovered = last_path / "reconciled.json"
            if recovered.exists():
                saved = read(recovered)
                if saved["job_id"] != last_job:
                    raise ValueError("reconciliation receipt belongs to another allocation")
                progress = saved["progress"]
            elif persist:
                emit({"state": "reconciling", "shard": shard["id"], "terminal_job_id": last_job})
                progress = runtime.reconcile_output(Path(shard["directory"]) / "results")
                save(recovered, {"job_id": last_job, "scheduler": row, "progress": progress,
                                 "reconciled_at_epoch": time.time()})
            else:
                item.update(state="awaiting_reconciliation", reason="terminal allocation needs saved-output reconciliation")
            item["progress"] = progress
            if "state" not in item:
                if progress.get("status") in EXHAUSTED:
                    item.update(state="exhausted", reason=progress["status"])
                elif (progress.get("status") == "not_started" and row.get("state") in STARTUP_FAILURE_STATES
                      and (row.get("elapsed_seconds") or STARTUP_FAILURE_SECONDS + 1) <= STARTUP_FAILURE_SECONDS
                      and not (Path(shard["directory"]) / "results/control/index.sqlite").exists()
                      and allocated < shard["max_allocations"]):
                    # Died during startup checks, before any judging (e.g. a node could not look up the user for
                    # the quota check). Retry within the shard's existing allocation budget; never beyond it.
                    item.update(state="eligible", reason="previous allocation failed at startup before any judging; retry within budget")
                elif progress.get("status") == "not_started" or completed(progress) <= last_intent["completed_before"]:
                    item.update(state="blocked", reason="terminal allocation saved no new outcomes; inspect model startup/runtime logs")
                elif progress.get("states", {}).get("running", 0):
                    item.update(state="blocked", reason="reconciliation left running tasks; ownership is unresolved")
                elif not progress.get("states", {}).get("pending", 0):
                    item.update(state="blocked", reason="unfinished output has no runnable pending tasks")
                elif allocated >= shard["max_allocations"]:
                    item.update(state="budget_exhausted", reason="pending work remains after the shard's finite allocation budget")
                else:
                    item.update(state="eligible", reason="previous owner terminal; only remaining pending work may resume")
        elif progress.get("status") in EXHAUSTED:
            item.update(state="exhausted", reason=progress["status"])
        elif (Path(shard["directory"]) / "results/control/index.sqlite").exists():
            item.update(state="blocked", reason="existing results lack sender allocation ownership; reconcile explicitly")
        else:
            item.update(state="eligible", reason="frozen unstarted shard")
        shards.append(item)
    if total_attempts > plan["maximum_allocations"]:
        raise ValueError("saved submissions exceed the frozen overall allocation budget")
    # A successful sbatch reply may precede squeue visibility. Account for every
    # unresolved marker. The user explicitly scoped the 200 cap to THIS plan;
    # other user allocations are reported and preserved, never used as a gate.
    accounted_nonterminal = {job for job in tracked if job not in observed["jobs"]
                            and job in observed["accounting"]
                            and observed["accounting"][job]["state"] not in TERMINAL}
    owned_live = len(tracked & observed["jobs"].keys())
    reserved = owned_live + unresolved + len(accounted_nonterminal)
    node_seconds = gpu_seconds = 0
    for row in allocation_rows.values():
        elapsed, nodes = row["elapsed_seconds"], row["allocated_nodes"]
        if elapsed is not None and nodes is not None:
            node_seconds += elapsed * nodes
        gpu = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", row["allocated_tres"])
        if elapsed is not None and gpu:
            gpu_seconds += elapsed * int(gpu[1])
    return {**input_scope(plan), "timestamp_utc": stamp(), "captured_at_epoch": observed["captured_at_epoch"],
            "deadline_epoch": plan["deadline_epoch"], "shards": shards,
            "allocations_attempted": total_attempts, "maximum_allocations": plan["maximum_allocations"],
            "all_user_queued_or_active": len(observed["jobs"]), "plan_queued_or_active": owned_live,
            "other_user_allocations": [row for job, row in observed["jobs"].items() if job not in tracked],
            "capacity_reserved": reserved,
            "ambiguous_or_missing_allocations": unresolved, "max_inflight": plan["max_inflight"],
            "accounting": list(allocation_rows.values()),
            "accounted_node_hours": node_seconds / 3600, "accounted_gpu_hours": gpu_seconds / 3600,
            "accounting_note": "reported sacct usage only; missing/unknown usage is not an estimate",
            "allocation_ceiling_node_hours": total_attempts * 5,
            "allocation_ceiling_gpu_hours": total_attempts * 20}


def excluded_nodes(plan) -> str:
    """Nodes to keep bouts off (e.g. one that cannot resolve the user), read fresh at every submission from
    <workspace>/control/gemma-exclude-nodes: one Slurm node list such as hkn0515 or hkn[0515,0601]."""
    path = Path(plan["workspace"]) / "control/gemma-exclude-nodes"
    if not path.is_file():
        return ""
    value = path.read_text().strip()
    if value and not re.fullmatch(r"[A-Za-z0-9\[\],-]+", value):
        raise ValueError(f"invalid node list in {path}")
    return value


def submit(root, plan, state, shard, item):
    if hashlib.sha256((Path(shard["directory"]) / "config.json").read_bytes()).hexdigest() != shard["config_sha256"]:
        raise ValueError("frozen shard configuration changed")
    number = item["submission_attempts"] + 1
    directory = Path(shard["directory"])
    attempt = directory / "submissions" / f"attempt-{number:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    token = hashlib.sha256((str(root) + state["plan_sha256"] + shard["id"] + str(number)).encode()).hexdigest()[:32]
    comment = "geodml-gemma-v4:" + token
    (directory / "logs").mkdir(exist_ok=True)
    exclude = excluded_nodes(plan)
    command = ["sbatch", "--parsable", f"--account={plan['account']}", "--partition=accelerated",
               *([f"--exclude={exclude}"] if exclude else []),
               "--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--gres=gpu:4", "--mem=0",
               "--exclusive", "--time=05:00:00", "--no-requeue", "--job-name=geodml-gemma-v4-bout",
               f"--comment={comment}", f"--chdir={plan['repository']}", "--open-mode=append",
               f"--output={directory / 'logs/slurm-%j.out'}", f"--error={directory / 'logs/slurm-%j.err'}",
               str(directory / "run.sh")]
    intent = {"shard_id": shard["id"], "attempt": number, "comment": comment,
              "created_at_epoch": time.time(), "git_commit": plan["git_commit"],
              "config_sha256": shard["config_sha256"], "plan_sha256": state["plan_sha256"],
              "completed_before": completed(item["progress"]), "command": command}
    save(attempt / "intent.json", intent)
    emit({"state": "submitting", "shard": shard["id"], "attempt": number, "comment": comment})
    try:
        response = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        receipt = {"returncode": response.returncode, "stdout": response.stdout, "stderr": response.stderr}
        match = re.fullmatch(r"(\d+)(?:;[^\s;]+)?\s*", response.stdout)
        if response.returncode == 0 and match:
            receipt["job_id"] = match[1]
        elif response.returncode != 0 and not response.stdout.strip() and re.search(
                r"MaxSubmitJob|maximum number of jobs",
                response.stderr, re.IGNORECASE):
            # These are explicit scheduler rejections, not lost replies. They
            # spend no allocation budget; retry only after the next 600s poll.
            receipt["disposition"] = "capacity_refused"
        elif response.returncode != 0 and not response.stdout.strip() and "Job violates accounting/QOS policy" in response.stderr:
            receipt["disposition"] = "policy_refused"
    except (OSError, subprocess.TimeoutExpired) as error:
        receipt = {"returncode": None, "stdout": str(getattr(error, "stdout", None) or ""),
                   "stderr": str(getattr(error, "stderr", None) or ""), "error": str(error)}
    save(attempt / "receipt.json", {**receipt, "finished_at_epoch": time.time()})
    return receipt


def emit_status(root, value):
    """Keep tmux logs bounded; the full evidence remains in status.json."""
    emit({key: value.get(key) for key in (
        "state", "reason", "allocations_attempted", "maximum_allocations",
        "plan_queued_or_active", "all_user_queued_or_active", "capacity_reserved",
        "ambiguous_or_missing_allocations", "accounted_node_hours", "accounted_gpu_hours")}
        | {"shards_by_state": dict(Counter(s["state"] for s in value.get("shards", []))),
           "status_path": str(root / "sender/status.json")})


def finish(root, value, state, reason=None):
    value = {**value, "timestamp_utc": stamp(), "state": state, "terminal_summary": True}
    if reason:
        value["reason"] = reason
    save(root / "sender/status.json", value)
    save(root / "sender/summary.json", value)
    emit_status(root, value)
    return 0 if state == "finished" else 130 if state == "interrupted" else 2


def send(root):
    from analysis.scripts import horeka_gemma_v4 as runtime
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("run the sender on a login host outside an allocation")
    plan = runtime.checked_plan(root)
    validate(plan)
    with sender_lock(plan["workspace"], lock_name(plan)):
        state = state_file(root, plan)
        since = datetime.fromtimestamp(state["created_at_epoch"] - 86400).strftime("%Y-%m-%d")
        value = {**input_scope(plan), "shards": [], "allocations_attempted": None}
        reserved = False
        try:
            while True:
                plan = runtime.checked_plan(root)
                validate(plan)
                state_file(root, plan)
                observed = snapshot(since)
                audit = root / "sender" / f"check-{time.time_ns()}"
                save(audit / "scheduler.json", observed)
                value = inspect(plan, observed, runtime=runtime, persist=True)
                if time.time() >= plan["deadline_epoch"]:
                    return finish(root, value, "expired", "finite sender deadline reached; live jobs preserved")
                eligible = [s for s in value["shards"] if s["state"] == "eligible"]
                budget = plan["maximum_allocations"] - value["allocations_attempted"]
                capacity = plan["max_inflight"] - value["capacity_reserved"]
                if eligible and budget > 0 and capacity > 0:
                    with submission_lock(plan["workspace"]):
                        room = GLOBAL_MAX_GEMMA_BOUTS - gemma_bouts_in_queue()
                        value["global_room"] = room
                        if room <= 0:
                            value["waiting_for_global_room"] = (
                                f"{GLOBAL_MAX_GEMMA_BOUTS} Gemma bouts already queued or running; next check in 600s")
                        else:
                            runtime.storage(plan, audit)
                            checked_at = time.time()
                            if not reserved:
                                runtime.reserve(plan, root)
                                reserved = True
                            by_id = {s["id"]: s for s in plan["shards"]}
                            submitted = []
                            for item in eligible[:max(0, min(budget, capacity, room))]:
                                if time.time() >= plan["deadline_epoch"]:
                                    break
                                if time.time() - checked_at >= 120:
                                    runtime.storage(plan, audit / f"refresh-{time.time_ns()}")
                                    checked_at = time.time()
                                if time.time() >= plan["deadline_epoch"]:
                                    break
                                receipt = submit(root, plan, state, by_id[item["id"]], item)
                                submitted.append({"shard": item["id"], "job_id": receipt.get("job_id"),
                                                  "disposition": receipt.get("disposition", "submitted_or_ambiguous")})
                                item["submission_attempts"] += 1
                                if receipt.get("disposition") == "policy_refused":
                                    return finish(root, value, "blocked", receipt["stderr"])
                                if receipt.get("disposition") == "capacity_refused":
                                    item.update(state="capacity_wait", reason="scheduler submit limit rejected this request; next check in 600s")
                                    break
                                item["allocations"] += 1
                                item.update(state="awaiting_accounting", last_job_id=receipt.get("job_id"),
                                            reason="submission recorded; awaiting the next scheduler snapshot")
                            value["submitted_this_pass"] = submitted
                            spent = sum(s["disposition"] != "capacity_refused" for s in submitted)
                            value["allocations_attempted"] += spent
                            value["capacity_reserved"] += spent
                            value["allocation_ceiling_node_hours"] = value["allocations_attempted"] * 5
                            value["allocation_ceiling_gpu_hours"] = value["allocations_attempted"] * 20
                if time.time() >= plan["deadline_epoch"]:
                    return finish(root, value, "expired", "finite sender deadline reached; live jobs preserved")
                pending_live = any(s["state"] in {"live", "awaiting_accounting"} for s in value["shards"])
                submitted_now = bool(value.get("submitted_this_pass"))
                if not pending_live and not submitted_now:
                    if all(s["state"] == "exhausted" for s in value["shards"]):
                        failures = any(s["progress"].get("status") == "finished_with_failures" for s in value["shards"])
                        return finish(root, value, "finished_with_failures" if failures else "finished")
                    if not eligible or budget <= 0:
                        blocked = any(s["state"] == "blocked" for s in value["shards"])
                        return finish(root, value, "blocked" if blocked else "budget_exhausted")
                value["state"] = "monitoring"
                save(root / "sender/status.json", value)
                emit_status(root, value)
                time.sleep(min(plan["poll_seconds"], max(0, plan["deadline_epoch"] - time.time())))
        except KeyboardInterrupt:
            return finish(root, value, "interrupted", "sender stopped; submission markers and live jobs preserved")
        except Exception as error:
            finish(root, value, "blocked", str(error))
            raise


def status(root):
    from analysis.scripts import horeka_gemma_v4 as runtime
    plan = runtime.checked_plan(root)
    validate(plan)
    state_path = root / "sender/state.json"
    created = read(state_path)["created_at_epoch"] if state_path.exists() else time.time()
    since = datetime.fromtimestamp(created - 86400).strftime("%Y-%m-%d")
    value = inspect(plan, snapshot(since), persist=False)
    saved = root / "sender/status.json"
    value["sender_status"] = read(saved).get("state") if saved.exists() else "not_started"
    emit(value)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["send", "status"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return send(args.output.resolve()) if args.command == "send" else status(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
