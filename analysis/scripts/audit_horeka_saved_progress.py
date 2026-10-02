#!/usr/bin/env python3
"""Collect scheduler, saved progress and bounded log evidence without changing runs.

This operational snapshot does not validate dataset record references or infer
unique corpus completion from overlapping bout manifests. Standard library only.
"""
import argparse
import fcntl
import gzip
import getpass
import hashlib
import json
import re
import sqlite3
import subprocess
import tarfile
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def unique_qwen_cells(division_dir, division, attempts):
    """Count division identities once from existing ledger files; never create locks."""
    report = {"scope": "unique cells in CURRENT division, including spill once",
              "evidence": "ledger states; result checksums not revalidated; per-stripe capture, not an atomic cluster snapshot"}
    try:
        wanted = set()
        for bout in division["bouts"]:
            path = division_dir / "bouts" / ("bout-%04d" % bout["number"]) / "bout.json"
            value = json.loads(path.read_text())
            for key in ("primary_fingerprints", "spill_fingerprints"):
                ids = value[key]
                if not isinstance(ids, list) or any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x) for x in ids):
                    raise ValueError("invalid planned fingerprint list")
                wanted.update(ids)
        if not wanted:
            raise ValueError("no division identities available")
        sizes = {r["ledger_stripes"] for r in attempts if r.get("ledger_stripes") is not None}
        if len(sizes) != 1:
            raise ValueError("one observed ledger stripe count required")
        stripes = sizes.pop()
        if not 1 <= stripes <= 4096:
            raise ValueError("invalid ledger stripe count")
        dataset = Path(division["dataset_root"]).resolve()
        if any(r.get("dataset_root") and Path(r["dataset_root"]).resolve() != dataset for r in attempts):
            raise ValueError("attempt dataset roots differ from CURRENT division")
        ledger = dataset / "control/task-ledger"
        if not (ledger / "events").is_dir() or not (ledger / "locks").is_dir():
            raise ValueError("ledger directories unavailable")
        by_stripe = {}
        for fingerprint in wanted:
            by_stripe.setdefault(int(fingerprint[:16], 16) % stripes, set()).add(fingerprint)
        counts, issues = Counter(), []
        for stripe, members in sorted(by_stripe.items()):
            path = ledger / "events" / ("stripe-%04d.jsonl" % stripe)
            lock = ledger / "locks" / ("stripe-%04d.lock" % stripe)
            latest = {}
            try:
                # Existing writers use exclusive flock on the same inode. Busy or absent
                # locks produce unknown counts, never invented unattempted cells.
                with lock.open("rb") as handle:
                    fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    if path.exists():
                        with path.open("rb") as stream:
                            for line in stream:
                                row = json.loads(line)
                                if row.get("format_version") != "geodml-agentic-task-event-v1":
                                    raise ValueError("unknown ledger event format")
                                fingerprint = row.get("fingerprint")
                                if fingerprint in members:
                                    latest[fingerprint] = row["state"]
                for fingerprint in members:
                    state = latest.get(fingerprint, "unattempted")
                    if state not in {"completed", "terminal_failed", "claimed", "running", "result_saved", "checkpointed", "retryable", "unattempted"}:
                        state = "unknown"
                    counts[state] += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                counts["unknown"] += len(members)
                issues.append(str(path) + ": " + str(error))
        completed = counts["completed"]
        report.update(total=len(wanted), ledger_stripes=stripes, recorded_finished=completed,
                      known_unfinished=len(wanted) - completed - counts["unknown"],
                      unknown=counts["unknown"], states=dict(counts), issues=issues)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["unavailable"] = str(error)
    return report


def qwen_counts(workspace):
    """Read only division identities and ledger-layout evidence for a quick count."""
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CURRENT Qwen division; cells, not Slurm jobs or API requests"}

    def read(path):
        before = path.stat()
        value = json.loads(path.read_text())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("file changed during capture: " + str(path))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object: " + str(path))
        return value

    try:
        workspace = workspace.resolve(strict=True)
        directory = Path((workspace / "qwen-bouts/CURRENT").read_text().strip())
        if not directory.is_absolute():
            raise ValueError("CURRENT must contain an absolute division path")
        report["division"] = str(directory)
        division = read(directory / "division.json")
        if (division.get("format_version") != "geodml-horeka-qwen-bouts-v1"
                or division.get("model") != "qwen38"):
            raise ValueError("CURRENT is not a supported Qwen bout division")
        if (not isinstance(division.get("bouts"), list)
                or any(not isinstance(bout, dict) or type(bout.get("number")) is not int
                       or bout["number"] < 1 for bout in division["bouts"])):
            raise ValueError("invalid Qwen bout list")
        attempts = []
        for bout in division["bouts"]:
            root = directory / "bouts" / ("bout-%04d" % bout["number"])
            saved_attempts = sorted((root / "attempts").glob("job*"))
            if not saved_attempts:
                continue
            config = read(root / "config.json")
            if not isinstance(config.get("dataset_root"), str) or not config["dataset_root"]:
                raise ValueError("attempt config lacks dataset_root: " + str(root / "config.json"))
            attempts.append({"dataset_root": config["dataset_root"], "ledger_stripes": None})
            for attempt in saved_attempts:
                if not attempt.is_dir():
                    continue
                for name in ("results/run_manifest.json", "bout-result.json"):
                    path = attempt / name
                    if not path.exists():
                        continue
                    direct = read(path).get("direct_dataset") or {}
                    if not isinstance(direct, dict):
                        raise ValueError("invalid direct_dataset: " + str(path))
                    stripes = direct.get("ledger_stripes")
                    if stripes is not None and (type(stripes) is not int or stripes < 1):
                        raise ValueError("invalid ledger stripe count: " + str(path))
                    attempts.append({"dataset_root": config.get("dataset_root"),
                                     "ledger_stripes": stripes})
        report["qwen_unique_cells"] = unique_qwen_cells(directory, division, attempts)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["qwen_unique_cells"] = {"unavailable": str(error)}
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def gemma_cells(inputs, rows):
    """Join frozen cells to one attempt's saved task results without merging runs."""
    manifest = json.loads((inputs / "manifest.json").read_text())
    path = inputs / "cells.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"]["cells.jsonl.gz"]:
        raise ValueError("frozen cells checksum mismatch")
    tasks = {tid: (kind, state, json.loads(raw) if raw else {}) for tid, kind, state, raw in rows}
    counts, unfinished, kinds = Counter(), [], Counter()
    for kind, state, result in tasks.values():
        outcome = "ok" if result.get("ok") is True else "failed" if result.get("ok") is False else "unknown"
        kinds[(kind, state, outcome)] += 1
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if not line.strip():
                continue
            cell = json.loads(line)
            mapper = tasks.get(cell.get("map_task_id"))
            ids = [s.get("dependency_id", s.get("judge_task_id")) for s in cell.get("sources", [])]
            sources = [tasks.get(tid) for tid in ids]
            relevant = ([mapper] if cell.get("map_task_id") else []) + sources
            def scored(task):
                if not task or task[1] != "done" or task[2].get("ok") is not True:
                    return False
                parsed = task[2].get("parsed_output", {})
                return parsed.get("status") == "scored" and type(parsed.get("importance")) is int and 0 <= parsed["importance"] <= 5
            if any(t and t[1] == "done" and t[2].get("ok") is False for t in relevant):
                status = "failed"
            elif mapper and mapper[1] == "done" and mapper[2].get("ok") is True and mapper[2].get("parsed_output", {}).get("eligibility") in {"map_unusable", "global_absence_only", "no_substantive_content"}:
                status = "not_assessable"
            elif sources and all(scored(t) for t in sources) and (not cell.get("map_task_id") or (mapper and mapper[1] == "done" and mapper[2].get("ok") is True and mapper[2].get("parsed_output", {}).get("eligibility") == "eligible")):
                status = "finished"
            elif any(t and t[1] == "blocked" for t in relevant):
                status = "blocked"
            elif any(t and t[1] in {"running", "saved"} for t in relevant):
                status = "in_progress"
            elif any(t is None for t in relevant) or not relevant:
                status = "missing_task_evidence"
            elif any(t and t[1] in {"pending", "waiting"} for t in relevant):
                status = "pending"
            else:
                status = "unresolved"
            counts[status] += 1
            if status != "finished":
                unfinished.append({"cell_id": cell["cell_id"], "model": cell.get("model"), "state": status,
                    "map_task_id": cell.get("map_task_id"),
                    "errors": [t[2].get("error") for t in relevant if t and t[2].get("error")],
                    "unresolved_source_ids": [tid for tid, task in zip(ids, sources) if not scored(task)]})
    return {"cell_counts": dict(counts), "total_cells": sum(counts.values()),
            "unfinished_cells": unfinished,
            "task_counts": [{"kind": k, "state": s, "outcome": o, "count": n} for (k, s, o), n in sorted(kinds.items())],
            "cell_count_evidence": "frozen-cell join to saved index; sealed result references not revalidated"}


def collect(workspace, account=None):
    workspace = workspace.resolve(strict=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "scheduler and saved progress; no dataset checksum revalidation",
              "commands": {}, "files": {}, "qwen": [], "gemma": [], "issues": []}

    def command(label, argv):
        try:
            p = subprocess.run(argv, text=True, capture_output=True, timeout=30, check=False)
            report["commands"][label] = {"argv": argv, "returncode": p.returncode,
                                         "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode:
                raise ValueError("command returned " + str(p.returncode))
            return p.stdout
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            report["issues"].append(label + ": " + str(error))
            return None

    def saved(path, tail=False):
        key = str(path)
        if key in report["files"]:
            return report["files"][key].get("value")
        try:
            with path.open("rb") as stream:
                before = path.stat()
                if tail:
                    stream.seek(max(0, before.st_size - 65536))
                raw = stream.read(65536 if tail else 2097153)
                after = path.stat()
            record = {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                      "captured_sha256": hashlib.sha256(raw).hexdigest(),
                      "tail_only": tail and before.st_size > len(raw)}
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("file changed during capture; repeat snapshot")
            if not tail and len(raw) > 2097152:
                raise ValueError("metadata exceeds 2 MiB; not read in full")
            value = raw.decode("utf-8", errors="replace") if tail else json.loads(raw)
            if not tail and not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            record["value"] = value
            report["files"][key] = record
            return value
        except (OSError, ValueError) as error:
            report["files"][key] = {"unavailable": str(error)}
            return None

    def count(value):
        return value if type(value) is int and value >= 0 else None

    fields = ["job", "name", "state", "elapsed_seconds", "limit", "exit_code", "start", "end"]
    accounting = command("accounting", ["sacct", "-X", "--user=" + getpass.getuser(),
        "--starttime=2026-09-25", "--noheader", "--parsable2",
        "--format=JobIDRaw,JobName%100,State%40,ElapsedRaw,Timelimit,ExitCode,Start,End"])
    live = command("live_queue", ["squeue", "--me", "--array", "--noheader",
        "--format=%i|%j|%T|%M|%l|%S|%R"])
    jobs = {}
    parsed_scheduler = True
    for raw, keys in ((accounting, fields),
                      (live, ["job", "name", "state", "elapsed", "limit", "start", "reason"])):
        if raw is None:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split("|")
            if parts and parts[-1] == "" and len(parts) == len(keys) + 1:
                parts.pop()
            if len(parts) != len(keys) or not parts[0].strip() or not parts[2].strip():
                report["issues"].append("Unparsed scheduler row: " + line)
                parsed_scheduler = False
                continue
            row = dict(zip(keys, (p.strip() for p in parts)))
            row["state"] = row["state"].split()[0].rstrip("+")
            jobs.setdefault(row["job"], {}).update(row)
    report["jobs"] = jobs
    report["scheduler_complete"] = accounting is not None and live is not None and parsed_scheduler
    current = workspace / "qwen-bouts/CURRENT"
    try:
        division_dir = Path(current.read_text().strip())
        if not division_dir.is_absolute():
            raise ValueError("CURRENT must contain an absolute division path")
        division = saved(division_dir / "division.json") or {}
    except (OSError, ValueError) as error:
        report["issues"].append("CURRENT: " + str(error))
        division_dir, division = None, {}
    report["division"] = str(division_dir) if division_dir else None
    fail_states = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL",
                   "CANCELLED", "PREEMPTED", "DEADLINE", "REVOKED"}
    for bout in division.get("bouts", []):
        number = bout["number"]
        directory = division_dir / "bouts" / ("bout-%04d" % number)
        receipt = saved(directory / "submission.json") or {}
        config = saved(directory / "config.json") or {}
        ids = {p.name[3:] for p in (directory / "attempts").glob("job*") if p.is_dir()}
        submitted = str(receipt.get("stdout", "")).strip().split(";")[0]
        if submitted.isdigit():
            ids.add(submitted)
        ids.update(job for job, row in jobs.items() if row["name"] == "geodml-qwen-bout-%04d" % number)
        for job in sorted(ids or {"unknown"}):
            attempt = directory / "attempts" / ("job" + job)
            manifest = saved(attempt / "results/run_manifest.json") or {}
            result = saved(attempt / "bout-result.json") or {}
            progress = manifest or result
            direct = progress.get("direct_dataset") or {}
            if not isinstance(direct, dict):
                direct = {}
            failed = manifest.get("failed_cell_ids")
            failed = sorted(set(failed)) if isinstance(failed, list) and all(isinstance(x, str) for x in failed) else None
            row = {"bout": number, "job": job, "state": jobs.get(job, {}).get("state", "UNKNOWN"),
                   "requested": count(bout.get("cell_count")), "primary": count(bout.get("primary_cells")),
                   "spill": count(bout.get("spill_cells")), "completed": count(progress.get("completed_count")),
                   "remaining": count(progress.get("remaining_count")),
                   "progress_source": "run_manifest" if manifest else "bout_result" if result else None,
                   "newly_committed": count(direct.get("committed_count")),
                   "reused": count(direct.get("reused_count")), "failed_cell_ids": failed,
                   "explicit_failed": len(failed) if failed is not None else None,
                   "stop_reason": progress.get("stop_reason"), "result": result,
                   "git_commit": config.get("git_commit"), "dataset_root": config.get("dataset_root"),
                   "writer_id": direct.get("writer_id"), "ledger_stripes": count(direct.get("ledger_stripes")), "attempt": str(attempt)}
            if all(row[k] is not None for k in ("requested", "completed", "remaining")):
                if row["completed"] + row["remaining"] != row["requested"]:
                    row["inconsistent_counts"] = True
            row["attention"] = (row["state"] in fail_states or result.get("status") in {"failed", "deadline"}
                                or bool(result.get("validation_error")) or row.get("inconsistent_counts", False)
                                or (row["state"] == "COMPLETED" and row["remaining"] != 0))
            report["qwen"].append(row)
            if row["attention"]:
                saved(attempt / "execution.json")
                for path in (directory / ("slurm-" + job + ".err"),
                             directory / ("slurm-" + job + ".out"), attempt / "server.log"):
                    saved(path, tail=True)
    report["unmatched_qwen_jobs"] = sorted(job for job, r in jobs.items()
        if r["name"].startswith("geodml-qwen-bout-") and job not in {r["job"] for r in report["qwen"]})
    report["qwen_unique_cells"] = unique_qwen_cells(division_dir, division, report["qwen"])
    roots = {p.parent for pattern in ("gemma*/config.json", "gemma*/run/config.json")
             for p in (workspace / "reviews").glob(pattern)}
    for root in sorted(roots):
        if not root.is_dir():
            continue
        config = saved(root / "config.json") or {}
        repair = saved(root / "HASH_REPAIR_VERIFIED.json")
        item = {"root": str(root), "walltime": config.get("walltime"), "repair_receipt": repair,
                "allocation_attempted": (root / "ALLOCATION_ATTEMPTED").exists(), "passes": []}
        for pattern in ("attempts/job*/trial-result.json", "attempts/job*/trial/summary.json"):
            for path in sorted(root.glob(pattern)):
                saved(path)
        for attempt in sorted(root.glob("attempts/job*")):
            saved(attempt / "execution.json")
            passes = [("v4-pass1", "v4-inputs")] if config.get("workload_mode") == "selected-v4-cells" else [
                ("v4-pass1", "v4-inputs"), ("v3-bridge", "v3-inputs"), ("constructed", "constructed-inputs"),
                ("v4-pass2", "v4-inputs"), ("v4-pass3", "v4-inputs")]
            for name, inputs in passes:
                values = config.get("inventories", {}).get(inputs, {}).get("unique_tasks")
                values = list(values.values()) if isinstance(values, dict) else [values]
                expected = sum(values) if all(count(v) is not None for v in values) else None
                entry = {"attempt": attempt.name, "pass": name, "expected": expected}
                path = attempt / "trial" / name / "control/index.sqlite"
                try:
                    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
                        db.execute("PRAGMA query_only=ON")
                        db.execute("BEGIN")
                        counts = Counter()
                        for state, raw in db.execute("SELECT state,result FROM tasks"):
                            label = state
                            if state in {"done", "saved"}:
                                value = json.loads(raw) if raw else {}
                                label += "_" + ("ok" if value.get("ok") is True else "failed" if value.get("ok") is False else "unknown")
                            counts[label] += 1
                        columns = {r[1] for r in db.execute("PRAGMA table_info(tasks)")}
                        if {"id", "kind"} <= columns:
                            task_rows = db.execute("SELECT id,kind,state,result FROM tasks").fetchall()
                            entry.update(gemma_cells(root / inputs, task_rows))
                    entry.update(total=sum(counts.values()), counts=dict(counts))
                except (OSError, sqlite3.Error, ValueError, AttributeError) as error:
                    entry["unavailable"] = str(error)
                summaries = sorted((attempt / "trial" / name / "reports").glob("*/summary.json"))
                if summaries:
                    entry["latest_saved_summary"] = saved(summaries[-1])
                item["passes"].append(entry)
            saved(attempt / "server.log", tail=True)
        for pattern in ("console-job*.log", "slurm-*.err", "slurm-*.out"):
            for path in sorted(root.glob(pattern)):
                saved(path, tail=True)
        report["gemma"].append(item)
    troubled = [job for job, row in jobs.items() if job.isdigit() and row["state"] in fail_states
                and (row["name"].startswith("geodml-qwen-bout-") or row["name"] == "geodml-gemma-si-v4")]
    if troubled:
        command("failed_job_steps", ["sacct", "--noheader", "--parsable2", "--jobs=" + ",".join(troubled),
            "--format=JobID%50,State%40,ExitCode,DerivedExitCode,Elapsed,MaxRSS,MaxVMSize,NodeList%100"])
    for job, row in jobs.items():
        if row["name"] == "geodml-gemma-si-v4" and row["state"] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            command("gemma_" + job, ["scontrol", "show", "job", job, "-o"])
    command("disk_bytes", ["df", "-h", str(workspace)])
    command("disk_inodes", ["df", "-i", str(workspace)])
    command("work_quota", ["/usr/lpp/mmfs/bin/mmlsquota", "-u", getpass.getuser(),
                            "--block-size", "G", "-C", "hkn.scc.kit.edu", "hkfs-work"])
    if account:
        command("account_queue", ["squeue", "--account=" + account, "--array", "--noheader", "--format=%i|%j|%T"])
        command("home_quota", ["/usr/lpp/mmfs/bin/mmlsquota", "-j", account,
                                "--block-size", "G", "-C", "hkn.scc.kit.edu", "hkfs-home"])
    report["pending_reasons"] = dict(Counter(r.get("reason", "unknown") for r in jobs.values() if r["state"] == "PENDING"))
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def render(report):
    lines = ["SNAPSHOT " + report["started_utc"], "DIVISION " + str(report["division"]),
             "Scheduler complete: " + str(report["scheduler_complete"])]
    for label, accept in (("QWEN", lambda n: n.startswith("geodml-qwen-bout-")),
                          ("GEMMA", lambda n: n == "geodml-gemma-si-v4")):
        counts = Counter(r["state"] for r in report["jobs"].values() if accept(r["name"]))
        lines.append(label + " allocation states " + json.dumps(dict(counts), sort_keys=True))
    if not report["scheduler_complete"]:
        lines.append("INCOMPLETE SCHEDULER EVIDENCE: state counts above cover only available rows.")
    lines.append("PENDING_REASONS " + json.dumps(report.get("pending_reasons", {}), sort_keys=True))
    for job, row in report["jobs"].items():
        if row["name"] == "geodml-gemma-si-v4":
            lines.append("GEMMA_JOB " + json.dumps(row, sort_keys=True))
    lines.append("QWEN_UNIQUE_CELLS " + json.dumps(report["qwen_unique_cells"], sort_keys=True))
    failed = [r for r in report["qwen"] if r["state"] == "FAILED"]
    groups = Counter("unknown_checkpoint" if r["completed"] is None or r["remaining"] is None or r.get("inconsistent_counts")
                     else "zero_reported" if r["completed"] == 0
                     else "all_requested_reported" if r["remaining"] == 0 else "partial_reported" for r in failed)
    lines.append("FAILED QWEN saved progress " + json.dumps(dict(groups), sort_keys=True))
    lines.append("bout job state completed/requested new reused explicit_failed remaining exit stop_reason")
    for r in report["qwen"]:
        if r["attention"]:
            val = lambda k: "?" if r[k] is None else str(r[k])
            lines.append(f"{r['bout']:04d} {r['job']} {r['state']} {val('completed')}/{val('requested')} "
                         f"{val('newly_committed')} {val('reused')} {val('explicit_failed')} {val('remaining')} "
                         f"{report['jobs'].get(r['job'], {}).get('exit_code', '?')} {r['stop_reason']}")
    lines.append("UNMATCHED_QWEN_JOBS " + json.dumps(report["unmatched_qwen_jobs"]))
    for item in report["gemma"]:
        lines.append("GEMMA_PREPARATION " + item["root"])
        for run in item["passes"]:
            lines.append("GEMMA_PASS " + json.dumps({k: run[k] for k in ("attempt", "pass", "expected", "total", "counts", "cell_counts", "total_cells", "unavailable") if k in run}, sort_keys=True))
            for cell in run.get("unfinished_cells", [])[:20]:
                lines.append("GEMMA_UNFINISHED_CELL " + json.dumps(cell, sort_keys=True))
            if len(run.get("unfinished_cells", [])) > 20:
                lines.append("Additional unfinished cells are listed in audit.json.")
    for path, record in report["files"].items():
        if path.endswith((".err", "server.log")) and isinstance(record.get("value"), str):
            lines.append("LOG_TAIL " + path)
            lines.extend(record["value"].splitlines()[-8:])
    lines.extend("UNAVAILABLE " + issue for issue in report["issues"])
    for key in ("account_queue", "disk_bytes", "disk_inodes", "work_quota", "home_quota"):
        captured = report["commands"].get(key, {})
        if captured.get("returncode") == 0:
            if key == "account_queue":
                lines.append("ACCOUNT_QUEUE_JOBS " + str(len({line.split('|')[0] for line in captured["stdout"].splitlines() if line.strip()})))
            else:
                lines.append(key.upper() + "\n" + captured["stdout"].strip())
    lines += ["? means unavailable, not zero. Counts are saved checkpoint reports.",
              "Remaining includes failures, busy and unattempted work; it is not a failure count.",
              "Completed includes reuse/spill overlap. Do not sum it as unique corpus completion.",
              "Sources unchanged. No claims released, tasks retried, jobs changed or data uploaded.",
              "Full saved metadata and up to 64 KiB per selected log are in audit.json."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("account", nargs="?")
    parser.add_argument("--qwen-counts-only", action="store_true",
                        help="print current division cell counts without scheduler, logs, Gemma or output files")
    args = parser.parse_args()
    workspace = args.workspace
    if args.qwen_counts_only:
        report = qwen_counts(workspace)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if report["qwen_unique_cells"].get("unavailable") else 0
    report = collect(workspace, args.account)
    output = Path(tempfile.mkdtemp(prefix="horeka-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") ,
                                   dir=workspace / "reviews"))
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = render(report)
    (output / "summary.txt").write_text(summary)
    archive = output / "audit.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        for name in ("audit.json", "summary.txt"):
            stream.add(output / name, arcname=name)
    print(summary, end="")
    print("AUDIT_DIRECTORY", output)
    print("AUDIT_ARCHIVE", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
