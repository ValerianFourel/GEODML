#!/usr/bin/env python3
"""Collect scheduler, saved progress and bounded log evidence without changing runs.

This operational snapshot does not validate dataset record references or infer
unique corpus completion from overlapping bout manifests. Standard library only.
"""
import getpass
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


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
            row["state"] = row["state"].split()[0]
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
                   "writer_id": direct.get("writer_id"), "attempt": str(attempt)}
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
    for root in sorted((workspace / "reviews").glob("gemma-si-v4*")):
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
            for name, inputs in (("v4-pass1", "v4-inputs"), ("v3-bridge", "v3-inputs"),
                                 ("constructed", "constructed-inputs"), ("v4-pass2", "v4-inputs"), ("v4-pass3", "v4-inputs")):
                values = config.get("inventories", {}).get(inputs, {}).get("unique_tasks")
                values = list(values.values()) if isinstance(values, dict) else [values]
                expected = sum(values) if all(count(v) is not None for v in values) else None
                entry = {"attempt": attempt.name, "pass": name, "expected": expected}
                path = attempt / "trial" / name / "control/index.sqlite"
                try:
                    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
                        db.execute("PRAGMA query_only=ON")
                        counts = Counter()
                        for state, raw in db.execute("SELECT state,result FROM tasks"):
                            label = state
                            if state in {"done", "saved"}:
                                value = json.loads(raw) if raw else {}
                                label += "_" + ("ok" if value.get("ok") is True else "failed" if value.get("ok") is False else "unknown")
                            counts[label] += 1
                    entry.update(total=sum(counts.values()), counts=dict(counts))
                except (OSError, sqlite3.Error, ValueError, AttributeError) as error:
                    entry["unavailable"] = str(error)
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
    for job, row in report["jobs"].items():
        if row["name"] == "geodml-gemma-si-v4":
            lines.append("GEMMA_JOB " + json.dumps(row, sort_keys=True))
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
    lines.extend("GEMMA_PREPARATION " + json.dumps(item, sort_keys=True) for item in report["gemma"])
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
    workspace = Path(sys.argv[1])
    report = collect(workspace, sys.argv[2] if len(sys.argv) > 2 else None)
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


if __name__ == "__main__":
    main()
