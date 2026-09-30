#!/usr/bin/env python3
"""Export saved Gemma diagnostics and optionally upload one verified private-Hub ZIP.

Login-host CPU/I/O only. Never allocate, infer, alter runs or update shared claims.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import HubStore, atomic
from analysis.interpretability.pipeline.agentic_hours import canonical

ROOT_FILES = ("config.json", "inputs.json", "run.sh", "quota-before-fresh20.json")
ATTEMPT_FILES = ("boundary.json", "execution.json", "trial-result.json", "schema-check.json",
                 "serving-profile.json", "server.log", "server.log.runtime.json", "gpu.csv", "nvidia-smi.txt")
PASS_FILES = ("tasks.jsonl", "cells.jsonl", "results.jsonl", "audit.jsonl", "summary.json", "example.json")


def collect(runs):
    """Allowlist diagnostic artifacts, retaining raw bytes and reporting missing evidence."""
    files, report, jobs = {}, {"scientific_result": False, "attempts": [], "missing": []}, set()
    if len({p.name for p in runs}) != len(runs):
        raise ValueError("run directory names must be distinct")
    def read(run, relative, *, required=False):
        path = run / relative
        if not path.exists():
            if required:
                report["missing"].append(f"{run.name}/{relative}")
            return None
        if not path.is_file() or not path.resolve().is_relative_to(run.resolve()) or path.is_symlink():
            raise ValueError(f"refusing redirected artifact: {path}")
        raw = path.read_bytes()
        files[f"{run.name}/{relative}"] = raw
        return raw
    for run in runs:
        if not run.is_dir():
            raise ValueError(f"run not found: {run}")
        for name in ROOT_FILES:
            read(run, name, required=name in {"config.json", "inputs.json"})
        for path in sorted(run.glob("console-job*.log")):
            read(run, path.name)
        attempts = sorted((run / "attempts").glob("job*"))
        if not attempts:
            report["missing"].append(f"{run.name}/attempts/job*")
        for attempt in attempts:
            job = attempt.name.removeprefix("job")
            if not job.isdecimal():
                raise ValueError(f"invalid job directory: {attempt.name}")
            jobs.add(job)
            prefix = f"attempts/{attempt.name}"
            item = {"run": run.name, "job_id": job, "passes": {}}
            for name in ATTEMPT_FILES + ("trial/summary.json", "trial/comparison.json"):
                raw = read(run, f"{prefix}/{name}", required=name in {"trial-result.json", "trial/summary.json"})
                if raw and name in {"trial-result.json", "server.log.runtime.json"}:
                    item[name] = json.loads(raw)
            for pass_id in ("pass1", "pass2"):
                saved = {}
                for name in PASS_FILES:
                    raw = read(run, f"{prefix}/trial/{pass_id}/{name}", required=True)
                    if raw is not None:
                        saved[name] = raw
                summary = json.loads(saved.get("summary.json", b"{}"))
                rows = [json.loads(line) for line in saved.get("results.jsonl", b"").splitlines() if line.strip()]
                seconds = summary.get("seconds")
                latencies = [r["duration_seconds"] for r in rows if isinstance(r.get("duration_seconds"), (int, float))]
                item["passes"][pass_id] = {"summary": summary, "result_rows": len(rows),
                    "successful_rows": sum(r.get("ok") is True for r in rows),
                    "failed_rows": [r for r in rows if r.get("ok") is not True],
                    "rows_with_retry_failures": sum(bool(r.get("failure_categories")) for r in rows),
                    "judgments_per_second": len(rows) / seconds if seconds and rows else None,
                    "latency_seconds_mean": statistics.mean(latencies) if latencies else None,
                    "latency_seconds_median": statistics.median(latencies) if latencies else None,
                    "latency_seconds_max": max(latencies) if latencies else None}
            report["attempts"].append(item)
    return files, report, sorted(jobs)


def package(files, report, accounting):
    files = {**files, "report.json": canonical(report), "slurm-accounting.json": canonical(accounting)}
    manifest = {"format_version": "geodml-gemma-review-export-v1", "scientific_result": False,
                "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                          for name, raw in sorted(files.items())}}
    files["manifest.json"] = canonical(manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(zipfile.ZipInfo(name), raw, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def publish(raw, repo_id):
    """One content-addressed object; verify the remote copy at its exact commit."""
    sha = hashlib.sha256(raw).hexdigest()
    name = f"reviews/gemma-si-v3/exports/{sha}.zip"
    store = HubStore(repo_id)  # Existing private repository and saved authentication.
    revision = store.head()
    if not store.exists(name, revision):
        revision = store.commit(revision, {name: raw}, "Archive Gemma SI-v3 judgments and diagnostics")
    if store.read(name, revision) != raw:
        raise ValueError("remote archive verification failed")
    return {"repo_id": repo_id, "revision": revision, "path": name, "sha256": sha,
            "bytes": len(raw), "verified": True,
            "url": f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/{name}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    files, report, jobs = collect(args.run)
    command = ["sacct", "-j", ",".join(jobs), "--noheader", "--parsable2",
               "--format=JobIDRaw,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocCPUS,AllocTRES,MaxRSS"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        accounting = {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        accounting = {"command": command, "error": str(error)}
    raw = package(files, report, accounting)
    sha = hashlib.sha256(raw).hexdigest()
    target = args.output / f"{sha}.zip"
    atomic(target, raw)
    atomic(args.output / f"{sha}.report.json", canonical(report))
    print("LOCAL_EXPORT " + json.dumps({"path": str(target), "sha256": sha, "bytes": len(raw),
                                      "missing": report["missing"]}), flush=True)
    if args.upload:
        receipt = publish(raw, args.repo_id)
        atomic(args.output / f"{sha}.receipt.json", canonical(receipt))
        print("HF_EXPORT " + json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
