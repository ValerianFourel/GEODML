"""Observe an existing adaptive allocation. Never start or stop inference.

Standard library only. Reports are deliberately separate from experiment outputs.
Claim counts are structural observations, not scientific/result-hash validation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


def command(argv, timeout=15, env=None):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
        # Do not export arbitrary stderr: it can contain arguments or credentials.
        return {"returncode": result.returncode, "stdout": result.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "stdout": "", "error": type(error).__name__}


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def job_fields(text, job_id, uid):
    fields = dict(word.split("=", 1) for word in text.split() if "=" in word)
    keys = (
        "JobId",
        "JobState",
        "UserId",
        "NodeList",
        "NumNodes",
        "EndTime",
        "StartTime",
        "TimeLimit",
    )
    selected = {key: fields.get(key) for key in keys}
    selected["live"] = (
        fields.get("JobId") == job_id
        and fields.get("JobState") == "RUNNING"
        and fields.get("UserId", "").endswith(f"({uid})")
    )
    return selected


def step_command(job_id, nodes, payload, node=None):
    if not str(job_id).isdigit() or nodes < 1:
        raise ValueError(
            "explicit numeric existing job ID and positive node count required"
        )
    return [
        "srun",
        f"--jobid={job_id}",
        "--overlap",
        "--exact",
        f"--nodes={nodes}",
        f"--ntasks={nodes}",
        "--ntasks-per-node=1",
        "--cpus-per-task=1",
        "--mem=1G",
        "--cpu-bind=none",
        "--immediate=15",
        *([f"--nodelist={node}"] if node else []),
        *payload,
    ]


def step_environment():
    keep = {
        "SLURM_CONF",
        "SLURM_CONF_SERVER",
        "SLURM_JWT",
        "SLURM_CLUSTER_NAME",
        "SLURM_CLUSTERS",
    }
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SLURM_", "SRUN_")) or k in keep
    }


def safe_output(output, root):
    output, root = output.resolve(), root.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("diagnostics must be outside the experiment run")
    output.mkdir(parents=True, exist_ok=False)


def client_identity(argv, root):
    # The supervisor also contains its child's command in argv. Match the first
    # executed script, not a later argument describing a child process.
    scripts = [Path(arg).name for arg in argv[1:] if arg.endswith(".py")]
    if not scripts or scripts[0] != "run_agentic_search_integration_smoke.py":
        return None
    try:
        target = Path(argv[argv.index("--output") + 1]).resolve()
        relative = target.relative_to(root.resolve() / "workers")
        return relative.parts[0] if len(relative.parts) >= 2 else None
    except (ValueError, IndexError):
        return None


def clients(root, job_id):
    records = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = (
                process.joinpath("cmdline")
                .read_bytes()
                .decode(errors="replace")
                .split("\0")
            )
            worker = client_identity(argv, root)
            if not worker:
                continue
            environment = process.joinpath("environ").read_bytes().split(b"\0")
            if f"SLURM_JOB_ID={job_id}".encode() not in environment:
                continue
            status = dict(
                line.split(":", 1)
                for line in process.joinpath("status").read_text().splitlines()
                if ":" in line
            )
            stat = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            try:
                io = dict(
                    line.split(":", 1)
                    for line in process.joinpath("io").read_text().splitlines()
                )
            except OSError:
                io = {}
            records.append(
                {
                    "pid": int(process.name),
                    "worker": worker,
                    "start_ticks": int(stat[19]),
                    "cpu_seconds": (int(stat[11]) + int(stat[12]))
                    / os.sysconf("SC_CLK_TCK"),
                    "threads": status.get("Threads", "").strip(),
                    "rss": status.get("VmRSS", "").strip(),
                    "allowed_cpus": status.get("Cpus_allowed_list", "").strip(),
                    "io": {k: int(v) for k, v in io.items()},
                    "executable": str(process.joinpath("exe").resolve()),
                }
            )
        except (OSError, ValueError, IndexError):
            continue
    return records


class ClaimReader:
    def __init__(self):
        self.cache = {}

    def snapshot(self, root):
        roles = {}
        for role in ("original-llama4", "nemotron"):
            successes, failures, invalid, duplicate = {}, set(), 0, 0
            directory = root / "claims" / role
            for path in directory.glob("*/*.json"):
                if not re.fullmatch(r"[0-9a-f]{64}(?:\.failed)?\.json", path.name):
                    continue
                try:
                    stat = path.stat()
                    key = (str(path), stat.st_mtime_ns, stat.st_size)
                    if key not in self.cache:
                        value = read_json(path)
                        outcome = value["outcome"]
                        identity = value["identity_sha256"]
                        if identity != path.name[:64] or not isinstance(outcome, dict):
                            raise ValueError("invalid claim envelope")
                        cell = outcome.get("result", {}).get("cell_id", identity)
                        elapsed = outcome.get("diagnostics", {}).get("elapsed_seconds")
                        self.cache[key] = (cell, elapsed)
                    cell, elapsed = self.cache[key]
                    if path.name.endswith(".failed.json"):
                        failures.add(cell)
                    else:
                        duplicate += int(cell in successes)
                        successes[cell] = elapsed
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    invalid += 1
            latencies = sorted(
                v
                for v in successes.values()
                if isinstance(v, (float, int)) and math.isfinite(v) and v >= 0
            )
            roles[role] = {
                "completed": len(successes),
                "failed": len(failures),
                "invalid": invalid,
                "duplicates": duplicate,
                "exists": directory.exists(),
                "median_cell_seconds_all_saved": statistics.median(latencies)
                if latencies
                else None,
                "p95_cell_seconds_all_saved": latencies[
                    math.ceil(0.95 * len(latencies)) - 1
                ]
                if latencies
                else None,
            }
        return roles


def bounded_log(path, states):
    stat = path.stat()
    old_inode, offset = states.get(str(path), (None, 0))
    rotated = old_inode is not None and (
        old_inode != stat.st_ino or stat.st_size < offset
    )
    if old_inode != stat.st_ino or stat.st_size < offset:
        offset = 0
    # Limit the read even on first observation; never copy prompts or arbitrary log text.
    start = max(offset, stat.st_size - 65536)
    with path.open("rb") as stream:
        stream.seek(start)
        text = stream.read(65536).decode(errors="replace")
        states[str(path)] = (stat.st_ino, stream.tell())
    metrics = []
    for line in text.splitlines():
        match = re.search(
            r"Avg generation throughput: ([\d.]+) tokens/s.*Running: (\d+) reqs, Waiting: (\d+) reqs",
            line,
        )
        if match:
            metrics.append(
                {
                    "generation_tokens_s": float(match[1]),
                    "running": int(match[2]),
                    "waiting": int(match[3]),
                }
            )
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "bytes": stat.st_size,
        "rotated": rotated,
        "truncated": start > offset,
        "metrics": metrics,
        "http_200": text.count('POST /v1/chat/completions HTTP/1.1" 200'),
        "errors": len(re.findall(r"\bERROR\b|Traceback", text)),
        "startup_complete": "Application startup complete" in text,
    }


def worker_snapshot(root, states):
    workers = {}
    for worker in sorted((root / "workers").glob("worker-*")):
        result_ids = {p.stem for p in worker.glob("*/output/results/*.json")}
        configs = []
        for path in worker.glob("*/output/config.json"):
            try:
                config = read_json(path)
                configs.append(
                    {
                        k: config.get(k)
                        for k in (
                            "git_commit",
                            "model_id",
                            "model_revision",
                            "request_concurrency",
                            "cell_concurrency",
                            "execution_policy",
                        )
                    }
                )
            except (OSError, ValueError, AttributeError):
                configs.append({"error": "unreadable_config"})
        logs = []
        paths = list(worker.glob("*/server.log"))
        paths += list((root / "launcher-logs").glob(worker.name + ".*"))
        for path in paths:
            try:
                logs.append(bounded_log(path, states))
            except OSError:
                logs.append({"path": str(path), "error": "unreadable_log"})
        workers[worker.name] = {
            "materialized": len(result_ids),
            "configs": configs,
            "logs": logs,
        }
    return workers


def summarize(snapshots):
    if not snapshots:
        return {"status": "no_samples"}
    first, last = snapshots[0], snapshots[-1]
    seconds = last["time"] - first["time"]
    roles = {}
    for role, current in last["claims"].items():
        delta = current["completed"] - first["claims"].get(role, {}).get(
            "completed", current["completed"]
        )
        rate = delta * 60 / seconds if seconds > 0 and delta >= 0 else None
        remaining = (
            max(0, 6000 - current["completed"]) if role == "original-llama4" else None
        )
        roles[role] = {
            **current,
            "new_cells": delta,
            "cells_per_minute": rate,
            "eta_minutes": remaining / rate
            if remaining is not None and rate and rate > 0
            else None,
        }
    workers = {
        name: {
            "materialized_delta": current["materialized"]
            - first["workers"]
            .get(name, {})
            .get("materialized", current["materialized"])
        }
        for name, current in last["workers"].items()
    }
    return {
        "status": "observed",
        "baseline_seconds": seconds,
        "roles": roles,
        "workers": workers,
        "notes": [
            "Worker counts measure materialization, not proven producer attribution.",
            "Counts are structurally readable saved artifacts, not scientific validation.",
            "ETA is conditional on this window's rate; it excludes judging.",
            "CPU profiles are collected after this baseline, never inside it.",
        ],
    }


def node_probe(args):
    host = socket.gethostname().split(".")[0]
    write_json(
        args.output / f"hardware-{host}.json",
        {
            "host": host,
            "time": time.time(),
            "python": sys.version,
            "gpu": command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "vllm",
                    "py-spy",
                )
            },
        },
    )
    target = args.output / f"node-{host}.jsonl"
    previous = {}
    end = time.monotonic() + args.duration
    with target.open("x") as stream:
        while True:
            stamp = time.time()
            processes = clients(args.run_root, args.job_id)
            for record in processes:
                identity = (record["pid"], record["start_ticks"])
                old = previous.get(identity)
                if old and stamp > old[0]:
                    record["cpu_percent_interval"] = (
                        100 * (record["cpu_seconds"] - old[1]) / (stamp - old[0])
                    )
                previous[identity] = (stamp, record["cpu_seconds"])
            gpu = command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                timeout=4,
            )
            stream.write(
                json.dumps(
                    {"time": stamp, "host": host, "clients": processes, "gpu": gpu}
                )
                + "\n"
            )
            stream.flush()
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(5, remaining))


def node_summary(output):
    hosts = {}
    for path in sorted(output.glob("node-*.jsonl")):
        cpu, gpu, records, errors = [], [], 0, 0
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                records += 1
                cpu.extend(
                    r["cpu_percent_interval"]
                    for r in row["clients"]
                    if "cpu_percent_interval" in r
                )
                if row["gpu"]["returncode"] != 0:
                    errors += 1
                for fields in csv.reader(row["gpu"]["stdout"].splitlines()):
                    if len(fields) == 4:
                        try:
                            gpu.append(float(fields[1]))
                        except ValueError:
                            pass
            except (ValueError, KeyError, TypeError):
                errors += 1
        hosts[path.stem.removeprefix("node-")] = {
            "samples": records,
            "errors": errors,
            "mean_client_cpu_percent": statistics.mean(cpu) if cpu else None,
            "mean_sampled_gpu_utilization_percent": statistics.mean(gpu)
            if gpu
            else None,
            "gpu_idle_sample_fraction": sum(v == 0 for v in gpu) / len(gpu)
            if gpu
            else None,
        }
    return hosts


def profile_client(args):
    records = [
        r for r in clients(args.run_root, args.job_id) if r["worker"] == args.worker
    ]
    tool = shutil.which("py-spy")
    report = {"worker": args.worker, "started_at": time.time(), "status": "unavailable"}
    if tool and len(records) == 1:
        report["pid"] = records[0]["pid"]
        # No locals, subprocesses, sudo or environment dumps. Exclude this from baseline.
        result = command(
            [
                tool,
                "record",
                "--pid",
                str(records[0]["pid"]),
                "--duration",
                "30",
                "--rate",
                "25",
                "--idle",
                "--format",
                "speedscope",
                "--output",
                str(args.output / "cpu-profile.json"),
            ],
            timeout=45,
        )
        report["status"] = (
            "collected" if result["returncode"] == 0 else "attachment_failed"
        )
        report["returncode"] = result["returncode"]
    else:
        report["reason"] = (
            "py_spy_missing" if not tool else "client_missing_or_ambiguous"
        )
    report["finished_at"] = time.time()
    write_json(args.output / "profile-status.json", report)


def collect(args):
    safe_output(args.output, args.run_root)
    query = command(["scontrol", "show", "job", "--oneliner", args.job_id])
    job = job_fields(query["stdout"], args.job_id, os.getuid())
    job["query_returncode"] = query["returncode"]
    job["requested_job_id"] = args.job_id
    job["live"] = job["live"] and query["returncode"] == 0
    write_json(args.output / "allocation.json", job)
    plan = read_json(args.run_root / "run_manifest.json")
    write_json(
        args.output / "provenance.json",
        {
            "run_root": str(args.run_root),
            "experiment_commit": plan.get("source_git_commit"),
            "python": sys.version,
            "collector_commit": command(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[2]),
                    "rev-parse",
                    "HEAD",
                ]
            )["stdout"].strip(),
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "vllm",
                    "py-spy",
                )
            },
        },
    )
    reader, log_states, snapshots = ClaimReader(), {}, []
    node_process = None
    duration = args.duration if job["live"] else 0
    if job["live"]:
        try:
            seconds_left = (
                time.mktime(time.strptime(job["EndTime"], "%Y-%m-%dT%H:%M:%S"))
                - time.time()
            )
            duration = min(duration, max(0, int(seconds_left - 60)))
        except (ValueError, TypeError):
            # No controller deadline, no monitoring step. Saved artifacts remain readable.
            duration = 0
    try:
        if job["live"] and duration > 0:
            payload = [
                sys.executable,
                str(Path(__file__).resolve()),
                "node",
                "--job-id",
                args.job_id,
                "--run-root",
                str(args.run_root),
                "--output",
                str(args.output),
                "--duration",
                str(duration),
            ]
            # Raw srun output is not needed; report its exit status without leaking arguments.
            node_process = subprocess.Popen(
                step_command(args.job_id, int(job["NumNodes"]), payload),
                env=step_environment(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        end = time.monotonic() + duration
        with (args.output / "baseline.jsonl").open("x") as stream:
            while True:
                snapshot = {
                    "time": time.time(),
                    "claims": reader.snapshot(args.run_root),
                    "workers": worker_snapshot(args.run_root, log_states),
                }
                snapshots.append(snapshot)
                stream.write(json.dumps(snapshot) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {"time": snapshot["time"], "claims": snapshot["claims"]}
                    ),
                    flush=True,
                )
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(60, remaining))
        report = summarize(snapshots)
        report["live_allocation"] = job["live"]
        report["requested_baseline_seconds"] = args.duration
        report["deadline_limited_baseline_seconds"] = duration
        if node_process:
            try:
                report["probe_exit_code"] = node_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Kill only our bounded monitoring command, never any inference step/job.
                node_process.terminate()
                report["probe_exit_code"] = "monitor_timeout"
        report["nodes"] = node_summary(args.output)
        report["expected_node_count"] = (
            int(job["NumNodes"]) if str(job.get("NumNodes", "")).isdigit() else None
        )
        report["node_coverage_complete"] = (
            len(report["nodes"]) == report["expected_node_count"]
        )
        write_json(args.output / "summary.json", report)
    finally:
        if node_process and node_process.poll() is None:
            node_process.terminate()
    print("REPORT=" + str(args.output / "summary.json"), flush=True)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def profile(args):
    provenance = read_json(args.output / "provenance.json")
    allocation = read_json(args.output / "allocation.json")
    if (
        provenance["run_root"] != str(args.run_root)
        or allocation["requested_job_id"] != args.job_id
    ):
        raise ValueError("profile must match the baseline job and run")
    if (args.output / "profile-status.json").exists() or (
        args.output / "cpu-profile.json"
    ).exists():
        raise ValueError("profile already attempted; preserve the existing report")
    report = read_json(args.output / "summary.json")
    candidates = sorted(
        report.get("workers", {}),
        key=lambda w: (report["workers"][w]["materialized_delta"], w),
    )
    nodes = []
    for file in args.output.glob("node-*.jsonl"):
        try:
            last = json.loads(file.read_text().splitlines()[-1])
            nodes.extend((r["worker"], last["host"]) for r in last["clients"])
        except (OSError, ValueError, KeyError, IndexError):
            continue
    selected = next(
        ((worker, host) for worker in candidates for w, host in nodes if w == worker),
        None,
    )
    job = job_fields(
        command(["scontrol", "show", "job", "--oneliner", args.job_id])["stdout"],
        args.job_id,
        os.getuid(),
    )
    if not job["live"] or not selected:
        write_json(
            args.output / "profile-status.json",
            {"status": "skipped", "reason": "no_live_job_or_identified_client"},
        )
        return
    worker, host = selected
    try:
        seconds_left = (
            time.mktime(time.strptime(job["EndTime"], "%Y-%m-%dT%H:%M:%S"))
            - time.time()
        )
    except (ValueError, TypeError):
        seconds_left = 0
    allowed_nodes = command(["scontrol", "show", "hostnames", job["NodeList"]])[
        "stdout"
    ].split()
    if seconds_left < 90 or host not in allowed_nodes:
        write_json(
            args.output / "profile-status.json",
            {"status": "skipped", "reason": "deadline_or_node_mismatch"},
        )
        return
    result = command(
        step_command(
            args.job_id,
            1,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "profile-node",
                "--job-id",
                args.job_id,
                "--run-root",
                str(args.run_root),
                "--output",
                str(args.output),
                "--worker",
                worker,
            ],
            node=host,
        ),
        timeout=65,
        env=step_environment(),
    )
    if not (args.output / "profile-status.json").exists():
        write_json(
            args.output / "profile-status.json",
            {"status": "step_failed", "returncode": result["returncode"]},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("collect", "node", "profile", "profile-node", "summary")
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if not args.job_id.isdigit() or not 0 <= args.duration <= 600:
        parser.error("numeric job ID and duration between 0 and 600 seconds required")
    args.run_root, args.output = args.run_root.resolve(), args.output.resolve()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "node":
        node_probe(args)
    elif args.mode == "profile-node":
        profile_client(args)
    elif args.mode == "profile":
        profile(args)
    else:
        print(json.dumps(read_json(args.output / "summary.json"), indent=2))


if __name__ == "__main__":
    main()
