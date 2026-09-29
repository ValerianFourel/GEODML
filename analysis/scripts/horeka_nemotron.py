#!/usr/bin/env python3
"""Nemotron judge on HoreKa four-A100 nodes: download, submit and execute.

Same model, revision and scientific serving settings as the JUPITER pilot
(bf16, four-way tensor parallel, 73,728-token window, 0.85 GPU memory,
eager mode, four concurrent requests, thinking off, temperature 0); only the
accelerator differs. `execute` runs a trial driver on finished Qwen cells of the
HoreKa dataset (source importance SI-v3 by default, or the older claims-v3): a
diagnostic compatibility run, never results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hours import canonical

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
NEMOTRON = ((MODEL_ID, MODEL_REVISION),)
SERVING = {"tensor_parallel_size": 4, "data_parallel_size": 1, "dtype": "bfloat16", "max_model_len": 73728,
           "gpu_memory_utilization": 0.85, "request_concurrency": 4, "enforce_eager": True}
JOB_NAME = "geodml-nemotron-horeka-trial"
CLEANUP_MARGIN = 120
TRIALS = {  # trial name -> (driver script, stage name, default output-token cap)
    "source-importance": ("try_source_importance_judge.py", "nemotron-source-importance-horeka-trial", 640),
    "claims-v3": ("try_claims_v3_judge.py", "nemotron-claims-v3-horeka-trial", 4096),
}


def read(path):
    return json.loads(Path(path).read_bytes())


def preparation_dir(workspace: Path) -> Path:
    return workspace / "preparation" / f"nemotron-{MODEL_REVISION}"


def download(args) -> int:
    """Login node: inventory the pinned revision, check quota, download, verify checksums."""
    from huggingface_hub import HfApi

    from analysis.scripts.prepare_horeka_qwen import capture_quota, download_models, model_inventory
    workspace = args.workspace.resolve()
    out = preparation_dir(workspace)
    out.mkdir(parents=True, exist_ok=True)
    manifest = model_inventory(HfApi(), NEMOTRON)
    atomic(out / "models.json", canonical(manifest))
    quota = capture_quota(workspace, args.account)
    atomic(out / "quota.json", canonical(quota))
    verified = download_models(manifest, workspace / "models", quota, models=NEMOTRON)
    atomic(out / "models-verified.json", canonical(verified))
    print(json.dumps(verified, indent=2))
    return 0


def stage_commands(config: dict, *, python: str, attempt: Path, cache: Path) -> tuple[list[str], list[str]]:
    """The two stage-runner calls, with the frozen pilot settings and an A100 check."""
    repo = Path(config["repository"])
    script, stage, _ = TRIALS[config.get("trial", "claims-v3")]  # configs written before SI were claims-v3
    profile = attempt / "serving-profile.json"
    prepare = [python, str(repo / "analysis/scripts/search_vllm_stage.py"), "prepare",
               "--profile", str(profile), "--stage", stage,
               "--model-id", MODEL_ID, "--model-revision", MODEL_REVISION,
               "--vllm-executable", str(Path(python).parent / "vllm"), "--cache-base", str(cache),
               "--expected-gpu-name-pattern", "A100",
               "--data-parallel-size", str(SERVING["data_parallel_size"]),
               "--tensor-parallel-size", str(SERVING["tensor_parallel_size"]),
               "--dtype", SERVING["dtype"], "--max-model-len", str(SERVING["max_model_len"]),
               "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
               "--request-concurrency", str(SERVING["request_concurrency"]), "--enforce-eager"]
    run = [python, str(repo / "analysis/scripts/search_vllm_stage.py"), "run",
           "--profile", str(profile), "--server-log", str(attempt / "server.log"),
           "--cache-base", str(cache), "--startup-timeout-seconds", "1200", "--",
           python, str(repo / "analysis/scripts" / script),
           "--dataset-root", config["dataset_root"], "--output", str(attempt / "trial"),
           "--model", config.get("generator_model", "qwen38"), "--count", str(config["count"]),
           "--base-url", "http://127.0.0.1:8010/v1", "--server-model-name", MODEL_ID,
           "--max-tokens", str(config["max_tokens"]), "--concurrency", str(SERVING["request_concurrency"])]
    if config.get("cells_from"):
        run += ["--cells-from", config["cells_from"], "--cells-from-sha256", config["cells_from_sha256"]]
    return prepare, run


def submit(args) -> int:
    """Login node: freeze the run configuration and submit one approved allocation."""
    repo = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("clean committed checkout required")
    if not args.approval.strip():
        raise ValueError("a recorded approval is required")
    workspace, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("run directory already exists; inspect it instead of resubmitting")
    verified = preparation_dir(workspace) / "models-verified.json"
    if not verified.is_file() or read(verified).get("status") != "verified":
        raise ValueError("download and verify Nemotron first (horeka_nemotron.py download)")
    runtime = workspace / "environment/qwen-runtime/bin/python"
    if not runtime.is_file():
        raise ValueError("HoreKa runtime missing")
    if not (args.dataset.resolve() / "contract.json").is_file():
        raise ValueError("dataset root has no contract.json")
    cells_from = getattr(args, "cells_from", None)
    if cells_from and args.trial != "source-importance":
        raise ValueError("--cells-from is only supported by the source-importance trial")
    if args.count <= 0:
        raise ValueError("count must be positive")
    config = {"repository": str(repo), "git_commit": pin, "job_name": JOB_NAME, "walltime": args.walltime,
              "workspace": str(workspace), "dataset_root": str(args.dataset.resolve()), "count": args.count,
              "trial": args.trial,
              "max_tokens": args.max_tokens if args.max_tokens is not None else TRIALS[args.trial][2], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "serving": SERVING, "approval": args.approval, "scientific_result": False}
    config["generator_model"] = getattr(args, "model", "qwen38")
    if args.trial == "source-importance":
        from analysis.interpretability.pipeline.source_importance import PROTOCOL
        config["judge_protocol"] = PROTOCOL
    if cells_from:
        config.update(cells_from=str(cells_from.resolve()),
                      cells_from_sha256=hashlib.sha256(cells_from.read_bytes()).hexdigest())
    command = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--gres=gpu:4", "--exclusive",
               "--cpus-per-task=32", "--mem=0", "--time=" + args.walltime, "--account=" + args.account,
               "--partition=" + args.partition, "--job-name=" + JOB_NAME, "--chdir=" + str(repo),
               "--output=" + str(out / "slurm-%j.out"), "--error=" + str(out / "slurm-%j.err")]
    if args.reservation:
        command.append("--reservation=" + args.reservation)
    command += [str(out / "run.sh")]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config, "command": shlex.join(command)}, indent=2))
        return 0
    out.mkdir(parents=True)
    atomic(out / "config.json", canonical(config))
    atomic(out / "run.sh", (f"#!/bin/bash\nset -euo pipefail\nexec {shlex.quote(str(runtime))} "
                            f"{shlex.quote(str(Path(__file__).resolve()))} execute "
                            f"--config {shlex.quote(str(out / 'config.json'))}\n").encode())
    os.chmod(out / "run.sh", 0o755)
    if args.no_submit:
        # Interactive: the same frozen run, started by hand inside the salloc shell on the node.
        salloc = ["salloc", *[flag for flag in command[2:-1]
                              if not flag.startswith(("--output=", "--error=", "--chdir="))
                              and flag != "--no-requeue"]]  # sbatch-only options
        atomic(out / "interactive.json", canonical({"salloc": salloc, "run": str(out / "run.sh"),
                                                    "approval": args.approval}))
        print(json.dumps({"interactive": True, "run": str(out), "salloc": shlex.join(salloc),
                          "then_inside_the_salloc_shell": "bash " + shlex.quote(str(out / "run.sh"))}, indent=2))
        return 0
    with (out / "SUBMISSION_ATTEMPTED").open("x") as marker:  # never resubmit an ambiguous receipt
        marker.write(str(time.time()))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    atomic(out / "submission.json", canonical({"returncode": result.returncode, "stdout": result.stdout,
                                               "stderr": result.stderr, "command": command}))
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(json.dumps({"job_id": result.stdout.strip(), "run": str(out), "walltime": args.walltime,
                      "maximum_gpu_hours": 4 * _hours(args.walltime), "scientific_result": False}, indent=2))
    return 0


def _hours(walltime: str) -> float:
    h, m, s = (int(x) for x in walltime.split(":"))
    return h + m / 60 + s / 3600


def execute(config_path: Path) -> int:
    """Compute node: boundary and allocation checks first, then Nemotron and the trial."""
    from analysis.scripts.horeka_qwen_bouts import compiler_environment
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify("horeka")
    config = read(config_path)
    job = os.environ["SLURM_JOB_ID"]
    attempt = Path(config_path).parent / "attempts" / f"job{job}"
    attempt.mkdir(parents=True, exist_ok=False)
    atomic(attempt / "boundary.json", canonical(boundary))
    fields = dict(t.split("=", 1) for t in subprocess.check_output(
        ["scontrol", "show", "job", job, "-o"], text=True).split() if "=" in t)
    if fields.get("JobName") != config["job_name"] or fields.get("TimeLimit") != config["walltime"]:
        raise ValueError("allocation differs from the approved run")
    repo = Path(config["repository"])
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != config["git_commit"]:
        raise ValueError("execution commit mismatch")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("execution checkout is dirty")
    workspace = Path(config["workspace"])
    scratch = Path(os.environ.get("TMPDIR") or attempt / "tmp")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SEARCH_AGENTIC_", "GEODML_"))}
    env.update(compiler_environment())
    env.update(GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_CACHE=str(workspace / "models"), PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1",
               XDG_CACHE_HOME=str(scratch / "xdg"), TRITON_CACHE_DIR=str(scratch / "triton"),
               VLLM_CACHE_ROOT=str(scratch / "vllm"))
    end = int(datetime.fromisoformat(fields["EndTime"]).timestamp())
    cache = workspace / "serving-cache" / f"nemotron-job{job}"
    prepare, run = stage_commands(config, python=sys.executable, attempt=attempt, cache=cache)
    if config.get("trial") == "source-importance":
        from analysis.interpretability.pipeline.source_importance import PROTOCOL
        if config.get("judge_protocol") != PROTOCOL:
            raise ValueError("SI protocol mismatch; use the original checkout for historical runs")
        # Fail before server/model startup if this runtime rejects the SI schema.
        check = subprocess.run([sys.executable, str(repo / "analysis/scripts/try_source_importance_judge.py"),
                                "--check-schema"], cwd=repo, env=env, text=True, capture_output=True)
        atomic(attempt / "schema-check.json", canonical({"returncode": check.returncode,
               "stdout": check.stdout, "stderr": check.stderr}))
        if check.returncode:
            raise RuntimeError(f"SI schema check failed; inspect {attempt / 'schema-check.json'}")
    atomic(attempt / "nvidia-smi.txt", subprocess.check_output(["nvidia-smi"], text=True).encode())
    atomic(attempt / "execution.json", canonical({"job_id": job, "started_at": time.time(), "prepare": prepare,
           "run": run, "git_commit": config["git_commit"], "compiler": compiler_environment(),
           "serving": SERVING, "approval": config["approval"], "scientific_result": False}))
    subprocess.run(prepare, cwd=repo, env=env, check=True)
    telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                  "--format=csv,noheader", "-l", "30"],
                                 stdout=(attempt / "gpu.csv").open("w"), stderr=subprocess.DEVNULL)
    process = subprocess.Popen(run, cwd=repo, env=env, start_new_session=True)
    started = time.time()
    try:
        returncode = process.wait(timeout=max(1, end - time.time() - CLEANUP_MARGIN))
        status = "completed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        returncode, status = 124, "deadline"
    finally:
        for proc, group in ((process, True), (telemetry, False)):
            try:
                os.killpg(proc.pid, signal.SIGTERM) if group else proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL) if group else proc.kill()
                proc.wait()
    summary = attempt / "trial/summary.json"
    atomic(attempt / "trial-result.json", canonical({
        "status": status, "returncode": returncode, "job_id": job, "elapsed_seconds": time.time() - started,
        "summary": read(summary) if summary.is_file() else None, "scientific_result": False}))
    return returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--workspace", type=Path, required=True)
    d.add_argument("--account", required=True)
    s = sub.add_parser("submit")
    for name in ("workspace", "dataset", "output"):
        s.add_argument("--" + name, type=Path, required=True)
    s.add_argument("--account", required=True)
    s.add_argument("--partition", default="accelerated")
    s.add_argument("--reservation")
    s.add_argument("--walltime", required=True, choices=["01:00:00"],
                   help="approved wall-time; this trial allows only one hour")
    s.add_argument("--approval", required=True)
    s.add_argument("--trial", choices=sorted(TRIALS), default="source-importance")
    s.add_argument("--model", choices=["qwen38", "llama4"], default="qwen38", help="generator corpus to judge")
    s.add_argument("--cells-from", type=Path, help="replay the exact cells from a saved smoke cells.jsonl")
    s.add_argument("--count", type=int, default=12)
    s.add_argument("--max-tokens", type=int, help="default: 640 for source-importance, 4096 for claims-v3")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-submit", action="store_true",
                   help="prepare the run and print the salloc command for an interactive node")
    e = sub.add_parser("execute")
    e.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        return download(args)
    if args.command == "submit":
        return submit(args)
    return execute(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
