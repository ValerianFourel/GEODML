#!/usr/bin/env python3
"""Qwen BM25 rerun inside one HoreKa allocation, with the original Qwen bouts' serving setup.

  horeka_qwen.py execute --bout-config CONFIG --bundle B --run-dir R [--retriever bm25] [--score-cache C ...]
                         [--phase REQ:CELL[:SECONDS] ...] [--limit N]

CONFIG is an existing Qwen bout's config.json (qwen-bouts/<division>/bouts/bout-NNNN/config.json): it supplies the
frozen vLLM reference profile, the two search snapshots, the cross-encoder snapshot and the bout environment, so
the model, revision, chat template, parallelism and cross-encoder are the original run's. Like
horeka_qwen_bouts.execute this proves the exclusive compute node first, scrubs the environment, uses the system GCC
for FlashInfer, writes this job's vLLM profile, and runs generate.py under search_vllm_stage.py (one vLLM server,
loopback, per-run key). Phases run one after another against the same server, each generate.py call with its own
request and cell concurrency (concurrency is runtime-only, not a scientific setting); a phase with SECONDS stops
admitting cells after that long (GEODML_ROLE_END_TIME). Exit code: the last phase's (0 done, 4 deadline checkpoint,
3 breaker), 4 when the allocation ends first, 2 refused, 1 error.
"""
from __future__ import annotations

import argparse
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
from analysis.scripts.horeka_qwen_bouts import CLEANUP_MARGIN, START_MARGIN, compiler_environment

QUERY_MAX_TOKENS, FINAL_MAX_TOKENS = 256, 2048  # horeka_qwen_bouts.controller_command


def parse_phase(value: str) -> tuple[int, int, int | None]:
    parts = value.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise argparse.ArgumentTypeError("phase must be REQUEST:CELL or REQUEST:CELL:SECONDS (positive integers)")
    return int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) == 3 else None


def generate_command(python: str, repo: Path, config: dict, profile: dict, a, request: int, cell: int) -> list[str]:
    inputs = config["inputs"]
    command = [python, "-u", str(repo / "analysis/sigir_bm25/generate.py"), "run",
               "--bundle", str(a.bundle), "--run-dir", str(a.run_dir), "--model", "qwen38",
               "--retriever", a.retriever,
               "--snapshot", "duckduckgo=" + inputs["SEARCH_AGENTIC_DDG_SNAPSHOT"],
               "--snapshot", "searxng=" + inputs["SEARCH_AGENTIC_SEARXNG_SNAPSHOT"],
               "--base-url", profile["serving"]["public_base_url"], "--served-model", profile["model"]["model_id"],
               "--cross-encoder", config["cross_encoder"], "--device", "cpu",
               "--query-max-tokens", str(QUERY_MAX_TOKENS), "--final-max-tokens", str(FINAL_MAX_TOKENS),
               "--disable-thinking", "--request-concurrency", str(request), "--cell-concurrency", str(cell)]
    for path in a.score_cache:
        command += ["--score-cache", str(path)]
    if a.limit:
        command += ["--limit", str(a.limit)]
    return command + ["--shard", str(a.shard), "--shards", str(a.shards)]


def phase_script(commands: list[tuple[list[str], int | None]]) -> str:
    """Bash running each phase in turn; a timed phase ends admission via GEODML_ROLE_END_TIME. Last status wins."""
    lines = ["status=0"]
    for command, seconds in commands:
        prefix = f"GEODML_ROLE_END_TIME=$(( $(date +%s) + {seconds} )) " if seconds else ""
        lines.append(f"{prefix}{shlex.join(command)}; status=$?")
        lines.append('case $status in 0|4) ;; *) exit $status ;; esac')
    lines.append("exit $status")
    return "\n".join(lines)


def execute(a) -> int:
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify("horeka")  # compute-node proof precedes configuration, GPU imports and model loading
    config = json.loads(Path(a.bout_config).read_text())
    repo = Path(__file__).resolve().parents[2]
    job = os.environ["SLURM_JOB_ID"]
    attempt = Path(a.run_dir) / "attempts" / f"job{job}"
    attempt.mkdir(parents=True, exist_ok=False)
    atomic(attempt / "boundary.json", canonical(boundary))
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                               text=True).strip():
        print(f"refused: dirty checkout {repo}", file=sys.stderr)
        return 2
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    info = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(t.split("=", 1) for t in info.split() if "=" in t)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SEARCH_AGENTIC_", "GEODML_"))}
    env.update(config["environment"])
    env["PYTHONPATH"] = str(repo)
    env.update(compiler_environment())
    for field, variable in (("StartTime", "SLURM_JOB_START_TIME"), ("EndTime", "SLURM_JOB_END_TIME")):
        env[variable] = str(int(datetime.fromisoformat(fields[field]).timestamp()))
    env.update(GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY="1", GEODML_APPROVED_WALLTIME=fields["TimeLimit"],
               GEODML_START_MARGIN_SECONDS=str(START_MARGIN), GEODML_CLEANUP_MARGIN_SECONDS=str(CLEANUP_MARGIN),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("GEODML_PRIVATE_NETWORK_NAMESPACE", None)
    os.environ.clear()
    os.environ.update(env)
    from analysis.scripts.horeka_qwen_probe import local_profile
    from analysis.scripts.search_vllm_stage import (
        create_or_verify_profile,
        discover_visible_gpus,
        inspect_vllm,
        load_profile,
    )
    atomic(attempt / "nvidia-smi.txt", subprocess.check_output(["nvidia-smi"], text=True).encode())
    frozen = load_profile(config["reference_profile"])
    executable, version, help_text = inspect_vllm(Path(sys.executable).parent / "vllm")
    profile = local_profile(frozen, executable, version, help_text, discover_visible_gpus(),
                            os.environ.get("CUDA_VISIBLE_DEVICES"))
    profile_path = attempt / "horeka-profile.json"
    create_or_verify_profile(profile_path, profile)
    phases = a.phase or [(4, 12, None)]
    script = phase_script([(generate_command(sys.executable, repo, config, profile, a, r, c), s) for r, c, s in phases])
    cache = Path(a.cache_base or Path(config["cache"]).parent / "sigir-bm25")
    command = [sys.executable, str(repo / "analysis/scripts/search_vllm_stage.py"), "run",
               "--profile", str(profile_path), "--server-log", str(attempt / "server.log"),
               "--cache-base", str(cache), "--startup-timeout-seconds", "900", "--", "bash", "-c", script]
    started = time.time()
    atomic(attempt / "execution.json", canonical({"job_id": job, "started_at": started, "command": command,
                                                  "git_commit": commit, "bout_config": str(a.bout_config),
                                                  "retriever": a.retriever, "phases": phases,
                                                  "compiler": compiler_environment(), "scientific_result": False}))
    telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                  "--format=csv,noheader", "-l", "30"],
                                 stdout=(attempt / "gpu.csv").open("w"), stderr=subprocess.DEVNULL)
    timeout = int(env["SLURM_JOB_END_TIME"]) - time.time() - CLEANUP_MARGIN
    if timeout <= 0:
        raise ValueError("allocation has no remaining work time")
    process = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
    status = None
    try:
        status = process.wait(timeout=timeout)
        return 4 if status == 124 else status
    except subprocess.TimeoutExpired:
        status = "deadline"
        return 4
    finally:
        atomic(attempt / "result.json", canonical({"job_id": job, "returncode": status,
                                                   "elapsed_seconds": time.time() - started}))
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("execute")
    e.add_argument("--bout-config", required=True, type=Path)
    e.add_argument("--bundle", required=True, type=Path)
    e.add_argument("--run-dir", required=True, type=Path)
    e.add_argument("--retriever", default="bm25", choices=("bm25", "orig"))
    e.add_argument("--score-cache", action="append", default=[], type=Path)
    e.add_argument("--phase", action="append", type=parse_phase, help="REQUEST:CELL[:SECONDS], in order")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--shard", type=int, default=0, help="this allocation's slice of the cells")
    e.add_argument("--shards", type=int, default=1, help="slices run by parallel allocations")
    e.add_argument("--cache-base", type=Path, help="vLLM caches (default: next to the bout's serving cache)")
    a = p.parse_args(argv)
    return execute(a)


if __name__ == "__main__":
    raise SystemExit(main())
