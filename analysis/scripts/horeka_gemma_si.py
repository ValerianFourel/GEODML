#!/usr/bin/env python3
"""Prepare a pinned Gemma 4 31B SI-v3 replay. Never submits or acquires an allocation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hours import canonical
from analysis.interpretability.pipeline.source_importance import PROTOCOL
from analysis.scripts import horeka_nemotron as runner
from analysis.scripts.prepare_horeka_qwen import capture_quota, download_models, immutable_json, model_inventory
from analysis.scripts.replay_source_importance_judge import freeze_baseline

MODEL = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MODELS = ((MODEL, REVISION),)
JOB_NAME = "geodml-gemma-si-replay"


def prep(workspace):
    return workspace / "preparation" / f"gemma4-{REVISION}"


def remove_nemotron(workspace):
    """Delete only this model's HF cache, never results, other weights or allocations."""
    jobs = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%j"], text=True)
    if "nemotron" in jobs.lower():
        raise ValueError("Nemotron has queued/running allocations; leave them intact and remove its cache later")
    cache = workspace / "models"
    target = cache / ("models--" + runner.MODEL_ID.replace("/", "--"))
    if cache.is_symlink() or target.is_symlink() or target.resolve().parent != cache.resolve():
        raise ValueError("refusing redirected model cache")
    if target.exists():
        shutil.rmtree(target)
    # Invalidate its old verification receipt so a future Nemotron run cannot trust it.
    receipt = runner.preparation_dir(workspace) / "models-verified.json"
    if receipt.exists():
        receipt.unlink()
    print(f"REMOVED_MODEL_CACHE {target}", flush=True)


def download(args):
    from huggingface_hub import HfApi
    workspace = args.workspace.resolve()
    destination = prep(workspace)
    destination.mkdir(parents=True, exist_ok=True)
    # Save and validate the comparison before deleting any model files.
    immutable_json(destination / "nemotron-baseline.json", freeze_baseline(args.baseline.resolve()))
    inventory = model_inventory(HfApi(), MODELS)
    immutable_json(destination / "models.json", inventory)
    if args.remove_nemotron:
        remove_nemotron(workspace)
    quota = capture_quota(workspace, args.account)
    atomic(destination / "quota.json", canonical(quota))
    verified = download_models(inventory, workspace / "models", quota, models=MODELS)
    atomic(destination / "models-verified.json", canonical(verified))
    print(json.dumps(verified, indent=2))
    return 0


def runtime_check():
    """Login-safe static check; full GPU compatibility remains a compute-node check."""
    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.submodule_search_locations:
        raise ValueError("vLLM is not installed in this Python environment")
    registry = Path(next(iter(spec.submodule_search_locations))) / "model_executor/models/registry.py"
    if "Gemma4ForConditionalGeneration" not in registry.read_text():
        raise ValueError("installed vLLM does not register Gemma4ForConditionalGeneration; use a compatible isolated runtime")
    return {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "xgrammar")}


def prepare(args):
    if not args.approval.strip():
        raise ValueError("record explicit approval of this allocation and wall-time")
    repo = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("clean committed checkout required")
    workspace, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("output exists; inspect it instead of overwriting")
    source = prep(workspace)
    verified = runner.read(source / "models-verified.json")
    if verified.get("status") != "verified" or [(m["repo_id"], m["revision"]) for m in verified["models"]] != list(MODELS):
        raise ValueError("download and verify pinned Gemma first")
    for model in verified["models"]:
        if not Path(model["snapshot"]).is_dir():
            raise ValueError("verified model snapshot missing")
    versions = runtime_check()
    content = (source / "nemotron-baseline.json").read_bytes()
    # No GPU work until the existing structured-output dependency accepts SI-v3.
    subprocess.run([sys.executable, str(repo / "analysis/scripts/try_source_importance_judge.py"),
                    "--check-schema"], check=True)
    config = {"repository": str(repo), "git_commit": pin, "workspace": str(workspace),
        "count": 20, "max_tokens": 640, "cache_prefix": "gemma4",
        "trial": "source-importance", "judge_protocol": PROTOCOL,
        "model_id": MODEL, "model_revision": REVISION, "serving": runner.SERVING,
        "job_name": JOB_NAME, "walltime": args.walltime, "approval": args.approval,
        "replay_inputs": str(out / "inputs.json"), "replay_inputs_sha256": hashlib.sha256(content).hexdigest(),
        "runtime": versions, "scientific_result": False,
        "estimate": {"minutes": [15, 45], "basis": "Nemotron cold startup 11.6 min; Gemma dense throughput unmeasured",
                     "nodes": 1, "gpus": 4, "cpus_requested": 32, "memory": "whole node (512 GiB)",
                     "maximum_gpu_hours": 4 * runner._hours(args.walltime)}}
    out.mkdir(parents=True)
    atomic(out / "inputs.json", content)
    atomic(out / "config.json", canonical(config))
    command = [sys.executable, str(repo / "analysis/scripts/horeka_nemotron.py"), "execute", "--config", str(out / "config.json")]
    atomic(out / "run.sh", ("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n").encode())
    os.chmod(out / "run.sh", 0o755)
    print(json.dumps({"prepared": str(out), "commit": pin, "runtime": versions,
                      "allocation_submitted": False}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--workspace", type=Path, required=True)
    d.add_argument("--account", required=True)
    d.add_argument("--baseline", type=Path, required=True)
    d.add_argument("--remove-nemotron", action="store_true", help="delete only Nemotron weights after preserving the pilot")
    p = sub.add_parser("prepare")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--walltime", choices=("00:30:00", "00:45:00", "01:00:00"), required=True)
    p.add_argument("--approval", required=True)
    args = parser.parse_args(argv)
    return download(args) if args.command == "download" else prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
