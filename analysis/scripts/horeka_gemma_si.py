#!/usr/bin/env python3
"""Prepare a pinned Gemma 4 31B SI-v3 replay. Never submits or acquires an allocation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
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


def run_script(repo, out):
    command = [sys.executable, str(repo / "analysis/scripts/horeka_nemotron.py"), "execute", "--config", str(out / "config.json")]
    return ("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n").encode()


def input_path(args):
    return args.inputs.resolve() if getattr(args, "inputs", None) else prep(args.workspace.resolve()) / "nemotron-baseline.json"


def reuse_prepared(args):
    """Verify the saved run, preserving its original code pin, approval and artifacts."""
    out, workspace = args.output.resolve(), args.workspace.resolve()
    if not all((out / name).is_file() for name in ("config.json", "inputs.json", "run.sh")):
        raise ValueError("existing preparation is incomplete; inspect it without overwriting")
    config = runner.read(out / "config.json")
    expected = {"workspace": str(workspace), "walltime": args.walltime, "model_id": MODEL,
        "model_revision": REVISION, "job_name": JOB_NAME, "judge_protocol": PROTOCOL,
        "trial": "source-importance", "count": 20, "max_tokens": 640, "serving": runner.SERVING,
        "replay_inputs": str(out / "inputs.json"), "existing_job_id": getattr(args, "existing_job_id", None)}
    if any(config.get(k) != v for k, v in expected.items()) or not config.get("approval"):
        raise ValueError("existing preparation conflicts with the requested run")
    content = (out / "inputs.json").read_bytes()
    if hashlib.sha256(content).hexdigest() != config["replay_inputs_sha256"]:
        raise ValueError("existing replay input checksum changed")
    if content != input_path(args).read_bytes():
        raise ValueError("existing preparation conflicts with the frozen baseline")
    repo = Path(config["repository"])
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != config["git_commit"]:
        raise ValueError("saved execution checkout no longer matches its code pin")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("saved execution checkout is dirty")
    if (out / "run.sh").read_bytes() != run_script(repo, out):
        raise ValueError("saved run.sh differs from the frozen command or Python interpreter")
    print(json.dumps({"status": "reused", "prepared": str(out), "commit": config["git_commit"],
                      "allocation_submitted": False, "next": "check"}, indent=2))
    return 0


def prepare(args):
    if not args.approval.strip():
        raise ValueError("record explicit approval of this allocation and wall-time")
    bound = getattr(args, "existing_job_id", None)
    if bound:
        fields = dict(t.split("=", 1) for t in subprocess.check_output(
            ["scontrol", "show", "job", str(bound), "-o"], text=True).split() if "=" in t)
        runner.check_bound_allocation({"existing_job_id": bound, "minimum_remaining_seconds": 1200},
                                      os.environ.get("SLURM_JOB_ID"), fields)
        if fields.get("JobName") != JOB_NAME or fields.get("TimeLimit") != args.walltime:
            raise ValueError("existing allocation does not match the Gemma run")
    repo = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("clean committed checkout required")
    workspace, out = args.workspace.resolve(), args.output.resolve()
    if out.exists():
        return reuse_prepared(args)
    source = prep(workspace)
    verified = runner.read(source / "models-verified.json")
    if verified.get("status") != "verified" or [(m["repo_id"], m["revision"]) for m in verified["models"]] != list(MODELS):
        raise ValueError("download and verify pinned Gemma first")
    for model in verified["models"]:
        if not Path(model["snapshot"]).is_dir():
            raise ValueError("verified model snapshot missing")
    versions = runtime_check()
    content = input_path(args).read_bytes()
    from analysis.scripts.replay_source_importance_judge import frozen_cells
    frozen = frozen_cells(json.loads(content))
    if len(frozen) != 20 or sorted(c["model"] for c, _ in frozen) != ["llama4"] * 10 + ["qwen38"] * 10:
        raise ValueError("Gemma pilot requires ten frozen Qwen and ten frozen Llama cells")
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
    if bound:
        config.update(existing_job_id=bound, minimum_remaining_seconds=1200,
                      cache_prefix="gemma4-fresh20-" + hashlib.sha256(str(out).encode()).hexdigest()[:12],
                      estimate={"minutes": [10, 15], "minimum_remaining_minutes": 20,
                                "basis": "first 20-cell Gemma replay: 464s stage run, 92.3s judging; frozen inputs prepared separately",
                                "new_allocation": False, "nodes": 1, "gpus": 4, "cpus_requested": 32,
                                "memory": "whole node", "additional_allocation_hours": 0})
    out.mkdir(parents=True)
    atomic(out / "inputs.json", content)
    atomic(out / "config.json", canonical(config))
    atomic(out / "run.sh", run_script(repo, out))
    os.chmod(out / "run.sh", 0o755)
    print(json.dumps({"prepared": str(out), "commit": pin, "runtime": versions,
                      "allocation_submitted": False}, indent=2))
    return 0


def check(args):
    """Read-only Slurm checks for the single approved scheduling exception. Never allocates."""
    from analysis.scripts.capture_agentic_scheduler_snapshot import capture
    from analysis.scripts.prepare_horeka_qwen import capture_quota
    from analysis.scripts.manage_agentic_hours import health
    workspace, out = args.workspace.resolve(), args.output.resolve()
    config = json.loads((out / "config.json").read_text())
    if out.name != "gemma4-si-v3-ddc93fe" or config.get("job_name") != "geodml-gemma-si-replay" or config.get("walltime") != "01:00:00":
        raise SystemExit("Exception applies only to the already approved one-hour Gemma run.")
    if (out / "ALLOCATION_ATTEMPTED").exists() or any((out / "attempts").glob("job*")):
        raise SystemExit("A previous allocation attempt exists. Inspect it; do not request another.")
    queued = subprocess.check_output(["squeue", "--account=" + args.account, "--array", "--noheader", "--format=%i"], text=True)
    queue_count = len(set(queued.split()))
    if queue_count >= 295:
        raise SystemExit(f"Account queue is {queue_count}/295. Wait for a free slot; leave all jobs unchanged.")
    since = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    live = subprocess.check_output(["squeue", "--me", "--noheader", "--format=%i"], text=True)
    history = subprocess.check_output(["sacct", "-X", "--noheader", "--parsable2", "--starttime=" + since, "--format=JobIDRaw"], text=True)
    ids = sorted({line.strip().split("|")[0] for line in (live + "\n" + history).splitlines() if line.strip()})
    snapshot = capture(plan={"plan_id": "gemma-si-replay"}, since=since, include_job_ids=ids)
    if snapshot.get("complete") is not True or not 0 <= time.time() - snapshot["captured_at_epoch"] <= 120:
        raise SystemExit("Fresh complete scheduler evidence required.")
    if any(row.get("job_name") == "geodml-gemma-si-replay" for row in snapshot["jobs"]):
        raise SystemExit("A Gemma allocation is already queued or active. Reuse it; do not request another.")
    starts = [r["start_epoch"] for r in snapshot["jobs"] + snapshot.get("owners", []) if isinstance(r.get("start_epoch"), int)]
    remaining = 600 - (int(time.time()) - max(starts, default=0))
    if remaining > 0:
        raise SystemExit(f"Wait at least {remaining} seconds on the login host, then repeat this check. Ten-minute start-gap rule retained.")
    quota = out / "admission-quota.json"
    quota.write_text(json.dumps(capture_quota(workspace, args.account), indent=2))
    storage = health({"dataset_root": str(workspace), "cluster": "horeka"}, quota)
    if storage.get("safe_to_admit") is not True or storage.get("quota_verified") is not True:
        raise SystemExit("Storage check blocked admission. Leave all jobs unchanged.")
    evidence = {"scheduler": snapshot, "storage": storage, "account_queue_count": queue_count,
        "approved_exception": "Valerian approved one 01:00:00 Gemma allocation alongside the existing Qwen queue; waive five-active and no-pending guards for this run only. Cancel, hold or modify no jobs.",
        "walltime_seconds": 3600, "maximum_gpu_hours": 4, "checked_at_epoch": int(time.time())}
    (out / "admission.json").write_text(json.dumps(evidence, indent=2))
    print(f"CHECK PASSED: account queue {queue_count}/295. Now run the separate salloc command once.", flush=True)
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
    p.add_argument("--inputs", type=Path, help="verified fresh 20-cell bundle instead of the Nemotron baseline")
    p.add_argument("--existing-job-id", help="bind execution to this already approved live allocation")
    c = sub.add_parser("check", help="check admission for the approved Gemma run; never submits")
    c.add_argument("--workspace", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--account", required=True)
    args = parser.parse_args(argv)
    return {"download": download, "prepare": prepare, "check": check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
