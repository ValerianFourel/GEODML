"""Prepare a first dataset-backed Llama allocation, without fabricating calibration.

Preparation registers the existing frozen population and accepts exact recovered
results. Submission is a separate, explicitly approved, single allocation.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline.agentic_audit_progress import audit_stage
from analysis.interpretability.pipeline.agentic_hour_sync import HubStore, atomic
from analysis.interpretability.pipeline.agentic_hours import (
    canonical,
    digest,
    inventory,
)
from analysis.interpretability.pipeline.agentic_storage import storage_health
from analysis.interpretability.pipeline.agentic_verification_cache import (
    VerificationCache,
)
from analysis.scripts.accept_agentic_recovery_generations import accept
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
from analysis.scripts.prepare_agentic_qwen_inputs import identity
from analysis.scripts.register_agentic_dataset_tasks import register_generator_tasks
from analysis.scripts.run_agentic_search_integration_smoke import SmokeInputs
from analysis.scripts.search_vllm_stage import load_profile
from analysis.scripts.submit_agentic_paired_trial import _snapshot

MODEL = {"model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
         "model_revision": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"}
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FILES = ("SEARCH_AGENTIC_DDG_SNAPSHOT", "SEARCH_AGENTIC_SEARXNG_SNAPSHOT",
         "SEARCH_AGENTIC_PROMPTS_JSONL", "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL")
ESTIMATE = ("One bounded first Llama allocation, not a full-population completion estimate. "
            "No validated throughput for this backlog. Up to 55 minutes of startup and "
            "task admission, five-minute drain including two-minute cleanup. "
            "One node, four GH200, 32 requested CPUs, all node memory; maximum four GPU-hours.")


def read(path):
    return json.loads(Path(path).read_bytes())


def command(argv):
    options = {}
    if argv[0] == 'sbatch':
        options['env'] = {k: v for k, v in os.environ.items()
                          if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_API_TOKEN'}}
    return subprocess.run(argv, check=True, text=True, capture_output=True, **options).stdout.strip()


def clean_commit():
    sha = command(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"])
    if command(["git", "-C", str(REPOSITORY), "status", "--porcelain", "--untracked-files=all"]):
        raise ValueError("use a clean pinned checkout")
    return sha


def inputs_from_runtime(root, runtime, profile):
    if profile["model"] != MODEL:
        raise ValueError("unexpected Llama model/revision")
    serving = profile["serving"]
    expected = {"tensor_parallel_size": 4, "data_parallel_size": 1,
                "dtype": "bfloat16", "max_model_len": 40960, "request_concurrency": 4}
    if any(serving[key] != value for key, value in expected.items()):
        raise ValueError("Llama serving settings differ from the maintained worker")
    if runtime["SEARCH_AGENTIC_CROSS_ENCODER_REVISION"] != BGE_REVISION:
        raise ValueError("unexpected reranker revision")
    if int(runtime["SEARCH_AGENTIC_PROMPT_SELECTION_SEED"]) != 20260912:
        raise ValueError("unexpected population selection seed")
    return SmokeInputs(
        output=root / "local-only/llama-registration", base_url="http://127.0.0.1:8010/v1",
        **MODEL, cross_encoder_snapshot=Path(runtime["SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT"]),
        cross_encoder_revision=BGE_REVISION,
        search_snapshots={"duckduckgo": Path(runtime[FILES[0]]), "searxng": Path(runtime[FILES[1]])},
        seed=20260911, max_tokens=1024, query_max_tokens=256, final_max_tokens=4096,
        request_concurrency=4, cell_concurrency=12, disable_thinking=False,
        prompts_jsonl=Path(runtime[FILES[2]]), selection_records_jsonl=Path(runtime[FILES[3]]),
        prompt_count=int(runtime["SEARCH_AGENTIC_PROMPT_COUNT"]),
        prompt_selection_seed=20260912, production_conditions=True,
    )


@audit_stage("freeze_llama_backlog")
def freeze_backlog(root, output, *, stripes=256, model="llama4", local_inventory=None):
    tasks, completed, blocked = (inventory(root, stripes=stripes, reuse_verified=True)
                                 if local_inventory is None else local_inventory)
    tasks = [row for row in tasks if row["model"] == model]
    if not tasks or len({row["configuration_sha256"] for row in tasks}) != 1:
        raise ValueError("expected one registered Llama configuration")
    eligible = [row for row in tasks if row["fingerprint"] not in completed | blocked
                and not row.get("blocked_reason")
                and set(row.get("dependency_fingerprints", [])) <= completed]
    eligible.sort(key=lambda row: (row["priority_rank"], row["prompt_id"], row["task_id"]))
    if not eligible:
        raise ValueError("no eligible missing Llama cells")
    raw = b"".join(canonical({**row["runnable_task"], "geodml_keyword_id": row["keyword_id"],
                              "geodml_task_fingerprint": row["fingerprint"]}) + b"\n" for row in eligible)
    path = output / "wave/backlog.jsonl"
    atomic(path, raw)
    manifest = {"format_version": "geodml-inference-wave-v2", "status": "planned",
                "dispatch_mode": "backlog", "model": model, "direction": "forward",
                "backlog": {"path": str(path), **identity(path), "task_count": len(eligible)}}
    atomic(path.parent / "run_manifest.json", canonical(manifest))
    return {"registered": len(tasks), "completed": sum(t["fingerprint"] in completed for t in tasks),
            "blocked": sum(t["fingerprint"] in blocked for t in tasks), "eligible": len(eligible),
            "configuration_sha256": tasks[0]["configuration_sha256"],
            "hour_packages": None, "calibration_status": "measurement_required"}


def verify_prepared(output):
    record = read(output / "preparation.json")
    if record["git_commit"] != clean_commit():
        raise ValueError("preparation belongs to a different code commit")
    with VerificationCache(output) as verification:
        for path, expected in record["files"].items():
            if not verification.file(Path(path), expected):
                raise ValueError(f"prepared file changed: {path}")
    # External frozen inputs have their own root-bound receipt.
    root = Path(record["dataset_root"])
    with VerificationCache(root) as verification:
        for path, expected in record["dataset_files"].items():
            if not verification.file(Path(path), expected):
                raise ValueError(f"dataset input changed: {path}")
    for path, expected in record["external_files"].items():
        if identity(Path(path)) != expected:
            raise ValueError(f"input changed: {path}")
    return record


@audit_stage("prepare_llama")
def prepare(args):
    root, output = args.dataset.resolve(), args.output.resolve()
    profile = load_profile(args.profile)
    runtime = read(args.reference_runtime)
    inputs = inputs_from_runtime(root, runtime, profile)
    pin = clean_commit()
    if output.exists():
        raise FileExistsError("preparation already exists; use submit/status, not a new preparation")
    environment = {key: os.environ[key] for key in ("ACL_ARR_VENV", "HF_HUB_CACHE", "GEODML_CACHE_ROOT")}
    for key, value in environment.items():
        if not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError(f"missing absolute directory: {key}")
    if importlib.metadata.version("sentence-transformers") != "6.0.1":
        raise ValueError("sentence-transformers 6.0.1 required")
    if importlib.metadata.version("vllm") != profile["runtime"]["vllm_version"]:
        raise ValueError("vLLM version differs from the reference profile")
    model = _snapshot(Path(environment["HF_HUB_CACHE"]), MODEL["model_id"], MODEL["model_revision"])
    for path in [*(Path(runtime[key]) for key in FILES), args.keyword_priority,
                 args.recovery_hub / "publication-manifest.json",
                 inputs.cross_encoder_snapshot / "model.safetensors"]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing required input: {path}")
    output.mkdir(parents=True)
    atomic(output / "profile.json", args.profile.read_bytes())
    selected_files = [*(Path(runtime[key]).resolve() for key in FILES), args.keyword_priority.resolve()]
    proofs = {str(path): identity(path) for path in selected_files}
    registration = register_generator_tasks(
        dataset_root=root, model_slug="llama4", inputs=inputs,
        keyword_priority_path=args.keyword_priority, writer_id="register-llama4-" + digest(proofs)[:16])
    atomic(output / "registration.json", canonical(registration))
    recovery = audit_stage("accept_llama_recovery")(accept)(root, args.recovery_hub, model="llama4")
    atomic(output / "recovery.json", canonical(recovery))
    summary = freeze_backlog(root, output)
    environment.update({key: runtime[key] for key in (*FILES, "SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT",
                        "SEARCH_AGENTIC_CROSS_ENCODER_REVISION", "SEARCH_AGENTIC_PROMPT_COUNT",
                        "SEARCH_AGENTIC_PROMPT_SELECTION_SEED")})
    environment.update({
        "GEODML_EXECUTION_REPOSITORY": str(REPOSITORY), "GEODML_EXECUTION_COMMIT": pin,
        "GEODML_MODEL_SLUG": "llama4", "GEODML_DATASET_ROOT": str(root),
        "GEODML_WAVE_ROOT": str(output / "wave"), "GEODML_WAVE_OUTPUT_ROOT": str(output / "results"),
        "GEODML_WAVE_LOG_ROOT": str(output / "logs"),
        "GEODML_WORKER_LAUNCHER": str(REPOSITORY / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"),
        "GEODML_START_MARGIN_SECONDS": "300", "GEODML_CLEANUP_MARGIN_SECONDS": "120",
        "GEODML_WORKER_INDEX": "0", "GEODML_WORKER_COUNT": "1",
        "GEODML_WORKER_STDOUT": str(output / "slurm-%j.out"),
        "GEODML_WORKER_STDERR": str(output / "slurm-%j.err"),
        "SEARCH_AGENTIC_PROFILE": str(output / "profile.json"),
        "SEARCH_AGENTIC_CELL_CONCURRENCY": "12", "SEARCH_AGENTIC_REQUEST_CONCURRENCY": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    atomic(output / "runtime.json", canonical(environment))
    run = ("#!/usr/bin/env bash\nset -eo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\n"
           "if ! type module >/dev/null 2>&1; then source /etc/profile; fi\n"
           "module load Stages/2026 GCC Python CUDA git\nsource " +
           shlex.quote(environment["ACL_ARR_VENV"] + "/bin/activate") + "\nexec python3 " +
           shlex.quote(str(Path(__file__).resolve())) + " execute --output " + shlex.quote(str(output)) + "\n")
    atomic(output / "run.sh", run.encode())
    for path, expected in proofs.items():
        if identity(Path(path)) != expected:
            raise ValueError(f"input changed during registration: {path}")
    record = {"git_commit": pin, "dataset_root": str(root), "model_snapshot": model,
              "summary": summary, "files": {str(path): identity(path) for path in output.rglob("*") if path.is_file()},
              "dataset_files": {p: v for p, v in proofs.items() if Path(p).is_relative_to(root)},
              "external_files": {p: v for p, v in proofs.items() if not Path(p).is_relative_to(root)}}
    atomic(output / "preparation.json", canonical(record))
    verify_prepared(output)
    return {"status": "prepared", **summary, "output": str(output), "allocation_submitted": False}


def scheduler_gate(snapshot, now):
    if snapshot.get("complete") is not True or not 0 <= now - snapshot["captured_at_epoch"] <= 120:
        raise ValueError("fresh complete scheduler snapshot required")
    # This first measurement deliberately starts with no other GEODML allocations.
    if snapshot["jobs"]:
        raise ValueError("GEODML allocations are present; preserve them and review admission")
    starts = [row["start_epoch"] for row in snapshot["owners"] if row.get("start_epoch")]
    if starts and now - max(starts) < 600:
        raise ValueError("wait until ten minutes after the last observed start")


def current_scheduler(since):
    # Include unnamed interactive allocations as well as geodml-* batch jobs.
    ids = command(["squeue", "--me", "--noheader", "--format=%i"]).splitlines()
    return capture(plan={"plan_id": "llama-first-allocation"}, since=since, include_job_ids=ids)


def submit(args):
    output = args.output.resolve()
    record = verify_prepared(output)
    model = record.get("model", "llama4")
    walltime = record.get("approved_walltime", "01:00:00")
    if model not in {"llama4", "qwen38"} or walltime not in {"01:00:00", "03:00:00"}:
        raise ValueError("unsupported first measurement")
    if args.approved_walltime != walltime:
        raise ValueError("this first allocation requires its explicit one-hour approval")
    gpu_hours = 12 if walltime == "03:00:00" else 4
    if not args.approval.strip():
        raise ValueError("explicit approval evidence required")
    root = Path(record["dataset_root"])
    # A dataset-wide receipt also prevents a second attempt directory from creating
    # another bootstrap allocation. Any further job needs a measured new plan.
    label = "llama" if model == "llama4" else "qwen38"
    receipt = root / f"control/{label}-first-allocation.json"
    with (root / f"control/{label}-first-allocation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            raise ValueError(f"allocation already requested; inspect {receipt}; do not resubmit")
        store = HubStore(args.hf_repo)
        revision = store.head()
        registry = json.loads(store.read("coordination/hours.json", revision))
        if any(hour.get("model") == model for hour in registry["hours"].values()):
            raise ValueError("shared Llama hour packages exist; use their coordinated dispatcher")
        snapshot = current_scheduler(args.since)
        scheduler_gate(snapshot, int(time.time()))
        storage = storage_health(root)
        if not storage["safe_to_admit"]:
            raise ValueError(f"storage admission blocked: {storage['reasons']}")
        # Exercise file creation/fsync on the output filesystem; quota exhaustion
        # can reject writes even when filesystem-wide df reports free capacity.
        with tempfile.TemporaryFile(dir=root / "control") as probe:
            probe.write(b"geodml-storage-probe\n")
            probe.flush()
            os.fsync(probe.fileno())
        atomic(output / "scheduler.json", canonical(snapshot))
        atomic(output / "storage.json", canonical(storage))
        approval = {"evidence": args.approval, "walltime": args.approved_walltime,
                    "maximum_gpu_hours": gpu_hours, "estimate": record.get("estimate", ESTIMATE),
                    "resources": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "all"},
                    "hf_revision": revision, "hf_plan_updated": False}
        atomic(output / "approval.json", canonical(approval))
        argv = ["sbatch", "--parsable", "--hold", "--no-requeue", "--nodes=1", "--ntasks=1",
                "--gres=gpu:4", "--cpus-per-task=32", "--mem=0", "--exclusive",
                "--time=" + args.approved_walltime, "--account=" + args.account,
                "--partition=" + args.partition, "--job-name=geodml-" + label + "-first-hour",
                "--chdir=" + str(REPOSITORY), "--output=" + str(output / "slurm-%j.out"),
                "--error=" + str(output / "slurm-%j.err"), str(output / "run.sh")]
        request = {"status": "submission_requested", "output": str(output), "command": argv,
                   "git_commit": record["git_commit"], "approval": approval}
        atomic(receipt, canonical(request))
        atomic(output / "submission.json", canonical(request))
        # Persist intent before contacting Slurm; uncertain responses never retry.
        raw = command(argv)
        atomic(output / "sbatch-response.txt", raw.encode())
        job = raw.split(";")[0]
        if not re.fullmatch(r"[0-9]+", job):
            raise ValueError("uncertain submission response; inspect Slurm, do not retry")
        request.update(status="submitted_held", job_id=job)
        atomic(receipt, canonical(request))
        atomic(output / "submission.json", canonical(request))
        snapshot = current_scheduler(args.since)
        own = [row for row in snapshot["jobs"] if row["job_id"] == job]
        if len(own) != 1 or not own[0].get("held"):
            raise ValueError(f"job {job} cannot be confirmed held; inspect Slurm")
        scheduler_gate({**snapshot, "jobs": [row for row in snapshot["jobs"] if row["job_id"] != job]}, int(time.time()))
        command(["scontrol", "release", job])
        request["status"] = "submitted"
        atomic(receipt, canonical(request))
        atomic(output / "submission.json", canonical(request))
        return {"job_id": job, "output": str(output), "walltime": args.approved_walltime,
                "maximum_gpu_hours": gpu_hours, "eligible_cells": record["summary"]["eligible"]}


def execute(output):
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify("jupiter")  # Check the compute node before artifact reads or model loading.
    record = verify_prepared(output)
    submission = read(output / "submission.json")
    job = os.environ["SLURM_JOB_ID"]
    if submission.get("job_id") != job:
        raise ValueError("allocation differs from this submission receipt")
    info = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(token.split("=", 1) for token in info.split() if "=" in token)
    if fields.get("TimeLimit") != record.get("approved_walltime", "01:00:00") or fields.get("NumNodes") != "1":
        raise ValueError("allocation differs from approved one-hour single-node resources")
    times = {variable: str(int(datetime.fromisoformat(fields[field]).timestamp()))
             for field, variable in (("StartTime", "SLURM_JOB_START_TIME"), ("EndTime", "SLURM_JOB_END_TIME"))}
    atomic(output / "boundary.json", canonical(boundary))
    atomic(output / "allocation.json", canonical({"slurm": fields, "boundary": boundary}))
    runtime = read(output / "runtime.json")
    for name in list(os.environ):
        if name.startswith(("GEODML_", "SEARCH_AGENTIC_")):
            del os.environ[name]
    os.environ.update(runtime)
    os.environ.update(times)
    os.environ.update(GEODML_APPROVED_WALLTIME=submission["approval"]["walltime"],
                      GEODML_ALLOCATION_ESTIMATE=submission["approval"]["estimate"])
    if record["git_commit"] != runtime["GEODML_EXECUTION_COMMIT"]:
        raise ValueError("runtime commit mismatch")
    os.execv("/bin/bash", ["bash", str(REPOSITORY / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("dataset", "reference-runtime", "profile", "keyword-priority", "recovery-hub", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("--output", type=Path, required=True)
    for name in ("account", "partition", "since", "hf-repo", "approval"):
        launch.add_argument("--" + name, required=True)
    launch.add_argument("--approved-walltime", required=True, choices=["01:00:00"])
    run = sub.add_parser("execute")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else submit(args) if args.command == "submit" else execute(args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
