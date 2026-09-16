#!/usr/bin/env bash
# Submission supplies the specifically approved time and one-node resources.
# Historical fixed mode remains 24 tasks and one hour. The queue wrapper selects
# throughput mode with an allocation deadline and all available planned work.
set +u
set -eo pipefail
export DEBUGINFOD_URLS="${DEBUGINFOD_URLS:-}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module load Stages/2026 GCC Python CUDA
module load git
source "${ACL_ARR_VENV:?}/bin/activate"
set -u

: "${SLURM_JOB_ID:?Run through Slurm}"
: "${GEODML_EXECUTION_REPOSITORY:?}"
: "${GEODML_EXECUTION_COMMIT:?}"
: "${GEODML_JUDGE_PILOT_ROOT:?Use a new pilot directory}"
: "${GEODML_CACHE_ROOT:?}"
: "${HF_HUB_CACHE:?}"
: "${GEODML_APPROVED_WALLTIME:?}"
: "${GEODML_ALLOCATION_ESTIMATE:?}"
export GEODML_JUDGE_EXECUTION_MODE="${GEODML_JUDGE_EXECUTION_MODE:-fixed}"
export GEODML_JUDGE_PLAN_ROOT="${GEODML_JUDGE_PLAN_ROOT:-$GEODML_JUDGE_PILOT_ROOT/plan}"
export GEODML_JUDGE_WORKER_INDEX="${GEODML_JUDGE_WORKER_INDEX:-0}"
export GEODML_JUDGE_WORKER_COUNT="${GEODML_JUDGE_WORKER_COUNT:-1}"
export GEODML_JUDGE_DISPATCH_MODE="${GEODML_JUDGE_DISPATCH_MODE:-partition}"
geodml_claim_args=(--worker-index "$GEODML_JUDGE_WORKER_INDEX" --worker-count "$GEODML_JUDGE_WORKER_COUNT" --dispatch-mode "$GEODML_JUDGE_DISPATCH_MODE")
case "$GEODML_JUDGE_EXECUTION_MODE" in
  fixed) test "$GEODML_APPROVED_WALLTIME" = 01:00:00 ;;
  throughput)
    : "${SLURM_JOB_END_TIME:?Throughput mode requires the actual allocation end time}"
    : "${GEODML_JUDGE_CLAIM_ROOT:?All related judge jobs must share one claim directory}"
    [[ "$GEODML_JUDGE_CLAIM_ROOT" = /* ]] || { printf '%s\n' 'Claim root must be an absolute shared path.' >&2; exit 2; }
    geodml_claim_args+=(--claim-root "$GEODML_JUDGE_CLAIM_ROOT")
    ;;
  *) printf 'Unknown judge execution mode: %s\n' "$GEODML_JUDGE_EXECUTION_MODE" >&2; exit 2 ;;
esac
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_GPUS_ON_NODE:?}" = 4
test "${SLURM_CPUS_PER_TASK:?}" = 32
test "${SLURM_MEM_PER_NODE:?}" = 524288
test "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)" = "$GEODML_EXECUTION_COMMIT"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"
cd "$GEODML_EXECUTION_REPOSITORY"

test -s "$GEODML_JUDGE_PLAN_ROOT/run_manifest.json"
test -s "$GEODML_JUDGE_PLAN_ROOT/bulk_tasks.jsonl"
mkdir -p "$GEODML_JUDGE_PILOT_ROOT"
command -v flock >/dev/null
exec 9>>"$GEODML_JUDGE_PILOT_ROOT/.pilot.lock"
if ! flock -n 9; then
  printf 'Pilot already has an active worker: %s\n' "$GEODML_JUDGE_PILOT_ROOT" >&2
  exit 75
fi

geodml_log_root="$GEODML_JUDGE_PILOT_ROOT/logs"
geodml_profile="$GEODML_JUDGE_PILOT_ROOT/serving-profile.json"
mkdir -p "$geodml_log_root"
test ! -e "$geodml_log_root/server.log"
test ! -e "$geodml_log_root/gpu.csv"
test ! -e "$geodml_log_root/allocation.json"
test ! -e "$geodml_profile"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GEODML_PILOT_MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
export GEODML_PILOT_MODEL_REVISION=bf77c3174f68ad409e1c2aa60daeb46e32d1c606
export PYTHONUNBUFFERED=1

python3 - <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re

import xgrammar
from analysis.interpretability.pipeline.agentic_judging import (
    TRANSCRIPT_FORMAT_VERSION,
    agentic_judge_schema,
)
from analysis.interpretability.pipeline.inference_budget import AllocationBudget
from analysis.scripts.run_acl_arr_vllm import _agentic_judge_context, task_in_worker

env = os.environ
root = Path(env["GEODML_JUDGE_PILOT_ROOT"])
plan_root = Path(env["GEODML_JUDGE_PLAN_ROOT"])
manifest_path = plan_root / "run_manifest.json"
queue = plan_root / "bulk_tasks.jsonl"
manifest = json.loads(manifest_path.read_text())
mode = env["GEODML_JUDGE_EXECUTION_MODE"]
budget = AllocationBudget.from_environment(require=mode == "throughput")
if mode == "throughput" and not budget.can_start():
    raise ValueError("Insufficient allocation time remains to start the judge server")
tasks, _, model_id, revision = _agentic_judge_context(
    manifest_path, queue, judge_role="bulk"
)
if (model_id, revision) != (
    env["GEODML_PILOT_MODEL_ID"], env["GEODML_PILOT_MODEL_REVISION"]
):
    raise ValueError("Pilot judge identity differs from the approved Nemotron snapshot")
if len({task.judge_task_id for task in tasks}) != len(tasks):
    raise ValueError("Judge task IDs must be unique")
if mode == "fixed" and len(tasks) != 24:
    raise ValueError("This pilot requires exactly 24 unique judge tasks")
if mode == "throughput":
    if manifest.get("pilot", {}).get("execution_mode") != "throughput":
        raise ValueError("Throughput mode requires an all-available throughput plan")
    summary = manifest.get("summary", {})
    if (summary.get("pending_task_count") != len(tasks)
            or summary.get("bulk_task_count") != len(tasks)
            or summary.get("available_task_count") != len(tasks) + summary.get("excluded_task_count", -1)):
        raise ValueError("Throughput queue coverage counts disagree")
if manifest.get("scientific_result") is not False or manifest.get("pilot_only") is not True:
    raise ValueError("A judge pilot must not claim scientific results")
source_task_count = len(tasks)
worker_index = int(env["GEODML_JUDGE_WORKER_INDEX"])
worker_count = int(env["GEODML_JUDGE_WORKER_COUNT"])
dispatch_mode = env["GEODML_JUDGE_DISPATCH_MODE"]
if dispatch_mode not in {"partition", "backlog"}:
    raise ValueError("Unknown judge dispatch mode")
if mode == "fixed" and (worker_index, worker_count) != (0, 1):
    raise ValueError("Historical fixed pilot cannot be divided into worker slots")
if mode == "fixed" and dispatch_mode != "partition":
    raise ValueError("Historical fixed pilot must retain partition dispatch")
task_in_worker("validate-slot", worker_index, worker_count)
if dispatch_mode == "partition":
    tasks = [task for task in tasks if task_in_worker(task.judge_task_id, worker_index, worker_count)]
if not tasks:
    raise ValueError("This worker slot has no tasks; model was not loaded")
snapshot = Path(env["HF_HUB_CACHE"]) / (
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/" + revision
)
config = json.loads((snapshot / "config.json").read_text())
if config.get("architectures") != ["NemotronHForCausalLM"]:
    raise ValueError("Unexpected cached Nemotron architecture")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
weights = sorted(set(index["weight_map"].values()))
if len(weights) != 13:
    raise ValueError("The pinned Nemotron snapshot must contain 13 weight shards")
for name in weights + ["config.json", "tokenizer.json", "tokenizer_config.json"]:
    if Path(name).name != name or not (snapshot / name).is_file():
        raise ValueError(f"Missing or invalid cached model file: {name}")
    if (snapshot / name).stat().st_size <= 0:
        raise ValueError(f"Empty cached model file: {name}")
context_budget = None
if manifest["format_version"] == TRANSCRIPT_FORMAT_VERSION:
    from transformers import AutoTokenizer
    from analysis.scripts.check_agentic_judge_context import check_agentic_judge_context_budget

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
    )
    context_budget = check_agentic_judge_context_budget(
        tasks, tokenizer=tokenizer, max_model_len=16384,
        max_output_tokens=2048, disable_thinking=True,
    )
    context_budget = {key: value for key, value in context_budget.items() if key != "tasks"}
for task in tasks:
    schema = agentic_judge_schema([item.evidence_id for item in task.evidence])
    xgrammar.Grammar.from_json_schema(json.dumps(schema))
print(f"NEMOTRON_PILOT_PREFLIGHT=PASS mode={mode} tasks={len(tasks)} weight_shards=13 schemas={len(tasks)}", flush=True)
versions = {}
for package in ("vllm", "transformers", "huggingface-hub", "xgrammar", "torch"):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
walltime = env["GEODML_APPROVED_WALLTIME"]
days, separator, clock = walltime.partition("-")
if not separator:
    days, clock = "0", days
parts = [int(part) for part in clock.split(":")]
if len(parts) != 3 or int(days) < 0 or parts[0] < 0 or not all(0 <= part < 60 for part in parts[1:]):
    raise ValueError("Approved walltime must use [days-]HH:MM:SS")
approved_seconds = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
if approved_seconds <= 0:
    raise ValueError("Approved walltime must be positive")


def log_path(key, suffix):
    value = env.get(key, str(root / "logs" / f"slurm-{env['SLURM_JOB_ID']}.{suffix}"))
    replacements = {
        "%%": "%", "%j": env["SLURM_JOB_ID"],
        "%A": env.get("SLURM_ARRAY_JOB_ID", env["SLURM_JOB_ID"]),
        "%a": env.get("SLURM_ARRAY_TASK_ID", "0"),
    }
    return re.sub(r"%%|%[Aaj]", lambda match: replacements[match.group()], value)


record = {
    "status": "starting",
    "scientific_result": False,
    "approved_walltime": env["GEODML_APPROVED_WALLTIME"],
    "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
    "maximum_gpu_hours": approved_seconds * 4 / 3600,
    "execution_mode": mode,
    "allocation_budget": budget.record(),
    "slurm_job_id": env["SLURM_JOB_ID"],
    "git_commit": env["GEODML_EXECUTION_COMMIT"],
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "finished_at_utc": None,
    "model_id": model_id,
    "model_revision": revision,
    "model_snapshot": str(snapshot),
    "task_count": len(tasks),
    "source_task_count": source_task_count,
    "worker_index": worker_index,
    "worker_count": worker_count,
    "dispatch_mode": dispatch_mode,
    "context_budget": context_budget,
    "claim_root": str(Path(env["GEODML_JUDGE_CLAIM_ROOT"]).resolve()) if mode == "throughput" else None,
    "available_task_count": manifest.get("summary", {}).get("available_task_count", len(tasks)),
    "excluded_task_count": manifest.get("summary", {}).get("excluded_task_count", 0),
    "tasks_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
    "plan_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "virtual_environment": env["ACL_ARR_VENV"],
    "platform": platform.machine(),
    "versions": versions,
    "output": str(root / "nemotron"),
    "stdout": log_path("GEODML_JUDGE_STDOUT", "out"),
    "stderr": log_path("GEODML_JUDGE_STDERR", "err"),
    "server_log": str(root / "logs/server.log"),
    "gpu_telemetry": str(root / "logs/gpu.csv"),
    "resources": {
        key: env.get(key) for key in (
            "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_GPUS_ON_NODE",
            "SLURM_JOB_NODELIST",
        )
    },
}
path = root / "logs/allocation.json"
temporary = path.with_suffix(".tmp")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(path)
print("NEMOTRON_PILOT_ALLOCATION=" + json.dumps(record, sort_keys=True), flush=True)
PY

geodml_telemetry_pid=""
geodml_stage_pid=""
geodml_cleanup() {
  geodml_exit_status=$?
  trap - EXIT INT TERM
  if [[ -n "$geodml_stage_pid" ]]; then
    kill -TERM "$geodml_stage_pid" 2>/dev/null || true
    wait "$geodml_stage_pid" 2>/dev/null || true
  fi
  if [[ -n "$geodml_telemetry_pid" ]]; then
    kill "$geodml_telemetry_pid" 2>/dev/null || true
    wait "$geodml_telemetry_pid" 2>/dev/null || true
  fi
  python3 - "$geodml_exit_status" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path = Path(os.environ["GEODML_JUDGE_PILOT_ROOT"]) / "logs/allocation.json"
record = json.loads(path.read_text())
status = int(sys.argv[1])
runtime_path = Path(record["output"]) / "run_manifest.json"
try:
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime manifest must be an object")
except (OSError, ValueError) as error:
    runtime = {}
    record["runtime_manifest_error"] = str(error)
terminal = "failed_or_interrupted"
if status == 0:
    completed = runtime.get("completed_count")
    remaining = runtime.get("remaining_count")
    valid_counts = (type(completed) is int and type(remaining) is int
                    and completed >= 0 and remaining >= 0
                    and completed + remaining == record["task_count"])
    if valid_counts and runtime.get("status") == "complete" and remaining == 0:
        terminal = "complete"
    elif valid_counts and runtime.get("status") == "checkpointed" and remaining > 0:
        terminal = "checkpointed"
    else:
        status = 2
record.update(
    status=terminal,
    exit_code=status,
    runtime_status=runtime.get("status"),
    completed_count=runtime.get("completed_count"),
    remaining_count=runtime.get("remaining_count"),
    stop_reason=runtime.get("stop_reason"),
    shared_claims=runtime.get("shared_claims"),
    finished_at_utc=datetime.now(timezone.utc).isoformat(),
)
temporary = path.with_suffix(".tmp")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(path)
print("NEMOTRON_PILOT_COMMAND_STATUS=" + str(status), flush=True)
if status != int(sys.argv[1]):
    raise SystemExit(status)
PY
  exit "$geodml_exit_status"
}
trap geodml_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits --loop=10 > "$geodml_log_root/gpu.csv" &
geodml_telemetry_pid=$!

python3 analysis/scripts/search_vllm_stage.py prepare \
  --profile "$geodml_profile" \
  --stage nemotron-judge-pilot \
  --model-id "$GEODML_PILOT_MODEL_ID" \
  --model-revision "$GEODML_PILOT_MODEL_REVISION" \
  --vllm-executable "$ACL_ARR_VENV/bin/vllm" \
  --cache-base "$GEODML_CACHE_ROOT/compile-cache" \
  --expected-gpu-name-pattern GH200 \
  --data-parallel-size 1 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --request-concurrency 4 --enforce-eager

python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$geodml_profile" \
  --server-log "$geodml_log_root/server.log" \
  --cache-base "$GEODML_CACHE_ROOT/compile-cache" \
  --startup-timeout-seconds 900 \
  -- \
  python3 analysis/scripts/run_acl_arr_vllm.py agentic-judge \
    --tasks "$GEODML_JUDGE_PLAN_ROOT/bulk_tasks.jsonl" \
    --judge-manifest "$GEODML_JUDGE_PLAN_ROOT/run_manifest.json" \
    --judge-role bulk --output-dir "$GEODML_JUDGE_PILOT_ROOT/nemotron" \
    --base-url http://127.0.0.1:8010/v1 \
    --server-model-name "$GEODML_PILOT_MODEL_ID" \
    --server-model-revision "$GEODML_PILOT_MODEL_REVISION" \
    --max-concurrency 4 --max-output-tokens 2048 --max-attempts 3 \
    --request-timeout 120 --scheduler rolling --pilot-only --resume --disable-thinking \
    "${geodml_claim_args[@]}" &
geodml_stage_pid=$!
if wait "$geodml_stage_pid"; then
  geodml_stage_status=0
else
  geodml_stage_status=$?
fi
geodml_stage_pid=""
exit "$geodml_stage_status"
