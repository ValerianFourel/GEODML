#!/usr/bin/env bash
# Submission supplies the approved one-hour, one-node resource request.
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
test "$GEODML_APPROVED_WALLTIME" = 01:00:00
test "${SLURM_JOB_NUM_NODES:?}" = 1
test "${SLURM_GPUS_ON_NODE:?}" = 4
test "${SLURM_CPUS_PER_TASK:?}" = 32
test "${SLURM_MEM_PER_NODE:?}" = 524288
test "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)" = "$GEODML_EXECUTION_COMMIT"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"
cd "$GEODML_EXECUTION_REPOSITORY"

test -s "$GEODML_JUDGE_PILOT_ROOT/plan/run_manifest.json"
test -s "$GEODML_JUDGE_PILOT_ROOT/plan/bulk_tasks.jsonl"
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

import xgrammar
from analysis.interpretability.pipeline.agentic_judging import agentic_judge_schema
from analysis.scripts.run_acl_arr_vllm import _agentic_judge_context

env = os.environ
root = Path(env["GEODML_JUDGE_PILOT_ROOT"])
manifest_path = root / "plan/run_manifest.json"
queue = root / "plan/bulk_tasks.jsonl"
manifest = json.loads(manifest_path.read_text())
tasks, _, model_id, revision = _agentic_judge_context(
    manifest_path, queue, judge_role="bulk"
)
if (model_id, revision) != (
    env["GEODML_PILOT_MODEL_ID"], env["GEODML_PILOT_MODEL_REVISION"]
):
    raise ValueError("Pilot judge identity differs from the approved Nemotron snapshot")
if len(tasks) != 24 or len({task.judge_task_id for task in tasks}) != 24:
    raise ValueError("This pilot requires exactly 24 unique judge tasks")
if manifest.get("scientific_result") is not False or manifest.get("pilot_only") is not True:
    raise ValueError("A judge pilot must not claim scientific results")
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
for task in tasks:
    schema = agentic_judge_schema([item.evidence_id for item in task.evidence])
    xgrammar.Grammar.from_json_schema(json.dumps(schema))
print("NEMOTRON_PILOT_PREFLIGHT=PASS tasks=24 weight_shards=13 schemas=24", flush=True)
versions = {}
for package in ("vllm", "transformers", "huggingface-hub", "xgrammar", "torch"):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
record = {
    "status": "starting",
    "scientific_result": False,
    "approved_walltime": env["GEODML_APPROVED_WALLTIME"],
    "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
    "maximum_gpu_hours": 4,
    "slurm_job_id": env["SLURM_JOB_ID"],
    "git_commit": env["GEODML_EXECUTION_COMMIT"],
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "finished_at_utc": None,
    "model_id": model_id,
    "model_revision": revision,
    "model_snapshot": str(snapshot),
    "task_count": len(tasks),
    "tasks_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
    "plan_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "virtual_environment": env["ACL_ARR_VENV"],
    "platform": platform.machine(),
    "versions": versions,
    "output": str(root / "nemotron"),
    "stdout": str(root / "logs" / f"slurm-{env['SLURM_JOB_ID']}.out"),
    "stderr": str(root / "logs" / f"slurm-{env['SLURM_JOB_ID']}.err"),
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
record.update(
    status="complete" if status == 0 else "failed_or_interrupted",
    exit_code=status,
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
    --tasks "$GEODML_JUDGE_PILOT_ROOT/plan/bulk_tasks.jsonl" \
    --judge-manifest "$GEODML_JUDGE_PILOT_ROOT/plan/run_manifest.json" \
    --judge-role bulk --output-dir "$GEODML_JUDGE_PILOT_ROOT/nemotron" \
    --base-url http://127.0.0.1:8010/v1 \
    --server-model-name "$GEODML_PILOT_MODEL_ID" \
    --server-model-revision "$GEODML_PILOT_MODEL_REVISION" \
    --max-concurrency 4 --max-output-tokens 2048 --max-attempts 3 \
    --request-timeout 120 --scheduler rolling --pilot-only --resume --disable-thinking &
geodml_stage_pid=$!
if wait "$geodml_stage_pid"; then
  geodml_stage_status=0
else
  geodml_stage_status=$?
fi
geodml_stage_pid=""
exit "$geodml_stage_status"
