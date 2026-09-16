#!/usr/bin/env bash
# Submit with an explicitly approved --time, resource request, and --array range.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set +u
set -eo pipefail

: "${ACL_ARR_VENV:?}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
: "${GEODML_EXECUTION_REPOSITORY:?}"
: "${GEODML_EXECUTION_COMMIT:?}"
: "${GEODML_WAVE_ROOT:?}"
: "${GEODML_WAVE_OUTPUT_ROOT:?}"
: "${GEODML_WORKER_LAUNCHER:?}"
: "${GEODML_APPROVED_WALLTIME:?Record the specifically approved wall-time}"
: "${GEODML_ALLOCATION_ESTIMATE:?Record the estimate supporting approval}"

export DEBUGINFOD_URLS="${DEBUGINFOD_URLS:-}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module load Stages/2026 GCC Python CUDA
module load git
source "$ACL_ARR_VENV/bin/activate"
set -u
: "${SLURM_JOB_ID:?Run through Slurm}"
test "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)" = \
  "$GEODML_EXECUTION_COMMIT"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"

python3 - "$GEODML_EXECUTION_REPOSITORY" "$GEODML_WORKER_LAUNCHER" <<'PY'
import sys
from pathlib import Path

repository, launcher = (Path(value).resolve(strict=True) for value in sys.argv[1:])
if repository not in launcher.parents:
    raise ValueError("Worker launcher must belong to the pinned repository")
PY
test -x "$GEODML_WORKER_LAUNCHER"

export GEODML_ALLOCATION_BUDGET_JSON
GEODML_ALLOCATION_BUDGET_JSON="$(python3 - "$GEODML_EXECUTION_REPOSITORY" <<'PY'
import json
import sys
sys.path.insert(0, sys.argv[1])
from analysis.interpretability.pipeline.inference_budget import AllocationBudget

budget = AllocationBudget.from_environment(require=True)
if not budget.can_start():
    raise SystemExit("No inference admission time remains in this allocation; model was not loaded")
print(json.dumps(budget.record(), sort_keys=True))
PY
)"
printf 'ALLOCATION_WORK_POLICY=%s\n' "$GEODML_ALLOCATION_BUDGET_JSON"

geodml_worker_index="$SLURM_ARRAY_TASK_ID"
geodml_worker_name="$(printf 'worker-%05d' "$geodml_worker_index")"
geodml_worker_tasks="$GEODML_WAVE_ROOT/workers/$geodml_worker_name.jsonl"
geodml_worker_output="$GEODML_WAVE_OUTPUT_ROOT/$geodml_worker_name"

python3 - "$GEODML_WAVE_ROOT/run_manifest.json" "$geodml_worker_index" \
  "$geodml_worker_tasks" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
index = int(sys.argv[2])
path = Path(sys.argv[3]).resolve()
assert manifest["format_version"] == "geodml-inference-wave-v1"
assert manifest["status"] == "planned"
assert 0 <= index < manifest["worker_count"]
worker = manifest["workers"][index]
assert worker["worker_index"] == index
assert Path(worker["path"]).resolve() == path
assert hashlib.sha256(path.read_bytes()).hexdigest() == worker["sha256"]
assert sum(1 for line in path.read_text().splitlines() if line.strip()) == worker["task_count"]
print(f"INFERENCE_WAVE_PREFLIGHT=PASS worker={index} tasks={worker['task_count']}")
PY

mkdir -p "$geodml_worker_output"
command -v flock >/dev/null
exec 9>>"$geodml_worker_output/.worker.lock"
if ! flock -n 9; then
  printf 'Worker output already has an active writer: %s\n' "$geodml_worker_output" >&2
  exit 75
fi

export GEODML_WORKER_TASKS="$geodml_worker_tasks"
export GEODML_WORKER_OUTPUT="$geodml_worker_output"
export GEODML_WORKER_ATTEMPT_ID
GEODML_WORKER_ATTEMPT_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export GEODML_WORKER_STARTED_AT
GEODML_WORKER_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
geodml_record_allocation() {
  python3 - "$geodml_worker_output/allocation_attempts.jsonl" "$1" "${2:-}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
env = os.environ
wave_manifest = Path(env["GEODML_WAVE_ROOT"]) / "run_manifest.json"
tasks = Path(env["GEODML_WORKER_TASKS"])
runtime_path = Path(env["GEODML_WORKER_OUTPUT"]) / "run_manifest.json"
runtime = json.loads(runtime_path.read_text()) if status != "running" and runtime_path.is_file() else {}
if status == "complete" and runtime.get("status") == "checkpointed":
    status = "checkpointed"


def log_path(key):
    value = env.get(key)
    if value is None:
        return None
    replacements = {
        "%%": "%",
        "%A": env.get("SLURM_ARRAY_JOB_ID") or env["SLURM_JOB_ID"],
        "%a": env["SLURM_ARRAY_TASK_ID"],
        "%j": env["SLURM_JOB_ID"],
    }
    return re.sub(r"%%|%[Aaj]", lambda match: replacements[match.group()], value)


row = {
    "attempt_id": env["GEODML_WORKER_ATTEMPT_ID"],
    "status": status,
    "exit_code": int(sys.argv[3]) if sys.argv[3] else None,
    "started_at_utc": env["GEODML_WORKER_STARTED_AT"],
    "finished_at_utc": None if status == "running" else datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": env["SLURM_JOB_ID"],
    "slurm_array_job_id": env.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": env["SLURM_ARRAY_TASK_ID"],
    "git_commit": env["GEODML_EXECUTION_COMMIT"],
    "repository": env["GEODML_EXECUTION_REPOSITORY"],
    "launcher": env["GEODML_WORKER_LAUNCHER"],
    "model_slug": env.get("GEODML_MODEL_SLUG"),
    "virtual_environment": env["ACL_ARR_VENV"],
    "output": env["GEODML_WORKER_OUTPUT"],
    "wave_manifest": str(wave_manifest),
    "wave_manifest_sha256": hashlib.sha256(wave_manifest.read_bytes()).hexdigest(),
    "tasks": str(tasks),
    "tasks_sha256": hashlib.sha256(tasks.read_bytes()).hexdigest(),
    "stdout": log_path("GEODML_WORKER_STDOUT"),
    "stderr": log_path("GEODML_WORKER_STDERR"),
    "approved_walltime": env["GEODML_APPROVED_WALLTIME"],
    "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
    "allocation_budget": json.loads(env["GEODML_ALLOCATION_BUDGET_JSON"]),
    "stop_reason": runtime.get("stop_reason"),
    "resources": {key: env.get(key) for key in (
        "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_NODELIST",
    )},
}
with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}
geodml_record_allocation running

geodml_launcher_pid=""
geodml_finish() {
  geodml_exit_status=$?
  trap - EXIT INT TERM
  if [[ -n "$geodml_launcher_pid" ]]; then
    kill -TERM "$geodml_launcher_pid" 2>/dev/null || true
    wait "$geodml_launcher_pid" 2>/dev/null || true
  fi
  if [[ "$geodml_exit_status" = 0 ]]; then
    geodml_terminal_status=complete
  else
    geodml_terminal_status=failed_or_interrupted
  fi
  geodml_record_allocation "$geodml_terminal_status" "$geodml_exit_status" || \
    printf '%s\n' 'Could not append terminal allocation event; consult Slurm accounting.' >&2
  exit "$geodml_exit_status"
}
trap geodml_finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$GEODML_EXECUTION_REPOSITORY"
"$GEODML_WORKER_LAUNCHER" &
geodml_launcher_pid=$!
if wait "$geodml_launcher_pid"; then
  geodml_launcher_status=0
else
  geodml_launcher_status=$?
fi
geodml_launcher_pid=""
exit "$geodml_launcher_status"
