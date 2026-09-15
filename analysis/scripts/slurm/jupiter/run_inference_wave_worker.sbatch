#!/usr/bin/env bash
# Submit with an explicitly approved --time, resource request, and --array range.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail

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
test "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)" = \
  "$GEODML_EXECUTION_COMMIT"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"

case "$GEODML_WORKER_LAUNCHER" in
  "$GEODML_EXECUTION_REPOSITORY"/*) ;;
  *) printf '%s\n' 'Worker launcher must belong to the pinned repository.' >&2; exit 2 ;;
esac
test -x "$GEODML_WORKER_LAUNCHER"

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
python3 - "$geodml_worker_output/allocation_attempts.jsonl" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "approved_walltime": os.environ["GEODML_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["GEODML_ALLOCATION_ESTIMATE"],
}
with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
PY

export GEODML_WORKER_TASKS="$geodml_worker_tasks"
export GEODML_WORKER_OUTPUT="$geodml_worker_output"
cd "$GEODML_EXECUTION_REPOSITORY"
exec "$GEODML_WORKER_LAUNCHER"
