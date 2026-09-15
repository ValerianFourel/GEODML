#!/usr/bin/env bash
# Submission supplies the specifically approved time and resource request.
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
: "${GEODML_APPROVED_WALLTIME:?Record the specifically approved wall-time}"
: "${GEODML_ALLOCATION_ESTIMATE:?Record the estimate supporting approval}"
: "${GEODML_RESUME_ATTEMPT_DIR:?Use a dedicated attempt directory}"
: "${SEARCH_AGENTIC_OUTPUT:?Use the existing shard output}"
: "${SEARCH_AGENTIC_PROFILE:?Use the pinned serving profile}"

test "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)" = \
  "$GEODML_EXECUTION_COMMIT"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"
test -d "$SEARCH_AGENTIC_OUTPUT"
command -v flock >/dev/null
exec 9>>"$SEARCH_AGENTIC_OUTPUT/.resume.lock"
if ! flock -n 9; then
  printf 'Shard output is already being resumed: %s\n' "$SEARCH_AGENTIC_OUTPUT" >&2
  exit 75
fi
test -s "$SEARCH_AGENTIC_OUTPUT/config.json"
test -s "$SEARCH_AGENTIC_PROFILE"

export GEODML_EXPECTED_JOB_ID="$SLURM_JOB_ID"
unset SEARCH_AGENTIC_CELL_IDS_JSONL
export SEARCH_AGENTIC_SERVER_LOG="$GEODML_RESUME_ATTEMPT_DIR/qwen38-job${SLURM_JOB_ID}.server.log"
export SEARCH_AGENTIC_GPU_TELEMETRY="$GEODML_RESUME_ATTEMPT_DIR/qwen38-job${SLURM_JOB_ID}.gpu.csv"
test ! -e "$SEARCH_AGENTIC_SERVER_LOG"
test ! -e "$SEARCH_AGENTIC_GPU_TELEMETRY"
mkdir -p "$GEODML_RESUME_ATTEMPT_DIR"

python3 - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path

env = os.environ
attempt = Path(env["GEODML_RESUME_ATTEMPT_DIR"])
config = json.loads((Path(env["SEARCH_AGENTIC_OUTPUT"]) / "config.json").read_text())
job_id = env["SLURM_JOB_ID"]
row = {
    "slurm_job_id": job_id,
    "approved_walltime": env["GEODML_APPROVED_WALLTIME"],
    "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
    "git_commit": env["GEODML_EXECUTION_COMMIT"],
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "output": env["SEARCH_AGENTIC_OUTPUT"],
    "source_config_sha256": config["config_sha256"],
    "profile": env["SEARCH_AGENTIC_PROFILE"],
    "stdout": env.get("GEODML_RESUME_STDOUT", str(attempt / "slurm-%j.out")).replace("%j", job_id),
    "stderr": env.get("GEODML_RESUME_STDERR", str(attempt / "slurm-%j.err")).replace("%j", job_id),
    "server_log": env["SEARCH_AGENTIC_SERVER_LOG"],
    "gpu_telemetry": env["SEARCH_AGENTIC_GPU_TELEMETRY"],
    "resources": {
        key: env.get(key)
        for key in (
            "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_GPUS",
            "SLURM_GPUS_ON_NODE", "SLURM_JOB_GRES", "SLURM_JOB_NODELIST",
        )
    },
}
with (attempt / "allocation_attempts.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
print("QWEN38_RESUME_ALLOCATION=" + json.dumps(row, sort_keys=True), flush=True)
PY

cd "$GEODML_EXECUTION_REPOSITORY"
exec bash analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh
