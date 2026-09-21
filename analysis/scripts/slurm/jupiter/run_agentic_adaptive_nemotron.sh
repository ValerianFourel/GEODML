#!/usr/bin/env bash
set -euo pipefail

test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
test "${GEODML_EXECUTION_COMMIT:?}" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"
: "${GEODML_ADAPTIVE_PLAN:?}"
: "${GEODML_ADAPTIVE_WORKER_INDEX:?}"
: "${GEODML_NEMOTRON_PROFILE:?}"
: "${GEODML_NEMOTRON_SNAPSHOT:?}"
: "${GEODML_ADAPTIVE_ROLE_ROOT:?}"
: "${GEODML_CACHE_ROOT:?}"

test -s "$GEODML_ADAPTIVE_PLAN"
test -s "$GEODML_NEMOTRON_PROFILE"
test -d "$GEODML_NEMOTRON_SNAPSHOT"
test "$SLURM_JOB_NUM_NODES" = 5
test "$SLURM_STEP_NUM_NODES" = 1

profile_sha="$(python3 - "$GEODML_NEMOTRON_PROFILE" <<'PY'
import sys
from analysis.scripts.search_vllm_stage import load_profile
p = load_profile(sys.argv[1])
assert p["model"] == {
    "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "model_revision": "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
}
assert p["serving"]["tensor_parallel_size"] == 4
assert p["serving"]["data_parallel_size"] == 1
assert p["serving"]["max_model_len"] == 73728
assert p["serving"]["dtype"] == "bfloat16"
assert p["features"]["enforce_eager"] is True
print(p["profile_sha256"])
PY
)"

role_root="$GEODML_ADAPTIVE_ROLE_ROOT/nemotron"
server_log="$role_root/server.log"
telemetry="$role_root/gpu.csv"
event="$role_root/role-event.json"
mkdir -p "$role_root"
test ! -e "$server_log"
test ! -e "$telemetry"

python3 - "$event" "$profile_sha" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
value = {
    "format_version": "agentic-adaptive-role-event-v1", "role": "nemotron",
    "status": "starting", "worker_index": int(os.environ["GEODML_ADAPTIVE_WORKER_INDEX"]),
    "job_id": os.environ["SLURM_JOB_ID"], "profile_sha256": sys.argv[2],
    "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

telemetry_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  python3 - "$event" "$status" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); value = json.loads(path.read_text())
value.update(status="complete" if int(sys.argv[2]) == 0 else "failed_or_interrupted",
             exit_code=int(sys.argv[2]), finished_at=datetime.now(timezone.utc).isoformat())
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
  exit "$status"
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits --loop=10 > "$telemetry" &
telemetry_pid=$!

python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$GEODML_NEMOTRON_PROFILE" \
  --server-log "$server_log" \
  --cache-base "$GEODML_CACHE_ROOT/compile-cache" \
  --startup-timeout-seconds 900 \
  -- \
  python3 analysis/scripts/run_agentic_nemotron_adaptive.py \
    --plan "$GEODML_ADAPTIVE_PLAN" \
    --profile "$GEODML_NEMOTRON_PROFILE" \
    --tokenizer-snapshot "$GEODML_NEMOTRON_SNAPSHOT" \
    --telemetry "$telemetry" \
    --worker-index "$GEODML_ADAPTIVE_WORKER_INDEX" \
    --worker-count 5

python3 - <<'PY'
import subprocess
rows = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    text=True, capture_output=True, check=True,
).stdout.splitlines()
if any(row.strip() for row in rows):
    raise SystemExit("GPU process remained after Nemotron stage shutdown")
print("NEMOTRON_GPU_RELEASE=PASS")
PY
