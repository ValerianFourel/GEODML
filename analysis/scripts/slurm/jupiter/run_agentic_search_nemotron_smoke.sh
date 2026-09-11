#!/usr/bin/env bash
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module load Stages/2026 GCC Python CUDA
module load git
source "${ACL_ARR_VENV:?}/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
test "${GEODML_EXECUTION_COMMIT:?}" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

: "${ACL_ARR_VENV:?}"
: "${SEARCH_AGENTIC_OUTPUT:?}"
: "${SEARCH_AGENTIC_PROFILE:?}"
: "${SEARCH_AGENTIC_MODEL_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_DOWNLOAD_MANIFEST:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_REVISION:?}"
: "${SEARCH_AGENTIC_DDG_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_SEARXNG_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_SERVER_LOG:?}"
: "${SEARCH_AGENTIC_GPU_TELEMETRY:?}"

test -d "$SEARCH_AGENTIC_MODEL_SNAPSHOT"
test "$(basename "$SEARCH_AGENTIC_MODEL_SNAPSHOT")" = 2dc98e2afe4face0e4ce40972a915c45368bd34a
test -d "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT"
test -s "$SEARCH_AGENTIC_DDG_SNAPSHOT"
test -s "$SEARCH_AGENTIC_SEARXNG_SNAPSHOT"
test ! -e "$SEARCH_AGENTIC_SERVER_LOG"
test ! -e "$SEARCH_AGENTIC_GPU_TELEMETRY"

agentic_request_concurrency="${SEARCH_AGENTIC_REQUEST_CONCURRENCY:-1}"
case "$agentic_request_concurrency" in
  1|2|3|4) ;;
  *) printf 'STOP: SEARCH_AGENTIC_REQUEST_CONCURRENCY must be 1, 2, 3, or 4\n' >&2; exit 2 ;;
esac

python3 analysis/scripts/download_nemotron3_super.py \
  --cache-dir "${HF_HUB_CACHE:?}" \
  --output "${SEARCH_AGENTIC_DOWNLOAD_MANIFEST:?}" \
  --verify-only

python3 - <<'PY'
import sentence_transformers

assert sentence_transformers.__version__ == "6.0.1"
print("AGENTIC_COMPACTOR_RUNTIME_GATE=PASS")
PY

mkdir -p "$(dirname "$SEARCH_AGENTIC_OUTPUT")" \
  "$(dirname "$SEARCH_AGENTIC_PROFILE")" \
  "$(dirname "$SEARCH_AGENTIC_SERVER_LOG")" \
  "$(dirname "$SEARCH_AGENTIC_GPU_TELEMETRY")"

python3 analysis/scripts/search_vllm_stage.py prepare \
  --profile "$SEARCH_AGENTIC_PROFILE" \
  --stage agentic-nemotron-smoke \
  --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
  --model-revision 2dc98e2afe4face0e4ce40972a915c45368bd34a \
  --vllm-executable "$ACL_ARR_VENV/bin/vllm" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --expected-gpu-name-pattern GH200 \
  --port 8010 \
  --data-parallel-size 1 \
  --tensor-parallel-size 4 \
  --request-concurrency "$agentic_request_concurrency" \
  --dtype bfloat16 \
  --max-model-len 43008 \
  --gpu-memory-utilization 0.82 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --enforce-eager

python3 - "$SEARCH_AGENTIC_PROFILE" <<'PY'
import sys
from analysis.scripts.search_vllm_stage import load_profile

profile = load_profile(sys.argv[1])
assert profile["model"] == {
    "model_id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "model_revision": "2dc98e2afe4face0e4ce40972a915c45368bd34a",
}
assert profile["serving"]["tensor_parallel_size"] == 4
assert profile["serving"]["data_parallel_size"] == 1
assert profile["serving"]["max_model_len"] == 43008
assert profile["serving"]["gpu_memory_utilization"] == 0.82
assert "--enforce-eager" in profile["server_argv"]
print("AGENTIC_NEMOTRON_PROFILE_GATE=PASS")
PY

telemetry_pid=""
cleanup() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 10
  done
) > "$SEARCH_AGENTIC_GPU_TELEMETRY" &
telemetry_pid=$!

python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$SEARCH_AGENTIC_PROFILE" \
  --server-log "$SEARCH_AGENTIC_SERVER_LOG" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --startup-timeout-seconds 1500 \
  -- \
  python3 analysis/scripts/run_agentic_search_integration_smoke.py \
    --output "$SEARCH_AGENTIC_OUTPUT" \
    --base-url http://127.0.0.1:8010/v1 \
    --model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --model-revision 2dc98e2afe4face0e4ce40972a915c45368bd34a \
    --cross-encoder-snapshot "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT" \
    --cross-encoder-revision "$SEARCH_AGENTIC_CROSS_ENCODER_REVISION" \
    --search-snapshot "duckduckgo=$SEARCH_AGENTIC_DDG_SNAPSHOT" \
    --search-snapshot "searxng=$SEARCH_AGENTIC_SEARXNG_SNAPSHOT" \
    --seed 20260911 \
    --max-tokens 1024 \
    --request-concurrency "$agentic_request_concurrency" \
    --disable-thinking

python3 - "$SEARCH_AGENTIC_OUTPUT/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "status": "complete",
    "cell_count": 12,
    "completed_count": 12,
    "remaining_count": 0,
    "scientific_result": False,
    "disable_thinking": True,
}
actual = {key: manifest.get(key) for key in expected}
assert actual == expected, actual
print("AGENTIC_NEMOTRON_12_CELL_SMOKE=PASS")
print("SUMMARY=" + json.dumps(actual, sort_keys=True))
PY
