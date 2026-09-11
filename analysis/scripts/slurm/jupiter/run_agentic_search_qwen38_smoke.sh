#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
test "${GEODML_EXECUTION_COMMIT:?}" = "$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

: "${SEARCH_AGENTIC_OUTPUT:?}"
: "${SEARCH_AGENTIC_PROFILE:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_REVISION:?}"
: "${SEARCH_AGENTIC_DDG_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_SEARXNG_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_SERVER_LOG:?}"
: "${SEARCH_AGENTIC_GPU_TELEMETRY:?}"

test -s "$SEARCH_AGENTIC_PROFILE"
test -d "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT"
test -s "$SEARCH_AGENTIC_DDG_SNAPSHOT"
test -s "$SEARCH_AGENTIC_SEARXNG_SNAPSHOT"
test ! -e "$SEARCH_AGENTIC_SERVER_LOG"
test ! -e "$SEARCH_AGENTIC_GPU_TELEMETRY"

python3 - "$SEARCH_AGENTIC_PROFILE" <<'PY'
import sys
from analysis.scripts.search_vllm_stage import load_profile

profile = load_profile(sys.argv[1])
assert profile["model"] == {
    "model_id": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}, profile["model"]
assert profile["serving"]["tensor_parallel_size"] == 4
assert profile["serving"]["data_parallel_size"] == 1
print("AGENTIC_QWEN38_PROFILE_GATE=PASS")
PY

python3 - <<'PY'
import sentence_transformers

assert sentence_transformers.__version__ == "6.0.1"
print("AGENTIC_COMPACTOR_RUNTIME_GATE=PASS")
PY

mkdir -p "$(dirname "$SEARCH_AGENTIC_OUTPUT")" \
  "$(dirname "$SEARCH_AGENTIC_SERVER_LOG")" \
  "$(dirname "$SEARCH_AGENTIC_GPU_TELEMETRY")"

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
  --startup-timeout-seconds 900 \
  -- \
  python3 analysis/scripts/run_agentic_search_integration_smoke.py \
    --output "$SEARCH_AGENTIC_OUTPUT" \
    --base-url http://127.0.0.1:8010/v1 \
    --model-id Qwen/Qwen3.8-27B \
    --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
    --cross-encoder-snapshot "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT" \
    --cross-encoder-revision "$SEARCH_AGENTIC_CROSS_ENCODER_REVISION" \
    --search-snapshot "duckduckgo=$SEARCH_AGENTIC_DDG_SNAPSHOT" \
    --search-snapshot "searxng=$SEARCH_AGENTIC_SEARXNG_SNAPSHOT" \
    --seed 20260911 \
    --max-tokens 1024

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
}
actual = {key: manifest.get(key) for key in expected}
assert actual == expected, actual
print("AGENTIC_QWEN38_12_CELL_SMOKE=PASS")
print("SUMMARY=" + json.dumps(actual, sort_keys=True))
PY
