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

geodml_prompt_args=()
geodml_expected_cells="${SEARCH_AGENTIC_EXPECTED_CELL_COUNT:-12}"
if [[ -n "${SEARCH_AGENTIC_PROMPTS_JSONL:-}" ]]; then
  : "${SEARCH_AGENTIC_SELECTION_RECORDS_JSONL:?}"
  : "${SEARCH_AGENTIC_PROMPT_COUNT:?}"
  test -s "$SEARCH_AGENTIC_PROMPTS_JSONL"
  test -s "$SEARCH_AGENTIC_SELECTION_RECORDS_JSONL"
  geodml_prompt_args=(
    --prompts-jsonl "$SEARCH_AGENTIC_PROMPTS_JSONL"
    --selection-records-jsonl "$SEARCH_AGENTIC_SELECTION_RECORDS_JSONL"
    --prompt-count "$SEARCH_AGENTIC_PROMPT_COUNT"
    --prompt-selection-seed "${SEARCH_AGENTIC_PROMPT_SELECTION_SEED:-20260912}"
  )
fi

python3 - "$SEARCH_AGENTIC_PROFILE" <<'PY'
import sys
from analysis.scripts.search_vllm_stage import load_profile

profile = load_profile(sys.argv[1])
assert profile["model"] == {
    "model_id": "Qwen/Qwen2.5-72B-Instruct",
    "model_revision": "495f39366efef23836d0cfae4fbe635880d2be31",
}, profile["model"]
assert profile["serving"]["tensor_parallel_size"] == 4
assert profile["serving"]["data_parallel_size"] == 1
assert profile["features"]["rope_scaling"] == {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "rope_type": "yarn",
}, profile["features"]
print("AGENTIC_QWEN25_PROFILE_GATE=PASS")
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
    --model-id Qwen/Qwen2.5-72B-Instruct \
    --model-revision 495f39366efef23836d0cfae4fbe635880d2be31 \
    --cross-encoder-snapshot "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT" \
    --cross-encoder-revision "$SEARCH_AGENTIC_CROSS_ENCODER_REVISION" \
    --search-snapshot "duckduckgo=$SEARCH_AGENTIC_DDG_SNAPSHOT" \
    --search-snapshot "searxng=$SEARCH_AGENTIC_SEARXNG_SNAPSHOT" \
    --seed 20260911 \
    --max-tokens 1024 \
    --request-concurrency "${SEARCH_AGENTIC_REQUEST_CONCURRENCY:-1}" \
    "${geodml_prompt_args[@]}"

python3 - "$SEARCH_AGENTIC_OUTPUT/run_manifest.json" "$geodml_expected_cells" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
expected_cells = int(sys.argv[2])
expected = {
    "status": "complete",
    "cell_count": expected_cells,
    "completed_count": expected_cells,
    "remaining_count": 0,
    "scientific_result": False,
}
actual = {key: manifest.get(key) for key in expected}
assert actual == expected, actual
if expected_cells == 12:
    print("AGENTIC_QWEN25_12_CELL_SMOKE=PASS")
else:
    print(f"AGENTIC_QWEN25_{expected_cells}_CELL_RUN=PASS")
print("SUMMARY=" + json.dumps(actual, sort_keys=True))
PY
