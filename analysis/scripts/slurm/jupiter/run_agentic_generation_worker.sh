#!/usr/bin/env bash
set -euo pipefail

: "${GEODML_WORKER_TASKS:?}"
: "${GEODML_WORKER_OUTPUT:?}"
: "${GEODML_MODEL_SLUG:?Use qwen38 or llama4}"

case "$GEODML_MODEL_SLUG" in
  qwen38)
    geodml_launcher=analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh
    ;;
  llama4)
    geodml_launcher=analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh
    ;;
  *)
    printf 'Unsupported generator model slug: %s\n' "$GEODML_MODEL_SLUG" >&2
    exit 2
    ;;
esac

geodml_log_root="${GEODML_WAVE_LOG_ROOT:-$(dirname "$GEODML_WORKER_OUTPUT")/logs}"
mkdir -p "$geodml_log_root"
geodml_log_stem="$geodml_log_root/${GEODML_MODEL_SLUG}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"

export SEARCH_AGENTIC_OUTPUT="$GEODML_WORKER_OUTPUT"
export SEARCH_AGENTIC_CELL_IDS_JSONL="$GEODML_WORKER_TASKS"
export SEARCH_AGENTIC_EXPECTED_CELL_COUNT="$(awk 'NF {count++} END {print count+0}' "$GEODML_WORKER_TASKS")"
export SEARCH_AGENTIC_SERVER_LOG="$geodml_log_stem.server.log"
export SEARCH_AGENTIC_GPU_TELEMETRY="$geodml_log_stem.gpu.csv"
export SEARCH_AGENTIC_PROMPT_SHARD_INDEX="${SEARCH_AGENTIC_PROMPT_SHARD_INDEX:-0}"
export SEARCH_AGENTIC_PROMPT_SHARD_COUNT="${SEARCH_AGENTIC_PROMPT_SHARD_COUNT:-1}"
export SEARCH_AGENTIC_PRODUCTION_CONDITIONS=1
export GEODML_EXPECTED_JOB_ID="$SLURM_JOB_ID"
if [[ "${GEODML_DISPATCH_MODE:-partition}" = backlog ]]; then
  : "${GEODML_INFERENCE_CLAIM_ROOT:?Backlog generation requires a shared claim registry}"
  export SEARCH_AGENTIC_SHARED_CLAIM_ROOT="$GEODML_INFERENCE_CLAIM_ROOT"
  export SEARCH_AGENTIC_WORKER_INDEX="${GEODML_WORKER_INDEX:?}"
  export SEARCH_AGENTIC_WORKER_COUNT="${GEODML_WORKER_COUNT:?}"
fi

exec bash "$geodml_launcher"
