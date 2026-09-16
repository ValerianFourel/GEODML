#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${GEODML_WORKER_TASKS:?}"
: "${GEODML_WORKER_OUTPUT:?}"
: "${GEODML_JUDGE_MANIFEST:?}"
: "${GEODML_JUDGE_ROLE:?Use bulk or validation}"
: "${GEODML_JUDGE_PROFILE:?Pinned vLLM serving profile}"
export GEODML_JUDGE_CLAIM_ROOT="${GEODML_JUDGE_CLAIM_ROOT:-${GEODML_INFERENCE_CLAIM_ROOT:-}}"
: "${GEODML_JUDGE_CLAIM_ROOT:?All related judge jobs must share one claim directory}"
[[ "$GEODML_JUDGE_CLAIM_ROOT" = /* ]] || { printf '%s\n' 'Claim root must be an absolute shared path.' >&2; exit 2; }

readarray -t geodml_model < <(
  python3 - "$GEODML_JUDGE_MANIFEST" "$GEODML_JUDGE_ROLE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = manifest[f"{sys.argv[2]}_model"]
print(model["model_id"])
print(model["model_revision"])
PY
)
test "${#geodml_model[@]}" -eq 2

geodml_log_root="${GEODML_WAVE_LOG_ROOT:-$(dirname "$GEODML_WORKER_OUTPUT")/logs}"
mkdir -p "$geodml_log_root"
geodml_server_log="$geodml_log_root/judge-${GEODML_JUDGE_ROLE}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}.server.log"
geodml_thinking_args=()
if [[ "${GEODML_JUDGE_DISABLE_THINKING:-0}" == 1 ]]; then
  geodml_thinking_args=(--disable-thinking)
fi

python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$GEODML_JUDGE_PROFILE" \
  --server-log "$geodml_server_log" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --startup-timeout-seconds 900 \
  -- \
  python3 analysis/scripts/run_acl_arr_vllm.py agentic-judge \
    --tasks "$GEODML_WORKER_TASKS" \
    --judge-manifest "$GEODML_JUDGE_MANIFEST" \
    --judge-role "$GEODML_JUDGE_ROLE" \
    --output-dir "$GEODML_WORKER_OUTPUT" \
    --base-url http://127.0.0.1:8010/v1 \
    --server-model-name "${geodml_model[0]}" \
    --server-model-revision "${geodml_model[1]}" \
    --max-concurrency "${GEODML_JUDGE_CONCURRENCY:-32}" \
    --max-output-tokens "${GEODML_JUDGE_MAX_OUTPUT_TOKENS:-512}" \
    --claim-root "$GEODML_JUDGE_CLAIM_ROOT" \
    --dispatch-mode "${GEODML_DISPATCH_MODE:-partition}" \
    --worker-index "${GEODML_WORKER_INDEX:-0}" \
    --worker-count "${GEODML_WORKER_COUNT:-1}" \
    --resume \
    "${geodml_thinking_args[@]}"
