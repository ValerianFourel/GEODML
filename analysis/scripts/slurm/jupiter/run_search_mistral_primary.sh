#!/usr/bin/env bash
set -eo pipefail
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA
module load git
set -u
source "${ACL_ARR_VENV:?}/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
test "$(git rev-parse HEAD)" = "${GEODML_EXECUTION_COMMIT:?}"
test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
git diff --quiet HEAD --
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077
geodml_model=mistralai/Mistral-Small-4-119B-2603
geodml_revision=a11f36bebf709121056b1dbcc943d1c6afbe494d
geodml_output="${SEARCH_PRIMARY_OUTPUT:-$SEARCH_PILOT_ROOT/primary-schema-fix-0b3ce8acb5d6/model-config-c860fb2fb61da06a8443}"
geodml_dp="${SEARCH_PRIMARY_DATA_PARALLEL_SIZE:-1}"
geodml_tp="${SEARCH_PRIMARY_TENSOR_PARALLEL_SIZE:-4}"
geodml_concurrency="${SEARCH_PRIMARY_REQUEST_CONCURRENCY:-8}"
geodml_port="${SEARCH_PRIMARY_PORT:-8010}"
geodml_answer_max_tokens="${SEARCH_PRIMARY_ANSWER_MAX_TOKENS:-}"
geodml_base_url="http://127.0.0.1:${geodml_port}/v1"
geodml_profile="${geodml_output}.serving-profile.json"
geodml_args=(run-primary --bundle-dir "$SEARCH_PILOT_ROOT/bundle"
  --model-configuration-id model-config-c860fb2fb61da06a8443
  --server-model-revision "$geodml_revision" --base-url "$geodml_base_url"
  --max-concurrency "$geodml_concurrency" --output-dir "$geodml_output" --resume)
if [[ -n "$geodml_answer_max_tokens" ]]; then
  geodml_args+=(--answer-max-tokens "$geodml_answer_max_tokens")
fi
if python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" --preflight-only; then
  printf 'ALREADY_COMPLETE; no model loaded\n'
  exit 0
else
  geodml_status=$?
  test "$geodml_status" = 3
fi
if [[ -e "$geodml_output/run_manifest.json" ]]; then
  if ! python3 -c 'import json, sys; from pathlib import Path; manifest=json.loads(Path(sys.argv[1]).read_text()); expected=str(Path(sys.argv[2]).resolve()); raise SystemExit(0 if manifest["serving_profile"]["path"] == expected else 1)' \
      "$geodml_output/run_manifest.json" "$geodml_profile"; then
    printf 'STOP: historical partial results have no serving profile provenance\n'
    exit 2
  fi
fi
geodml_profile_hash="$(python3 analysis/scripts/search_vllm_stage.py prepare \
  --profile "$geodml_profile" --stage mistral-primary \
  --model-id "$geodml_model" --model-revision "$geodml_revision" \
  --vllm-executable "$ACL_ARR_VENV/bin/vllm" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --expected-gpu-name-pattern GH200 \
  --port "$geodml_port" --data-parallel-size "$geodml_dp" \
  --tensor-parallel-size "$geodml_tp" --request-concurrency "$geodml_concurrency" \
  --dtype bfloat16 --max-model-len 41472 --gpu-memory-utilization 0.90 \
  --language-model-only --tokenizer-mode mistral --attention-backend FLASH_ATTN_MLA \
  --config-format mistral --load-format mistral \
  --structured-outputs-config '{"backend":"xgrammar"}')"
printf 'TEXT_ONLY_CLI=PASS; no multimodal inputs permitted\n'
printf 'SERVING_PROFILE=%s\nSERVING_PROFILE_SHA256=%s\n' "$geodml_profile" "$geodml_profile_hash"
geodml_approval_args=()
if (( geodml_dp > 1 )); then
  geodml_approval_path="${SEARCH_PRIMARY_BENCHMARK_APPROVAL_PATH:?DP requires benchmark approval}"
  python3 analysis/scripts/search_vllm_stage.py validate-approval \
    --profile "$geodml_profile" \
    --approval "$geodml_approval_path"
  geodml_approval_args=(--benchmark-approval "$geodml_approval_path")
fi
geodml_args+=(--serving-profile "$geodml_profile")
geodml_native_report="$(python3 analysis/scripts/check_search_mistral_native.py --model-snapshots "${ACL_ARR_RUN_ROOT:?}/model-snapshots.json")"
printf 'NATIVE_CONFIG_PREFLIGHT=PASS; no model weights loaded\n'
python3 analysis/scripts/check_search_experience_grammar.py
geodml_context_args=(--bundle-dir "$SEARCH_PILOT_ROOT/bundle"
  --model-configuration-id model-config-c860fb2fb61da06a8443
  --model-snapshots "${ACL_ARR_RUN_ROOT:?}/model-snapshots.json"
  --max-model-len 41472)
if [[ -n "$geodml_answer_max_tokens" ]]; then
  geodml_context_args+=(--answer-max-tokens "$geodml_answer_max_tokens")
fi
python3 analysis/scripts/check_search_mistral_context.py "${geodml_context_args[@]}"
mkdir -p "$SEARCH_PILOT_ROOT/logs"
geodml_log="$(mktemp "$SEARCH_PILOT_ROOT/logs/mistral-primary.XXXXXX")"
scontrol show job "$SLURM_JOB_ID" > "$geodml_log.allocation"
printf '%s\n' "$geodml_native_report" > "$geodml_log.native-config.json"
printf 'COMMIT=%s\nMODEL=%s\nREVISION=%s\nTOKENIZER=mistral\nLANGUAGE_MODEL_ONLY=true\nCONTEXT=41472\nDP=%s\nTP=%s\nDTYPE=bfloat16\nCONCURRENCY=%s\nPORT=%s\nANSWER_MAX_TOKENS=%s\nSERVING_PROFILE_SHA256=%s\nSERVER_STARTUP_TIMEOUT=00:30:00\nSTEP_CAP=00:45:00\nESTIMATE=10-30 minutes; loading and inference unmeasured for this arm\n' \
  "$GEODML_EXECUTION_COMMIT" "$geodml_model" "$geodml_revision" "$geodml_dp" "$geodml_tp" \
  "$geodml_concurrency" "$geodml_port" "${geodml_answer_max_tokens:-frozen-plan}" \
  "$geodml_profile_hash" > "$geodml_log.settings"
printf 'SERVER_LOG=%s\n' "$geodml_log"
if python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$geodml_profile" --server-log "$geodml_log" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --startup-timeout-seconds 1800 "${geodml_approval_args[@]}" -- \
  python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" \
  2>&1 | tee "$geodml_log.controller"; then
  geodml_status=0
else
  geodml_status=$?
fi
printf 'MISTRAL_PRIMARY_STATUS=%s\nRESULTS=%s\n' "$geodml_status" "$geodml_output"
python3 - "$geodml_output" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
root = Path(sys.argv[1])
if (root / "run_manifest.json").is_file():
    m = json.loads((root / "run_manifest.json").read_text())
    print("SUMMARY=" + json.dumps({k:m.get(k) for k in ("status","task_count","completed_count","remaining_count","failures_this_invocation")}))
if (root / "failures.jsonl").is_file():
    rows = [json.loads(x) for x in (root / "failures.jsonl").read_text().splitlines() if x.strip()]
    print("ERROR_COUNTS=" + json.dumps(dict(Counter(r.get("error","UNKNOWN") for r in rows))))
PY
exit "$geodml_status"
