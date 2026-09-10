#!/usr/bin/env bash
set -eo pipefail
trap 'printf "HF_PRIMARY_WORKER_FAILED line=%s status=%s\n" "$LINENO" "$?" >&2' ERR
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA
module load git
set -u
source "${ACL_ARR_VENV:?}/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
test "$(git rev-parse HEAD)" = "${GEODML_EXECUTION_COMMIT:?}"
test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
git diff --quiet HEAD --
: "${SEARCH_PILOT_ROOT:?}" "${SEARCH_PRIMARY_OUTPUT:?}"
: "${SEARCH_PRIMARY_MODEL_ID:?}" "${SEARCH_PRIMARY_MODEL_REVISION:?}"
: "${SEARCH_PRIMARY_MODEL_CONFIGURATION_ID:?}" "${ACL_ARR_RUN_ROOT:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
umask 077

geodml_dp="${SEARCH_PRIMARY_DATA_PARALLEL_SIZE:-1}"
geodml_tp="${SEARCH_PRIMARY_TENSOR_PARALLEL_SIZE:-4}"
geodml_concurrency="${SEARCH_PRIMARY_REQUEST_CONCURRENCY:-8}"
geodml_port="${SEARCH_PRIMARY_PORT:-8010}"
geodml_max_model_len="${SEARCH_PRIMARY_MAX_MODEL_LEN:?}"
geodml_gpu_memory_utilization="${SEARCH_PRIMARY_GPU_MEMORY_UTILIZATION:-0.90}"
geodml_max_tasks="${SEARCH_PRIMARY_MAX_TASKS:-0}"
geodml_answer_max_tokens="${SEARCH_PRIMARY_ANSWER_MAX_TOKENS:-2048}"
geodml_startup_timeout_seconds="${SEARCH_PRIMARY_STARTUP_TIMEOUT_SECONDS:-1200}"
geodml_enforce_eager="${SEARCH_PRIMARY_ENFORCE_EAGER:-0}"
geodml_disable_custom_all_reduce="${SEARCH_PRIMARY_DISABLE_CUSTOM_ALL_REDUCE:-0}"
geodml_rope_scaling="${SEARCH_PRIMARY_ROPE_SCALING:-}"
[[ "$geodml_dp" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_tp" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_concurrency" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_port" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_max_model_len" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_max_tasks" =~ ^[0-9]+$ ]]
[[ "$geodml_answer_max_tokens" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_startup_timeout_seconds" =~ ^[1-9][0-9]*$ ]]
[[ "$geodml_enforce_eager" =~ ^[01]$ ]]
[[ "$geodml_disable_custom_all_reduce" =~ ^[01]$ ]]

geodml_base_url="http://127.0.0.1:${geodml_port}/v1"
geodml_profile="${SEARCH_PRIMARY_OUTPUT}.serving-profile.json"
geodml_args=(run-primary
  --bundle-dir "$SEARCH_PILOT_ROOT/bundle"
  --model-configuration-id "$SEARCH_PRIMARY_MODEL_CONFIGURATION_ID"
  --server-model-revision "$SEARCH_PRIMARY_MODEL_REVISION"
  --base-url "$geodml_base_url"
  --answer-max-tokens "$geodml_answer_max_tokens"
  --max-concurrency "$geodml_concurrency"
  --max-tasks "$geodml_max_tasks"
  --output-dir "$SEARCH_PRIMARY_OUTPUT"
  --resume)

if python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" --preflight-only; then
  printf 'ALREADY_COMPLETE; no model loaded\n'
  exit 0
else
  geodml_status=$?
  test "$geodml_status" = 3
fi
if [[ -e "$SEARCH_PRIMARY_OUTPUT/run_manifest.json" ]]; then
  if [[ "${SEARCH_ALLOW_RESUME:-0}" != 1 ]]; then
    printf 'STOP: partial run requires SEARCH_ALLOW_RESUME=1 after retry review\n' >&2
    exit 2
  fi
  if ! python3 -c 'import json, sys; from pathlib import Path; manifest=json.loads(Path(sys.argv[1]).read_text()); expected=str(Path(sys.argv[2]).resolve()); raise SystemExit(0 if manifest["serving_profile"]["path"] == expected else 1)' \
      "$SEARCH_PRIMARY_OUTPUT/run_manifest.json" "$geodml_profile"; then
    printf 'STOP: partial results have incompatible serving profile provenance\n' >&2
    exit 2
  fi
fi

mkdir -p "${SEARCH_PRIMARY_OUTPUT%/*}" "$SEARCH_PILOT_ROOT/logs"
geodml_prepare_args=(prepare
  --profile "$geodml_profile" --stage search-hf-primary
  --model-id "$SEARCH_PRIMARY_MODEL_ID"
  --model-revision "$SEARCH_PRIMARY_MODEL_REVISION"
  --vllm-executable "$ACL_ARR_VENV/bin/vllm"
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache"
  --expected-gpu-name-pattern GH200
  --port "$geodml_port"
  --data-parallel-size "$geodml_dp"
  --tensor-parallel-size "$geodml_tp"
  --request-concurrency "$geodml_concurrency"
  --dtype bfloat16
  --max-model-len "$geodml_max_model_len"
  --gpu-memory-utilization "$geodml_gpu_memory_utilization"
  --language-model-only
  --structured-outputs-config '{"backend":"xgrammar"}')
if [[ "$geodml_enforce_eager" == 1 ]]; then
  geodml_prepare_args+=(--enforce-eager)
fi
if [[ "$geodml_disable_custom_all_reduce" == 1 ]]; then
  geodml_prepare_args+=(--disable-custom-all-reduce)
fi
if [[ -n "$geodml_rope_scaling" ]]; then
  geodml_prepare_args+=(--rope-scaling "$geodml_rope_scaling")
fi
geodml_profile_hash="$(python3 analysis/scripts/search_vllm_stage.py "${geodml_prepare_args[@]}")"
printf 'SERVING_PROFILE=%s\nSERVING_PROFILE_SHA256=%s\n' "$geodml_profile" "$geodml_profile_hash"
geodml_approval_args=()
if (( geodml_dp > 1 )); then
  geodml_approval_path="${SEARCH_PRIMARY_BENCHMARK_APPROVAL_PATH:?DP requires benchmark approval}"
  python3 analysis/scripts/search_vllm_stage.py validate-approval \
    --profile "$geodml_profile" --approval "$geodml_approval_path"
  geodml_approval_args=(--benchmark-approval "$geodml_approval_path")
fi
geodml_args+=(--serving-profile "$geodml_profile")

python3 analysis/scripts/check_search_experience_grammar.py
python3 - "$geodml_max_model_len" "$geodml_answer_max_tokens" "$geodml_rope_scaling" <<'PY'
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

from analysis.scripts.check_acl_arr_context_budget import (
    _input_token_count,
    _native_context,
)
from analysis.scripts.run_search_experience import primary_items

locks = json.loads(
    (Path(os.environ["ACL_ARR_RUN_ROOT"]) / "model-snapshots.json").read_text()
)["models"]
model_id = os.environ["SEARCH_PRIMARY_MODEL_ID"]
revision = os.environ["SEARCH_PRIMARY_MODEL_REVISION"]
matches = [
    row
    for row in locks
    if row["model_id"] == model_id and row["revision"] == revision
]
if len(matches) != 1:
    raise ValueError("expected exactly one pinned model snapshot")
snapshot = Path(matches[0]["snapshot"])
native, source = _native_context(snapshot)
configured = int(sys.argv[1])
rope_scaling = json.loads(sys.argv[3]) if sys.argv[3] else None
if native is None:
    raise ValueError("native context is unknown; do not override silently")
if native < configured:
    expected = {
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "type": "yarn",
    }
    if model_id != "Qwen/Qwen2.5-72B-Instruct" or rope_scaling != expected:
        raise ValueError(
            "serving context exceeds native context without the approved "
            "Qwen2.5 YaRN configuration"
        )
    scaled_context = int(
        expected["factor"] * expected["original_max_position_embeddings"]
    )
    if configured > scaled_context:
        raise ValueError("serving context exceeds the configured YaRN context")
tokenizer = AutoTokenizer.from_pretrained(
    str(snapshot), local_files_only=True, trust_remote_code=True
)
items, identity, _ = primary_items(
    Path(os.environ["SEARCH_PILOT_ROOT"]) / "bundle",
    os.environ["SEARCH_PRIMARY_MODEL_CONFIGURATION_ID"],
    answer_max_tokens=int(sys.argv[2]),
)
if identity["model_id"] != model_id or identity["model_revision"] != revision:
    raise ValueError("frozen model identity differs from requested server")
required = max(
    _input_token_count(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    + item["max_tokens"]
    for item in items
)
if required > configured:
    raise ValueError("task exceeds configured context; no truncation permitted")
print(
    f"HF_PRIMARY_TOKEN_PREFLIGHT=PASS tasks={len(items)} required={required} "
    f"context={configured} native={native} source={source} "
    f"rope_scaling={json.dumps(rope_scaling, sort_keys=True)}"
)
PY

geodml_log="$(mktemp "$SEARCH_PILOT_ROOT/logs/hf-primary.XXXXXX")"
scontrol show job "$SLURM_JOB_ID" > "$geodml_log.allocation"
git rev-parse HEAD > "$geodml_log.commit"
printf 'MODEL=%s\nREVISION=%s\nCONFIGURATION=%s\nCONTEXT=%s\nDP=%s\nTP=%s\nCONCURRENCY=%s\nGPU_MEMORY_UTILIZATION=%s\nMAX_TASKS=%s\nANSWER_MAX_TOKENS=%s\nROPE_SCALING=%s\nSERVING_PROFILE_SHA256=%s\n' \
  "$SEARCH_PRIMARY_MODEL_ID" "$SEARCH_PRIMARY_MODEL_REVISION" \
  "$SEARCH_PRIMARY_MODEL_CONFIGURATION_ID" "$geodml_max_model_len" \
  "$geodml_dp" "$geodml_tp" "$geodml_concurrency" \
  "$geodml_gpu_memory_utilization" "$geodml_max_tasks" \
  "$geodml_answer_max_tokens" "$geodml_rope_scaling" \
  "$geodml_profile_hash" > "$geodml_log.settings"
printf 'SERVER_LOG=%s\n' "$geodml_log"
if python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$geodml_profile" --server-log "$geodml_log" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --startup-timeout-seconds "$geodml_startup_timeout_seconds" \
  "${geodml_approval_args[@]}" -- \
  python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" \
  2>&1 | tee "$geodml_log.controller"; then
  geodml_status=0
else
  geodml_status=$?
fi
printf 'HF_PRIMARY_STATUS=%s\nRESULTS=%s\nCONTROLLER_LOG=%s\n' \
  "$geodml_status" "$SEARCH_PRIMARY_OUTPUT" "$geodml_log.controller"
python3 - "$SEARCH_PRIMARY_OUTPUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "run_manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text())
    print("SUMMARY=" + json.dumps({
        key: manifest.get(key)
        for key in (
            "status", "task_count", "completed_count", "remaining_count",
            "failures_this_invocation", "answer_max_tokens_override",
        )
    }, sort_keys=True))
failures_path = root / "failures.jsonl"
if failures_path.is_file():
    failures = [
        json.loads(line)
        for line in failures_path.read_text().splitlines()
        if line.strip()
    ]
    print("ERROR_COUNTS=" + json.dumps(dict(Counter(
        row.get("error", "UNKNOWN") for row in failures
    )), sort_keys=True))
PY
exit "$geodml_status"
