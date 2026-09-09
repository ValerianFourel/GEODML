#!/usr/bin/env bash
set -eo pipefail
trap 'printf "JUDGE_WORKER_FAILED line=%s status=%s\n" "$LINENO" "$?" >&2' ERR
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA
module load git
set -u
source "${ACL_ARR_VENV:?}/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
test "$(git rev-parse HEAD)" = "${GEODML_EXECUTION_COMMIT:?}"
git diff --quiet HEAD --
test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
: "${SEARCH_BUNDLE_DIR:?}" "${SEARCH_PRIMARY_OUTPUT:?}" "${SEARCH_JUDGE_OUTPUT:?}"
: "${SEARCH_JUDGE_MODEL:?}" "${SEARCH_JUDGE_REVISION:?}" "${ACL_ARR_RUN_ROOT:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
geodml_dp="${SEARCH_JUDGE_DATA_PARALLEL_SIZE:-1}"
geodml_tp="${SEARCH_JUDGE_TENSOR_PARALLEL_SIZE:-4}"
geodml_concurrency="${SEARCH_JUDGE_REQUEST_CONCURRENCY:-8}"
geodml_port="${SEARCH_JUDGE_PORT:-8010}"
geodml_base_url="http://127.0.0.1:${geodml_port}/v1"
geodml_profile="${SEARCH_JUDGE_OUTPUT}.serving-profile.json"
geodml_args=(run-judge --bundle-dir "$SEARCH_BUNDLE_DIR" --primary-output "$SEARCH_PRIMARY_OUTPUT"
  --judge-model-id "$SEARCH_JUDGE_MODEL" --judge-model-revision "$SEARCH_JUDGE_REVISION"
  --judge-contract search-experience-judge-quotes-v2 --output-dir "$SEARCH_JUDGE_OUTPUT"
  --base-url "$geodml_base_url" --max-concurrency "$geodml_concurrency" --resume)
if python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" --preflight-only; then
  geodml_status=0
else
  geodml_status=$?
fi
if [[ "$geodml_status" == 0 ]]; then
  printf 'ALREADY_COMPLETE; verified saved judgments; no model loaded\n'
  exit 0
fi
if [[ "$geodml_status" != 3 ]]; then exit "$geodml_status"; fi
if [[ -e "$SEARCH_JUDGE_OUTPUT/run_manifest.json" ]]; then
  if [[ "${SEARCH_ALLOW_RESUME:-0}" != 1 ]]; then
    printf 'STOP: partial run requires SEARCH_ALLOW_RESUME=1 after retry review\n' >&2
    exit 2
  fi
  if ! python3 -c 'import json, sys; from pathlib import Path; manifest=json.loads(Path(sys.argv[1]).read_text()); expected=str(Path(sys.argv[2]).resolve()); raise SystemExit(0 if manifest["serving_profile"]["path"] == expected else 1)' \
      "$SEARCH_JUDGE_OUTPUT/run_manifest.json" "$geodml_profile"; then
    printf 'STOP: historical partial results have no serving profile provenance\n' >&2
    exit 2
  fi
fi
umask 077
geodml_profile_hash="$(python3 analysis/scripts/search_vllm_stage.py prepare \
  --profile "$geodml_profile" --stage search-quote-judge \
  --model-id "$SEARCH_JUDGE_MODEL" --model-revision "$SEARCH_JUDGE_REVISION" \
  --vllm-executable "$ACL_ARR_VENV/bin/vllm" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --expected-gpu-name-pattern GH200 \
  --port "$geodml_port" --data-parallel-size "$geodml_dp" \
  --tensor-parallel-size "$geodml_tp" --request-concurrency "$geodml_concurrency" \
  --dtype bfloat16 --max-model-len 49152 --gpu-memory-utilization 0.90 \
  --structured-outputs-config '{"backend":"xgrammar"}')"
printf 'SERVING_PROFILE=%s\nSERVING_PROFILE_SHA256=%s\n' "$geodml_profile" "$geodml_profile_hash"
geodml_approval_args=()
if (( geodml_dp > 1 )); then
  geodml_approval_path="${SEARCH_JUDGE_BENCHMARK_APPROVAL_PATH:?DP requires benchmark approval}"
  python3 analysis/scripts/search_vllm_stage.py validate-approval \
    --profile "$geodml_profile" \
    --approval "$geodml_approval_path"
  geodml_approval_args=(--benchmark-approval "$geodml_approval_path")
fi
geodml_args+=(--serving-profile "$geodml_profile")
python3 analysis/scripts/check_search_experience_grammar.py
python3 - <<'PY'
import json, os
from pathlib import Path
from transformers import AutoTokenizer
from analysis.scripts.run_search_experience import judge_items
from analysis.scripts.check_acl_arr_context_budget import _input_token_count, _native_context
locks = json.loads((Path(os.environ['ACL_ARR_RUN_ROOT']) / 'model-snapshots.json').read_text())['models']
matches = [r for r in locks if r['model_id'] == os.environ['SEARCH_JUDGE_MODEL']
           and r['revision'] == os.environ['SEARCH_JUDGE_REVISION']]
if len(matches) != 1:
    raise ValueError('expected exactly one pinned snapshot')
snapshot = Path(matches[0]['snapshot'])
native, _ = _native_context(snapshot)
if native < 49152:
    raise ValueError('native context below serving context; do not override silently')
tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
items, _, _ = judge_items(os.environ['SEARCH_BUNDLE_DIR'], os.environ['SEARCH_PRIMARY_OUTPUT'],
    os.environ['SEARCH_JUDGE_MODEL'], os.environ['SEARCH_JUDGE_REVISION'],
    judge_contract='search-experience-judge-quotes-v2')
required = max(_input_token_count(tokenizer.apply_chat_template(
    [{'role': 'user', 'content': item['prompt']}], tokenize=True, add_generation_prompt=True))
    + item['max_tokens'] for item in items)
if required > 49152:
    raise ValueError('task exceeds context; no truncation permitted')
print(f'TOKEN_PREFLIGHT=PASS tasks={len(items)} required={required} context=49152')
PY
mkdir -p "${SEARCH_JUDGE_OUTPUT%/*}"
geodml_log="$(mktemp "${SEARCH_JUDGE_OUTPUT}.server.XXXXXX")"
scontrol show job "$SLURM_JOB_ID" > "$geodml_log.allocation"
git rev-parse HEAD > "$geodml_log.commit"
printf 'SERVER_LOG=%s\n' "$geodml_log"
if python3 analysis/scripts/search_vllm_stage.py run \
  --profile "$geodml_profile" --server-log "$geodml_log" \
  --cache-base "${GEODML_CACHE_ROOT:?}/compile-cache" \
  --startup-timeout-seconds 900 "${geodml_approval_args[@]}" -- \
  python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" \
  2>&1 | tee "$geodml_log.controller"; then
  geodml_status=0
else
  geodml_status=$?
fi
printf 'JUDGE_STATUS=%s\nRESULTS=%s\nCONTROLLER_LOG=%s\n' "$geodml_status" "$SEARCH_JUDGE_OUTPUT" "$geodml_log.controller"
if ! python3 - "$SEARCH_JUDGE_OUTPUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
manifest = root / 'run_manifest.json'
if manifest.is_file():
    record = json.loads(manifest.read_text())
    print('SUMMARY=' + json.dumps({key: record.get(key) for key in (
        'status', 'completed_count', 'remaining_count', 'failures_this_invocation', 'judge_contract')}))
else:
    print('MANIFEST=MISSING; inspect controller log')
path = root / 'failures.jsonl'
counts, examples = Counter(), {}
if path.is_file():
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            error = row.get('error', 'UNKNOWN')
            counts[error] += 1
            examples.setdefault(error, {key: row.get(key) for key in ('task_id', 'error', 'raw_output', 'usage')})
print('HISTORICAL_FAILURE_COUNTS=' + json.dumps(dict(counts)))
for row in examples.values():
    print('EXAMPLE=' + json.dumps(row, ensure_ascii=False))
PY
then
  printf 'RESULT_DIAGNOSTIC_FAILED; preserve results and inspect controller log\n' >&2
fi
exit "$geodml_status"
