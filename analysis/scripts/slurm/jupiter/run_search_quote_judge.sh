#!/usr/bin/env bash
# Run inside an existing GPU Slurm step. Never requests or cancels an allocation.
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
geodml_args=(run-judge --bundle-dir "$SEARCH_BUNDLE_DIR" --primary-output "$SEARCH_PRIMARY_OUTPUT"
  --judge-model-id "$SEARCH_JUDGE_MODEL" --judge-model-revision "$SEARCH_JUDGE_REVISION"
  --judge-contract search-experience-judge-quotes-v2 --output-dir "$SEARCH_JUDGE_OUTPUT"
  --base-url http://127.0.0.1:8010/v1 --max-concurrency 8 --resume)
set +e
python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" --preflight-only
geodml_status=$?
set -e
if [[ "$geodml_status" == 0 ]]; then
  printf 'ALREADY_COMPLETE; verified saved judgments; no model loaded\n'
  exit 0
fi
if [[ "$geodml_status" != 3 ]]; then exit "$geodml_status"; fi
# Partial runs need explicit retry authorization; complete runs never retry.
if [[ -e "$SEARCH_JUDGE_OUTPUT/run_manifest.json" && "${SEARCH_ALLOW_RESUME:-0}" != 1 ]]; then
  printf 'STOP: partial run requires SEARCH_ALLOW_RESUME=1 after retry review\n' >&2
  exit 2
fi
umask 077
geodml_cache="${GEODML_CACHE_ROOT:?}/compile-cache/vllm028-torch213-job${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="$geodml_cache/vllm" TORCHINDUCTOR_CACHE_DIR="$geodml_cache/inductor"
export TRITON_CACHE_DIR="$geodml_cache/triton" CUDA_CACHE_PATH="$geodml_cache/cuda"
python3 analysis/scripts/check_search_experience_grammar.py
python3 - <<'PY'
import json, os, socket, tempfile
from pathlib import Path
from transformers import AutoTokenizer
from analysis.scripts.run_search_experience import judge_items
from analysis.scripts.check_acl_arr_context_budget import _input_token_count, _native_context
for key in ('VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
    path = Path(os.environ[key]).resolve()
    home = Path.home().resolve()
    if path == home or home in path.parents:
        raise ValueError('cache must be outside home')
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as f:
        f.write(b'check'); f.flush(); os.fsync(f.fileno())
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
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 8010))
print(f'TOKEN_PREFLIGHT=PASS tasks={len(items)} required={required} context=49152')
PY
command -v setsid >/dev/null
command -v curl >/dev/null
mkdir -p "${SEARCH_JUDGE_OUTPUT%/*}"
geodml_log="$(mktemp "${SEARCH_JUDGE_OUTPUT}.server.XXXXXX")"
scontrol show job "$SLURM_JOB_ID" > "$geodml_log.allocation"
git rev-parse HEAD > "$geodml_log.commit"
geodml_pid=''
cleanup() {
  if [[ -n "$geodml_pid" ]]; then
    kill -TERM -- "-$geodml_pid" 2>/dev/null || true
    for ((i=0; i<30; i++)); do
      kill -0 -- "-$geodml_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$geodml_pid" 2>/dev/null || true
    wait "$geodml_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +m
printf 'SERVER_LOG=%s\n' "$geodml_log"
setsid "$ACL_ARR_VENV/bin/vllm" serve "$SEARCH_JUDGE_MODEL" \
  --revision "$SEARCH_JUDGE_REVISION" --served-model-name "$SEARCH_JUDGE_MODEL" \
  --host 127.0.0.1 --port 8010 --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 49152 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --no-enable-log-requests --trust-remote-code \
  --structured-outputs-config '{"backend":"xgrammar"}' > "$geodml_log" 2>&1 &
geodml_pid=$!
geodml_ready=0
for ((i=0; i<180; i++)); do
  if ! kill -0 "$geodml_pid" 2>/dev/null; then tail -n 100 "$geodml_log"; exit 1; fi
  if curl --max-time 2 -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
    geodml_ready=1
    break
  fi
  sleep 5
done
if [[ "$geodml_ready" != 1 ]]; then tail -n 100 "$geodml_log"; exit 1; fi
python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" 2>&1 | tee "$geodml_log.controller"
