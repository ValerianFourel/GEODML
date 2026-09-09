#!/usr/bin/env bash
# Existing-allocation worker for the frozen text-only Mistral search pilot.
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
geodml_output="$SEARCH_PILOT_ROOT/primary-schema-fix-0b3ce8acb5d6/model-config-c860fb2fb61da06a8443"
geodml_args=(run-primary --bundle-dir "$SEARCH_PILOT_ROOT/bundle"
  --model-configuration-id model-config-c860fb2fb61da06a8443
  --server-model-revision "$geodml_revision" --base-url http://127.0.0.1:8010/v1
  --max-concurrency 8 --output-dir "$geodml_output" --resume)
if python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" --preflight-only; then
  printf 'ALREADY_COMPLETE; no model loaded\n'
  exit 0
else
  geodml_status=$?
  test "$geodml_status" = 3
fi
if [[ -e "$geodml_output/run_manifest.json" ]]; then
  printf 'STOP: partial results exist; preserve them for retry review\n'
  exit 2
fi
geodml_cache="${GEODML_CACHE_ROOT:?}/compile-cache/vllm028-torch213-job${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="$geodml_cache/vllm" TORCHINDUCTOR_CACHE_DIR="$geodml_cache/inductor"
export TRITON_CACHE_DIR="$geodml_cache/triton" CUDA_CACHE_PATH="$geodml_cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$geodml_cache/flashinfer-workspace"
python3 - <<'PY'
import os, socket, tempfile
from pathlib import Path
for key in ("VLLM_CACHE_ROOT","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","CUDA_CACHE_PATH",
            "FLASHINFER_WORKSPACE_BASE"):
    path = Path(os.environ[key]).resolve()
    home = Path.home().resolve()
    assert path != home and home not in path.parents
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as f:
        f.write(b"check"); f.flush(); os.fsync(f.fileno())
with socket.socket() as s:
    s.bind(("127.0.0.1",8010))
print("CACHE_AND_PORT=PASS")
PY
geodml_help="$("$ACL_ARR_VENV/bin/vllm" serve --help=all)"
if [[ "$geodml_help" != *"--language-model-only"* ]]; then
  printf 'STOP: installed vLLM lacks language-model-only; no weights loaded\n' >&2
  exit 2
fi
printf 'TEXT_ONLY_CLI=PASS; no multimodal inputs permitted\n'
geodml_native_report="$(python3 analysis/scripts/check_search_mistral_native.py --model-snapshots "${ACL_ARR_RUN_ROOT:?}/model-snapshots.json")"
printf 'NATIVE_CONFIG_PREFLIGHT=PASS; no model weights loaded\n'
python3 analysis/scripts/check_search_experience_grammar.py
command -v setsid >/dev/null
command -v curl >/dev/null
mkdir -p "$SEARCH_PILOT_ROOT/logs"
geodml_log="$(mktemp "$SEARCH_PILOT_ROOT/logs/mistral-primary.XXXXXX")"
scontrol show job "$SLURM_JOB_ID" > "$geodml_log.allocation"
printf '%s\n' "$geodml_native_report" > "$geodml_log.native-config.json"
printf 'COMMIT=%s\nMODEL=%s\nREVISION=%s\nTOKENIZER=mistral\nLANGUAGE_MODEL_ONLY=true\nCONTEXT=41472\nTP=4\nDTYPE=bfloat16\nCONCURRENCY=8\nSTEP_CAP=00:45:00\nESTIMATE=10-30 minutes; loading and inference unmeasured for this arm\n' "$GEODML_EXECUTION_COMMIT" "$geodml_model" "$geodml_revision" > "$geodml_log.settings"
geodml_pid=""
cleanup() {
  if [[ -n "$geodml_pid" ]]; then
    kill -TERM -- "-$geodml_pid" 2>/dev/null || true
    for ((i=0;i<30;i++)); do
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
setsid "$ACL_ARR_VENV/bin/vllm" serve "$geodml_model" \
  --revision "$geodml_revision" --served-model-name "$geodml_model" \
  --language-model-only --tokenizer-mode mistral --attention-backend FLASH_ATTN_MLA \
  --config-format mistral --load-format mistral \
  --host 127.0.0.1 --port 8010 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 41472 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --no-enable-log-requests --trust-remote-code \
  --structured-outputs-config '{"backend":"xgrammar"}' > "$geodml_log" 2>&1 &
geodml_pid=$!
geodml_ready=0
for ((i=0;i<180;i++)); do
  if ! kill -0 "$geodml_pid" 2>/dev/null; then tail -n 80 "$geodml_log"; exit 1; fi
  if curl --max-time 2 -fsS http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
    geodml_ready=1
    break
  fi
  sleep 5
done
if [[ "$geodml_ready" != 1 ]]; then tail -n 80 "$geodml_log"; exit 1; fi
if python3 analysis/scripts/run_search_experience.py "${geodml_args[@]}" 2>&1 | tee "$geodml_log.controller"; then
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
