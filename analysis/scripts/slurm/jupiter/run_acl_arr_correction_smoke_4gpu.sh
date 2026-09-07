#!/usr/bin/env bash
set -euo pipefail
step_start_epoch="$(date +%s)"
stop_submit_epoch=$((step_start_epoch + 900))

# Invoke with bash in a job step of the existing approved allocation. Never source
# this worker in the allocation-owning shell. Cleanup owns only its vLLM group.
: "${ACL_ARR_RUN_ROOT:?set the original pilot run directory}"
: "${ACL_ARR_CORRECTION_SOURCE:?set original answer results}"
: "${ACL_ARR_CORRECTION_REPAIR:?set latest formatting repair results}"
: "${ACL_ARR_VENV:?set the existing vLLM virtual environment}"
: "${ACL_ARR_CORRECTION_ROOT:?set a fresh answer/log directory}"
: "${ACL_ARR_CORRECTION_JOB_ID:?set the existing approved allocation ID}"
: "${ACL_ARR_CORRECTION_APPROVED_WALLTIME:?set the approved answer step limit}"
: "${ACL_ARR_CORRECTION_ESTIMATE:?set the approved answer estimate}"
: "${GEODML_CORRECTION_COMMIT:?set the answer checkout commit}"

if [[ "$ACL_ARR_CORRECTION_APPROVED_WALLTIME" != "00:20:00" ]]; then
    echo "ERROR: this answer step was approved only for 00:20:00" >&2
    exit 2
fi
if [[ "${SLURM_JOB_ID:-}" != "$ACL_ARR_CORRECTION_JOB_ID" ||
      -z "${SLURM_STEP_ID:-}" ]]; then
    echo "ERROR: run in a job step of the specified existing allocation" >&2
    exit 2
fi

if ! type module >/dev/null 2>&1; then
    set +u
    source /etc/profile
    set -u
fi
module load Stages/2026 GCC Python CUDA
module load git
hash -r
command -v git >/dev/null
command -v setsid >/dev/null
command -v curl >/dev/null

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"
test "$(git rev-parse HEAD)" = "$GEODML_CORRECTION_COMMIT"
git diff --quiet HEAD --
test -x "$ACL_ARR_VENV/bin/python"
test -x "$ACL_ARR_VENV/bin/vllm"
test -d "$ACL_ARR_CORRECTION_ROOT"
test ! -e "$ACL_ARR_CORRECTION_ROOT/results"

source "$ACL_ARR_VENV/bin/activate"
export PYTHONPATH="$REPOSITORY_ROOT:$REPOSITORY_ROOT/analysis"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_REVISION="92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"
SERVER_PORT=8003
SERVER_URL="http://127.0.0.1:${SERVER_PORT}/v1"
SERVER_LOG="$ACL_ARR_CORRECTION_ROOT/vllm-answer.log"
answer_args=(
    --source-results "$ACL_ARR_CORRECTION_SOURCE"
    --output-dir "$ACL_ARR_CORRECTION_ROOT/results"
    --base-url "$SERVER_URL"
    --repair-results "$ACL_ARR_CORRECTION_REPAIR"
    --stop-submit-epoch "$stop_submit_epoch"
    --approved-walltime "$ACL_ARR_CORRECTION_APPROVED_WALLTIME"
    --allocation-estimate "$ACL_ARR_CORRECTION_ESTIMATE"
)

# Validate frozen answer inputs and the completed rerank recovery before weights.
python3 analysis/scripts/run_acl_arr_correction_smoke.py \
    "${answer_args[@]}" --preflight-only

visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$visible_gpus" != "4" ]]; then
    echo "ERROR: expected four visible GPUs, found $visible_gpus" >&2
    exit 2
fi

server_args=(
    serve "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --served-model-name "$MODEL_ID"
    --host 127.0.0.1 --port "$SERVER_PORT"
    --tensor-parallel-size 4 --dtype bfloat16
    --max-model-len 49152 --gpu-memory-utilization 0.90
    --enable-prefix-caching --no-enable-log-requests --trust-remote-code
    --structured-outputs-config '{"backend":"xgrammar"}'
)

# Parse the exact server argv without invoking ServeSubcommand.cmd or creating
# an engine. This catches removed vLLM flags before another model load.
python3 - "${server_args[@]}" <<'PY'
import sys
from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
command = ServeSubcommand()
command.subparser_init(parser.add_subparsers(dest="subparser"))
args = parser.parse_args(sys.argv[1:])
command.validate(args)
print("CORRECTION_VLLM_CLI=PASS; no model engine started")
PY

# Validate the unchanged answer schema before any model weights are loaded.
python3 - <<'PY'
import json
import xgrammar
from analysis.scripts.run_acl_arr_vllm import _answer_schema
xgrammar.Grammar.from_json_schema(json.dumps(_answer_schema()))
print("CORRECTION_GRAMMAR=PASS; original answer schema; no model engine loaded")
PY

python3 - "$SERVER_PORT" "$ACL_ARR_RUN_ROOT" "$ACL_ARR_CORRECTION_ROOT" <<'PY'
import socket
import sys
from pathlib import Path

original, recovery = map(lambda p: Path(p).resolve(), sys.argv[2:])
if original == recovery or recovery in original.parents:
    raise SystemExit("ERROR: answer root must not be the original root or its ancestor")
with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as error:
        raise SystemExit(f"ERROR: answer port unavailable: {error}") from error
print("CORRECTION_PORT=AVAILABLE")
PY

# Atomic one-use guard keeps logs and incomplete attempts intact on failure.
mkdir "$ACL_ARR_CORRECTION_ROOT/worker-started"
python3 -m pip freeze > "$ACL_ARR_CORRECTION_ROOT/pip-freeze.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
    > "$ACL_ARR_CORRECTION_ROOT/gpu-before.csv"
printf 'CORRECTION_JOB_ID=%s\nCORRECTION_COMMIT=%s\nCORRECTION_ROOT=%s\n' \
    "$SLURM_JOB_ID" "$GEODML_CORRECTION_COMMIT" "$ACL_ARR_CORRECTION_ROOT"

server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]]; then
        # setsid establishes this child's private process group. Negative PID
        # signals only that group, never the Slurm allocation or owning shell.
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt=0; attempt<30; attempt++)); do
            if ! kill -0 -- "-$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "START_CORRECTION_SERVER model=$MODEL_ID revision=$MODEL_REVISION context=49152"
# With job control disabled, the child is not a process-group leader, so setsid
# execs vLLM without forking and its PID is also the private process-group ID.
set +m
setsid "$ACL_ARR_VENV/bin/vllm" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
server_pid=$!
server_ready=0
server_deadline=$((SECONDS + 900))
while (( SECONDS < server_deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: answer vLLM exited while loading" >&2
        tail -n 120 "$SERVER_LOG" >&2
        exit 1
    fi
    if curl --max-time 2 -fsS "$SERVER_URL/models" >/dev/null 2>&1; then
        server_ready=1
        break
    fi
    sleep 5
done
if [[ "$server_ready" != "1" ]]; then
    echo "ERROR: answer vLLM was not ready within 15 minutes" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
fi
echo "CORRECTION_SERVER_READY model=$MODEL_ID"
python3 analysis/scripts/run_acl_arr_correction_smoke.py "${answer_args[@]}"
echo "CORRECTION_WORKER=PASS; existing allocation remains owned by its original shell"
