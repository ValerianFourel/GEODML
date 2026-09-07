#!/usr/bin/env bash
set -euo pipefail

# Invoke with bash in a job step of the existing approved allocation. Never source
# this worker in the allocation-owning shell. Cleanup owns only its vLLM group.
: "${ACL_ARR_RUN_ROOT:?set the original pilot run directory}"
: "${ACL_ARR_VENV:?set the existing vLLM virtual environment}"
: "${ACL_ARR_RECOVERY_ROOT:?set a fresh recovery/log directory}"
: "${ACL_ARR_RECOVERY_JOB_ID:?set the existing approved allocation ID}"
: "${ACL_ARR_RECOVERY_APPROVED_WALLTIME:?set the approved recovery step limit}"
: "${ACL_ARR_RECOVERY_ESTIMATE:?set the approved recovery estimate}"
: "${GEODML_RECOVERY_COMMIT:?set the recovery checkout commit}"

if [[ "$ACL_ARR_RECOVERY_APPROVED_WALLTIME" != "00:30:00" ]]; then
    echo "ERROR: this recovery step was approved only for 00:30:00" >&2
    exit 2
fi
if [[ "${SLURM_JOB_ID:-}" != "$ACL_ARR_RECOVERY_JOB_ID" ||
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
test "$(git rev-parse HEAD)" = "$GEODML_RECOVERY_COMMIT"
git diff --quiet HEAD --
test -x "$ACL_ARR_VENV/bin/python"
test -x "$ACL_ARR_VENV/bin/vllm"
test -d "$ACL_ARR_RECOVERY_ROOT"
test ! -e "$ACL_ARR_RECOVERY_ROOT/results"

source "$ACL_ARR_VENV/bin/activate"
export PYTHONPATH="$REPOSITORY_ROOT:$REPOSITORY_ROOT/analysis"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_ID="meta-llama/Llama-4-Scout-17B-16E-Instruct"
MODEL_REVISION="92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"
SERVER_PORT=8001
SERVER_URL="http://127.0.0.1:${SERVER_PORT}/v1"
SERVER_LOG="$ACL_ARR_RECOVERY_ROOT/vllm-recovery.log"
recovery_args=(
    --run-root "$ACL_ARR_RUN_ROOT"
    --output-dir "$ACL_ARR_RECOVERY_ROOT/results"
    --base-url "$SERVER_URL"
    --max-model-len 40960
    --max-concurrency 8
    --approved-walltime "$ACL_ARR_RECOVERY_APPROVED_WALLTIME"
    --allocation-estimate "$ACL_ARR_RECOVERY_ESTIMATE"
    --expected-plan-commit f9c35e499b708a4c812b49a3f550e097306ce25d
)

# Read-only validation checks the original 384 tasks, 319 successes and 65
# unresolved failures. The original plan commit deliberately differs from this
# recovery checkout; both are recorded by the recovery manifest.
python3 analysis/scripts/recover_acl_arr_llama_rerank.py \
    "${recovery_args[@]}" --preflight-only

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
    --max-model-len 40960 --gpu-memory-utilization 0.90
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
print("RECOVERY_VLLM_CLI=PASS; no model engine started")
PY

# Use the installed grammar library to construct both recovery schema shapes.
# This needs neither a tokenizer nor model weights. Any unsupported enum,
# prefixItems, or const constraint must fail before the server is started.
python3 - <<'PY'
import json
import xgrammar
from analysis.scripts.acl_arr_recovery_decode import _schema
from analysis.scripts.run_acl_arr_vllm import _rerank_schema

item = {"schema": _rerank_schema(3)}
allowed = ["C001", "C002", "C003", "C004"]
for prefix in ([], ["C002"]):
    schema = _schema(item, allowed, 3, prefix)
    xgrammar.Grammar.from_json_schema(json.dumps(schema))
print("RECOVERY_GRAMMAR=PASS schemas=2; no tokenizer or model engine loaded")
PY

python3 - "$SERVER_PORT" "$ACL_ARR_RUN_ROOT" "$ACL_ARR_RECOVERY_ROOT" <<'PY'
import socket
import sys
from pathlib import Path

original, recovery = map(lambda p: Path(p).resolve(), sys.argv[2:])
if original == recovery or recovery in original.parents:
    raise SystemExit("ERROR: recovery root must not be the original root or its ancestor")
with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as error:
        raise SystemExit(f"ERROR: recovery port unavailable: {error}") from error
print("RECOVERY_PORT=AVAILABLE")
PY

# Atomic one-use guard keeps logs and incomplete attempts intact on failure.
mkdir "$ACL_ARR_RECOVERY_ROOT/worker-started"
python3 -m pip freeze > "$ACL_ARR_RECOVERY_ROOT/pip-freeze.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
    > "$ACL_ARR_RECOVERY_ROOT/gpu-before.csv"
printf 'RECOVERY_JOB_ID=%s\nRECOVERY_COMMIT=%s\nRECOVERY_ROOT=%s\n' \
    "$SLURM_JOB_ID" "$GEODML_RECOVERY_COMMIT" "$ACL_ARR_RECOVERY_ROOT"

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

echo "START_RECOVERY_SERVER model=$MODEL_ID revision=$MODEL_REVISION context=40960"
# With job control disabled, the child is not a process-group leader, so setsid
# execs vLLM without forking and its PID is also the private process-group ID.
set +m
setsid "$ACL_ARR_VENV/bin/vllm" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
server_pid=$!
server_ready=0
server_deadline=$((SECONDS + 900))
while (( SECONDS < server_deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: recovery vLLM exited while loading" >&2
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
    echo "ERROR: recovery vLLM was not ready within 15 minutes" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
fi
echo "RECOVERY_SERVER_READY model=$MODEL_ID"
python3 analysis/scripts/recover_acl_arr_llama_rerank.py "${recovery_args[@]}"
echo "RECOVERY_WORKER=PASS; existing allocation remains owned by its original shell"
