#!/usr/bin/env bash
set -euo pipefail

# Run inside an approved four-GH200 Slurm allocation after the pilot plan and
# all pinned model snapshots are available on shared storage.

: "${ACL_ARR_RUN_ROOT:?set ACL_ARR_RUN_ROOT to the pilot run directory}"
: "${ACL_ARR_VENV:?set ACL_ARR_VENV to the vLLM virtual environment}"
: "${ACL_ARR_APPROVED_WALLTIME:?set ACL_ARR_APPROVED_WALLTIME}"
: "${ACL_ARR_ALLOCATION_ESTIMATE:?set ACL_ARR_ALLOCATION_ESTIMATE}"

if [[ "$ACL_ARR_APPROVED_WALLTIME" != "03:00:00" ]]; then
    echo "ERROR: this pilot was approved only for 03:00:00" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this script inside the approved Slurm allocation" >&2
    exit 2
fi

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPOSITORY_ROOT"
PLAN_ROOT="$ACL_ARR_RUN_ROOT/plan"
PLAN_MANIFEST="$PLAN_ROOT/run_manifest.json"
RESULTS_ROOT="$ACL_ARR_RUN_ROOT/results"
JUDGE_PLAN_ROOT="$ACL_ARR_RUN_ROOT/judge-plan"
ANALYSIS_ROOT="$ACL_ARR_RUN_ROOT/analysis"
LOG_ROOT="$ACL_ARR_RUN_ROOT/logs"
RUNTIME_MANIFEST="$ACL_ARR_RUN_ROOT/pilot-runtime-manifest.json"
SERVER_PORT="${ACL_ARR_SERVER_PORT:-8000}"
SERVER_URL="http://127.0.0.1:${SERVER_PORT}/v1"
MAX_CONCURRENCY="${ACL_ARR_MAX_CONCURRENCY:-16}"
MAX_MODEL_LEN="${ACL_ARR_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${ACL_ARR_TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${ACL_ARR_GPU_MEMORY_UTILIZATION:-0.90}"

test -s "$PLAN_MANIFEST"
test -x "$ACL_ARR_VENV/bin/python"
test -x "$ACL_ARR_VENV/bin/vllm"
mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

module load Stages/2026 GCC Python CUDA >/dev/null 2>&1 || true
source "$ACL_ARR_VENV/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

current_commit="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"
planned_commit="$(python3 - "$PLAN_MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["source_git_commit"])
PY
)"
if [[ "$current_commit" != "$planned_commit" ]]; then
    echo "ERROR: checkout $current_commit does not match plan $planned_commit" >&2
    exit 2
fi

visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$visible_gpus" != "4" ]]; then
    echo "ERROR: expected four visible GPUs, found $visible_gpus" >&2
    exit 2
fi

python3 - "$RUNTIME_MANIFEST" "$PLAN_MANIFEST" <<'PY'
import datetime
import json
import os
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
manifest = {
    "format_version": "acl-arr-document-pilot-runtime-v1",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "plan_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "git_commit_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["ACL_ARR_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["ACL_ARR_ALLOCATION_ESTIMATE"],
    "resources": {
        "nodes": os.environ.get("SLURM_JOB_NUM_NODES", "1"),
        "gpus": 4,
        "cpus": int(os.environ.get("SLURM_CPUS_ON_NODE", "32")),
        "memory": "512G",
    },
    "scientific_result": False,
    "purpose": "throughput and protocol pilot",
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
python3 -m pip freeze > "$ACL_ARR_RUN_ROOT/pip-freeze.txt"

server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT INT TERM

start_server() {
    local model_id="$1"
    local revision="$2"
    local configuration_id="$3"
    local server_log="$LOG_ROOT/vllm-${configuration_id}.log"
    local extra_args=()

    stop_server
    if [[ "$model_id" == mistralai/Mistral-Small-4-* ]]; then
        extra_args+=(--attention-backend FLASH_ATTN_MLA)
    fi
    echo "START_SERVER model=$model_id revision=$revision"
    vllm serve "$model_id" \
        --revision "$revision" \
        --served-model-name "$model_id" \
        --host 127.0.0.1 \
        --port "$SERVER_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enable-prefix-caching \
        --disable-log-requests \
        --trust-remote-code \
        "${extra_args[@]}" >"$server_log" 2>&1 &
    server_pid=$!

    for _ in $(seq 1 180); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: vLLM exited while loading $model_id" >&2
            tail -n 120 "$server_log" >&2
            return 1
        fi
        if curl -fsS "$SERVER_URL/models" >/dev/null 2>&1; then
            echo "SERVER_READY model=$model_id"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not become ready for $model_id" >&2
    tail -n 120 "$server_log" >&2
    return 1
}

mapfile -t model_rows < <(python3 - "$PLAN_MANIFEST" <<'PY'
import json
import sys
for model in json.load(open(sys.argv[1], encoding="utf-8"))["models"]:
    print("\t".join((model["configuration_id"], model["model_id"], model["model_revision"])))
PY
)
if [[ "${#model_rows[@]}" != "4" ]]; then
    echo "ERROR: expected four model configurations" >&2
    exit 2
fi

answer_arguments=()
rerank_arguments=()
analysis_answer_arguments=()
for model_row in "${model_rows[@]}"; do
    IFS=$'\t' read -r configuration_id model_id revision <<< "$model_row"
    start_server "$model_id" "$revision" "$configuration_id"
    for pipeline in rerank answer; do
        task_file="$PLAN_ROOT/tasks/$configuration_id/$pipeline.jsonl"
        output_root="$RESULTS_ROOT/$configuration_id/$pipeline"
        test -s "$task_file"
        python3 analysis/scripts/run_acl_arr_vllm.py primary \
            --tasks "$task_file" \
            --plan-manifest "$PLAN_MANIFEST" \
            --base-url "$SERVER_URL" \
            --server-model-name "$model_id" \
            --server-model-revision "$revision" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --request-timeout 600 \
            --max-attempts 3 \
            --resume \
            --output-dir "$output_root"
    done
    stop_server
    rerank_arguments+=(--rerank-outcomes "$RESULTS_ROOT/$configuration_id/rerank/outcomes.jsonl")
    analysis_answer_arguments+=(--answer-outcomes "$RESULTS_ROOT/$configuration_id/answer/outcomes.jsonl")
    answer_arguments+=(--answer-outcomes "$RESULTS_ROOT/$configuration_id/answer/outcomes.jsonl")
done

judge_model_id="Qwen/Qwen2.5-72B-Instruct"
judge_row="$(printf '%s\n' "${model_rows[@]}" | awk -F '\t' -v model="$judge_model_id" '$2 == model {print; exit}')"
if [[ -z "$judge_row" ]]; then
    echo "ERROR: approved judge model is absent from the plan" >&2
    exit 2
fi
IFS=$'\t' read -r judge_configuration_id _ judge_revision <<< "$judge_row"

if [[ ! -s "$JUDGE_PLAN_ROOT/judge_manifest.json" ]]; then
    python3 analysis/scripts/prepare_acl_arr_judge_tasks.py \
        "${answer_arguments[@]}" \
        --plan-manifest "$PLAN_MANIFEST" \
        --judge-model-id "$judge_model_id" \
        --judge-model-revision "$judge_revision" \
        --master-seed 20260905 \
        --output-dir "$JUDGE_PLAN_ROOT"
fi

start_server "$judge_model_id" "$judge_revision" "judge-${judge_configuration_id}"
python3 analysis/scripts/run_acl_arr_vllm.py judge \
    --tasks "$JUDGE_PLAN_ROOT/judge_tasks.jsonl" \
    --judge-manifest "$JUDGE_PLAN_ROOT/judge_manifest.json" \
    --base-url "$SERVER_URL" \
    --server-model-name "$judge_model_id" \
    --server-model-revision "$judge_revision" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-timeout 600 \
    --max-attempts 3 \
    --resume \
    --output-dir "$RESULTS_ROOT/judge"
stop_server

if [[ ! -d "$ANALYSIS_ROOT" ]]; then
    python3 analysis/scripts/analyze_acl_arr_experiment.py \
        --plan-manifest "$PLAN_MANIFEST" \
        "${rerank_arguments[@]}" \
        "${analysis_answer_arguments[@]}" \
        --judge-outcomes "$RESULTS_ROOT/judge/outcomes.jsonl" \
        --private-judge-mapping "$JUDGE_PLAN_ROOT/private_judge_mapping.jsonl" \
        --output-dir "$ANALYSIS_ROOT"
fi

python3 - "$RUNTIME_MANIFEST" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest["status"] = "PASS"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
echo "ACL_ARR_PILOT=PASS"
echo "RUN_ROOT=$ACL_ARR_RUN_ROOT"
