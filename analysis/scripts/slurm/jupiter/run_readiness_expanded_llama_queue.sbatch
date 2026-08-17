#!/bin/bash -l
# Label the frozen transfer extension with three existing judges and Llama 3.3 70B.

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=readiness-expand

set -euo pipefail
umask 077

: "${PROJECT:?PROJECT is required}"
: "${FSCRATCH:?FSCRATCH is required}"
: "${USER:?USER is required}"
: "${SLURM_JOB_ID:?submit this script through Slurm}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"
: "${READINESS_TRANSFER_TASKS:?READINESS_TRANSFER_TASKS is required}"
: "${READINESS_TRANSFER_TASKS_SHA256:?READINESS_TRANSFER_TASKS_SHA256 is required}"
: "${READINESS_TRANSFER_TASKS_PER_SLOT:?READINESS_TRANSFER_TASKS_PER_SLOT is required}"
: "${READINESS_EXPANDED_TASKS:?READINESS_EXPANDED_TASKS is required}"
: "${READINESS_EXPANDED_TASKS_SHA256:?READINESS_EXPANDED_TASKS_SHA256 is required}"
: "${READINESS_EXPANDED_TASKS_PER_SLOT:?READINESS_EXPANDED_TASKS_PER_SLOT is required}"
: "${LLAMA_MODEL:?LLAMA_MODEL must name a local Llama 3.3 70B snapshot}"
: "${LLAMA_REVISION:?LLAMA_REVISION must be an immutable snapshot revision}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
GEODML_REPOSITORY="$GEODML_PROJECT_ROOT/src/geodml-mono"
GEODML_MODEL_VENV="${GEODML_MODEL_VENV:-$FSCRATCH/$USER/geodml/python/.venv-model-panel-transformers5141}"
GEODML_MODULE_PYTHONPATH="${PYTHONPATH-}"

source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages:$GEODML_MODULE_PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export GEODML_REQUIRED_GPU_COUNT=4
export GEODML_DEVICE_MAP=balanced
export GEODML_ATTENTION_IMPLEMENTATION=sdpa

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "scientific queue requires a clean checkout" >&2
    exit 2
fi

visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpus" != "4" ]]; then
    echo "expected four visible GPUs, found $visible_gpus" >&2
    exit 2
fi
while IFS= read -r gpu_name; do
    if [[ "$gpu_name" != *GH200* ]]; then
        echo "expected GH200 GPUs, found $gpu_name" >&2
        exit 2
    fi
done < <(nvidia-smi --query-gpu=name --format=csv,noheader)

validate_bank() {
    local path="$1"
    local expected_sha="$2"
    local expected_per_slot="$3"
    local actual_sha
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "task-bank hash mismatch: path=$path expected=$expected_sha actual=$actual_sha" >&2
        exit 2
    fi
    python - "$path" "$expected_per_slot" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
counts = collections.Counter()
task_ids = set()
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["task_id"] in task_ids:
        raise SystemExit(f"duplicate task ID: {row['task_id']}")
    task_ids.add(row["task_id"])
    counts[row["judge_slot"]] += 1
required = {
    "primary-frontier",
    "replicate-frontier-a",
    "replicate-frontier-b",
    "replicate-frontier-c",
}
if set(counts) != required:
    raise SystemExit(f"unexpected judge slots: {dict(counts)}")
if any(value != expected for value in counts.values()):
    raise SystemExit(f"task counts do not equal {expected}: {dict(counts)}")
print(f"task bank OK: {path}; per-slot={expected}; total={sum(counts.values())}")
PY
}

validate_bank \
    "$READINESS_TRANSFER_TASKS" \
    "$READINESS_TRANSFER_TASKS_SHA256" \
    "$READINESS_TRANSFER_TASKS_PER_SLOT"
validate_bank \
    "$READINESS_EXPANDED_TASKS" \
    "$READINESS_EXPANDED_TASKS_SHA256" \
    "$READINESS_EXPANDED_TASKS_PER_SLOT"

qwen32_revision="9216db5781bf21249d130ec9da846c4624c16137"
qwen32_model="$GEODML_PROJECT_ROOT/models/qwen/Qwen3-32B/$qwen32_revision"
ministral_revision="f6fae9795746f63c9be8344932f01275f3c63734"
ministral_model="$GEODML_PROJECT_ROOT/models/mistral/Ministral-3-8B-Instruct-2512-BF16/$ministral_revision"
gemma_revision="842da3794eaa0b77d5f08bae87a17459d91ff475"
gemma_model="$GEODML_PROJECT_ROOT/models/gemma/gemma-4-31B-it/$gemma_revision"

for model in "$qwen32_model" "$ministral_model" "$gemma_model" "$LLAMA_MODEL"; do
    if [[ ! -s "$model/config.json" ]]; then
        echo "missing model snapshot: $model" >&2
        exit 2
    fi
done

queue_root="${READINESS_EXPANDED_QUEUE_ROOT:-$GEODML_PROJECT_ROOT/runs/semantic-readiness-expanded-panel/$GEODML_EXPECTED_COMMIT}"
mkdir -p "$queue_root/logs" "$queue_root/smoke" "$queue_root/full"
printf '%s  %s\n' "$READINESS_TRANSFER_TASKS_SHA256" "$READINESS_TRANSFER_TASKS" > "$queue_root/transfer-task-bank-sha256.txt"
printf '%s  %s\n' "$READINESS_EXPANDED_TASKS_SHA256" "$READINESS_EXPANDED_TASKS" > "$queue_root/expanded-task-bank-sha256.txt"
git rev-parse HEAD > "$queue_root/git-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$queue_root/slurm-job-$SLURM_JOB_ID.txt"
nvidia-smi > "$queue_root/gpu-environment-start-$SLURM_JOB_ID.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$queue_root/started-at-$SLURM_JOB_ID.txt"

run_data_parallel() {
    local stage="$1"
    local tag="$2"
    local slot="$3"
    local model="$4"
    local family="$5"
    local revision="$6"
    local batch_size="$7"
    local limit="$8"
    local disable_thinking="$9"
    local output="$queue_root/$stage/$tag"
    local log="$queue_root/logs/$stage-$tag.log"
    local purpose=production
    local -a args=(
        analysis/scripts/run_semantic_readiness_judge_4gpu.py
        --tasks "$READINESS_TRANSFER_TASKS"
        --tasks-sha256 "$READINESS_TRANSFER_TASKS_SHA256"
        --expected-tasks "$READINESS_TRANSFER_TASKS_PER_SLOT"
        --judge-slot "$slot"
        --model "$model"
        --model-family "$family"
        --model-revision "$revision"
        --output-dir "$output"
        --batch-size "$batch_size"
        --max-input-tokens 2048
        --max-new-tokens 300
        --maximum-attempts 5
        --expected-world-size 4
        --attention-implementation sdpa
    )
    if [[ -n "$limit" ]]; then
        args+=(--limit "$limit")
        purpose=debug
    fi
    args+=(--run-purpose "$purpose")
    if [[ "$disable_thinking" == true ]]; then
        args+=(--disable-thinking)
    fi
    if [[ -e "$output" ]]; then
        args+=(--resume)
    fi
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting $stage/$tag" | tee -a "$queue_root/queue.log"
    python -m torch.distributed.run --standalone --nproc-per-node=4 "${args[@]}" 2>&1 | tee -a "$log"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] finished $stage/$tag exit=0" | tee -a "$queue_root/queue.log"
}

run_llama() {
    local stage="$1"
    local limit="$2"
    local output="$queue_root/$stage/llama3.3-70b-replicate-c"
    local log="$queue_root/logs/$stage-llama3.3-70b-replicate-c.log"
    local purpose=production
    local -a args=(
        analysis/scripts/run_semantic_readiness_judge.py
        --tasks "$READINESS_EXPANDED_TASKS"
        --judge-slot replicate-frontier-c
        --backend local
        --model "$LLAMA_MODEL"
        --model-family llama
        --model-revision "$LLAMA_REVISION"
        --precision full
        --batch-size "${LLAMA_BATCH_SIZE:-16}"
        --max-new-tokens 300
        --maximum-attempts 5
        --output-dir "$output"
    )
    if [[ -n "$limit" ]]; then
        args+=(--limit "$limit")
        purpose=debug
    fi
    args+=(--run-purpose "$purpose")
    if [[ -e "$output" ]]; then
        args+=(--resume)
    fi
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting $stage/llama3.3-70b-replicate-c" | tee -a "$queue_root/queue.log"
    python "${args[@]}" 2>&1 | tee -a "$log"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] finished $stage/llama3.3-70b-replicate-c exit=0" | tee -a "$queue_root/queue.log"
}

run_data_parallel smoke qwen3-32b-replicate-a replicate-frontier-a "$qwen32_model" qwen "$qwen32_revision" 16 8 true
run_data_parallel full qwen3-32b-replicate-a replicate-frontier-a "$qwen32_model" qwen "$qwen32_revision" 16 "" true
run_data_parallel smoke ministral3-8b-replicate-b replicate-frontier-b "$ministral_model" mistral "$ministral_revision" 32 8 false
run_data_parallel full ministral3-8b-replicate-b replicate-frontier-b "$ministral_model" mistral "$ministral_revision" 32 "" false
run_data_parallel smoke gemma4-31b-primary primary-frontier "$gemma_model" gemma "$gemma_revision" 8 8 false
run_data_parallel full gemma4-31b-primary primary-frontier "$gemma_model" gemma "$gemma_revision" 8 "" false
run_llama smoke 8
run_llama full ""

nvidia-smi > "$queue_root/gpu-environment-end-$SLURM_JOB_ID.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$queue_root/completed-at-$SLURM_JOB_ID.txt"
find "$queue_root" -type f ! -name artifact-sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$queue_root/artifact-sha256.txt"
echo "queue complete: $queue_root" | tee -a "$queue_root/queue.log"
