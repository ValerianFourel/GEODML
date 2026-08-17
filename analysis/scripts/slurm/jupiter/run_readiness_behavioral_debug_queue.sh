#!/usr/bin/env bash
set -euo pipefail
umask 077

: "${PROJECT:?PROJECT is required}"
: "${FSCRATCH:?FSCRATCH is required}"
: "${USER:?USER is required}"
: "${SLURM_JOB_ID:?run this inside an active four-GPU allocation}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1
jutil env activate -p scifi

GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
GEODML_REPOSITORY="$GEODML_PROJECT_ROOT/src/geodml-mono"
GEODML_MODEL_VENV="$FSCRATCH/$USER/geodml/python/.venv-model-panel-transformers5141"
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

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "repository checkout is dirty" >&2
    exit 2
fi

visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpus" != "4" ]]; then
    echo "expected four visible GPUs, found $visible_gpus" >&2
    exit 2
fi

tasks="$GEODML_PROJECT_ROOT/runs/semantic-readiness-base-axis/f6d9e6df42c90b425e4035bb9f28cb551be63175/label-tasks/readiness_label_tasks_blinded.jsonl"
tasks_sha256="9c1e084332d4fc3129a1f1c5400b8118d7a3425a01f3c771edb133d66d496775"
queue_root="$GEODML_PROJECT_ROOT/runs/semantic-readiness-debug/$SLURM_JOB_ID/unattended-behavioral-queue-$GEODML_EXPECTED_COMMIT"
mkdir -p "$queue_root/logs" "$queue_root/smoke" "$queue_root/full"

actual_tasks_sha256="$(sha256sum "$tasks" | awk '{print $1}')"
if [[ "$actual_tasks_sha256" != "$tasks_sha256" ]]; then
    echo "task-bank hash mismatch: expected=$tasks_sha256 actual=$actual_tasks_sha256" >&2
    exit 2
fi
printf '%s  %s\n' "$actual_tasks_sha256" "$tasks" > "$queue_root/task-bank-sha256.txt"
git rev-parse HEAD > "$queue_root/git-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$queue_root/slurm-job.txt"
nvidia-smi > "$queue_root/gpu-environment-start.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$queue_root/started-at.txt"

run_stage() {
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
    local -a args=(
        analysis/scripts/run_semantic_readiness_judge_4gpu.py
        --tasks "$tasks"
        --tasks-sha256 "$tasks_sha256"
        --expected-tasks 5091
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
        --run-purpose debug
    )
    if [[ -n "$limit" ]]; then
        args+=(--limit "$limit")
    fi
    if [[ "$disable_thinking" == "true" ]]; then
        args+=(--disable-thinking)
    fi
    if [[ -e "$output" ]]; then
        args+=(--resume)
    fi

    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting $stage/$tag" | tee -a "$queue_root/queue.log"
    set +e
    python -m torch.distributed.run --standalone --nproc-per-node=4 "${args[@]}" 2>&1 | tee -a "$log"
    local exit_code="${PIPESTATUS[0]}"
    set -e
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] finished $stage/$tag exit=$exit_code" | tee -a "$queue_root/queue.log"
    return "$exit_code"
}

run_model() {
    local tag="$1"
    local slot="$2"
    local model="$3"
    local family="$4"
    local revision="$5"
    local batch_size="$6"
    local disable_thinking="$7"

    if [[ ! -s "$model/config.json" ]]; then
        echo "missing model snapshot: $model" | tee -a "$queue_root/queue.log"
        return 0
    fi
    if ! run_stage smoke "$tag" "$slot" "$model" "$family" "$revision" "$batch_size" 8 "$disable_thinking"; then
        echo "smoke failed; skipping full run for $tag" | tee -a "$queue_root/queue.log"
        return 0
    fi
    if ! run_stage full "$tag" "$slot" "$model" "$family" "$revision" "$batch_size" "" "$disable_thinking"; then
        echo "full run failed or was interrupted for $tag" | tee -a "$queue_root/queue.log"
    fi
}

qwen32_revision="9216db5781bf21249d130ec9da846c4624c16137"
qwen32_model="$GEODML_PROJECT_ROOT/models/qwen/Qwen3-32B/$qwen32_revision"
ministral_revision="f6fae9795746f63c9be8344932f01275f3c63734"
ministral_model="$GEODML_PROJECT_ROOT/models/mistral/Ministral-3-8B-Instruct-2512-BF16/$ministral_revision"
gemma_revision="842da3794eaa0b77d5f08bae87a17459d91ff475"
gemma_model="$GEODML_PROJECT_ROOT/models/gemma/gemma-4-31B-it/$gemma_revision"

run_model qwen3-32b-replicate-a replicate-frontier-a "$qwen32_model" qwen "$qwen32_revision" 16 true
run_model ministral3-8b-replicate-b replicate-frontier-b "$ministral_model" mistral "$ministral_revision" 32 false
run_model gemma4-31b-primary primary-frontier "$gemma_model" gemma "$gemma_revision" 8 false

nvidia-smi > "$queue_root/gpu-environment-end.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$queue_root/completed-at.txt"
find "$queue_root" -type f ! -name artifact-sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$queue_root/artifact-sha256.txt"
echo "queue complete: $queue_root" | tee -a "$queue_root/queue.log"
