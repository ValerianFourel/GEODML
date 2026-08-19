#!/bin/bash -l
# Run three restart-safe readiness embedding views inside an existing allocation.
# This script never requests Slurm resources.

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"
: "${GEODML_REPOSITORY:?GEODML_REPOSITORY is required}"
: "${GEODML_MODEL_VENV:?GEODML_MODEL_VENV is required}"
: "${READINESS_HF_BUNDLE_ROOT:?READINESS_HF_BUNDLE_ROOT is required}"
: "${READINESS_HF_EMBEDDING_ROOT:?READINESS_HF_EMBEDDING_ROOT is required}"
: "${QWEN3_8B_SNAPSHOT:?QWEN3_8B_SNAPSHOT is required}"
: "${QWEN3_8B_REVISION:?QWEN3_8B_REVISION is required}"
: "${LLM2VEC_MNTP_SNAPSHOT:?LLM2VEC_MNTP_SNAPSHOT is required}"
: "${LLM2VEC_MNTP_REPO:?LLM2VEC_MNTP_REPO is required}"
: "${LLM2VEC_MNTP_REVISION:?LLM2VEC_MNTP_REVISION is required}"
: "${LLM2VEC_UNSUP_SIMCSE_SNAPSHOT:?LLM2VEC_UNSUP_SIMCSE_SNAPSHOT is required}"
: "${LLM2VEC_UNSUP_SIMCSE_REPO:?LLM2VEC_UNSUP_SIMCSE_REPO is required}"
: "${LLM2VEC_UNSUP_SIMCSE_REVISION:?LLM2VEC_UNSUP_SIMCSE_REVISION is required}"
: "${LLM2VEC_SUPERVISED_SNAPSHOT:?LLM2VEC_SUPERVISED_SNAPSHOT is required}"
: "${LLM2VEC_SUPERVISED_REPO:?LLM2VEC_SUPERVISED_REPO is required}"
: "${LLM2VEC_SUPERVISED_REVISION:?LLM2VEC_SUPERVISED_REVISION is required}"
: "${LLM2VEC_GEN_SNAPSHOT:?LLM2VEC_GEN_SNAPSHOT is required}"
: "${LLM2VEC_GEN_REPO:?LLM2VEC_GEN_REPO is required}"
: "${LLM2VEC_GEN_REVISION:?LLM2VEC_GEN_REVISION is required}"

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "embedding export requires a clean checkout" >&2
    exit 2
fi

prompts="$READINESS_HF_BUNDLE_ROOT/restricted-local/prompts.jsonl"
test -s "$prompts"
for path in \
    "$QWEN3_8B_SNAPSHOT" \
    "$LLM2VEC_MNTP_SNAPSHOT" \
    "$LLM2VEC_UNSUP_SIMCSE_SNAPSHOT" \
    "$LLM2VEC_SUPERVISED_SNAPSHOT" \
    "$LLM2VEC_GEN_SNAPSHOT"; do
    if [[ ! -d "$path" ]]; then
        echo "missing frozen model snapshot: $path" >&2
        exit 2
    fi
done

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

module_pythonpath="${PYTHONPATH-}"
source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages:$module_pythonpath"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

visible_gpus="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpus" != 1 ]]; then
    echo "embedding views require exactly one visible GPU; found $visible_gpus" >&2
    exit 2
fi

mkdir -p "$READINESS_HF_EMBEDDING_ROOT"
view_list="${READINESS_HF_EMBEDDING_VIEWS:-qwen3-8b-mntp-unsup-simcse,qwen3-8b-mntp-supervised,qwen3-8b-llm2vec-gen}"
batch_size="${READINESS_HF_EMBEDDING_BATCH_SIZE:-8}"
shard_size="${READINESS_HF_EMBEDDING_SHARD_SIZE:-512}"
max_length="${READINESS_HF_EMBEDDING_MAX_LENGTH:-512}"

run_llm2vec() {
    local view_name="$1" peft_snapshot="$2" peft_revision="$3" peft_model_id="$4"
    python analysis/scripts/build_readiness_hf_dataset.py embed \
        --prompts "$prompts" \
        --output-dir "$READINESS_HF_EMBEDDING_ROOT/$view_name" \
        --view-name "$view_name" \
        --backend llm2vec \
        --embedding-model "$QWEN3_8B_SNAPSHOT" \
        --embedding-model-id "Qwen/Qwen3-8B" \
        --embedding-model-revision "$QWEN3_8B_REVISION" \
        --mntp-model "$LLM2VEC_MNTP_SNAPSHOT" \
        --mntp-model-id "$LLM2VEC_MNTP_REPO" \
        --mntp-model-revision "$LLM2VEC_MNTP_REVISION" \
        --peft-model "$peft_snapshot" \
        --peft-model-id "$peft_model_id" \
        --peft-model-revision "$peft_revision" \
        --batch-size "$batch_size" \
        --max-length "$max_length" \
        --shard-size "$shard_size" \
        --git-commit-sha "$GEODML_EXPECTED_COMMIT"
}

IFS=',' read -r -a views <<< "$view_list"
for view in "${views[@]}"; do
    case "$view" in
        qwen3-8b-mntp-unsup-simcse)
            run_llm2vec \
                "$view" \
                "$LLM2VEC_UNSUP_SIMCSE_SNAPSHOT" \
                "$LLM2VEC_UNSUP_SIMCSE_REVISION" \
                "$LLM2VEC_UNSUP_SIMCSE_REPO"
            ;;
        qwen3-8b-mntp-supervised)
            run_llm2vec \
                "$view" \
                "$LLM2VEC_SUPERVISED_SNAPSHOT" \
                "$LLM2VEC_SUPERVISED_REVISION" \
                "$LLM2VEC_SUPERVISED_REPO"
            ;;
        qwen3-8b-llm2vec-gen)
            python analysis/scripts/build_readiness_hf_dataset.py embed \
                --prompts "$prompts" \
                --output-dir "$READINESS_HF_EMBEDDING_ROOT/$view" \
                --view-name "$view" \
                --backend llm2vec-gen \
                --embedding-model "$LLM2VEC_GEN_SNAPSHOT" \
                --embedding-model-id "$LLM2VEC_GEN_REPO" \
                --embedding-model-revision "$LLM2VEC_GEN_REVISION" \
                --batch-size "$batch_size" \
                --max-length "$max_length" \
                --shard-size "$shard_size" \
                --git-commit-sha "$GEODML_EXPECTED_COMMIT"
            ;;
        *)
            echo "unknown READINESS_HF_EMBEDDING_VIEWS entry: $view" >&2
            exit 2
            ;;
    esac
done

echo "READINESS EMBEDDING VIEWS: PASS"
