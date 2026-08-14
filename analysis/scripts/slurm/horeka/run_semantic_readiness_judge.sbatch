#!/bin/bash -l
# Resume one semantic-readiness judge slot on one HoreKa GPU.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=readiness-judge

set -euo pipefail

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:?Submit from the GEODML repository root}"
cd "$REPOSITORY_ROOT"

: "${JUDGE_SLOT:?Set JUDGE_SLOT}"
: "${JUDGE_TAG:?Set JUDGE_TAG}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to a local model snapshot}"
: "${READINESS_RUN:?Set READINESS_RUN}"
: "${READINESS_JUDGE_TASKS:?Set READINESS_JUDGE_TASKS}"

GEODML_VENV="${GEODML_VENV:-$REPOSITORY_ROOT/.venv311}"
if [[ ! -f "$GEODML_VENV/bin/activate" ]]; then
    echo "Missing virtual environment: $GEODML_VENV" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$GEODML_VENV/bin/activate"

export HF_HOME="${HF_HOME:-$REPOSITORY_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GEODML_REQUIRED_GPU_COUNT=1
export GEODML_DEVICE_MAP=auto

if [[ -n "${GEODML_EXPECTED_COMMIT:-}" ]]; then
    actual_commit="$(git rev-parse HEAD)"
    if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
        echo "Commit mismatch: expected $GEODML_EXPECTED_COMMIT, found $actual_commit" >&2
        exit 2
    fi
fi
if [[ ! -f "$READINESS_JUDGE_TASKS" ]]; then
    echo "Missing judge tasks: $READINESS_JUDGE_TASKS" >&2
    exit 2
fi
if [[ ! -f "$JUDGE_MODEL/config.json" ]]; then
    echo "Missing model snapshot: $JUDGE_MODEL" >&2
    exit 2
fi

output_directory="$READINESS_RUN/judges/$JUDGE_TAG"
mkdir -p "$output_directory"

echo "[readiness] job=$SLURM_JOB_ID"
echo "[readiness] commit=$(git rev-parse HEAD)"
echo "[readiness] slot=$JUDGE_SLOT"
echo "[readiness] model=$JUDGE_MODEL"
echo "[readiness] output=$output_directory"
nvidia-smi -L

srun --ntasks=1 python3 analysis/scripts/run_semantic_readiness_judge.py \
    --tasks "$READINESS_JUDGE_TASKS" \
    --judge-slot "$JUDGE_SLOT" \
    --backend local \
    --model "$JUDGE_MODEL" \
    --precision full \
    --max-new-tokens 300 \
    --maximum-attempts 5 \
    --resume \
    --output-dir "$output_directory"
