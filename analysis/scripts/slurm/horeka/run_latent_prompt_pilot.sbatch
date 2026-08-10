#!/bin/bash -l
# HoreKa GPU job for the query-conditioned informational-to-transactional
# latent-prompt pilot. Account, partition and GPU count are supplied by the
# submission helper rather than hard-coded here.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=geodml-latent-prompts
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the GEODML repository root}"
cd "$REPOSITORY_ROOT"

if [[ -n "${HOREKA_MODULES:-}" ]]; then
    module purge
    read -r -a requested_modules <<< "$HOREKA_MODULES"
    module load "${requested_modules[@]}"
fi

if [[ -z "${GEODML_VENV:-}" ]]; then
    if [[ -f "$REPOSITORY_ROOT/.venv311/bin/activate" ]]; then
        GEODML_VENV="$REPOSITORY_ROOT/.venv311"
    else
        GEODML_VENV="$REPOSITORY_ROOT/.venv"
    fi
fi
if [[ ! -f "$GEODML_VENV/bin/activate" ]]; then
    echo "Missing virtual environment: $GEODML_VENV" >&2
    exit 2
fi
export GEODML_VENV
# shellcheck disable=SC1090
source "$GEODML_VENV/bin/activate"

export GEODML_DATA_ROOT="${GEODML_DATA_ROOT:-$REPOSITORY_ROOT/geodml_data}"
export HF_HOME="${HF_HOME:-$REPOSITORY_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
export PROMPT_GENERATOR_PRECISION="${PROMPT_GENERATOR_PRECISION:-4bit}"
export LATENT_PROMPT_ENGINE="${LATENT_PROMPT_ENGINE:-searxng}"
export LATENT_PROMPT_POOL="${LATENT_PROMPT_POOL:-20}"
export LATENT_PROMPT_TOP_N="${LATENT_PROMPT_TOP_N:-10}"
export LATENT_PROMPT_MAX_KEYWORDS="${LATENT_PROMPT_MAX_KEYWORDS:-8}"
export LATENT_PROMPT_TARGET_GRID="${LATENT_PROMPT_TARGET_GRID:-0,0.25,0.5,0.75,1}"
export LATENT_PROMPT_STYLE_SEEDS="${LATENT_PROMPT_STYLE_SEEDS:-2}"
export LATENT_PROMPT_FIRST_STYLE_SEED="${LATENT_PROMPT_FIRST_STYLE_SEED:-0}"
export LATENT_PROMPT_CANDIDATES="${LATENT_PROMPT_CANDIDATES:-24}"
export LATENT_PROMPT_MASTER_SEED="${LATENT_PROMPT_MASTER_SEED:-20260810}"
export LATENT_PROMPT_PILOT_OUTPUT="${LATENT_PROMPT_PILOT_OUTPUT:-$REPOSITORY_ROOT/runs/latent_prompt_pilot/${SLURM_JOB_ID}}"
export LATENT_PROMPT_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$HF_HOME" "$LATENT_PROMPT_PILOT_OUTPUT"
RUN_MANIFEST="$LATENT_PROMPT_PILOT_OUTPUT/run_manifest.json"

record_status() {
    local status="$1"
    local exit_code="${2:-}"
    local arguments=(
        analysis/scripts/slurm/horeka/record_latent_run_manifest.py
        --output "$RUN_MANIFEST"
        --status "$status"
    )
    if [[ -n "$exit_code" ]]; then
        arguments+=(--exit-code "$exit_code")
    fi
    python3 "${arguments[@]}"
}

on_exit() {
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        record_status complete 0 || true
    else
        record_status failed "$exit_code" || true
    fi
}
trap on_exit EXIT
record_status started

echo "[horeka] job_id=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION:-unknown} account=${SLURM_JOB_ACCOUNT:-unknown}"
echo "[horeka] commit=$(git rev-parse HEAD)"
echo "[horeka] data_root=$GEODML_DATA_ROOT"
echo "[horeka] output=$LATENT_PROMPT_PILOT_OUTPUT"
echo "[horeka] venv=$GEODML_VENV"

SERP_TABLE="$GEODML_DATA_ROOT/data/serp/phase0_top${LATENT_PROMPT_POOL}_${LATENT_PROMPT_ENGINE}.parquet"
if [[ ! -f "$SERP_TABLE" ]]; then
    echo "Missing cached SERP table: $SERP_TABLE" >&2
    exit 2
fi

if [[ "${LATENT_PROMPT_VALIDATE_ONLY:-0}" == "1" ]]; then
    echo "[horeka] validation only: no model weights will be loaded"
    python3 --version
    python3 -c 'import numpy, pandas, torch, transformers, sentence_transformers; print("Python dependencies: OK")'
    python3 analysis/scripts/generate_latent_prompt_pilot.py --help >/dev/null
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    fi
    echo "[horeka] validation complete"
    exit 0
fi

: "${PROMPT_GENERATOR_MODEL:?Set PROMPT_GENERATOR_MODEL to a cached Hugging Face model ID or local model path}"

arguments=(
    analysis/scripts/generate_latent_prompt_pilot.py
    --provider local
    --generator-model "$PROMPT_GENERATOR_MODEL"
    --embedding-model "$EMBEDDING_MODEL"
    --embedding-device "$EMBEDDING_DEVICE"
    --precision "$PROMPT_GENERATOR_PRECISION"
    --data-root "$GEODML_DATA_ROOT"
    --engine "$LATENT_PROMPT_ENGINE"
    --pool "$LATENT_PROMPT_POOL"
    --top-n "$LATENT_PROMPT_TOP_N"
    --max-keywords "$LATENT_PROMPT_MAX_KEYWORDS"
    --target-grid "$LATENT_PROMPT_TARGET_GRID"
    --number-style-seeds "$LATENT_PROMPT_STYLE_SEEDS"
    --first-style-seed "$LATENT_PROMPT_FIRST_STYLE_SEED"
    --number-candidates "$LATENT_PROMPT_CANDIDATES"
    --master-seed "$LATENT_PROMPT_MASTER_SEED"
    --output-dir "$LATENT_PROMPT_PILOT_OUTPUT"
)
if [[ "${LATENT_PROMPT_OVERWRITE:-0}" == "1" ]]; then
    arguments+=(--overwrite)
fi

echo "[horeka] starting prompt generation (no reranking model is invoked)"
srun --ntasks=1 python3 "${arguments[@]}"
echo "[horeka] prompt-generation pilot complete"
