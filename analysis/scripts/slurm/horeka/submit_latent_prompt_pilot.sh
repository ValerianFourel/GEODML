#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_latent_prompt_pilot.sh [--validate-only] [--dry-run]

Required for generation:
  HOREKA_ACCOUNT          Slurm project account
  PROMPT_GENERATOR_MODEL  Cached Hugging Face model ID or local model path

Common overrides:
  HOREKA_PARTITION, HOREKA_GPUS, HOREKA_CPUS, HOREKA_TIME
  GEODML_VENV, GEODML_DATA_ROOT, HF_HOME, HOREKA_MODULES
  LATENT_PROMPT_PILOT_OUTPUT and LATENT_PROMPT_* experiment variables
EOF
}

validate_only=0
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --validate-only) validate_only=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

: "${HOREKA_ACCOUNT:?Set HOREKA_ACCOUNT to your HoreKa Slurm project account}"
if [[ $validate_only -eq 0 ]]; then
    : "${PROMPT_GENERATOR_MODEL:?Set PROMPT_GENERATOR_MODEL to a cached model ID or local path}"
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_directory/../../../.." && pwd)"
cd "$repository_root"
mkdir -p logs

if [[ $validate_only -eq 1 ]]; then
    export LATENT_PROMPT_VALIDATE_ONLY=1
    : "${HOREKA_PARTITION:=dev_accelerated}"
    : "${HOREKA_GPUS:=1}"
    : "${HOREKA_TIME:=00:15:00}"
else
    export LATENT_PROMPT_VALIDATE_ONLY=0
    : "${HOREKA_PARTITION:=accelerated}"
    : "${HOREKA_GPUS:=2}"
    : "${HOREKA_TIME:=02:00:00}"
fi
: "${HOREKA_CPUS:=16}"

export HOREKA_PARTITION HOREKA_GPUS HOREKA_TIME HOREKA_CPUS
export GEODML_VENV="${GEODML_VENV:-$repository_root/.venv311}"
export GEODML_DATA_ROOT="${GEODML_DATA_ROOT:-$repository_root/geodml_data}"

command=(
    sbatch
    --account="$HOREKA_ACCOUNT"
    --partition="$HOREKA_PARTITION"
    --gres="gpu:$HOREKA_GPUS"
    --cpus-per-task="$HOREKA_CPUS"
    --time="$HOREKA_TIME"
    --export=ALL
    analysis/scripts/slurm/horeka/run_latent_prompt_pilot.sbatch
)

printf '[submit]'
printf ' %q' "${command[@]}"
printf '\n'
if [[ $dry_run -eq 0 ]]; then
    "${command[@]}"
fi
