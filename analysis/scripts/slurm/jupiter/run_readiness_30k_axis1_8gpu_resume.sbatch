#!/bin/bash -l
# Resume the strict axis-one loop on two JUPITER Booster nodes.
# Approved allocation: 2026-08-24, 12 hours, 8 GPUs, 96 GPU-hours maximum.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=readiness-30k-axis1-8gpu

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?Export the exact pushed Git commit before sbatch}"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"

export READINESS_AXIS1_BASE_ROOT="${READINESS_AXIS1_BASE_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/axis1-30330-71f7a30-job1457686-planfixed}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"
export READINESS_INITIAL_CANDIDATE_ROOT="${READINESS_INITIAL_CANDIDATE_ROOT:-$READINESS_AXIS1_BASE_ROOT/round-00/generation/candidates}"
export READINESS_RECOVERY_PIPELINE_ROOT="${READINESS_RECOVERY_PIPELINE_ROOT:-$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-7ceb4cc6}"
export READINESS_30K_PIPELINE_ROOT="${READINESS_30K_PIPELINE_ROOT:-$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-${GEODML_EXPECTED_COMMIT:0:8}}"
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="${READINESS_VALIDATION_CACHE_SEARCH_ROOTS:-$READINESS_AXIS1_BASE_ROOT}"

export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"

export READINESS_APPROVED_WALLTIME="12:00:00"
export READINESS_ALLOCATION_ESTIMATE="Approved two-node JUPITER Booster strict axis-one continuation: 8 H100 GPUs for 12 hours, maximum 96 GPU-hours. Resume 29944 candidates and 2885 verified selections from job 1468481; reuse exact latest projections and validation caches; use eight generation and validation workers with 1024 refinement targets per round. Expected productive runtime 10.5-11.5 hours plus checkpoint reserve; estimated 3000-7000 additional strict prompts."
export READINESS_ALLOCATED_GPU_COUNT="8"
export READINESS_VALIDATION_SHARD_COUNT="8"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="1024"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="4"
export READINESS_GENERATION_SECONDS="10800"
export READINESS_FINALIZATION_RESERVE_SECONDS="900"
export READINESS_MAX_REFINEMENT_ROUNDS="1000"
export READINESS_MASTER_SEED="20260820"

[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 2 ]] || {
    echo "this production resume requires exactly two allocated nodes" >&2
    exit 2
}
[[ -d "$GEODML_REPOSITORY" ]]
[[ -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json" ]]
[[ -d "$READINESS_INITIAL_CANDIDATE_ROOT" ]]
[[ -d "$READINESS_RECOVERY_PIPELINE_ROOT" ]]

latest_verified_summary="$({
    find "$READINESS_RECOVERY_PIPELINE_ROOT" -mindepth 2 -maxdepth 2 \
        -path '*/round-*/verified_round_summary.json' -type f -print
} | sort -V | tail -n 1)"
[[ -s "$latest_verified_summary" ]] || {
    echo "recovery pipeline has no verified round summary" >&2
    exit 2
}
latest_verified_round="$(dirname "$latest_verified_summary")"
export READINESS_INITIAL_PROJECTION_ROOT="${READINESS_INITIAL_PROJECTION_ROOT:-$latest_verified_round/projections}"
[[ -s "$READINESS_INITIAL_PROJECTION_ROOT/qwen/projection_manifest.json" ]]
[[ -s "$READINESS_INITIAL_PROJECTION_ROOT/mistral/projection_manifest.json" ]]

mkdir -p "$READINESS_30K_PIPELINE_ROOT"
printf '%s\n' "$SLURM_JOB_ID" > "$READINESS_30K_PIPELINE_ROOT/current-job-id.txt"

echo "EIGHT-GPU STRICT AXIS-ONE RESUME"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "job=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NUM_NODES"
echo "recovery=$READINESS_RECOVERY_PIPELINE_ROOT"
echo "initial_projections=$READINESS_INITIAL_PROJECTION_ROOT"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"

exec "$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"
