#!/bin/bash -l
# Continue the fused strict axis-one checkpoint on one four-GPU Booster node.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
# Wall time is intentionally omitted.  Supply sbatch --time only after the
# allocation-specific estimate and approval have been recorded.
#SBATCH --job-name=readiness-30k-global

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run through sbatch on JUPITER}"
: "${GEODML_EXPECTED_COMMIT:?Export the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting this allocation}"
: "${READINESS_GLOBAL_CHECKPOINT_ROOT:?Set the fused verified final checkpoint}"
: "${READINESS_30K_PIPELINE_ROOT:?Set a fresh FSCRATCH pipeline root}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load git
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"

[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]
[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ]] || {
    echo "global continuation requires exactly one allocated node" >&2
    exit 2
}

export READINESS_AXIS1_BASE_ROOT="${READINESS_AXIS1_BASE_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/axis1-30330-71f7a30-job1457686-planfixed}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"

checkpoint="$(realpath "$READINESS_GLOBAL_CHECKPOINT_ROOT")"
checkpoint_candidates="$checkpoint/merged/candidates.jsonl"
checkpoint_projections="$checkpoint/merged/projections"
checkpoint_validation="$checkpoint/merged/validation.jsonl"
checkpoint_manifest="$checkpoint/merged/merge_manifest.json"

required_files=(
    "$checkpoint/verified_round_summary.json"
    "$checkpoint/strict-selection/run_manifest.json"
    "$checkpoint_candidates"
    "$checkpoint_candidates.manifest.json"
    "$checkpoint_projections/qwen/projection_manifest.json"
    "$checkpoint_projections/mistral/projection_manifest.json"
    "$checkpoint_validation"
    "$checkpoint_validation.manifest.json"
    "$checkpoint_manifest"
    "$READINESS_30K_PLAN_ROOT/plan_manifest.json"
)
for required_file in "${required_files[@]}"; do
    [[ -s "$required_file" ]] || {
        echo "missing global checkpoint artifact: $required_file" >&2
        exit 2
    }
done

bootstrap="$READINESS_30K_PIPELINE_ROOT/bootstrap"
empty_cache_source="$bootstrap/empty-validation-cache-source"
candidate_list="$bootstrap/initial-candidate-files.txt"
mkdir -p "$empty_cache_source"
printf '%s\n' "$checkpoint_candidates" > "$candidate_list"

export READINESS_INITIAL_CANDIDATE_FILE_LIST="$candidate_list"
export READINESS_INITIAL_PROJECTION_ROOT="$checkpoint_projections"
export READINESS_INITIAL_VALIDATION_OUTPUT="$checkpoint_validation"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["maximum_candidate_round_index"])' \
        "$checkpoint_manifest"
)"
export READINESS_RECOVERY_PIPELINE_ROOT="$empty_cache_source"
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="$empty_cache_source"
export READINESS_VALIDATION_CACHE_ROOT="$READINESS_30K_PIPELINE_ROOT/cache"
export READINESS_30K_PIPELINE_POINTER="$(dirname "$READINESS_30K_PIPELINE_ROOT")/latest-global-pipeline.txt"

export READINESS_WORK_PARTITION_COUNT="1"
export READINESS_WORK_PARTITION_INDEX="0"
export READINESS_WORK_PARTITION_SALT="axis1-30330-global-v1"
export READINESS_ALLOCATED_GPU_COUNT="4"
export READINESS_VALIDATION_SHARD_COUNT="4"
export READINESS_VALIDATION_BATCH_SIZE="16"
export READINESS_EMBEDDING_BATCH_SIZE="32"
export READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION="eager"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024}"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="4"
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-10800}"
export READINESS_FINALIZATION_RESERVE_SECONDS="${READINESS_FINALIZATION_RESERVE_SECONDS:-1800}"
export READINESS_MAX_REFINEMENT_ROUNDS="${READINESS_MAX_REFINEMENT_ROUNDS:-1000}"
export READINESS_MASTER_SEED="20260820"
export READINESS_DISTANCE_TOLERANCE="0.017"
export READINESS_DISAGREEMENT_WEIGHT="0.10"
export READINESS_GENERATOR_A_ID="qwen3-32b"
export READINESS_GENERATOR_B_ID="gemma4-31b"

export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
[[ -s "$subspace_pointer" ]]
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$subspace_pointer")}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"
[[ -s "$READINESS_BATTERY_ROOT/battery_manifest.json" ]]

printf '%s\n' "$READINESS_30K_PIPELINE_ROOT" > "$HOME/geodml-readiness-global-latest.txt.tmp"
mv "$HOME/geodml-readiness-global-latest.txt.tmp" "$HOME/geodml-readiness-global-latest.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$HOME/geodml-readiness-global-job-id.txt.tmp"
mv "$HOME/geodml-readiness-global-job-id.txt.tmp" "$HOME/geodml-readiness-global-job-id.txt"

echo "FOUR-GPU GLOBAL STRICT AXIS-ONE CONTINUATION"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "job=$SLURM_JOB_ID"
echo "approved_walltime=$READINESS_APPROVED_WALLTIME"
echo "checkpoint=$checkpoint"
echo "initial_logical_round=$READINESS_INITIAL_LOGICAL_ROUND_INDEX"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"
echo "live_audit_history=$READINESS_30K_PIPELINE_ROOT/live-fully-compliant-history.jsonl"

exec "$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"
