#!/bin/bash -l
# One-stop, resumable strict axis-one construction loop inside a GPU allocation.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting this allocation}"
: "${READINESS_AXIS1_BASE_ROOT:?Set the persistent axis-one experiment root}"
: "${READINESS_30K_PLAN_ROOT:?Set the frozen 30,330-target uniform plan}"
: "${READINESS_RECOVERY_PIPELINE_ROOT:?Set the pipeline containing partial refinement candidates}"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"
export READINESS_INITIAL_CANDIDATE_ROOT="${READINESS_INITIAL_CANDIDATE_ROOT:-$READINESS_AXIS1_BASE_ROOT/round-00/generation/candidates}"
export READINESS_30K_PIPELINE_ROOT="${READINESS_30K_PIPELINE_ROOT:-$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-${GEODML_EXPECTED_COMMIT:0:8}}"

# Discover and safely union all earlier judgments under the experiment root.
# The merger verifies immutable cache identities and fails on conflicting
# scientific reviews instead of silently overwriting one result with another.
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="${READINESS_VALIDATION_CACHE_SEARCH_ROOTS:-$READINESS_AXIS1_BASE_ROOT}"

export READINESS_ALLOCATED_GPU_COUNT="${READINESS_ALLOCATED_GPU_COUNT:-4}"
export READINESS_VALIDATION_SHARD_COUNT="${READINESS_VALIDATION_SHARD_COUNT:-$READINESS_ALLOCATED_GPU_COUNT}"
export READINESS_DISTANCE_TOLERANCE="0.017"
export READINESS_DISAGREEMENT_WEIGHT="0.10"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="${READINESS_REFINEMENT_CANDIDATES_PER_TASK:-4}"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024}"
export READINESS_MAX_REFINEMENT_ROUNDS="${READINESS_MAX_REFINEMENT_ROUNDS:-1000}"
export READINESS_MASTER_SEED="${READINESS_MASTER_SEED:-20260820}"
export READINESS_FINALIZATION_RESERVE_SECONDS="${READINESS_FINALIZATION_RESERVE_SECONDS:-900}"

[[ -d "$READINESS_AXIS1_BASE_ROOT" ]]
[[ -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json" ]]
[[ -d "$READINESS_INITIAL_CANDIDATE_ROOT" ]]
[[ -d "$READINESS_RECOVERY_PIPELINE_ROOT" ]]
[[ -d "$GEODML_REPOSITORY" ]]

mkdir -p "$READINESS_30K_PIPELINE_ROOT"
printf '%s\n' "$SLURM_JOB_ID" > "$READINESS_30K_PIPELINE_ROOT/current-job-id.txt"

controller_log="$READINESS_30K_PIPELINE_ROOT/controller-job-$SLURM_JOB_ID.log"
echo "STRICT AXIS-ONE LOOP"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "job=$SLURM_JOB_ID"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"
echo "plan=$READINESS_30K_PLAN_ROOT"
echo "cache_search=$READINESS_VALIDATION_CACHE_SEARCH_ROOTS"

set +e
"$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_end_to_end.sh" \
    2>&1 | tee -a "$controller_log"
pipeline_exit="${PIPESTATUS[0]}"
set -e

printf '%s\n' "$pipeline_exit" \
    > "$READINESS_30K_PIPELINE_ROOT/controller-exit-code.txt"
echo "pipeline_exit=$pipeline_exit"
echo "pipeline_root=$READINESS_30K_PIPELINE_ROOT"
exit "$pipeline_exit"
