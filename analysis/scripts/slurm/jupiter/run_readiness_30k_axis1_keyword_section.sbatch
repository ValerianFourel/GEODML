#!/bin/bash -l
# Run one independently resumable keyword-owned section of a frozen checkpoint.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
# Wall time is supplied by sbatch only after allocation-specific approval.
#SBATCH --job-name=readiness-30k-section

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run through sbatch on JUPITER}"
: "${GEODML_EXPECTED_COMMIT:?Export the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting this allocation}"
: "${READINESS_KEYWORD_SECTION_PLAN:?Set the immutable ten-section plan}"
: "${READINESS_TEN_SECTION_RUN_ROOT:?Set the shared ten-section run root}"
: "${READINESS_WORK_PARTITION_INDEX:?Set the independently launched section index}"

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
    echo "keyword section job requires exactly one allocated node" >&2
    exit 2
}

section_plan="$(realpath "$READINESS_KEYWORD_SECTION_PLAN")"
python3 "$GEODML_REPOSITORY/analysis/scripts/prepare_readiness_keyword_sections.py" \
    --verify-plan "$section_plan"

readarray -t section_identity < <(
    python3 - "$section_plan" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
print(plan["checkpoint_root"])
print(plan["section_count"])
print(plan["partition_salt"])
print(plan["initial_logical_round_index"])
PY
)
checkpoint="${section_identity[0]}"
export READINESS_WORK_PARTITION_COUNT="${section_identity[1]}"
export READINESS_WORK_PARTITION_SALT="${section_identity[2]}"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="${section_identity[3]}"

[[ "$READINESS_WORK_PARTITION_INDEX" =~ ^[0-9]+$ ]]
[[ "$READINESS_WORK_PARTITION_INDEX" -lt "$READINESS_WORK_PARTITION_COUNT" ]] || {
    echo "section index must be smaller than section count" >&2
    exit 2
}

required_checkpoint_files=(
    "$checkpoint/verified_round_summary.json"
    "$checkpoint/candidate-files.txt"
    "$checkpoint/validation.jsonl"
    "$checkpoint/validation.jsonl.manifest.json"
    "$checkpoint/projections/qwen/projection_manifest.json"
    "$checkpoint/projections/mistral/projection_manifest.json"
    "$checkpoint/strict-selection/run_manifest.json"
)
for required_file in "${required_checkpoint_files[@]}"; do
    [[ -s "$required_file" ]] || {
        echo "missing frozen checkpoint artifact: $required_file" >&2
        exit 2
    }
done

export READINESS_AXIS1_BASE_ROOT="${READINESS_AXIS1_BASE_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/axis1-30330-71f7a30-job1457686-planfixed}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"

section_name="section-$READINESS_WORK_PARTITION_INDEX-of-$READINESS_WORK_PARTITION_COUNT"
export READINESS_30K_PIPELINE_ROOT="$READINESS_TEN_SECTION_RUN_ROOT/$section_name"
bootstrap="$READINESS_30K_PIPELINE_ROOT/bootstrap"
empty_cache_source="$bootstrap/empty-validation-cache-source"
mkdir -p "$empty_cache_source" "$READINESS_TEN_SECTION_RUN_ROOT/coordination"

export READINESS_INITIAL_CANDIDATE_FILE_LIST="$checkpoint/candidate-files.txt"
export READINESS_INITIAL_PROJECTION_ROOT="$checkpoint/projections"
export READINESS_INITIAL_VALIDATION_OUTPUT="$checkpoint/validation.jsonl"
export READINESS_RECOVERY_PIPELINE_ROOT="$empty_cache_source"
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="$empty_cache_source"
export READINESS_VALIDATION_CACHE_ROOT="$READINESS_30K_PIPELINE_ROOT/cache"
export READINESS_30K_PIPELINE_POINTER="$READINESS_TEN_SECTION_RUN_ROOT/$section_name-latest.txt"

export READINESS_ALLOCATED_GPU_COUNT="4"
export READINESS_VALIDATION_SHARD_COUNT="4"
export READINESS_VALIDATION_BATCH_SIZE="16"
export READINESS_EMBEDDING_BATCH_SIZE="32"
export READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION="eager"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024}"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="4"
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
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

echo "FOUR-GPU KEYWORD SECTION"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "job=$SLURM_JOB_ID"
echo "approved_walltime=$READINESS_APPROVED_WALLTIME"
echo "checkpoint=$checkpoint"
echo "section=$READINESS_WORK_PARTITION_INDEX/$READINESS_WORK_PARTITION_COUNT"
echo "partition_salt=$READINESS_WORK_PARTITION_SALT"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"

set +e
"$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"
section_exit=$?
set -e
if [[ "$section_exit" -ne 0 && "$section_exit" -ne 3 ]]; then
    echo "keyword section failed with exit $section_exit" >&2
    exit "$section_exit"
fi

latest_summary="$({
    find "$READINESS_30K_PIPELINE_ROOT" -mindepth 2 -maxdepth 2 \
        -path '*/round-*/verified_round_summary.json' -type f -print
} | sort -V | tail -n 1)"
[[ -s "$latest_summary" ]]

python3 - "$READINESS_TEN_SECTION_RUN_ROOT/coordination" \
    "$READINESS_WORK_PARTITION_INDEX" "$READINESS_WORK_PARTITION_COUNT" \
    "$READINESS_30K_PIPELINE_ROOT" "$latest_summary" "$section_exit" \
    "$section_plan" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import sys

coordination = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
count = int(sys.argv[3])
pipeline = pathlib.Path(sys.argv[4]).resolve()
summary = pathlib.Path(sys.argv[5]).resolve()
section_exit = int(sys.argv[6])
plan = pathlib.Path(sys.argv[7]).resolve()
payload = {
    "format_version": "readiness-keyword-section-checkpoint-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "section_count": count,
    "section_index": index,
    "partition_salt": os.environ["READINESS_WORK_PARTITION_SALT"],
    "section_plan": str(plan),
    "section_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    "pipeline_root": str(pipeline),
    "verified_summary": str(summary),
    "section_exit_code": section_exit,
}
output = coordination / f"section-{index}-of-{count}.ready.json"
temporary = output.with_suffix(output.suffix + f".{os.environ['SLURM_JOB_ID']}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(f"section_marker={output}")
PY

echo "KEYWORD SECTION CHECKPOINTED: section=$READINESS_WORK_PARTITION_INDEX/$READINESS_WORK_PARTITION_COUNT exit=$section_exit"
