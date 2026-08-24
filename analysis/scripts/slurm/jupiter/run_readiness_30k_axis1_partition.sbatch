#!/bin/bash -l
# One of two collision-free strict axis-one producer/finalizer jobs.
# Retry approved per partition: 2026-08-24, 12 hours, 4 GPUs, 48 GPU-hours.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=readiness-30k-axis1-part

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?Export the exact pushed Git commit before sbatch}"
: "${READINESS_WORK_PARTITION_INDEX:?Submit once with index 0 and once with index 1}"
export READINESS_WORK_PARTITION_COUNT="2"
export READINESS_WORK_PARTITION_SALT="axis1-30330-two-way-v1"
[[ "$READINESS_WORK_PARTITION_INDEX" == "0" || "$READINESS_WORK_PARTITION_INDEX" == "1" ]] || {
    echo "READINESS_WORK_PARTITION_INDEX must be 0 or 1" >&2
    exit 2
}

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"

export READINESS_AXIS1_BASE_ROOT="${READINESS_AXIS1_BASE_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/axis1-30330-71f7a30-job1457686-planfixed}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"
export READINESS_INITIAL_CANDIDATE_ROOT="${READINESS_INITIAL_CANDIDATE_ROOT:-$READINESS_AXIS1_BASE_ROOT/round-00/generation/candidates}"
export READINESS_RECOVERY_PIPELINE_ROOT="${READINESS_RECOVERY_PIPELINE_ROOT:-$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-7ceb4cc6}"

partition_prefix="$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-${GEODML_EXPECTED_COMMIT:0:8}"
export READINESS_PARTITION_ROOT_0="${READINESS_PARTITION_ROOT_0:-$partition_prefix-partition-0-of-2}"
export READINESS_PARTITION_ROOT_1="${READINESS_PARTITION_ROOT_1:-$partition_prefix-partition-1-of-2}"
if [[ "$READINESS_WORK_PARTITION_INDEX" == "0" ]]; then
    export READINESS_30K_PIPELINE_ROOT="$READINESS_PARTITION_ROOT_0"
else
    export READINESS_30K_PIPELINE_ROOT="$READINESS_PARTITION_ROOT_1"
fi
export READINESS_PARTITION_COORDINATION_ROOT="${READINESS_PARTITION_COORDINATION_ROOT:-$partition_prefix-partitioned}"
export READINESS_30K_PIPELINE_POINTER="$READINESS_PARTITION_COORDINATION_ROOT/partition-$READINESS_WORK_PARTITION_INDEX-latest.txt"

# Read only the earlier shared cache tree.  Each live partition writes its own
# cache, preventing concurrent cache discovery or mutation across producers.
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="$READINESS_RECOVERY_PIPELINE_ROOT"

export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"

export READINESS_APPROVED_WALLTIME="12:00:00"
export READINESS_ALLOCATION_ESTIMATE="Freshly approved 2026-08-24 retry of partition $READINESS_WORK_PARTITION_INDEX after jobs 1471643 and 1471645 failed during checkpoint preflight before GPU generation: one JUPITER Booster node, 4 H100 GPUs for 12 hours, maximum 48 GPU-hours for this job and 96 GPU-hours across both jobs. The retry starts from the exact 29944-candidate verified checkpoint; stable target ownership prevents duplicate refinement tasks, and a global final strict selection combines both checkpoints. Estimated productive generation and per-partition verification: 10.5-12 hours including a 90-minute finalization reserve."
export READINESS_ALLOCATED_GPU_COUNT="4"
export READINESS_VALIDATION_SHARD_COUNT="4"
export READINESS_VALIDATION_BATCH_SIZE="16"
export READINESS_EMBEDDING_BATCH_SIZE="32"
export READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION="eager"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="1024"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="4"
export READINESS_GENERATION_SECONDS="10800"
export READINESS_FINALIZATION_RESERVE_SECONDS="5400"
export READINESS_MAX_REFINEMENT_ROUNDS="1000"
export READINESS_MASTER_SEED="20260820"
export READINESS_DISTANCE_TOLERANCE="0.017"
export READINESS_DISAGREEMENT_WEIGHT="0.10"
export READINESS_GENERATOR_A_ID="qwen3-32b"
export READINESS_GENERATOR_B_ID="gemma4-31b"

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
[[ -s "$subspace_pointer" ]]
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$subspace_pointer")}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"

[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ]] || {
    echo "each partition job requires exactly one allocated node" >&2
    exit 2
}
[[ -d "$GEODML_REPOSITORY" ]]
[[ -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json" ]]
[[ -d "$READINESS_INITIAL_CANDIDATE_ROOT" ]]
[[ -d "$READINESS_RECOVERY_PIPELINE_ROOT" ]]
[[ -s "$READINESS_BATTERY_ROOT/battery_manifest.json" ]]

latest_verified_summary="$({
    find "$READINESS_RECOVERY_PIPELINE_ROOT" -mindepth 2 -maxdepth 2 \
        -path '*/round-*/verified_round_summary.json' -type f -print
} | sort -V | tail -n 1)"
[[ -s "$latest_verified_summary" ]] || {
    echo "recovery pipeline has no verified round summary" >&2
    exit 2
}
latest_verified_round="$(dirname "$latest_verified_summary")"
export READINESS_INITIAL_CANDIDATE_FILE_LIST="${READINESS_INITIAL_CANDIDATE_FILE_LIST:-$latest_verified_round/candidate-files.txt}"
latest_verified_round_name="$(basename "$latest_verified_round")"
latest_verified_round_number="${latest_verified_round_name#round-}"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="$((10#$latest_verified_round_number))"
export READINESS_INITIAL_PROJECTION_ROOT="${READINESS_INITIAL_PROJECTION_ROOT:-$latest_verified_round/projections}"
export READINESS_INITIAL_VALIDATION_OUTPUT="${READINESS_INITIAL_VALIDATION_OUTPUT:-$latest_verified_round/validation.jsonl}"
[[ -s "$READINESS_INITIAL_CANDIDATE_FILE_LIST" ]]
[[ -s "$READINESS_INITIAL_PROJECTION_ROOT/qwen/projection_manifest.json" ]]
[[ -s "$READINESS_INITIAL_PROJECTION_ROOT/mistral/projection_manifest.json" ]]
[[ -s "$READINESS_INITIAL_VALIDATION_OUTPUT" ]]
[[ -s "$READINESS_INITIAL_VALIDATION_OUTPUT.manifest.json" ]]

mkdir -p "$READINESS_30K_PIPELINE_ROOT" "$READINESS_PARTITION_COORDINATION_ROOT"
own_marker="$READINESS_PARTITION_COORDINATION_ROOT/producer-$READINESS_WORK_PARTITION_INDEX.ready.json"
if [[ -e "$own_marker" ]]; then
    mv "$own_marker" "$own_marker.superseded-by-job-$SLURM_JOB_ID"
fi
printf '%s\n' "$SLURM_JOB_ID" > "$READINESS_30K_PIPELINE_ROOT/current-job-id.txt"

echo "FOUR-GPU STRICT AXIS-ONE PARTITION"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "job=$SLURM_JOB_ID"
echo "partition=$READINESS_WORK_PARTITION_INDEX/$READINESS_WORK_PARTITION_COUNT"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"
echo "coordination=$READINESS_PARTITION_COORDINATION_ROOT"

set +e
"$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"
producer_exit=$?
set -e
if [[ "$producer_exit" -ne 0 && "$producer_exit" -ne 3 ]]; then
    echo "partition producer failed with exit $producer_exit" >&2
    exit "$producer_exit"
fi

partition_summary="$({
    find "$READINESS_30K_PIPELINE_ROOT" -mindepth 2 -maxdepth 2 \
        -path '*/round-*/verified_round_summary.json' -type f -print
} | sort -V | tail -n 1)"
[[ -s "$partition_summary" ]] || {
    echo "partition produced no verified checkpoint" >&2
    exit 2
}

python - "$READINESS_PARTITION_COORDINATION_ROOT" "$READINESS_WORK_PARTITION_INDEX" \
    "$READINESS_30K_PIPELINE_ROOT" "$partition_summary" "$producer_exit" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys

coordination = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
pipeline = pathlib.Path(sys.argv[3]).resolve()
summary = pathlib.Path(sys.argv[4]).resolve()
producer_exit = int(sys.argv[5])
payload = {
    "format_version": "readiness-two-partition-producer-ready-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "partition_count": 2,
    "partition_index": index,
    "partition_salt": os.environ["READINESS_WORK_PARTITION_SALT"],
    "pipeline_root": str(pipeline),
    "verified_summary": str(summary),
    "producer_exit_code": producer_exit,
}
output = coordination / f"producer-{index}.ready.json"
temporary = output.with_suffix(output.suffix + f".{os.environ['SLURM_JOB_ID']}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(f"producer_marker={output}")
PY

"$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/finalize_readiness_30k_partitions.sh"
echo "PARTITION JOB COMPLETE: producer_exit=$producer_exit"
