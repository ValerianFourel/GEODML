#!/bin/bash -l
# Recover the two interrupted round-19 partitions sequentially, then finalize.
# Freshly approved 2026-08-24 after job 1484067 failed during repository
# preflight: one JUPITER Booster node, four GPUs, one hour, maximum four
# GPU-hours. This job must not start a new generation round.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=readiness-30k-recover

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run through sbatch on JUPITER}"
: "${GEODML_RECOVERY_ORCHESTRATOR_COMMIT:?Export the pushed recovery commit}"
: "${GEODML_RECOVERY_ORCHESTRATOR_REPOSITORY:?Export the exact recovery checkout}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load git
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

orchestrator_repository="$(realpath "$GEODML_RECOVERY_ORCHESTRATOR_REPOSITORY")"
[[ -d "$orchestrator_repository" ]] || {
    echo "recovery orchestrator repository is missing: $orchestrator_repository" >&2
    exit 2
}
actual_orchestrator_commit="$(git -C "$orchestrator_repository" rev-parse HEAD)"
[[ "$actual_orchestrator_commit" == "$GEODML_RECOVERY_ORCHESTRATOR_COMMIT" ]] || {
    echo "recovery orchestrator commit mismatch: expected=$GEODML_RECOVERY_ORCHESTRATOR_COMMIT actual=$actual_orchestrator_commit" >&2
    exit 2
}
[[ -z "$(git -C "$orchestrator_repository" status --porcelain)" ]] || {
    echo "recovery orchestration requires a clean exact-commit checkout" >&2
    exit 2
}

science_commit="f77b16f453a9421218d44a4d2e896cb7eb5fb589"
export GEODML_EXPECTED_COMMIT="$science_commit"
export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$science_commit}"

actual_science_commit="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
[[ "$actual_science_commit" == "$science_commit" ]] || {
    echo "scientific pipeline commit mismatch: expected=$science_commit actual=$actual_science_commit" >&2
    exit 2
}
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]] || {
    echo "scientific pipeline checkout must remain clean" >&2
    exit 2
}

export READINESS_AXIS1_BASE_ROOT="${READINESS_AXIS1_BASE_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/axis1-30330-71f7a30-job1457686-planfixed}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"
export READINESS_INITIAL_CANDIDATE_ROOT="${READINESS_INITIAL_CANDIDATE_ROOT:-$READINESS_AXIS1_BASE_ROOT/round-00/generation/candidates}"
export READINESS_RECOVERY_PIPELINE_ROOT="${READINESS_RECOVERY_PIPELINE_ROOT:-$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-7ceb4cc6}"

partition_prefix="$READINESS_AXIS1_BASE_ROOT/axis1-strict-loop-${science_commit:0:8}"
export READINESS_PARTITION_ROOT_0="${READINESS_PARTITION_ROOT_0:-$partition_prefix-partition-0-of-2}"
export READINESS_PARTITION_ROOT_1="${READINESS_PARTITION_ROOT_1:-$partition_prefix-partition-1-of-2}"
export READINESS_PARTITION_COORDINATION_ROOT="${READINESS_PARTITION_COORDINATION_ROOT:-$partition_prefix-partitioned}"

export READINESS_WORK_PARTITION_COUNT="2"
export READINESS_WORK_PARTITION_SALT="axis1-30330-two-way-v1"
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="$READINESS_RECOVERY_PIPELINE_ROOT"

export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"

export READINESS_APPROVED_WALLTIME="01:00:00"
export READINESS_ALLOCATION_ESTIMATE="Approved 2026-08-24 after batch job 1484067 failed during repository preflight before scientific work because Slurm executes a copied spool script: one JUPITER Booster node, four H100 GPUs, 32 CPUs, one hour, maximum four GPU-hours. Partition 0 remains complete. Partition 1 has 106,996 round-19 candidates; shards 0, 1, and 3 preserve 80,523 validated rows with completion manifests. Shard 2 is a nonempty 18 MB zero-record JSONL without a completion manifest; per-candidate reviews are written durably before aggregate checkpoints. Estimated cache-backed shard reconstruction, strict selection, two-partition reconciliation, diagnostics, and export readiness: 10-20 minutes, with 40-50 minutes of safety margin. Expected consumption is approximately 0.7-1.3 GPU-hours. An explicit operational checkpoint after physical round 19 forbids a new generation round."
export READINESS_ALLOCATED_GPU_COUNT="4"
export READINESS_VALIDATION_SHARD_COUNT="4"
export READINESS_VALIDATION_BATCH_SIZE="16"
export READINESS_EMBEDDING_BATCH_SIZE="32"
export READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION="eager"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="1024"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="4"
export READINESS_GENERATION_SECONDS="10800"
export READINESS_FINALIZATION_RESERVE_SECONDS="900"
export READINESS_MAX_REFINEMENT_ROUNDS="1000"
export READINESS_STOP_AFTER_PHYSICAL_ROUND="19"
export READINESS_END_TO_END_RUNNER="$orchestrator_repository/analysis/scripts/slurm/jupiter/run_readiness_30k_end_to_end.sh"
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
    echo "one-node recovery requires exactly one allocated node" >&2
    exit 2
}
[[ -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json" ]]
[[ -d "$READINESS_INITIAL_CANDIDATE_ROOT" ]]
[[ -d "$READINESS_RECOVERY_PIPELINE_ROOT" ]]
[[ -s "$READINESS_BATTERY_ROOT/battery_manifest.json" ]]
[[ -s "$READINESS_PARTITION_ROOT_0/round-18/verified_round_summary.json" ]]
[[ -s "$READINESS_PARTITION_ROOT_1/round-18/verified_round_summary.json" ]]
[[ -d "$READINESS_PARTITION_ROOT_0/round-19/generation" ]]
[[ -d "$READINESS_PARTITION_ROOT_1/round-19/generation" ]]

latest_verified_summary="$({
    find "$READINESS_RECOVERY_PIPELINE_ROOT" -mindepth 2 -maxdepth 2 \
        -path '*/round-*/verified_round_summary.json' -type f -print
} | sort -V | tail -n 1)"
[[ -s "$latest_verified_summary" ]]
latest_verified_round="$(dirname "$latest_verified_summary")"
latest_verified_round_name="$(basename "$latest_verified_round")"
latest_verified_round_number="${latest_verified_round_name#round-}"
export READINESS_INITIAL_CANDIDATE_FILE_LIST="${READINESS_INITIAL_CANDIDATE_FILE_LIST:-$latest_verified_round/candidate-files.txt}"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="$((10#$latest_verified_round_number))"
export READINESS_INITIAL_PROJECTION_ROOT="${READINESS_INITIAL_PROJECTION_ROOT:-$latest_verified_round/projections}"
export READINESS_INITIAL_VALIDATION_OUTPUT="${READINESS_INITIAL_VALIDATION_OUTPUT:-$latest_verified_round/validation.jsonl}"

mkdir -p "$READINESS_PARTITION_COORDINATION_ROOT"
recovery_manifest="$READINESS_PARTITION_COORDINATION_ROOT/recovery-job-$SLURM_JOB_ID.json"
recovery_status="failed"

python3 - "$recovery_manifest" "$actual_orchestrator_commit" "$science_commit" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "format_version": "readiness-30k-one-node-recovery-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "orchestrator_git_commit_sha": sys.argv[2],
    "scientific_pipeline_git_commit_sha": sys.argv[3],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "allocated_nodes": 1,
    "allocated_gpus": 4,
    "allocated_cpus": 32,
    "maximum_gpu_hours": 4,
    "source_slurm_job_ids": ["1471783", "1471784", "1480390", "1480473", "1481430", "1484067"],
    "target_round": 19,
    "status": "running",
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY

record_recovery_exit() {
    local exit_status="$?"
    set +e
    python3 - "$recovery_manifest" "$recovery_status" "$exit_status" <<'PY'
from datetime import datetime, timezone
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.exists():
    payload = json.loads(path.read_text())
    payload["finished_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["status"] = sys.argv[2]
    payload["exit_status"] = int(sys.argv[3])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY
}
trap record_recovery_exit EXIT

corrupt_validation="$READINESS_PARTITION_ROOT_1/round-19/validation-shard-2.jsonl"
quarantined_validation="$corrupt_validation.corrupt-source-job-1481430"
quarantine_manifest="$READINESS_PARTITION_COORDINATION_ROOT/validation-shard-2-quarantine-source-job-1481430.json"
python3 \
    "$orchestrator_repository/analysis/scripts/quarantine_corrupt_readiness_validation.py" \
    --source "$corrupt_validation" \
    --quarantine "$quarantined_validation" \
    --manifest "$quarantine_manifest" \
    --source-job-id 1481430 \
    --recovery-job-id "$SLURM_JOB_ID"

write_producer_marker() {
    local partition_index="$1" pipeline_root="$2" summary="$3" producer_exit="$4"
    python3 - "$READINESS_PARTITION_COORDINATION_ROOT" "$partition_index" \
        "$pipeline_root" "$summary" "$producer_exit" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys

coordination = pathlib.Path(sys.argv[1])
index = int(sys.argv[2])
payload = {
    "format_version": "readiness-two-partition-producer-ready-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "partition_count": 2,
    "partition_index": index,
    "partition_salt": os.environ["READINESS_WORK_PARTITION_SALT"],
    "pipeline_root": str(pathlib.Path(sys.argv[3]).resolve()),
    "verified_summary": str(pathlib.Path(sys.argv[4]).resolve()),
    "producer_exit_code": int(sys.argv[5]),
}
output = coordination / f"producer-{index}.ready.json"
if output.exists():
    output.replace(output.with_suffix(output.suffix + f".superseded-{os.environ['SLURM_JOB_ID']}"))
temporary = output.with_suffix(output.suffix + f".{os.environ['SLURM_JOB_ID']}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(f"producer_marker={output}")
PY
}

recover_partition() {
    local partition_index="$1" pipeline_root producer_exit partition_summary
    if [[ "$partition_index" -eq 0 ]]; then
        pipeline_root="$READINESS_PARTITION_ROOT_0"
    else
        pipeline_root="$READINESS_PARTITION_ROOT_1"
    fi
    export READINESS_WORK_PARTITION_INDEX="$partition_index"
    export READINESS_30K_PIPELINE_ROOT="$pipeline_root"
    export READINESS_30K_PIPELINE_POINTER="$READINESS_PARTITION_COORDINATION_ROOT/partition-$partition_index-latest.txt"
    printf '%s\n' "$SLURM_JOB_ID" > "$pipeline_root/current-job-id.txt"

    echo "===== RECOVER PARTITION $partition_index ====="
    set +e
    "$orchestrator_repository/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_strict_loop.sh"
    producer_exit="$?"
    set -e
    if [[ "$producer_exit" -ne 0 && "$producer_exit" -ne 3 ]]; then
        echo "partition $partition_index recovery failed with exit $producer_exit" >&2
        return "$producer_exit"
    fi
    partition_summary="$({
        find "$pipeline_root" -mindepth 2 -maxdepth 2 \
            -path '*/round-*/verified_round_summary.json' -type f -print
    } | sort -V | tail -n 1)"
    [[ "$(basename "$(dirname "$partition_summary")")" == "round-19" ]] || {
        echo "partition $partition_index did not complete the intended round-19 checkpoint" >&2
        return 2
    }
    write_producer_marker "$partition_index" "$pipeline_root" \
        "$partition_summary" "$producer_exit"
}

recover_partition 0
recover_partition 1

"$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/finalize_readiness_30k_partitions.sh"
[[ -s "$READINESS_PARTITION_COORDINATION_ROOT/final-latest.txt" ]]
final_root="$(<"$READINESS_PARTITION_COORDINATION_ROOT/final-latest.txt")"
[[ -s "$final_root/verified_round_summary.json" ]]
echo "RECOVERY FINAL RESULT=$final_root"
python3 -m json.tool "$final_root/verified_round_summary.json"
recovery_status="complete"
