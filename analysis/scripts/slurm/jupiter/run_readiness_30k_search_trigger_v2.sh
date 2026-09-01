#!/bin/bash -l
# Resume the global axis-one population under the relaxed search-trigger-v2 contract.
# This controller runs inside an existing four-GPU allocation; it never allocates.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside the specifically approved Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting this allocation}"
: "${READINESS_GLOBAL_COORDINATE_ROOT:?Set the immutable global coordinate checkpoint}"
: "${READINESS_30K_PLAN_ROOT:?Set the frozen 30,330-target uniform plan}"
: "${READINESS_SUBSPACE_ROOT:?Set the frozen readiness subspace root}"
: "${READINESS_30K_PIPELINE_ROOT:?Set a persistent, fresh v2 pipeline root}"

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "SEARCH_TRIGGER_V2_CONTROL_RUNTIME=PASS"

approved_walltime_seconds="$(python3 - "$READINESS_APPROVED_WALLTIME" <<'PY'
import re
import sys

value = sys.argv[1]
match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
if match is None:
    raise SystemExit("approved walltime must use [days-]HH:MM:SS")
days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
if minutes >= 60 or seconds >= 60:
    raise SystemExit("approved walltime contains an invalid minute or second")
total = days * 86400 + hours * 3600 + minutes * 60 + seconds
if total <= 0:
    raise SystemExit("approved walltime must be positive")
print(total)
PY
)"
export READINESS_APPROVED_WALLTIME_SECONDS="$approved_walltime_seconds"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-${GEODML_EXPECTED_COMMIT:0:12}}"
export SLURM_MPI_TYPE="none"
export READINESS_ALLOCATED_GPU_COUNT="${READINESS_ALLOCATED_GPU_COUNT:-4}"
[[ "$READINESS_ALLOCATED_GPU_COUNT" == "4" ]] || {
    echo "search-trigger-v2 production requires exactly four allocated GPUs" >&2
    exit 2
}

coordinate_root="$(realpath "$READINESS_GLOBAL_COORDINATE_ROOT")"
candidate_file="$coordinate_root/candidates.jsonl"
validation_file="$coordinate_root/validation.jsonl"
merge_manifest="$coordinate_root/merge_manifest.json"
for path in \
    "$candidate_file" \
    "$candidate_file.manifest.json" \
    "$validation_file" \
    "$validation_file.manifest.json" \
    "$merge_manifest" \
    "$coordinate_root/projections/qwen/question_projections.jsonl" \
    "$coordinate_root/projections/qwen/projection_manifest.json" \
    "$coordinate_root/projections/mistral/question_projections.jsonl" \
    "$coordinate_root/projections/mistral/projection_manifest.json" \
    "$READINESS_30K_PLAN_ROOT/plan_manifest.json"
do
    [[ -s "$path" ]] || {
        echo "missing immutable v2 input: $path" >&2
        exit 2
    }
done

read -r initial_round candidate_count embedding_arrays_included < <(
    python3 - "$merge_manifest" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    int(row["maximum_candidate_round_index"]),
    int(row["candidate_count"]),
    str(bool(row.get("embedding_arrays_included", True))).lower(),
)
PY
)
[[ "$candidate_count" -gt 0 ]] || {
    echo "global coordinate checkpoint has no candidates" >&2
    exit 2
}
[[ "$embedding_arrays_included" == "false" ]] || {
    echo "expected the deliberate coordinate-only global checkpoint" >&2
    exit 2
}
export READINESS_BASELINE_CANDIDATE_COUNT="$candidate_count"

mkdir -p "$READINESS_30K_PIPELINE_ROOT/bootstrap"
candidate_list="$READINESS_30K_PIPELINE_ROOT/bootstrap/global-candidate-files.txt"
printf '%s\n' "$candidate_file" > "$candidate_list"

export READINESS_INITIAL_CANDIDATE_FILE_LIST="$candidate_list"
export READINESS_INITIAL_PROJECTION_ROOT="$coordinate_root/projections"
export READINESS_INITIAL_VALIDATION_OUTPUT="$validation_file"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="$initial_round"
export READINESS_COORDINATE_ONLY_PROJECTION_REUSE="1"

export READINESS_TEXT_CONTRACT="search-trigger-v2"
export READINESS_ACCEPTANCE_CONTRACT="search-trigger-v2"
export READINESS_DISTANCE_TOLERANCE="0.035"
export READINESS_DISAGREEMENT_WEIGHT="0.10"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="${READINESS_REFINEMENT_CANDIDATES_PER_TASK:-4}"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024}"
export READINESS_MAX_REFINEMENT_ROUNDS="${READINESS_MAX_REFINEMENT_ROUNDS:-1000}"
export READINESS_VALIDATION_SHARD_COUNT="4"
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
export READINESS_FINALIZATION_RESERVE_SECONDS="${READINESS_FINALIZATION_RESERVE_SECONDS:-2400}"
export READINESS_MASTER_SEED="${READINESS_MASTER_SEED:-20260820}"
export READINESS_WORK_PARTITION_COUNT="1"
export READINESS_WORK_PARTITION_INDEX="0"
export READINESS_WORK_PARTITION_SALT="search-trigger-v2-global-${GEODML_EXPECTED_COMMIT:0:12}"
default_validation_cache_search_root="$(dirname "$coordinate_root")"
export READINESS_VALIDATION_CACHE_SEARCH_ROOTS="${READINESS_VALIDATION_CACHE_SEARCH_ROOTS:-$default_validation_cache_search_root}"

python3 - "$READINESS_30K_PIPELINE_ROOT/search-trigger-v2-allocation.json" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path

path = Path(os.sys.argv[1])
record = {
    "format_version": "readiness-search-trigger-v2-allocation-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "nodes": 1,
    "gpus": 4,
    "maximum_gpu_hours": (
        4 * int(os.environ["READINESS_APPROVED_WALLTIME_SECONDS"]) / 3600
    ),
    "scheduler_allocated_cpus": int(os.environ.get("SLURM_CPUS_ON_NODE", "288")),
    "distance_tolerance": 0.035,
    "text_contract": "search-trigger-v2",
    "acceptance_contract_version": "search-trigger-v2",
    "generation_profile": os.environ.get(
        "READINESS_GENERATION_PROFILE", "balanced-v1"
    ),
    "refinement_minimum_target_axis_1": (
        float(os.environ["READINESS_REFINEMENT_MIN_TARGET_AXIS_1"])
        if os.environ.get("READINESS_REFINEMENT_MIN_TARGET_AXIS_1")
        else None
    ),
    "refinement_task_priority": os.environ.get(
        "READINESS_REFINEMENT_TASK_PRIORITY", "stable-hash"
    ),
    "global_coordinate_root": os.environ["READINESS_GLOBAL_COORDINATE_ROOT"],
    "baseline_candidate_count": int(os.environ["READINESS_BASELINE_CANDIDATE_COUNT"]),
    "baseline_maximum_round": int(os.environ["READINESS_INITIAL_LOGICAL_ROUND_INDEX"]),
    "coordinate_only_projection_reuse": True,
    "scientific_guard": (
        "Prompt embeddings describe generated text and do not define randomized policy B."
    ),
}
if path.exists():
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault("generation_profile", "balanced-v1")
    existing.setdefault("refinement_minimum_target_axis_1", None)
    existing.setdefault("refinement_task_priority", "stable-hash")
    stable = set(record) - {"created_at", "slurm_job_id"}
    if any(existing.get(key) != record[key] for key in stable):
        raise SystemExit("existing v2 allocation record has a different identity")
else:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY

echo "SEARCH-TRIGGER-V2 ADDITIVE LOOP"
echo "job=$SLURM_JOB_ID"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "baseline_candidates=$candidate_count"
echo "baseline_maximum_round=$initial_round"
echo "tolerance=$READINESS_DISTANCE_TOLERANCE"
echo "pipeline=$READINESS_30K_PIPELINE_ROOT"
echo "coordinate_only_reuse=1"

runner="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_end_to_end.sh"
exec "$runner"
