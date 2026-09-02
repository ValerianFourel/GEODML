#!/bin/bash -l
# Run one disjoint high-axis search-trigger-v2 section inside an approved allocation.
# This wrapper never requests or creates a Slurm allocation.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside a specifically approved Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact repository checkout}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting the allocation}"
: "${READINESS_KEYWORD_SECTION_PLAN:?Set the immutable ten-section plan}"
: "${READINESS_TEN_SECTION_RUN_ROOT:?Set the shared ten-section output root}"
: "${READINESS_WORK_PARTITION_INDEX:?Set one unique section index from 0 through 9}"

[[ "$READINESS_APPROVED_WALLTIME" == "05:00:00" ]] || {
    echo "this ten-section wave was approved specifically for 05:00:00 per job" >&2
    exit 2
}

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "HIGH_AXIS_V2_SECTION_CONTROL_RUNTIME=PASS"

readarray -t section_identity < <(
    python3 - "$READINESS_KEYWORD_SECTION_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(pathlib.Path(plan["checkpoint_root"]).resolve())
print(int(plan["section_count"]))
PY
)
checkpoint="${section_identity[0]}"
section_count="${section_identity[1]}"
[[ "$section_count" == "10" ]] || {
    echo "high-axis v2 wave requires exactly ten disjoint sections" >&2
    exit 2
}

baseline_selected="$checkpoint/strict-selection/spatially_selected_questions.jsonl"
test -s "$baseline_selected"
python3 - "$checkpoint" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
selection = json.loads(
    (checkpoint / "strict-selection/run_manifest.json").read_text(encoding="utf-8")
)
pipeline = json.loads(
    (checkpoint.parent / "pipeline_manifest.json").read_text(encoding="utf-8")
)
if selection.get("text_contract", "question-v1") != "search-trigger-v2":
    raise SystemExit("section checkpoint is not search-trigger-v2")
if selection.get("acceptance_contract_version", "question-v1") != "search-trigger-v2":
    raise SystemExit("section checkpoint does not use search-trigger-v2 acceptance")
distance = float(selection["coordinate_acceptance_contract"]["distance_tolerance"])
if distance != 0.035:
    raise SystemExit(f"section checkpoint tolerance must be 0.035; found {distance}")
if pipeline.get("generation_profile") != "high-axis-action-v1":
    raise SystemExit("section checkpoint is not a high-axis-action-v1 pipeline")
if float(pipeline.get("refinement_minimum_target_axis_1", -1)) != 0.7:
    raise SystemExit("section checkpoint does not preserve the 0.700 minimum")
if pipeline.get("refinement_task_priority") != "descending-axis-1":
    raise SystemExit("section checkpoint does not preserve descending-axis-1 priority")
print("HIGH_AXIS_V2_SECTION_CHECKPOINT=PASS")
PY

export READINESS_GENERATION_PROFILE="high-axis-action-v1"
export READINESS_TEXT_CONTRACT="search-trigger-v2"
export READINESS_ACCEPTANCE_CONTRACT="search-trigger-v2"
export READINESS_DISTANCE_TOLERANCE="0.035"
export READINESS_COORDINATE_ONLY_PROJECTION_REUSE="1"
export READINESS_REFINEMENT_MIN_TARGET_AXIS_1="0.700"
export READINESS_REFINEMENT_TASK_PRIORITY="descending-axis-1"
export READINESS_HIGH_AXIS_BASELINE_SELECTED="$baseline_selected"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-1024}"
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
export READINESS_FINALIZATION_RESERVE_SECONDS="${READINESS_FINALIZATION_RESERVE_SECONDS:-2400}"

echo "SEARCH-TRIGGER-V2 HIGH-AXIS SECTION"
echo "job=$SLURM_JOB_ID"
echo "section=$READINESS_WORK_PARTITION_INDEX/$section_count"
echo "checkpoint=$checkpoint"
echo "baseline_selected=$READINESS_HIGH_AXIS_BASELINE_SELECTED"
echo "approved_walltime=$READINESS_APPROVED_WALLTIME"

driver="$jupiter_dir/run_readiness_30k_axis1_keyword_section.sbatch"
test -s "$driver"
exec bash "$driver"
