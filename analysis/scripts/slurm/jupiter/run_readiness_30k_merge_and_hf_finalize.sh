#!/bin/bash -l
# Merge ten verified sections and build one counted, checksum-verified text dataset.
# This wrapper runs inside an approved allocation and never allocates resources.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside the specifically approved Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact clean repository checkout}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved merge wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the supporting runtime estimate}"
: "${READINESS_SOURCE_TEN_SECTION_RUN_ROOT:?Set the completed section root}"
: "${READINESS_REPARTITION_ROOT:?Set a fresh global merge root}"
: "${READINESS_NEW_PARTITION_SALT:?Set the next deterministic partition salt}"
: "${READINESS_LIKERT_DATASET_ROOT:?Set the verified HF-safe Likert dataset}"
: "${READINESS_TEXT_DATASET_ROOT:?Set a fresh unified text dataset path}"
: "${GEODML_LLM2VEC_EXPORT_VENV:?Set the existing export environment}"

[[ "$READINESS_APPROVED_WALLTIME" == "03:00:00" ]] || {
    echo "this merge and finalization was approved specifically for 03:00:00" >&2
    exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ]] || {
    echo "merge and finalization requires exactly one allocated node" >&2
    exit 2
}
[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]
[[ -d "$READINESS_SOURCE_TEN_SECTION_RUN_ROOT" ]]
[[ -d "$READINESS_REPARTITION_ROOT" ]]
[[ ! -e "$READINESS_TEXT_DATASET_ROOT" ]]
[[ -s "$READINESS_LIKERT_DATASET_ROOT/dataset_manifest.json" ]]
[[ -x "$GEODML_LLM2VEC_EXPORT_VENV/bin/python" ]]

python3 - "$READINESS_SOURCE_TEN_SECTION_RUN_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
markers = sorted((root / "coordination").glob("section-*-of-10.ready.json"))
if len(markers) != 10:
    raise SystemExit(
        f"expected exactly ten ready section markers; found {len(markers)}"
    )
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in markers]
if {int(row["section_index"]) for row in payloads} != set(range(10)):
    raise SystemExit("ready markers do not cover section indices 0 through 9")
if len({str(row["slurm_job_id"]) for row in payloads}) != 10:
    raise SystemExit("ready markers do not identify ten distinct Slurm jobs")
if any(int(row["section_exit_code"]) not in {0, 3} for row in payloads):
    raise SystemExit("at least one section marker records a failed producer")
if len({str(row["section_plan_sha256"]) for row in payloads}) != 1:
    raise SystemExit("ready markers do not share one immutable section plan")
for row in payloads:
    pipeline = pathlib.Path(str(row["pipeline_root"])).resolve()
    if pipeline.parent != root:
        raise SystemExit("ready marker points outside the ten-section root")
    if not pathlib.Path(str(row["verified_summary"])).is_file():
        raise SystemExit("ready marker points to a missing verified summary")
print("TEN_READY_SECTION_MARKERS=PASS")
PY

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "MERGE_AND_HF_FINALIZE_CONTROL_RUNTIME=PASS"
source "$GEODML_LLM2VEC_EXPORT_VENV/bin/activate"
python -c 'import huggingface_hub, pyarrow'

cd "$GEODML_REPOSITORY"
python analysis/scripts/build_readiness_hf_dataset.py verify \
    --dataset-dir "$READINESS_LIKERT_DATASET_ROOT"

export READINESS_TEXT_CONTRACT="search-trigger-v2"
export READINESS_ACCEPTANCE_CONTRACT="search-trigger-v2"
export READINESS_GENERATION_PROFILE="high-axis-action-v1"
export READINESS_DISTANCE_TOLERANCE="0.035"
export READINESS_REFINEMENT_MIN_TARGET_AXIS_1="0.700"
export READINESS_REFINEMENT_TASK_PRIORITY="descending-axis-1"

bash "$jupiter_dir/run_readiness_30k_repartition_keyword_sections.sbatch"

final_root="$READINESS_REPARTITION_ROOT/checkpoint"
audit_report="$READINESS_REPARTITION_ROOT/fully-compliant-audit.json"
[[ -s "$final_root/verified_round_summary.json" ]]
[[ -s "$final_root/merged/candidates.jsonl" ]]
[[ -s "$final_root/merged/validation.jsonl" ]]
[[ -s "$final_root/strict-selection/spatially_selected_questions.jsonl" ]]
[[ -s "$audit_report" ]]

python analysis/scripts/build_readiness_text_hf_dataset.py finalize \
    --likert-dataset-root "$READINESS_LIKERT_DATASET_ROOT" \
    --prompt-population-root "$final_root" \
    --output-dir "$READINESS_TEXT_DATASET_ROOT" \
    --rows-per-shard 100000 \
    --git-commit-sha "$GEODML_EXPECTED_COMMIT"

python analysis/scripts/build_readiness_text_hf_dataset.py verify \
    --dataset-dir "$READINESS_TEXT_DATASET_ROOT"

python - "$final_root" "$audit_report" "$READINESS_TEXT_DATASET_ROOT" \
    "$READINESS_REPARTITION_ROOT/unification-summary.json" <<'PY'
from datetime import datetime, timezone
import json
import pathlib
import sys

final_root = pathlib.Path(sys.argv[1]).resolve()
audit_path = pathlib.Path(sys.argv[2]).resolve()
dataset_root = pathlib.Path(sys.argv[3]).resolve()
output = pathlib.Path(sys.argv[4]).resolve()
summary = json.loads(
    (final_root / "verified_round_summary.json").read_text(encoding="utf-8")
)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
dataset = json.loads(
    (dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
)
candidate_count = int(summary["candidate_count"])
selected_count = int(summary["selected_count"])
counts = dataset["table_counts"]
expected = {
    "generated_candidates": candidate_count,
    "candidate_compliance_annotations": candidate_count,
    "fully_compliant_prompts": selected_count,
}
observed = {key: int(counts.get(key, -1)) for key in expected}
if observed != expected:
    raise SystemExit(
        f"unified dataset counts differ: observed={observed} expected={expected}"
    )
if audit.get("audit_passed") is not True:
    raise SystemExit("independent fully compliant prompt audit did not pass")
if int(audit["fully_compliant_prompt_count"]) != selected_count:
    raise SystemExit("audit fully compliant count differs from global selection")
if int(audit["ready_to_export_count"]) != selected_count:
    raise SystemExit("audit ready-to-export count differs from global selection")
payload = {
    "format_version": "axisgeo-unified-count-summary-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "final_checkpoint": str(final_root),
    "dataset_root": str(dataset_root),
    "candidate_count": candidate_count,
    "independently_accepted_count": int(summary["independently_accepted_count"]),
    "fully_compliant_prompt_count": selected_count,
    "remaining_target_count": int(summary["refinement_task_count"]),
    "verified_population_passed": bool(summary["verified_population_passed"]),
    "complete_30330_population_passed": bool(
        audit["complete_30330_population_passed"]
    ),
    "dataset_table_counts": counts,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "MERGE_AND_HF_FINALIZE=PASS"
echo "checkpoint=$final_root"
echo "dataset=$READINESS_TEXT_DATASET_ROOT"
echo "counts=$READINESS_REPARTITION_ROOT/unification-summary.json"
