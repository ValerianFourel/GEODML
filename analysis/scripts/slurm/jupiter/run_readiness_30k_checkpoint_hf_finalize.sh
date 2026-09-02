#!/bin/bash -l
# Audit one verified cumulative checkpoint and build one counted text dataset.
# This wrapper runs inside an approved allocation and never allocates resources.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside the specifically approved Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact clean repository checkout}"
: "${GEODML_LLM2VEC_EXPORT_VENV:?Set the existing export environment}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved merge wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the supporting runtime estimate}"
: "${READINESS_PROMPT_POPULATION_ROOT:?Set the verified cumulative checkpoint}"
: "${READINESS_LIKERT_DATASET_ROOT:?Set the verified HF-safe Likert dataset}"
: "${READINESS_CHECKPOINT_MERGE_ROOT:?Set a fresh merge output root}"
: "${READINESS_TEXT_DATASET_ROOT:?Set a fresh unified text dataset path}"

[[ "$READINESS_APPROVED_WALLTIME" == "03:00:00" ]] || {
    echo "this checkpoint merge was approved specifically for 03:00:00" >&2
    exit 2
}
[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ]] || {
    echo "checkpoint merge requires exactly one allocated node" >&2
    exit 2
}
[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]
[[ -d "$READINESS_PROMPT_POPULATION_ROOT" ]]
[[ -d "$READINESS_CHECKPOINT_MERGE_ROOT" ]]
[[ ! -e "$READINESS_TEXT_DATASET_ROOT" ]]
[[ -s "$READINESS_LIKERT_DATASET_ROOT/dataset_manifest.json" ]]
[[ -x "$GEODML_LLM2VEC_EXPORT_VENV/bin/python" ]]

required_population_files=(
    "$READINESS_PROMPT_POPULATION_ROOT/verified_round_summary.json"
    "$READINESS_PROMPT_POPULATION_ROOT/candidate-files.txt"
    "$READINESS_PROMPT_POPULATION_ROOT/validation.jsonl"
    "$READINESS_PROMPT_POPULATION_ROOT/validation.jsonl.manifest.json"
    "$READINESS_PROMPT_POPULATION_ROOT/strict-selection/run_manifest.json"
    "$READINESS_PROMPT_POPULATION_ROOT/strict-selection/spatially_selected_questions.jsonl"
    "$READINESS_PROMPT_POPULATION_ROOT/selected-diversity/run_manifest.json"
    "$READINESS_PROMPT_POPULATION_ROOT/selected-diversity/question_diversity_audit.json"
)
for required_file in "${required_population_files[@]}"; do
    [[ -s "$required_file" ]] || {
        echo "missing cumulative checkpoint artifact: $required_file" >&2
        exit 2
    }
done

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "CHECKPOINT_MERGE_CONTROL_RUNTIME=PASS"
source "$GEODML_LLM2VEC_EXPORT_VENV/bin/activate"
python -c 'import huggingface_hub, pyarrow'

cd "$GEODML_REPOSITORY"
python analysis/scripts/build_readiness_hf_dataset.py verify \
    --dataset-dir "$READINESS_LIKERT_DATASET_ROOT"

audit_report="$READINESS_CHECKPOINT_MERGE_ROOT/fully-compliant-audit.json"
summary_report="$READINESS_CHECKPOINT_MERGE_ROOT/unification-summary.json"

python analysis/scripts/audit_fully_compliant_readiness_prompts.py \
    --final-root "$READINESS_PROMPT_POPULATION_ROOT" \
    --report-file "$audit_report" \
    --json

python analysis/scripts/build_readiness_text_hf_dataset.py finalize \
    --likert-dataset-root "$READINESS_LIKERT_DATASET_ROOT" \
    --prompt-population-root "$READINESS_PROMPT_POPULATION_ROOT" \
    --output-dir "$READINESS_TEXT_DATASET_ROOT" \
    --rows-per-shard 100000 \
    --git-commit-sha "$GEODML_EXPECTED_COMMIT"

python analysis/scripts/build_readiness_text_hf_dataset.py verify \
    --dataset-dir "$READINESS_TEXT_DATASET_ROOT"

python - "$READINESS_PROMPT_POPULATION_ROOT" "$audit_report" \
    "$READINESS_TEXT_DATASET_ROOT" "$summary_report" <<'PY'
from datetime import datetime, timezone
import json
import pathlib
import sys

population_root = pathlib.Path(sys.argv[1]).resolve()
audit_path = pathlib.Path(sys.argv[2]).resolve()
dataset_root = pathlib.Path(sys.argv[3]).resolve()
output = pathlib.Path(sys.argv[4]).resolve()

summary = json.loads(
    (population_root / "verified_round_summary.json").read_text(encoding="utf-8")
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
if int(audit["ready_to_export_count"]) != selected_count:
    raise SystemExit("audit ready-to-export count differs from global selection")

payload = {
    "format_version": "axisgeo-unified-count-summary-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "prompt_population_root": str(population_root),
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

echo "CHECKPOINT_MERGE_AND_HF_FINALIZE=PASS"
echo "population=$READINESS_PROMPT_POPULATION_ROOT"
echo "dataset=$READINESS_TEXT_DATASET_ROOT"
echo "counts=$summary_report"
