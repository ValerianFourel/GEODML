#!/usr/bin/env bash
# Build and verify the deterministic 128-prompt ACL ARR pilot plan.

set -euo pipefail

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; prepare the pilot environment first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"

CURRENT_COMMIT="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
[[ "$CURRENT_COMMIT" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -s "$ACL_ARR_RUN_ROOT/model-downloads.complete" ]]
[[ -s "$ACL_ARR_RUN_ROOT/models.json" ]]
[[ -x "$ACL_ARR_VENV/bin/python" ]]

source "$ACL_ARR_VENV/bin/activate"
cd "$GEODML_REPOSITORY"
export PYTHONPATH="$GEODML_REPOSITORY:$GEODML_REPOSITORY/analysis"
PREPARED_ROOT="$ACL_ARR_RUN_ROOT/prepared"
mkdir -p "$PREPARED_ROOT"

python3 analysis/scripts/download_acl_arr_pilot_models.py \
    --run-root "$ACL_ARR_RUN_ROOT" \
    --verify-only

python3 analysis/scripts/prepare_acl_arr_pilot_inputs.py \
    --audit-root "$AUDIT_ROOT" \
    --serp "$ACL_ARR_SEARCH_SNAPSHOT" \
    --data-root "$GEODML_DATA_ROOT" \
    --pilot-size 128 \
    --master-seed 20260904 \
    --engine searxng \
    --pool 20 \
    --minimum-documents 11 \
    --max-document-characters 12000 \
    --output-dir "$PREPARED_ROOT"

if [[ ! -s "$ACL_ARR_RUN_ROOT/document-freeze/document_freeze_manifest.json" ]]; then
    python3 analysis/scripts/prepare_acl_arr_document_sets.py \
        --serp "$PREPARED_ROOT/pilot-serp.jsonl" \
        --page-text "$PREPARED_ROOT/pilot-page-text.jsonl" \
        --minimum-documents 11 \
        --maximum-documents 20 \
        --max-document-characters 12000 \
        --source-git-commit "$GEODML_EXPECTED_COMMIT" \
        --output-dir "$ACL_ARR_RUN_ROOT/document-freeze"
fi

if [[ ! -s "$ACL_ARR_RUN_ROOT/plan/run_manifest.json" ]]; then
    python3 analysis/scripts/prepare_acl_arr_experiment.py \
        --prompts-jsonl "$PREPARED_ROOT/pilot-prompts.jsonl" \
        --axis-map-jsonl "$PREPARED_ROOT/pilot-axis.jsonl" \
        --document-sets-jsonl "$ACL_ARR_RUN_ROOT/document-freeze/frozen_document_sets.jsonl" \
        --models-json "$ACL_ARR_RUN_ROOT/models.json" \
        --top-n 10 \
        --master-seed 20260904 \
        --expected-prompt-count 128 \
        --expected-model-count 4 \
        --source-git-commit "$GEODML_EXPECTED_COMMIT" \
        --output-dir "$ACL_ARR_RUN_ROOT/plan"
fi

python3 - "$ACL_ARR_RUN_ROOT/plan/run_manifest.json" "$GEODML_EXPECTED_COMMIT" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
summary = manifest["summary"]
assert manifest["source_git_commit"] == expected_commit
assert summary["prompt_count"] == 128
assert summary["model_count"] == 4
assert summary["tasks_per_model_pipeline"] == 384
assert summary["primary_task_count"] == 3072
assert summary["planned_judge_task_count"] == 1536
assert summary["planned_total_inference_count"] == 4608
for artifact in manifest["artifacts"]["tasks"].values():
    path = pathlib.Path(artifact["path"])
    count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    assert count == 384, (path, count)
print(json.dumps(summary, indent=2, sort_keys=True))
print("PILOT_PLAN=PASS")
PY

python3 -m json.tool "$ACL_ARR_RUN_ROOT/document-freeze/document_freeze_manifest.json"
sha256sum \
    "$PREPARED_ROOT/pilot-prompts.jsonl" \
    "$PREPARED_ROOT/pilot-axis.jsonl" \
    "$ACL_ARR_RUN_ROOT/document-freeze/frozen_document_sets.jsonl" \
    "$ACL_ARR_RUN_ROOT/plan/run_manifest.json"
