#!/bin/bash -l
# Launch one approved CPU-only checkpoint merge in a detached tmux session.

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact clean repository checkout}"
: "${READINESS_PROMPT_POPULATION_ROOT:?Set the verified cumulative checkpoint}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved merge wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the supporting runtime estimate}"
: "${READINESS_HF_REPO_ID:?Set the private Hugging Face dataset repository}"
: "${READINESS_HF_CONFIRM_REPO_ID:?Confirm the same dataset repository}"

[[ "$READINESS_APPROVED_WALLTIME" == "03:00:00" ]] || {
    echo "this checkpoint merge was approved specifically for 03:00:00" >&2
    exit 2
}

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-${PROJECT:?}/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-${FSCRATCH:?}/$USER/geodml}"
export GEODML_LLM2VEC_EXPORT_VENV="${GEODML_LLM2VEC_EXPORT_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec}"

[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]
[[ -s "$READINESS_PROMPT_POPULATION_ROOT/verified_round_summary.json" ]]
[[ -s "$READINESS_PROMPT_POPULATION_ROOT/candidate-files.txt" ]]
[[ -s "$READINESS_PROMPT_POPULATION_ROOT/validation.jsonl" ]]
[[ -s "$READINESS_PROMPT_POPULATION_ROOT/strict-selection/spatially_selected_questions.jsonl" ]]
[[ -x "$GEODML_LLM2VEC_EXPORT_VENV/bin/python" ]]

python3 - "$READINESS_PROMPT_POPULATION_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
summary = json.loads((root / "verified_round_summary.json").read_text())
selection = json.loads((root / "strict-selection/run_manifest.json").read_text())
pipeline = json.loads((root.parent / "pipeline_manifest.json").read_text())

if selection.get("text_contract") != "search-trigger-v2":
    raise SystemExit("checkpoint is not search-trigger-v2")
if selection.get("acceptance_contract_version") != "search-trigger-v2":
    raise SystemExit("checkpoint does not use search-trigger-v2 acceptance")
if float(selection["coordinate_acceptance_contract"]["distance_tolerance"]) != 0.035:
    raise SystemExit("checkpoint does not use distance tolerance 0.035")
if pipeline.get("generation_profile") != "high-axis-action-v1":
    raise SystemExit("checkpoint is not high-axis-action-v1")

print(json.dumps({
    "CHECKPOINT_MERGE_PREFLIGHT": "PASS",
    "prompt_population_root": str(root),
    "candidate_count": int(summary["candidate_count"]),
    "selected_count": int(summary["selected_count"]),
    "remaining_target_count": int(summary["refinement_task_count"]),
}, indent=2, sort_keys=True))
PY

likert_export_root="$(<"$HOME/geodml-readiness-hf-export-latest.txt")"
if [[ -s "$likert_export_root/export-environment.sh" ]]; then
    export READINESS_LIKERT_DATASET_ROOT="$(
        bash -c 'source "$1/export-environment.sh"; printf "%s" "$READINESS_HF_DATASET_ROOT"' \
            _ "$likert_export_root"
    )"
else
    export READINESS_LIKERT_DATASET_ROOT="$likert_export_root/huggingface-dataset"
fi
[[ -s "$READINESS_LIKERT_DATASET_ROOT/dataset_manifest.json" ]]

run_id="checkpoint-merge-$(date -u +%Y%m%dT%H%M%SZ)"
export READINESS_CHECKPOINT_MERGE_ROOT="${READINESS_CHECKPOINT_MERGE_ROOT:-$GEODML_CACHE_ROOT/runs/readiness-30k-checkpoint-merge/$run_id}"
export READINESS_TEXT_DATASET_ROOT="${READINESS_TEXT_DATASET_ROOT:-$READINESS_CHECKPOINT_MERGE_ROOT/axisgeo-unified-hf-dataset}"
export READINESS_MERGE_TMUX_SESSION="${READINESS_MERGE_TMUX_SESSION:-axisv2-$run_id}"
export READINESS_MERGE_LOG="${READINESS_MERGE_LOG:-$READINESS_CHECKPOINT_MERGE_ROOT/merge-and-finalize.tmux.log}"
export READINESS_SECONDARY_STATE_FILE="${READINESS_SECONDARY_STATE_FILE:-$HOME/geodml-axis-v2-secondary-latest.env}"
export READINESS_HF_PUBLISH_RECEIPT="${READINESS_HF_PUBLISH_RECEIPT:-$READINESS_CHECKPOINT_MERGE_ROOT/hf-publication-receipt.json}"
export READINESS_PUBLISH_TMUX_SESSION="${READINESS_PUBLISH_TMUX_SESSION:-axisv2-publish-$run_id}"
export READINESS_PUBLISH_LOG="${READINESS_PUBLISH_LOG:-$READINESS_CHECKPOINT_MERGE_ROOT/hf-private-publish.tmux.log}"

[[ ! -e "$READINESS_CHECKPOINT_MERGE_ROOT" ]]
if tmux has-session -t "$READINESS_MERGE_TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $READINESS_MERGE_TMUX_SESSION" >&2
    exit 2
fi
mkdir -p "$READINESS_CHECKPOINT_MERGE_ROOT"

declare -px \
    GEODML_EXPECTED_COMMIT GEODML_PROJECT_ROOT GEODML_CACHE_ROOT \
    GEODML_REPOSITORY GEODML_LLM2VEC_EXPORT_VENV \
    READINESS_APPROVED_WALLTIME READINESS_ALLOCATION_ESTIMATE \
    READINESS_PROMPT_POPULATION_ROOT READINESS_LIKERT_DATASET_ROOT \
    READINESS_CHECKPOINT_MERGE_ROOT READINESS_TEXT_DATASET_ROOT \
    READINESS_MERGE_TMUX_SESSION READINESS_MERGE_LOG \
    READINESS_HF_REPO_ID READINESS_HF_CONFIRM_REPO_ID \
    READINESS_HF_PUBLISH_RECEIPT READINESS_PUBLISH_TMUX_SESSION \
    READINESS_PUBLISH_LOG \
    READINESS_SECONDARY_STATE_FILE > "$READINESS_SECONDARY_STATE_FILE"
chmod 600 "$READINESS_SECONDARY_STATE_FILE"

worker="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_checkpoint_hf_finalize.sh"
[[ -x "$worker" ]]

tmux new-session -d -s "$READINESS_MERGE_TMUX_SESSION" \
    "set -o pipefail; env GEODML_EXPECTED_COMMIT='$GEODML_EXPECTED_COMMIT' GEODML_PROJECT_ROOT='$GEODML_PROJECT_ROOT' GEODML_CACHE_ROOT='$GEODML_CACHE_ROOT' GEODML_REPOSITORY='$GEODML_REPOSITORY' GEODML_LLM2VEC_EXPORT_VENV='$GEODML_LLM2VEC_EXPORT_VENV' READINESS_APPROVED_WALLTIME='$READINESS_APPROVED_WALLTIME' READINESS_ALLOCATION_ESTIMATE='$READINESS_ALLOCATION_ESTIMATE' READINESS_PROMPT_POPULATION_ROOT='$READINESS_PROMPT_POPULATION_ROOT' READINESS_LIKERT_DATASET_ROOT='$READINESS_LIKERT_DATASET_ROOT' READINESS_CHECKPOINT_MERGE_ROOT='$READINESS_CHECKPOINT_MERGE_ROOT' READINESS_TEXT_DATASET_ROOT='$READINESS_TEXT_DATASET_ROOT' salloc --account=scifi --partition=booster --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --gres=none --time=03:00:00 --job-name='axisv2-merge-hf' bash '$worker' 2>&1 | tee '$READINESS_MERGE_LOG'"

echo "CHECKPOINT_MERGE_TMUX_LAUNCHED=PASS"
echo "session=$READINESS_MERGE_TMUX_SESSION"
echo "log=$READINESS_MERGE_LOG"
echo "state=$READINESS_SECONDARY_STATE_FILE"
tmux list-sessions -F '#{session_name} #{session_windows} #{session_created_string}' |
    awk -v name="$READINESS_MERGE_TMUX_SESSION" '$1 == name'
squeue --me --format='%.18i %.28j %.9T %.10M %.10l %R'
