#!/bin/bash -l
# Launch ten disjoint high-axis prompt-generation sections with interactive
# Slurm allocations in detached tmux sessions.

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact clean repository checkout}"
: "${AXIS_V2_CHECKPOINT:?Set the verified high-axis checkpoint}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved wall time per section}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the supporting runtime estimate}"

[[ "$READINESS_APPROVED_WALLTIME" == "05:00:00" ]] || {
    echo "this generation wave was approved specifically for 05:00:00 per node" >&2
    exit 2
}

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-${PROJECT:?}/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-${FSCRATCH:?}/$USER/geodml}"

[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]

python3 - "$AXIS_V2_CHECKPOINT" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1]).resolve()
required = [
    checkpoint / "verified_round_summary.json",
    checkpoint / "candidate-files.txt",
    checkpoint / "validation.jsonl",
    checkpoint / "validation.jsonl.manifest.json",
    checkpoint / "projections/qwen/projection_manifest.json",
    checkpoint / "projections/mistral/projection_manifest.json",
    checkpoint / "strict-selection/run_manifest.json",
    checkpoint / "strict-selection/spatially_selected_questions.jsonl",
]
missing = [
    str(path)
    for path in required
    if not path.is_file() or path.stat().st_size == 0
]
if missing:
    raise SystemExit("missing checkpoint artifacts: " + ", ".join(missing))

summary = json.loads((checkpoint / "verified_round_summary.json").read_text())
selection = json.loads((checkpoint / "strict-selection/run_manifest.json").read_text())
pipeline = json.loads((checkpoint.parent / "pipeline_manifest.json").read_text())

if selection.get("text_contract") != "search-trigger-v2":
    raise SystemExit("checkpoint is not search-trigger-v2")
if selection.get("acceptance_contract_version") != "search-trigger-v2":
    raise SystemExit("checkpoint does not use search-trigger-v2 acceptance")
if float(selection["coordinate_acceptance_contract"]["distance_tolerance"]) != 0.035:
    raise SystemExit("checkpoint does not use distance tolerance 0.035")
if pipeline.get("generation_profile") != "high-axis-action-v1":
    raise SystemExit("checkpoint is not high-axis-action-v1")
if float(pipeline.get("refinement_minimum_target_axis_1", -1)) != 0.7:
    raise SystemExit("checkpoint does not preserve the 0.700 high-axis minimum")
if pipeline.get("refinement_task_priority") != "descending-axis-1":
    raise SystemExit("checkpoint does not use descending-axis-1 priority")

print(json.dumps({
    "HIGH_AXIS_TEN_NODE_PREFLIGHT": "PASS",
    "checkpoint": str(checkpoint),
    "candidate_count": int(summary["candidate_count"]),
    "selected_count": int(summary["selected_count"]),
    "remaining_target_count": int(summary["refinement_task_count"]),
}, indent=2, sort_keys=True))
PY

wave_id="ten-section-$(date -u +%Y%m%dT%H%M%SZ)"
export AXIS_V2_WAVE_ROOT="${AXIS_V2_WAVE_ROOT:-$GEODML_CACHE_ROOT/runs/readiness-30k-high-axis-v2/$wave_id}"
export READINESS_KEYWORD_SECTION_PLAN="${READINESS_KEYWORD_SECTION_PLAN:-$AXIS_V2_WAVE_ROOT/keyword-section-plan.json}"
export READINESS_TEN_SECTION_RUN_ROOT="${READINESS_TEN_SECTION_RUN_ROOT:-$AXIS_V2_WAVE_ROOT/sections}"
export AXIS_V2_TMUX_PREFIX="${AXIS_V2_TMUX_PREFIX:-axisv2-gen-${wave_id#ten-section-}}"
export AXIS_V2_STATE_FILE="${AXIS_V2_STATE_FILE:-$HOME/geodml-axis-v2-ten-section-latest.env}"

[[ ! -e "$AXIS_V2_WAVE_ROOT" ]]
mkdir -p "$AXIS_V2_WAVE_ROOT/logs" "$READINESS_TEN_SECTION_RUN_ROOT"

python3 "$GEODML_REPOSITORY/analysis/scripts/prepare_readiness_keyword_sections.py" \
    --checkpoint-root "$AXIS_V2_CHECKPOINT" \
    --output "$READINESS_KEYWORD_SECTION_PLAN" \
    --section-count 10 \
    --partition-salt "search-trigger-v2-high-axis-ten-${GEODML_EXPECTED_COMMIT:0:12}"

python3 - "$READINESS_KEYWORD_SECTION_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
tasks = [
    json.loads(line)
    for line in pathlib.Path(plan["source_task_file"]).read_text().splitlines()
    if line.strip()
]
owners = {
    keyword: int(section["section_index"])
    for section in plan["sections"]
    for keyword in section["keyword_ids"]
}
high_by_section = [0] * 10
for row in tasks:
    value = float(row["target"]["normalized_axis_1"])
    if value >= 0.7:
        high_by_section[owners[str(row["keyword_id"])]] += 1
if not all(high_by_section):
    raise SystemExit(f"at least one section has no high-axis work: {high_by_section}")
print(json.dumps({
    "TEN_SECTION_PLAN": "PASS",
    "high_axis_tasks_by_section": high_by_section,
    "high_axis_tasks_total": sum(high_by_section),
}, indent=2, sort_keys=True))
PY

python3 "$GEODML_REPOSITORY/analysis/scripts/prepare_readiness_keyword_sections.py" \
    --verify-plan "$READINESS_KEYWORD_SECTION_PLAN"

export READINESS_TMUX_LAUNCH_MODE="${READINESS_TMUX_LAUNCH_MODE:-automatic}"
[[ "$READINESS_TMUX_LAUNCH_MODE" == "automatic" || \
    "$READINESS_TMUX_LAUNCH_MODE" == "manual" ]] || {
    echo "READINESS_TMUX_LAUNCH_MODE must be automatic or manual" >&2
    exit 2
}

declare -px \
    GEODML_EXPECTED_COMMIT GEODML_PROJECT_ROOT GEODML_CACHE_ROOT \
    GEODML_REPOSITORY AXIS_V2_CHECKPOINT AXIS_V2_WAVE_ROOT \
    READINESS_KEYWORD_SECTION_PLAN READINESS_TEN_SECTION_RUN_ROOT \
    AXIS_V2_TMUX_PREFIX READINESS_APPROVED_WALLTIME \
    READINESS_ALLOCATION_ESTIMATE READINESS_TMUX_LAUNCH_MODE \
    AXIS_V2_STATE_FILE > "$AXIS_V2_STATE_FILE"
chmod 600 "$AXIS_V2_STATE_FILE"

if [[ "$READINESS_TMUX_LAUNCH_MODE" == "manual" ]]; then
    echo "TEN_INTERACTIVE_MANUAL_PREPARE=PASS"
    echo "wave=$AXIS_V2_WAVE_ROOT"
    echo "state=$AXIS_V2_STATE_FILE"
    exit 0
fi

collision_count=0
for index in {0..9}; do
    session="$(printf '%s-%02d' "$AXIS_V2_TMUX_PREFIX" "$index")"
    if tmux has-session -t "$session" 2>/dev/null; then
        printf 'TMUX_COLLISION=%s\n' "$session"
        collision_count=$((collision_count + 1))
    fi
done
[[ "$collision_count" -eq 0 ]] || {
    echo "launch stopped because $collision_count tmux sessions already exist" >&2
    exit 2
}

worker="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_search_trigger_v2_high_axis_section.sh"
[[ -x "$worker" ]]

for index in {0..9}; do
    session="$(printf '%s-%02d' "$AXIS_V2_TMUX_PREFIX" "$index")"
    log="$AXIS_V2_WAVE_ROOT/logs/section-$(printf '%02d' "$index").tmux.log"
    tmux new-session -d -s "$session" \
        "set -o pipefail; env GEODML_EXPECTED_COMMIT='$GEODML_EXPECTED_COMMIT' GEODML_PROJECT_ROOT='$GEODML_PROJECT_ROOT' GEODML_CACHE_ROOT='$GEODML_CACHE_ROOT' GEODML_REPOSITORY='$GEODML_REPOSITORY' READINESS_APPROVED_WALLTIME='$READINESS_APPROVED_WALLTIME' READINESS_ALLOCATION_ESTIMATE='$READINESS_ALLOCATION_ESTIMATE' READINESS_KEYWORD_SECTION_PLAN='$READINESS_KEYWORD_SECTION_PLAN' READINESS_TEN_SECTION_RUN_ROOT='$READINESS_TEN_SECTION_RUN_ROOT' READINESS_WORK_PARTITION_INDEX='$index' salloc --account=scifi --partition=booster --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 --time=05:00:00 --job-name='axv2-s$(printf '%02d' "$index")' bash '$worker' 2>&1 | tee '$log'"
    printf 'INTERACTIVE_NODE_REQUESTED section=%s tmux=%s log=%s\n' \
        "$index" "$session" "$log"
done

echo "TEN_INTERACTIVE_TMUX_LAUNCH=PASS"
echo "wave=$AXIS_V2_WAVE_ROOT"
echo "state=$AXIS_V2_STATE_FILE"
tmux list-sessions -F '#{session_name} #{session_windows} #{session_created_string}' |
    awk -v prefix="$AXIS_V2_TMUX_PREFIX-" 'index($1, prefix) == 1'
squeue --me --format='%.18i %.24j %.9T %.10M %.10l %R'
