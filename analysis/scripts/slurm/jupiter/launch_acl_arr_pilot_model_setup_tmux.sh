#!/usr/bin/env bash
# Launch reconnect-safe pilot setup and model downloads on the login node.

set -euo pipefail

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; prepare the pilot environment first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"
SESSION="${ACL_ARR_MODEL_SETUP_SESSION:-acl-arr-model-${GEODML_EXPECTED_COMMIT:0:7}}"

WORKER="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_acl_arr_pilot_model_setup.sh"
LOG="$ACL_ARR_RUN_ROOT/model-download.log"
MARKER="$ACL_ARR_RUN_ROOT/model-downloads.complete"
[[ -x "$WORKER" ]]

if [[ -s "$MARKER" ]]; then
    echo "MODEL_DOWNLOADS=ALREADY_COMPLETE"
elif tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "MODEL_DOWNLOADS=RUNNING session=$SESSION"
else
    tmux new-session -d -s "$SESSION" \
        "set -o pipefail; ACL_ARR_ENVIRONMENT_FILE='$ENVIRONMENT_FILE' bash -l '$WORKER' 2>&1 | tee '$LOG'"
    echo "MODEL_DOWNLOADS=STARTED session=$SESSION"
fi

echo "session=$SESSION"
echo "log=$LOG"
echo "run_root=$ACL_ARR_RUN_ROOT"
tmux list-sessions -F '#{session_name} #{session_windows} #{session_created_string}' |
    awk -v name="$SESSION" '$1 == name'
