#!/usr/bin/env bash
# Print model-download progress and verify the finished four-model panel.

set -euo pipefail

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; run the environment stage first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"
SESSION="${ACL_ARR_MODEL_SETUP_SESSION:-acl-arr-model-${GEODML_EXPECTED_COMMIT:0:7}}"

LOG="$ACL_ARR_RUN_ROOT/model-download.log"
MARKER="$ACL_ARR_RUN_ROOT/model-downloads.complete"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "MODEL_SETUP_STATE=RUNNING"
else
    echo "MODEL_SETUP_STATE=NOT_RUNNING"
fi
echo "log=$LOG"
tail -n 160 "$LOG" 2>/dev/null || true

if [[ ! -s "$MARKER" ]]; then
    echo "MODEL_SETUP_READY=NO"
    echo "Wait for MODEL_DOWNLOADS=PASS. If the session stopped, rerun the model-setup launcher." >&2
    exit 3
fi

source "$ACL_ARR_VENV/bin/activate"
python3 "$GEODML_REPOSITORY/analysis/scripts/download_acl_arr_pilot_models.py" \
    --run-root "$ACL_ARR_RUN_ROOT" \
    --verify-only
python3 -c 'import aiohttp, huggingface_hub, vllm; print("vllm", vllm.__version__)'
du -sh "$HF_HUB_CACHE"
echo "MODEL_SETUP_READY=YES"
