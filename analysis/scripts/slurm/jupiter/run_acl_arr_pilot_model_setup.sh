#!/usr/bin/env bash
# Install the pilot runtime and download each pinned model snapshot.

set -euo pipefail
trap 'echo "MODEL_SETUP_FAILED line=$LINENO status=$?" >&2' ERR

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; prepare the pilot environment first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"

CURRENT_COMMIT="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
[[ "$CURRENT_COMMIT" == "$GEODML_EXPECTED_COMMIT" ]] || {
    echo "ERROR: checkout $CURRENT_COMMIT does not match $GEODML_EXPECTED_COMMIT" >&2
    exit 2
}

module load Stages/2026 GCC Python CUDA
if [[ ! -x "$ACL_ARR_VENV/bin/python" ]]; then
    python3 -m venv "$ACL_ARR_VENV"
fi
source "$ACL_ARR_VENV/bin/activate"
python3 -m pip install --upgrade pip wheel
python3 -m pip install --upgrade -r "$GEODML_REPOSITORY/analysis/requirements.txt" vllm
hf auth whoami
df -h "$GEODML_CACHE_ROOT"

python3 "$GEODML_REPOSITORY/analysis/scripts/download_acl_arr_pilot_models.py" \
    --run-root "$ACL_ARR_RUN_ROOT" \
    --model-template "$GEODML_REPOSITORY/analysis/config/acl_arr_model_panel.template.json"
python3 -c 'import aiohttp, huggingface_hub, vllm; print("vllm", vllm.__version__)'
du -sh "$HF_HUB_CACHE"

TEMPORARY_MARKER="$ACL_ARR_RUN_ROOT/model-downloads.complete.tmp"
printf 'completed_at=%s\ncommit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GEODML_EXPECTED_COMMIT" \
    > "$TEMPORARY_MARKER"
mv "$TEMPORARY_MARKER" "$ACL_ARR_RUN_ROOT/model-downloads.complete"
echo "MODEL_DOWNLOADS=PASS"
