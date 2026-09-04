#!/usr/bin/env bash
# Launch the specifically approved four-GH200 pilot allocation under tmux.

set -euo pipefail

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; prepare the pilot environment first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"
SESSION="${ACL_ARR_PILOT_SESSION:-acl-arr-pilot-${GEODML_EXPECTED_COMMIT:0:7}}"

CURRENT_COMMIT="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
[[ "$CURRENT_COMMIT" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -s "$ACL_ARR_RUN_ROOT/model-downloads.complete" ]]
[[ -s "$ACL_ARR_RUN_ROOT/plan/run_manifest.json" ]]
[[ -x "$ACL_ARR_VENV/bin/vllm" ]]

WORKER="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_acl_arr_document_pilot_4gpu.sh"
MARKER="$ACL_ARR_RUN_ROOT/allocation-requested-once.txt"
LOG="$ACL_ARR_RUN_ROOT/controller.log"
[[ -x "$WORKER" ]]

if [[ -e "$MARKER" ]]; then
    echo "REFUSING_SECOND_ALLOCATION_REQUEST=$MARKER" >&2
    echo "A retry or resume requires a new runtime estimate and wall-time approval." >&2
    exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session already exists: $SESSION" >&2
    exit 2
fi

printf 'approved_walltime=03:00:00\nrequested_at=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARKER"

tmux new-session -d -s "$SESSION" \
    "set -o pipefail; source '$ENVIRONMENT_FILE'; export ACL_ARR_APPROVED_WALLTIME='03:00:00'; export ACL_ARR_ALLOCATION_ESTIMATE='128 prompts; 4608 requests; estimated 1.5-2.5h; 3h approved with loading margin; maximum 12 GH200 GPU-hours'; export ACL_ARR_MAX_CONCURRENCY='16'; export ACL_ARR_MAX_MODEL_LEN='32768'; export ACL_ARR_TENSOR_PARALLEL_SIZE='4'; export ACL_ARR_GPU_MEMORY_UTILIZATION='0.90'; salloc --account=scifi --partition=booster --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=512G --gres=gpu:4 --time=03:00:00 --job-name=acl-arr-pilot srun --exact --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:4 --cpu-bind=none bash '$WORKER' 2>&1 | tee '$LOG'"

echo "ACL_ARR_PILOT_LAUNCH=PASS"
echo "session=$SESSION"
echo "log=$LOG"
echo "run_root=$ACL_ARR_RUN_ROOT"
tmux list-sessions -F '#{session_name} #{session_windows} #{session_created_string}' |
    awk -v name="$SESSION" '$1 == name'
squeue --me --name=acl-arr-pilot \
    --format='%.18i %.24j %.10T %.10M %.10l %.6D %.6C %R'
