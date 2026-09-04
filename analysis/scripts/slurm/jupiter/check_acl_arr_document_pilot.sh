#!/usr/bin/env bash
# Print Slurm state, logs, result counts, and final pilot artifacts.

set -u

ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
[[ -s "$ENVIRONMENT_FILE" ]] || {
    echo "ERROR: missing $ENVIRONMENT_FILE; prepare the pilot environment first" >&2
    exit 2
}
source "$ENVIRONMENT_FILE"

echo "===== SLURM QUEUE ====="
squeue --me --name=acl-arr-pilot \
    --format='%.18i %.24j %.10T %.10M %.10l %.6D %.6C %R'

echo
echo "===== SLURM ACCOUNTING ====="
sacct --name=acl-arr-pilot --starttime today \
    --format=JobID,JobName%24,State,Elapsed,Timelimit,AllocTRES%55,MaxRSS,ExitCode -X

echo
echo "===== CONTROLLER ====="
tail -n 160 "$ACL_ARR_RUN_ROOT/controller.log" 2>/dev/null || true

echo
echo "===== ACTIVE VLLM LOG ====="
LATEST_VLLM_LOG="$(
    find "$ACL_ARR_RUN_ROOT/logs" -maxdepth 1 -type f -name 'vllm-*.log' \
        -print 2>/dev/null | sort | tail -n 1
)"
if [[ -n "$LATEST_VLLM_LOG" ]]; then
    echo "log=$LATEST_VLLM_LOG"
    tail -n 100 "$LATEST_VLLM_LOG"
else
    echo "No vLLM log yet."
fi

echo
echo "===== OUTCOME COUNTS ====="
find "$ACL_ARR_RUN_ROOT/results" -type f -name outcomes.jsonl -print \
    2>/dev/null | sort |
while IFS= read -r PATHNAME; do
    printf '%8s  %s\n' "$(wc -l < "$PATHNAME")" "$PATHNAME"
done

echo
echo "===== FAILURE COUNTS ====="
find "$ACL_ARR_RUN_ROOT/results" -type f -name failures.jsonl -print \
    2>/dev/null | sort |
while IFS= read -r PATHNAME; do
    printf '%8s  %s\n' "$(wc -l < "$PATHNAME")" "$PATHNAME"
done

echo
echo "===== FINAL ARTIFACTS ====="
if [[ -s "$ACL_ARR_RUN_ROOT/pilot-runtime-manifest.json" ]]; then
    python3 -m json.tool "$ACL_ARR_RUN_ROOT/pilot-runtime-manifest.json"
else
    echo "pilot-runtime-manifest.json is not available yet."
fi
if [[ -d "$ACL_ARR_RUN_ROOT/analysis" ]]; then
    find "$ACL_ARR_RUN_ROOT/analysis" -maxdepth 2 -type f -print | sort
fi
