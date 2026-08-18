#!/bin/bash
# Read-only progress and throughput audit for the last four-judge incremental run.

set -euo pipefail

: "${GEODML_RUNS_ROOT:?source geodml_setup.sh first}"

input_commit="${READINESS_INCREMENTAL_INPUT_COMMIT:-40527b0fbc1fddb9f3a377f70e1405effd4ebc19}"
run_root="${READINESS_INCREMENTAL_ROOT:-$GEODML_RUNS_ROOT/semantic-readiness-incremental/$input_commit-six-source-atleast10000-four-judge}"
queue_root="$run_root/judge-queue"
job_id_path="$run_root/slurm-job-id.txt"

for path in "$job_id_path" "$run_root/launch-environment.txt" "$queue_root/queue.log"; do
    if [[ ! -s "$path" ]]; then
        echo "missing last-run artifact: $path" >&2
        exit 2
    fi
done
job_id="$(<"$job_id_path")"

echo "===== LAST JOB ====="
echo "JOB_ID=$job_id"
echo "RUN_ROOT=$run_root"
sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList
squeue -j "$job_id" 2>/dev/null || true

echo
echo "===== QUEUE LOG ====="
tail -n 100 "$queue_root/queue.log"

echo
echo "===== TASK AND CACHE COUNTS ====="
grep -E '^export READINESS_INCREMENTAL_TASKS_PER_SLOT=' "$run_root/launch-environment.txt"
for stage in smoke full; do
    for output in "$queue_root/$stage"/*; do
        [[ -d "$output" ]] || continue
        successful=0
        failed=0
        if [[ -d "$output/task_cache" ]]; then
            successful="$(find "$output/task_cache" -maxdepth 1 -type f -name '*.json' ! -name '*.failed.json' | wc -l)"
            failed="$(find "$output/task_cache" -maxdepth 1 -type f -name '*.failed.json' | wc -l)"
        fi
        printf '%-5s %-38s successful=%-7s failed=%s\n' \
            "$stage" "$(basename "$output")" "$successful" "$failed"
    done
done

echo
echo "===== OBSERVED STAGE DURATIONS ====="
python3 - "$queue_root/queue.log" <<'PY'
from datetime import datetime
import pathlib
import re
import sys

pattern = re.compile(r"^\[([^]]+)] (starting|finished) (\S+)(?: exit=\d+)?$")
starts = {}
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    match = pattern.match(line)
    if not match:
        continue
    timestamp = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    action, stage = match.group(2), match.group(3)
    if action == "starting":
        starts[stage] = timestamp
    elif stage in starts:
        seconds = (timestamp - starts[stage]).total_seconds()
        print(f"{stage:50s} {seconds / 60:8.2f} minutes")
PY

echo
echo "===== ERROR SCAN ====="
if grep -R -n -E 'Traceback|OutOfMemory|CUDA out of memory|FAILED|TIMEOUT|CANCELLED' \
    "$queue_root/logs" "$run_root/slurm" 2>/dev/null; then
    echo "Review the matching lines above."
else
    echo "No matching fatal errors found."
fi

echo
echo "===== COMPLETION ====="
if grep -F 'queue complete:' "$queue_root/queue.log"; then
    echo "LAST RUN COMPLETE"
else
    echo "LAST RUN INCOMPLETE"
fi
