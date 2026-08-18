#!/bin/bash
# Read-only status audit for a resumable 20k abstention-v2 run.

set -euo pipefail

: "${GEODML_RUNS_ROOT:?source geodml_setup.sh first}"

code_commit="${READINESS_20K_CODE_COMMIT:?READINESS_20K_CODE_COMMIT is required}"
run_root="${READINESS_20K_ROOT:-$GEODML_RUNS_ROOT/semantic-readiness-20k-abstention/$code_commit-combined-atleast20000-four-judge-v2}"
queue_root="$run_root/judge-queue"
job_id_path="$run_root/latest-slurm-job-id.txt"
launch_environment="$run_root/launch-environment.txt"

for path in "$job_id_path" "$launch_environment"; do
    if [[ ! -s "$path" ]]; then
        echo "missing v2 run artifact: $path" >&2
        exit 2
    fi
done

job_id="$(<"$job_id_path")"
tasks="$(sed -n 's/^export READINESS_20K_TASKS=//p' "$launch_environment")"
expected="$(sed -n 's/^export READINESS_20K_TASKS_PER_SLOT=//p' "$launch_environment")"

echo "===== EXPERIMENT ====="
echo "JOB_ID=$job_id"
echo "PROMPTS=$expected"
echo "REQUESTED_JUDGMENTS=$((4 * expected))"
echo "ROOT=$run_root"

echo
echo "===== SLURM STATUS ====="
squeue -j "$job_id" 2>/dev/null || true
sacct -j "$job_id" --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList

echo
echo "===== QUEUE LOG ====="
tail -n 100 "$queue_root/queue.log" 2>/dev/null || echo "No queue log yet."

echo
echo "===== PERSISTED CACHE COUNTS ====="
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
echo "===== V2 ANSWER TYPES ====="
python3 - "$tasks" "$queue_root" <<'PY'
import collections
import json
import pathlib
import sys

task_path = pathlib.Path(sys.argv[1])
queue_root = pathlib.Path(sys.argv[2])
task_ids = {
    json.loads(line)["task_id"]
    for line in task_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}

def object_from_text(raw):
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object")

full_root = queue_root / "full"
if not full_root.exists():
    print("No full-stage caches yet.")
    raise SystemExit(0)

for output in sorted(full_root.iterdir()):
    cache_root = output / "task_cache"
    if not cache_root.is_dir():
        continue
    counts = collections.Counter()
    rejected = 0
    invalid = 0
    for path in cache_root.glob("*.json"):
        if path.name.endswith(".failed.json"):
            continue
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["task_id"] not in task_ids:
                raise ValueError("unknown task")
            payload = object_from_text(str(cached["raw_response"]))
            answer_type = payload["answer_type"]
            if answer_type not in {"rating", "not_applicable", "dont_know"}:
                raise ValueError("unknown answer type")
            counts[answer_type] += 1
            rejected += len(cached.get("rejected_attempts", []))
        except Exception:
            invalid += 1
    print(
        f"{output.name}: total={sum(counts.values())} "
        f"rating={counts['rating']} "
        f"not_applicable={counts['not_applicable']} "
        f"dont_know={counts['dont_know']} "
        f"rejected_attempts={rejected} invalid_caches={invalid}"
    )
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
if [[ -s "$queue_root/queue.complete" ]]; then
    cat "$queue_root/queue.complete"
else
    echo "The full four-model v2 experiment is not complete yet."
fi
