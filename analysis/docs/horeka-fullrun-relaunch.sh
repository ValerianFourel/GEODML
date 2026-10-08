#!/bin/bash
# What went wrong, and relaunch only what is needed (login node). Without options it only reports: failed tasks with the
# tail of their latest log, checkpoints, the last allocations and their exit codes. Options:
#   --repin NEWCODE   after a fix: NEWCODE is a clean checkout of the fixed commit; every unfinished task is pointed at it
#                     (done tasks keep their outputs and commits; refused while any task is claimed)
#   --retry-all       give every failed task two fresh attempts (their failure records move to failed-archive/)
#   --retry ID        the same for one task (repeatable)
#   --launch          then start the launcher in the background (same caps, counted across relaunches)
# Usage: CODE=<checkout the ledger uses now> LEDGER=<ledger> bash horeka-fullrun-relaunch.sh [options]
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?}; LEDGER=${LEDGER:?}; FR=$(dirname "$LEDGER"); mkdir -p "$FR/launch"
NEW=""; RETRY=(); LAUNCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repin) NEW=$2; shift 2 ;;
    --retry-all) RETRY=(all); shift ;;
    --retry) RETRY+=(--task "$2"); shift 2 ;;
    --launch) LAUNCH=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
fr() { (cd "$CODE" && "$RT/bin/python" -m analysis.fullrun "$@"); }
fr reconcile --ledger "$LEDGER"
fr status --ledger "$LEDGER" > "$FR/launch/status.json"
"$RT/bin/python" - "$FR/launch/status.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("states", s["states"], "| ready cpu", s["ready_cpu"], "gpu", s["ready_gpu"], "| running", s["running"])
print("failed:", " ".join(s["failed_tasks"]) or "none")
PY
for f in "$LEDGER"/failed/*.jsonl; do
  [ -e "$f" ] || continue; t=$(basename "$f" .jsonl); lg=$(ls -t "$LEDGER/logs/$t".*.log 2>/dev/null | head -n 1)
  echo "== $t: $(wc -l < "$f") failure(s); last record $(tail -n 1 "$f")"; [ -n "$lg" ] && tail -n 15 "$lg" | sed 's/^/    | /'
done
ls "$LEDGER/checkpoint" 2>/dev/null | sed 's/^/checkpoint: /'
JOBS=$(cut -f1 "$FR/launch/submissions.tsv" 2>/dev/null | paste -sd, - || true)
[ -n "$JOBS" ] && { sacct -X -j "$JOBS" -o JobID,JobName%24,State,Elapsed,ExitCode || true; }
if [ -n "$NEW" ]; then
  test -z "$(git -C "$NEW" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $NEW" >&2; exit 2; }
  fr repin --ledger "$LEDGER" --old-code "$CODE" --new-code "$NEW"; CODE=$NEW
  echo "now CODE=$CODE ($(git -C "$CODE" rev-parse HEAD))"
fi
if [ "${RETRY[0]:-}" = all ]; then fr retry --ledger "$LEDGER"; elif [ ${#RETRY[@]} -gt 0 ]; then fr retry --ledger "$LEDGER" "${RETRY[@]}"; fi
if [ "$LAUNCH" = 1 ]; then
  if pgrep -u "$USER" -f horeka-fullrun-launch.sh > /dev/null; then echo "a launcher already runs: not starting a second one"; exit 0; fi
  L=$FR/launch/launch-$(date +%Y%m%d-%H%M%S).log
  CODE=$CODE LEDGER=$LEDGER nohup bash "$CODE/analysis/docs/horeka-fullrun-launch.sh" > "$L" 2>&1 &
  echo "launcher started (pid $!), log $L"
fi
