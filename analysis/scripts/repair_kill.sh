#!/usr/bin/env bash
# Emergency teardown for the repair pipeline. Stops the resubmission loop and
# scancels every GEODML job in the queue so nothing can re-post itself —
# including pending `afterany` chain successors left over from before the
# single-authority patch (those carry the sbatch default job-name geodml-*).
#
# Usage:
#   ./scripts/repair_kill.sh             # show what will die, then ask to confirm
#   ./scripts/repair_kill.sh --yes       # non-interactive: just do it
#   ./scripts/repair_kill.sh --dry-run   # show only, cancel nothing
#   ./scripts/repair_kill.sh --user foo  # target a different SLURM user (default $USER)
#
# What it does, in order:
#   1. Stops any running scripts/repair_loop.sh (SIGTERM — it exits cleanly
#      after the current cycle) and clears its lock file.
#   2. scancels every queued job of yours whose name matches the GEODML stages.
#   3. Leaves the manifest intact; re-audit with repair_dispatch.py --status.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY=0
YES=0
USER_NAME="${USER:-$(whoami)}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY=1; shift;;
    --yes|-y)   YES=1; shift;;
    --user)     USER_NAME="$2"; shift 2;;
    -h|--help)  sed -n '1,22p' "$0"; exit 0;;
    *) echo "[repair_kill] unknown flag: $1"; exit 2;;
  esac
done

# Job-name prefixes used across the pipeline (dispatch_all / dispatch_bcd /
# repair_dispatch job-names, plus the sbatch #SBATCH --job-name defaults that a
# chained successor inherits: geodml-rerank, geodml-abl, ...). Note both
# order_probe spellings (op- from repair_dispatch, ord- from dispatch_all) and
# both features spellings (feat- from dispatch_bcd, features- from dispatch_all).
NAME_RE='^(geodml-|rerank-|op-|ord-|prob-|abl-|sal-|wgt-|dml-|feat-|features-)'

echo "== 1. repair_loop processes =="
LOOP_PIDS="$(pgrep -f 'repair_loop\.sh' 2>/dev/null || true)"
if [ -n "$LOOP_PIDS" ]; then
  # shellcheck disable=SC2086
  ps -o pid=,etime=,args= -p $LOOP_PIDS 2>/dev/null | sed 's/^/  /' || echo "  $LOOP_PIDS"
else
  echo "  (none running)"
fi

echo "== 2. your GEODML SLURM jobs (user=$USER_NAME) =="
if ! command -v squeue >/dev/null 2>&1; then
  echo "  [warn] squeue not found — are you on a login node? Skipping job scan."
  JOBS=""
else
  JOBS="$(squeue -u "$USER_NAME" -h -o '%i|%j|%T' 2>/dev/null \
            | awk -F'|' -v re="$NAME_RE" '$2 ~ re {print}')"
fi
if [ -n "$JOBS" ]; then
  printf '%s\n' "$JOBS" | awk -F'|' '{printf "  %-12s %-40s %s\n", $1, $2, $3}'
else
  echo "  (none in queue)"
fi
JOBIDS="$(printf '%s\n' "$JOBS" | awk -F'|' 'NF{print $1}' | tr '\n' ' ')"
NJOBS="$(printf '%s' "$JOBIDS" | wc -w | tr -d ' ')"

echo
echo "Plan: stop loop pids [${LOOP_PIDS:-none}] and scancel $NJOBS job(s)."

if [ "$DRY" = "1" ]; then
  echo "[dry-run] nothing changed."
  exit 0
fi

if [ "$YES" != "1" ]; then
  printf "Proceed with teardown? [y/N] "
  read -r ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "aborted."; exit 0;;
  esac
fi

# 1. stop the loop (TERM → exits after current cycle) and clear its lock.
if [ -n "$LOOP_PIDS" ]; then
  # shellcheck disable=SC2086
  kill -TERM $LOOP_PIDS 2>/dev/null && echo "[killed] repair_loop pids: $LOOP_PIDS"
fi
rm -f logs/.repair_loop.lock logs/.repair_loop.lock.pid 2>/dev/null || true

# 2. cancel jobs by id (covers pending afterany successors too).
if [ -n "${JOBIDS// /}" ]; then
  # shellcheck disable=SC2086
  scancel $JOBIDS && echo "[scancel] cancelled: $JOBIDS"
else
  echo "[scancel] no matching jobs to cancel."
fi

echo
echo "[repair_kill] done. The manifest is untouched. Re-check state with:"
echo "  set -a; source .env; set +a"
echo "  .venv/bin/python scripts/repair_dispatch.py --status"
