#!/usr/bin/env bash
# Resumable repair loop. Runs `repair_audit.py` + `repair_dispatch.py` on a
# timer, re-submitting cells with gap > 0 until the manifest reports nothing
# left to do. Designed to live on the JUPITER login node (or any host that
# can `sbatch`).
#
# Safe to Ctrl-C and re-run — each cycle is idempotent, the manifest is the
# source of truth, and a flock guarantees only one loop at a time.
#
# Usage:
#   ./scripts/repair_loop.sh                   # default: 30 min between cycles
#   ./scripts/repair_loop.sh --interval 1200   # custom sleep (seconds)
#   ./scripts/repair_loop.sh --max-cycles 20   # cap; default: until done
#   ./scripts/repair_loop.sh --stage rerank    # only one stage
#   ./scripts/repair_loop.sh --max-submissions 16
#   ./scripts/repair_loop.sh --with-downstream # once Stage A done, kick off
#                                              #   dispatch_bcd.sh --with-stage-f
#   ./scripts/repair_loop.sh --dry-run         # print plan, submit nothing
#
# Logs to logs/repair_loop_<date>.log (rotated daily). Re-running picks up.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── flags ───────────────────────────────────────────────────────────────────
INTERVAL=1800           # seconds between cycles (default 30 min)
MAX_CYCLES=0            # 0 = unbounded
STAGE_FILTER=""         # "" = all stages
MAX_SUBMISSIONS=32
DRY_RUN=""
WITH_DOWNSTREAM=0
DOWNSTREAM_LAUNCHED=0   # tracks one-shot trigger of dispatch_bcd

while [ $# -gt 0 ]; do
  case "$1" in
    --interval)         INTERVAL="$2"; shift 2;;
    --max-cycles)       MAX_CYCLES="$2"; shift 2;;
    --stage)            STAGE_FILTER="$2"; shift 2;;
    --max-submissions)  MAX_SUBMISSIONS="$2"; shift 2;;
    --dry-run)          DRY_RUN="--dry-run"; shift;;
    --with-downstream)  WITH_DOWNSTREAM=1; shift;;
    -h|--help)          sed -n '1,30p' "$0"; exit 0;;
    *) echo "[repair_loop] unknown flag: $1"; exit 2;;
  esac
done

if [ ! -d .venv ]; then
  echo "[repair_loop] no .venv at $(pwd) — bootstrap with python -m venv .venv first"
  exit 2
fi
PY=.venv/bin/python

mkdir -p logs manifests
LOG="logs/repair_loop_$(date +%Y-%m-%d).log"
LOCK="logs/.repair_loop.lock"

# ── single-instance lock ────────────────────────────────────────────────────
exec 9>"$LOCK"
if ! command -v flock >/dev/null 2>&1; then
  # macOS may not have flock; fall back to a stale-pid check.
  if [ -f "$LOCK.pid" ] && kill -0 "$(cat "$LOCK.pid")" 2>/dev/null; then
    echo "[repair_loop] another instance running (pid $(cat "$LOCK.pid"))"
    exit 1
  fi
  echo $$ > "$LOCK.pid"
  trap 'rm -f "$LOCK.pid"' EXIT
else
  if ! flock -n 9; then
    echo "[repair_loop] another instance is holding the lock — exiting"
    exit 1
  fi
fi

# ── ctrl-c handling ─────────────────────────────────────────────────────────
INTERRUPTED=0
on_signal() { INTERRUPTED=1; echo; echo "[repair_loop] caught signal — exiting after current cycle"; }
trap on_signal INT TERM

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$LOG"
}

run_one_cycle() {
  local cycle="$1"
  log "─── cycle $cycle (max=${MAX_CYCLES:-∞}) ───"

  # 1. audit
  log "audit …"
  if [ -n "$STAGE_FILTER" ]; then
    "$PY" scripts/repair_audit.py --stage "$STAGE_FILTER" >>"$LOG" 2>&1 || {
      log "audit FAILED — see $LOG. Sleeping then retrying."
      return 1
    }
  else
    "$PY" scripts/repair_audit.py >>"$LOG" 2>&1 || {
      log "audit FAILED — see $LOG. Sleeping then retrying."
      return 1
    }
  fi

  # 2. dispatch
  log "dispatch (max-submissions=$MAX_SUBMISSIONS) …"
  set +u
  local extra_stage=""
  [ -n "$STAGE_FILTER" ] && extra_stage="--stage $STAGE_FILTER"
  "$PY" scripts/repair_dispatch.py \
    --max-submissions "$MAX_SUBMISSIONS" \
    $extra_stage \
    $DRY_RUN >>"$LOG" 2>&1
  local disp_rc=$?
  set -u
  if [ "$disp_rc" -ne 0 ]; then
    log "dispatch FAILED (rc=$disp_rc) — see $LOG. Sleeping then retrying."
    return 1
  fi

  # 3. compact rollup → stdout AND log
  local rollup
  rollup="$("$PY" - <<'PY' 2>/dev/null
import pyarrow.parquet as pq, pandas as pd
from pathlib import Path
m = Path("manifests/repair_manifest.parquet")
if not m.exists():
    print("(no manifest)"); raise SystemExit
df = pq.read_table(m).to_pandas()
for s in sorted(df["stage"].unique()):
    sub = df[df["stage"] == s]
    open_ = int((sub["gap"] > 0).sum())
    done  = len(sub) - open_
    inflight = int((sub["status"].isin(["SUBMITTED","RUNNING"])).sum())
    gap = int(sub["gap"].sum())
    print(f"  {s:12s} done={done}/{len(sub)}  open={open_}  in_flight={inflight}  kw_gap={gap:,}")
PY
)"
  echo "$rollup" | tee -a "$LOG"

  # 4. optional downstream kickoff (once)
  if [ "$WITH_DOWNSTREAM" -eq 1 ] && [ "$DOWNSTREAM_LAUNCHED" -eq 0 ]; then
    local stage_a_open
    stage_a_open="$("$PY" - <<'PY' 2>/dev/null
import pyarrow.parquet as pq
df = pq.read_table("manifests/repair_manifest.parquet").to_pandas()
sub = df[df["stage"].isin(["rerank", "order_probe"])]
print(int((sub["gap"] > 0).sum()))
PY
)"
    if [ "${stage_a_open:-1}" = "0" ] && [ -x scripts/slurm/dispatch_bcd.sh ]; then
      log "Stage A/A' all done — launching dispatch_bcd.sh --with-stage-f"
      if [ -n "$DRY_RUN" ]; then
        log "[DRY] ./scripts/slurm/dispatch_bcd.sh --with-stage-f"
      else
        ./scripts/slurm/dispatch_bcd.sh --with-stage-f >>"$LOG" 2>&1 || {
          log "dispatch_bcd.sh exited non-zero — see $LOG"
        }
      fi
      DOWNSTREAM_LAUNCHED=1
    fi
  fi

  # 5. report (overwrites docs/repair_report_<date>.md)
  "$PY" scripts/repair_report.py >>"$LOG" 2>&1 || true

  # 6. done?
  local total_gap
  total_gap="$("$PY" - <<'PY' 2>/dev/null
import pyarrow.parquet as pq
df = pq.read_table("manifests/repair_manifest.parquet").to_pandas()
mask = (df["gap"] > 0)
print(int(mask.sum()))
PY
)"
  if [ "${total_gap:-1}" = "0" ]; then
    log "All cells at gap=0. Exiting loop."
    return 99
  fi
  return 0
}

# ── main loop ───────────────────────────────────────────────────────────────
log "==============================================================="
log "repair_loop starting"
log "  interval=$INTERVAL s   max-cycles=${MAX_CYCLES:-unbounded}"
log "  stage_filter=${STAGE_FILTER:-all}"
log "  max-submissions=$MAX_SUBMISSIONS"
log "  with_downstream=$WITH_DOWNSTREAM"
log "  dry_run=${DRY_RUN:-no}"
log "  log=$LOG"
log "==============================================================="

cycle=0
while :; do
  cycle=$((cycle + 1))
  set +e
  run_one_cycle "$cycle"
  rc=$?
  set -e

  if [ "$rc" = "99" ]; then
    log "DONE — all gaps closed."
    exit 0
  fi

  if [ "$INTERRUPTED" = "1" ]; then
    log "exiting (interrupted)"
    exit 0
  fi

  if [ "$MAX_CYCLES" -gt 0 ] && [ "$cycle" -ge "$MAX_CYCLES" ]; then
    log "hit --max-cycles=$MAX_CYCLES; stopping"
    exit 0
  fi

  log "sleeping ${INTERVAL}s before next cycle …"
  # Sleep in short increments so signal handling is responsive.
  remaining="$INTERVAL"
  while [ "$remaining" -gt 0 ] && [ "$INTERRUPTED" = "0" ]; do
    sleep 15
    remaining=$((remaining - 15))
  done
done
