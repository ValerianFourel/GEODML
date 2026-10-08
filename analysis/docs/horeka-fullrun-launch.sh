#!/bin/bash
# Finite launcher of the whole full run (login node; it computes nothing itself). Every POLL seconds it reconciles the
# ledger, reports new failures with the tail of their logs, and admits at most one allocation when all checks pass:
#   - no full-run job of ours is still PENDING (so starts can be observed one at a time);
#   - fewer than MAX_ALLOC (5) RUNNING allocations of $USER, all kinds counted (Gemma bouts and sallocs too); pending jobs
#     hold no resources and are not counted (a queue of pending Gemma bouts would otherwise block the run forever);
#   - the latest observed start of any running job of $USER is at least GAP_MIN (10) minutes ago;
#   - storage: at least MIN_FREE_GB free and MIN_FREE_INODES inodes;
#   - GPU first: ready GPU tasks, fewer than GPU_MAX (1) GPU jobs of ours, fewer than GPU_CAP (3) GPU submissions so far,
#     wall time GPU_TIME (03:00:00); else CPU: ready CPU tasks (another node only when none of ours runs or at least
#     MIN_READY_CORES cores of ready work wait), fewer than CPU_MAX (4) CPU jobs, fewer than CPU_CAP (8) CPU submissions.
# FILL=1 (Valerian, 2026-10-08: "send it all 10 mins and fill the queue"): instead of the first and third checks, submit
# whenever GAP_MIN minutes have passed since the previous submission, while running allocations of $USER (all kinds) plus
# pending full-run jobs stay below MAX_ALLOC. Starts are then no longer observed one at a time (an explicit override).
# Caps count every submission in $FR/launch/submissions.tsv, across relaunches (raising them needs Valerian's approval).
# Never cancels, extends or requeues anything. Stops with
#   0 every task done | 3 blocked: failed tasks hold the rest (see horeka-fullrun-relaunch.sh) | 5 caps used with work left
#   2 storage below the threshold.
# Usage (login shell): CODE=<pinned checkout> LEDGER=<ledger> nohup bash horeka-fullrun-launch.sh > $FR/launch/launch.log 2>&1 &
set -uo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}; LEDGER=${LEDGER:?LEDGER must point to the planned ledger}
FR=$(dirname "$LEDGER"); LAUNCH=$FR/launch; mkdir -p "$LAUNCH" "$FR/slurm"
SUBS=$LAUNCH/submissions.tsv; SEEN=$LAUNCH/reported-failures.txt; touch "$SUBS" "$SEEN"
CPU_CAP=${CPU_CAP:-8}; CPU_MAX=${CPU_MAX:-4}; GPU_CAP=${GPU_CAP:-3}; GPU_MAX=${GPU_MAX:-1}; GPU_TIME=${GPU_TIME:-03:00:00}
MAX_ALLOC=${MAX_ALLOC:-5}; GAP_MIN=${GAP_MIN:-10}; POLL=${POLL:-300}; MIN_READY_CORES=${MIN_READY_CORES:-16}
FILL=${FILL:-0}; GAP_SEC=${GAP_SEC:-$((60 * GAP_MIN))}
MIN_FREE_GB=${MIN_FREE_GB:-100}; MIN_FREE_INODES=${MIN_FREE_INODES:-2000000}
[ "$CPU_MAX" -le 4 ] && [ "$GPU_MAX" -le 1 ] && [ "$MAX_ALLOC" -le 5 ] || { echo "concurrency above the approved limits" >&2; exit 2; }
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
grep -q -- "$CODE" "$LEDGER/tasks.jsonl" || { echo "the ledger's tasks do not use $CODE (repin first: horeka-fullrun-relaunch.sh)" >&2; exit 2; }
CPU_NAME=geodml-fullrun-cpu; GPU_NAME=geodml-fullrun-gpustats
fr() { (cd "$CODE" && "$RT/bin/python" -m analysis.fullrun "$@"); }
field() { "$RT/bin/python" -c "import json,sys; s=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, s))" "$LAUNCH/status.json" "$1"; }
log() { echo "$(date -u +%FT%TZ) $*"; }
log "launcher: fill $FILL gap ${GAP_SEC}s; code $CODE commit $(git -C "$CODE" rev-parse HEAD) ledger $LEDGER caps cpu $CPU_CAP gpu $GPU_CAP gpu-time $GPU_TIME"
while true; do
  fr reconcile --ledger "$LEDGER" > /dev/null
  fr status --ledger "$LEDGER" > "$LAUNCH/status.json.tmp" && mv "$LAUNCH/status.json.tmp" "$LAUNCH/status.json"
  total=$(field tasks); done_n=$(field "states.get('done',0)"); ready_cpu=$(field ready_cpu); ready_gpu=$(field ready_gpu)
  ready_cores=$(field ready_cpu_cores); open_n=$(field "open_cpu+open_gpu"); running_n=$(field "len(running)")
  free_gb=$(field "int(disk_free_gb)"); inodes=$(field inodes_free)
  for t in $(field "' '.join(failed_tasks)"); do
    grep -qx "$t" "$SEEN" && continue
    echo "$t" >> "$SEEN"; lg=$(ls -t "$LEDGER/logs/$t".*.log 2>/dev/null | head -n 1)
    log "FAILED twice: $t (log $lg)"; [ -n "$lg" ] && tail -n 15 "$lg" | sed 's/^/    | /'
  done
  cpu_jobs=$(squeue -h -u "$USER" -n "$CPU_NAME" -o %i | wc -l); gpu_jobs=$(squeue -h -u "$USER" -n "$GPU_NAME" -o %i | wc -l)
  ours_pending=$(squeue -h -u "$USER" -n "$CPU_NAME,$GPU_NAME" -t PD -o %i | wc -l)
  all_jobs=$(squeue -h -u "$USER" -t R -o %i | wc -l)
  cpu_subs=$(grep -c $'\tcpu\t' "$SUBS"); gpu_subs=$(grep -c $'\tgpu\t' "$SUBS")
  last=$(squeue -h -u "$USER" -t R -o %S | sort | tail -n 1); gap_ok=1
  if [ -n "$last" ]; then [ $(( $(date +%s) - $(date -d "$last" +%s) )) -ge $(( 60 * GAP_MIN )) ] || gap_ok=0; fi
  log "tasks $done_n/$total done, running $running_n, ready cpu $ready_cpu ($ready_cores cores) gpu $ready_gpu, open $open_n |" \
      "jobs: ours cpu $cpu_jobs gpu $gpu_jobs pending $ours_pending, running allocations (all kinds) $all_jobs | submitted cpu $cpu_subs/$CPU_CAP gpu $gpu_subs/$GPU_CAP |" \
      "disk ${free_gb} GB, inodes $inodes"
  [ "$done_n" -eq "$total" ] && { log "every task done: run block 7 of horeka-fullrun.md"; exit 0; }
  [ "$free_gb" -ge "$MIN_FREE_GB" ] && [ "$inodes" -ge "$MIN_FREE_INODES" ] || { log "STOP: storage below the threshold"; exit 2; }
  if [ $((cpu_jobs + gpu_jobs)) -eq 0 ] && [ "$running_n" -eq 0 ] && [ "$ready_cpu" -eq 0 ] && [ "$ready_gpu" -eq 0 ]; then
    log "STOP: blocked; $open_n tasks wait on failed tasks: $(field "' '.join(failed_tasks)")"
    log "fix, then: bash $CODE/analysis/docs/horeka-fullrun-relaunch.sh (see its header)"; exit 3
  fi
  want=""
  admit=0
  if [ "$FILL" = 1 ]; then
    since=$(( $(date +%s) - $(cat "$LAUNCH/last-submit.epoch" 2>/dev/null || echo 0) ))
    [ $((all_jobs + ours_pending)) -lt "$MAX_ALLOC" ] && [ "$since" -ge "$GAP_SEC" ] && admit=1
  elif [ "$ours_pending" -eq 0 ] && [ "$all_jobs" -lt "$MAX_ALLOC" ] && [ "$gap_ok" -eq 1 ]; then admit=1; fi
  if [ "$admit" -eq 1 ]; then
    if [ "$ready_gpu" -gt 0 ] && [ "$gpu_jobs" -lt "$GPU_MAX" ] && [ "$gpu_subs" -lt "$GPU_CAP" ]; then want=gpu
    elif [ "$ready_cpu" -gt 0 ] && [ "$cpu_jobs" -lt "$CPU_MAX" ] && [ "$cpu_subs" -lt "$CPU_CAP" ] \
         && { [ "$cpu_jobs" -eq 0 ] || [ "$ready_cores" -ge "$MIN_READY_CORES" ]; }; then want=cpu; fi
  fi
  [ -n "$want" ] && date +%s > "$LAUNCH/last-submit.epoch"
  if [ "$want" = gpu ]; then
    JOB=$(sbatch --parsable --time="$GPU_TIME" --export=ALL,CODE="$CODE",LEDGER="$LEDGER" --output="$FR/slurm/gpustats-%j.out" \
          "$CODE/analysis/docs/horeka-fullrun-gpu-stats.sbatch") && printf '%s\tgpu\t%s\t%s\n' "$JOB" "$(date -u +%FT%TZ)" "$CODE" >> "$SUBS" \
      && log "submitted GPU job $JOB ($((gpu_subs + 1))/$GPU_CAP, $GPU_TIME)"
  elif [ "$want" = cpu ]; then
    JOB=$(sbatch --parsable --export=ALL,CODE="$CODE",LEDGER="$LEDGER" --output="$FR/slurm/cpu-%j.out" \
          "$CODE/analysis/docs/horeka-fullrun-cpu.sbatch") && printf '%s\tcpu\t%s\t%s\n' "$JOB" "$(date -u +%FT%TZ)" "$CODE" >> "$SUBS" \
      && log "submitted CPU job $JOB ($((cpu_subs + 1))/$CPU_CAP)"
  fi
  if [ -z "$want" ] && [ $((cpu_jobs + gpu_jobs)) -eq 0 ] && [ "$running_n" -eq 0 ] \
     && { { [ "$ready_cpu" -gt 0 ] && [ "$cpu_subs" -ge "$CPU_CAP" ]; } || { [ "$ready_gpu" -gt 0 ] && [ "$gpu_subs" -ge "$GPU_CAP" ]; }; }; then
    log "STOP: caps used with ready work left (ask Valerian before raising CPU_CAP/GPU_CAP)"; exit 5
  fi
  sleep "$POLL"
done
