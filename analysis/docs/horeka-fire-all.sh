#!/bin/bash
# Valerian asked (2026-10-06) for one script that sends all missing work now.
# Rerunnable: starts only senders that are not running; never touches finished
# results, running jobs or other senders. Run from a HoreKa login shell.
#  1. Gemma v4 on Qwen answers: existing frozen pass, tmux gemma-v4-qwen.
#  2. Qwen generation: second finite recovery sweep (pin 405b408), tmux qwen-recovery-2.
#  3. For 6 hours, moves our waiting cpuonly preparation jobs to dev_cpuonly (tmux dev-mover).
set -uo pipefail
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
ACCOUNT=hk-project-p0026831
GCODE="$W/checkouts/gemma-qwen-469b7861b0bd7eaa0f51a8051dcee0a23e359c8d"
GRUN="$W/reviews/gemma-si-v4-qwen-cpu-5h-20261006"
QPIN=405b408cda3d2bab926c90975cb55476f8cfcd0c
QCODE="$W/checkouts/qwen-recovery-$QPIN"
QRUN="$W/reviews/qwen-recovery-20261006"
QPREV="$W/reviews/qwen-recovery-5h-20261003"
LOG="$W/reviews/fire-all.log"

echo "== 1. Gemma v4 on Qwen answers"
if tmux has-session -t '=gemma-v4-qwen' 2>/dev/null; then
  echo "running: $(tail -n 1 "$GRUN/sender.log")"
else
  echo "not running; restarting the same run (type the HF write token in the window)"
  bash "$GCODE/analysis/docs/horeka-gemma-v4-qwen.sh" "$W" "$W/shared-hours/dataset" "$GRUN" "$ACCOUNT" \
    "$W/reviews/si-v4-r2-cycle-20261002/development/candidate-inputs"
fi

echo "== 2. Qwen generation, second recovery sweep"
if tmux has-session -t '=qwen-recovery-2' 2>/dev/null; then
  echo "running: $(tail -n 1 "$QRUN/sender.log" 2>/dev/null)"
else
  git -C "$REPO" fetch -q origin codex/qwen-failed-attempt-idempotent-20261006
  test -d "$QCODE" || git -C "$REPO" worktree add -q --detach "$QCODE" "$QPIN"
  if [ "$(git -C "$QCODE" rev-parse HEAD)" != "$QPIN" ] || [ -n "$(git -C "$QCODE" status --porcelain --untracked-files=all)" ]; then
    echo "Qwen checkout is not clean at $QPIN; not starting"
  else
    mkdir -p "$QRUN"
    cat > "$QRUN/sender-command.sh" <<EOF
#!/bin/bash
set -euo pipefail
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$QCODE" GEODML_AUDIT_PROGRESS=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
echo "qwen recovery sender start \$(date -u +%FT%TZ) pin $QPIN"
exec "\$RT/bin/python" -u "$QCODE/analysis/scripts/horeka_qwen_recovery.py" send \\
  --workspace "$W" --account $ACCOUNT --output "$QRUN" \\
  --walltime 05:00:00 --previous-recovery "$QPREV" --gpu-all-at-once
EOF
    tmux new-session -d -s qwen-recovery-2 "bash $QRUN/sender-command.sh >> $QRUN/sender.log 2>&1"
    echo "started tmux qwen-recovery-2; log $QRUN/sender.log"
  fi
fi

echo "== 3. Move waiting preparation jobs to dev_cpuonly (every 2 min for 6 h)"
if tmux has-session -t '=dev-mover' 2>/dev/null; then
  echo "mover already running"
else
  tmux new-session -d -s dev-mover "for i in \$(seq 1 180); do
    squeue --me -h -t PD -p cpuonly -o '%i %j' | while read -r J N; do
      case \"\$N\" in geodml-gemma-v4-prepare|geodml-qwen-recovery-preparation|geodml-qwen-recovery-final)
        scontrol update JobId=\$J Partition=dev_cpuonly && echo \"\$(date -u +%FT%TZ) moved \$J \$N\" >> $LOG ;;
      esac
    done
    sleep 120
  done"
  echo "mover started; moves logged to $LOG"
fi

sleep 20
echo "== queue"; squeue --me -o '%.10i %.14P %.34j %.8T %.10M %.10l %R'
echo "== recheck: tail -n 3 $GRUN/sender.log $QRUN/sender.log $LOG"
