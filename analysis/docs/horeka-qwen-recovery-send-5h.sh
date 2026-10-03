#!/bin/bash
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  test -z "${SLURM_JOB_ID:-}" || { echo 'Use a separate HoreKa login shell; preserve the existing allocation.'; exit 1; }
  QWEN_PIN=76622b72dd4346669ccad9bbd10b8a98d27c5f5d
  QWEN_CODE="$W/checkouts/qwen-recovery-$QWEN_PIN"
  QWEN_RUN="$W/reviews/qwen-recovery-5h-20261003"
  git -C "$REPO" fetch origin codex/qwen-recovery-20261003
  if [ ! -d "$QWEN_CODE" ]; then
    git -C "$REPO" worktree add --detach "$QWEN_CODE" "$QWEN_PIN"
  fi
  test "$(git -C "$QWEN_CODE" rev-parse HEAD)" = "$QWEN_PIN"
  test -z "$(git -C "$QWEN_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$QWEN_CODE" GEODML_AUDIT_PROGRESS=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  mkdir -p "$QWEN_RUN"
  nohup "$RT/bin/python" -u "$QWEN_CODE/analysis/scripts/horeka_qwen_recovery.py" send \
    --workspace "$W" --account hk-project-p0026831 --walltime 05:00:00 \
    --previous-recovery "$W/reviews/qwen-recovery-20261003" \
    --output "$QWEN_RUN" >> "$QWEN_RUN/sender.log" 2>&1 < /dev/null &
  QWEN_SENDER_PID=$!
  sleep 1
  if ! kill -0 "$QWEN_SENDER_PID" 2>/dev/null; then
    tail -n 20 "$QWEN_RUN/sender.log"
    exit 1
  fi
  printf 'Sender PID: %s\nLog: %s\nFinal audit: %s\n' \
    "$QWEN_SENDER_PID" "$QWEN_RUN/sender.log" "$QWEN_RUN/final/coverage.json"
)
