#!/bin/bash
# Render __V4_OPERATOR_PIN__ to the published launcher commit. Run inside login-host tmux.
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  test -z "${SLURM_JOB_ID:-}" || { echo 'Use a separate login shell; preserve the existing allocation.'; exit 1; }
  V4_OPERATOR_PIN=__V4_OPERATOR_PIN__
  V4_OPERATOR="$W/checkouts/v4-interactive-$V4_OPERATOR_PIN"
  V4_RUN="$W/reviews/si-v4-r2-cycle-20261002/development"
  git -C "$REPO" fetch origin codex/v4-interactive-20261003
  if [ ! -d "$V4_OPERATOR" ]; then
    git -C "$REPO" worktree add --detach "$V4_OPERATOR" "$V4_OPERATOR_PIN"
  fi
  test "$(git -C "$V4_OPERATOR" rev-parse HEAD)" = "$V4_OPERATOR_PIN"
  test -z "$(git -C "$V4_OPERATOR" status --porcelain --untracked-files=all)"
  test -f "$V4_RUN/config.json"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$V4_OPERATOR"
  V4_LOG=$(mktemp "$V4_RUN/interactive-20261003-XXXXXX.log")
  printf 'Interactive controller log: %s\n' "$V4_LOG"
  "$RT/bin/python" -u "$V4_OPERATOR/analysis/scripts/run_si_v4_cycle.py" submit \
    --config "$V4_RUN/config.json" --interactive --account-existing-job 5175818 \
    2>&1 | tee -a "$V4_LOG"
)
