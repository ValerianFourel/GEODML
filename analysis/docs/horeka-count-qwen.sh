#!/bin/bash
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  INFERENCE_PIN=7a5d3bb89f2b3349cfdd2c80748f2fa5bf3a3f11
  INFERENCE_CODE="$W/checkouts/inference-status-$INFERENCE_PIN"
  git -C "$REPO" fetch origin codex/gemma-selected-cells
  if [ ! -d "$INFERENCE_CODE" ]; then
    git -C "$REPO" worktree add --detach "$INFERENCE_CODE" "$INFERENCE_PIN"
  fi
  test "$(git -C "$INFERENCE_CODE" rev-parse HEAD)" = "$INFERENCE_PIN"
  test -z "$(git -C "$INFERENCE_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$INFERENCE_CODE"
  "$RT/bin/python" "$INFERENCE_CODE/analysis/scripts/audit_horeka_saved_progress.py" \
    "$W" --qwen-counts-only
)
