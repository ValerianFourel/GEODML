#!/bin/bash
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  QWEN_RECOVERY_PIN=4a668d72bb2727518e9aa30ee08442d848fbc254
  QWEN_RECOVERY_CODE="$W/checkouts/qwen-recovery-$QWEN_RECOVERY_PIN"
  QWEN_RECOVERY_RUN="$W/reviews/qwen-recovery-20261003"
  test "$(git -C "$QWEN_RECOVERY_CODE" rev-parse HEAD)" = "$QWEN_RECOVERY_PIN"
  test -z "$(git -C "$QWEN_RECOVERY_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$QWEN_RECOVERY_CODE"
  export GEODML_AUDIT_PROGRESS=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  "$RT/bin/python" -u "$QWEN_RECOVERY_CODE/analysis/scripts/horeka_qwen_recovery.py" next \
    --output "$QWEN_RECOVERY_RUN"
)
