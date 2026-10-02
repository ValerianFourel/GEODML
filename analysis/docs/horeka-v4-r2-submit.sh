#!/bin/bash
# Submit one identified segment. The existing budget, scheduler and storage guards apply.
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  CODE="$W/checkouts/si-v4-r2-__SI_V4_PIN__"
  CYCLE="$W/reviews/si-v4-r2-cycle-20261002"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
  "$RT/bin/python" "$CODE/analysis/scripts/run_si_v4_cycle.py" submit \
    --config "$CYCLE/development/config.json"
)
