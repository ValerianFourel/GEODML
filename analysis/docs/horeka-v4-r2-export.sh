#!/bin/bash
# Capture all submitted resource use and export quiescent results for review.
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  CODE="$W/checkouts/si-v4-r2-__SI_V4_PIN__"
  CYCLE="$W/reviews/si-v4-r2-cycle-20261002"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
  EXPORT="$(mktemp -d "$CYCLE/export-XXXXXXXX")"
  "$RT/bin/python" "$CODE/analysis/scripts/run_si_v4_cycle.py" resource-usage \
    --budget "$CYCLE/cycle-budget.json" --output "$EXPORT/resource-usage.json"
  "$RT/bin/python" "$CODE/analysis/scripts/run_si_v4_cycle.py" assess \
    --config "$CYCLE/development/config.json" --resource-usage "$EXPORT/resource-usage.json" \
    --output "$EXPORT/development-progress.json"
  "$RT/bin/python" "$CODE/analysis/scripts/run_si_v4_cycle.py" export \
    --config "$CYCLE/development/config.json" --output "$EXPORT/evaluation-evidence.tar.gz"
  printf 'RETURN_THIS_DIRECTORY=%s\n' "$EXPORT"
)
