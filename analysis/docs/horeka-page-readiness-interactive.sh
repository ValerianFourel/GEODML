#!/bin/bash
# Run the whole page-readiness pipeline inside one existing four-A100 allocation.
# Usage (from a HoreKa login shell that holds the allocation):
#   bash horeka-page-readiness-interactive.sh SLURM_JOB_ID
# Every stage is skipped when its output exists, and embedding resumes from saved
# shards, so rerunning the same command in a later allocation continues the work.
JOB=${1:?usage: horeka-page-readiness-interactive.sh SLURM_JOB_ID}
LOG_NAME=interactive
source "$(dirname -- "$0")/horeka-page-readiness-lib.sh"
EXTRACT="$R/extract-llama-v2"
if [ ! -f "$EXTRACT/manifest.json" ]; then
  echo "== extract llama answers and page texts (32 readers)"
  step "$RT/bin/python" -u "$SCRIPT" extract --source "$W/llama-hf/dataset:llama4" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --workers 32 --output "$EXTRACT"
fi
"$RT/bin/python" -c "import json,sys;print('EXTRACT', json.load(open(sys.argv[1]))['counts'])" "$EXTRACT/manifest.json"
ensure_relocation
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$EXTRACT/pages.jsonl.gz" "$R/embed-pages-$VIEW"
done
if [ ! -f "$R/analysis-llama/report.html" ]; then
  echo "== ranking model, keyword bootstrap, permutation null, HTML report"
  step "$RT/bin/python" -u "$SCRIPT" analyze --extract "$EXTRACT" --qwen "$R/embed-pages-qwen.merged" \
    --mistral "$R/embed-pages-mistral.merged" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --workers 32 --output "$R/analysis-llama"
fi
echo "REPORT $R/analysis-llama/report.html  end $(date -u +%FT%TZ)"
