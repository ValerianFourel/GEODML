#!/bin/bash
# Read-only: count glued-title rows in the frozen DuckDuckGo and SearXNG snapshots and
# how often llama was shown them. Run on a login node: bash horeka-audit-snapshot-glued-rows.sh
set -uo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
FILES=$(find "$W" -maxdepth 8 -type f \( -name '*.parquet' -o -name '*.jsonl' \) 2>/dev/null \
  | grep -i snapshot | grep -v -e reviews/ -e checkouts/)
echo '--- snapshot files found'
echo "$FILES" | grep -i -e ddg -e duck -e searx | head -10
DDG=$(echo "$FILES" | grep -i -e ddg -e duck | head -1)
SRX=$(echo "$FILES" | grep -i searx | head -1)
echo "using DDG=$DDG"
echo "using SEARXNG=$SRX"
if [ -z "$DDG" ] || [ -z "$SRX" ]; then
  echo 'snapshot files not found; paste the list above'
  exit 1
fi
"$RT/bin/python" "$CODE/analysis/scripts/audit_snapshot_glued_rows.py" \
  --snapshot "duckduckgo=$DDG" --snapshot "searxng=$SRX" \
  --pages "$W/reviews/page-readiness-20261004/extract-llama-v2/pages.jsonl.gz"
