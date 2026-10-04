#!/bin/bash
# Read-only: count glued-title rows in the frozen DuckDuckGo and SearXNG snapshots and
# how often llama was shown them. Run on a login node: bash horeka-audit-snapshot-glued-rows.sh
set -uo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
# The traces record each engine's snapshot path and SHA-256; a same-named copy under the
# workspace is accepted only if its hash matches.
"$RT/bin/python" "$CODE/analysis/scripts/audit_snapshot_glued_rows.py" \
  --locate-from "$W/shared-hours/dataset" --locate-from "$W/llama-hf/dataset" --search-root "$W" \
  --pages "$W/reviews/page-readiness-20261004/extract-llama-v2/pages.jsonl.gz"
