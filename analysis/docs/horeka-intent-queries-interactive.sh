#!/bin/bash
# Embed the AI's search queries (intent surfacing study) with both frozen LLM2Vec views inside one
# existing four-A100 allocation, then merge each view. Usage, from the login shell that holds it:
#   bash horeka-intent-queries-interactive.sh SLURM_JOB_ID
# Needs the trace-extract output of horeka-intent-stages.sbatch. Rerun in a later allocation to
# continue: embedding resumes from saved shards and finished views are skipped. Then resubmit
# horeka-intent-stages.sbatch for the analysis.
JOB=${1:?usage: horeka-intent-queries-interactive.sh SLURM_JOB_ID}
LOG_NAME=intent-queries
source "$(dirname -- "$0")/horeka-page-readiness-lib.sh"
QUERIES="$R/intent-trace-extract-v1/queries.jsonl.gz"
test -f "$QUERIES" || { echo "missing $QUERIES: run horeka-intent-stages.sbatch first" >&2; exit 2; }
"$RT/bin/python" -c "import json, sys; print('QUERIES', json.load(open(sys.argv[1]))['counts']['unique_queries'])" \
  "$R/intent-trace-extract-v1/manifest.json"
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$QUERIES" "$R/embed-queries-$VIEW"
done
echo "QUERIES embedded: $R/embed-queries-qwen.merged $R/embed-queries-mistral.merged end $(date -u +%FT%TZ)"
echo "Next: resubmit horeka-intent-stages.sbatch for the analysis."
