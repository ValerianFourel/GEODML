#!/bin/bash
# Embed every distinct evidence snippet shown to llama and Qwen (both search engines)
# with both frozen LLM2Vec views, keeping the full vectors, and package them for the
# private dataset. Usage, from the login shell that holds a four-A100 allocation:
#   bash horeka-snippet-embeddings-interactive.sh SLURM_JOB_ID
# Rerun the same command in a later allocation to continue; finished stages are skipped.
# Publishing needs internet and a write token, so it runs afterwards on a login node.
JOB=${1:?usage: horeka-snippet-embeddings-interactive.sh SLURM_JOB_ID}
LOG_NAME=snippets
source "$(dirname -- "$0")/horeka-page-readiness-lib.sh"
EXTRACT="$R/extract-all-v1"
PACKAGE="$R/snippet-embeddings-v1"
if [ ! -f "$EXTRACT/manifest.json" ]; then
  echo "== extract snippets shown to llama and Qwen (32 readers)"
  step "$RT/bin/python" -u "$SCRIPT" extract --source "$W/llama-hf/dataset:llama4" \
    --source "$W/shared-hours/dataset:qwen38" --final-axis-map "$A/final-audit/final-axis-map.jsonl" \
    --workers 32 --output "$EXTRACT"
fi
"$RT/bin/python" -c "import json,sys;print('EXTRACT', json.load(open(sys.argv[1]))['counts'])" "$EXTRACT/manifest.json"
ensure_relocation
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$EXTRACT/pages.jsonl.gz" "$R/embed-all-$VIEW" --save-embeddings
done
if [ ! -f "$PACKAGE/manifest.json" ]; then
  echo "== package snippet table, vectors and manifest"
  step "$RT/bin/python" -u "$SCRIPT" package --extract "$EXTRACT" --qwen "$R/embed-all-qwen.merged" \
    --mistral "$R/embed-all-mistral.merged" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --relocation "$R/relocation-fresh.json" --output "$PACKAGE"
fi
echo "PACKAGE $PACKAGE  end $(date -u +%FT%TZ)"
echo "Next, on a login node: $RT/bin/python $SCRIPT publish --package $PACKAGE"
