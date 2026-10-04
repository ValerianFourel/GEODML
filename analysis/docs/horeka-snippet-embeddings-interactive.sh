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
echo "PACKAGE $PACKAGE"

# The whole mini internet: every servable row of both frozen snapshots, shown or not.
CORPUS="$R/corpus-v1"
CORPUS_PACKAGE="$R/snippet-embeddings-corpus-v1"
if [ ! -f "$CORPUS/manifest.json" ]; then
  echo "== corpus: every servable snapshot row, located by the hashes recorded in the traces"
  step "$RT/bin/python" -u "$SCRIPT" corpus --locate-from "$W/shared-hours/dataset" --locate-from "$W/llama-hf/dataset" \
    --search-root "$W" --shown "$EXTRACT" --output "$CORPUS"
fi
"$RT/bin/python" -c "import json,sys;print('CORPUS', json.load(open(sys.argv[1]))['counts'])" "$CORPUS/manifest.json"
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$CORPUS/pages.jsonl.gz" "$R/embed-corpus-$VIEW" --save-embeddings
done
if [ ! -f "$CORPUS_PACKAGE/manifest.json" ]; then
  echo "== package the corpus snippets, vectors and manifest"
  step "$RT/bin/python" -u "$SCRIPT" package --extract "$CORPUS" --qwen "$R/embed-corpus-qwen.merged" \
    --mistral "$R/embed-corpus-mistral.merged" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --relocation "$R/relocation-fresh.json" --output "$CORPUS_PACKAGE"
fi
echo "CORPUS_PACKAGE $CORPUS_PACKAGE  end $(date -u +%FT%TZ)"
echo "Next, on a login node, publish each package:"
echo "  $RT/bin/python $SCRIPT publish --package $PACKAGE"
echo "  $RT/bin/python $SCRIPT publish --package $CORPUS_PACKAGE"
