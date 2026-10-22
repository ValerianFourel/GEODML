#!/bin/bash
# SIGIR BM25 baseline: Llama-4-Scout through Cheaper Inference's OpenAI-compatible API (model llama-4-scout), on the
# Mac, natural condition.
#   bash analysis/sigir_bm25/run_llama_api.sh ARM [LIMIT]
# ARM is bm25 or orig (the original lexical search, through the same API: the clean contrast, since the provider's
# weights and precision are not ours). PROMPT_IDS=<file> runs only those prompts' cells (build_bundle.py
# sample-prompts writes such a list, e.g. 1,000 held-out prompts = 4,000 cells per arm); LIMIT stops once that many
# cells are finished. Rerun the same command to continue; finished cells are skipped. Stop gracefully:
# touch "$DATA/runs/llama4-$ARM/STOP".
# The API key comes from CHEAPER_INFERENCE_API_KEY (exported, or the env block of ~/.claude/settings.json, as the
# inference skill reads it), else a hidden prompt; it is passed to this run only and never written or printed.
# Settings are the original Llama run's (query/final max tokens 256/4096, no thinking switch, seed 20260911 + cell),
# the original cross-encoder on CPU; known Parallel pairs come from the score cache.
set -euo pipefail
ARM=${1:?usage: run_llama_api.sh bm25|orig [LIMIT]}
LIMIT=${2:-0}
[[ "$ARM" == bm25 || "$ARM" == orig ]] || { echo "ARM must be bm25 or orig" >&2; exit 2; }
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${SIGIR_BM25_DATA:-$HOME/Hamburg/GEODML_Unified/sigir_bm25_data}
PY=${PYTHON:-$DATA/venv/bin/python}
BUNDLE=${BUNDLE:-$DATA/bundle-20261022}
BGE=${BGE:-$HOME/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e}
CACHE=${CACHE:-$DATA/score-cache-pre.parquet}
DDG=${DDG_SNAPSHOT:-$(ls "$DATA"/snapshots/*duckduckgo* 2>/dev/null | head -1)}
SXG=${SEARXNG_SNAPSHOT:-$(ls "$DATA"/snapshots/*searxng* 2>/dev/null | head -1)}
RUN="$DATA/runs/llama4-$ARM"
for p in "$PY" "$BUNDLE/manifest.json" "$BGE/config.json" "$CACHE" "$DDG" "$SXG" ${PROMPT_IDS:+"$PROMPT_IDS"}; do
  test -e "$p" || { echo "missing ${p:-a snapshot (copy both from HoreKa into $DATA/snapshots/)}" >&2; exit 2; }
done
test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
mkdir -p "$RUN"
if [ -z "${CHEAPER_INFERENCE_API_KEY:-}" ]; then
  CHEAPER_INFERENCE_API_KEY=$("$PY" -c 'import json, os; print(json.load(open(os.path.expanduser("~/.claude/settings.json"))).get("env", {}).get("CHEAPER_INFERENCE_API_KEY", ""))' 2>/dev/null || true)
fi
if [ -z "$CHEAPER_INFERENCE_API_KEY" ]; then
  read -r -s -p "Cheaper Inference API key (hidden): " CHEAPER_INFERENCE_API_KEY; echo
fi
export CHEAPER_INFERENCE_API_KEY GEODML_ADMISSION_STOP_FILE="$RUN/STOP" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARGS=(--bundle "$BUNDLE" --run-dir "$RUN" --model llama4 --retriever "$ARM"
      --snapshot "duckduckgo=$DDG" --snapshot "searxng=$SXG"
      --base-url https://api.cheaperinference.com/v1 --served-model llama-4-scout --api-key-env CHEAPER_INFERENCE_API_KEY
      --cross-encoder "$BGE" --device cpu --score-cache "$CACHE" --query-max-tokens 256 --final-max-tokens 4096
      --timeout 300 --transport-retries 5 --request-concurrency "${REQUEST_CONCURRENCY:-32}"
      --cell-concurrency "${CELL_CONCURRENCY:-48}" --limit "$LIMIT")
[ -n "${PROMPT_IDS:-}" ] && ARGS+=(--prompt-ids "$PROMPT_IDS")
status=0
caffeinate -i "$PY" -u "$CODE/analysis/sigir_bm25/generate.py" run "${ARGS[@]}" 2>&1 | tee -a "$RUN/run.log" \
  || status=${PIPESTATUS[0]}
unset CHEAPER_INFERENCE_API_KEY
"$PY" "$CODE/analysis/sigir_bm25/generate.py" status --bundle "$BUNDLE" --run-dir "$RUN" --model llama4
echo "exit $status (0 done, 4 stopped or not finished, 3 breaker: transport failures)"
exit $status
