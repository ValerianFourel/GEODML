#!/bin/bash
# SIGIR BM25 baseline: Llama-4-Scout through Cheaper Inference's OpenAI-compatible API, on the Mac.
#   bash analysis/sigir_bm25/run_llama_api.sh ARM [LIMIT]
# ARM is bm25 or orig (the original lexical search, through the same API: the clean contrast, since the provider's
# weights and precision are not ours). LIMIT stops once that many cells of the arm are finished (pilots); without it
# the arm runs to the end. Rerun the same command to continue; finished cells are skipped. The API key is read
# hidden into this process only and never written anywhere. Stop gracefully: touch "$DATA/runs/llama4-$ARM/STOP".
# Settings are the original Llama run's (query/final max tokens 256/4096, no thinking switch), the original
# cross-encoder on CPU, scores for known Parallel pairs from the cache.
set -euo pipefail
ARM=${1:?usage: run_llama_api.sh bm25|orig [LIMIT]}
LIMIT=${2:-0}
[[ "$ARM" == bm25 || "$ARM" == orig ]] || { echo "ARM must be bm25 or orig" >&2; exit 2; }
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${SIGIR_BM25_DATA:-$HOME/Hamburg/GEODML_Unified/sigir_bm25_data}
PY=${PYTHON:-$DATA/venv/bin/python}
BUNDLE=${BUNDLE:-$DATA/bundle-20261022}
BGE=${BGE:?BGE: local snapshot of BAAI/bge-reranker-v2-m3 at 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e}
CACHE=${CACHE:-$DATA/score-cache.parquet}
DDG=${DDG_SNAPSHOT:?DDG_SNAPSHOT: the DuckDuckGo snapshot file copied from HoreKa (bout config SEARCH_AGENTIC_DDG_SNAPSHOT)}
SXG=${SEARXNG_SNAPSHOT:?SEARXNG_SNAPSHOT: the SearXNG snapshot file copied from HoreKa}
RUN="$DATA/runs/llama4-$ARM"
for p in "$PY" "$BUNDLE/manifest.json" "$BGE/config.json" "$CACHE" "$DDG" "$SXG"; do
  test -e "$p" || { echo "missing $p" >&2; exit 2; }
done
test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
mkdir -p "$RUN"
read -r -s -p "Cheaper Inference API key (hidden): " CI_API_KEY; echo
export CI_API_KEY GEODML_ADMISSION_STOP_FILE="$RUN/STOP" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
status=0
caffeinate -i "$PY" -u "$CODE/analysis/sigir_bm25/generate.py" run --bundle "$BUNDLE" --run-dir "$RUN" \
  --model llama4 --retriever "$ARM" \
  --snapshot "duckduckgo=$DDG" --snapshot "searxng=$SXG" \
  --base-url https://api.cheaperinference.com/v1 --served-model llama-4-scout --api-key-env CI_API_KEY \
  --cross-encoder "$BGE" --device cpu --score-cache "$CACHE" --query-max-tokens 256 --final-max-tokens 4096 \
  --timeout 300 --transport-retries 5 --request-concurrency "${REQUEST_CONCURRENCY:-32}" \
  --cell-concurrency "${CELL_CONCURRENCY:-48}" --limit "$LIMIT" 2>&1 | tee -a "$RUN/run.log" || status=${PIPESTATUS[0]}
unset CI_API_KEY
"$PY" "$CODE/analysis/sigir_bm25/generate.py" status --bundle "$BUNDLE" --run-dir "$RUN" --model llama4
echo "exit $status (0 done, 4 stopped or limit not yet reached, 3 breaker: transport failures)"
exit $status
