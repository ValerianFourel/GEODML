#!/bin/bash
# SIGIR BM25 baseline, local: Llama-4-Scout through Cheaper Inference (model llama-4-scout) from this Mac, natural
# condition, Parallel Expansion, on a prompt list (default: the first 1,000 of the 26,008 prompts).
#   bash analysis/sigir_bm25/run_llama_local.sh ARM          ARM = bm25 | orig (the original search, same API)
# Everything heavy is precomputed or remote: BM25 or the original search runs locally (the release's snapshot rows),
# the reranker scores come from the score cache (no local model; a cell with an unscored pair is retried later and
# never recorded), and generation goes to the API with the pipeline's own requests: temperature 0, seed
# 20260911 + cell id + call index, strict JSON schema, max tokens 256/4096 (the original Llama settings), the logged
# Parallel queries replayed. The Reactive Loop is not run here: its queries are written live, so its reranker pairs
# cannot be precomputed, and this Mac's CPU scores about one pair per second.
# Key and endpoint as the inference skill resolves them: CHEAPER_INFERENCE_API_KEY (or INFERENCE_API_KEY), exported
# or from the env block of ~/.claude/settings.json; INFERENCE_BASE_URL (default https://api.cheaperinference.com/v1).
# The key stays in this process's environment; it is never printed or written. Rerun to continue; finished cells are
# skipped. Stop gracefully: touch "$RUN/STOP". Options: PROMPT_IDS, CACHE, REQUEST_CONCURRENCY, CELL_CONCURRENCY.
set -euo pipefail
ARM=${1:?usage: run_llama_local.sh bm25|orig}
[[ "$ARM" == bm25 || "$ARM" == orig ]] || { echo "ARM must be bm25 or orig" >&2; exit 2; }
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${SIGIR_BM25_DATA:-$HOME/Hamburg/GEODML_Unified/sigir_bm25_data}
PY=${PYTHON:-$DATA/venv/bin/python}  # needs aiohttp + brotli (the API answers brotli-compressed), pandas, pyarrow,
                                    # numpy, scipy; no torch: $DATA/venv/bin/pip install aiohttp brotli pandas pyarrow numpy scipy
BUNDLE=${BUNDLE:-$DATA/bundle-20261022}
CACHE=${CACHE:-$DATA/score-cache-pre.parquet}
SNAP=${SNAPSHOTS:-$DATA/snapshots-release}
if [ "${ALL:-0}" = 1 ]; then  # every Parallel cell of the bundle
  SELECT=(); LIST=$BUNDLE/manifest.json; NAME=parallel
elif [ -n "${CELL_IDS:-}" ]; then  # a sampled cell list (build_bundle.py sample-cells) instead of a prompt list
  SELECT=(--cell-ids "$CELL_IDS"); LIST=$CELL_IDS; NAME=$(basename "$LIST" .txt)
else
  PROMPT_IDS=${PROMPT_IDS:-$DATA/first-1000.txt}; SELECT=(--prompt-ids "$PROMPT_IDS"); LIST=$PROMPT_IDS
  NAME=$(basename "$LIST" .txt)
fi
RUN="$DATA/runs/llama4-$ARM-local-${RUN_NAME:-$NAME}"
for p in "$PY" "$BUNDLE/manifest.json" "$CACHE" "$LIST" "$SNAP/duckduckgo.jsonl" "$SNAP/searxng.jsonl"; do
  test -e "$p" || { echo "missing $p" >&2; exit 2; }
done
test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
KEY=${CHEAPER_INFERENCE_API_KEY:-${INFERENCE_API_KEY:-}}
if [ -z "$KEY" ]; then
  KEY=$("$PY" -c 'import json, pathlib
env = json.loads((pathlib.Path.home() / ".claude" / "settings.json").read_text()).get("env", {})
print(env.get("CHEAPER_INFERENCE_API_KEY") or env.get("INFERENCE_API_KEY") or "")' 2>/dev/null || true)
fi
[ -n "$KEY" ] || { echo "set CHEAPER_INFERENCE_API_KEY (see the inference skill's SKILL.md)" >&2; exit 2; }
BASE=${INFERENCE_BASE_URL:-https://api.cheaperinference.com/v1}
mkdir -p "$RUN"
export SIGIR_API_KEY="$KEY" GEODML_ADMISSION_STOP_FILE="$RUN/STOP" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
unset KEY
status=0
caffeinate -i "$PY" -u "$CODE/analysis/sigir_bm25/generate.py" run --bundle "$BUNDLE" --run-dir "$RUN" \
  --model llama4 --retriever "$ARM" --methods Parallel-Expansion-v1 ${SELECT[@]+"${SELECT[@]}"} \
  --snapshot "duckduckgo=$SNAP/duckduckgo.jsonl" --snapshot "searxng=$SNAP/searxng.jsonl" \
  --base-url "$BASE" --served-model llama-4-scout --api-key-env SIGIR_API_KEY \
  --cache-only --score-cache "$CACHE" --query-max-tokens 256 --final-max-tokens 4096 \
  --timeout 300 --transport-retries 5 --breaker 50 \
  --request-concurrency "${REQUEST_CONCURRENCY:-32}" --cell-concurrency "${CELL_CONCURRENCY:-48}" \
  2>&1 | tee -a "$RUN/run.log" || status=${PIPESTATUS[0]}
unset SIGIR_API_KEY
"$PY" "$CODE/analysis/sigir_bm25/generate.py" status --bundle "$BUNDLE" --run-dir "$RUN" --model llama4 \
  ${SELECT[@]+"${SELECT[@]}"} --methods Parallel-Expansion-v1 || true
echo "exit $status (0 done, 4 cells left (unscored pairs or stopped), 3 breaker: API failures)"
exit $status
