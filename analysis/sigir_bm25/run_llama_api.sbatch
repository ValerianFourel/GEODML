#!/bin/bash
#SBATCH --job-name=geodml-sigir-llama-api
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# SIGIR BM25 baseline: Llama-4-Scout through Cheaper Inference's API (model llama-4-scout), natural condition, on a
# HoreKa GPU node: generation is remote, the bge-reranker runs as one replica per GPU (no vLLM here). The Mac cannot
# do this part: its 4-core Intel CPU scores about one reranker pair per second.
# Submit: sbatch --account=<account> --export=ALL,CODE=<clean checkout>,BUNDLE=<bundle>,ARM=<bm25|orig>[,PROMPT_IDS=<file>]
#   run_llama_api.sbatch            (dev_accelerated 1 h for the pilot; add --partition=accelerated --time=06:00:00
#   for the full arms). The key is read from KEY_FILE (default ~/.config/geodml/cheaper_inference_key, mode 600) into
#   this job's environment only. Search snapshots: SNAPSHOTS (default the run inputs' snapshots/ folder, copied from
#   the original Qwen bout inputs). Resumable: resubmit with the same ARM; finished cells are skipped. Exit codes:
#   0 done, 4 deadline checkpoint, 3 breaker (API failures), 2 refused.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=${CODE:?CODE must point to the clean, pinned checkout}
BUNDLE=${BUNDLE:?BUNDLE must point to the copied bundle directory}
ARM=${ARM:?ARM: bm25 or orig}
JOB=${SLURM_JOB_ID:?run under sbatch or inside a salloc shell}
[[ "$ARM" == bm25 || "$ARM" == orig ]] || { echo "ARM must be bm25 or orig" >&2; exit 2; }
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
PY="$W/environment/qwen-runtime/bin/python"
BGE="$W/models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
SNAPSHOTS=${SNAPSHOTS:-$W/runs/sigir-bm25/inputs/snapshots}
KEY_FILE=${KEY_FILE:-$HOME/.config/geodml/cheaper_inference_key}
RUN="$W/runs/sigir-bm25/llama4-$ARM-${RUN_TAG:-20261022}"
DDG=$(ls "$SNAPSHOTS"/duckduckgo.* | head -1); SXG=$(ls "$SNAPSHOTS"/searxng.* | head -1)
for p in "$PY" "$BGE/config.json" "$BUNDLE/manifest.json" "$DDG" "$SXG" "$KEY_FILE" ${PROMPT_IDS:+"$PROMPT_IDS"}; do
  test -e "$p" || { echo "missing $p" >&2; exit 2; }
done
[ "$(stat -c %a "$KEY_FILE")" = 600 ] || { echo "$KEY_FILE must be mode 600" >&2; exit 2; }
mkdir -p "$RUN/logs"
echo "sigir llama api ($ARM): commit $(git -C "$CODE" rev-parse HEAD) job $JOB host $(hostname) start $(date -u +%FT%TZ)"
code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' https://api.cheaperinference.com/v1/models || true)
[[ "$code" == 200 || "$code" == 401 ]] || { echo "no route from this node to api.cheaperinference.com (HTTP '$code')" >&2; exit 2; }
END=$(squeue -h -j "$JOB" -o %e)
export SLURM_JOB_END_TIME=$(date -d "$END" +%s) GEODML_START_MARGIN_SECONDS=300 GEODML_CLEANUP_MARGIN_SECONDS=120
export GEODML_ADMISSION_STOP_FILE="$RUN/STOP" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CHEAPER_INFERENCE_API_KEY=$(cat "$KEY_FILE"); export CHEAPER_INFERENCE_API_KEY
ARGS=(--bundle "$BUNDLE" --run-dir "$RUN" --model llama4 --retriever "$ARM"
      --snapshot "duckduckgo=$DDG" --snapshot "searxng=$SXG"
      --base-url https://api.cheaperinference.com/v1 --served-model llama-4-scout --api-key-env CHEAPER_INFERENCE_API_KEY
      --cross-encoder "$BGE" --device cuda-pool --query-max-tokens 256 --final-max-tokens 4096
      --timeout 300 --transport-retries 5 --request-concurrency "${REQUEST_CONCURRENCY:-32}"
      --cell-concurrency "${CELL_CONCURRENCY:-48}" --limit "${LIMIT:-0}")
[ -n "${PROMPT_IDS:-}" ] && ARGS+=(--prompt-ids "$PROMPT_IDS")
[ -n "${CELL_IDS:-}" ] && ARGS+=(--cell-ids "$CELL_IDS")
[ -n "${METHODS:-}" ] && ARGS+=(--methods "$METHODS")
status=0
"$PY" -u "$CODE/analysis/sigir_bm25/generate.py" run "${ARGS[@]}" > "$RUN/logs/generate-$JOB.log" 2>&1 || status=$?
unset CHEAPER_INFERENCE_API_KEY
tail -n 3 "$RUN/logs/generate-$JOB.log"
"$PY" "$CODE/analysis/sigir_bm25/generate.py" status --bundle "$BUNDLE" --run-dir "$RUN" --model llama4 \
  ${PROMPT_IDS:+--prompt-ids "$PROMPT_IDS"} ${CELL_IDS:+--cell-ids "$CELL_IDS"} ${METHODS:+--methods "$METHODS"} || true
echo "end $(date -u +%FT%TZ) exit $status (0 done, 4 deadline checkpoint, 3 breaker, 2 refused)"
exit $status
