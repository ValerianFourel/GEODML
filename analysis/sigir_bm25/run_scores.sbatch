#!/bin/bash
#SBATCH --job-name=geodml-sigir-scores
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# SIGIR BM25 baseline: bge-reranker-v2-m3 scores for the Parallel pairs no logged score covers (build_bundle.py
# pairs), with the rebuttal's score_pairs.py `score` step unchanged: one worker per GPU, length-sorted batches,
# no shard started that could not finish 5 minutes before the end. No generator is called.
# Submit: sbatch --account=<account> --export=ALL,CODE=<clean checkout>,PAIRS=<pairs bundle>,RUN_ID=<id> run_scores.sbatch
# Resumable and chainable like run_qwen_bm25.sbatch (exit 4 = deadline checkpoint, CHAIN_STOP on failure).
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=${CODE:?CODE must point to the clean, pinned checkout}
PAIRS=${PAIRS:?PAIRS must point to the copied pairs bundle}
RUN_ID=${RUN_ID:?RUN_ID, e.g. scores-20261023}
JOB=${SLURM_JOB_ID:?run under sbatch or inside a salloc shell}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
PY="$W/environment/qwen-runtime/bin/python"
BGE="$W/models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RUN="$W/runs/sigir-bm25/$RUN_ID"
SP="$CODE/analysis/www_robustness/score_pairs.py"
mkdir -p "$RUN/logs" "$RUN/results"
echo "sigir scores: commit $(git -C "$CODE" rev-parse HEAD) job $JOB host $(hostname) start $(date -u +%FT%TZ)"
if [ -s "$RUN/CHAIN_STOP" ]; then
  echo "stopped by an earlier job, nothing run (delete $RUN/CHAIN_STOP to go on):"; cat "$RUN/CHAIN_STOP"; exit 0
fi
[ -s "$RUN/results/shards.parquet" ] || cp "$PAIRS/shards.parquet" "$RUN/results/shards.parquet"
cmp -s "$PAIRS/shards.parquet" "$RUN/results/shards.parquet" || { echo "shard plan differs from $PAIRS" >&2; exit 2; }
END=$(squeue -h -j "$JOB" -o %e)
DEADLINE=$(( $(date -d "$END" +%s) - 300 ))
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pids=()
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$PY" -u "$SP" score --bundle "$PAIRS" --run-dir "$RUN" --model "$BGE" \
    --worker $GPU --workers 4 --deadline-epoch "$DEADLINE" > "$RUN/logs/score-worker$GPU-$JOB.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  s=0; wait "$pid" || s=$?
  if [ $s -ne 0 ] && [ $s -ne 4 ]; then status=1; elif [ $s -eq 4 ] && [ $status -eq 0 ]; then status=4; fi
done
"$PY" "$SP" status --run-dir "$RUN" || true
case $status in
  0) echo "== done $(date -u +%FT%TZ): copy $RUN/results/scores back" ;;
  4) echo "== deadline checkpoint $(date -u +%FT%TZ): resubmit with the same RUN_ID" ;;
  *) tail -n 20 "$RUN"/logs/score-worker*-"$JOB".log; echo "$(date -u +%FT%TZ) job $JOB: scoring failed" >> "$RUN/CHAIN_STOP" ;;
esac
echo "end $(date -u +%FT%TZ) exit $status"
exit $status
