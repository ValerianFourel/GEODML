#!/bin/bash
# WWW 2027 brief, T1 (retriever robustness) inside one existing four-A100 HoreKa allocation:
#   embed (4 GPUs, e5-large-v2) -> retrieve (CPU) -> score (4 GPUs, bge-reranker-v2-m3) -> status.
# Usage, from the login shell that holds the allocation:
#   bash analysis/www_robustness/run_horeka.sh SLURM_JOB_ID BUNDLE_DIR RUN_ID
# Rerun with the same RUN_ID in a later allocation to continue: finished steps and shards are skipped.
# Scoring admits no shard that could not finish 5 minutes before the allocation ends (exit 4 = checkpoint).
# It never cancels, extends or releases the allocation.
set -euo pipefail
set +x
JOB=${1:?usage: run_horeka.sh SLURM_JOB_ID BUNDLE_DIR RUN_ID}
BUNDLE=${2:?usage: run_horeka.sh SLURM_JOB_ID BUNDLE_DIR RUN_ID}
RUN_ID=${3:?usage: run_horeka.sh SLURM_JOB_ID BUNDLE_DIR RUN_ID}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "allocation job id must be numeric" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run id: letters, digits, dot, dash, underscore" >&2; exit 2; }
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
PY="$W/environment/qwen-runtime/bin/python"
E5="$W/models/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6"
BGE="$W/models/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RUN="$W/runs/www-robustness/$RUN_ID"
SP="$CODE/analysis/www_robustness/score_pairs.py"
for p in "$PY" "$E5/config.json" "$BGE/config.json" "$BUNDLE/manifest.json"; do
  test -e "$p" || { echo "missing $p" >&2; exit 2; }
done
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/run-$JOB.log") 2>&1
END=$(squeue -h -j "$JOB" -o %e)
DEADLINE=$(( $(date -d "$END" +%s) - 300 ))
echo "commit $(git -C "$CODE" rev-parse HEAD) allocation $JOB ends $END start $(date -u +%FT%TZ)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PY SP BUNDLE RUN JOB

gpu4() {  # one 4-GPU step, one pinned worker per GPU; exit 4 if any worker checkpointed, 1 if any failed
  local step=$1; shift
  STEP="$step" STEP_ARGS="$*" srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 bash -c '
    set -u
    pids=()
    for GPU in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 "$PY" -u "$SP" "$STEP" --bundle "$BUNDLE" --run-dir "$RUN" \
        --worker $GPU --workers 4 $STEP_ARGS > "$RUN/logs/$STEP-worker$GPU-$JOB.log" 2>&1 &
      pids+=($!)
    done
    worst=0
    for pid in "${pids[@]}"; do
      wait "$pid"; s=$?
      if [ $s -ne 0 ] && [ $s -ne 4 ]; then worst=1; elif [ $s -eq 4 ] && [ $worst -eq 0 ]; then worst=4; fi
    done
    exit $worst'
}

echo "== embed $(date -u +%FT%TZ)"
gpu4 embed --model "$E5" || { tail -n 20 "$RUN"/logs/embed-worker*-"$JOB".log; exit 1; }
echo "== retrieve $(date -u +%FT%TZ)"
OMP_NUM_THREADS=32 srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=none \
  "$PY" -u "$SP" retrieve --bundle "$BUNDLE" --run-dir "$RUN" --workers 4 > "$RUN/logs/retrieve-$JOB.log" 2>&1 \
  || { tail -n 20 "$RUN/logs/retrieve-$JOB.log"; exit 1; }
echo "== score $(date -u +%FT%TZ)"
status=0
gpu4 score --model "$BGE" --deadline-epoch "$DEADLINE" || status=$?
"$PY" "$SP" status --run-dir "$RUN"
case $status in
  0) echo "== done $(date -u +%FT%TZ): copy $RUN/results and $RUN/diagnostics back" ;;
  4) echo "== deadline checkpoint $(date -u +%FT%TZ): rerun with the same RUN_ID in a new allocation" ;;
  *) tail -n 20 "$RUN"/logs/score-worker*-"$JOB".log; exit 1 ;;
esac
exit $status
