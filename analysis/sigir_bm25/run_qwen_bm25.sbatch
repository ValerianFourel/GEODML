#!/bin/bash
#SBATCH --job-name=geodml-sigir-qwen-bm25
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# SIGIR BM25 baseline: Qwen3.8-27B generation with BM25 retrieval, natural condition, one four-A100 allocation.
# horeka_qwen.py starts the original Qwen bouts' vLLM server (frozen profile from BOUT_CONFIG) and runs generate.py.
# Submit (smoke, dev_accelerated, 1 h):
#   sbatch --account=<account> --export=ALL,CODE=<clean checkout>,BUNDLE=<bundle>,RUN_ID=<id>,BOUT_CONFIG=<config.json>
#     run_qwen_bm25.sbatch
# Production: add --partition=accelerated --time=<approved walltime>. Optional exports: RETRIEVER (bm25 | orig),
#   SCORE_CACHE (score-cache parquet), PHASES (space-separated REQUEST:CELL[:SECONDS], default "4:12"), LIMIT.
# Resumable: resubmit with the same RUN_ID; finished cells are skipped. Exit 4 = deadline checkpoint.
# Chain: --dependency=afternotok:<previous> --kill-on-invalid-dep=yes. Exit 4 starts the next job, exit 0 (all cells
#   done) cancels the rest; any other failure writes $RUN/CHAIN_STOP and the next job exits at once.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=${CODE:?CODE must point to the clean, pinned checkout}
BUNDLE=${BUNDLE:?BUNDLE must point to the copied bundle directory}
RUN_ID=${RUN_ID:?RUN_ID, e.g. qwen-bm25-20261023}
BOUT_CONFIG=${BOUT_CONFIG:?BOUT_CONFIG: an existing Qwen bout config.json (frozen serving setup)}
JOB=${SLURM_JOB_ID:?run under sbatch or inside a salloc shell}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
PY="$W/environment/qwen-runtime/bin/python"
RUN="$W/runs/sigir-bm25/$RUN_ID"
mkdir -p "$RUN/logs"
echo "sigir qwen bm25: commit $(git -C "$CODE" rev-parse HEAD) job $JOB host $(hostname) start $(date -u +%FT%TZ)"
if [ -s "$RUN/CHAIN_STOP" ]; then
  echo "stopped by an earlier job, nothing run (delete $RUN/CHAIN_STOP to go on):"; cat "$RUN/CHAIN_STOP"; exit 0
fi
stop() { echo "$(date -u +%FT%TZ) job $JOB: $1" | tee -a "$RUN/CHAIN_STOP" >&2; }
ARGS=(execute --bout-config "$BOUT_CONFIG" --bundle "$BUNDLE" --run-dir "$RUN" --retriever "${RETRIEVER:-bm25}")
[ -n "${SCORE_CACHE:-}" ] && ARGS+=(--score-cache "$SCORE_CACHE")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
for phase in ${PHASES:-4:12}; do ARGS+=(--phase "$phase"); done
status=0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" "$PY" -u "$CODE/analysis/sigir_bm25/horeka_qwen.py" "${ARGS[@]}" || status=$?
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" "$PY" "$CODE/analysis/sigir_bm25/generate.py" status \
  --bundle "$BUNDLE" --run-dir "$RUN" --model qwen38 || true
case $status in
  0) echo "== done $(date -u +%FT%TZ)" ;;
  4) echo "== deadline checkpoint $(date -u +%FT%TZ): resubmit with the same RUN_ID" ;;
  *) stop "horeka_qwen.py exit $status (see $RUN/attempts/job$JOB/)" ;;
esac
echo "end $(date -u +%FT%TZ) exit $status"
exit $status
