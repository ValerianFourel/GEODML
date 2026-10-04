#!/bin/bash
#SBATCH --job-name=geodml-page-readiness-embed
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Embed one LLM2Vec view of the page texts (and the prompt re-embedding sample)
# with one worker per GPU. Shards are atomic: resubmitting the same command
# resumes and never repeats a finished shard. Required environment (exported by
# the submitting shell): CODE PY VIEW MAP MODEL MNTP PEFT PAGES OUT [SAMPLE SAMPLE_OUT].
set -euo pipefail
: "${CODE:?}" "${PY:?}" "${VIEW:?}" "${MAP:?}" "${MODEL:?}" "${MNTP:?}" "${PEFT:?}" "${PAGES:?}" "${OUT:?}"
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "commit $(git -C "$CODE" rev-parse HEAD) job $SLURM_JOB_ID node $(hostname) view $VIEW start $(date -u +%FT%TZ)"

run_view() {  # pages output
  local pids=()
  for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$CODE/analysis/scripts/page_readiness_ordering.py" embed \
      --pages "$1" --view "$VIEW" --map "$MAP" --embedding-model "$MODEL" \
      --mntp-model "$MNTP" --peft-model "$PEFT" --workers 4 --worker-index "$GPU" \
      --output "$2" > "$2.worker$GPU.$SLURM_JOB_ID.log" 2>&1 &
    pids+=($!)
  done
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return $status
}

mkdir -p "$OUT"
if [ -n "${SAMPLE:-}" ]; then
  mkdir -p "$SAMPLE_OUT"
  run_view "$SAMPLE" "$SAMPLE_OUT"
fi
run_view "$PAGES" "$OUT"
echo "end $(date -u +%FT%TZ)"
