#!/bin/bash
#SBATCH --job-name=geodml-fullrun-gpu
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --no-requeue
# The GPU tasks of the full-run ledger: both LLM2Vec views of the agent's queries and of the answers, on the validated
# four-A100 profile (analysis/docs/horeka-fullrun-embed.sh). Two hours, cap 3 GPU allocations (Valerian, 2026-10-08).
# Needs CODE and LEDGER. Exit 0: no GPU task left that may start; 4: deadline (resubmit).
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
LEDGER=${LEDGER:?LEDGER must point to the planned ledger}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1
END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
echo "fullrun gpu: commit $(git -C "$CODE" rev-parse HEAD) job $SLURM_JOB_ID node $(hostname) end $(date -u -d @"$END" +%FT%TZ)"
nvidia-smi -L
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun reconcile --ledger "$LEDGER" --job "$SLURM_JOB_ID")
rc=0
(cd "$CODE" && "$RT/bin/python" -u -m analysis.fullrun worker --ledger "$LEDGER" --gpu --cores 1 --end-epoch "$END" \
  --margin-minutes 10 --min-start-minutes 30 --job "$SLURM_JOB_ID") || rc=$?
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun status --ledger "$LEDGER") || true
exit "$rc"
