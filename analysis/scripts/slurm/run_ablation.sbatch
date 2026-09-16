#!/bin/bash -l
# One ablation chunk: one model, one treatment, both frames, all keywords.
# Does not self-resubmit. Any retry needs a fresh estimate and wall-time approval.
#
# Required exports: MODEL, TREATMENT
# Optional:         SAMPLE_N (default huge so all eligible kws are taken),
#                   FRAME (default both)
#
# Submit (or use scripts/slurm/dispatch_all.sh):
#   sbatch --account=$JUWELS_ACCOUNT \
#          --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,TREATMENT=T7_source_earned \
#          scripts/slurm/run_ablation.sbatch

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=05:55:00
#SBATCH --job-name=geodml-abl
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Node failures, wall-time limits, and transient errors never submit another job.
# Preserve checkpoints and obtain fresh wall-time approval before any retry.

# shellcheck source=scripts/slurm/_common.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_common.sh"

: "${MODEL:?MODEL must be set, e.g. meta-llama/Llama-3.3-70B-Instruct}"
: "${TREATMENT:?TREATMENT must be set, e.g. T7_source_earned}"
: "${SAMPLE_N:=100000}"
: "${FRAME:=both}"
: "${PROMPT_VARIANT:=biased}"
export PROMPT_VARIANT

MODEL_TAG="${MODEL##*/}"
OUTDIR="$SLURM_SUBMIT_DIR/interpretability/output/ablation_${TREATMENT}_${MODEL_TAG}_${PROMPT_VARIANT}"
mkdir -p "$OUTDIR"

echo "[ablation] model=$MODEL treatment=$TREATMENT frame=$FRAME sample_n=$SAMPLE_N variant=$PROMPT_VARIANT"
echo "[ablation] outdir=$OUTDIR"

set +e
python -m interpretability.ablation \
  --backend local \
  --models "$MODEL" \
  --treatments "$TREATMENT" \
  --sample-n "$SAMPLE_N" \
  --frame "$FRAME" \
  --resume \
  --output-dir "$OUTDIR"
RC=$?
set -e

echo "[ablation] python rc=$RC"
if [ "$RC" -eq 0 ]; then
  touch "$OUTDIR/.done_${MODEL_TAG}_${TREATMENT}"
  echo "[ablation] complete; no new allocation submitted."
  exit 0
fi

# Compatibility helper reports manual recovery only; it never calls sbatch.
chain_resubmit "$0" \
  MODEL="$MODEL" TREATMENT="$TREATMENT" SAMPLE_N="$SAMPLE_N" FRAME="$FRAME" \
  PROMPT_VARIANT="$PROMPT_VARIANT" \
  || exit "$RC"
exit 0
