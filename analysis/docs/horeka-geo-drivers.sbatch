#!/bin/bash
#SBATCH --job-name=geodml-geo-drivers
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# GEO drivers study (analysis/docs/geo_drivers_study.md): assemble, then analyze. CPU only.
# Each stage is skipped when its output exists; rerunning continues after an interruption.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
R="$W/reviews/page-readiness-20261004"
A="$W/geoaxis-archive"
SCRIPT="$CODE/analysis/scripts/geo_drivers_study.py"
echo "commit $(git -C "$CODE" rev-parse HEAD) job ${SLURM_JOB_ID:-none} start $(date -u +%FT%TZ)"
if [ ! -f "$R/geo-drivers-assembled-v1/manifest.json" ]; then
  "$RT/bin/python" -u "$SCRIPT" assemble --extract "$R/extract-all-v1" --corpus-package "$R/snippet-embeddings-corpus-v1" \
    --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" \
    --output "$R/geo-drivers-assembled-v1"
fi
if [ ! -f "$R/geo-drivers-v1/report.html" ]; then
  "$RT/bin/python" -u "$SCRIPT" analyze --assembled "$R/geo-drivers-assembled-v1" --bootstrap 200 --permutations 200 \
    --workers "${SLURM_CPUS_PER_TASK:-1}" --output "$R/geo-drivers-v1"
fi
echo "REPORT $R/geo-drivers-v1/report.html end $(date -u +%FT%TZ)"
