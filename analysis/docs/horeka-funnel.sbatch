#!/bin/bash
#SBATCH --job-name=geodml-funnel
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Funnel study (analysis/docs/funnel_study.md): extract -> replay -> assemble -> analyze (SHARD=k/n) -> report.
# Each stage is skipped when its output exists; analyze reuses finished tasks from its cache, so resubmitting
# continues the work. Interrupted .partial folders are set aside (never deleted).
# Needs: CODE (pinned clean checkout), the feature table at $R/funnel-features-v1 (built on the Mac).
# Optional: SHARD (default 1/1), SPECS (default main,visible,complete), SNAPSHOT_DDG / SNAPSHOT_SEARXNG.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="$W/reviews/page-readiness-20261004"
A="$W/geoaxis-archive"
SCRIPT="$CODE/analysis/scripts/funnel_study.py"
FEATURES="$R/funnel-features-v1"
EXTRACT="$R/funnel-extract-v1"
REPLAY="$R/funnel-replay-v1"
ASSEMBLED="$R/funnel-assembled-v1"
ANALYSIS="$R/funnel-analysis-v1"
REPORT="$R/funnel-report-v1"
SHARD=${SHARD:-1/1}
SPECS=${SPECS:-main,visible,complete}
CPUS=${SLURM_CPUS_PER_TASK:-1}
echo "commit $(git -C "$CODE" rev-parse HEAD) job ${SLURM_JOB_ID:-none} shard $SHARD start $(date -u +%FT%TZ)"
test -f "$FEATURES/manifest.json" || { echo "missing $FEATURES: fetch the Mac feature table first" >&2; exit 2; }

find_snapshot() {  # the snapshot mirrored next to the datasets (the extract checks its sha256 against the traces)
  find "$W/shared-hours/dataset/artifacts" "$W/llama-hf/dataset/artifacts" -name "$1" -type f 2>/dev/null | head -n 1
}
DDG=${SNAPSHOT_DDG:-$(find_snapshot phase0_top20_ddg.parquet)}
SEARXNG=${SNAPSHOT_SEARXNG:-$(find_snapshot phase0_top20_searxng.parquet)}
test -f "$DDG" && test -f "$SEARXNG" || { echo "snapshots not found; set SNAPSHOT_DDG and SNAPSHOT_SEARXNG" >&2; exit 2; }
SNAPS=(--snapshot "duckduckgo=$DDG" --snapshot "searxng=$SEARXNG")

set_aside() {
  if [ -d "$1.partial" ]; then
    mv "$1.partial" "$1.partial.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "set aside interrupted $1.partial"
  fi
}

"$RT/bin/python" -u "$CODE/analysis/scripts/funnel_features.py" verify --features "$FEATURES" \
  --snapshot-ddg "$DDG" --snapshot-searxng "$SEARXNG"

if [ ! -f "$EXTRACT/manifest.json" ]; then
  set_aside "$EXTRACT"
  echo "== extract $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    "${SNAPS[@]}" --final-axis-map "$A/final-audit/final-axis-map.jsonl" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" \
    --workers "$CPUS" --output "$EXTRACT"
fi
if [ ! -f "$REPLAY/manifest.json" ]; then
  set_aside "$REPLAY"
  echo "== replay $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" replay "${SNAPS[@]}" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" --output "$REPLAY"
fi
if [ ! -f "$ASSEMBLED/manifest.json" ]; then
  set_aside "$ASSEMBLED"
  echo "== assemble $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" assemble --extract "$EXTRACT" --features "$FEATURES" --replay "$REPLAY" \
    --corpus-package "$R/snippet-embeddings-corpus-v1" \
    --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --output "$ASSEMBLED"
fi
echo "== analyze shard $SHARD $(date -u +%FT%TZ)"
"$RT/bin/python" -u "$SCRIPT" analyze --assembled "$ASSEMBLED" --output "$ANALYSIS" --specs "$SPECS" \
  --shard "$SHARD" --workers "$CPUS"
if [ ! -f "$REPORT/final-report.html" ]; then
  echo "== report (exit 3 = other shards still missing) $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" report --assembled "$ASSEMBLED" --output "$ANALYSIS" --specs "$SPECS" --report "$REPORT" || [ $? -eq 3 ]
fi
echo "end $(date -u +%FT%TZ)"
