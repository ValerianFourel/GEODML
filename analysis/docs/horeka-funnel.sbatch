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
# Each stage is skipped when its output exists; analyze reuses finished tasks and checkpointed fits from its
# cache, so resubmitting continues the work. Interrupted .partial folders are set aside (never deleted).
# MODE=prep runs extract, replay and assemble only; MODE=analyze requires the assembled tables (so parallel
# shards never build them concurrently); MODE=all (default) does both.
# Analyze starts no new fit later than MARGIN_MIN minutes before Slurm's end time for this job; exit 4 is
# that deadline checkpoint (finished fits saved; resubmit the same shard), 0 full completion, else failure.
# Needs: CODE (pinned clean checkout), the feature table at $R/funnel-features-v1 (built on the Mac).
# Optional: SHARD (default 1/1), SPECS (default main,visible,complete), SPLIT, MODE, MARGIN_MIN (default 12),
# SNAPSHOT_DDG / SNAPSHOT_SEARXNG.
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
SPLIT=${SPLIT:-confirmation}  # addendum A1: confirmatory results use the held-out keywords
MODE=${MODE:-all}
MARGIN_MIN=${MARGIN_MIN:-12}  # one cold scored-candidate fit on the confirmation split, plus report and cleanup
case "$MODE" in prep|analyze|all) ;; *) echo "MODE must be prep, analyze or all" >&2; exit 2 ;; esac
CPUS=${SLURM_CPUS_PER_TASK:-1}
echo "commit $(git -C "$CODE" rev-parse HEAD) job ${SLURM_JOB_ID:-none} mode $MODE shard $SHARD split $SPLIT start $(date -u +%FT%TZ)"
test -f "$FEATURES/manifest.json" || { echo "missing $FEATURES: fetch the Mac feature table first" >&2; exit 2; }

find_snapshot() {  # the snapshot mirrored next to the datasets (the extract checks its sha256 against the traces)
  find "$W/shared-hours/dataset/artifacts" "$W/llama-hf/dataset/artifacts" -name "$1" -type f 2>/dev/null | head -n 1 || true
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

minutes_left() {  # minutes until this job's Slurm end time, minus the margin (45 outside Slurm)
  local end
  end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null || true)
  if [ -n "$end" ] && [ "$end" != "N/A" ]; then
    echo $(( ($(date -d "$end" +%s) - $(date +%s)) / 60 - MARGIN_MIN ))
  else
    echo 45
  fi
}

if [ "$MODE" = analyze ]; then
  test -f "$ASSEMBLED/manifest.json" || { echo "missing $ASSEMBLED: run MODE=prep first" >&2; exit 2; }
fi
if [ "$MODE" != analyze ] && [ ! -f "$EXTRACT/manifest.json" ]; then
  set_aside "$EXTRACT"
  echo "== extract $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    "${SNAPS[@]}" --final-axis-map "$A/final-audit/final-axis-map.jsonl" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" \
    --workers "$CPUS" --output "$EXTRACT"
fi
if [ "$MODE" != analyze ] && [ ! -f "$REPLAY/manifest.json" ]; then
  set_aside "$REPLAY"
  echo "== replay $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" replay "${SNAPS[@]}" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" --output "$REPLAY"
fi
if [ "$MODE" != analyze ] && [ ! -f "$ASSEMBLED/manifest.json" ]; then
  set_aside "$ASSEMBLED"
  echo "== assemble $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" assemble --extract "$EXTRACT" --features "$FEATURES" --replay "$REPLAY" \
    --corpus-package "$R/snippet-embeddings-corpus-v1" \
    --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --output "$ASSEMBLED"
fi
if [ "$MODE" = prep ]; then
  echo "end (prep) $(date -u +%FT%TZ)"
  exit 0
fi
STOP=$(minutes_left)
echo "== analyze shard $SHARD, new fits for $STOP min $(date -u +%FT%TZ)"
rc=0
"$RT/bin/python" -u "$SCRIPT" analyze --assembled "$ASSEMBLED" --output "$ANALYSIS" --specs "$SPECS" --split "$SPLIT" \
  --shard "$SHARD" --workers "$CPUS" --stop-after-minutes "$STOP" || rc=$?
if [ "$rc" -eq 4 ]; then
  echo "deadline checkpoint $(date -u +%FT%TZ): finished fits are saved; resubmit the same shard to continue"
  exit 4
fi
[ "$rc" -eq 0 ] || exit "$rc"
if [ ! -f "$REPORT/final-report.html" ]; then
  echo "== report (exit 3 = other shards still missing) $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" report --assembled "$ASSEMBLED" --output "$ANALYSIS" --specs "$SPECS" --split "$SPLIT" \
    --report "$REPORT" || [ $? -eq 3 ]
fi
echo "end $(date -u +%FT%TZ)"
