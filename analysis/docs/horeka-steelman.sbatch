#!/bin/bash
#SBATCH --job-name=geodml-steelman
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Steelman of the shortlist claim (analysis/steelman/PREREG.md, addendum B1) on HoreKa: both models, the
# held-out confirmation keywords, from existing traces only (no inference).
# Prerequisites, each built here with the same command as the existing runners when its output is missing
# (skipped when present, so this job and horeka-funnel.sbatch / horeka-intent-stages.sbatch share them):
#   funnel extract -> replay -> assemble ($R/funnel-{extract,replay,assembled}-v1), intent trace-extract
#   ($R/intent-trace-extract-v1, the agent's queries). Extract and trace-extract are not resumable: an
#   interrupted attempt is set aside and restarts in the next job.
# Steelman parts: chain, followup, pairs, generator, fe, lexical (BM25 selector; shortlisting refits skipped),
# report; each skipped when its JSON exists. generator and fe checkpoint every replicate / stratum and stop
# before Slurm's end time: exit 4 = deadline checkpoint, resubmit the same command; 0 = everything done.
# Needs: CODE (pinned clean checkout), $R/funnel-features-v1. Optional: SPLIT (default confirmation),
# OUT (default $R/steelman-$SPLIT-v1), MARGIN_MIN (default 10), GEN_WORKERS (default 16).
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
R="$W/reviews/page-readiness-20261004"
A="$W/geoaxis-archive"
FUNNEL="$CODE/analysis/scripts/funnel_study.py"
STAGES="$CODE/analysis/scripts/intent_stages_study.py"
FEATURES="$R/funnel-features-v1"; CORPUS="$R/snippet-embeddings-corpus-v1"
EXTRACT="$R/funnel-extract-v1"; REPLAY="$R/funnel-replay-v1"; ASSEMBLED="$R/funnel-assembled-v1"; TRACE="$R/intent-trace-extract-v1"
POP="$A/registration/population-registration-v1/population-prompts.jsonl"; AXIS="$A/final-audit/final-axis-map.jsonl"
SPLIT=${SPLIT:-confirmation}
OUT=${OUT:-$R/steelman-$SPLIT-v1}
MARGIN_MIN=${MARGIN_MIN:-10}
GEN_WORKERS=${GEN_WORKERS:-16}  # memory: one Monte-Carlo chunk per worker (~2 GB at the full confirmation split)
CPUS=${SLURM_CPUS_PER_TASK:-1}
COMMIT=$(git -C "$CODE" rev-parse HEAD)
echo "steelman commit $COMMIT job ${SLURM_JOB_ID:-none} split $SPLIT out $OUT start $(date -u +%FT%TZ)"
"$RT/bin/python" -c "import sys, numpy, scipy, pandas; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
df -h "$W" | tail -n 1

minutes_left() {
  local end
  end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null || true)
  if [ -n "$end" ] && [ "$end" != "N/A" ]; then echo $(( ($(date -d "$end" +%s) - $(date +%s)) / 60 - MARGIN_MIN )); else echo 45; fi
}
need_time() {
  local left; left=$(minutes_left)
  if [ "$left" -lt "$1" ]; then echo "deadline checkpoint: $left min left before the margin; resubmit $(date -u +%FT%TZ)"; exit 4; fi
}
set_aside() {
  if [ -d "$1.partial" ]; then mv "$1.partial" "$1.partial.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"; echo "set aside interrupted $1.partial"; fi
}
find_snapshot() {
  find "$W/shared-hours/dataset/artifacts" "$W/llama-hf/dataset/artifacts" -name "$1" -type f 2>/dev/null | head -n 1 || true
}

# ---------------------------------------------------------------- prerequisites (shared with the funnel and intent-stages runners)
test -f "$FEATURES/manifest.json" || { echo "missing $FEATURES (fetch derived/funnel-features-v1, HF 5f89299a)" >&2; exit 2; }
if [ ! -f "$EXTRACT/manifest.json" ] || [ ! -f "$REPLAY/manifest.json" ] || [ ! -f "$ASSEMBLED/manifest.json" ]; then
  DDG=${SNAPSHOT_DDG:-$(find_snapshot phase0_top20_ddg.parquet)}
  SEARXNG=${SNAPSHOT_SEARXNG:-$(find_snapshot phase0_top20_searxng.parquet)}
  test -f "$DDG" && test -f "$SEARXNG" || { echo "snapshots not found; set SNAPSHOT_DDG and SNAPSHOT_SEARXNG" >&2; exit 2; }
  SNAPS=(--snapshot "duckduckgo=$DDG" --snapshot "searxng=$SEARXNG")
fi
if [ ! -f "$EXTRACT/manifest.json" ]; then
  need_time 40; set_aside "$EXTRACT"; echo "== funnel extract $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$FUNNEL" extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    "${SNAPS[@]}" --final-axis-map "$AXIS" --population-prompts "$POP" --workers "$CPUS" --output "$EXTRACT"
fi
if [ ! -f "$REPLAY/manifest.json" ]; then
  need_time 10; set_aside "$REPLAY"; echo "== funnel replay $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$FUNNEL" replay "${SNAPS[@]}" --population-prompts "$POP" --output "$REPLAY"
fi
if [ ! -f "$ASSEMBLED/manifest.json" ]; then
  need_time 20; set_aside "$ASSEMBLED"; echo "== funnel assemble $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$FUNNEL" assemble --extract "$EXTRACT" --features "$FEATURES" --replay "$REPLAY" \
    --corpus-package "$CORPUS" --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --output "$ASSEMBLED"
fi
if [ ! -f "$TRACE/manifest.json" ]; then
  need_time 40; set_aside "$TRACE"; echo "== intent trace-extract (the agent's queries) $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$STAGES" trace-extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    --final-axis-map "$AXIS" --corpus-package "$CORPUS" --population-prompts "$POP" --workers "$CPUS" --output "$TRACE"
fi
"$RT/bin/python" -c "import json, sys; [print(p.split('/')[-2], json.dumps(json.load(open(p)).get('counts', {}))[:400]) for p in sys.argv[1:]]" \
  "$EXTRACT/manifest.json" "$ASSEMBLED/manifest.json"

# ---------------------------------------------------------------- steelman parts
ARGS=(--assembled "$ASSEMBLED" --extract "$EXTRACT" --replay "$REPLAY" --trace-extract "$TRACE" --features "$FEATURES"
      --prompts "$POP" --split "$SPLIT" --input-root "$R" --output "$OUT")
part() {  # part name, workers, minimum minutes to start
  local name=$1 workers=$2 minimum=$3 rc=0
  [ -f "$OUT/$name.json" ] && { echo "== $name: done"; return 0; }
  need_time "$minimum"
  echo "== steelman $name, $(minutes_left) min before the margin $(date -u +%FT%TZ)"
  "$RT/bin/python" -u -m analysis.steelman "$name" "${ARGS[@]}" "${@:4}" --workers "$workers" \
    --stop-after-minutes "$(minutes_left)" > >(tee -a "$OUT.$name.log") 2>&1 || rc=$?
  [ "$rc" -eq 4 ] && { echo "deadline checkpoint in $name (finished units saved); resubmit"; exit 4; }
  [ "$rc" -eq 0 ] || { echo "steelman $name failed with $rc" >&2; exit "$rc"; }
}
mkdir -p "$OUT"
part chain 1 15
part followup 1 10
part pairs 1 10
part generator "$GEN_WORKERS" 8
part fe "$CPUS" 8
part lexical 1 15 --selection-draws 0
"$RT/bin/python" -u -m analysis.steelman report --output "$OUT"
TARBALL="$W/reviews/steelman-$SPLIT-${COMMIT:0:7}-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -C "$(dirname "$OUT")" --exclude='*.cache' -czf "$TARBALL" "$(basename "$OUT")" $(cd "$(dirname "$OUT")" && ls -d "$(basename "$OUT")".*.log 2>/dev/null)
echo "STEELMAN DONE $OUT/RESULTS.md  tarball $TARBALL  end $(date -u +%FT%TZ)"
