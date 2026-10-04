#!/bin/bash
#SBATCH --job-name=geodml-intent-stages
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Intent surfacing study (analysis/docs/intent_surfacing_study.md), CPU stages:
#   trace-extract -> replay -> [merge the answer views] -> analyze (once both query views are embedded).
# Each stage is skipped when its output exists, and analyze reuses finished strata from its cache,
# so resubmitting continues the work. An interrupted stage's .partial folder is renamed (never deleted).
# Optional: ANSWER_EXPORT, ANSWER_EMBED_QWEN, ANSWER_EMBED_MISTRAL (answer_readiness.py export and its two
# embed folders, merged here into this study's own folder), RANKING_CHANGE (report_axis_ranking_change outputs).
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="$W/reviews/page-readiness-20261004"
A="$W/geoaxis-archive"
SCRIPT="$CODE/analysis/scripts/intent_stages_study.py"
READINESS="$CODE/analysis/scripts/page_readiness_ordering.py"
CORPUS="$R/snippet-embeddings-corpus-v1"
TRACE="$R/intent-trace-extract-v1"
REPLAY="$R/intent-replay-v1"
OUT="$R/intent-stages-v1"
CPUS=${SLURM_CPUS_PER_TASK:-1}
echo "commit $(git -C "$CODE" rev-parse HEAD) job ${SLURM_JOB_ID:-none} start $(date -u +%FT%TZ)"
"$RT/bin/python" -c "import sys, numpy; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"

set_aside() {  # keep an interrupted attempt for inspection, out of the rerun's way
  if [ -d "$1.partial" ]; then
    mv "$1.partial" "$1.partial.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "set aside interrupted $1.partial"
  fi
}

if [ ! -f "$TRACE/manifest.json" ]; then
  set_aside "$TRACE"
  echo "== trace-extract $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" trace-extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --corpus-package "$CORPUS" \
    --population-prompts "$A/registration/population-registration-v1/population-prompts.jsonl" \
    --workers "$CPUS" --output "$TRACE"
fi
"$RT/bin/python" -c "import json, sys; m = json.load(open(sys.argv[1])); print('TRACE', json.dumps({k: m[k] for k in ('counts', 'consistency_failures', 'mean_pages_per_answer')}))" "$TRACE/manifest.json"

if [ ! -f "$REPLAY/manifest.json" ] && [ ! -f "$REPLAY.fidelity-failed.json" ]; then
  set_aside "$REPLAY"
  echo "== replay $(date -u +%FT%TZ)"
  # A failed fidelity gate stops only this stage (exit 2 plus its report); any other failure stops the job.
  "$RT/bin/python" -u "$SCRIPT" replay --trace-extract "$TRACE" --corpus-package "$CORPUS" \
    --dataset-root "$W/shared-hours/dataset" --dataset-root "$W/llama-hf/dataset" --search-root "$W" \
    --workers "$CPUS" --output "$REPLAY" || [ -f "$REPLAY.fidelity-failed.json" ]
fi
REPLAYED=()
if [ -f "$REPLAY/manifest.json" ]; then
  REPLAYED=(--replay "$REPLAY")
else
  echo "No replay stage: the fidelity gate failed ($REPLAY.fidelity-failed.json); the analysis runs without R0 and Rk."
fi

ANSWERS=()
if [ -n "${ANSWER_EXPORT:-}" ]; then
  for VIEW in qwen mistral; do
    if [ "$VIEW" = qwen ]; then SOURCE=${ANSWER_EMBED_QWEN:?set ANSWER_EMBED_QWEN}; else SOURCE=${ANSWER_EMBED_MISTRAL:?set ANSWER_EMBED_MISTRAL}; fi
    if [ ! -f "$R/answers-embed-$VIEW.merged/projection_manifest.json" ]; then
      set_aside "$R/answers-embed-$VIEW.merged"
      echo "== merge the $VIEW answer view $(date -u +%FT%TZ)"
      "$RT/bin/python" -u "$READINESS" merge --input "$SOURCE" --pages "$ANSWER_EXPORT/answers.jsonl.gz" \
        --output "$R/answers-embed-$VIEW.merged"
    fi
  done
  ANSWERS=(--answers-index "$ANSWER_EXPORT/observations.jsonl.gz" --answers-qwen "$R/answers-embed-qwen.merged"
           --answers-mistral "$R/answers-embed-mistral.merged")
fi

if [ ! -f "$R/embed-queries-qwen.merged/projection_manifest.json" ] || [ ! -f "$R/embed-queries-mistral.merged/projection_manifest.json" ]; then
  echo "Queries are not embedded yet: run horeka-intent-queries-interactive.sh in a four-A100 allocation, then resubmit."
  exit 0
fi
EXTRA=()
[ -f "$R/relocation-fresh.json" ] && EXTRA+=(--relocation "$R/relocation-fresh.json")
for DIR in ${RANKING_CHANGE:-}; do EXTRA+=(--ranking-change "$DIR"); done
if [ ! -f "$OUT/final-report.html" ]; then
  set_aside "$OUT"
  echo "== analyze $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$SCRIPT" analyze --trace-extract "$TRACE" ${REPLAYED[@]+"${REPLAYED[@]}"} --corpus-package "$CORPUS" \
    --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" \
    --queries-qwen "$R/embed-queries-qwen.merged" --queries-mistral "$R/embed-queries-mistral.merged" \
    ${ANSWERS[@]+"${ANSWERS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
    --bootstrap 200 --permutations 200 --workers "$CPUS" --output "$OUT"
fi
echo "REPORT $OUT/final-report.html end $(date -u +%FT%TZ)"
