#!/bin/bash
# Finish every computation the paper needs inside one existing HoreKa allocation (one node, four A100s),
# and end with one results bundle. Usage, from a login shell other than the one holding the allocation:
#   bash horeka-paper-interactive.sh SLURM_JOB_ID
# Every stage is skipped when its output exists; long fits are checkpointed and stop before the
# allocation's end (exit 4 = deadline checkpoint: rerun the same command in the next allocation).
# Stages, in order:
#   1 funnel (CPU): verify features, extract, replay, assemble, analyze (confirmation split, main
#     specification first, then the secondary ones), report
#   2 generator decisions (CPU): which shown snippets each answer keeps, and how it orders them
#   3 intent stages (CPU): trace extract, replay, Gemma extract (development) when GEMMA_RUN exists
#   4 queries (GPU): both LLM2Vec views of the AI's search queries
#   5 answers (CPU + GPU): export the natural answers, embed both views, place them on the axis
#   6 intent-stages analysis (CPU) with queries, answers and Gemma
#   7 bundle: paper_results.py -> $R/paper-results-<commit>/ and a tarball in $W/reviews
# Never cancels, extends or releases the allocation. Optional: GEMMA_RUN, MARGIN_MIN (default 8), CPUS.
JOB=${1:?usage: horeka-paper-interactive.sh SLURM_JOB_ID}
LOG_NAME=paper
source "$(dirname -- "$0")/horeka-page-readiness-lib.sh"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FUNNEL="$CODE/analysis/scripts/funnel_study.py"
STAGES="$CODE/analysis/scripts/intent_stages_study.py"
DECISIONS="$CODE/analysis/scripts/funnel_importance.py"
ANSWERS="$CODE/analysis/scripts/answer_readiness.py"
CORPUS="$R/snippet-embeddings-corpus-v1"
POP="$A/registration/population-registration-v1/population-prompts.jsonl"
AXIS="$A/final-audit/final-axis-map.jsonl"
MARGIN_MIN=${MARGIN_MIN:-8}
GEMMA_RUN=${GEMMA_RUN:-$W/reviews/gemma-si-v4-llama-reuse-5h-20261004}
CPUS=${CPUS:-$(scontrol show job "$JOB" 2>/dev/null | grep -o 'NumCPUs=[0-9]*' | head -n 1 | cut -d= -f2 || true)}
CPUS=${CPUS:-32}
COMMIT=$(git -C "$CODE" rev-parse --short HEAD)

cpu() {  # CPU step on the allocated node with every allocated core
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --gres=none "$@"
}
minutes_left() {  # minutes until the allocation ends, minus the margin
  local end
  end=$(squeue -h -j "$JOB" -o %e 2>/dev/null || true)
  if [ -n "$end" ] && [ "$end" != "N/A" ]; then
    echo $(( ($(date -d "$end" +%s) - $(date +%s)) / 60 - MARGIN_MIN ))
  else
    echo 45
  fi
}
need_time() {  # stop cleanly when fewer than $1 minutes remain
  local left
  left=$(minutes_left)
  if [ "$left" -lt "$1" ]; then
    echo "deadline checkpoint: $left min left before the margin; rerun in the next allocation $(date -u +%FT%TZ)"
    exit 4
  fi
}
set_aside() {
  if [ -d "$1.partial" ]; then
    mv "$1.partial" "$1.partial.interrupted-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "set aside interrupted $1.partial"
  fi
}
find_snapshot() {
  find "$W/shared-hours/dataset/artifacts" "$W/llama-hf/dataset/artifacts" -name "$1" -type f 2>/dev/null | head -n 1 || true
}
echo "paper runner: $CPUS CPUs, $(minutes_left) min of fits before the margin, Gemma run $([ -d "$GEMMA_RUN/shards" ] && echo present || echo absent)"

# ---------------------------------------------------------------- 1 funnel, confirmation split
FEATURES="$R/funnel-features-v1"
test -f "$FEATURES/manifest.json" || { echo "missing $FEATURES: fetch derived/funnel-features-v1 (HF 5f89299a) first" >&2; exit 2; }
DDG=${SNAPSHOT_DDG:-$(find_snapshot phase0_top20_ddg.parquet)}
SEARXNG=${SNAPSHOT_SEARXNG:-$(find_snapshot phase0_top20_searxng.parquet)}
test -f "$DDG" && test -f "$SEARXNG" || { echo "snapshots not found; set SNAPSHOT_DDG and SNAPSHOT_SEARXNG" >&2; exit 2; }
SNAPS=(--snapshot "duckduckgo=$DDG" --snapshot "searxng=$SEARXNG")
cpu "$RT/bin/python" -u "$CODE/analysis/scripts/funnel_features.py" verify --features "$FEATURES" --snapshot-ddg "$DDG" --snapshot-searxng "$SEARXNG"
F_EXTRACT="$R/funnel-extract-v1"; F_REPLAY="$R/funnel-replay-v1"; F_ASSEMBLED="$R/funnel-assembled-v1"
F_ANALYSIS="$R/funnel-analysis-v1"; F_REPORT="$R/funnel-report-v1"
if [ ! -f "$F_EXTRACT/manifest.json" ]; then
  need_time 30; set_aside "$F_EXTRACT"; echo "== funnel extract $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$FUNNEL" extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    "${SNAPS[@]}" --final-axis-map "$AXIS" --population-prompts "$POP" --workers "$CPUS" --output "$F_EXTRACT"
fi
if [ ! -f "$F_REPLAY/manifest.json" ]; then
  need_time 10; set_aside "$F_REPLAY"; echo "== funnel replay $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$FUNNEL" replay "${SNAPS[@]}" --population-prompts "$POP" --output "$F_REPLAY"
fi
if [ ! -f "$F_ASSEMBLED/manifest.json" ]; then
  need_time 20; set_aside "$F_ASSEMBLED"; echo "== funnel assemble $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$FUNNEL" assemble --extract "$F_EXTRACT" --features "$FEATURES" --replay "$F_REPLAY" \
    --corpus-package "$CORPUS" --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --output "$F_ASSEMBLED"
fi
if [ ! -f "$F_REPORT/results.json" ]; then
  for SPECS in main main,visible,complete; do  # the confirmatory family (main) first, robustness afterwards
    need_time 5; echo "== funnel analyze $SPECS, $(minutes_left) min $(date -u +%FT%TZ)"
    rc=0
    cpu "$RT/bin/python" -u "$FUNNEL" analyze --assembled "$F_ASSEMBLED" --output "$F_ANALYSIS" --specs "$SPECS" --split confirmation \
      --workers "$CPUS" --stop-after-minutes "$(minutes_left)" || rc=$?
    [ "$rc" -eq 4 ] && { echo "deadline checkpoint in funnel analyze ($SPECS); rerun in the next allocation"; exit 4; }
    [ "$rc" -eq 0 ] || exit "$rc"
  done
  cpu "$RT/bin/python" -u "$FUNNEL" report --assembled "$F_ASSEMBLED" --output "$F_ANALYSIS" --specs main,visible,complete \
    --split confirmation --report "$F_REPORT"
fi

# ---------------------------------------------------------------- 2 the generator's keep and order decisions
DEC="$R/funnel-decisions-v1"
if [ ! -f "$DEC/decisions.json" ]; then
  need_time 5; echo "== generator decisions, $(minutes_left) min $(date -u +%FT%TZ)"
  rc=0
  cpu "$RT/bin/python" -u "$DECISIONS" --assembled "$F_ASSEMBLED" --split confirmation --bootstrap 100 --permutations 100 \
    --workers "$CPUS" --stop-after-minutes "$(minutes_left)" --output "$DEC" || rc=$?
  [ "$rc" -eq 4 ] && { echo "deadline checkpoint in generator decisions; rerun in the next allocation"; exit 4; }
  [ "$rc" -eq 0 ] || exit "$rc"
fi

# ---------------------------------------------------------------- 3 intent stages: traces, replay, Gemma
TRACE="$R/intent-trace-extract-v1"; I_REPLAY="$R/intent-replay-v1"; GEMMA_OUT="$R/intent-gemma-extract-v1"
if [ ! -f "$TRACE/manifest.json" ]; then
  need_time 30; set_aside "$TRACE"; echo "== intent trace-extract $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$STAGES" trace-extract --source "$W/llama-hf/dataset:llama4" --source "$W/shared-hours/dataset:qwen38" \
    --final-axis-map "$AXIS" --corpus-package "$CORPUS" --population-prompts "$POP" --workers "$CPUS" --output "$TRACE"
fi
if [ ! -f "$I_REPLAY/manifest.json" ] && [ ! -f "$I_REPLAY.fidelity-failed.json" ]; then
  need_time 15; set_aside "$I_REPLAY"; echo "== intent replay $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$STAGES" replay --trace-extract "$TRACE" --corpus-package "$CORPUS" \
    --dataset-root "$W/shared-hours/dataset" --dataset-root "$W/llama-hf/dataset" --search-root "$W" \
    --workers "$CPUS" --output "$I_REPLAY" || [ -f "$I_REPLAY.fidelity-failed.json" ]
fi
GEMMA=()
if [ -d "$GEMMA_RUN/shards" ]; then
  if [ ! -f "$GEMMA_OUT/manifest.json" ]; then
    need_time 10; set_aside "$GEMMA_OUT"; echo "== gemma-extract (development judgments) $(date -u +%FT%TZ)"
    cpu "$RT/bin/python" -u "$STAGES" gemma-extract --gemma-run "$GEMMA_RUN" --corpus-package "$CORPUS" --workers "$CPUS" --output "$GEMMA_OUT"
  fi
  GEMMA=(--gemma "$GEMMA_OUT")
fi

# ---------------------------------------------------------------- 4 queries on the GPUs
need_time 15
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$TRACE/queries.jsonl.gz" "$R/embed-queries-$VIEW"
done

# ---------------------------------------------------------------- 5 answers: export, embed, place on the axis
EXPORT="$R/answers-export-v1"
if [ ! -f "$EXPORT/manifest.json" ]; then
  need_time 20; set_aside "$EXPORT"; echo "== answer export (natural condition) $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$ANSWERS" export --dataset "llama4=$W/llama-hf/dataset" --dataset "qwen38=$W/shared-hours/dataset" \
    --axis-map "$AXIS" --output "$EXPORT"
fi
need_time 15
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$EXPORT/answers.jsonl.gz" "$R/answers-embed-$VIEW"
done
ANS_ANALYSIS="$R/answers-analysis-v1"
if [ ! -f "$ANS_ANALYSIS/results.json" ] && [ ! -d "$ANS_ANALYSIS" ]; then
  need_time 10; echo "== answers on the axis $(date -u +%FT%TZ)"
  cpu "$RT/bin/python" -u "$ANSWERS" analyze --export "$EXPORT" --qwen "$R/answers-embed-qwen.merged" --mistral "$R/answers-embed-mistral.merged" \
    --battery "$A/battery" --axis-map "$AXIS" --output "$ANS_ANALYSIS"
fi

# ---------------------------------------------------------------- 6 intent-stages analysis
I_OUT="$R/intent-stages-v1"
if [ ! -f "$I_OUT/final-report.html" ]; then
  need_time 20; set_aside "$I_OUT"; echo "== intent-stages analyze $(date -u +%FT%TZ)"
  REPLAYED=(); [ -f "$I_REPLAY/manifest.json" ] && REPLAYED=(--replay "$I_REPLAY")
  EXTRA=(); [ -f "$R/relocation-fresh.json" ] && EXTRA+=(--relocation "$R/relocation-fresh.json")
  cpu "$RT/bin/python" -u "$STAGES" analyze --trace-extract "$TRACE" ${REPLAYED[@]+"${REPLAYED[@]}"} --corpus-package "$CORPUS" \
    --qwen-prompts "$A/final-audit/projections/qwen" --mistral-prompts "$A/final-audit/projections/mistral" \
    --qwen-map "$A/maps/qwen" --mistral-map "$A/maps/mistral" --battery "$A/battery" --final-axis-map "$AXIS" \
    --queries-qwen "$R/embed-queries-qwen.merged" --queries-mistral "$R/embed-queries-mistral.merged" \
    --answers-index "$EXPORT/observations.jsonl.gz" --answers-id-field answer_id \
    --answers-qwen "$R/answers-embed-qwen.merged" --answers-mistral "$R/answers-embed-mistral.merged" \
    ${GEMMA[@]+"${GEMMA[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} --bootstrap 200 --permutations 200 --workers "$CPUS" --output "$I_OUT"
fi

# ---------------------------------------------------------------- 7 the bundle
BUNDLE="$R/paper-results-$COMMIT"
if [ ! -f "$BUNDLE/report.md" ]; then
  echo "== results bundle $(date -u +%FT%TZ)"
  "$RT/bin/python" -u "$CODE/analysis/scripts/paper_results.py" --output "$BUNDLE" \
    --input "funnel-confirmation=$F_REPORT" --input "funnel-extract=$F_EXTRACT" --input "funnel-assembled=$F_ASSEMBLED" \
    --input "generator-decisions=$DEC" --input "intent-stages=$I_OUT" --input "intent-trace-extract=$TRACE" \
    --input "intent-replay=$I_REPLAY" --input "gemma-extract=$GEMMA_OUT" --input "answers-export=$EXPORT" \
    --input "answers-analysis=$ANS_ANALYSIS"
  tar -C "$R" -czf "$W/reviews/paper-results-$COMMIT-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" "$(basename "$BUNDLE")"
fi
echo "BUNDLE $BUNDLE/report.md  tarball: $(ls -t "$W"/reviews/paper-results-"$COMMIT"-*.tar.gz 2>/dev/null | head -n 1)  end $(date -u +%FT%TZ)"
