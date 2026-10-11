#!/bin/bash
# Axis-2 readout on HoreKa: CPU only, inside one allocation on an accelerated node (never cpuonly).
#   1 texts: the agent's queries' and the answers' raw axis coordinates (both LLM2Vec views) from the full run's merged
#     projections -> results/texts/text_coordinates.parquet (small; the Mac readout takes it with --text-coordinates)
#   2 run:   the complete readout (prompts, pages, queries, answers; random directions for prompts and pages)
#     -> results/readout/{results.json, numbers-axis2.tex, stages-axis2.csv, face-validity.txt, ...}
# Usage, from a login shell, through one allocation-bearing srun (see the handoff for the full command):
#   srun ... bash $CODE/analysis/docs/horeka-axis2-readout.sh OUT_DIR
# Optional: TEXTS_ONLY=1 stops after step 1 (the cheaper path: finish the readout on the Mac).
# Writes OUT_DIR/{config.yaml, run_manifest.json, stdout.log, stderr.log, results/, diagnostics/}; refuses an existing
# OUT_DIR. Reads only; no GPU, no model, no Hub access. Never cancels, extends or releases an allocation.
set -euo pipefail
OUT=${1:?usage: horeka-axis2-readout.sh OUT_DIR}
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
R=$W/reviews/page-readiness-20261004
RUN=$R/fullrun-v1/run2
A=$W/geoaxis-archive
SIF=${GEODML_SIF:-$W/containers/geodml-cpu.sif}
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PIN=$(git -C "$CODE" rev-parse HEAD)
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT/results" "$OUT/diagnostics"
exec > >(tee -a "$OUT/stdout.log") 2> >(tee -a "$OUT/stderr.log" >&2)
START=$(date -u +%FT%TZ)
echo "axis2-readout commit $PIN job ${SLURM_JOB_ID:-none} node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} start $START"

TRACES=$RUN/release-traces-v1
PROMPT_RECORDS=$A/registration/population-registration-v1/population-prompts.jsonl
for path in "$TRACES/trace_answers.parquet" "$TRACES/stage_items.parquet" "$TRACES/answer_queries.parquet" "$TRACES/queries.parquet" \
    "$RUN/funnel-replay/replay.npz" "$R/funnel-features-v1/rows.parquet" "$R/snippet-embeddings-corpus-v1/snippets.parquet" \
    "$A/final-audit/projections/qwen" "$A/final-audit/projections/mistral" "$A/maps/qwen/readiness_embedding_map.json" \
    "$A/maps/mistral/readiness_embedding_map.json" "$A/battery/readiness_robustness_battery.json" "$A/final-audit/final-axis-map.jsonl" \
    "$RUN/embed-queries-qwen.merged/projection_manifest.json" "$RUN/embed-queries-mistral.merged/projection_manifest.json" \
    "$RUN/embed-answers-qwen.merged/projection_manifest.json" "$RUN/embed-answers-mistral.merged/projection_manifest.json" \
    "$RUN/answers-export/observations.jsonl.gz" "$SIF"; do
  test -e "$path" || { echo "missing input $path" >&2; exit 2; }
done
EXTRA=()
test -f "$PROMPT_RECORDS" && EXTRA+=(--prompt-records "$PROMPT_RECORDS")
test -f "$A/maps/qwen/readiness_supervised_subspace_coordinates.jsonl" && EXTRA+=(--calibration "$A")
test -f "$RUN/steelman-confirmation/chain.json" && EXTRA+=(--stored-chain "$RUN/steelman-confirmation/chain.json")

cat > "$OUT/config.yaml" <<EOF
experiment: axis2-readout
commit: $PIN
slurm_job_id: ${SLURM_JOB_ID:-none}
container: $SIF
full_run: $RUN
traces: $TRACES
archive: $A
corpus: $R/snippet-embeddings-corpus-v1
bootstrap: 200
permutations: 200
seeds: {keyword_draws: 20261007, shuffles: 20261008, random_directions: 20261012}
random_directions_per_view: 5
texts_only: ${TEXTS_ONLY:-0}
extra_arguments: "${EXTRA[*]}"
EOF
{ nproc; free -g; df -h "$W" | tail -n 1; } > "$OUT/diagnostics/node.txt" 2>&1 || true

py() {
  apptainer exec --no-home --bind "$W" \
    --env "GEODML_GIT_COMMIT=$PIN,PYTHONPATH=$CODE,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1" \
    "$SIF" python -u "$@"
}
TEXTS_STATUS=0
RUN_STATUS=skipped
echo "== texts $(date -u +%FT%TZ)"
py -m analysis.scripts.axis2_readout texts --traces "$TRACES" \
  --queries-qwen "$RUN/embed-queries-qwen.merged" --queries-mistral "$RUN/embed-queries-mistral.merged" \
  --answers-qwen "$RUN/embed-answers-qwen.merged" --answers-mistral "$RUN/embed-answers-mistral.merged" \
  --answers-index "$RUN/answers-export/observations.jsonl.gz" --output "$OUT/results/texts" || TEXTS_STATUS=$?
if [ "$TEXTS_STATUS" -eq 0 ] && [ "${TEXTS_ONLY:-0}" != 1 ]; then
  echo "== run $(date -u +%FT%TZ)"
  RUN_STATUS=0
  py -m analysis.scripts.axis2_readout run --traces "$TRACES" --replay "$RUN/funnel-replay" \
    --rows "$R/funnel-features-v1/rows.parquet" --corpus "$R/snippet-embeddings-corpus-v1" \
    --prompt-embeddings "$A/final-audit/projections" --maps "$A/maps" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" ${EXTRA[@]+"${EXTRA[@]}"} \
    --text-coordinates "$OUT/results/texts/text_coordinates.parquet" --output "$OUT/results/readout" || RUN_STATUS=$?
fi
END=$(date -u +%FT%TZ)
cat > "$OUT/run_manifest.json" <<EOF
{"experiment": "axis2-readout", "commit": "$PIN", "slurm_job_id": "${SLURM_JOB_ID:-none}", "node": "$(hostname)",
 "cpus_per_task": "${SLURM_CPUS_PER_TASK:-unknown}", "start": "$START", "end": "$END",
 "texts_exit": $TEXTS_STATUS, "run_exit": "$RUN_STATUS", "texts_only": "${TEXTS_ONLY:-0}",
 "outputs": {"texts": "results/texts", "readout": "results/readout"}, "container": "$SIF"}
EOF
echo "axis2-readout end $END texts_exit $TEXTS_STATUS run_exit $RUN_STATUS"
[ "$TEXTS_STATUS" -eq 0 ] && { [ "$RUN_STATUS" = skipped ] || [ "$RUN_STATUS" = 0 ]; }
