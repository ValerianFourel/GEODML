#!/bin/bash
#SBATCH --job-name=geodml-rebuttal-retrieval
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# ARR author response, retrieval baseline (exploratory), one four-A100 allocation; no generator is called:
#   1. Lucene check (CPU, a few minutes): Anserini BM25 (k1 0.9, b 0.4, default English analyzer) over each engine
#      snapshot for a seeded sample of logged queries and prompts, to validate the Python bm25l (lucene_check.py)
#   2. run_horeka.sh: e5-large-v2 dense retrieval and bge-reranker-v2-m3 scores of every new (prompt, row) pair of
#      nokw, bm25, bm25l and dense, plus the 1,000 validation answers (gate G2 on the Mac)
# Submit:  sbatch --account=<account> --export=ALL,CODE=<clean checkout>,BUNDLE=<bundle dir>,RUN_ID=<id> run_rebuttal.sbatch
# Or, inside an existing salloc (same steps, no new allocation):  CODE=... BUNDLE=... RUN_ID=... bash run_rebuttal.sbatch
# Resumable: resubmit with the same RUN_ID; finished steps and shards are skipped. Exit 4 = deadline checkpoint.
# Optional: LUCENE_JAR (default: the jar setup_lucene.sh downloaded; the check is skipped when there is none).
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=${CODE:?CODE must point to the clean, pinned checkout}
BUNDLE=${BUNDLE:?BUNDLE must point to the copied bundle directory}
RUN_ID=${RUN_ID:?RUN_ID, e.g. rebuttal-20261019}
JOB=${SLURM_JOB_ID:?run under sbatch or inside a salloc shell}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
PY="$W/environment/qwen-runtime/bin/python"
RUN="$W/runs/www-robustness/$RUN_ID"
mkdir -p "$RUN/logs"
echo "rebuttal retrieval: commit $(git -C "$CODE" rev-parse HEAD) job $JOB host $(hostname) start $(date -u +%FT%TZ)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"

# 1. Lucene check (skipped if done or no jar)
LJ=${LUCENE_JAR:-$(ls "$W"/environment/lucene/anserini-*-fatjar.jar 2>/dev/null | tail -1 || true)}
JAVA="$W/environment/lucene/jdk/bin/java"
if [ -s "$RUN/results/lucene/hits.parquet" ]; then
  echo "== lucene: done before"
elif [ -n "$LJ" ] && [ -x "$JAVA" ]; then
  echo "== lucene $(date -u +%FT%TZ) $(basename "$LJ")"
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=none bash -c '
    set -euo pipefail
    "$0" -u "$1/analysis/www_robustness/lucene_check.py" prepare --bundle "$2" --run-dir "$3"
    for d in "$3"/docs/*/; do
      e=$(basename "$d")
      rm -rf "$3/index/$e.tmp"
      [ -d "$3/index/$e" ] || { "$4" -cp "$5" io.anserini.index.IndexCollection -collection JsonCollection \
        -input "$d" -index "$3/index/$e.tmp" -generator DefaultLuceneDocumentGenerator -threads 8 \
        && mv "$3/index/$e.tmp" "$3/index/$e"; }
      mkdir -p "$3/runs"
      "$4" -cp "$5" io.anserini.search.SearchCollection -index "$3/index/$e" -topics "$3/topics/$e.tsv" \
        -topicReader TsvString -output "$3/runs/$e.trec.tmp" -bm25 -bm25.k1 0.9 -bm25.b 0.4 -hits 100 -threads 8
      mv "$3/runs/$e.trec.tmp" "$3/runs/$e.trec"
    done
    "$0" -u "$1/analysis/www_robustness/lucene_check.py" collect --run-dir "$3"
  ' "$PY" "$CODE" "$BUNDLE" "$RUN" "$JAVA" "$LJ" > "$RUN/logs/lucene-$JOB.log" 2>&1 \
    || { echo "lucene check failed (GPU steps still run):"; tail -n 20 "$RUN/logs/lucene-$JOB.log"; }
else
  echo "== lucene: skipped (run setup_lucene.sh on a login node first)"
fi

# 2. dense retrieval and cross-encoder scores (the 18 Oct runner, unchanged)
rc=0
bash "$CODE/analysis/www_robustness/run_horeka.sh" "$JOB" "$BUNDLE" "$RUN_ID" || rc=$?
echo "end $(date -u +%FT%TZ) exit $rc"
exit "$rc"
