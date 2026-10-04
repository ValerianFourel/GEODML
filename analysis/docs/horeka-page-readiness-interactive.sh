#!/bin/bash
# Run the whole page-readiness pipeline inside one existing four-A100 allocation.
# Usage (from a HoreKa login shell that holds the allocation):
#   bash horeka-page-readiness-interactive.sh SLURM_JOB_ID
# Every stage is skipped when its output exists, and embedding resumes from saved
# shards, so rerunning the same command in a later allocation continues the work.
# Nothing here cancels, extends or releases the allocation.
set -euo pipefail
set +x
JOB=${1:?usage: horeka-page-readiness-interactive.sh SLURM_JOB_ID}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "job id must be numeric" >&2; exit 2; }
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
R="$W/reviews/page-readiness-20261004"
A="$W/geoaxis-archive"
M="$W/models"
SCRIPT="$CODE/analysis/scripts/page_readiness_ordering.py"
EXTRACT="$R/extract-llama-v2"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1
mkdir -p "$R"
LOG="$R/interactive-$JOB.log"
exec > >(tee -a "$LOG") 2>&1
echo "commit $(git -C "$CODE" rev-parse HEAD) allocation $JOB start $(date -u +%FT%TZ)"

step() {  # CPU step on the allocated node
  srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=none "$@"
}

if [ ! -f "$EXTRACT/manifest.json" ]; then
  echo "== extract llama answers and page texts (32 readers)"
  step "$RT/bin/python" -u "$SCRIPT" extract --source "$W/llama-hf/dataset:llama4" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --workers 32 --output "$EXTRACT"
fi
"$RT/bin/python" -c "import json,sys;print('EXTRACT', json.load(open(sys.argv[1]))['counts'])" "$EXTRACT/manifest.json"
if [ ! -f "$R/prompt-sample/pages.jsonl.gz" ]; then
  step "$RT/bin/python" -u "$SCRIPT" sample-prompts --prompts "$A/final-audit/compliant-candidates.jsonl" \
    --count 512 --output "$R/prompt-sample"
fi

view_settings() {
  if [ "$1" = qwen ]; then
    PY="$W/environment/llm2vec-qwen/bin/python"
    MODEL="$M/qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218"
    MNTP="$M/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/c84774c1366ea79f033504994bd254155d956d57"
    PEFT="$M/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/86b17660b1b1a8efe0b822e90c995f1ac7294645"
  else
    PY="$W/environment/llm2vec-mistral/bin/python"
    MODEL="$M/mistralai/Mistral-7B-Instruct-v0.2/63a8b081895390a26e140280378bc85ec8bce07a"
    MNTP="$M/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp/e76f9757923897a0c5204b3075f1062f484d033b"
    PEFT="$M/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8"
  fi
}

embed_view() {  # view pages output: one 4-GPU step, one pinned worker per GPU, shared shard list
  local view=$1 pages=$2 out=$3
  view_settings "$view"
  [ -f "$out.merged/projection_manifest.json" ] && return 0
  echo "== embed $view: $(basename "$out") $(date -u +%FT%TZ)"
  export PY MODEL MNTP PEFT SCRIPT A JOB VIEW_NAME="$view" PAGES_FILE="$pages" OUT_DIR="$out"
  if ! HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 bash -c '
        set -uo pipefail
        status=0
        for GPU in 0 1 2 3; do
          CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$SCRIPT" embed --pages "$PAGES_FILE" --view "$VIEW_NAME" \
            --map "$A/maps/$VIEW_NAME" --embedding-model "$MODEL" --mntp-model "$MNTP" --peft-model "$PEFT" \
            --workers 4 --worker-index "$GPU" --output "$OUT_DIR" > "$OUT_DIR.worker$GPU.$JOB.log" 2>&1 &
        done
        for pid in $(jobs -p); do wait "$pid" || status=1; done
        exit $status'; then
    tail -n 20 "$out".worker*."$JOB".log
    return 1
  fi
  step "$RT/bin/python" -u "$SCRIPT" merge --input "$out" --pages "$pages" --output "$out.merged"
}

for VIEW in qwen mistral; do
  mkdir -p "$R/embed-sample-$VIEW" "$R/embed-pages-$VIEW"
  embed_view "$VIEW" "$R/prompt-sample/pages.jsonl.gz" "$R/embed-sample-$VIEW"
done
if [ ! -f "$R/relocation-fresh.json" ]; then
  echo "== relocation with fresh re-embedding of 512 archived prompts"
  step "$RT/bin/python" -u "$SCRIPT" relocate --final-axis-map "$A/final-audit/final-axis-map.jsonl" \
    --qwen-projections "$A/final-audit/merged/qwen" --mistral-projections "$A/final-audit/merged/mistral" \
    --battery "$A/battery" --fresh-qwen "$R/embed-sample-qwen.merged" \
    --fresh-mistral "$R/embed-sample-mistral.merged" --output "$R/relocation-fresh.json" \
    || { echo "fresh embeddings do not reproduce the archived axis; page embedding stopped" >&2; exit 2; }
fi
for VIEW in qwen mistral; do
  embed_view "$VIEW" "$EXTRACT/pages.jsonl.gz" "$R/embed-pages-$VIEW"
done
if [ ! -f "$R/analysis-llama/report.html" ]; then
  echo "== ranking model, keyword bootstrap, permutation null, HTML report"
  step "$RT/bin/python" -u "$SCRIPT" analyze --extract "$EXTRACT" --qwen "$R/embed-pages-qwen.merged" \
    --mistral "$R/embed-pages-mistral.merged" --battery "$A/battery" \
    --final-axis-map "$A/final-audit/final-axis-map.jsonl" --workers 32 --output "$R/analysis-llama"
fi
echo "REPORT $R/analysis-llama/report.html  end $(date -u +%FT%TZ)"
