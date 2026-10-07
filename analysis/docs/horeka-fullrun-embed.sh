#!/bin/bash
# One GPU task of the full run: both halves of the validated four-A100 LLM2Vec profile of horeka-page-readiness-lib.sh
# (same models, revisions, venvs, batch size, length and attention; one pinned worker per GPU), then the merge.
# Usage (inside a GPU allocation, run by `python -m analysis.fullrun worker --gpu`): horeka-fullrun-embed.sh VIEW PAGES OUT
# Resumable: finished 20k-text shards are kept; the merged folder marks completion.
set -euo pipefail
VIEW=$1 PAGES=$2 OUT=$3
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
A="$W/geoaxis-archive"; M="$W/models"; SCRIPT="$CODE/analysis/scripts/page_readiness_ordering.py"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ "$VIEW" = qwen ]; then
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
[ -f "$OUT.merged/projection_manifest.json" ] && { echo "embed $VIEW $(basename "$OUT"): done"; exit 0; }
test -f "$PAGES" || { echo "missing input $PAGES" >&2; exit 2; }
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -eq 4 ] || { echo "expected 4 GPUs (validated profile), found $NGPU" >&2; exit 2; }
mkdir -p "$OUT"
echo "== embed $VIEW $(basename "$OUT") $(date -u +%FT%TZ)"
status=0
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$SCRIPT" embed --pages "$PAGES" --view "$VIEW" --map "$A/maps/$VIEW" \
    --embedding-model "$MODEL" --mntp-model "$MNTP" --peft-model "$PEFT" --workers 4 --worker-index "$GPU" \
    --output "$OUT" > "$OUT.worker$GPU.${SLURM_JOB_ID:-local}.log" 2>&1 &
done
for pid in $(jobs -p); do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then tail -n 20 "$OUT".worker*."${SLURM_JOB_ID:-local}".log; exit 1; fi
"$RT/bin/python" -u "$SCRIPT" merge --input "$OUT" --pages "$PAGES" --output "$OUT.merged"
echo "== embed $VIEW $(basename "$OUT") merged $(date -u +%FT%TZ)"
