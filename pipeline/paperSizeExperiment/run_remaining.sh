#!/bin/bash
# Run remaining SearXNG experiments (Runs 6, 7, 8) sequentially
set -e
cd /Users/valerianfourel/Hamburg/GEODML
source venv312/bin/activate

echo "=== Run 6: SearXNG / Llama-3.3-70B / serp50 ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "meta-llama/Llama-3.3-70B-Instruct" \
  --pool-sizes "50,10" \
  --force
echo "Run 6 finished: $(date)"

echo "=== Run 6 RETRY: retrying 356 failed keywords ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "meta-llama/Llama-3.3-70B-Instruct" \
  --pool-sizes "50,10" \
  --force
echo "Run 6 retry finished: $(date)"

echo "=== Run 7: SearXNG / Qwen2.5-72B / serp20 ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "20,10" \
  --force
echo "Run 7 finished: $(date)"

echo "=== Run 8: SearXNG / Qwen2.5-72B / serp50 ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "50,10" \
  --force
echo "Run 8 finished: $(date)"

echo "=== All runs complete. Running cross-model analysis ==="
python paperSizeExperiment/analyze_cross_model.py \
  --input paperSizeExperiment/output/merged_all_runs.csv \
  --output-dir paperSizeExperiment/output/cross_model_analysis
echo "Cross-model analysis done: $(date)"
