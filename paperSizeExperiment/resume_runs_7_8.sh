#!/bin/bash
# Resume Run 7 (Phase 3 + retry failed Phase 1 keywords) and start Run 8.
# Run under caffeinate so macOS sleep doesn't kill the process.
set -u
cd /Users/valerianfourel/Hamburg/GEODML
source venv312/bin/activate

echo "=== Run 7 RESUME: SearXNG / Qwen2.5-72B / serp20 ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "20,10" \
  --force
RC7=$?
echo "Run 7 finished rc=$RC7: $(date)"

echo "=== Run 8: SearXNG / Qwen2.5-72B / serp50 ==="
echo "Started: $(date)"
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "50,10" \
  --force
RC8=$?
echo "Run 8 finished rc=$RC8: $(date)"

echo "=== Cross-model analysis ==="
python paperSizeExperiment/analyze_cross_model.py \
  --input paperSizeExperiment/output/merged_all_runs.csv \
  --output-dir paperSizeExperiment/output/cross_model_analysis
echo "Cross-model done: $(date)"
