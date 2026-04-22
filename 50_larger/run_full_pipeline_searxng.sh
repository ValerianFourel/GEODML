#!/bin/bash
# Full SearXNG pipeline for 50_larger experiment
# Run with: nohup bash 50_larger/run_full_pipeline_searxng.sh > 50_larger/pipeline.log 2>&1 &
set -e
cd /Users/valerianfourel/Hamburg/GEODML
source venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== Step 1: AI Search with SearXNG ==="
python 50_larger/run_ai_search.py --engine searxng

echo ""
echo "=== Step 2: Extract Results ==="
python 50_larger/extract_all_results.py --filter searxng

echo ""
echo "=== Step 3: Page Scraper (all phases) ==="
python 50_larger/run_page_scraper.py --input 50_larger/results/all_results_searxng.csv --all

echo ""
echo "=== Step 4: Build Clean Dataset ==="
python 50_larger/build_clean_dataset.py

echo ""
echo "=== Step 5: Run DML Study ==="
python 50_larger/run_dml_study.py

echo ""
echo "=== Step 6: Test Suite (32 models) ==="
python 50_larger/test/run_experiments.py

echo ""
echo "=== Step 7: Test Diff (16 models) ==="
python 50_larger/test_diff/run_experiments.py

echo ""
echo "=== Step 8: Full Diagnostics LGBM ==="
python 50_larger/test_full/run_full_diagnostics.py

echo ""
echo "=== Step 9: Full Diagnostics RF ==="
python 50_larger/test_full_rf/run_full_diagnostics.py

echo ""
echo "=== ALL STEPS COMPLETE ==="
date
