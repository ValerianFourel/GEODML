#!/usr/bin/env python3
"""Run ONLY Phase 3 (LLM features) on an existing features.csv.

Usage:
  python run_phase3_only.py --run-dir paperSizeExperiment/output/searxng_Qwen2.5-72B-Instruct_serp20_top10
"""
import argparse
import csv
import json
import os
import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from gather_data import (
    run_phase_llm_features, _save_features_csv, FEATURE_COLS, LLM_TREATMENT_COLS,
)

HF_TOKEN = os.getenv("HF_TOKEN", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-72B-Instruct")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    features_csv = run_dir / "features.csv"
    html_cache_dir = run_dir / "html_cache"
    progress_file = run_dir / "progress.json"

    if not features_csv.exists():
        print(f"ERROR: {features_csv} not found")
        sys.exit(1)

    # Load features.csv
    print(f"Loading {features_csv}...")
    features = {}
    with open(features_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            if url:
                features[url] = row

    total = len(features)
    already_done = sum(1 for f in features.values() if f.get("T1_llm_statistical_density"))
    needs_processing = sum(1 for f in features.values() if not f.get("error") and not f.get("T1_llm_statistical_density"))
    print(f"Total URLs: {total}")
    print(f"Already have LLM features: {already_done}")
    print(f"Need LLM processing: {needs_processing}")
    print(f"Errors (skipped): {total - already_done - needs_processing}")

    # Update progress
    if progress_file.exists():
        with open(progress_file) as f:
            prog = json.load(f)
    else:
        prog = {}
    prog["phase"] = "phase3_llm_features"
    prog["phase3_llm_features_total"] = needs_processing
    prog["phase3_llm_features_done"] = 0
    with open(progress_file, "w") as f:
        json.dump(prog, f, indent=2)

    # Run Phase 3
    print(f"\nStarting Phase 3 LLM features with {args.llm_model}...")
    features = run_phase_llm_features(
        features, {},  # empty html_cache — will lazy-load from disk
        HF_TOKEN, args.llm_model,
        html_cache_dir=html_cache_dir,
        features_csv=features_csv,
    )

    # Final save
    _save_features_csv(features, features_csv)
    print(f"\nPhase 3 complete. Saved {features_csv}")

    # Update progress
    prog["phase"] = "done"
    prog["phase3_llm_features_done"] = needs_processing
    with open(progress_file, "w") as f:
        json.dump(prog, f, indent=2)


if __name__ == "__main__":
    main()
