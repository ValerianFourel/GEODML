#!/usr/bin/env python3
"""Standalone Phase 3: LLM Feature Extraction.

Runs just the LLM feature extraction on an existing run directory that already
has features.csv and html_cache/ from phases 1-2.

Fully resumable — reads features.csv on start, skips URLs that already have
LLM features, saves after every 5 URLs. Safe to Ctrl-C and re-run.

Usage:
  python pipeline/run_phase3.py paperSizeExperiment/output/duckduckgo_Llama-3.3-70B-Instruct_serp20_top10
  python pipeline/run_phase3.py paperSizeExperiment/output/duckduckgo_Llama-3.3-70B-Instruct_serp20_top10 --model meta-llama/Llama-3.3-70B-Instruct
  python pipeline/run_phase3.py <run_dir> --max-errors 30
"""

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env.local")

from gather_data import (
    FEATURE_COLS,
    _save_features_csv,
    _url_to_cache_key,
    build_page_digest,
    llm_extract_treatments,
)

HF_TOKEN = os.getenv("HF_TOKEN", "")

# Graceful shutdown on Ctrl-C
_stop_requested = False

def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n\n  >>> Ctrl-C received — finishing current URL and saving... <<<\n")

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def load_features(features_csv: Path) -> dict[str, dict]:
    features = {}
    with open(features_csv, newline="") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "").strip()
            if url:
                features[url] = row
    return features


def run(run_dir: Path, model_id: str, max_consecutive_errors: int = 20, checkpoint_interval: int = 5):
    features_csv = run_dir / "features.csv"
    html_cache_dir = run_dir / "html_cache"
    progress_json = run_dir / "progress.json"

    if not features_csv.exists():
        print(f"Error: {features_csv} not found. Run phases 1-2 first.")
        sys.exit(1)
    if not html_cache_dir.exists():
        print(f"Error: {html_cache_dir}/ not found. Run phases 1-2 first.")
        sys.exit(1)
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set in .env.local")
        sys.exit(1)

    # Load features
    print(f"Loading features from {features_csv} ...")
    features = load_features(features_csv)
    print(f"  Total URLs: {len(features)}")

    # Determine which URLs need LLM features
    urls_to_process = [
        url for url, f in features.items()
        if not f.get("error") and not f.get("T1_llm_statistical_density")
    ]
    already_done = sum(1 for f in features.values() if f.get("T1_llm_statistical_density"))
    print(f"  Already have LLM features: {already_done}")
    print(f"  Need LLM features: {len(urls_to_process)}")

    if not urls_to_process:
        print("\nAll URLs already have LLM features. Nothing to do.")
        return

    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)
    print(f"\n  Model: {model_id}")
    print(f"  Checkpoint every {checkpoint_interval} URLs")
    print(f"  Press Ctrl-C to stop gracefully\n")

    consecutive_errors = 0
    processed_count = 0
    skipped_count = 0

    def save_progress():
        _save_features_csv(features, features_csv)
        # Update progress.json
        done_now = sum(1 for f in features.values() if f.get("T1_llm_statistical_density"))
        total_eligible = sum(1 for f in features.values() if not f.get("error"))
        if progress_json.exists():
            with open(progress_json) as pf:
                prog = json.load(pf)
        else:
            prog = {}
        prog["phase"] = "phase3_llm_features"
        prog["phase3_llm_features_total"] = total_eligible
        prog["phase3_llm_features_done"] = done_now
        prog["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        with open(progress_json, "w") as pf:
            json.dump(prog, pf, indent=2)

    try:
        for i, url in enumerate(urls_to_process, 1):
            if _stop_requested:
                print(f"\nStopping after {processed_count} processed, {skipped_count} skipped.")
                break

            feat = features[url]
            domain = feat.get("domain", "")
            print(f"[{i}/{len(urls_to_process)}] {url} ({domain})", end=" ")

            # Lazy-load HTML from disk cache
            cache_path = html_cache_dir / f"{_url_to_cache_key(url)}.html"
            if not cache_path.exists():
                print("  No cached HTML, skipping.")
                feat["llm_error"] = "no_cached_html"
                skipped_count += 1
                continue

            html = cache_path.read_text(encoding="utf-8")
            digest = build_page_digest(html, url, domain)
            llm_result = llm_extract_treatments(digest, client, model_id)
            for key, val in llm_result.items():
                feat[key] = val

            err = feat.get("llm_error", "")
            if err:
                print(f"ERROR: {err}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n  CIRCUIT BREAKER: {max_consecutive_errors} consecutive errors. "
                          f"Stopping. Re-run to resume.")
                    break
            else:
                print(f"T1={feat.get('T1_llm_statistical_density', '?')} "
                      f"T2={feat.get('T2_llm_question_heading', '?')} "
                      f"T3={feat.get('T3_llm_structured_data', '?')} "
                      f"T4={feat.get('T4_llm_citation_authority', '?')}")
                consecutive_errors = 0
                processed_count += 1

            # Checkpoint
            if i % checkpoint_interval == 0:
                save_progress()
                print(f"  [checkpoint {already_done + processed_count} done]")

            time.sleep(random.uniform(0.5, 1.5))
    finally:
        if processed_count > 0 or skipped_count > 0:
            save_progress()
            print(f"\nSaved. Processed: {processed_count}, Skipped: {skipped_count}, "
                  f"Total with LLM features: {already_done + processed_count}")
        else:
            print("\nNo new URLs processed.")


def main():
    parser = argparse.ArgumentParser(description="Standalone Phase 3: LLM Feature Extraction")
    parser.add_argument("run_dir", type=Path, help="Path to run output directory")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID (default: meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--max-errors", type=int, default=20,
                        help="Max consecutive LLM errors before stopping (default: 20)")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Save to CSV every N URLs (default: 5)")
    args = parser.parse_args()

    run(args.run_dir, args.model, args.max_errors, args.checkpoint_interval)


if __name__ == "__main__":
    main()
