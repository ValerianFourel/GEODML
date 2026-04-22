#!/usr/bin/env python3
"""Rebuild features.csv from cached HTML files + rankings.csv.

Used when gather_data.py was interrupted before saving features.
Reads HTML from html_cache/, extracts code-based features, runs PageRank + WHOIS.

Usage:
  python pipeline/rebuild_features.py --input-dir output/deepseek-r1/
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# Add parent so we can import from gather_data
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gather_data import (
    extract_html_features,
    run_phase_pagerank,
    run_phase_whois,
    compute_keyword_difficulty,
    FEATURE_COLS,
)

# Load env
from dotenv import load_dotenv
import os
load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")
OPENPAGERANK_KEY = os.getenv("OPENPAGERANK_KEY", "")


def _url_to_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Rebuild features.csv from cached HTML")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with rankings.csv and html_cache/")
    parser.add_argument("--pagerank", action="store_true", default=True,
                        help="Run PageRank phase (default: True)")
    parser.add_argument("--whois", action="store_true", default=True,
                        help="Run WHOIS phase (default: True)")
    parser.add_argument("--no-pagerank", action="store_true",
                        help="Skip PageRank")
    parser.add_argument("--no-whois", action="store_true",
                        help="Skip WHOIS")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    html_cache_dir = input_dir / "html_cache"
    rankings_csv = input_dir / "rankings.csv"
    features_csv = input_dir / "features.csv"

    if not rankings_csv.exists():
        print(f"Missing: {rankings_csv}")
        sys.exit(1)
    if not html_cache_dir.exists():
        print(f"Missing: {html_cache_dir}")
        sys.exit(1)

    # Load rankings to get URL list
    rankings_rows = []
    with open(rankings_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings_rows.append(row)
    print(f"Loaded {len(rankings_rows)} ranking rows")

    # Deduplicate URLs
    seen = set()
    url_list = []
    for row in rankings_rows:
        url = row.get("url", "").strip()
        domain = row.get("domain", "").strip()
        if not url:
            url = f"https://{domain}/"
        if url not in seen:
            seen.add(url)
            url_list.append({"url": url, "domain": domain})
    print(f"Unique URLs: {len(url_list)}")

    # Extract features from cached HTML
    features = {}
    ok_count = 0
    miss_count = 0
    err_count = 0

    for i, entry in enumerate(url_list, 1):
        url = entry["url"]
        domain = entry["domain"]
        cache_key = _url_to_cache_key(url)
        cache_path = html_cache_dir / f"{cache_key}.html"

        if cache_path.exists():
            html = cache_path.read_text(encoding="utf-8", errors="replace")
            feat = extract_html_features(html, url, domain)
            if feat.get("error"):
                err_count += 1
            else:
                ok_count += 1
            if i <= 5 or i % 50 == 0:
                print(f"  [{i}/{len(url_list)}] {url}  Words={feat.get('X3_word_count', '?')}  T1={feat.get('T1_statistical_density', '?')}")
        else:
            miss_count += 1
            feat = {
                "url": url, "domain": domain,
                "T1_statistical_density": None, "T2_question_heading_match": None,
                "T3_structured_data": None, "T4_citation_authority": None,
                "X3_word_count": None, "X6_readability": None,
                "X7_internal_links": None, "X7B_outbound_links": None,
                "X9_images_with_alt": None,
                "X10_https": 1 if url.lower().startswith("https://") else 0,
                "error": "no_cached_html",
            }

        features[url] = feat

    print(f"\nFeature extraction: {ok_count} OK, {err_count} errors, {miss_count} missing HTML")

    # PageRank
    if not args.no_pagerank:
        features = run_phase_pagerank(features, OPENPAGERANK_KEY)

    # WHOIS
    if not args.no_whois:
        features = run_phase_whois(features)

    # Keyword difficulty
    kw_difficulty = compute_keyword_difficulty(rankings_rows, features)
    if kw_difficulty:
        for url, feat in features.items():
            for row in rankings_rows:
                if row.get("url", "").strip() == url or f"https://{row.get('domain', '')}/" == url:
                    kw = row.get("keyword", "")
                    if kw in kw_difficulty:
                        feat["X8_keyword_difficulty"] = kw_difficulty[kw]
                    break

    # Save
    cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    with open(features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for url in sorted(features):
            writer.writerow(features[url])
    print(f"\nSaved features CSV: {features_csv} ({len(features)} rows)")


if __name__ == "__main__":
    main()
