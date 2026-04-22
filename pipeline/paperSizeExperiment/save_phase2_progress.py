#!/usr/bin/env python3
"""Recovery script: rebuild features.csv from cached HTML files in html_cache/.

Reads rankings.csv to get the URL list, checks which URLs have cached HTML,
extracts features, and writes features.csv — saving the progress of a
running/crashed Phase 2.
"""

import csv
import hashlib
import sys
from pathlib import Path

# Add project root so we can import from pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.gather_data import extract_html_features, FEATURE_COLS, _extract_domain


def _url_to_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "paperSizeExperiment/output/duckduckgo_Llama-3.3-70B-Instruct_serp20_top10"
    )
    rankings_csv = output_dir / "rankings.csv"
    html_cache_dir = output_dir / "html_cache"
    features_csv = output_dir / "features.csv"

    if not rankings_csv.exists():
        print(f"No rankings.csv found at {rankings_csv}")
        sys.exit(1)

    # Build unique URL list from rankings
    url_list = []
    seen = set()
    with open(rankings_csv, newline="") as f:
        for row in csv.DictReader(f):
            url = row.get("url", "").strip()
            domain = row.get("domain", "").strip()
            if not url:
                url = f"https://{domain}/"
            if url not in seen:
                seen.add(url)
                url_list.append({"url": url, "domain": domain})

    print(f"Rankings has {len(url_list)} unique URLs")
    print(f"HTML cache has {len(list(html_cache_dir.glob('*.html')))} files")

    # Extract features from cached HTML
    cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    ok = 0
    missing = 0

    with open(features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()

        for i, entry in enumerate(url_list, 1):
            url = entry["url"]
            domain = entry["domain"]
            cache_path = html_cache_dir / f"{_url_to_cache_key(url)}.html"

            if cache_path.exists():
                html = cache_path.read_text(encoding="utf-8")
                feat = extract_html_features(html, url, domain)
                feat["fetch_status_code"] = 200
                feat["error"] = None
                writer.writerow(feat)
                ok += 1
            else:
                # URL wasn't fetched yet — write error row
                feat = {
                    "url": url, "domain": domain,
                    "fetch_status_code": None,
                    "error": "not_fetched_yet",
                }
                writer.writerow(feat)
                missing += 1

            if i % 500 == 0:
                print(f"  [{i}/{len(url_list)}] OK={ok} Missing={missing}")

    print(f"\nDone! Saved {features_csv}")
    print(f"  OK (from cache): {ok}")
    print(f"  Missing (not fetched): {missing}")


if __name__ == "__main__":
    main()
