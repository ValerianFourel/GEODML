#!/usr/bin/env python3
"""Traditional search engine ranking: keywords → DuckDuckGo + Google → ranked domains."""

import sys
from datetime import datetime, timezone

from src.keywords import load_keywords
from src.engine_scraper import search_duckduckgo, search_google
from src.results_io import save_results, results_to_csv
from src.config import TOP_N


def main():
    keywords = load_keywords()
    print(f"Loaded {len(keywords)} keywords")
    print(f"Top N domains per keyword: {TOP_N}\n")

    ddg_rankings = {}
    google_rankings = {}
    ddg_errors = []
    google_errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # DuckDuckGo
        ddg_domains = search_duckduckgo(keyword, num_results=TOP_N)
        if ddg_domains:
            ddg_rankings[keyword] = ddg_domains
            print(f"  DDG: {len(ddg_domains)} domains — {ddg_domains[:3]}...")
        else:
            ddg_errors.append(keyword)
            print(f"  DDG: no results")

        # Google
        google_domains = search_google(keyword, num_results=TOP_N)
        if google_domains:
            google_rankings[keyword] = google_domains
            print(f"  Google: {len(google_domains)} domains — {google_domains[:3]}...")
        else:
            google_errors.append(keyword)
            print(f"  Google: no results")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Save DuckDuckGo results
    ddg_data = {
        "source": "duckduckgo",
        "method": "duckduckgo_search library",
        "top_n": TOP_N,
        "timestamp": timestamp,
        "total_keywords": len(keywords),
        "successful_keywords": len(ddg_rankings),
        "failed_keywords": ddg_errors,
        "rankings": ddg_rankings,
    }
    save_results(ddg_data, "ddg_rankings.json")
    results_to_csv(ddg_data, "ddg_rankings.csv")

    # Save Google results
    google_data = {
        "source": "google",
        "method": "googlesearch-python library",
        "top_n": TOP_N,
        "timestamp": timestamp,
        "total_keywords": len(keywords),
        "successful_keywords": len(google_rankings),
        "failed_keywords": google_errors,
        "rankings": google_rankings,
    }
    save_results(google_data, "google_rankings.json")
    results_to_csv(google_data, "google_rankings.csv")

    # Combined output for convenience
    combined_data = {
        "source": "engine_search",
        "method": "DuckDuckGo + Google combined",
        "top_n": TOP_N,
        "timestamp": timestamp,
        "total_keywords": len(keywords),
        "rankings": {
            kw: {"duckduckgo": ddg_rankings.get(kw, []), "google": google_rankings.get(kw, [])}
            for kw in keywords
        },
    }
    save_results(combined_data, "engine_rankings.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"Engine Search Complete")
    print(f"  DuckDuckGo: {len(ddg_rankings)}/{len(keywords)} keywords")
    print(f"  Google: {len(google_rankings)}/{len(keywords)} keywords")
    if ddg_errors:
        print(f"  DDG failures: {ddg_errors}")
    if google_errors:
        print(f"  Google failures: {google_errors}")
    print(f"  Results saved to results/")


if __name__ == "__main__":
    main()
