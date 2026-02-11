#!/usr/bin/env python3
"""Traditional search engine ranking: keywords -> DuckDuckGo + Google -> ranked domains.

Captures full provenance: experiment context (IP, geo, machine),
per-query timestamps, raw SERP data with original positions.
"""

from src.keywords import load_keywords
from src.engine_scraper import search_duckduckgo, search_google
from src.results_io import save_results, results_to_csv
from src.experiment_context import collect_experiment_context, utcnow_iso
from src.config import TOP_N


def main():
    # Collect experiment provenance BEFORE any queries
    print("Collecting experiment context (IP, geolocation, machine info)...")
    context = collect_experiment_context()
    print(f"  IP: {context['network']['public_ip']}")
    geo = context["network"]["geolocation"]
    print(f"  Location: {geo['city']}, {geo['region']}, {geo['country']}")
    print(f"  ISP: {geo['isp']}")
    print()

    keywords = load_keywords()
    print(f"Loaded {len(keywords)} keywords")
    print(f"Top N domains per keyword: {TOP_N}\n")

    ddg_query_results = []
    google_query_results = []
    ddg_errors = []
    google_errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # DuckDuckGo (returns full provenance dict)
        ddg_result = search_duckduckgo(keyword, num_results=TOP_N)
        if ddg_result["ranked_domains"]:
            ddg_query_results.append(ddg_result)
            print(f"  DDG: {len(ddg_result['ranked_domains'])} domains — "
                  f"{ddg_result['ranked_domains'][:3]}...")
        else:
            ddg_errors.append({"keyword": keyword, "error": ddg_result.get("error")})
            print(f"  DDG: no results")

        # Google (returns full provenance dict)
        google_result = search_google(keyword, num_results=TOP_N)
        if google_result["ranked_domains"]:
            google_query_results.append(google_result)
            print(f"  Google: {len(google_result['ranked_domains'])} domains — "
                  f"{google_result['ranked_domains'][:3]}...")
        else:
            google_errors.append({"keyword": keyword, "error": google_result.get("error")})
            print(f"  Google: no results")

    end_time = utcnow_iso()

    # Save DuckDuckGo full provenance
    ddg_data = {
        "experiment_context": context,
        "experiment_end_utc": end_time,
        "source": "duckduckgo",
        "method": "duckduckgo_search library",
        "top_n": TOP_N,
        "total_keywords": len(keywords),
        "successful_keywords": len(ddg_query_results),
        "failed_keywords": ddg_errors,
        "per_keyword_results": ddg_query_results,
    }
    save_results(ddg_data, "ddg_rankings.json")
    results_to_csv(ddg_query_results, "duckduckgo", "ddg_rankings.csv")

    # Save Google full provenance
    google_data = {
        "experiment_context": context,
        "experiment_end_utc": end_time,
        "source": "google",
        "method": "googlesearch-python library",
        "top_n": TOP_N,
        "total_keywords": len(keywords),
        "successful_keywords": len(google_query_results),
        "failed_keywords": google_errors,
        "per_keyword_results": google_query_results,
    }
    save_results(google_data, "google_rankings.json")
    results_to_csv(google_query_results, "google", "google_rankings.csv")

    # Combined reference file
    combined_data = {
        "experiment_context": context,
        "experiment_end_utc": end_time,
        "source": "engine_search",
        "method": "DuckDuckGo + Google combined",
        "top_n": TOP_N,
        "total_keywords": len(keywords),
        "per_keyword_results": {
            kw: {
                "duckduckgo": next(
                    (r for r in ddg_query_results if r["query"] == kw), {}
                ),
                "google": next(
                    (r for r in google_query_results if r["query"] == kw), {}
                ),
            }
            for kw in keywords
        },
    }
    save_results(combined_data, "engine_rankings.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"Engine Search Complete")
    print(f"  Experiment start: {context['experiment_start_utc']}")
    print(f"  Location: {geo['city']}, {geo['country']} (IP: {context['network']['public_ip']})")
    print(f"  DuckDuckGo: {len(ddg_query_results)}/{len(keywords)} keywords")
    print(f"  Google: {len(google_query_results)}/{len(keywords)} keywords")
    if ddg_errors:
        print(f"  DDG failures: {[e['keyword'] for e in ddg_errors]}")
    if google_errors:
        print(f"  Google failures: {[e['keyword'] for e in google_errors]}")
    print(f"  Results saved to results/")


if __name__ == "__main__":
    main()
