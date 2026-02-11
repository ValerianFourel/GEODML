#!/usr/bin/env python3
"""AI-powered search ranking: keywords → SearXNG → LLM re-ranking → ranked domains."""

import sys
import time
from datetime import datetime, timezone

from src.keywords import load_keywords
from src.searxng_client import search_searxng
from src.llm_ranker import rank_domains_with_llm
from src.results_io import save_results, results_to_csv
from src.config import TOP_N


def main():
    keywords = load_keywords()
    print(f"Loaded {len(keywords)} keywords")
    print(f"Top N domains per keyword: {TOP_N}\n")

    rankings = {}
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # Step 1: Get raw search results from SearXNG
        search_results = search_searxng(keyword, num_results=20)
        if not search_results:
            print(f"  No SearXNG results for '{keyword}', skipping")
            errors.append(keyword)
            continue
        print(f"  Got {len(search_results)} SearXNG results")

        # Step 2: LLM re-ranks and filters to top B2B SaaS domains
        domains = rank_domains_with_llm(keyword, search_results, top_n=TOP_N)
        print(f"  LLM ranked {len(domains)} domains: {domains[:3]}...")

        rankings[keyword] = domains

    # Build output data
    data = {
        "source": "ai_search",
        "method": "SearXNG + Mistral-7B-Instruct LLM re-ranking",
        "top_n": TOP_N,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_keywords": len(keywords),
        "successful_keywords": len(rankings),
        "failed_keywords": errors,
        "rankings": rankings,
    }

    # Save results
    save_results(data, "ai_search_rankings.json")
    results_to_csv(data, "ai_search_rankings.csv")

    # Summary
    print(f"\n{'='*60}")
    print(f"AI Search Complete")
    print(f"  Keywords processed: {len(rankings)}/{len(keywords)}")
    print(f"  Failed: {len(errors)}")
    if errors:
        print(f"  Failed keywords: {errors}")
    print(f"  Results saved to results/ai_search_rankings.json and .csv")


if __name__ == "__main__":
    main()
