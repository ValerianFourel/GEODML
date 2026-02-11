#!/usr/bin/env python3
"""Compare pre-LLM (raw search) vs post-LLM (re-ranked) domain rankings.

Runs each keyword through: Search Engine → LLM re-ranker
Then prints a side-by-side comparison showing what the LLM changed.

Usage:
  python run_ranking_comparison.py
  python run_ranking_comparison.py --top 5
  python run_ranking_comparison.py --keywords 3   # only first 3 keywords
"""

import argparse
import json
import os
from datetime import datetime, timezone

import tldextract

from src.keywords import load_keywords
from src.searxng_client import search_searxng
from src.llm_ranker import rank_domains_with_llm, MODEL_ID
from src.config import TOP_N, RESULTS_DIR


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def run_comparison(keywords: list[str], top_n: int):
    experiment_start = datetime.now(timezone.utc).isoformat()
    all_results = []

    for i, kw in enumerate(keywords, 1):
        print(f"\n  [{i}/{len(keywords)}] Searching: {kw}")

        # Raw search
        search = search_searxng(kw, num_results=20)
        raw = search["raw_results"]

        if not raw:
            print(f"    No results, skipping")
            continue

        # Extract pre-LLM domain ranking (original SERP order, skip non-product)
        pre_llm_domains = []
        for r in raw:
            d = extract_domain(r["url"])
            if d and d not in pre_llm_domains:
                pre_llm_domains.append(d)

        # LLM re-ranking
        llm = rank_domains_with_llm(kw, raw, top_n=top_n)
        post_llm_domains = llm["ranked_domains"]

        all_results.append({
            "keyword": kw,
            "search_backend": search["search_backend"],
            "searxng_instance": search.get("searxng_instance"),
            "search_timestamp": search["query_timestamp_utc"],
            "llm_timestamp": llm["llm_query_timestamp_utc"],
            "pre_llm_domains": pre_llm_domains[:top_n],
            "post_llm_domains": post_llm_domains,
            "llm_error": llm["error"],
            "used_fallback": llm["used_fallback"],
        })

    experiment_end = datetime.now(timezone.utc).isoformat()

    return {
        "experiment_start": experiment_start,
        "experiment_end": experiment_end,
        "search_backend": all_results[0]["search_backend"] if all_results else "unknown",
        "searxng_instance": all_results[0].get("searxng_instance") if all_results else None,
        "llm_model": MODEL_ID,
        "top_n": top_n,
        "results": all_results,
    }


def print_report(data: dict):
    w = 72

    # Header
    print("\n" + "=" * w)
    print("  RANKING COMPARISON: Search Engine vs LLM Re-ranking")
    print("=" * w)

    backend = data["search_backend"]
    instance = data.get("searxng_instance")
    search_label = f"{backend}" + (f" ({instance})" if instance else "")

    print(f"  Search engine:  {search_label}")
    print(f"  LLM re-ranker:  {data['llm_model']}")
    print(f"  Top N:          {data['top_n']}")
    print(f"  Start:          {data['experiment_start']}")
    print(f"  End:            {data['experiment_end']}")
    print(f"  Keywords:       {len(data['results'])}")
    print("=" * w)

    for r in data["results"]:
        pre = r["pre_llm_domains"]
        post = r["post_llm_domains"]
        top_n = len(pre)

        print(f"\n  Keyword: {r['keyword']}")
        print(f"  Search: {r['search_timestamp']}  |  LLM: {r['llm_timestamp']}")
        if r["llm_error"]:
            print(f"  LLM error: {r['llm_error'][:80]}")
        print(f"  {'─' * (w - 4)}")

        # Header row
        col = max(top_n, len(post))
        print(f"  {'#':<4} {'Raw Search':<30} {'LLM Re-ranked':<30} {'Change'}")
        print(f"  {'─'*4} {'─'*30} {'─'*30} {'─'*10}")

        for rank in range(max(len(pre), len(post))):
            pre_d = pre[rank] if rank < len(pre) else ""
            post_d = post[rank] if rank < len(post) else ""

            # Compute rank change
            change = ""
            if post_d and post_d in pre:
                old_rank = pre.index(post_d) + 1
                new_rank = rank + 1
                diff = old_rank - new_rank
                if diff > 0:
                    change = f"  +{diff}"
                elif diff < 0:
                    change = f"  {diff}"
                else:
                    change = "   ="
            elif post_d and post_d not in pre:
                change = " NEW"
            elif not post_d:
                change = ""

            print(f"  {rank+1:<4} {pre_d:<30} {post_d:<30} {change}")

        # Summary for this keyword
        promoted = []
        demoted = []
        dropped = []
        added = []

        pre_set = set(pre)
        post_set = set(post)

        for d in post:
            if d not in pre_set:
                added.append(d)
            elif post.index(d) < pre.index(d):
                promoted.append(d)
            elif post.index(d) > pre.index(d):
                demoted.append(d)

        for d in pre:
            if d not in post_set:
                dropped.append(d)

        if promoted:
            print(f"  Promoted: {', '.join(promoted)}")
        if demoted:
            print(f"  Demoted:  {', '.join(demoted)}")
        if dropped:
            print(f"  Dropped:  {', '.join(dropped)}")
        if added:
            print(f"  Added:    {', '.join(added)}")

    # Global summary
    print(f"\n{'=' * w}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * w}")

    total_pre = 0
    total_post = 0
    total_overlap = 0
    total_reordered = 0

    for r in data["results"]:
        pre = r["pre_llm_domains"]
        post = r["post_llm_domains"]
        pre_set = set(pre)
        post_set = set(post)
        overlap = pre_set & post_set
        total_pre += len(pre)
        total_post += len(post)
        total_overlap += len(overlap)

        # Count domains whose rank changed
        for d in overlap:
            if pre.index(d) != post.index(d):
                total_reordered += 1

    n = len(data["results"])
    print(f"  Keywords processed:       {n}")
    print(f"  Avg raw domains/keyword:  {total_pre / n:.1f}")
    print(f"  Avg LLM domains/keyword:  {total_post / n:.1f}")
    print(f"  Avg overlap:              {total_overlap / n:.1f}")
    print(f"  Avg reordered:            {total_reordered / n:.1f}")
    print(f"  Overlap rate:             {total_overlap / max(total_post, 1) * 100:.0f}%")
    print(f"{'=' * w}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare search vs LLM rankings")
    parser.add_argument("--top", type=int, default=TOP_N, help="Top N domains to compare")
    parser.add_argument("--keywords", type=int, default=0, help="Limit to first N keywords (0 = all)")
    args = parser.parse_args()

    keywords = load_keywords()
    if args.keywords > 0:
        keywords = keywords[:args.keywords]

    print(f"Running ranking comparison: {len(keywords)} keywords, top {args.top}")

    data = run_comparison(keywords, args.top)

    # Print report
    print_report(data)

    # Save JSON
    out_path = RESULTS_DIR / f"ranking_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Full data saved to: {out_path}")


if __name__ == "__main__":
    main()
