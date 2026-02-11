#!/usr/bin/env python3
"""Read a results JSON file and print a ranking comparison summary.

Usage:
  python summarize_results.py results/ai_search_rankings.json
  python summarize_results.py results/ranking_comparison_20260211.json
"""

import json
import sys
import tldextract
from collections import Counter


def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def parse_ai_search_format(data: dict) -> dict:
    """Parse the output of run_ai_search.py (ai_search_rankings.json)."""
    context = data.get("experiment_context", {})
    network = context.get("network", {})
    geo = network.get("geolocation", {})

    results = []
    for r in data.get("per_keyword_results", []):
        keyword = r.get("query") or r.get("keyword", "")

        # Pre-LLM: extract domains from raw search results
        pre_llm = []
        raw = r.get("raw_results", [])
        if not raw and "searxng" in r:
            raw = r["searxng"].get("raw_results", [])
        if not raw:
            raw = r.get("sources", [])

        for rr in raw:
            d = rr.get("domain") or extract_domain(rr.get("url", ""))
            if d and d not in pre_llm:
                pre_llm.append(d)

        # Post-LLM: the LLM re-ranked domains
        post_llm = r.get("ranked_domains", [])
        if not post_llm and "llm" in r:
            post_llm = r["llm"].get("ranked_domains", [])

        # Timestamps
        search_ts = r.get("query_timestamp_utc", "")
        if not search_ts and "searxng" in r:
            search_ts = r["searxng"].get("query_timestamp_utc", "")

        llm_ts = ""
        if "llm" in r:
            llm_ts = r["llm"].get("llm_query_timestamp_utc", "")

        # Search backend
        backend = "unknown"
        if "searxng" in r:
            backend = r["searxng"].get("search_backend") or r["searxng"].get("searxng_instance", "searxng")
        elif "search_backend" in r:
            backend = r["search_backend"]

        results.append({
            "keyword": keyword,
            "pre_llm_domains": pre_llm,
            "post_llm_domains": post_llm,
            "search_timestamp": search_ts,
            "llm_timestamp": llm_ts,
            "search_backend": backend,
        })

    return {
        "experiment_start": context.get("experiment_start_utc", data.get("experiment_start", "")),
        "experiment_end": data.get("experiment_end_utc", data.get("experiment_end", "")),
        "search_backend": data.get("mode", "unknown"),
        "llm_model": data.get("chat_model") or data.get("method", "unknown"),
        "location": f"{geo.get('city', '?')}, {geo.get('country', '?')}" if geo else "unknown",
        "ip": network.get("public_ip", "unknown"),
        "results": results,
    }


def parse_comparison_format(data: dict) -> dict:
    """Parse the output of run_ranking_comparison.py."""
    return {
        "experiment_start": data.get("experiment_start", ""),
        "experiment_end": data.get("experiment_end", ""),
        "search_backend": data.get("search_backend", "unknown"),
        "llm_model": data.get("llm_model", "unknown"),
        "location": "unknown",
        "ip": "unknown",
        "results": [
            {
                "keyword": r["keyword"],
                "pre_llm_domains": r["pre_llm_domains"],
                "post_llm_domains": r["post_llm_domains"],
                "search_timestamp": r.get("search_timestamp", ""),
                "llm_timestamp": r.get("llm_timestamp", ""),
                "search_backend": r.get("search_backend", ""),
            }
            for r in data.get("results", [])
        ],
    }


def print_report(parsed: dict):
    w = 76

    print("\n" + "=" * w)
    print("  RANKING COMPARISON: Raw Search Engine vs LLM Re-ranking")
    print("=" * w)
    print(f"  Search engine:    {parsed['search_backend']}")
    print(f"  LLM re-ranker:    {parsed['llm_model']}")
    print(f"  Location:         {parsed['location']}")
    print(f"  IP:               {parsed['ip']}")
    print(f"  Experiment start: {parsed['experiment_start']}")
    print(f"  Experiment end:   {parsed['experiment_end']}")
    print(f"  Keywords:         {len(parsed['results'])}")
    print("=" * w)

    # Per-keyword stats accumulators
    all_promoted = Counter()
    all_demoted = Counter()
    all_dropped = Counter()
    all_added = Counter()
    total_overlap = 0
    total_reordered = 0
    total_pre = 0
    total_post = 0

    for r in parsed["results"]:
        pre = r["pre_llm_domains"]
        post = r["post_llm_domains"]
        top_n = max(len(pre), len(post))

        pre_set = set(pre)
        post_set = set(post)
        overlap = pre_set & post_set

        total_pre += len(pre)
        total_post += len(post)
        total_overlap += len(overlap)

        print(f"\n  Keyword: {r['keyword']}")
        print(f"  Search: {r['search_timestamp']}")
        print(f"  LLM:    {r['llm_timestamp']}")
        print(f"  {'─' * (w - 4)}")
        print(f"  {'#':<4} {'Raw Search':<32} {'LLM Re-ranked':<32} {'Change'}")
        print(f"  {'─'*4} {'─'*32} {'─'*32} {'─'*8}")

        for rank in range(top_n):
            pre_d = pre[rank] if rank < len(pre) else ""
            post_d = post[rank] if rank < len(post) else ""

            change = ""
            if post_d and post_d in pre:
                old_rank = pre.index(post_d) + 1
                new_rank = rank + 1
                diff = old_rank - new_rank
                if diff > 0:
                    change = f" +{diff}"
                elif diff < 0:
                    change = f" {diff}"
                else:
                    change = "  ="
            elif post_d:
                change = " NEW"

            print(f"  {rank+1:<4} {pre_d:<32} {post_d:<32} {change}")

        # Per-keyword changes
        promoted = []
        demoted = []
        dropped = []
        added = []

        for d in post:
            if d not in pre_set:
                added.append(d)
                all_added[d] += 1
            elif post.index(d) < pre.index(d):
                promoted.append(d)
                all_promoted[d] += 1
            elif post.index(d) > pre.index(d):
                demoted.append(d)
                all_demoted[d] += 1

        for d in pre:
            if d not in post_set:
                dropped.append(d)
                all_dropped[d] += 1

        for d in overlap:
            if pre.index(d) != post.index(d):
                total_reordered += 1

        if promoted:
            print(f"  Promoted: {', '.join(promoted)}")
        if demoted:
            print(f"  Demoted:  {', '.join(demoted)}")
        if dropped:
            print(f"  Dropped:  {', '.join(dropped)}")
        if added:
            print(f"  Added:    {', '.join(added)}")

    # Global summary
    n = len(parsed["results"])
    if n == 0:
        print("\n  No results to summarize.")
        return

    print(f"\n{'=' * w}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * w}")
    print(f"  Keywords processed:         {n}")
    print(f"  Avg domains/keyword (raw):  {total_pre / n:.1f}")
    print(f"  Avg domains/keyword (LLM):  {total_post / n:.1f}")
    print(f"  Avg overlap:                {total_overlap / n:.1f}")
    print(f"  Avg reordered:              {total_reordered / n:.1f}")
    print(f"  Overlap rate:               {total_overlap / max(total_post, 1) * 100:.0f}%")

    # Most frequently dropped / added domains
    if all_dropped:
        print(f"\n  Most dropped by LLM (removed from raw ranking):")
        for domain, count in all_dropped.most_common(10):
            print(f"    {domain:<35} dropped {count}x")

    if all_added:
        print(f"\n  Most added by LLM (not in raw ranking):")
        for domain, count in all_added.most_common(10):
            print(f"    {domain:<35} added {count}x")

    if all_promoted:
        print(f"\n  Most promoted by LLM (moved up):")
        for domain, count in all_promoted.most_common(10):
            print(f"    {domain:<35} promoted {count}x")

    print(f"\n{'=' * w}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <results_file.json>")
        print("  e.g. python summarize_results.py results/ai_search_rankings.json")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath) as f:
        data = json.load(f)

    # Detect format
    if "per_keyword_results" in data:
        parsed = parse_ai_search_format(data)
    elif "results" in data and data["results"] and "pre_llm_domains" in data["results"][0]:
        parsed = parse_comparison_format(data)
    else:
        print(f"ERROR: Unrecognized JSON format in {filepath}")
        print(f"  Top-level keys: {list(data.keys())}")
        sys.exit(1)

    print(f"  Loaded: {filepath}")
    print_report(parsed)


if __name__ == "__main__":
    main()
