#!/usr/bin/env python3
"""Read a results JSON file and print a ranking comparison summary.

Decomposes the LLM re-ranking into two independent effects:
  1. FILTERING — which domains the LLM drops/keeps from the raw SERP
  2. REORDERING — among kept domains, how the LLM shuffles their relative order

The raw "rank delta" conflates both effects. This script separates them
using Spearman rho and Kendall tau on relative ranks within the overlap set.

Usage:
  python summarize_results.py results/ai_search_rankings.json
  python summarize_results.py results/ranking_comparison_20260211.json
"""

import json
import math
import sys
import tldextract
from collections import Counter, defaultdict


# ── Helpers ───────────────────────────────────────────────────────────────

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _stats(values: list[float]) -> dict:
    """Compute mean, median, std, min, max for a list of numbers."""
    if not values:
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {"n": n, "mean": mean, "median": median, "std": std, "min": min(values), "max": max(values)}


def _to_relative_ranks(values: list[float]) -> list[float]:
    """Convert absolute values to relative ranks (1-indexed, dense)."""
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    for rank_0, (orig_idx, _) in enumerate(sorted_vals):
        ranks[orig_idx] = rank_0 + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two sequences.

    Converts to ranks internally, handles ties via dense ranking.
    Returns rho in [-1, +1].  +1 = identical order, -1 = reversed.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    rx = _to_relative_ranks(x)
    ry = _to_relative_ranks(y)
    d_sq_sum = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))


def kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall tau-b rank correlation between two sequences.

    Returns tau in [-1, +1].  +1 = identical order, -1 = reversed.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # ties (product == 0) are ignored in tau-b numerator
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return float("nan")
    return (concordant - discordant) / total_pairs


def compute_rank_change_vector(pre_domains: list[str], post_domains: list[str]) -> list[dict]:
    """Compute the rank change vector between pre-LLM and post-LLM domain orderings.

    For each domain in post_domains:
        - pre_rank: 1-indexed position in pre list (None if not present)
        - post_rank: 1-indexed position in post list
        - rank_delta: pre_rank - post_rank (positive = promoted, None if new)
    """
    pre_rank_map = {d: i + 1 for i, d in enumerate(pre_domains)}
    changes = []
    for post_rank_0, domain in enumerate(post_domains):
        post_rank = post_rank_0 + 1
        pre_rank = pre_rank_map.get(domain)
        rank_delta = (pre_rank - post_rank) if pre_rank is not None else None
        changes.append({
            "domain": domain,
            "pre_rank": pre_rank,
            "post_rank": post_rank,
            "rank_delta": rank_delta,
        })
    return changes


def compute_reorder_metrics(pre_domains: list[str], post_domains: list[str]) -> dict:
    """Compute reordering metrics on the OVERLAP set only.

    Isolates the pure reordering effect from the filtering effect by
    looking only at domains present in both lists, using their original
    absolute ranks as paired observations for Spearman / Kendall.

    Returns dict with:
        overlap_n, overlap_domains,
        pre_ranks, post_ranks,
        spearman_rho, kendall_tau,
        relative_deltas  (within-overlap rank shifts)
    """
    pre_rank_map = {d: i + 1 for i, d in enumerate(pre_domains)}
    post_rank_map = {d: i + 1 for i, d in enumerate(post_domains)}
    overlap = [d for d in post_domains if d in pre_rank_map]

    if len(overlap) < 2:
        return {
            "overlap_n": len(overlap),
            "overlap_domains": overlap,
            "pre_ranks": [pre_rank_map[d] for d in overlap] if overlap else [],
            "post_ranks": [post_rank_map[d] for d in overlap] if overlap else [],
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
            "relative_deltas": [],
        }

    # Absolute ranks of overlap domains in each list
    pre_ranks = [pre_rank_map[d] for d in overlap]
    post_ranks = [post_rank_map[d] for d in overlap]

    rho = spearman_rho(pre_ranks, post_ranks)
    tau = kendall_tau(pre_ranks, post_ranks)

    # Relative deltas: among the overlap set, assign dense relative ranks
    # and measure how each domain shifted within that subset
    pre_rel = _to_relative_ranks(pre_ranks)
    post_rel = _to_relative_ranks(post_ranks)
    relative_deltas = [int(pr - po) for pr, po in zip(pre_rel, post_rel)]

    return {
        "overlap_n": len(overlap),
        "overlap_domains": overlap,
        "pre_ranks": pre_ranks,
        "post_ranks": post_ranks,
        "spearman_rho": rho,
        "kendall_tau": tau,
        "relative_deltas": relative_deltas,
    }


# ── Parsers ───────────────────────────────────────────────────────────────

def parse_ai_search_format(data: dict) -> dict:
    """Parse the output of run_ai_search.py (ai_search_rankings.json)."""
    context = data.get("experiment_context", {})
    network = context.get("network", {})
    geo = network.get("geolocation", {})

    results = []
    for r in data.get("per_keyword_results", []):
        keyword = r.get("query") or r.get("keyword", "")

        # Pre-LLM: extract domains + URLs from raw search results
        pre_llm = []
        pre_llm_urls = {}
        raw = r.get("raw_results", [])
        if not raw and "serp" in r:
            raw = r["serp"].get("raw_results", [])
        if not raw and "searxng" in r:
            raw = r["searxng"].get("raw_results", [])
        if not raw:
            raw = r.get("sources", [])

        for rr in raw:
            url = rr.get("url", "")
            d = rr.get("domain") or extract_domain(url)
            if d and d not in pre_llm:
                pre_llm.append(d)
                pre_llm_urls[d] = url

        # Post-LLM: the LLM re-ranked domains + URLs
        post_llm = r.get("ranked_domains", [])
        if not post_llm and "llm" in r:
            post_llm = r["llm"].get("ranked_domains", [])

        post_llm_urls = {}
        ranked_results = r.get("ranked_results", [])
        if not ranked_results and "llm" in r:
            ranked_results = r["llm"].get("ranked_results", [])
        for entry in ranked_results:
            post_llm_urls[entry["domain"]] = entry.get("url", "")

        # Rank changes: use stored vector if available, otherwise compute
        rank_changes = r.get("rank_changes")
        if rank_changes is None:
            rank_changes = compute_rank_change_vector(pre_llm, post_llm)

        # Timestamps
        search_ts = r.get("query_timestamp_utc", "")
        if not search_ts and "serp" in r:
            search_ts = r["serp"].get("query_timestamp_utc", "")
        if not search_ts and "searxng" in r:
            search_ts = r["searxng"].get("query_timestamp_utc", "")

        llm_ts = ""
        if "llm" in r:
            llm_ts = r["llm"].get("llm_query_timestamp_utc", "")

        # Search backend
        backend = "unknown"
        if "serp" in r:
            backend = r["serp"].get("search_backend", "unknown")
        elif "searxng" in r:
            backend = r["searxng"].get("search_backend") or r["searxng"].get("searxng_instance", "searxng")
        elif "search_backend" in r:
            backend = r["search_backend"]

        results.append({
            "keyword": keyword,
            "pre_llm_domains": pre_llm,
            "pre_llm_urls": pre_llm_urls,
            "post_llm_domains": post_llm,
            "post_llm_urls": post_llm_urls,
            "rank_changes": rank_changes,
            "search_timestamp": search_ts,
            "llm_timestamp": llm_ts,
            "search_backend": backend,
        })

    return {
        "experiment_start": context.get("experiment_start_utc", data.get("experiment_start", "")),
        "experiment_end": data.get("experiment_end_utc", data.get("experiment_end", "")),
        "search_backend": data.get("serp_engine") or data.get("mode", "unknown"),
        "llm_model": data.get("chat_model") or data.get("method", "unknown"),
        "location": f"{geo.get('city', '?')}, {geo.get('country', '?')}" if geo else "unknown",
        "ip": network.get("public_ip", "unknown"),
        "results": results,
    }


def parse_comparison_format(data: dict) -> dict:
    """Parse the output of run_ranking_comparison.py."""
    parsed_results = []
    for r in data.get("results", []):
        pre = r["pre_llm_domains"]
        post = r["post_llm_domains"]
        rank_changes = compute_rank_change_vector(pre, post)
        parsed_results.append({
            "keyword": r["keyword"],
            "pre_llm_domains": pre,
            "post_llm_domains": post,
            "rank_changes": rank_changes,
            "search_timestamp": r.get("search_timestamp", ""),
            "llm_timestamp": r.get("llm_timestamp", ""),
            "search_backend": r.get("search_backend", ""),
        })

    return {
        "experiment_start": data.get("experiment_start", ""),
        "experiment_end": data.get("experiment_end", ""),
        "search_backend": data.get("search_backend", "unknown"),
        "llm_model": data.get("llm_model", "unknown"),
        "location": "unknown",
        "ip": "unknown",
        "results": parsed_results,
    }


# ── Report ────────────────────────────────────────────────────────────────

def print_report(parsed: dict):
    w = 80

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

    # ── Accumulators ──────────────────────────────────────────────────────
    all_promoted = Counter()
    all_demoted = Counter()
    all_dropped = Counter()
    all_added = Counter()
    total_overlap = 0
    total_pre = 0
    total_post = 0

    # Raw delta accumulators
    all_raw_deltas = []
    per_keyword_raw_deltas = []

    # Reorder accumulators
    per_kw_spearman = []
    per_kw_kendall = []
    per_kw_reorder = []       # (keyword, metrics_dict)
    all_relative_deltas = []  # relative deltas across all keywords

    # Filtering accumulators
    kept_serp_positions = []    # SERP position of domains the LLM kept
    dropped_serp_positions = [] # SERP position of domains the LLM dropped

    # ── Per-keyword loop ──────────────────────────────────────────────────
    for r in parsed["results"]:
        pre = r["pre_llm_domains"]
        post = r["post_llm_domains"]
        pre_urls = r.get("pre_llm_urls", {})
        post_urls = r.get("post_llm_urls", {})
        rank_changes = r.get("rank_changes", [])
        top_n = max(len(pre), len(post))

        pre_set = set(pre)
        post_set = set(post)
        overlap = pre_set & post_set

        total_pre += len(pre)
        total_post += len(post)
        total_overlap += len(overlap)

        # Compute reorder metrics for this keyword
        reorder = compute_reorder_metrics(pre, post)
        per_kw_reorder.append((r["keyword"], reorder))
        if not math.isnan(reorder["spearman_rho"]):
            per_kw_spearman.append(reorder["spearman_rho"])
            per_kw_kendall.append(reorder["kendall_tau"])
        all_relative_deltas.extend(reorder["relative_deltas"])

        # Filtering: track SERP positions of kept vs dropped
        pre_rank_map = {d: i + 1 for i, d in enumerate(pre)}
        for d in pre:
            pos = pre_rank_map[d]
            if d in post_set:
                kept_serp_positions.append(pos)
            else:
                dropped_serp_positions.append(pos)

        # ── Per-keyword table ─────────────────────────────────────────────
        print(f"\n  Keyword: {r['keyword']}")
        print(f"  Search: {r['search_timestamp']}")
        print(f"  LLM:    {r['llm_timestamp']}")
        print(f"  {'─' * (w - 4)}")
        print(f"  {'#':<4} {'Raw Search':<32} {'LLM Re-ranked':<32} {'Delta'}")
        print(f"  {'─'*4} {'─'*32} {'─'*32} {'─'*8}")

        for rank in range(top_n):
            pre_d = pre[rank] if rank < len(pre) else ""
            post_d = post[rank] if rank < len(post) else ""

            change = ""
            if post_d and post_d in pre_set:
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

        # Raw rank change vector
        keyword_deltas = [rc["rank_delta"] for rc in rank_changes if rc["rank_delta"] is not None]
        new_count = sum(1 for rc in rank_changes if rc["rank_delta"] is None)
        per_keyword_raw_deltas.append(keyword_deltas)
        all_raw_deltas.extend(keyword_deltas)

        # Reorder info for this keyword
        print(f"  {'─' * (w - 4)}")
        if keyword_deltas or new_count:
            print(f"  Raw delta vector: {keyword_deltas}"
                  + (f"  (+{new_count} new)" if new_count else ""))
        if reorder["relative_deltas"]:
            print(f"  Relative reorder: {reorder['relative_deltas']}"
                  f"  (among {reorder['overlap_n']} shared domains)")
        if not math.isnan(reorder["spearman_rho"]):
            print(f"  Spearman rho={reorder['spearman_rho']:+.3f}"
                  f"   Kendall tau={reorder['kendall_tau']:+.3f}")

        # URLs
        if post_urls:
            print(f"  {'─' * (w - 4)}")
            print(f"  URLs (LLM re-ranked):")
            for rank, d in enumerate(post, 1):
                url = post_urls.get(d, pre_urls.get(d, ""))
                if url:
                    print(f"    {rank}. {url}")

        # Promoted / demoted / dropped / added
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

        if promoted:
            print(f"  Promoted: {', '.join(promoted)}")
        if demoted:
            print(f"  Demoted:  {', '.join(demoted)}")
        if dropped:
            print(f"  Dropped:  {', '.join(dropped)}")
        if added:
            print(f"  Added:    {', '.join(added)}")

    # ══════════════════════════════════════════════════════════════════════
    #  GLOBAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
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
    print(f"  Overlap rate:               {total_overlap / max(total_post, 1) * 100:.0f}%")

    if all_dropped:
        print(f"\n  Most dropped by LLM:")
        for domain, count in all_dropped.most_common(10):
            print(f"    {domain:<35} dropped {count}x")
    if all_added:
        print(f"\n  Most added by LLM (hallucinated or from training data):")
        for domain, count in all_added.most_common(10):
            print(f"    {domain:<35} added {count}x")

    # ══════════════════════════════════════════════════════════════════════
    #  EFFECT 1: FILTERING (which domains the LLM drops/keeps)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * w}")
    print("  EFFECT 1: FILTERING (which SERP domains the LLM drops vs keeps)")
    print(f"{'=' * w}")

    if kept_serp_positions and dropped_serp_positions:
        ks = _stats(kept_serp_positions)
        ds = _stats(dropped_serp_positions)
        print(f"\n  The LLM selects {total_post} domains from {total_pre} raw SERP results.")
        print(f"  This is the dominant source of the positive raw-delta bias.")
        print(f"\n  SERP position of KEPT domains:    mean={ks['mean']:.1f}  median={ks['median']:.1f}  "
              f"range=[{ks['min']:.0f}, {ks['max']:.0f}]")
        print(f"  SERP position of DROPPED domains: mean={ds['mean']:.1f}  median={ds['median']:.1f}  "
              f"range=[{ds['min']:.0f}, {ds['max']:.0f}]")

        expected_delta = ks["mean"] - (total_post / n + 1) / 2
        print(f"\n  Avg SERP position of kept domains: {ks['mean']:.1f}")
        print(f"  Avg post-LLM position (mechanical): {(total_post / n + 1) / 2:.1f}")
        print(f"  => Expected raw delta from filtering alone: +{expected_delta:.1f}")
        if all_raw_deltas:
            raw_s = _stats(all_raw_deltas)
            print(f"  => Observed mean raw delta:                +{raw_s['mean']:.1f}")
            residual = raw_s["mean"] - expected_delta
            print(f"  => Residual (attributable to reordering):  {residual:+.1f}")

        # Distribution: where do kept vs dropped domains come from?
        print(f"\n  Kept domains by SERP position bucket:")
        kept_c = Counter()
        drop_c = Counter()
        for p in kept_serp_positions:
            bucket = f"{((p-1)//5)*5+1}-{((p-1)//5)*5+5}"
            kept_c[bucket] += 1
        for p in dropped_serp_positions:
            bucket = f"{((p-1)//5)*5+1}-{((p-1)//5)*5+5}"
            drop_c[bucket] += 1
        all_buckets = sorted(set(list(kept_c.keys()) + list(drop_c.keys())))
        print(f"    {'Bucket':<10} {'Kept':>6} {'Dropped':>8} {'Keep %':>8}")
        for b in all_buckets:
            k = kept_c.get(b, 0)
            d = drop_c.get(b, 0)
            total_b = k + d
            pct = k / total_b * 100 if total_b else 0
            print(f"    {b:<10} {k:>6} {d:>8} {pct:>7.0f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  EFFECT 2: REORDERING (how the LLM shuffles kept domains)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * w}")
    print("  EFFECT 2: REORDERING (among shared domains, does the LLM change order?)")
    print(f"{'=' * w}")

    if per_kw_spearman:
        sp_s = _stats(per_kw_spearman)
        kt_s = _stats(per_kw_kendall)
        print(f"\n  Spearman rho (1.0 = SERP order preserved, -1.0 = reversed):")
        print(f"    Mean:    {sp_s['mean']:+.3f}")
        print(f"    Median:  {sp_s['median']:+.3f}")
        print(f"    Std:     {sp_s['std']:.3f}")
        print(f"    Range:   [{sp_s['min']:+.3f}, {sp_s['max']:+.3f}]")

        print(f"\n  Kendall tau (1.0 = all pairs concordant, -1.0 = all discordant):")
        print(f"    Mean:    {kt_s['mean']:+.3f}")
        print(f"    Median:  {kt_s['median']:+.3f}")
        print(f"    Std:     {kt_s['std']:.3f}")
        print(f"    Range:   [{kt_s['min']:+.3f}, {kt_s['max']:+.3f}]")

        # Interpretation
        print(f"\n  Interpretation:")
        if sp_s["mean"] > 0.7:
            print(f"    The LLM mostly PRESERVES the search engine's relative ordering.")
            print(f"    Its main effect is filtering (dropping non-product domains),")
            print(f"    not reordering the product domains it keeps.")
        elif sp_s["mean"] > 0.3:
            print(f"    The LLM PARTIALLY preserves the search engine's ordering")
            print(f"    but also does meaningful reordering of the kept domains.")
        elif sp_s["mean"] > -0.3:
            print(f"    The LLM shows LITTLE correlation with the SERP ordering.")
            print(f"    It effectively ranks from scratch rather than re-ranking.")
        else:
            print(f"    The LLM INVERTS the search engine's ordering.")

        # Per-keyword Spearman / Kendall table
        print(f"\n  Per-keyword Spearman / Kendall:")
        for r_data, (kw, reorder) in zip(parsed["results"], per_kw_reorder):
            if not math.isnan(reorder["spearman_rho"]):
                print(f"    {kw:<40} rho={reorder['spearman_rho']:+.3f}"
                      f"  tau={reorder['kendall_tau']:+.3f}"
                      f"  (n={reorder['overlap_n']})")
            else:
                print(f"    {kw:<40} (< 2 shared domains)")
        print(f"  {'─' * (w - 4)}")
        print(f"  Mean Spearman: {sp_s['mean']:+.3f}   Mean Kendall: {kt_s['mean']:+.3f}")

        # Relative delta histogram (within-overlap reordering)
        if all_relative_deltas:
            print(f"\n  Relative reorder delta histogram (within overlap set only):")
            print(f"  (positive = domain moved up relative to other kept domains)")
            rd_buckets = Counter(all_relative_deltas)
            for delta in sorted(rd_buckets.keys()):
                bar = "#" * rd_buckets[delta]
                print(f"    {delta:+3d}: {rd_buckets[delta]:>3}  {bar}")

            rel_s = _stats([float(d) for d in all_relative_deltas])
            print(f"\n  Relative delta stats:")
            print(f"    Mean:   {rel_s['mean']:+.2f}  (should be ~0 by construction)")
            print(f"    Std:    {rel_s['std']:.2f}  (measures reorder intensity)")
            abs_rel = _stats([abs(float(d)) for d in all_relative_deltas])
            print(f"    Mean |delta|: {abs_rel['mean']:.2f}  (avg positions moved among kept)")
    else:
        print("\n  Not enough overlapping domains to compute rank correlations.")

    # ══════════════════════════════════════════════════════════════════════
    #  RAW DELTA ANALYSIS (conflated — for reference)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * w}")
    print("  RAW DELTA ANALYSIS (filtering + reordering conflated)")
    print(f"{'=' * w}")

    if all_raw_deltas:
        gs = _stats(all_raw_deltas)
        abs_s = _stats([abs(d) for d in all_raw_deltas])
        print(f"\n  NOTE: These numbers are dominated by the filtering effect.")
        print(f"  The mean delta of ~+{gs['mean']:.0f} mostly reflects that the LLM")
        print(f"  drops top-ranked non-product domains, mechanically inflating deltas.")
        print(f"\n  Total observations: {gs['n']}")
        print(f"  Mean delta:     {gs['mean']:+.2f}")
        print(f"  Median delta:   {gs['median']:+.1f}")
        print(f"  Std:            {gs['std']:.2f}")
        print(f"  Mean |delta|:   {abs_s['mean']:.2f}")

        # Delta histogram
        buckets = Counter(all_raw_deltas)
        print(f"\n  Raw delta histogram:")
        for delta in sorted(buckets.keys()):
            bar = "#" * buckets[delta]
            print(f"    {delta:+3d}: {buckets[delta]:>3}  {bar}")

        # Per-keyword mean raw delta
        print(f"\n  Per-keyword mean raw delta:")
        kw_means = []
        for r, deltas in zip(parsed["results"], per_keyword_raw_deltas):
            if deltas:
                s = _stats(deltas)
                kw_means.append(s["mean"])
                print(f"    {r['keyword']:<40} mean={s['mean']:+.1f}  n={s['n']}")
            else:
                print(f"    {r['keyword']:<40} (no overlap)")
        if kw_means:
            kw_s = _stats(kw_means)
            print(f"  {'─' * (w - 4)}")
            print(f"  Mean of per-keyword means: {kw_s['mean']:+.2f}  std={kw_s['std']:.2f}")

    print(f"\n{'=' * w}\n")


# ── Main ──────────────────────────────────────────────────────────────────

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
