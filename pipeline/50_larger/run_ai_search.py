#!/usr/bin/env python3
"""AI-powered search ranking experiment — 50_larger variant.

Same pipeline as the root experiment but with:
  - 50 raw SERP results from SearXNG (instead of 20)
  - LLM re-ranks top 20 domains (instead of 10)

Pipeline: keyword → search engine (SERP, 50 results) → LLM re-ranking (top 20)

Usage:
  python 50_larger/run_ai_search.py
  python 50_larger/run_ai_search.py --engine searxng --keywords 5
"""

import argparse
import sys
import tldextract
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keywords import load_keywords
from src.engine_scraper import search, ENGINES
from src.llm_ranker import rank_domains_with_llm, MODEL_ID
from src.experiment_context import collect_experiment_context, utcnow_iso
from src.results_io import save_results, results_to_csv
from src.config import RESULTS_DIR as _ORIG_RESULTS_DIR

# ── 50_larger overrides ──────────────────────────────────────────────────────
NUM_SERP_RESULTS = 50   # SearXNG returns 50 results (was 20)
TOP_N = 20              # LLM re-ranks top 20 (was 10)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def compute_rank_changes(raw_results: list[dict], post_llm_domains: list[str]) -> list[dict]:
    """Compute the rank change vector between pre-LLM and post-LLM domain orderings.

    Returns a list of dicts (one per post-LLM domain):
        domain, pre_rank (1-indexed or null), post_rank (1-indexed),
        rank_delta (pre - post; positive = promoted, null if new)
    """
    # Build pre-LLM domain ranking (deduplicated, first occurrence wins)
    pre_domains = []
    for r in raw_results:
        d = _extract_domain(r.get("url", ""))
        if d and d not in pre_domains:
            pre_domains.append(d)
    pre_rank_map = {d: i + 1 for i, d in enumerate(pre_domains)}

    changes = []
    for post_rank_0, domain in enumerate(post_llm_domains):
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


def main():
    parser = argparse.ArgumentParser(description="AI-powered search ranking experiment (50_larger)")
    parser.add_argument("--engine", type=str, default="searxng",
                        choices=ENGINES,
                        help=f"Search engine for SERP (default: searxng). Choices: {', '.join(ENGINES)}")
    parser.add_argument("--keywords", type=int, default=0, help="Limit to first N keywords (0 = all)")
    parser.add_argument("--top", type=int, default=TOP_N, help="Top N domains per keyword (default: 20)")
    args = parser.parse_args()

    # Collect experiment provenance
    print("Collecting experiment context (IP, geolocation, machine info)...")
    context = collect_experiment_context()
    print(f"  IP: {context['network']['public_ip']}")
    geo = context["network"]["geolocation"]
    print(f"  Location: {geo['city']}, {geo['region']}, {geo['country']}")
    print(f"  ISP: {geo['isp']}")
    print()

    keywords = load_keywords()
    if args.keywords > 0:
        keywords = keywords[:args.keywords]

    print(f"Loaded {len(keywords)} keywords")
    print(f"SERP engine: {args.engine}")
    print(f"SERP results requested: {NUM_SERP_RESULTS}")
    print(f"Top N domains per keyword: {args.top}")
    print(f"LLM: {MODEL_ID}")
    print()

    per_keyword_data = []
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # Step 1: Raw search results — 50 results (was 20)
        serp_result = search(args.engine, keyword, num_results=NUM_SERP_RESULTS)
        raw_results = serp_result["raw_results"]

        if not raw_results:
            print(f"  No results, skipping")
            errors.append({"keyword": keyword, "error": serp_result.get("error")})
            continue

        print(f"  Got {len(raw_results)} results via {args.engine}")

        # Step 2: LLM re-ranks top 20 (was 10)
        llm_result = rank_domains_with_llm(keyword, raw_results, top_n=args.top)
        domains = llm_result["ranked_domains"]
        print(f"  LLM re-ranked: {domains[:5]}{'...' if len(domains) > 5 else ''}")

        if llm_result["used_fallback"]:
            print(f"  (used fallback due to LLM error)")

        # Step 3: Compute rank change vector (pre-LLM vs post-LLM)
        rank_changes = compute_rank_changes(raw_results, domains)
        deltas = [rc["rank_delta"] for rc in rank_changes if rc["rank_delta"] is not None]
        print(f"  Rank deltas: {deltas}")

        per_keyword_data.append({
            "query": keyword,
            "query_timestamp_utc": serp_result["query_timestamp_utc"],
            "serp": serp_result,
            "llm": llm_result,
            "ranked_domains": domains,
            "ranked_results": llm_result["ranked_results"],
            "rank_changes": rank_changes,
        })

    # Build output
    data = {
        "experiment_context": context,
        "experiment_end_utc": utcnow_iso(),
        "source": "ai_search",
        "serp_engine": args.engine,
        "method": f"{args.engine} + {MODEL_ID} LLM re-ranking",
        "chat_model": MODEL_ID,
        "top_n": args.top,
        "num_serp_results": NUM_SERP_RESULTS,
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }

    # Build filename: {engine}_{llm_model}_{date}.{ext}
    llm_label = MODEL_ID.split("/")[-1]
    date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    base_name = f"{args.engine}_{llm_label}_{date_label}"

    # Save to 50_larger/results/
    json_path = RESULTS_DIR / f"{base_name}.json"
    import json
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {json_path}")

    csv_rows = []
    for kd in per_keyword_data:
        raw = kd["serp"].get("raw_results", [])
        csv_rows.append({
            "query": kd["query"],
            "query_timestamp_utc": kd["query_timestamp_utc"],
            "raw_results": raw,
            "ranked_domains": kd["ranked_domains"],
            "ranked_results": kd["ranked_results"],
        })

    # Write CSV manually to 50_larger/results/
    import csv
    csv_path = RESULTS_DIR / f"{base_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "keyword", "rank", "domain", "url", "source",
            "query_timestamp_utc", "first_serp_position", "total_raw_results",
        ])
        for qr in csv_rows:
            keyword = qr.get("query", "")
            timestamp = qr.get("query_timestamp_utc", "")
            raw_results = qr.get("raw_results", [])
            total_raw = len(raw_results)

            domain_first_pos = {}
            domain_url = {}
            for rr in raw_results:
                url = rr.get("url", "")
                d = _extract_domain(url)
                if d and d not in domain_first_pos:
                    domain_first_pos[d] = rr.get("position", "")
                    domain_url[d] = url

            ranked_results = qr.get("ranked_results", [])
            if ranked_results:
                for rank, entry in enumerate(ranked_results, 1):
                    domain = entry["domain"]
                    url = entry.get("url") or domain_url.get(domain, "")
                    first_pos = domain_first_pos.get(domain, "")
                    writer.writerow([
                        keyword, rank, domain, url, "ai_search",
                        timestamp, first_pos, total_raw,
                    ])
            else:
                domains_list = qr.get("ranked_domains", [])
                for rank, domain in enumerate(domains_list, 1):
                    url = domain_url.get(domain, "")
                    first_pos = domain_first_pos.get(domain, "")
                    writer.writerow([
                        keyword, rank, domain, url, "ai_search",
                        timestamp, first_pos, total_raw,
                    ])
    print(f"  Saved CSV: {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"AI Search Complete (50_larger variant)")
    print(f"  SERP engine: {args.engine}")
    print(f"  SERP results: {NUM_SERP_RESULTS}")
    print(f"  LLM: {MODEL_ID}")
    print(f"  LLM top-N: {args.top}")
    print(f"  Method: {data['method']}")
    print(f"  Experiment start: {context['experiment_start_utc']}")
    print(f"  Location: {geo['city']}, {geo['country']} (IP: {context['network']['public_ip']})")
    print(f"  Keywords processed: {len(per_keyword_data)}/{len(keywords)}")
    print(f"  Failed: {len(errors)}")
    if errors:
        print(f"  Failed keywords: {[e['keyword'] for e in errors]}")
    print(f"  Results: {RESULTS_DIR}/{base_name}.json + .csv")


if __name__ == "__main__":
    main()
