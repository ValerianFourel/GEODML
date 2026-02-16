#!/usr/bin/env python3
"""AI-powered search ranking experiment.

Pipeline: keyword → search engine (SERP) → LLM re-ranking (HF Inference API)

- The keyword is passed as a bare search term (no sentence wrapping)
- The LLM re-ranks the results — this re-ranking IS the experimental variable
- Full provenance is captured (timestamps, raw responses, geo context)

Usage:
  python run_ai_search.py                          # default: searxng
  python run_ai_search.py --engine duckduckgo
  python run_ai_search.py --engine google
  python run_ai_search.py --engine yahoo
  python run_ai_search.py --engine kagi
  python run_ai_search.py --engine searxng --keywords 5 --top 5
"""

import argparse
import tldextract
from datetime import datetime, timezone

from src.keywords import load_keywords
from src.engine_scraper import search, ENGINES
from src.llm_ranker import rank_domains_with_llm, MODEL_ID
from src.results_io import save_results, results_to_csv
from src.experiment_context import collect_experiment_context, utcnow_iso
from src.config import TOP_N


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
    parser = argparse.ArgumentParser(description="AI-powered search ranking experiment")
    parser.add_argument("--engine", type=str, default="searxng",
                        choices=ENGINES,
                        help=f"Search engine for SERP (default: searxng). Choices: {', '.join(ENGINES)}")
    parser.add_argument("--keywords", type=int, default=0, help="Limit to first N keywords (0 = all)")
    parser.add_argument("--top", type=int, default=TOP_N, help="Top N domains per keyword")
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
    print(f"Top N domains per keyword: {args.top}")
    print(f"LLM: {MODEL_ID}")
    print()

    per_keyword_data = []
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # Step 1: Raw search results from chosen engine
        serp_result = search(args.engine, keyword, num_results=20)
        raw_results = serp_result["raw_results"]

        if not raw_results:
            print(f"  No results, skipping")
            errors.append({"keyword": keyword, "error": serp_result.get("error")})
            continue

        print(f"  Got {len(raw_results)} results via {args.engine}")

        # Step 2: LLM re-ranks (the experimental variable)
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
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }

    # Build filename: {engine}_{llm_model}_{date}.{ext}
    llm_label = MODEL_ID.split("/")[-1]
    date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    base_name = f"{args.engine}_{llm_label}_{date_label}"

    # Save
    save_results(data, f"{base_name}.json")

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
    results_to_csv(csv_rows, "ai_search", f"{base_name}.csv")

    # Summary
    print(f"\n{'='*60}")
    print(f"AI Search Complete")
    print(f"  SERP engine: {args.engine}")
    print(f"  LLM: {MODEL_ID}")
    print(f"  Method: {data['method']}")
    print(f"  Experiment start: {context['experiment_start_utc']}")
    print(f"  Location: {geo['city']}, {geo['country']} (IP: {context['network']['public_ip']})")
    print(f"  Keywords processed: {len(per_keyword_data)}/{len(keywords)}")
    print(f"  Failed: {len(errors)}")
    if errors:
        print(f"  Failed keywords: {[e['keyword'] for e in errors]}")
    print(f"  Results: results/{base_name}.json + .csv")


if __name__ == "__main__":
    main()
