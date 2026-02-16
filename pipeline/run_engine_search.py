#!/usr/bin/env python3
"""Pure search engine ranking (no LLM re-ranking).

Pipeline: keyword → search engine (SERP) → domain extraction → results/

This is the baseline: raw SERP domain rankings with no LLM involved.
Compare its output against run_ai_search.py to isolate the LLM effect.

Usage:
  python run_engine_search.py                          # default: searxng
  python run_engine_search.py --engine duckduckgo
  python run_engine_search.py --engine google
  python run_engine_search.py --engine google_api
  python run_engine_search.py --engine yahoo
  python run_engine_search.py --engine kagi
  python run_engine_search.py --engine brave
  python run_engine_search.py --engine searxng --keywords 5 --top 10
"""

import argparse
import tldextract
from datetime import datetime, timezone

from src.keywords import load_keywords
from src.engine_scraper import search, ENGINES
from src.results_io import save_results, results_to_csv
from src.experiment_context import collect_experiment_context, utcnow_iso
from src.config import TOP_N


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def extract_domains(raw_results: list[dict], top_n: int) -> tuple[list[str], list[dict]]:
    """Extract unique domains from raw SERP results (first occurrence wins).

    Returns:
        ranked_domains: list of domain strings, up to top_n
        ranked_results: list of {domain, url} dicts
    """
    domains = []
    domain_url = {}
    for r in raw_results:
        url = r.get("url", "")
        d = _extract_domain(url)
        if d and d not in domain_url:
            domains.append(d)
            domain_url[d] = url
        if len(domains) >= top_n:
            break
    ranked_results = [{"domain": d, "url": domain_url[d]} for d in domains]
    return domains, ranked_results


def main():
    parser = argparse.ArgumentParser(description="Pure search engine ranking (no LLM)")
    parser.add_argument("--engine", type=str, default="searxng",
                        choices=ENGINES,
                        help=f"Search engine (default: searxng). Choices: {', '.join(ENGINES)}")
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
    print(f"Mode: pure SERP (no LLM re-ranking)")
    print()

    per_keyword_data = []
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        serp_result = search(args.engine, keyword, num_results=20)
        raw_results = serp_result["raw_results"]

        if not raw_results:
            print(f"  No results, skipping")
            errors.append({"keyword": keyword, "error": serp_result.get("error")})
            continue

        print(f"  Got {len(raw_results)} results via {args.engine}")

        # Extract domains directly from SERP order (no LLM)
        domains, ranked_results = extract_domains(raw_results, args.top)
        print(f"  Domains: {domains[:5]}{'...' if len(domains) > 5 else ''}")

        per_keyword_data.append({
            "query": keyword,
            "query_timestamp_utc": serp_result["query_timestamp_utc"],
            "serp": serp_result,
            "ranked_domains": domains,
            "ranked_results": ranked_results,
        })

    # Build output
    data = {
        "experiment_context": context,
        "experiment_end_utc": utcnow_iso(),
        "source": "engine_search",
        "serp_engine": args.engine,
        "method": f"{args.engine} (pure SERP, no LLM)",
        "chat_model": "none",
        "top_n": args.top,
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }

    # Build filename: engine_nollm_{date}.{ext}
    date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    base_name = f"{args.engine}_nollm_{date_label}"

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
    results_to_csv(csv_rows, args.engine, f"{base_name}.csv")

    # Summary
    print(f"\n{'='*60}")
    print(f"Engine Search Complete (no LLM)")
    print(f"  SERP engine: {args.engine}")
    print(f"  Mode: pure SERP (no LLM re-ranking)")
    print(f"  Experiment start: {context['experiment_start_utc']}")
    print(f"  Location: {geo['city']}, {geo['country']} (IP: {context['network']['public_ip']})")
    print(f"  Keywords processed: {len(per_keyword_data)}/{len(keywords)}")
    print(f"  Failed: {len(errors)}")
    if errors:
        print(f"  Failed keywords: {[e['keyword'] for e in errors]}")
    print(f"  Results: results/{base_name}.json + .csv")


if __name__ == "__main__":
    main()
