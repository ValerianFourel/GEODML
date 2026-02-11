#!/usr/bin/env python3
"""AI-powered search ranking experiment.

Two modes:
  --mode perplexica  (default)  Perplexica framework (SearXNG + Qwen3-32B re-ranking)
  --mode standalone             SearXNG + HF Inference API LLM re-ranking

In both modes:
  - The keyword is passed as a bare search term (no sentence wrapping)
  - The LLM re-ranks the results — this re-ranking IS the experimental variable
  - Full provenance is captured (timestamps, raw responses, geo context)
"""

import argparse

from src.keywords import load_keywords
from src.results_io import save_results, results_to_csv
from src.experiment_context import collect_experiment_context, utcnow_iso
from src.config import TOP_N


def run_perplexica(keywords, context, top_n):
    """Perplexica mode: keyword -> Perplexica (SearXNG + LLM re-ranking) -> domains."""
    from src.perplexica_client import (
        discover_providers, _find_provider_and_model,
        _find_embedding_model, search_perplexica,
    )

    # Discover available models
    print("Discovering Perplexica providers...")
    providers = discover_providers()
    if not providers.get("providers"):
        print("ERROR: No providers found. Is Perplexica running?")
        return None

    # Find Qwen3 chat model
    chat_pid, chat_key = _find_provider_and_model(providers, "qwen3")
    if not chat_pid:
        # Fall back to any available chat model
        for p in providers["providers"]:
            if p.get("chatModels"):
                chat_pid = p["id"]
                chat_key = p["chatModels"][0]["key"]
                break
    if not chat_pid:
        print("ERROR: No chat model found in Perplexica providers.")
        return None
    print(f"  Chat model: {chat_key} (provider: {chat_pid})")

    # Find embedding model
    embed_pid, embed_key = _find_embedding_model(providers)
    if not embed_pid:
        print("ERROR: No embedding model found in Perplexica providers.")
        return None
    print(f"  Embedding model: {embed_key} (provider: {embed_pid})")
    print()

    per_keyword_data = []
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        result = search_perplexica(
            query=keyword,
            providers=providers,
            chat_provider_id=chat_pid,
            chat_model_key=chat_key,
            embed_provider_id=embed_pid,
            embed_model_key=embed_key,
        )

        if result["error"]:
            errors.append({"keyword": keyword, "error": result["error"]})
            print(f"  Error: {result['error']}")
            continue

        domains = result["ranked_domains"]
        print(f"  LLM re-ranked {len(domains)} domains: {domains[:3]}...")

        per_keyword_data.append(result)

    return {
        "experiment_context": context,
        "experiment_end_utc": utcnow_iso(),
        "source": "ai_search",
        "method": f"Perplexica (SearXNG + {chat_key} LLM re-ranking)",
        "mode": "perplexica",
        "chat_model": chat_key,
        "embedding_model": embed_key,
        "top_n": top_n,
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }


def run_standalone(keywords, context, top_n):
    """Standalone mode: keyword -> SearXNG -> LLM re-ranking -> domains."""
    from src.searxng_client import search_searxng
    from src.llm_ranker import rank_domains_with_llm, MODEL_ID

    per_keyword_data = []
    errors = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i}/{len(keywords)}] Searching: {keyword}")

        # Step 1: Raw SearXNG results
        searxng_result = search_searxng(keyword, num_results=20)
        raw_results = searxng_result["raw_results"]

        if not raw_results:
            print(f"  No SearXNG results, skipping")
            errors.append({"keyword": keyword, "error": searxng_result.get("error")})
            continue
        print(f"  Got {len(raw_results)} SearXNG results")

        # Step 2: LLM re-ranks (this is the experimental variable)
        llm_result = rank_domains_with_llm(keyword, raw_results, top_n=top_n)
        domains = llm_result["ranked_domains"]
        print(f"  LLM re-ranked {len(domains)} domains: {domains[:3]}...")
        if llm_result["used_fallback"]:
            print(f"  (used fallback due to LLM error)")

        per_keyword_data.append({
            "query": keyword,
            "query_timestamp_utc": searxng_result["query_timestamp_utc"],
            "searxng": searxng_result,
            "llm": llm_result,
            "ranked_domains": domains,
        })

    return {
        "experiment_context": context,
        "experiment_end_utc": utcnow_iso(),
        "source": "ai_search",
        "method": f"SearXNG + {MODEL_ID} LLM re-ranking",
        "mode": "standalone",
        "chat_model": MODEL_ID,
        "top_n": top_n,
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered search ranking experiment"
    )
    parser.add_argument(
        "--mode", choices=["perplexica", "standalone"], default="perplexica",
        help="perplexica = Perplexica framework (default), "
             "standalone = SearXNG + HF Inference API",
    )
    args = parser.parse_args()

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
    print(f"Top N domains per keyword: {TOP_N}")
    print(f"Mode: {args.mode}\n")

    if args.mode == "perplexica":
        data = run_perplexica(keywords, context, TOP_N)
    else:
        data = run_standalone(keywords, context, TOP_N)

    if data is None:
        print("Experiment aborted due to setup errors.")
        return

    # Save full provenance JSON
    save_results(data, "ai_search_rankings.json")

    # Save flat CSV
    csv_rows = []
    for kd in data["per_keyword_results"]:
        query = kd.get("query") or kd.get("keyword", "")
        timestamp = kd.get("query_timestamp_utc", "")
        # Raw results may be nested differently per mode
        raw = kd.get("raw_results", [])
        if not raw and "searxng" in kd:
            raw = kd["searxng"].get("raw_results", [])
        if not raw:
            raw = kd.get("sources", [])
        csv_rows.append({
            "query": query,
            "query_timestamp_utc": timestamp,
            "raw_results": raw,
            "ranked_domains": kd.get("ranked_domains", []),
        })
    results_to_csv(csv_rows, "ai_search", "ai_search_rankings.csv")

    # Summary
    errors = data.get("failed_keywords", [])
    successful = data.get("successful_keywords", 0)
    print(f"\n{'='*60}")
    print(f"AI Search Complete ({args.mode} mode)")
    print(f"  Method: {data['method']}")
    print(f"  Experiment start: {context['experiment_start_utc']}")
    print(f"  Location: {geo['city']}, {geo['country']} (IP: {context['network']['public_ip']})")
    print(f"  Keywords processed: {successful}/{len(keywords)}")
    print(f"  Failed: {len(errors)}")
    if errors:
        print(f"  Failed keywords: {[e['keyword'] for e in errors]}")
    print(f"  Results: results/ai_search_rankings.json + .csv")


if __name__ == "__main__":
    main()
