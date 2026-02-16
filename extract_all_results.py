#!/usr/bin/env python3
"""Extract domain, url, keyword, engine, source from all results JSON files into a single CSV."""

import csv
import json
import sys
import tldextract
from pathlib import Path


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _build_domain_url_map(raw_results: list[dict]) -> dict:
    """Map domain -> best (first) URL from raw SERP results."""
    m = {}
    for r in raw_results:
        url = r.get("url", "")
        d = _extract_domain(url)
        if d and d not in m:
            m[d] = url
    return m


def extract_rows(json_path: Path) -> list[dict]:
    """Extract ranked results from a single JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    source = data.get("source", "")          # "ai_search" or "engine_search"
    engine = data.get("serp_engine", "")      # "serpapi", "duckduckgo", "searxng", ...
    model = data.get("chat_model", "none")    # LLM model or "none"
    file_str = str(json_path)

    # Skip files with no engine or different structure (e.g. ranking_comparison)
    if not source:
        return []

    rows = []
    for kw_data in data.get("per_keyword_results", []):
        keyword = kw_data.get("query", "")

        # Build domain->url map from raw SERP for URL recovery
        raw_results = kw_data.get("serp", {}).get("raw_results", [])
        domain_url_map = _build_domain_url_map(raw_results)

        # Get ranked_results (list of {domain, url}) or ranked_domains (list of str)
        ranked = []
        domains = []

        if source == "ai_search" and "llm" in kw_data:
            ranked = kw_data["llm"].get("ranked_results", [])
            domains = kw_data["llm"].get("ranked_domains", [])
        else:
            ranked = kw_data.get("ranked_results", [])
            domains = kw_data.get("ranked_domains", [])

        # Also check top-level ranked_domains (older format)
        if not ranked and not domains:
            domains = kw_data.get("ranked_domains", [])

        if ranked:
            for rank, item in enumerate(ranked, 1):
                domain = item.get("domain", "")
                url = item.get("url", "") or domain_url_map.get(domain, "")
                rows.append({
                    "keyword": keyword,
                    "rank": rank,
                    "domain": domain,
                    "url": url,
                    "source": source,
                    "engine": engine,
                    "model": model,
                    "source_file": file_str,
                })
        elif domains:
            for rank, domain in enumerate(domains, 1):
                url = domain_url_map.get(domain, "")
                rows.append({
                    "keyword": keyword,
                    "rank": rank,
                    "domain": domain,
                    "url": url,
                    "source": source,
                    "engine": engine,
                    "model": model,
                    "source_file": file_str,
                })

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract results from JSON files into CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only include JSON files whose name contains this string (e.g. 'searxng')")
    args = parser.parse_args()

    results_dir = Path("results")
    if not results_dir.is_dir():
        print("Error: results/ directory not found", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(results_dir.glob("*.json"))
    if args.filter:
        json_files = [f for f in json_files if args.filter in f.name]

    if not json_files:
        print("No matching JSON files found in results/", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for jf in json_files:
        try:
            rows = extract_rows(jf)
            all_rows.extend(rows)
            print(f"  {jf.name}: {len(rows)} rows")
        except Exception as e:
            print(f"  {jf.name}: SKIPPED ({e})", file=sys.stderr)

    suffix = f"_{args.filter}" if args.filter else ""
    out_path = results_dir / f"all_results{suffix}.csv"
    fieldnames = ["keyword", "rank", "domain", "url", "source", "engine", "model", "source_file"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
