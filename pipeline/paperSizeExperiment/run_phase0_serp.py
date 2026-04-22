#!/usr/bin/env python3
"""Phase 0: Collect and save raw SERP JSON responses.

Fetches raw search results from DuckDuckGo and SearXNG for all 1011 keywords
at both pool sizes (top 20 and top 50), and saves them as JSON files in the
consolidated results folder.

Output files:
  consolidated_results/phase0_top20_ddg.json
  consolidated_results/phase0_top50_ddg.json
  consolidated_results/phase0_top20_searxng.json
  consolidated_results/phase0_top50_searxng.json

Each file contains:
  {
    "metadata": { engine, pool_size, n_keywords, timestamp, ... },
    "serp_results": {
      "keyword1": { query, timestamp, raw_results: [{position, title, url, snippet, ...}], ... },
      "keyword2": { ... },
      ...
    }
  }

Usage:
  python paperSizeExperiment/run_phase0_serp.py
  python paperSizeExperiment/run_phase0_serp.py --keywords 5      # test with 5
  python paperSizeExperiment/run_phase0_serp.py --engine duckduckgo --pool-size 20
  python paperSizeExperiment/run_phase0_serp.py --resume           # skip already-fetched keywords
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env.local")

import requests

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

KEYWORDS_FILE = SCRIPT_DIR / "keywords.txt"
OUTPUT_DIR = SCRIPT_DIR / "consolidated_results"


# ── Search backends ──────────────────────────────────────────────────────────

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ddg_fetch(query: str, num_results: int, timeout: int = 30) -> list:
    """Fetch DDG results with a hard timeout to prevent hangs."""
    import concurrent.futures
    from ddgs import DDGS
    def _do():
        return list(DDGS().text(query, max_results=num_results))
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        fut = ex.submit(_do)
        return fut.result(timeout=timeout)


def search_duckduckgo(query: str, num_results: int) -> dict:
    result = {
        "query": query,
        "query_timestamp_utc": utcnow_iso(),
        "response_timestamp_utc": None,
        "search_backend": "duckduckgo",
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }
    max_retries = 4
    for attempt in range(max_retries):
        try:
            raw = _ddg_fetch(query, num_results, timeout=30)
            for pos, r in enumerate(raw, 1):
                result["raw_results"].append({
                    "position": pos,
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(code in err_str for code in ["429", "403", "Too Many Requests", "Forbidden", "TimeoutError"])
            if is_rate_limit and attempt < max_retries - 1:
                backoff = (2 ** attempt) * 10 + random.uniform(0, 5)
                print(f"  [DDG] Rate limited (attempt {attempt+1}/{max_retries}), backing off {backoff:.0f}s...")
                time.sleep(backoff)
                result["raw_results"] = []
                continue
            result["error"] = err_str
            print(f"  [DDG] Error: {e}")
            break
    result["response_timestamp_utc"] = utcnow_iso()
    delay = random.uniform(8, 12) if num_results > 20 else random.uniform(2, 4)
    time.sleep(delay)
    return result


def search_searxng(query: str, num_results: int) -> dict:
    result = {
        "query": query,
        "query_timestamp_utc": utcnow_iso(),
        "response_timestamp_utc": None,
        "search_backend": "searxng",
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            headers=headers, timeout=30,
        )
        if resp.status_code in (403, 429):
            result["error"] = f"SearXNG returned {resp.status_code}"
            result["response_timestamp_utc"] = utcnow_iso()
            return result
        resp.raise_for_status()
        data = resp.json()
        for pos, item in enumerate(data.get("results", [])[:num_results], 1):
            result["raw_results"].append({
                "position": pos,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "engines": item.get("engines", []),
                "score": item.get("score"),
            })
    except requests.RequestException as e:
        result["error"] = str(e)
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


SEARCH_FNS = {
    "duckduckgo": search_duckduckgo,
    "searxng": search_searxng,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def load_keywords(limit: int = 0) -> list[str]:
    with open(KEYWORDS_FILE) as f:
        keywords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if limit > 0:
        keywords = keywords[:limit]
    return keywords


def output_filename(engine: str, pool_size: int) -> str:
    engine_short = "ddg" if engine == "duckduckgo" else engine
    return f"phase0_top{pool_size}_{engine_short}.json"


def run_phase0(engine: str, pool_size: int, keywords: list[str], resume: bool = False):
    search_fn = SEARCH_FNS[engine]
    engine_short = "ddg" if engine == "duckduckgo" else engine
    out_file = OUTPUT_DIR / output_filename(engine, pool_size)

    # Load existing data for resume
    existing = {}
    if resume and out_file.exists():
        with open(out_file) as f:
            data = json.load(f)
            existing = data.get("serp_results", {})
        print(f"  Resuming: {len(existing)} keywords already fetched")

    serp_results = dict(existing)
    total = len(keywords)
    skipped = 0
    errors = 0

    print(f"\n{'='*60}")
    print(f"Phase 0: {engine_short} top-{pool_size} | {total} keywords")
    print(f"Output: {out_file}")
    print(f"{'='*60}")

    for i, kw in enumerate(keywords, 1):
        if kw in serp_results:
            skipped += 1
            continue

        print(f"  [{i}/{total}] {kw}...", end=" ", flush=True)
        result = search_fn(kw, pool_size)
        n_results = len(result["raw_results"])
        serp_results[kw] = result

        if result["error"]:
            errors += 1
            print(f"ERROR: {result['error']}")
        else:
            print(f"{n_results} results")

        # Save checkpoint every 50 keywords
        if i % 50 == 0:
            _save(out_file, engine, pool_size, serp_results, total)
            print(f"  [checkpoint saved: {len(serp_results)} keywords]")

    # Final save
    _save(out_file, engine, pool_size, serp_results, total)

    fetched = len(serp_results) - len(existing)
    print(f"\nDone: {fetched} fetched, {skipped} skipped (resume), {errors} errors")
    print(f"Total: {len(serp_results)} keywords saved to {out_file}")
    return out_file


def _save(out_file: str, engine: str, pool_size: int, serp_results: dict, total_kw: int):
    engine_short = "ddg" if engine == "duckduckgo" else engine
    data = {
        "metadata": {
            "engine": engine,
            "engine_short": engine_short,
            "pool_size": pool_size,
            "n_keywords_target": total_kw,
            "n_keywords_fetched": len(serp_results),
            "n_keywords_with_results": sum(1 for r in serp_results.values() if r["raw_results"]),
            "n_keywords_with_errors": sum(1 for r in serp_results.values() if r["error"]),
            "total_serp_entries": sum(len(r["raw_results"]) for r in serp_results.values()),
            "created_utc": utcnow_iso(),
            "description": f"Raw SERP JSON responses from {engine} requesting top {pool_size} results per keyword",
        },
        "serp_results": serp_results,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Collect raw SERP JSON responses")
    parser.add_argument("--keywords", type=int, default=0,
                        help="Limit number of keywords (0 = all)")
    parser.add_argument("--engine", choices=["duckduckgo", "searxng", "all"], default="all",
                        help="Which search engine(s) to query")
    parser.add_argument("--pool-size", choices=["20", "50", "all"], default="all",
                        help="Which pool size(s) to fetch")
    parser.add_argument("--resume", action="store_true",
                        help="Skip keywords already fetched in existing output")
    args = parser.parse_args()

    keywords = load_keywords(args.keywords)
    print(f"Loaded {len(keywords)} keywords from {KEYWORDS_FILE}")

    engines = ["duckduckgo", "searxng"] if args.engine == "all" else [args.engine]
    pool_sizes = [20, 50] if args.pool_size == "all" else [int(args.pool_size)]

    results_files = []
    for engine in engines:
        for pool_size in pool_sizes:
            out = run_phase0(engine, pool_size, keywords, resume=args.resume)
            results_files.append(out)

    print(f"\n{'='*60}")
    print("All Phase 0 collections complete:")
    for f in results_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
