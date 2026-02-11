import time
import random
import requests
from src.config import SEARXNG_URL
from src.experiment_context import utcnow_iso


def _try_searxng(query: str, num_results: int) -> dict | None:
    """Try the local SearXNG instance. Returns parsed JSON or None."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            headers=headers,
            timeout=30,
        )
        if resp.status_code in (403, 429):
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _ddg_fallback(query: str, num_results: int) -> list[dict]:
    """DuckDuckGo fallback when SearXNG is unavailable."""
    from ddgs import DDGS

    results = []
    try:
        ddgs = DDGS()
        raw = ddgs.text(query, max_results=num_results)
        for position, r in enumerate(raw, 1):
            results.append({
                "position": position,
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "engines": ["duckduckgo"],
                "score": None,
            })
    except Exception as e:
        print(f"  [DDG fallback] Error: {e}")
    return results


def search_searxng(query: str, num_results: int = 20) -> dict:
    """Query SearXNG (local Apptainer instance) with DuckDuckGo fallback.

    Returns dict with:
        query, query_timestamp_utc, searxng_instance, search_backend,
        num_requested, raw_results, error
    """
    query_time = utcnow_iso()

    result = {
        "query": query,
        "query_timestamp_utc": query_time,
        "searxng_instance": None,
        "search_backend": None,
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }

    # Try local SearXNG
    data = _try_searxng(query, num_results)
    if data is not None:
        result["searxng_instance"] = SEARXNG_URL
        result["search_backend"] = "searxng"
        result["response_timestamp_utc"] = utcnow_iso()

        for position, item in enumerate(data.get("results", [])[:num_results], 1):
            result["raw_results"].append({
                "position": position,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "engines": item.get("engines", []),
                "score": item.get("score"),
            })

        if result["raw_results"]:
            time.sleep(random.uniform(2, 4))
            return result

    # Fallback to DuckDuckGo
    print(f"  [SearXNG] Unavailable, using DuckDuckGo fallback")
    result["search_backend"] = "duckduckgo_fallback"
    result["raw_results"] = _ddg_fallback(query, num_results)
    result["response_timestamp_utc"] = utcnow_iso()

    if not result["raw_results"]:
        result["error"] = "SearXNG unavailable and DuckDuckGo fallback returned no results"

    time.sleep(random.uniform(2, 4))
    return result
