import time
import random
import requests
from src.config import SEARXNG_FALLBACK_URLS
from src.experiment_context import utcnow_iso

# Track which instances are known-bad to avoid retrying them every call
_blocked_instances: set[str] = set()


def _try_searxng_instance(instance_url: str, query: str, num_results: int) -> dict | None:
    """Try a single SearXNG instance. Returns parsed JSON or None on failure."""
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(
            f"{instance_url}/search",
            params=params,
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 403:
            _blocked_instances.add(instance_url)
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _ddg_fallback(query: str, num_results: int) -> list[dict]:
    """Use duckduckgo_search library as a direct fallback when all SearXNG instances fail."""
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
    """Query SearXNG with automatic instance fallback + DuckDuckGo fallback.

    Tries each SearXNG instance in SEARXNG_FALLBACK_URLS. If all fail (403, timeout, etc.),
    falls back to the duckduckgo_search library as the raw search source.

    Returns dict with:
        query, query_timestamp_utc, searxng_instance, num_requested,
        raw_results (list of {position, title, url, snippet, engines}),
        error (str or None)
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

    # Try SearXNG instances (skip known-blocked ones)
    instances_to_try = [u for u in SEARXNG_FALLBACK_URLS if u not in _blocked_instances]

    for instance_url in instances_to_try:
        data = _try_searxng_instance(instance_url, query, num_results)
        if data is not None:
            result["searxng_instance"] = instance_url
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
                print(f"  [SearXNG] Got {len(result['raw_results'])} results from {instance_url}")
                time.sleep(random.uniform(2, 4))
                return result

    # All SearXNG instances failed — fall back to DuckDuckGo library
    if _blocked_instances:
        print(f"  [SearXNG] All instances blocked/failed, using DuckDuckGo fallback")
    else:
        print(f"  [SearXNG] No instances available, using DuckDuckGo fallback")

    result["search_backend"] = "duckduckgo_fallback"
    result["raw_results"] = _ddg_fallback(query, num_results)
    result["response_timestamp_utc"] = utcnow_iso()

    if not result["raw_results"]:
        result["error"] = "All SearXNG instances failed and DuckDuckGo fallback returned no results"

    time.sleep(random.uniform(2, 4))
    return result
