import time
import random
import requests
from src.config import SEARXNG_URL
from src.experiment_context import utcnow_iso


def search_searxng(query: str, num_results: int = 20) -> dict:
    """Query a SearXNG instance and return timestamped raw results.

    Returns dict with:
        query, query_timestamp_utc, searxng_instance, num_requested,
        raw_results (list of {position, title, url, snippet, engines}),
        error (str or None)
    """
    query_time = utcnow_iso()

    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    headers = {
        "User-Agent": "GEODML-Research/1.0",
        "Accept": "application/json",
    }

    result = {
        "query": query,
        "query_timestamp_utc": query_time,
        "searxng_instance": SEARXNG_URL,
        "request_params": params,
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }

    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        result["error"] = str(e)
        print(f"  [SearXNG] Error querying '{query}': {e}")
        return result

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

    # Rate limit: sleep 2-4s between calls
    time.sleep(random.uniform(2, 4))

    return result
