import time
import random
import requests
from src.config import SEARXNG_URL


def search_searxng(query: str, num_results: int = 20) -> list[dict]:
    """Query a SearXNG instance and return raw search results.

    Returns list of dicts with keys: title, url, snippet.
    """
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    headers = {
        "User-Agent": "GEODML-Research/1.0",
        "Accept": "application/json",
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
        print(f"  [SearXNG] Error querying '{query}': {e}")
        return []

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        })

    # Rate limit: sleep 2-4s between calls
    time.sleep(random.uniform(2, 4))

    return results
