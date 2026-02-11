import time
import random
import tldextract
from duckduckgo_search import DDGS
from googlesearch import search as google_search
from src.experiment_context import utcnow_iso


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL."""
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def search_duckduckgo(query: str, num_results: int = 10) -> dict:
    """Search DuckDuckGo and return timestamped raw results + ranked domains.

    Returns dict with:
        query, query_timestamp_utc, source, num_requested,
        raw_results (list of {position, url, domain, title, snippet}),
        ranked_domains (list of unique domains in SERP order),
        error (str or None)
    """
    query_time = utcnow_iso()

    result = {
        "query": query,
        "query_timestamp_utc": query_time,
        "source": "duckduckgo",
        "num_requested": num_results,
        "raw_results": [],
        "ranked_domains": [],
        "error": None,
    }

    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=num_results * 2)
            for position, r in enumerate(raw, 1):
                url = r.get("href", "")
                domain = _extract_domain(url)
                result["raw_results"].append({
                    "position": position,
                    "url": url,
                    "domain": domain,
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [DDG] Error searching '{query}': {e}")

    result["response_timestamp_utc"] = utcnow_iso()

    # Extract unique domains preserving SERP order
    seen = set()
    for r in result["raw_results"]:
        d = r["domain"]
        if d and d not in seen:
            seen.add(d)
            result["ranked_domains"].append(d)
        if len(result["ranked_domains"]) >= num_results:
            break

    # Rate limit
    time.sleep(random.uniform(2, 5))
    return result


def search_google(query: str, num_results: int = 10) -> dict:
    """Search Google and return timestamped raw results + ranked domains.

    Returns dict with:
        query, query_timestamp_utc, source, num_requested,
        raw_results (list of {position, url, domain}),
        ranked_domains (list of unique domains in SERP order),
        error (str or None)
    """
    query_time = utcnow_iso()

    result = {
        "query": query,
        "query_timestamp_utc": query_time,
        "source": "google",
        "num_requested": num_results,
        "raw_results": [],
        "ranked_domains": [],
        "error": None,
    }

    try:
        raw = google_search(query, num_results=num_results * 2, sleep_interval=2)
        for position, url in enumerate(raw, 1):
            domain = _extract_domain(url)
            result["raw_results"].append({
                "position": position,
                "url": url,
                "domain": domain,
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Google] Error searching '{query}': {e}")

    result["response_timestamp_utc"] = utcnow_iso()

    # Extract unique domains preserving SERP order
    seen = set()
    for r in result["raw_results"]:
        d = r["domain"]
        if d and d not in seen:
            seen.add(d)
            result["ranked_domains"].append(d)
        if len(result["ranked_domains"]) >= num_results:
            break

    # Rate limit
    time.sleep(random.uniform(3, 5))
    return result
