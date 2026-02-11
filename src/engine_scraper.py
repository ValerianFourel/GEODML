import time
import random
import tldextract
from duckduckgo_search import DDGS
from googlesearch import search as google_search


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL."""
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def search_duckduckgo(query: str, num_results: int = 10) -> list[str]:
    """Search DuckDuckGo and return ordered list of root domains."""
    domains = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results * 2)
            for r in results:
                domain = _extract_domain(r.get("href", ""))
                if domain and domain not in domains:
                    domains.append(domain)
                if len(domains) >= num_results:
                    break
    except Exception as e:
        print(f"  [DDG] Error searching '{query}': {e}")

    # Rate limit
    time.sleep(random.uniform(2, 5))
    return domains[:num_results]


def search_google(query: str, num_results: int = 10) -> list[str]:
    """Search Google and return ordered list of root domains."""
    domains = []
    try:
        results = google_search(query, num_results=num_results * 2, sleep_interval=2)
        for url in results:
            domain = _extract_domain(url)
            if domain and domain not in domains:
                domains.append(domain)
            if len(domains) >= num_results:
                break
    except Exception as e:
        print(f"  [Google] Error searching '{query}': {e}")

    # Rate limit
    time.sleep(random.uniform(3, 5))
    return domains[:num_results]
