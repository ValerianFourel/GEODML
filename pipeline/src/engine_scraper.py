"""Search engine backends: duckduckgo, google, yahoo, kagi, searxng.

Each function returns a common dict format:
    query, query_timestamp_utc, response_timestamp_utc, search_backend,
    num_requested, raw_results, error

raw_results is a list of:
    {position, title, url, snippet, engines (optional), score (optional)}
"""

import time
import random
import requests
import tldextract
from src.experiment_context import utcnow_iso
from src.config import SEARXNG_URL, KAGI_TOKEN, BRAVE_API_KEY, GOOGLE_API_KEY, GOOGLE_CX, SERPAPI_KEY


ENGINES = ["searxng", "duckduckgo", "google", "google_api", "yahoo", "kagi", "brave", "serpapi"]


def _make_result(query: str, backend: str, num_results: int) -> dict:
    """Create an empty result dict with common fields."""
    return {
        "query": query,
        "query_timestamp_utc": utcnow_iso(),
        "response_timestamp_utc": None,
        "search_backend": backend,
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


# ── SearXNG ──────────────────────────────────────────────────────────────

def search_searxng(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "searxng", num_results)
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


# ── DuckDuckGo ───────────────────────────────────────────────────────────

def search_duckduckgo(query: str, num_results: int = 20) -> dict:
    from ddgs import DDGS

    result = _make_result(query, "duckduckgo", num_results)
    try:
        ddgs = DDGS()
        raw = ddgs.text(query, max_results=num_results)
        for pos, r in enumerate(raw, 1):
            result["raw_results"].append({
                "position": pos,
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [DDG] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


# ── Google ───────────────────────────────────────────────────────────────

def search_google(query: str, num_results: int = 20) -> dict:
    from googlesearch import search as google_search

    result = _make_result(query, "google", num_results)
    try:
        raw = google_search(query, num_results=num_results, sleep_interval=2)
        for pos, url in enumerate(raw, 1):
            result["raw_results"].append({
                "position": pos,
                "title": "",
                "url": url,
                "snippet": "",
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Google] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(3, 5))
    return result


# ── Yahoo ────────────────────────────────────────────────────────────────

def search_yahoo(query: str, num_results: int = 20) -> dict:
    from bs4 import BeautifulSoup

    result = _make_result(query, "yahoo", num_results)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    }
    try:
        resp = requests.get(
            "https://search.yahoo.com/search",
            params={"p": query, "n": num_results},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        pos = 0
        for link in soup.select("div.algo h3 a, div.dd a.ac-algo"):
            href = link.get("href", "")
            if not href or "yahoo.com" in href:
                continue
            pos += 1
            result["raw_results"].append({
                "position": pos,
                "title": link.get_text(strip=True),
                "url": href,
                "snippet": "",
            })
            if pos >= num_results:
                break
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Yahoo] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


# ── Kagi ─────────────────────────────────────────────────────────────────

def search_kagi(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "kagi", num_results)

    if not KAGI_TOKEN:
        result["error"] = "KAGI_TOKEN not set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result

    try:
        resp = requests.get(
            "https://kagi.com/api/v0/search",
            params={"q": query, "limit": num_results},
            headers={"Authorization": f"Bot {KAGI_TOKEN}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for pos, item in enumerate(data.get("data", [])[:num_results], 1):
            if item.get("t") != 0:  # t=0 is organic result
                continue
            result["raw_results"].append({
                "position": pos,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Kagi] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


# ── Google API (Custom Search JSON API) ──────────────────────────────────

def search_google_api(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "google_api", num_results)

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        result["error"] = "GOOGLE_API_KEY and GOOGLE_CX must be set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result

    try:
        # API returns max 10 per request, so paginate if needed
        collected = []
        for start in range(1, num_results + 1, 10):
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX,
                    "q": query,
                    "start": start,
                    "num": min(10, num_results - len(collected)),
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", []):
                collected.append(item)
            if len(collected) >= num_results:
                break
            time.sleep(0.5)

        for pos, item in enumerate(collected[:num_results], 1):
            result["raw_results"].append({
                "position": pos,
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Google API] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


# ── Brave ────────────────────────────────────────────────────────────────

def search_brave(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "brave", num_results)

    if not BRAVE_API_KEY:
        result["error"] = "BRAVE_API_KEY not set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for pos, item in enumerate(data.get("web", {}).get("results", [])[:num_results], 1):
            result["raw_results"].append({
                "position": pos,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Brave] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


# ── SerpAPI (Google results via serpapi.com) ──────────────────────────────

def search_serpapi(query: str, num_results: int = 20) -> dict:
    from serpapi import GoogleSearch

    result = _make_result(query, "serpapi_google", num_results)

    if not SERPAPI_KEY:
        result["error"] = "SERPAPI_KEY not set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result

    try:
        collected = []
        seen_urls = set()
        start = 0
        # Paginate until we have enough organic results (each page ~8-10)
        while len(collected) < num_results:
            params = {
                "q": query,
                "num": 10,
                "start": start,
                "api_key": SERPAPI_KEY,
                "engine": "google",
            }
            data = GoogleSearch(params).get_dict()
            organic = data.get("organic_results", [])
            if not organic:
                break
            for item in organic:
                url = item.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    collected.append(item)
            start += len(organic)
            if start >= 30:  # cap at 3 pages to save API credits
                break
            time.sleep(0.5)

        for pos, item in enumerate(collected[:num_results], 1):
            result["raw_results"].append({
                "position": pos,
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [SerpAPI] Error: {e}")

    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


# ── Dispatcher ───────────────────────────────────────────────────────────

_ENGINE_MAP = {
    "searxng": search_searxng,
    "duckduckgo": search_duckduckgo,
    "google": search_google,
    "google_api": search_google_api,
    "yahoo": search_yahoo,
    "kagi": search_kagi,
    "brave": search_brave,
    "serpapi": search_serpapi,
}


def search(engine: str, query: str, num_results: int = 20) -> dict:
    """Search using the specified engine. Returns common result dict."""
    fn = _ENGINE_MAP.get(engine)
    if fn is None:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {ENGINES}")
    return fn(query, num_results)
