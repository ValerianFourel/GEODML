#!/usr/bin/env python3
"""Gather data: search engine SERP + LLM re-ranking + page feature extraction.

Self-contained script — imports nothing from ../src/.

Pipeline: keywords → search engine (--serp-results) → LLM re-rank (--llm-top-n)
          → fetch HTML → extract code-based features → optional phases (LLM, PageRank, WHOIS)
          → save experiment.json + rankings.csv + features.csv

Usage:
  python pipeline/gather_data.py --keywords 2 --serp-results 5 --llm-top-n 3
  python pipeline/gather_data.py --all-features --output-dir output/full/
  python pipeline/gather_data.py --engine duckduckgo --serp-results 50 --llm-top-n 20
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import socket
import sys
import time
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import textstat
import tldextract
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env.local")

HF_TOKEN = os.getenv("HF_TOKEN", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
KAGI_TOKEN = os.getenv("KAGI_TOKEN", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
OPENPAGERANK_KEY = os.getenv("OPENPAGERANK_KEY", "")

ENGINES = ["searxng", "duckduckgo", "google", "google_api", "yahoo", "kagi", "brave", "serpapi"]
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
FETCH_TIMEOUT = 30
MAX_HTML_SIZE = 5 * 1024 * 1024  # 5 MB

# ── Utilities ─────────────────────────────────────────────────────────────────

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


# ── Experiment Context ────────────────────────────────────────────────────────

def _get_public_ip() -> str:
    for url in ["https://api.ipify.org", "https://ifconfig.me/ip"]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text.strip()
        except Exception:
            continue
    return "unknown"


def _get_geo_from_ip(ip: str) -> dict:
    if ip == "unknown":
        return {"city": "unknown", "region": "unknown", "country": "unknown",
                "lat": None, "lon": None, "isp": "unknown", "query_ip": ip}
    try:
        resp = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,lat,lon,isp,query",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            return {
                "city": data.get("city", "unknown"),
                "region": data.get("regionName", "unknown"),
                "country": data.get("country", "unknown"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "isp": data.get("isp", "unknown"),
                "query_ip": data.get("query", ip),
            }
    except Exception:
        pass
    return {"city": "unknown", "region": "unknown", "country": "unknown",
            "lat": None, "lon": None, "isp": "unknown", "query_ip": ip}


def _get_library_versions() -> dict:
    versions = {}
    for pkg in ["requests", "huggingface_hub", "tldextract", "dotenv"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def collect_experiment_context() -> dict:
    ip = _get_public_ip()
    geo = _get_geo_from_ip(ip)
    return {
        "experiment_start_utc": utcnow_iso(),
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "network": {"public_ip": ip, "geolocation": geo},
        "library_versions": _get_library_versions(),
    }


# ── Search Engine Backends ────────────────────────────────────────────────────

def _make_result(query: str, backend: str, num_results: int) -> dict:
    return {
        "query": query,
        "query_timestamp_utc": utcnow_iso(),
        "response_timestamp_utc": None,
        "search_backend": backend,
        "num_requested": num_results,
        "raw_results": [],
        "error": None,
    }


def search_searxng(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "searxng", num_results)
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
                "position": pos, "title": item.get("title", ""),
                "url": item.get("url", ""), "snippet": item.get("content", ""),
                "engines": item.get("engines", []), "score": item.get("score"),
            })
    except requests.RequestException as e:
        result["error"] = str(e)
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


def search_duckduckgo(query: str, num_results: int = 20) -> dict:
    from ddgs import DDGS
    result = _make_result(query, "duckduckgo", num_results)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            ddgs = DDGS()
            raw = ddgs.text(query, max_results=num_results)
            for pos, r in enumerate(raw, 1):
                result["raw_results"].append({
                    "position": pos, "title": r.get("title", ""),
                    "url": r.get("href", ""), "snippet": r.get("body", ""),
                })
            break  # success
        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(code in err_str for code in ["429", "403", "Too Many Requests", "Forbidden"])
            if is_rate_limit and attempt < max_retries - 1:
                backoff = (2 ** attempt) * 10 + random.uniform(0, 5)  # 10s, 25s, 45s, ...
                print(f"  [DDG] Rate limited (attempt {attempt+1}/{max_retries}), backing off {backoff:.0f}s...")
                time.sleep(backoff)
                result["raw_results"] = []  # reset for retry
                continue
            result["error"] = err_str
            print(f"  [DDG] Error: {e}")
            break
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


def search_google(query: str, num_results: int = 20) -> dict:
    from googlesearch import search as google_search
    result = _make_result(query, "google", num_results)
    try:
        raw = google_search(query, num_results=num_results, sleep_interval=2)
        for pos, url in enumerate(raw, 1):
            result["raw_results"].append({
                "position": pos, "title": "", "url": url, "snippet": "",
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Google] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(3, 5))
    return result


def search_google_api(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "google_api", num_results)
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        result["error"] = "GOOGLE_API_KEY and GOOGLE_CX must be set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result
    try:
        collected = []
        for start in range(1, num_results + 1, 10):
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query,
                        "start": start, "num": min(10, num_results - len(collected))},
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
                "position": pos, "title": item.get("title", ""),
                "url": item.get("link", ""), "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Google API] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


def search_yahoo(query: str, num_results: int = 20) -> dict:
    result = _make_result(query, "yahoo", num_results)
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(
            "https://search.yahoo.com/search",
            params={"p": query, "n": num_results},
            headers=headers, timeout=30,
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
                "position": pos, "title": link.get_text(strip=True),
                "url": href, "snippet": "",
            })
            if pos >= num_results:
                break
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Yahoo] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(2, 4))
    return result


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
            if item.get("t") != 0:
                continue
            result["raw_results"].append({
                "position": pos, "title": item.get("title", ""),
                "url": item.get("url", ""), "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Kagi] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


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
            headers={"Accept": "application/json", "Accept-Encoding": "gzip",
                     "X-Subscription-Token": BRAVE_API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for pos, item in enumerate(data.get("web", {}).get("results", [])[:num_results], 1):
            result["raw_results"].append({
                "position": pos, "title": item.get("title", ""),
                "url": item.get("url", ""), "snippet": item.get("description", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [Brave] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


def search_serpapi(query: str, num_results: int = 20) -> dict:
    from serpapi import GoogleSearch
    result = _make_result(query, "serpapi_google", num_results)
    if not SERPAPI_KEY:
        result["error"] = "SERPAPI_KEY not set in .env.local"
        result["response_timestamp_utc"] = utcnow_iso()
        return result
    try:
        collected, seen_urls, start = [], set(), 0
        while len(collected) < num_results:
            params = {"q": query, "num": 10, "start": start,
                      "api_key": SERPAPI_KEY, "engine": "google"}
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
            if start >= 30:
                break
            time.sleep(0.5)
        for pos, item in enumerate(collected[:num_results], 1):
            result["raw_results"].append({
                "position": pos, "title": item.get("title", ""),
                "url": item.get("link", ""), "snippet": item.get("snippet", ""),
            })
    except Exception as e:
        result["error"] = str(e)
        print(f"  [SerpAPI] Error: {e}")
    result["response_timestamp_utc"] = utcnow_iso()
    time.sleep(random.uniform(1, 2))
    return result


_ENGINE_MAP = {
    "searxng": search_searxng, "duckduckgo": search_duckduckgo,
    "google": search_google, "google_api": search_google_api,
    "yahoo": search_yahoo, "kagi": search_kagi,
    "brave": search_brave, "serpapi": search_serpapi,
}


def search(engine: str, query: str, num_results: int = 20) -> dict:
    fn = _ENGINE_MAP.get(engine)
    if fn is None:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {ENGINES}")
    return fn(query, num_results)


# ── LLM Re-Ranker ────────────────────────────────────────────────────────────

def _build_rerank_prompt(keyword: str, search_results: list[dict], top_n: int) -> str:
    results_text = ""
    for r in search_results:
        domain = _extract_domain(r["url"])
        results_text += f"{r['position']}. [{domain}] {r['title']} — {r['snippet'][:150]}\n"
    return f"""Search keyword: {keyword}

Below are search engine results for the above keyword. Re-rank the results and return the top {top_n} software product domains, ordered by relevance to the keyword.

Exclude non-product sites: review aggregators, directories, Wikipedia, news, blogs, forums, YouTube.

Return only root domains, one per line. No explanations.

Search results:
{results_text}

Re-ranked product domains:"""


def _parse_domains(llm_output: str) -> list[str]:
    domains = []
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in ["- ", "* "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
            if len(parts) > 1:
                line = parts[1].strip()
        ext = tldextract.extract(line)
        if ext.domain and ext.suffix:
            domain = f"{ext.domain}.{ext.suffix}"
            if domain not in domains:
                domains.append(domain)
    return domains


def _build_domain_url_map(search_results: list[dict]) -> dict:
    domain_url = {}
    for r in search_results:
        url = r.get("url", "")
        domain = _extract_domain(url)
        if domain and domain not in domain_url:
            domain_url[domain] = url
    return domain_url


def _attach_urls(domains: list[str], domain_url_map: dict) -> list[dict]:
    return [{"domain": d, "url": domain_url_map.get(d, "")} for d in domains]


def _fallback_extract(search_results: list[dict], top_n: int) -> list[str]:
    skip_domains = {
        "g2.com", "capterra.com", "wikipedia.org", "youtube.com",
        "reddit.com", "quora.com", "forbes.com", "techcrunch.com",
        "gartner.com", "trustradius.com", "softwareadvice.com",
        "getapp.com", "pcmag.com", "techradar.com", "cnet.com",
    }
    domains = []
    for r in search_results:
        d = _extract_domain(r["url"])
        if d and d not in domains and d not in skip_domains:
            domains.append(d)
        if len(domains) >= top_n:
            break
    return domains


def rank_domains_with_llm(keyword: str, search_results: list[dict],
                          top_n: int = 10, model_id: str = "meta-llama/Llama-3.3-70B-Instruct") -> dict:
    from huggingface_hub import InferenceClient

    result = {
        "keyword": keyword,
        "llm_role": "re-ranker (LLM re-orders results by relevance)",
        "llm_model": model_id,
        "llm_parameters": {"max_tokens": 500, "temperature": 0.1},
        "prompt": None, "raw_llm_response": None,
        "llm_query_timestamp_utc": None, "llm_response_timestamp_utc": None,
        "ranked_domains": [], "ranked_results": [],
        "used_fallback": False, "error": None,
    }

    if not search_results:
        result["error"] = "no search results provided"
        return result

    domain_url_map = _build_domain_url_map(search_results)
    client = InferenceClient(token=HF_TOKEN)
    prompt = _build_rerank_prompt(keyword, search_results, top_n)
    result["prompt"] = prompt
    result["llm_query_timestamp_utc"] = utcnow_iso()

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, max_tokens=500, temperature=0.1,
        )
        llm_output = response.choices[0].message.content
        # Strip DeepSeek R1 <think>...</think> reasoning tags (no-op for other models)
        llm_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL).strip()
    except Exception as e:
        result["llm_response_timestamp_utc"] = utcnow_iso()
        result["error"] = str(e)
        result["used_fallback"] = True
        domains = _fallback_extract(search_results, top_n)
        result["ranked_domains"] = domains
        result["ranked_results"] = _attach_urls(domains, domain_url_map)
        print(f"  [LLM] Error ranking for '{keyword}': {e}")
        return result

    result["llm_response_timestamp_utc"] = utcnow_iso()
    result["raw_llm_response"] = llm_output
    domains = _parse_domains(llm_output)[:top_n]
    result["ranked_domains"] = domains
    result["ranked_results"] = _attach_urls(domains, domain_url_map)
    return result


# ── Rank Changes ──────────────────────────────────────────────────────────────

def compute_rank_changes(raw_results: list[dict], post_llm_domains: list[str]) -> list[dict]:
    pre_domains = []
    for r in raw_results:
        d = _extract_domain(r.get("url", ""))
        if d and d not in pre_domains:
            pre_domains.append(d)
    pre_rank_map = {d: i + 1 for i, d in enumerate(pre_domains)}

    changes = []
    for post_rank_0, domain in enumerate(post_llm_domains):
        post_rank = post_rank_0 + 1
        pre_rank = pre_rank_map.get(domain)
        rank_delta = (pre_rank - post_rank) if pre_rank is not None else None
        changes.append({
            "domain": domain, "pre_rank": pre_rank,
            "post_rank": post_rank, "rank_delta": rank_delta,
        })
    return changes


# ── HTML Feature Extraction ───────────────────────────────────────────────────

def _get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _extract_body_text(soup: BeautifulSoup) -> str:
    soup = copy(soup)
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                               "noscript", "iframe", "svg"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    body = soup.find("body")
    if body is None:
        return ""
    return body.get_text(separator=" ", strip=True)


_STAT_PATTERNS = [
    re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'),
    re.compile(r'\b\d+\.?\d*%'),
    re.compile(r'\b(?:19|20)\d{2}\b'),
    re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'),
    re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
    re.compile(r'\b\d+(?:\.\d+)?[BMKbmk]\b'),
]


def t1_statistical_density(body_text: str) -> float | None:
    if not body_text.strip():
        return None
    words = body_text.split()
    word_count = len(words)
    if word_count == 0:
        return None
    found = set()
    for pat in _STAT_PATTERNS:
        for match in pat.finditer(body_text):
            found.add(match.group())
    return round(len(found) / (word_count / 500), 2)


_QUESTION_RE = re.compile(
    r'^\s*(?:what|how|why|when|where|which|who|can|does|is|are|should|will|do)\b',
    re.IGNORECASE,
)


def t2_question_heading_match(soup: BeautifulSoup) -> int:
    for heading in soup.find_all(["h2", "h3"]):
        if _QUESTION_RE.match(heading.get_text(strip=True)):
            return 1
    return 0


def _check_ld_type(data, target_types: set) -> bool:
    if isinstance(data, dict):
        type_val = data.get("@type", "")
        if isinstance(type_val, str) and type_val.lower() in target_types:
            return True
        if isinstance(type_val, list):
            if any(t.lower() in target_types for t in type_val if isinstance(t, str)):
                return True
        if "@graph" in data:
            return _check_ld_type(data["@graph"], target_types)
    elif isinstance(data, list):
        return any(_check_ld_type(item, target_types) for item in data)
    return False


def t3_structured_data_presence(soup: BeautifulSoup) -> int:
    target_types = {"faqpage", "faq", "product", "howto"}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if _check_ld_type(data, target_types):
            return 1
    return 0


_AUTHORITY_SUFFIXES = {"edu", "gov", "gov.uk", "ac.uk", "mil"}
_AUTHORITY_DOMAINS = {
    "wikipedia.org", "scholar.google.com", "ncbi.nlm.nih.gov",
    "arxiv.org", "nature.com", "sciencedirect.com", "ieee.org",
    "acm.org", "researchgate.net", "pubmed.ncbi.nlm.nih.gov",
}


def t4_external_citation_authority(soup: BeautifulSoup, page_domain: str) -> int:
    count = 0
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        try:
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https"):
                continue
            ext = tldextract.extract(href)
            link_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
            if not link_domain or link_domain == page_domain:
                continue
            if ext.suffix in _AUTHORITY_SUFFIXES or link_domain in _AUTHORITY_DOMAINS:
                count += 1
        except Exception:
            continue
    return count


def x3_word_count(body_text: str) -> int | None:
    if not body_text.strip():
        return None
    return len(body_text.split())


def x6_flesch_kincaid(body_text: str) -> float | None:
    if not body_text.strip() or len(body_text.split()) < 100:
        return None
    try:
        return round(textstat.flesch_kincaid_grade(body_text), 2)
    except Exception:
        return None


def x7_internal_link_count(soup: BeautifulSoup, page_domain: str) -> int:
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        if not parsed.scheme and not parsed.netloc:
            if href.startswith("/") or href.startswith("#") or href.startswith("?"):
                count += 1
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        ext = tldextract.extract(href)
        link_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
        if link_domain == page_domain:
            count += 1
    return count


def x7b_outbound_link_count(soup: BeautifulSoup, page_domain: str) -> int:
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        if parsed.scheme not in ("http", "https"):
            continue
        ext = tldextract.extract(href)
        link_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
        if link_domain and link_domain != page_domain:
            count += 1
    return count


def x9_images_with_alt(soup: BeautifulSoup) -> int:
    return sum(1 for img in soup.find_all("img") if img.get("alt", "").strip())


def x10_https_status(url: str) -> int:
    return 1 if url.lower().startswith("https://") else 0


def extract_html_features(html: str, url: str, domain: str) -> dict:
    result = {
        "url": url, "domain": domain,
        "T1_statistical_density": None, "T2_question_heading_match": None,
        "T3_structured_data": None, "T4_citation_authority": None,
        "X3_word_count": None, "X6_readability": None,
        "X7_internal_links": None, "X7B_outbound_links": None,
        "X9_images_with_alt": None, "X10_https": x10_https_status(url),
        "error": None,
    }
    if not html or not html.strip():
        result["error"] = "empty_html"
        return result
    try:
        soup = _get_soup(html)
        body_text = _extract_body_text(soup)
        result["T1_statistical_density"] = t1_statistical_density(body_text)
        result["T2_question_heading_match"] = t2_question_heading_match(soup)
        result["T3_structured_data"] = t3_structured_data_presence(soup)
        result["T4_citation_authority"] = t4_external_citation_authority(soup, domain)
        result["X3_word_count"] = x3_word_count(body_text)
        result["X6_readability"] = x6_flesch_kincaid(body_text)
        result["X7_internal_links"] = x7_internal_link_count(soup, domain)
        result["X7B_outbound_links"] = x7b_outbound_link_count(soup, domain)
        result["X9_images_with_alt"] = x9_images_with_alt(soup)
    except Exception as e:
        result["error"] = f"extraction_error: {str(e)[:200]}"
    return result


# ── LLM Treatment Extraction ─────────────────────────────────────────────────

def _collect_ld_types(data, types_list: list):
    if isinstance(data, dict):
        t = data.get("@type", "")
        if isinstance(t, str) and t:
            types_list.append(t)
        elif isinstance(t, list):
            types_list.extend(s for s in t if isinstance(s, str))
        if "@graph" in data:
            _collect_ld_types(data["@graph"], types_list)
    elif isinstance(data, list):
        for item in data:
            _collect_ld_types(item, types_list)


def build_page_digest(html: str, url: str, domain: str, max_body_chars: int = 3000) -> str:
    soup = _get_soup(html)
    parts = [f"URL: {url}", f"Domain: {domain}", ""]

    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(f"  <{tag.name}> {text}")
    if headings:
        parts.append("HEADINGS:")
        parts.extend(headings[:40])
    else:
        parts.append("HEADINGS: (none found)")
    parts.append("")

    body_text = _extract_body_text(soup)
    if body_text:
        sample = body_text[:max_body_chars]
        if len(body_text) > max_body_chars:
            sample += f"\n  ... [truncated, total {len(body_text)} chars]"
        parts.append(f"BODY TEXT ({len(body_text.split())} words):")
        parts.append(sample)
    else:
        parts.append("BODY TEXT: (empty)")
    parts.append("")

    jsonld_types = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            _collect_ld_types(data, jsonld_types)
        except (json.JSONDecodeError, TypeError):
            continue
    parts.append(f"JSON-LD TYPES: {', '.join(jsonld_types)}" if jsonld_types else "JSON-LD TYPES: (none)")
    parts.append("")

    outbound = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        try:
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https"):
                continue
            ext = tldextract.extract(href)
            link_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
            if link_domain and link_domain != domain:
                suffix_tag = ""
                if ext.suffix in _AUTHORITY_SUFFIXES:
                    suffix_tag = f" [.{ext.suffix}]"
                elif link_domain in _AUTHORITY_DOMAINS:
                    suffix_tag = " [academic]"
                outbound.append(f"  {link_domain}{suffix_tag}")
        except Exception:
            continue
    if outbound:
        parts.append(f"OUTBOUND LINKS ({len(outbound)} unique external domains):")
        parts.extend(outbound[:50])
        if len(outbound) > 50:
            parts.append(f"  ... and {len(outbound) - 50} more")
    else:
        parts.append("OUTBOUND LINKS: (none)")
    return "\n".join(parts)


_LLM_TREATMENT_PROMPT = """You are analyzing a web page to measure 4 treatment variables for a causal inference experiment on search engine optimization.

Given the page digest below, evaluate each treatment and return ONLY a JSON object with these exact keys:

- "T1_llm_statistical_density": (float) Count of unique statistics, numbers, percentages, dollar amounts, or dates per 500 words of body text. Be precise.
- "T2_llm_question_heading": (0 or 1) Does the page contain H2 or H3 headings that closely match natural language questions (e.g. "What is...", "How to...", "Why should...")? 1 if yes, 0 if no.
- "T3_llm_structured_data": (0 or 1) Does the page have JSON-LD structured data of type FAQ, Product, or HowTo? 1 if yes, 0 if no.
- "T4_llm_citation_authority": (integer) Count of outbound links to authoritative sources (.edu, .gov, academic journals, Wikipedia, government sites). Only count genuinely authoritative citations, not marketing links.
- "T1_reasoning": (string) Brief explanation for T1 score.
- "T2_reasoning": (string) Brief explanation for T2 score.
- "T3_reasoning": (string) Brief explanation for T3 score.
- "T4_reasoning": (string) Brief explanation for T4 score.

Return ONLY valid JSON. No markdown, no extra text.

PAGE DIGEST:
{digest}"""


def llm_extract_treatments(digest: str, client, model_id: str) -> dict:
    result = {
        "T1_llm_statistical_density": None, "T2_llm_question_heading": None,
        "T3_llm_structured_data": None, "T4_llm_citation_authority": None,
        "T1_reasoning": None, "T2_reasoning": None,
        "T3_reasoning": None, "T4_reasoning": None,
        "llm_error": None,
    }
    prompt = _LLM_TREATMENT_PROMPT.format(digest=digest)
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, max_tokens=800, temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip DeepSeek R1 <think>...</think> reasoning tags (no-op for other models)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        for key in result:
            if key in parsed:
                result[key] = parsed[key]
    except json.JSONDecodeError as e:
        result["llm_error"] = f"json_parse: {str(e)[:100]}"
    except Exception as e:
        result["llm_error"] = f"llm_error: {str(e)[:150]}"
    return result


# ── HTML Fetching ─────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _url_to_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class _FetchTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FetchTimeout("hard timeout")


def fetch_page(url: str, session: requests.Session) -> tuple[str | None, int | None, str | None]:
    import signal
    # Hard 45-second wall-clock timeout to catch drip-feed stalls
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(45)
    try:
        resp = session.get(url, timeout=(10, FETCH_TIMEOUT), allow_redirects=True)
        signal.alarm(0)  # cancel alarm
        status = resp.status_code
        ct = resp.headers.get("Content-Type", "")
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None, status, f"non_html: {ct[:60]}"
        if len(resp.content) > MAX_HTML_SIZE:
            return None, status, f"too_large: {len(resp.content)}"
        if status >= 400:
            return None, status, f"http_{status}"
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text, status, None
    except _FetchTimeout:
        return None, None, "hard_timeout"
    except requests.exceptions.SSLError:
        return None, None, "ssl_error"
    except requests.exceptions.Timeout:
        return None, None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, None, "connection_error"
    except requests.RequestException as e:
        return None, None, f"request_error: {str(e)[:100]}"
    except Exception as e:
        return None, None, f"unexpected_error: {str(e)[:100]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Optional Phases ───────────────────────────────────────────────────────────

def run_phase_pagerank(features: dict[str, dict], api_key: str) -> dict[str, dict]:
    if not api_key:
        print("\n  OPENPAGERANK_KEY not set. Skipping.")
        return features
    domains = sorted({f.get("domain", "") for f in features.values()} - {""})
    done = {f.get("domain", "") for f in features.values() if f.get("X1_domain_authority") is not None}
    remaining = [d for d in domains if d not in done]
    print(f"\nPhase: Open PageRank (Domain Authority)")
    print(f"  Unique domains: {len(domains)}, remaining: {len(remaining)}")
    cache = {}
    for batch_start in range(0, len(remaining), 100):
        batch = remaining[batch_start:batch_start + 100]
        print(f"  Batch {batch_start // 100 + 1}: {len(batch)} domains...", end=" ")
        try:
            params = [("domains[]", d) for d in batch]
            resp = requests.get(
                "https://openpagerank.com/api/v1.0/getPageRank",
                params=params, headers={"API-OPR": api_key}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("response", []):
                domain = item.get("domain", "")
                pr = item.get("page_rank_decimal")
                rank = item.get("rank")
                if domain:
                    cache[domain] = {"authority": pr, "global_rank": int(rank) if rank else None}
            print(f"OK ({len(cache)} total)")
        except Exception as e:
            print(f"Error: {str(e)[:80]}")
        time.sleep(1)
    for url, feat in features.items():
        d = feat.get("domain", "")
        if d in cache:
            feat["X1_domain_authority"] = cache[d]["authority"]
            feat["X1_global_rank"] = cache[d]["global_rank"]
    return features


def run_phase_whois(features: dict[str, dict]) -> dict[str, dict]:
    try:
        import whois as whois_lib
    except ImportError:
        print("\n  python-whois not installed. Run: pip install python-whois")
        return features
    domains = sorted({f.get("domain", "") for f in features.values()} - {""})
    done = {f.get("domain", "") for f in features.values() if f.get("X2_domain_age_years") is not None}
    remaining = [d for d in domains if d not in done]
    print(f"\nPhase: WHOIS Domain Age")
    print(f"  Unique domains: {len(domains)}, remaining: {len(remaining)}")
    now = datetime.now(timezone.utc)
    cache = {}
    for i, domain in enumerate(remaining, 1):
        print(f"  [{i}/{len(remaining)}] {domain}", end=" ")
        try:
            w = whois_lib.whois(domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation:
                if hasattr(creation, "tzinfo") and creation.tzinfo is None:
                    creation = creation.replace(tzinfo=timezone.utc)
                age_years = round((now - creation).days / 365.25, 1)
                cache[domain] = {"date": creation.isoformat(), "years": age_years}
                print(f"-> {age_years} years")
            else:
                cache[domain] = {"date": None, "years": None}
                print("-> unknown")
        except Exception as e:
            cache[domain] = {"date": None, "years": None}
            print(f"-> error: {str(e)[:60]}")
        time.sleep(2)
    for url, feat in features.items():
        d = feat.get("domain", "")
        if d in cache:
            feat["X2_domain_age_date"] = cache[d]["date"]
            feat["X2_domain_age_years"] = cache[d]["years"]
    return features


def run_phase_llm_features(features: dict[str, dict], html_cache: dict[str, str],
                           hf_token: str, model_id: str,
                           html_cache_dir: Path = None,
                           features_csv: Path = None,
                           max_consecutive_errors: int = 20) -> dict[str, dict]:
    if not hf_token:
        print("\n  HF_TOKEN not set, skipping LLM phase.")
        return features
    from huggingface_hub import InferenceClient
    urls_to_process = [
        url for url, f in features.items()
        if not f.get("error") and not f.get("T1_llm_statistical_density")
    ]
    print(f"\nPhase LLM: LLM Treatment Extraction")
    print(f"  Model: {model_id}, URLs to analyze: {len(urls_to_process)}")
    if not urls_to_process:
        print("  Nothing to do.")
        return features
    client = InferenceClient(token=hf_token)
    consecutive_errors = 0
    processed_count = 0
    for i, url in enumerate(urls_to_process, 1):
        feat = features[url]
        domain = feat.get("domain", "")
        print(f"\n[{i}/{len(urls_to_process)}] {url} ({domain})")
        html = html_cache.get(url)
        # Lazy-load from disk cache if not in memory
        if not html and html_cache_dir:
            cache_path = html_cache_dir / f"{_url_to_cache_key(url)}.html"
            if cache_path.exists():
                html = cache_path.read_text(encoding="utf-8")
        if not html:
            print("  No cached HTML, skipping.")
            feat["llm_error"] = "no_cached_html"
            continue
        digest = build_page_digest(html, url, domain)
        llm_result = llm_extract_treatments(digest, client, model_id)
        for key, val in llm_result.items():
            feat[key] = val
        err = feat.get("llm_error", "")
        if err:
            print(f"  LLM Error: {err}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n  CIRCUIT BREAKER: {max_consecutive_errors} consecutive LLM errors. "
                      f"Stopping Phase 3 early ({processed_count} succeeded, {i} attempted).")
                print(f"  Re-run to resume from where we left off.")
                break
        else:
            print(f"  LLM  T1={feat.get('T1_llm_statistical_density', '?')}  "
                  f"T2={feat.get('T2_llm_question_heading', '?')}  "
                  f"T3={feat.get('T3_llm_structured_data', '?')}  "
                  f"T4={feat.get('T4_llm_citation_authority', '?')}")
            consecutive_errors = 0
            processed_count += 1
        # Incremental save every 5 URLs
        if features_csv and i % 5 == 0:
            _save_features_csv(features, features_csv)
            print(f"  [checkpoint] Saved {processed_count} LLM features to {features_csv}")
        time.sleep(random.uniform(0.5, 1.5))
    # Final incremental save
    if features_csv and processed_count > 0:
        _save_features_csv(features, features_csv)
        print(f"  [final] Saved {processed_count} LLM features to {features_csv}")
    return features


def _save_features_csv(features: dict[str, dict], features_csv: Path):
    """Write full features dict to CSV (atomic overwrite)."""
    cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    tmp = features_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for url in sorted(features):
            writer.writerow(features[url])
    tmp.replace(features_csv)


def compute_keyword_difficulty(rankings: list[dict], features: dict[str, dict]) -> dict[str, float]:
    from collections import defaultdict
    kw_authorities = defaultdict(list)
    for row in rankings:
        keyword = row.get("keyword", "")
        url = row.get("url", "").strip()
        domain = row.get("domain", "").strip()
        lookup = url if url else f"https://{domain}/"
        feat = features.get(lookup, {})
        auth = feat.get("X1_domain_authority")
        if auth is not None and auth != "" and keyword:
            try:
                kw_authorities[keyword].append(float(auth))
            except (ValueError, TypeError):
                pass
    kw_difficulty = {}
    for keyword, auths in kw_authorities.items():
        if auths:
            kw_difficulty[keyword] = round(sum(auths) / len(auths), 2)
    return kw_difficulty


# ── Output Columns ────────────────────────────────────────────────────────────

LLM_TREATMENT_COLS = [
    "T1_llm_statistical_density", "T2_llm_question_heading",
    "T3_llm_structured_data", "T4_llm_citation_authority",
    "T1_reasoning", "T2_reasoning", "T3_reasoning", "T4_reasoning",
    "llm_error",
]

FEATURE_COLS = [
    "T1_statistical_density", "T2_question_heading_match",
    "T3_structured_data", "T4_citation_authority",
] + LLM_TREATMENT_COLS + [
    "X1_domain_authority", "X1_global_rank",
    "X2_domain_age_years", "X2_domain_age_date",
    "X3_word_count", "X6_readability",
    "X7_internal_links", "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt", "X10_https",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gather data: SERP + LLM re-ranking + page features")
    parser.add_argument("--engine", type=str, default="searxng", choices=ENGINES,
                        help=f"Search engine backend (default: searxng)")
    parser.add_argument("--serp-results", type=int, default=20,
                        help="Number of results to collect from search engine (default: 20)")
    parser.add_argument("--llm-top-n", type=int, default=10,
                        help="Number of results the LLM re-ranks to (default: 10)")
    parser.add_argument("--llm-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID for LLM re-ranking")
    parser.add_argument("--keywords-file", type=str, default=str(PROJECT_ROOT / "keywords.txt"),
                        help="Path to keywords file (default: ../keywords.txt)")
    parser.add_argument("--keywords", type=int, default=0,
                        help="Limit to first N keywords (0=all)")
    parser.add_argument("--output-dir", type=str, default="output/",
                        help="Output directory (default: output/)")
    parser.add_argument("--llm-features", action="store_true",
                        help="Enable LLM-based treatment extraction (T1-T4 via LLM)")
    parser.add_argument("--pagerank", action="store_true",
                        help="Enable Open PageRank API (X1 domain authority)")
    parser.add_argument("--whois", action="store_true",
                        help="Enable WHOIS domain age (X2)")
    parser.add_argument("--all-features", action="store_true",
                        help="Enable all optional feature phases")
    parser.add_argument("--max-urls", type=int, default=0,
                        help="Limit URLs for HTML fetching (0=all)")
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Path to write live progress JSON (updated after each keyword/URL)")
    args = parser.parse_args()

    if args.all_features:
        args.llm_features = args.pagerank = args.whois = True

    # ── Progress file helper ─────────────────────────────────────────────
    _progress_state = {
        "phase": "init",
        "phase1_keywords_total": 0,
        "phase1_keywords_done": 0,
        "phase1_keywords_failed": 0,
        "phase1_current_keyword": None,
        "phase1_llm_errors": 0,
        "phase1_llm_fallbacks": 0,
        "phase2_urls_total": 0,
        "phase2_urls_done": 0,
        "phase2_urls_failed": 0,
        "phase2_current_url": None,
        "phase3_llm_features_total": 0,
        "phase3_llm_features_done": 0,
        "phase_pagerank": "pending",
        "phase_whois": "pending",
        "last_updated_utc": None,
    }

    def _write_progress(**updates):
        if not args.progress_file:
            return
        _progress_state.update(updates)
        _progress_state["last_updated_utc"] = utcnow_iso()
        try:
            with open(args.progress_file, "w") as pf:
                json.dump(_progress_state, pf, indent=2)
        except Exception:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_cache_dir = output_dir / "html_cache"
    html_cache_dir.mkdir(exist_ok=True)

    # ── Load keywords ─────────────────────────────────────────────────────
    kw_path = Path(args.keywords_file)
    if not kw_path.exists():
        print(f"Keywords file not found: {kw_path}")
        sys.exit(1)
    with open(kw_path) as f:
        keywords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if args.keywords > 0:
        keywords = keywords[:args.keywords]

    # ── Experiment context ────────────────────────────────────────────────
    print("Collecting experiment context...")
    context = collect_experiment_context()
    geo = context["network"]["geolocation"]
    print(f"  IP: {context['network']['public_ip']}")
    print(f"  Location: {geo['city']}, {geo['region']}, {geo['country']}")
    print()

    print(f"Keywords: {len(keywords)}")
    print(f"SERP engine: {args.engine}, results per query: {args.serp_results}")
    print(f"LLM re-rank to top: {args.llm_top_n}")
    print(f"LLM model: {args.llm_model}")
    phases = ["SERP+LLM", "HTML"]
    if args.llm_features: phases.append("LLM-Features")
    if args.pagerank: phases.append("PageRank")
    if args.whois: phases.append("WHOIS")
    print(f"Phases: {' -> '.join(phases)}")
    print()

    # ── Phase 1: SERP + LLM Re-Ranking ───────────────────────────────────
    per_keyword_data = []
    errors = []
    rankings_rows = []

    # ── Resume logic: detect already-processed keywords ────────────────
    rankings_csv = output_dir / "rankings.csv"
    done_keywords = set()
    if rankings_csv.exists():
        with open(rankings_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kw = row.get("keyword", "").strip()
                if kw:
                    done_keywords.add(kw)
                    rankings_rows.append(row)
        if done_keywords:
            print(f"  Resuming: {len(done_keywords)} keywords already processed, skipping them")

    # ── Prepare incremental CSV writer ─────────────────────────────────
    _rankings_fields = ["keyword", "domain", "url", "engine", "model"]
    _csv_needs_header = not rankings_csv.exists() or rankings_csv.stat().st_size == 0
    _rankings_fh = open(rankings_csv, "a", newline="")
    _rankings_writer = csv.DictWriter(_rankings_fh, fieldnames=_rankings_fields)
    if _csv_needs_header:
        _rankings_writer.writeheader()
        _rankings_fh.flush()

    # ── Incremental JSONL for per-keyword data ─────────────────────────
    jsonl_path = output_dir / "keywords.jsonl"
    _jsonl_fh = open(jsonl_path, "a")

    _write_progress(phase="phase1_serp_llm", phase1_keywords_total=len(keywords))

    try:
        for i, keyword in enumerate(keywords, 1):
            if keyword in done_keywords:
                continue
            print(f"[{i}/{len(keywords)}] Searching: {keyword}")
            _write_progress(phase1_current_keyword=keyword, phase1_keywords_done=len(done_keywords))

            serp_result = search(args.engine, keyword, num_results=args.serp_results)
            raw_results = serp_result["raw_results"]

            if not raw_results:
                print(f"  No results, skipping")
                errors.append({"keyword": keyword, "error": serp_result.get("error")})
                _write_progress(phase1_keywords_failed=len(errors))
                done_keywords.add(keyword)
                continue

            print(f"  Got {len(raw_results)} results via {args.engine}")

            llm_result = rank_domains_with_llm(keyword, raw_results,
                                               top_n=args.llm_top_n, model_id=args.llm_model)
            domains = llm_result["ranked_domains"]
            print(f"  LLM re-ranked: {domains[:5]}{'...' if len(domains) > 5 else ''}")

            if llm_result["used_fallback"]:
                print(f"  (used fallback due to LLM error)")
                _write_progress(phase1_llm_fallbacks=_progress_state["phase1_llm_fallbacks"] + 1)
            if llm_result.get("error"):
                _write_progress(phase1_llm_errors=_progress_state["phase1_llm_errors"] + 1)

            rank_changes = compute_rank_changes(raw_results, domains)
            deltas = [rc["rank_delta"] for rc in rank_changes if rc["rank_delta"] is not None]
            print(f"  Rank deltas: {deltas}")

            kw_data = {
                "query": keyword,
                "query_timestamp_utc": serp_result["query_timestamp_utc"],
                "serp": serp_result, "llm": llm_result,
                "ranked_domains": domains,
                "ranked_results": llm_result["ranked_results"],
                "rank_changes": rank_changes,
            }
            per_keyword_data.append(kw_data)

            # Incremental JSONL save
            _jsonl_fh.write(json.dumps(kw_data, ensure_ascii=False) + "\n")
            _jsonl_fh.flush()

            # Build ranking rows for CSV + incremental save
            for rr in llm_result["ranked_results"]:
                row = {
                    "keyword": keyword,
                    "domain": rr["domain"],
                    "url": rr["url"],
                    "engine": args.engine,
                    "model": args.llm_model,
                }
                rankings_rows.append(row)
                _rankings_writer.writerow(row)
            _rankings_fh.flush()

            done_keywords.add(keyword)
            _write_progress(phase1_keywords_done=len(done_keywords))
    finally:
        _rankings_fh.close()
        _jsonl_fh.close()

    _write_progress(phase="phase1_complete", phase1_keywords_done=len(keywords))

    # ── Save experiment JSON ──────────────────────────────────────────────
    llm_label = args.llm_model.split("/")[-1]
    date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")

    experiment_data = {
        "experiment_context": context,
        "experiment_end_utc": utcnow_iso(),
        "source": "ai_search",
        "serp_engine": args.engine,
        "serp_results_requested": args.serp_results,
        "llm_top_n": args.llm_top_n,
        "method": f"{args.engine} + {args.llm_model} LLM re-ranking",
        "chat_model": args.llm_model,
        "total_keywords": len(keywords),
        "successful_keywords": len(per_keyword_data),
        "failed_keywords": errors,
        "per_keyword_results": per_keyword_data,
    }

    json_path = output_dir / "experiment.json"
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved experiment JSON: {json_path}")

    # rankings.csv already written incrementally above
    print(f"Rankings CSV: {rankings_csv} ({len(rankings_rows)} rows)")

    # ── Phase 2: HTML Feature Extraction ──────────────────────────────────
    # Build unique URL list from rankings
    url_list = []
    seen_urls = set()
    for row in rankings_rows:
        url = row.get("url", "").strip()
        domain = row.get("domain", "").strip()
        if not url:
            url = f"https://{domain}/"
        if url not in seen_urls:
            seen_urls.add(url)
            url_list.append({"url": url, "domain": domain})

    if args.max_urls > 0:
        url_list = url_list[:args.max_urls]

    print(f"\nPhase 2: HTML Feature Extraction")
    print(f"  Unique URLs: {len(url_list)}")
    _write_progress(phase="phase2_html_features", phase2_urls_total=len(url_list))

    session = _make_session()
    features = {}  # url -> feature dict
    html_cache = {}  # url -> html (in-memory for LLM phase)

    # ── Resume logic: load already-processed URLs from features.csv ────
    features_csv = output_dir / "features.csv"
    _feat_cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    done_urls = set()
    if features_csv.exists() and features_csv.stat().st_size > 0:
        with open(features_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row.get("url", "").strip()
                if u:
                    done_urls.add(u)
                    features[u] = row
        if done_urls:
            print(f"  Resuming: {len(done_urls)} URLs already processed, skipping them")
            # NOTE: cached HTML for already-done URLs will be loaded lazily
            # in Phase 3 (LLM features) to avoid loading 1GB+ into memory now

    # ── Prepare incremental features CSV writer ────────────────────────
    _feat_needs_header = not features_csv.exists() or features_csv.stat().st_size == 0
    _feat_fh = open(features_csv, "a", newline="")
    _feat_writer = csv.DictWriter(_feat_fh, fieldnames=_feat_cols, extrasaction="ignore")
    if _feat_needs_header:
        _feat_writer.writeheader()
        _feat_fh.flush()

    try:
        for i, entry in enumerate(url_list, 1):
            url = entry["url"]
            domain = entry["domain"]
            if url in done_urls:
                continue
            print(f"  [{i}/{len(url_list)}] {url}", end=" ")
            _write_progress(phase2_current_url=url, phase2_urls_done=len(done_urls))

            html, status, fetch_err = fetch_page(url, session)

            if fetch_err:
                feat = {
                    "url": url, "domain": domain,
                    "T1_statistical_density": None, "T2_question_heading_match": None,
                    "T3_structured_data": None, "T4_citation_authority": None,
                    "X3_word_count": None, "X6_readability": None,
                    "X7_internal_links": None, "X7B_outbound_links": None,
                    "X9_images_with_alt": None,
                    "X10_https": 1 if url.lower().startswith("https://") else 0,
                    "error": fetch_err,
                }
                print(f"Error: {fetch_err}")
                _write_progress(phase2_urls_failed=_progress_state["phase2_urls_failed"] + 1)
            else:
                # Cache HTML on disk + in memory
                cache_path = html_cache_dir / f"{_url_to_cache_key(url)}.html"
                with open(cache_path, "w", encoding="utf-8") as hf:
                    hf.write(html)
                html_cache[url] = html
                feat = extract_html_features(html, url, domain)
                print(f"OK  Words={feat.get('X3_word_count', '?')}  "
                      f"T1={feat.get('T1_statistical_density', '?')}")

            feat["fetch_status_code"] = status
            feat["fetch_timestamp_utc"] = utcnow_iso()
            features[url] = feat
            done_urls.add(url)

            # Incremental save to features.csv
            _feat_writer.writerow(feat)
            _feat_fh.flush()

            _write_progress(phase2_urls_done=len(done_urls))

            time.sleep(random.uniform(1, 3))
    finally:
        _feat_fh.close()

    _write_progress(phase="phase2_complete", phase2_urls_done=len(url_list))

    # ── Optional Phase: LLM Features ──────────────────────────────────────
    if args.llm_features:
        llm_urls = [u for u, f in features.items() if not f.get("error") and not f.get("T1_llm_statistical_density")]
        _write_progress(phase="phase3_llm_features", phase3_llm_features_total=len(llm_urls))
        features = run_phase_llm_features(features, html_cache, HF_TOKEN, args.llm_model,
                                              html_cache_dir=html_cache_dir,
                                              features_csv=features_csv)
        _write_progress(phase3_llm_features_done=len(llm_urls))

    # ── Optional Phase: PageRank ──────────────────────────────────────────
    if args.pagerank:
        _write_progress(phase="phase_pagerank", phase_pagerank="running")
        features = run_phase_pagerank(features, OPENPAGERANK_KEY)
        _write_progress(phase_pagerank="completed")

    # ── Optional Phase: WHOIS ─────────────────────────────────────────────
    if args.whois:
        _write_progress(phase="phase_whois", phase_whois="running")
        features = run_phase_whois(features)
        _write_progress(phase_whois="completed")

    # ── Compute keyword difficulty ────────────────────────────────────────
    kw_difficulty = compute_keyword_difficulty(rankings_rows, features)
    if kw_difficulty:
        for url, feat in features.items():
            # Find keyword for this URL from rankings
            for row in rankings_rows:
                if row.get("url", "").strip() == url or f"https://{row.get('domain', '')}/" == url:
                    kw = row.get("keyword", "")
                    if kw in kw_difficulty:
                        feat["X8_keyword_difficulty"] = kw_difficulty[kw]
                    break

    # ── Save features CSV (full rewrite with enriched data) ──────────────
    cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    with open(features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for url in sorted(features):
            writer.writerow(features[url])
    print(f"\nSaved features CSV: {features_csv} ({len(features)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────
    ok = sum(1 for f in features.values() if not f.get("error"))
    fail = len(features) - ok
    print(f"\n{'='*60}")
    print(f"Gather Data Complete")
    print(f"  SERP engine: {args.engine}")
    print(f"  SERP results per query: {args.serp_results}")
    print(f"  LLM re-rank top-N: {args.llm_top_n}")
    print(f"  LLM model: {args.llm_model}")
    print(f"  Keywords processed: {len(per_keyword_data)}/{len(keywords)}")
    print(f"  URLs fetched: {len(features)} (OK: {ok}, Failed: {fail})")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")
    _write_progress(phase="done")


if __name__ == "__main__":
    main()
