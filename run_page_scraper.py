#!/usr/bin/env python3
"""Scrape page features for DML causal inference experiment.

Reads a results CSV, fetches each unique URL once, extracts treatments
(T1-T4) and confounders (X1-X10) via code + LLM + APIs.

Usage:
  python run_page_scraper.py                          # code extraction only
  python run_page_scraper.py --llm                    # + LLM treatment analysis
  python run_page_scraper.py --pagerank               # + X1 domain authority
  python run_page_scraper.py --whois                  # + X2 domain age
  python run_page_scraper.py --lcp                    # + X4 page speed
  python run_page_scraper.py --all                    # all phases
  python run_page_scraper.py --max-urls 5             # test with 5 URLs
"""

import argparse
import csv
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime, timezone

import requests
import tldextract
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import RESULTS_DIR, GOOGLE_API_KEY, HF_TOKEN, OPENPAGERANK_KEY
from src.page_features import extract_html_features, build_page_digest, llm_extract_treatments


# ── Constants ────────────────────────────────────────────────────────────

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
FETCH_TIMEOUT = 30
MAX_HTML_SIZE = 5 * 1024 * 1024  # 5 MB
HTML_CACHE_DIR = RESULTS_DIR / "html_cache"
LLM_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


# ── Helpers ──────────────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _url_to_cache_key(url: str) -> str:
    """Convert URL to a safe filename for HTML cache."""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _save_html_cache(url: str, html: str):
    """Save fetched HTML to disk cache."""
    HTML_CACHE_DIR.mkdir(exist_ok=True)
    path = HTML_CACHE_DIR / f"{_url_to_cache_key(url)}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _load_html_cache(url: str) -> str | None:
    """Load HTML from disk cache. Returns None if not cached."""
    path = HTML_CACHE_DIR / f"{_url_to_cache_key(url)}.html"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    })
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ── Input / Progress ─────────────────────────────────────────────────────

def load_input_csv(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        return list(csv.DictReader(f))


def build_url_list(rows: list[dict]) -> list[dict]:
    """Deduplicate URLs; construct https://domain/ fallback for missing ones."""
    seen_urls = {}  # url -> domain
    domains_with_url = set()

    for row in rows:
        url = row.get("url", "").strip()
        domain = row.get("domain", "").strip()
        if url and url not in seen_urls:
            seen_urls[url] = domain
            domains_with_url.add(domain)

    # Fallback for domains that never had a URL
    all_domains = {row.get("domain", "").strip() for row in rows} - {""}
    for domain in sorted(all_domains - domains_with_url):
        fallback = f"https://{domain}/"
        if fallback not in seen_urls:
            seen_urls[fallback] = domain

    result = [
        {"url": url, "domain": domain}
        for url, domain in sorted(seen_urls.items(), key=lambda x: x[1])
    ]
    return result


def _progress_path(input_path: Path) -> Path:
    stem = input_path.stem.replace("all_results", "page_features_progress")
    return RESULTS_DIR / f"{stem}.json"


def load_progress(path: Path) -> dict[str, dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_progress(path: Path, features: dict[str, dict]):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(features, f, ensure_ascii=False)
    os.rename(tmp, path)


# ── Fetching ─────────────────────────────────────────────────────────────

def fetch_page(url: str, session: requests.Session) -> tuple[str | None, int | None, str | None]:
    """Fetch a URL. Returns (html, status_code, error)."""
    try:
        resp = session.get(url, timeout=FETCH_TIMEOUT, allow_redirects=True)
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

    except requests.exceptions.SSLError:
        return None, None, "ssl_error"
    except requests.exceptions.Timeout:
        return None, None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, None, "connection_error"
    except requests.RequestException as e:
        return None, None, f"request_error: {str(e)[:100]}"


# ── Phase 1: HTML Features ──────────────────────────────────────────────

def run_phase1(url_list: list[dict], existing: dict[str, dict],
               session: requests.Session, max_urls: int,
               rate_min: float, rate_max: float) -> dict[str, dict]:
    """Fetch pages and extract HTML features. Resumable."""
    remaining = [u for u in url_list if u["url"] not in existing]
    if max_urls > 0:
        remaining = remaining[:max_urls]

    total = len(remaining)
    print(f"\nPhase 1: HTML Feature Extraction")
    print(f"  Total unique URLs: {len(url_list)}")
    print(f"  Already completed: {len(existing)}")
    print(f"  Remaining:         {total}")

    if total == 0:
        print("  Nothing to do.")
        return existing

    progress_file = None  # set by caller via save_progress

    for i, entry in enumerate(remaining, 1):
        url = entry["url"]
        domain = entry["domain"]

        print(f"\n[{i}/{total}] {url} ({domain})")

        html, status, fetch_err = fetch_page(url, session)

        if fetch_err:
            features = {
                "url": url, "domain": domain,
                "T1_statistical_density": None,
                "T2_question_heading_match": None,
                "T3_structured_data": None,
                "T4_citation_authority": None,
                "X3_word_count": None,
                "X6_readability": None,
                "X7_internal_links": None,
                "X9_images_with_alt": None,
                "X10_https": 1 if url.lower().startswith("https://") else 0,
                "error": fetch_err,
            }
            print(f"  Error: {fetch_err}")
        else:
            # Cache raw HTML for LLM phase
            _save_html_cache(url, html)
            features = extract_html_features(html, url, domain)
            wc = features.get("X3_word_count", "?")
            t1 = features.get("T1_statistical_density", "?")
            t2 = features.get("T2_question_heading_match", "?")
            t3 = features.get("T3_structured_data", "?")
            t4 = features.get("T4_citation_authority", "?")
            print(f"  OK  Words={wc}  T1={t1}  T2={t2}  T3={t3}  T4={t4}")

        features["fetch_status_code"] = status
        features["fetch_timestamp_utc"] = _utcnow_iso()
        existing[url] = features

        # Save progress every 10 URLs
        if i % 10 == 0:
            print(f"  [progress saved: {len(existing)} URLs]")
            # caller handles save via returned dict

        time.sleep(random.uniform(rate_min, rate_max))

    return existing


# ── Phase: Open PageRank (Domain Authority) ─────────────────────────────

def run_phase_pagerank(features: dict[str, dict], api_key: str) -> dict[str, dict]:
    """Add X1_domain_authority via Open PageRank API (free, batch 100)."""
    if not api_key:
        print("\n  OPENPAGERANK_KEY not set. Sign up free at https://www.domcop.com/openpagerank/auth/signup")
        print("  Then add OPENPAGERANK_KEY=<key> to .env.local")
        return features

    domains = sorted({f.get("domain", "") for f in features.values()} - {""})
    # Skip already-done domains
    done = {f.get("domain", "") for f in features.values() if f.get("X1_domain_authority") is not None}
    remaining = [d for d in domains if d not in done]

    print(f"\nPhase: Open PageRank (Domain Authority)")
    print(f"  Unique domains: {len(domains)}")
    print(f"  Already done:   {len(done)}")
    print(f"  Remaining:      {len(remaining)}")

    cache = {}
    # Batch 100 domains per request
    for batch_start in range(0, len(remaining), 100):
        batch = remaining[batch_start:batch_start + 100]
        print(f"  Batch {batch_start // 100 + 1}: {len(batch)} domains...", end=" ")
        try:
            params = [("domains[]", d) for d in batch]
            resp = requests.get(
                "https://openpagerank.com/api/v1.0/getPageRank",
                params=params,
                headers={"API-OPR": api_key},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("response", []):
                domain = item.get("domain", "")
                pr = item.get("page_rank_decimal")
                rank = item.get("rank")
                if domain:
                    cache[domain] = {
                        "authority": pr,
                        "global_rank": int(rank) if rank else None,
                    }
            print(f"OK ({len(cache)} total)")
        except Exception as e:
            print(f"Error: {str(e)[:80]}")
        time.sleep(1)

    # Attach to features
    for url, feat in features.items():
        d = feat.get("domain", "")
        if d in cache:
            feat["X1_domain_authority"] = cache[d]["authority"]
            feat["X1_global_rank"] = cache[d]["global_rank"]

    return features


# ── Phase: WHOIS Domain Age ─────────────────────────────────────────────

def run_phase_whois(features: dict[str, dict]) -> dict[str, dict]:
    """Add X2_domain_age_years via WHOIS (creation date -> years)."""
    try:
        import whois as whois_lib
    except ImportError:
        print("\n  python-whois not installed. Run: pip install python-whois")
        return features

    domains = sorted({f.get("domain", "") for f in features.values()} - {""})
    done = {f.get("domain", "") for f in features.values() if f.get("X2_domain_age_years") is not None}
    remaining = [d for d in domains if d not in done]

    print(f"\nPhase: WHOIS Domain Age")
    print(f"  Unique domains: {len(domains)}")
    print(f"  Remaining:      {len(remaining)}")

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
                    from datetime import timezone as tz
                    creation = creation.replace(tzinfo=tz.utc)
                age_years = round((now - creation).days / 365.25, 1)
                cache[domain] = {"date": creation.isoformat(), "years": age_years}
                print(f"-> {age_years} years ({creation.date()})")
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


# ── Phase 3: PageSpeed LCP ──────────────────────────────────────────────

def run_phase3_lcp(features: dict[str, dict], api_key: str) -> dict[str, dict]:
    """Add X4_lcp_ms via Google PageSpeed Insights API."""
    urls = [url for url, f in features.items() if not f.get("error")]
    print(f"\nPhase 3: PageSpeed Insights LCP")
    print(f"  URLs to check: {len(urls)}")

    if not api_key:
        print("  GOOGLE_API_KEY not set, skipping.")
        return features

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] {url[:70]}...", end=" ")
        try:
            resp = requests.get(
                "https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
                params={
                    "url": url,
                    "key": api_key,
                    "category": "performance",
                    "strategy": "mobile",
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            lcp = (data
                   .get("lighthouseResult", {})
                   .get("audits", {})
                   .get("largest-contentful-paint", {})
                   .get("numericValue"))
            features[url]["X4_lcp_ms"] = round(float(lcp), 1) if lcp is not None else None
            print(f"-> {features[url]['X4_lcp_ms']} ms")
        except Exception as e:
            features[url]["X4_lcp_ms"] = None
            print(f"-> error: {str(e)[:60]}")
        time.sleep(1)

    return features


# ── Phase: Keyword Difficulty Estimation ─────────────────────────────────

def compute_keyword_difficulty(rankings_csv: Path, features: dict[str, dict]) -> dict[str, float]:
    """Estimate X8_keyword_difficulty from average domain authority of top results.

    For each keyword, takes the mean Open PageRank score of the top-10
    ranked domains. This is a standard KD estimation approach: harder
    keywords have stronger domains in the top results.

    Returns {keyword: difficulty_score} dict.
    """
    from collections import defaultdict

    kw_authorities = defaultdict(list)
    with open(rankings_csv) as f:
        for row in csv.DictReader(f):
            keyword = row.get("keyword", "")
            url = row.get("url", "").strip()
            domain = row.get("domain", "").strip()
            lookup = url if url else f"https://{domain}/"
            feat = features.get(lookup, {})
            auth = feat.get("X1_domain_authority")
            if auth is not None and keyword:
                kw_authorities[keyword].append(float(auth))

    kw_difficulty = {}
    for keyword, auths in kw_authorities.items():
        if auths:
            kw_difficulty[keyword] = round(sum(auths) / len(auths), 2)

    return kw_difficulty


# ── Phase LLM: LLM Treatment Extraction ─────────────────────────────────

def run_phase_llm(features: dict[str, dict], hf_token: str) -> dict[str, dict]:
    """Use LLM to evaluate T1-T4 from cached HTML for each URL."""
    from huggingface_hub import InferenceClient

    if not hf_token:
        print("\n  HF_TOKEN not set, skipping LLM phase.")
        return features

    # Only process URLs that succeeded in Phase 1 and haven't been LLM-analyzed
    urls_to_process = [
        url for url, f in features.items()
        if not f.get("error") and f.get("T1_llm_statistical_density") is None
    ]

    print(f"\nPhase LLM: LLM Treatment Extraction")
    print(f"  Model:          {LLM_MODEL_ID}")
    print(f"  URLs to analyze: {len(urls_to_process)}")

    if not urls_to_process:
        print("  Nothing to do.")
        return features

    client = InferenceClient(token=hf_token)

    for i, url in enumerate(urls_to_process, 1):
        feat = features[url]
        domain = feat.get("domain", "")

        print(f"\n[{i}/{len(urls_to_process)}] {url} ({domain})")

        # Load HTML from cache
        html = _load_html_cache(url)
        if not html:
            print("  No cached HTML, skipping.")
            feat["llm_error"] = "no_cached_html"
            continue

        # Build digest and call LLM
        digest = build_page_digest(html, url, domain)
        llm_result = llm_extract_treatments(digest, client, LLM_MODEL_ID)

        # Merge LLM results into features
        for key, val in llm_result.items():
            feat[key] = val

        t1l = feat.get("T1_llm_statistical_density", "?")
        t2l = feat.get("T2_llm_question_heading", "?")
        t3l = feat.get("T3_llm_structured_data", "?")
        t4l = feat.get("T4_llm_citation_authority", "?")
        err = feat.get("llm_error", "")
        if err:
            print(f"  LLM Error: {err}")
        else:
            print(f"  LLM  T1={t1l}  T2={t2l}  T3={t3l}  T4={t4l}")

        time.sleep(random.uniform(0.5, 1.5))

    return features


# ── Output ───────────────────────────────────────────────────────────────

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
    "X3_word_count", "X4_lcp_ms",
    "X6_readability", "X7_internal_links", "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt", "X10_https",
]


def save_features_json(features: dict[str, dict], filepath: Path):
    with open(filepath, "w") as f:
        json.dump({
            "created_utc": _utcnow_iso(),
            "total_urls": len(features),
            "features": features,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {filepath}")


def save_features_csv(features: dict[str, dict], filepath: Path):
    cols = ["url", "domain"] + FEATURE_COLS + ["fetch_status_code", "error"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for url in sorted(features):
            writer.writerow(features[url])
    print(f"  Saved CSV:  {filepath} ({len(features)} rows)")


def merge_dml_dataset(rankings_csv: Path, features: dict[str, dict],
                      kw_difficulty: dict[str, float], output: Path):
    """Join page features + keyword difficulty to the rankings CSV."""
    with open(rankings_csv) as fin:
        reader = csv.DictReader(fin)
        ranking_cols = reader.fieldnames

        rows = []
        for row in reader:
            url = row.get("url", "").strip()
            domain = row.get("domain", "").strip()
            keyword = row.get("keyword", "")
            lookup = url if url else f"https://{domain}/"
            feat = features.get(lookup, {})

            out = dict(row)
            for col in FEATURE_COLS:
                if col == "X8_keyword_difficulty":
                    out[col] = kw_difficulty.get(keyword, "")
                else:
                    out[col] = feat.get(col, "")
            out["fetch_error"] = feat.get("error", "no_features" if not feat else "")
            rows.append(out)

    out_cols = ranking_cols + FEATURE_COLS + ["fetch_error"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved DML:  {output} ({len(rows)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape page features for DML")
    parser.add_argument("--input", type=str, default="results/all_results_searxng.csv")
    parser.add_argument("--llm", action="store_true", help="LLM treatment extraction (T1-T4)")
    parser.add_argument("--pagerank", action="store_true", help="X1: Domain authority via Open PageRank")
    parser.add_argument("--whois", action="store_true", help="X2: Domain age via WHOIS")
    parser.add_argument("--lcp", action="store_true", help="X4: LCP via PageSpeed Insights")
    parser.add_argument("--all", action="store_true", help="Enable all optional phases")
    parser.add_argument("--max-urls", type=int, default=0, help="Limit URLs (0=all)")
    parser.add_argument("--rate-min", type=float, default=2.0)
    parser.add_argument("--rate-max", type=float, default=5.0)
    args = parser.parse_args()

    # --all enables everything
    if args.all:
        args.llm = args.pagerank = args.whois = args.lcp = True

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    # Derive output filenames from input
    tag = input_path.stem.replace("all_results", "").lstrip("_") or "default"
    prog_path = RESULTS_DIR / f"page_features_progress_{tag}.json"
    json_path = RESULTS_DIR / f"page_features_{tag}.json"
    csv_path = RESULTS_DIR / f"page_features_{tag}.csv"
    dml_path = RESULTS_DIR / f"dml_dataset_{tag}.csv"

    print(f"Input:    {input_path}")
    print(f"Progress: {prog_path}")
    phases = ["HTML"]
    if args.llm: phases.append("LLM")
    if args.pagerank: phases.append("PageRank")
    if args.whois: phases.append("WHOIS")
    if args.lcp: phases.append("LCP")
    print(f"Phases:   {' → '.join(phases)}")

    rows = load_input_csv(input_path)
    url_list = build_url_list(rows)
    features = load_progress(prog_path)

    session = _make_session()

    # Phase 1: HTML features (always)
    features = run_phase1(url_list, features, session, args.max_urls,
                          args.rate_min, args.rate_max)
    save_progress(prog_path, features)

    # Phase: LLM treatment extraction
    if args.llm:
        features = run_phase_llm(features, HF_TOKEN)
        save_progress(prog_path, features)

    # Phase: Open PageRank (domain authority)
    if args.pagerank:
        features = run_phase_pagerank(features, OPENPAGERANK_KEY)
        save_progress(prog_path, features)

    # Phase: WHOIS domain age
    if args.whois:
        features = run_phase_whois(features)
        save_progress(prog_path, features)

    # Phase: PageSpeed LCP
    if args.lcp:
        features = run_phase3_lcp(features, GOOGLE_API_KEY)
        save_progress(prog_path, features)

    # Compute keyword difficulty from domain authority data
    kw_difficulty = compute_keyword_difficulty(input_path, features)
    if kw_difficulty:
        print(f"\nX8 Keyword Difficulty (from avg domain authority of top results):")
        for kw in sorted(kw_difficulty, key=kw_difficulty.get, reverse=True)[:10]:
            print(f"  {kw:45s} KD={kw_difficulty[kw]}")
        if len(kw_difficulty) > 10:
            print(f"  ... and {len(kw_difficulty) - 10} more")

    # Save outputs
    print(f"\n{'='*60}")
    save_features_json(features, json_path)
    save_features_csv(features, csv_path)
    merge_dml_dataset(input_path, features, kw_difficulty, dml_path)

    # Summary
    ok = sum(1 for f in features.values() if not f.get("error"))
    fail = len(features) - ok
    print(f"\n{'='*60}")
    print(f"  Total URLs:   {len(features)}")
    print(f"  Successful:   {ok}")
    print(f"  Failed:       {fail}")
    print(f"  DML dataset:  {dml_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
