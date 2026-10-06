"""Page, URL, domain and keyword features for the funnel study (pure functions).

Feature blocks and who can see them (``analysis/docs/funnel_study.md``):
  A3 snippet surface (visible text), A4 URL string (visible to the generator only),
  B search (stored engine position, SearXNG score/engine count: used by the frozen search only),
  C1 off-page domain signals (DataForSEO domain data, Open PageRank, llms.txt, brand lists, Google rank),
  C2 page body from raw HTML (seen by no component: negative controls).
Nothing here reads the clock: page dates are measured against a fixed reference date.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REFERENCE_DATE = datetime(2026, 4, 15, tzinfo=timezone.utc)  # the snapshot was collected 2026-04-15/16
QUESTION_WORDS = ("what", "how", "why", "when", "where", "which", "who", "can", "does", "is", "are", "should", "will", "do")
USER_CONTENT = {"reddit.com", "quora.com", "medium.com", "youtube.com", "linkedin.com", "facebook.com", "x.com",
                "twitter.com", "stackexchange.com", "stackoverflow.com", "github.com", "substack.com", "tiktok.com",
                "instagram.com", "pinterest.com"}
CHALLENGE_MARKERS = ("captcha", "are you a robot", "just a moment", "attention required", "access denied",
                     "enable javascript and cookies", "request unsuccessful", "403 forbidden", "verify you are human",
                     "checking your browser")
_TLD = None


def _extractor():
    global _TLD
    if _TLD is None:
        import tldextract
        _TLD = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())  # offline bundled list: reproducible
    return _TLD


# ---------------------------------------------------------------- URLs and domains

def decode_bing_ad(url: str) -> str | None:
    """Advertiser landing URL of a Bing ad redirect (``bing.com/aclick?...&u=a1<base64>``), else None."""
    parts = urlsplit(url)
    if not (parts.hostname or "").endswith("bing.com") or not parts.path.startswith("/aclick"):
        return None
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "u" and value.startswith("a1"):
            payload = value[2:]
            try:
                decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return None
            return decoded if decoded.startswith(("http://", "https://")) else None
    return None


def normalize_url(url: str) -> str:
    """Join key for URLs: lower-case scheme and host, no ``www.``, no default port, no fragment,
    no trailing slash except at the root, query kept without Google's ``srsltid``."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    if parts.port and not ((scheme == "http" and parts.port == 80) or (scheme == "https" and parts.port == 443)):
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "srsltid"])
    return urlunsplit((scheme, host, path, query, ""))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def registrable_domain(url: str, known: dict | None = None) -> tuple[str, str]:
    """(domain, source). Ad redirects resolve to the advertiser; a URL known from experiment 1 keeps
    that experiment's domain; otherwise the offline public-suffix list; IP hosts keep the host."""
    target = decode_bing_ad(url)
    if target:
        domain, source = registrable_domain(target, known)
        return domain, f"ad_target:{source}"
    if known and url in known and known[url]:
        return str(known[url]).lower(), "experiment1"
    host = (urlsplit(url).hostname or "").lower()
    if _is_ip(host):
        return host, "ip_host"
    parts = _extractor()(host)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower(), "public_suffix_list"
    return host, "host"


def url_features(url: str, domain: str) -> dict:
    """Block A4: properties of the URL string (the generator sees the URL; the reranker does not)."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    bare = host[4:] if host.startswith("www.") else host
    path = [p for p in parts.path.split("/") if p]
    suffix = domain.rsplit(".", 1)[-1] if "." in domain else ""
    return {
        "url_https": int(parts.scheme == "https"),
        "url_path_depth": len(path),
        "url_length": len(url),
        "url_has_query": int(bool(parts.query)),
        "url_subdomain": int(bare != domain and not _is_ip(host)),
        "url_tld_com": int(suffix == "com"),
        "url_tld_org": int(suffix == "org"),
        "url_tld_edu_gov": int(domain.endswith((".edu", ".gov")) or ".gov." in domain or ".edu." in domain or ".ac." in domain),
        "url_user_content": int(domain in USER_CONTENT),
        "url_wikipedia": int(domain == "wikipedia.org"),
        "url_ad_redirect": int(decode_bing_ad(url) is not None),
    }


# ---------------------------------------------------------------- visible snippet text (block A3)

_NUMBER = re.compile(r"\d")
_PERCENT = re.compile(r"\d\s?%")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_CURRENCY = re.compile(r"[$€£]\s?\d")
_LISTICLE = re.compile(r"^\s*(top\s+)?\d+\b|\b(top|best)\s+\d+\b", re.I)


def snippet_features(title: str, snippet: str, domain: str, glued_titles: int) -> dict:
    text = f"{title} {snippet}"
    first = (title.strip().split() or [""])[0].casefold()
    name = domain.split(".")[0].casefold() if domain else ""
    return {
        "snip_title_chars": len(title),
        "snip_text_chars": len(snippet),
        "snip_text_words": len(snippet.split()),
        "snip_digits": len(_NUMBER.findall(text)),
        "snip_percent": int(bool(_PERCENT.search(text))),
        "snip_year": int(bool(_YEAR.search(text))),
        "snip_currency": int(bool(_CURRENCY.search(text))),
        "snip_title_question": int(title.strip().endswith("?") or first in QUESTION_WORDS),
        "snip_title_listicle": int(bool(_LISTICLE.search(title))),
        "snip_names_domain": int(len(name) >= 3 and name in text.casefold()),
        "snip_glued": int(glued_titles > 0),
    }


# ---------------------------------------------------------------- page body (block C2)

def usable_html(html: str | None, size: int) -> tuple[bool, str]:
    if html is None:
        return False, "missing"
    if size < 2048:
        return False, "too_small"
    head = html[:20000].casefold()
    if any(marker in head for marker in CHALLENGE_MARKERS) and len(head) < 20000:
        return False, "bot_challenge"
    return True, "ok"


def latest_page_date(soup, body_text: str):
    """Most recent date in date-like meta tags, JSON-LD date fields or <time datetime>; else the
    first plausible date in the body text. Same sources as ``features.extract_t6_freshness``."""
    import json
    from analysis.interpretability.pipeline import features as page

    found = []
    for meta in soup.find_all("meta"):
        name = (meta.get("name", "") or meta.get("property", "") or "").lower()
        if any(dn in name for dn in ("date", "published", "modified", "time")):
            dt = page._parse_date_str(meta.get("content", "") or "")
            if dt:
                found.append(dt)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for key in ("datePublished", "dateModified", "dateCreated"):
                if data.get(key):
                    dt = page._parse_date_str(str(data[key]))
                    if dt:
                        found.append(dt)
    for tag in soup.find_all("time"):
        if tag.get("datetime"):
            dt = page._parse_date_str(tag.get("datetime"))
            if dt:
                found.append(dt)
    if not found:
        for pattern in page._DATE_PATTERNS:
            for match in pattern.finditer(body_text[:5000]):
                dt = page._parse_date_str(match.group(1))
                if dt and dt.year >= 2015 and dt <= REFERENCE_DATE:
                    found.append(dt)
                    break
            if found:
                break
    return max(found) if found else None


def freshness_bucket(age_days: float | None) -> int:
    """The experiment-1 ordinal (0-4) at the reference date."""
    if age_days is None:
        return 0
    for cutoff, value in ((0, 4), (180, 4), (365, 3), (730, 2), (1825, 1)):
        if age_days <= cutoff:
            return value
    return 0


def body_features(html: str, link_domain: str, snippet: str) -> dict:
    """Block C2 from one HTML copy, with the experiment-1 extractors and a fixed reference date."""
    from analysis.interpretability.pipeline import features as page

    soup = page._get_soup(html)
    body = page._extract_body_text(soup)
    newest = latest_page_date(soup, body)
    age = None if newest is None else (REFERENCE_DATE - newest).total_seconds() / 86400
    tokens = [t for t in re.findall(r"[\w-]+", snippet.casefold()) if len(t) >= 4]
    folded = body.casefold()
    return {
        "body_stats_density": page.extract_t1b_stats_density(body),
        "body_question_headings": page.extract_t2a_question_headings(soup),
        "body_modularity": page.extract_t2b_structural_modularity(soup),
        "body_structured_data": page.extract_t3_structured_data(soup),
        "body_ext_citations": page.extract_t4a_ext_citations_any(soup, link_domain),
        "body_auth_citations": page.extract_t4b_auth_citations(soup, link_domain),
        "body_word_count": page.extract_word_count(body),
        "body_readability": page.extract_readability(body),
        "body_internal_links": page.extract_internal_links(soup, link_domain),
        "body_outbound_links": page.extract_outbound_links(soup, link_domain),
        "body_images_alt": page.extract_images_alt(soup),
        "body_latest_date": newest.date().isoformat() if newest else None,
        "body_age_days": age,
        "body_freshness": freshness_bucket(age),
        "body_has_date": int(newest is not None),
        "body_contains_snippet": int(bool(tokens) and sum(t in folded for t in tokens) >= 0.5 * len(tokens)),
        "body_words_ok": int(len(body.split()) >= 50),
    }


# ---------------------------------------------------------------- keywords and domains

def keyword_moderators(main_intent: str | None, difficulty: float | None, terciles: tuple[float, float]) -> dict:
    return {"kw_commercial_or_transactional": int(main_intent in ("commercial", "transactional")),
            "kw_difficulty_tercile": None if difficulty is None else int(difficulty > terciles[0]) + int(difficulty > terciles[1])}


def constant_value(values) -> tuple[object, int]:
    """Most common non-null value of a domain-level field and the number of distinct values seen."""
    from collections import Counter
    seen = Counter(v for v in values if v is not None and v == v)
    if not seen:
        return None, 0
    return seen.most_common(1)[0][0], len(seen)
