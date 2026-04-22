"""HTML-based feature extraction for DML causal inference.

All functions accept either raw HTML (str) or a BeautifulSoup object.
They return simple scalar values (int, float, bool, None).
None indicates the feature could not be computed.

Also provides LLM-based treatment extraction: the LLM reads a condensed
page digest (headings, body sample, JSON-LD, outbound links) and returns
structured T1-T4 judgments.
"""

import re
import json
from urllib.parse import urlparse

import tldextract
import textstat
from bs4 import BeautifulSoup, Comment


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _extract_body_text(soup: BeautifulSoup) -> str:
    """Extract visible body text, stripping non-content elements."""
    # Work on a copy so we don't mutate the caller's soup
    from copy import copy
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


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


# ── Treatment Variables ──────────────────────────────────────────────────

_STAT_PATTERNS = [
    re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'),    # 1,234,567
    re.compile(r'\b\d+\.?\d*%'),                           # 45%, 99.9%
    re.compile(r'\b(?:19|20)\d{2}\b'),                     # years 1900-2099
    re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'),  # dates 01/15/2024
    re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),          # $1,234.56
    re.compile(r'\b\d+(?:\.\d+)?[BMKbmk]\b'),             # 5M, 2.5B
]


def t1_statistical_density(body_text: str) -> float | None:
    """T1: Unique numbers/percentages/dates per 500 words."""
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
    """T2: Binary — 1 if any H2/H3 starts with a question word."""
    for heading in soup.find_all(["h2", "h3"]):
        if _QUESTION_RE.match(heading.get_text(strip=True)):
            return 1
    return 0


def t3_structured_data_presence(soup: BeautifulSoup) -> int:
    """T3: Binary — 1 if JSON-LD with @type FAQ/Product/HowTo found."""
    target_types = {"faqpage", "faq", "product", "howto"}

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if _check_ld_type(data, target_types):
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


_AUTHORITY_SUFFIXES = {"edu", "gov", "gov.uk", "ac.uk", "mil"}
_AUTHORITY_DOMAINS = {
    "wikipedia.org", "scholar.google.com", "ncbi.nlm.nih.gov",
    "arxiv.org", "nature.com", "sciencedirect.com", "ieee.org",
    "acm.org", "researchgate.net", "pubmed.ncbi.nlm.nih.gov",
}


def t4_external_citation_authority(soup: BeautifulSoup, page_domain: str) -> int:
    """T4: Count of outbound links to .edu/.gov/academic domains."""
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

            if ext.suffix in _AUTHORITY_SUFFIXES:
                count += 1
            elif link_domain in _AUTHORITY_DOMAINS:
                count += 1
        except Exception:
            continue

    return count


# ── Confounder Variables ─────────────────────────────────────────────────

def x3_word_count(body_text: str) -> int | None:
    """X3: Total word count of body text."""
    if not body_text.strip():
        return None
    return len(body_text.split())


def x6_flesch_kincaid(body_text: str) -> float | None:
    """X6: Flesch-Kincaid readability grade. None if <100 words."""
    if not body_text.strip() or len(body_text.split()) < 100:
        return None
    try:
        return round(textstat.flesch_kincaid_grade(body_text), 2)
    except Exception:
        return None


def x7_internal_link_count(soup: BeautifulSoup, page_domain: str) -> int:
    """X7: Count of internal links (same domain + relative)."""
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        # Relative links are internal
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
    """X7B: Count of outbound (external) links."""
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
    """X9: Count of <img> tags with non-empty alt text."""
    return sum(1 for img in soup.find_all("img") if img.get("alt", "").strip())


def x10_https_status(url: str) -> int:
    """X10: 1 if HTTPS, 0 otherwise."""
    return 1 if url.lower().startswith("https://") else 0


# ── Combined Extraction ──────────────────────────────────────────────────

def extract_html_features(html: str, url: str, domain: str) -> dict:
    """Extract all HTML-based features from a page.

    Returns dict with T1-T4, X3, X6, X7, X9, X10, error.
    """
    result = {
        "url": url,
        "domain": domain,
        "T1_statistical_density": None,
        "T2_question_heading_match": None,
        "T3_structured_data": None,
        "T4_citation_authority": None,
        "X3_word_count": None,
        "X6_readability": None,
        "X7_internal_links": None,
        "X7B_outbound_links": None,
        "X9_images_with_alt": None,
        "X10_https": x10_https_status(url),
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


# ── LLM-Based Treatment Extraction ──────────────────────────────────────

def build_page_digest(html: str, url: str, domain: str, max_body_chars: int = 3000) -> str:
    """Build a condensed page digest for LLM analysis.

    Includes: URL, headings (H1-H3), body text sample, JSON-LD types,
    and outbound links. Capped to fit within LLM context.
    """
    soup = _get_soup(html)

    parts = [f"URL: {url}", f"Domain: {domain}", ""]

    # Headings
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(f"  <{tag.name}> {text}")
    if headings:
        parts.append("HEADINGS:")
        parts.extend(headings[:40])  # cap at 40 headings
    else:
        parts.append("HEADINGS: (none found)")
    parts.append("")

    # Body text sample
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

    # JSON-LD structured data
    jsonld_types = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            _collect_ld_types(data, jsonld_types)
        except (json.JSONDecodeError, TypeError):
            continue
    if jsonld_types:
        parts.append(f"JSON-LD TYPES: {', '.join(jsonld_types)}")
    else:
        parts.append("JSON-LD TYPES: (none)")
    parts.append("")

    # Outbound links
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
        parts.extend(outbound[:50])  # cap at 50
        if len(outbound) > 50:
            parts.append(f"  ... and {len(outbound) - 50} more")
    else:
        parts.append("OUTBOUND LINKS: (none)")

    return "\n".join(parts)


def _collect_ld_types(data, types_list: list):
    """Recursively collect @type values from JSON-LD."""
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
    """Use an LLM to evaluate T1-T4 from a page digest.

    Args:
        digest: condensed page representation from build_page_digest()
        client: huggingface_hub.InferenceClient instance
        model_id: HF model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)

    Returns dict with T1-T4 LLM values and reasoning, or error.
    """
    result = {
        "T1_llm_statistical_density": None,
        "T2_llm_question_heading": None,
        "T3_llm_structured_data": None,
        "T4_llm_citation_authority": None,
        "T1_reasoning": None,
        "T2_reasoning": None,
        "T3_reasoning": None,
        "T4_reasoning": None,
        "llm_error": None,
    }

    prompt = _LLM_TREATMENT_PROMPT.format(digest=digest)

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            max_tokens=800,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)

        for key in ["T1_llm_statistical_density", "T2_llm_question_heading",
                     "T3_llm_structured_data", "T4_llm_citation_authority",
                     "T1_reasoning", "T2_reasoning", "T3_reasoning", "T4_reasoning"]:
            if key in parsed:
                result[key] = parsed[key]

    except json.JSONDecodeError as e:
        result["llm_error"] = f"json_parse: {str(e)[:100]}"
    except Exception as e:
        result["llm_error"] = f"llm_error: {str(e)[:150]}"

    return result
