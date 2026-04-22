#!/usr/bin/env python3
"""Extract new treatment and confounder features for DML analysis.

Post-processing step after gather_data.py. Reads experiment JSON, HTML cache,
url_mapping, and existing dataset to produce enriched features CSV.

Inputs:
  - results/searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json
  - data/url_mapping.csv
  - results/html_cache/*.html
  - data/geodml_dataset.csv (carry-forward features)

Output:
  - pipeline/intermediate/features_new.csv

Usage:
  python pipeline/extract_features.py
  python pipeline/extract_features.py --moz-api-key YOUR_KEY
  python pipeline/extract_features.py --refresh-moz
"""

import argparse
import hashlib
import json
import os
import re
import sys
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import textstat
import tldextract
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env.local")

EXPERIMENT_JSON = PROJECT_ROOT / "results" / "searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json"
URL_MAPPING_CSV = PROJECT_ROOT / "data" / "url_mapping.csv"
HTML_CACHE_DIR = PROJECT_ROOT / "results" / "html_cache"
EXISTING_DATASET = PROJECT_ROOT / "data" / "geodml_dataset.csv"
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"
OUTPUT_CSV = INTERMEDIATE_DIR / "features_new.csv"
EMBEDDINGS_CACHE = INTERMEDIATE_DIR / "embeddings.npz"
MOZ_CACHE = INTERMEDIATE_DIR / "moz_data.csv"
VALIDATION_REPORT = INTERMEDIATE_DIR / "validation_report.txt"

# ── Well-known brand domains (B2B SaaS) ──────────────────────────────────────

BRAND_DOMAINS = {
    "salesforce.com", "hubspot.com", "microsoft.com", "oracle.com",
    "sap.com", "adobe.com", "google.com", "ibm.com", "cisco.com",
    "servicenow.com", "workday.com", "zendesk.com", "atlassian.com",
    "slack.com", "zoom.us", "dropbox.com", "shopify.com", "twilio.com",
    "datadog.com", "snowflake.com", "cloudflare.com", "okta.com",
    "pagerduty.com", "elastic.co", "mongodb.com", "confluent.io",
    "hashicorp.com", "databricks.com", "stripe.com", "brevo.com",
    "mailchimp.com", "intercom.com", "freshworks.com", "zoho.com",
    "monday.com", "asana.com", "notion.so", "airtable.com",
    "clickup.com", "smartsheet.com", "wix.com", "squarespace.com",
    "bigcommerce.com", "klaviyo.com", "semrush.com", "ahrefs.com",
    "moz.com", "hootsuite.com", "buffer.com", "sproutsocial.com",
    "canva.com", "figma.com", "webflow.com", "unbounce.com",
    "activecampaign.com", "drift.com", "gong.io", "outreach.io",
    "salesloft.com", "docusign.com", "pandadoc.com", "calendly.com",
    "loom.com", "vidyard.com", "wistia.com", "typeform.com",
    "surveymonkey.com", "qualtrics.com", "amplitude.com", "mixpanel.com",
    "segment.com", "braze.com", "iterable.com", "appcues.com",
    "pendo.io", "gainsight.com", "totango.com", "churnzero.com",
    "chargebee.com", "recurly.com", "zuora.com", "bill.com",
    "xero.com", "quickbooks.intuit.com", "netsuite.com", "sage.com",
    "gusto.com", "rippling.com", "bamboohr.com", "paylocity.com",
    "paycom.com", "deel.com", "remote.com", "oysterhr.com",
    "lever.co", "greenhouse.io", "ashbyhq.com", "gem.com",
    "procore.com", "autodesk.com", "plangrid.com", "buildertrend.com",
    "toast.com", "lightspeed.com", "mindbodyonline.com",
    "zenoti.com", "veeva.com", "clio.com", "appfolio.com",
}

EARNED_DOMAINS = {
    # ── Software review / comparison platforms ────────────────────────────────
    "g2.com", "capterra.com", "trustradius.com", "softwareadvice.com",
    "getapp.com", "gartner.com", "forrester.com", "idc.com",
    "solutionsreview.com", "selecthub.com", "betterbuys.com",
    "peerspot.com", "sourceforge.net", "crozdesk.com", "financesonline.com",
    "goodfirms.co", "trustpilot.com", "alternativeto.net", "softwaresuggest.com",
    "technologyadvice.com", "saashub.com", "clutch.co", "stackshare.io",
    "featuredcustomers.com", "saasworthy.com", "betalist.com", "indiehackers.com",
    "serchen.com", "saasgenius.com", "crowdreviews.com", "f6s.com",
    "startupstash.com", "saasmag.com", "softwarereviews.com", "spiceworks.com",
    "infotech.com", "toolradar.com", "selectsoftwarereviews.com",
    "discovercrm.com", "emailvendorselection.com",
    # ── Tech media / editorial ────────────────────────────────────────────────
    "techcrunch.com", "venturebeat.com", "zdnet.com", "techradar.com",
    "pcmag.com", "cnet.com", "tomsguide.com", "theverge.com", "verge.com",
    "wired.com", "arstechnica.com", "infoworld.com", "computerworld.com",
    "engadget.com", "gizmodo.com", "mashable.com", "thenextweb.com",
    "digitaltrends.com", "fastcompany.com", "techrepublic.com",
    "theregister.com", "siliconangle.com", "readwrite.com", "geekwire.com",
    "cmswire.com", "slashdot.org", "technologyreview.com", "theinformation.com",
    "gigaom.com", "makeuseof.com", "techspot.com", "tomshardware.com",
    "9to5mac.com", "bgr.com", "techdirt.com", "hackaday.com",
    "informationweek.com", "pcworld.com", "extremetech.com",
    "siliconrepublic.com", "geekflare.com", "rtings.com",
    # ── Business media ────────────────────────────────────────────────────────
    "forbes.com", "businessinsider.com", "entrepreneur.com", "inc.com",
    "nytimes.com", "wsj.com", "bloomberg.com", "reuters.com",
    "fortune.com", "economist.com", "adweek.com", "marketwatch.com",
    "cnbc.com", "ft.com", "axios.com", "businesswire.com", "prnewswire.com",
    "globenewswire.com",
    # ── Marketing / AdTech / MarTech publications ─────────────────────────────
    "adage.com", "digiday.com", "marketingdive.com", "adexchanger.com",
    "martech.org", "thedrum.com", "mediapost.com", "chiefmarketer.com",
    "marketingweek.com", "searchengineland.com", "searchenginejournal.com",
    "marketingbrew.com", "brandingmag.com", "martechseries.com",
    "socialmediatoday.com", "contentmarketinginstitute.com",
    "seroundtable.com", "moz.com",
    # ── Cybersecurity publications ────────────────────────────────────────────
    "darkreading.com", "securityweek.com", "thehackernews.com",
    "krebsonsecurity.com", "csoonline.com", "threatpost.com",
    "hackread.com", "infosecurity-magazine.com", "bleepingcomputer.com",
    "cybersecuritydive.com",
    # ── Cloud / DevOps / Infrastructure ───────────────────────────────────────
    "thenewstack.io", "devops.com", "cloudnativenow.com",
    "containerjournal.com", "cloudwards.net",
    # ── Enterprise IT / CIO / CTO ─────────────────────────────────────────────
    "cio.com", "ciodive.com", "techtarget.com",
    # ── Consulting / analyst / research firms ─────────────────────────────────
    "hbr.org", "mckinsey.com", "bain.com", "bcg.com", "deloitte.com",
    "accenture.com", "pwc.com", "kpmg.com", "ey.com",
    "oliverwyman.com", "rolandberger.com", "451research.com", "omdia.com",
    "constellationr.com", "everestgrp.com", "frost.com", "nucleusresearch.com",
    "redmonk.com", "hfsresearch.com", "canalys.com", "verdantix.com",
    "abiresearch.com", "globaldata.com",
    # ── Industry Dive network (all vertical trade pubs) ───────────────────────
    "retaildive.com", "supplychaindive.com", "hrdive.com",
    "constructiondive.com", "fooddive.com", "grocerydive.com",
    "healthcaredive.com", "manufacturingdive.com", "utilitydive.com",
    "restaurantdive.com", "bankingdive.com", "biopharmadive.com",
    "paymentsdive.com", "hoteldive.com", "wastedive.com",
    # ── Industry vertical publications ────────────────────────────────────────
    "ecommercetimes.com", "digitalcommerce360.com", "healthcareitnews.com",
    "fintechfutures.com", "pymnts.com", "businessofapps.com",
    "supplychaindigital.com", "fintechmagazine.com",
    # ── Fintech / finance media ───────────────────────────────────────────────
    "fintechweekly.com", "thefintechtimes.com", "fintech.global",
    # ── HR / workforce media ──────────────────────────────────────────────────
    "shrm.org",
    # ── AI / data science ─────────────────────────────────────────────────────
    "kdnuggets.com", "aimagazine.com",
    # ── Community / UGC / knowledge ───────────────────────────────────────────
    "wikipedia.org", "reddit.com", "quora.com", "stackexchange.com",
    "stackoverflow.com", "medium.com", "substack.com",
    "youtube.com", "producthunt.com", "crunchbase.com",
    "dev.to", "github.com", "hashnode.com", "codeproject.com",
    "hackernoon.com", "dzone.com", "sitepoint.com", "smashingmagazine.com",
    "freecodecamp.org",
    # ── News aggregators / syndication (derivative content) ───────────────────
    "news.google.com", "news.yahoo.com", "msn.com", "flipboard.com",
    "smartnews.com", "newsbreak.com", "feedly.com", "allsides.com",
    "techmeme.com", "hacker-news.firebaseio.com",
    # ── Press release / wire services ─────────────────────────────────────────
    "accesswire.com", "prweb.com",
    # ── Startup / VC ecosystem ────────────────────────────────────────────────
    "angel.co", "wellfound.com", "startupgrind.com",
    # ── Employer review / talent ──────────────────────────────────────────────
    "glassdoor.com", "builtin.com",
    # ── Design / UX / creative ────────────────────────────────────────────────
    "dribbble.com", "behance.net", "awwwards.com", "designrush.com",
    "webdesignerdepot.com",
    # ── Hosting / web tool reviews ────────────────────────────────────────────
    "hostingadvice.com", "wpbeginner.com", "top10.com",
    # ── Reference / knowledge ─────────────────────────────────────────────────
    "britannica.com", "investopedia.com", "coursereport.com",
    # ── Third-party editorial / comparison / review blogs (not SaaS vendors) ──
    "zapier.com", "omr.com", "thedigitalprojectmanager.com", "toptal.com",
    "european-alternatives.eu", "softwaretestingmaterial.com",
    "thectoclub.com", "proposal.biz", "easyreplenish.com",
    "softwareworld.co", "appsumo.com", "killerstartups.com",
    "wirecutter.com", "consumerreports.org",
}

# Expanded JSON-LD types for T3
STRUCTURED_DATA_TYPES = {
    "faqpage", "faq", "product", "howto", "softwareapplication",
    "article", "blogposting", "review", "aggregaterating",
    "offer", "itemlist", "breadcrumblist", "videoobject",
    "dataset", "course", "event", "recipe", "qapage",
}

# Authority domains for T4
AUTHORITY_SUFFIXES = {"edu", "gov", "gov.uk", "ac.uk", "mil"}
AUTHORITY_DOMAINS = {
    "wikipedia.org", "scholar.google.com", "ncbi.nlm.nih.gov",
    "arxiv.org", "nature.com", "sciencedirect.com", "ieee.org",
    "acm.org", "researchgate.net", "pubmed.ncbi.nlm.nih.gov",
    "springer.com", "wiley.com", "jstor.org", "ssrn.com",
    "nber.org", "who.int", "un.org", "worldbank.org",
    "statista.com", "pewresearch.org", "gallup.com",
}

# Filter domains for external link counting (not authority, just noise)
LINK_FILTER_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com",
    "pinterest.com", "tiktok.com", "youtube.com", "apple.com",
    "play.google.com", "apps.apple.com",
    "cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com",
    "ajax.googleapis.com", "maxcdn.bootstrapcdn.com",
}


# ── HTML Helpers ──────────────────────────────────────────────────────────────

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


def _url_to_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _load_html(url: str, url_to_file: dict, cache_dir: Path = None) -> str | None:
    """Load HTML from cache, trying url_mapping first, then direct hash."""
    if cache_dir is None:
        cache_dir = HTML_CACHE_DIR
    # Try url_mapping lookup
    cache_file = url_to_file.get(url)
    if cache_file:
        path = cache_dir / cache_file
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Try direct hash
    cache_key = _url_to_cache_key(url)
    path = cache_dir / f"{cache_key}.html"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    return None


# ── Stat patterns (reused from gather_data.py) ───────────────────────────────

_STAT_PATTERNS = [
    re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'),
    re.compile(r'\b\d+\.?\d*%'),
    re.compile(r'\b(?:19|20)\d{2}\b'),
    re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'),
    re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
    re.compile(r'\b\d+(?:\.\d+)?[BMKbmk]\b'),
]

_QUESTION_RE = re.compile(
    r'^\s*(?:what|how|why|when|where|which|who|can|does|is|are|should|will|do)\b',
    re.IGNORECASE,
)


# ── Treatment Extractors ─────────────────────────────────────────────────────

def extract_t1a_stats_present(body_text: str) -> int:
    """T1a: Binary — any statistics present."""
    if not body_text.strip():
        return 0
    for pat in _STAT_PATTERNS:
        if pat.search(body_text):
            return 1
    return 0


def extract_t1b_stats_density(body_text: str) -> float | None:
    """T1b: Float — stats per 500 words."""
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


def extract_t2a_question_headings(soup: BeautifulSoup) -> int:
    """T2a: Binary — any H2/H3 with question words or ?."""
    for heading in soup.find_all(["h2", "h3"]):
        text = heading.get_text(strip=True)
        if _QUESTION_RE.match(text) or text.endswith("?"):
            return 1
    return 0


def extract_t2b_structural_modularity(soup: BeautifulSoup) -> int:
    """T2b: Int — count of H2 + H3 headings."""
    return len(soup.find_all(["h2", "h3"]))


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


def extract_t3_structured_data(soup: BeautifulSoup) -> int:
    """T3: Binary — JSON-LD with expanded type list."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if _check_ld_type(data, STRUCTURED_DATA_TYPES):
            return 1
    return 0


def extract_t4a_ext_citations_any(soup: BeautifulSoup, page_domain: str) -> int:
    """T4a: Binary — any external links (filtered)."""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        try:
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https"):
                continue
            ext = tldextract.extract(href)
            link_domain = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
            if not link_domain or link_domain == page_domain:
                continue
            if link_domain in LINK_FILTER_DOMAINS:
                continue
            return 1
        except Exception:
            continue
    return 0


def extract_t4b_auth_citations(soup: BeautifulSoup, page_domain: str) -> int:
    """T4b: Int — count of authority domain links."""
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
            if ext.suffix in AUTHORITY_SUFFIXES or link_domain in AUTHORITY_DOMAINS:
                count += 1
        except Exception:
            continue
    return count


# ── T6: Freshness ────────────────────────────────────────────────────────────

_DATE_META_NAMES = [
    "article:published_time", "article:modified_time",
    "datePublished", "dateModified", "date", "DC.date",
    "publication_date", "last-modified",
]

_DATE_PATTERNS = [
    re.compile(r'(\d{4}-\d{2}-\d{2})'),                    # 2025-01-15
    re.compile(r'(\d{4}/\d{2}/\d{2})'),                    # 2025/01/15
    re.compile(r'(\w+ \d{1,2},?\s*\d{4})', re.IGNORECASE), # January 15, 2025
    re.compile(r'(\d{1,2} \w+ \d{4})', re.IGNORECASE),     # 15 January 2025
]

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%B %d %Y",
    "%d %B %Y", "%b %d, %Y", "%b %d %Y", "%d %b %Y",
]


def _parse_date_str(s: str) -> datetime | None:
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Try ISO format
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass
    return None


def extract_t6_freshness(soup: BeautifulSoup, body_text: str) -> int:
    """T6: Ordinal 0-4 based on content recency.

    4 = < 6 months old
    3 = 6-12 months old
    2 = 1-2 years old
    1 = 2-5 years old
    0 = > 5 years old or no date found
    """
    now = datetime.now(timezone.utc)
    dates_found = []

    # Check meta tags
    for meta in soup.find_all("meta"):
        name = (meta.get("name", "") or meta.get("property", "") or "").lower()
        content = meta.get("content", "")
        if any(dn in name for dn in ["date", "published", "modified", "time"]):
            dt = _parse_date_str(content)
            if dt:
                dates_found.append(dt)

    # Check JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for key in ["datePublished", "dateModified", "dateCreated"]:
                val = data.get(key, "")
                if val:
                    dt = _parse_date_str(str(val))
                    if dt:
                        dates_found.append(dt)

    # Check <time> elements
    for time_tag in soup.find_all("time"):
        dt_attr = time_tag.get("datetime", "")
        if dt_attr:
            dt = _parse_date_str(dt_attr)
            if dt:
                dates_found.append(dt)

    # Last resort: scan body text for year patterns
    if not dates_found:
        for pat in _DATE_PATTERNS:
            for match in pat.finditer(body_text[:5000]):
                dt = _parse_date_str(match.group(1))
                if dt and dt.year >= 2015 and dt <= now:
                    dates_found.append(dt)
                    break
            if dates_found:
                break

    if not dates_found:
        return 0

    # Use the most recent date
    most_recent = max(dates_found)
    age_days = (now - most_recent).days

    if age_days < 0:
        return 4  # future date likely means very recent
    if age_days <= 180:
        return 4
    if age_days <= 365:
        return 3
    if age_days <= 730:
        return 2
    if age_days <= 1825:
        return 1
    return 0


# ── T7: Source type classification ────────────────────────────────────────────

def classify_source_type(domain: str) -> tuple[int, int, str]:
    """Classify domain as brand, earned, or other.

    Returns (is_brand, is_earned, source_type_str).
    """
    d = domain.lower().strip()
    if d in BRAND_DOMAINS:
        return 1, 0, "brand"
    if d in EARNED_DOMAINS:
        return 0, 1, "earned"
    return 0, 0, "other"


# ── Confounder Extractors ─────────────────────────────────────────────────────

def conf_title_has_kw(title: str, keyword: str) -> int:
    """S6: Binary — does title contain core keyword term."""
    if not title or not keyword:
        return 0
    # Check if any word of the keyword (length >= 3) appears in title
    kw_words = [w.lower() for w in keyword.split() if len(w) >= 3]
    title_lower = title.lower()
    for w in kw_words:
        if w in title_lower:
            return 1
    return 0


def conf_brand_recog(domain: str) -> int:
    """S5: Binary — domain in well-known brand set."""
    return 1 if domain.lower().strip() in BRAND_DOMAINS else 0


# ── HTML-based confounder extractors (carry forward or re-extract) ────────────

def extract_word_count(body_text: str) -> int | None:
    if not body_text.strip():
        return None
    return len(body_text.split())


def extract_readability(body_text: str) -> float | None:
    if not body_text.strip() or len(body_text.split()) < 100:
        return None
    try:
        return round(textstat.flesch_kincaid_grade(body_text), 2)
    except Exception:
        return None


def extract_internal_links(soup: BeautifulSoup, page_domain: str) -> int:
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


def extract_outbound_links(soup: BeautifulSoup, page_domain: str) -> int:
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


def extract_images_alt(soup: BeautifulSoup) -> int:
    return sum(1 for img in soup.find_all("img") if img.get("alt", "").strip())


# ── Moz API ──────────────────────────────────────────────────────────────────

def fetch_moz_data(domains: list[str], api_key: str) -> dict[str, dict]:
    """Fetch domain authority, backlinks, referring domains from Moz Links API v2.

    api_key can be:
      - "access_id:secret_key" format (Basic Auth)
      - a standalone API token (x-moz-token header)

    Returns dict: domain -> {domain_authority, backlinks, referring_domains}
    """
    import requests
    import time as _time

    results = {}

    # Determine auth method from key format
    if ":" in api_key:
        # Basic Auth: access_id:secret_key
        parts = api_key.split(":", 1)
        auth_tuple = (parts[0], parts[1])
        headers = {}
        print(f"  Auth: Basic Auth (access_id={parts[0][:12]}...)")
    else:
        # Token-based auth (post March 2024 keys)
        auth_tuple = None
        headers = {"x-moz-token": api_key}
        print(f"  Auth: x-moz-token header")

    # Moz Links API v2 supports batching up to 50 targets per request
    batch_size = 30
    total_batches = (len(domains) + batch_size - 1) // batch_size

    for batch_start in range(0, len(domains), batch_size):
        batch = domains[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"  Moz batch {batch_num}/{total_batches} ({len(batch)} domains)...", end=" ")

        try:
            resp = requests.post(
                "https://lsapi.seomoz.com/v2/url_metrics",
                json={"targets": batch},
                auth=auth_tuple,
                headers=headers,
                timeout=60,
            )

            if resp.status_code == 401:
                print(f"Auth failed (401). Check your Moz API credentials.")
                print(f"    Response: {resp.text[:200]}")
                return results
            if resp.status_code == 429:
                print(f"Rate limited (429). Waiting 30s...")
                _time.sleep(30)
                # Retry this batch once
                resp = requests.post(
                    "https://lsapi.seomoz.com/v2/url_metrics",
                    json={"targets": batch},
                    auth=auth_tuple,
                    headers=headers,
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()

            batch_results = data.get("results", [])
            for i, r in enumerate(batch_results):
                if i < len(batch):
                    domain = batch[i]
                    results[domain] = {
                        "domain_authority": r.get("domain_authority"),
                        "backlinks": r.get("external_pages_to_root_domain"),
                        "referring_domains": r.get("root_domains_to_root_domain"),
                    }

            ok_count = sum(1 for d in batch if results.get(d, {}).get("domain_authority") is not None)
            print(f"OK ({ok_count}/{len(batch)} with DA)")

        except Exception as e:
            print(f"Error: {str(e)[:100]}")
            for domain in batch:
                if domain not in results:
                    results[domain] = {
                        "domain_authority": None,
                        "backlinks": None,
                        "referring_domains": None,
                    }

        _time.sleep(5)  # respect rate limits between batches

    return results


# ── Sentence Transformer embeddings ──────────────────────────────────────────

def compute_embeddings(texts: list[str], model, batch_size: int = 64) -> np.ndarray:
    """Batch encode texts using sentence-transformers model."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── BM25 ─────────────────────────────────────────────────────────────────────

def compute_bm25_scores(keyword: str, page_texts: list[str]) -> list[float]:
    """Compute BM25 relevance of keyword against each page text."""
    from rank_bm25 import BM25Okapi

    # Tokenize documents
    tokenized = []
    for text in page_texts:
        if text and text.strip():
            tokenized.append(text.lower().split()[:5000])  # cap at 5000 tokens
        else:
            tokenized.append([""])

    if not tokenized:
        return []

    bm25 = BM25Okapi(tokenized)
    query_tokens = keyword.lower().split()
    scores = bm25.get_scores(query_tokens)
    return scores.tolist()


# ── Main extraction ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract new treatment/confounder features")
    parser.add_argument("--experiment-json", type=str, default=str(EXPERIMENT_JSON),
                        help="Path to experiment JSON")
    parser.add_argument("--moz-api-key", type=str, default=os.getenv("MOZ_API_KEY", ""),
                        help="Moz API key for domain authority/backlinks")
    parser.add_argument("--refresh-moz", action="store_true",
                        help="Re-query Moz API even if cache exists")
    parser.add_argument("--html-cache-dir", type=str, default=str(HTML_CACHE_DIR),
                        help="Path to HTML cache directory")
    parser.add_argument("--existing-dataset", type=str, default=str(EXISTING_DATASET),
                        help="Path to existing dataset CSV for carry-forward features")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV),
                        help="Path for output features CSV")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip sentence-transformer embeddings (faster for debugging)")
    args = parser.parse_args()

    # Override globals from CLI args
    html_cache_dir = Path(args.html_cache_dir)
    existing_dataset_path = Path(args.existing_dataset)
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load experiment JSON ──────────────────────────────────────────────
    json_path = Path(args.experiment_json)
    if not json_path.exists():
        print(f"Experiment JSON not found: {json_path}")
        sys.exit(1)

    print(f"Loading experiment JSON: {json_path}")
    with open(json_path) as f:
        experiment = json.load(f)

    # ── Load url_mapping ──────────────────────────────────────────────────
    url_to_file = {}
    if URL_MAPPING_CSV.exists():
        mapping_df = pd.read_csv(URL_MAPPING_CSV)
        for _, row in mapping_df.iterrows():
            url = str(row.get("url", "")).strip()
            filename = str(row.get("filename", "")).strip()
            if url and filename:
                url_to_file[url] = filename
        print(f"Loaded url_mapping: {len(url_to_file)} entries")
    else:
        print(f"Warning: url_mapping.csv not found at {URL_MAPPING_CSV}")

    # ── Load existing dataset for carry-forward ───────────────────────────
    existing_df = None
    if existing_dataset_path.exists():
        existing_df = pd.read_csv(existing_dataset_path)
        print(f"Loaded existing dataset: {len(existing_df)} rows, {len(existing_df.columns)} cols")
    else:
        print(f"Warning: existing dataset not found at {existing_dataset_path}")

    # ── Build (keyword, url) rows from experiment JSON ────────────────────
    # Fall back to keywords.jsonl when per_keyword_results is empty (resume bug)
    kw_results = experiment.get("per_keyword_results", [])
    if not kw_results:
        jsonl_path = json_path.parent / "keywords.jsonl"
        if jsonl_path.exists():
            print(f"  experiment.json has no per_keyword_results, falling back to {jsonl_path}")
            kw_results = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        kw_results.append(json.loads(line))
            print(f"  Loaded {len(kw_results)} keyword results from keywords.jsonl")

    rows_data = []
    for kw_result in kw_results:
        keyword = kw_result["query"]
        serp = kw_result.get("serp", {})
        raw_results = serp.get("raw_results", [])
        rank_changes = kw_result.get("rank_changes", [])
        ranked_results = kw_result.get("ranked_results", [])

        # Build lookup maps
        rc_by_domain = {rc["domain"]: rc for rc in rank_changes}
        serp_by_url = {r["url"]: r for r in raw_results}
        serp_by_domain = {}
        for r in raw_results:
            d = _extract_domain(r.get("url", ""))
            if d and d not in serp_by_domain:
                serp_by_domain[d] = r

        for rr in ranked_results:
            domain = rr["domain"]
            url = rr.get("url", "")
            if not url:
                url = f"https://{domain}/"

            # Get SERP data
            serp_item = serp_by_url.get(url, serp_by_domain.get(domain, {}))
            title = serp_item.get("title", "")
            snippet = serp_item.get("snippet", "")
            position = serp_item.get("position")

            # Get rank changes
            rc = rc_by_domain.get(domain, {})
            pre_rank = rc.get("pre_rank")
            post_rank = rc.get("post_rank")

            rows_data.append({
                "keyword": keyword,
                "domain": domain,
                "url": url,
                "title": title,
                "snippet": snippet,
                "serp_position": position,
                "pre_rank": pre_rank,
                "post_rank": post_rank,
            })

    print(f"\nTotal (keyword, url) pairs: {len(rows_data)}")
    if not rows_data:
        print("No data found. Exiting.")
        sys.exit(1)

    # ── Load carry-forward features from existing dataset ─────────────────
    carry_forward = {}
    if existing_df is not None:
        carry_cols = {
            "X3_word_count": "conf_word_count",
            "X6_readability": "conf_readability",
            "X7_internal_links": "conf_internal_links",
            "X7B_outbound_links": "conf_outbound_links",
            "X9_images_with_alt": "conf_images_alt",
            "X1_domain_authority": "conf_domain_authority",
        }
        for _, row in existing_df.iterrows():
            key = (row.get("keyword", ""), row.get("url", ""))
            cf = {}
            for old_col, new_col in carry_cols.items():
                if old_col in existing_df.columns:
                    val = row.get(old_col)
                    if pd.notna(val):
                        cf[new_col] = val
            if cf:
                carry_forward[key] = cf

    # ── Load HTML and extract HTML-based features ─────────────────────────
    print("\nExtracting HTML-based features...")
    html_features = {}  # url -> feature dict
    unique_urls = list({r["url"] for r in rows_data})
    url_to_domain = {r["url"]: r["domain"] for r in rows_data}

    loaded_count = 0
    failed_count = 0
    page_texts = {}  # url -> body text (for embeddings + BM25)

    for i, url in enumerate(unique_urls):
        domain = url_to_domain.get(url, _extract_domain(url))
        html = _load_html(url, url_to_file, html_cache_dir)

        if html is None:
            failed_count += 1
            html_features[url] = {
                "treat_stats_present": None,
                "treat_stats_density": None,
                "treat_question_headings": None,
                "treat_structural_modularity": None,
                "treat_structured_data": None,
                "treat_ext_citations_any": None,
                "treat_auth_citations": None,
                "treat_freshness": None,
                "conf_word_count": None,
                "conf_readability": None,
                "conf_internal_links": None,
                "conf_outbound_links": None,
                "conf_images_alt": None,
                "conf_https": 1 if url.lower().startswith("https://") else 0,
            }
            page_texts[url] = ""
            continue

        loaded_count += 1
        try:
            soup = _get_soup(html)
            body_text = _extract_body_text(soup)
            page_texts[url] = body_text

            html_features[url] = {
                "treat_stats_present": extract_t1a_stats_present(body_text),
                "treat_stats_density": extract_t1b_stats_density(body_text),
                "treat_question_headings": extract_t2a_question_headings(soup),
                "treat_structural_modularity": extract_t2b_structural_modularity(soup),
                "treat_structured_data": extract_t3_structured_data(soup),
                "treat_ext_citations_any": extract_t4a_ext_citations_any(soup, domain),
                "treat_auth_citations": extract_t4b_auth_citations(soup, domain),
                "treat_freshness": extract_t6_freshness(soup, body_text),
                "conf_word_count": extract_word_count(body_text),
                "conf_readability": extract_readability(body_text),
                "conf_internal_links": extract_internal_links(soup, domain),
                "conf_outbound_links": extract_outbound_links(soup, domain),
                "conf_images_alt": extract_images_alt(soup),
                "conf_https": 1 if url.lower().startswith("https://") else 0,
            }
        except Exception as e:
            print(f"  Error extracting from {url}: {str(e)[:100]}")
            html_features[url] = {
                "treat_stats_present": None,
                "treat_stats_density": None,
                "treat_question_headings": None,
                "treat_structural_modularity": None,
                "treat_structured_data": None,
                "treat_ext_citations_any": None,
                "treat_auth_citations": None,
                "treat_freshness": None,
                "conf_word_count": None,
                "conf_readability": None,
                "conf_internal_links": None,
                "conf_outbound_links": None,
                "conf_images_alt": None,
                "conf_https": 1 if url.lower().startswith("https://") else 0,
            }
            page_texts[url] = ""

    print(f"  HTML loaded: {loaded_count}, failed: {failed_count}")

    # ── Sentence-transformer embeddings ───────────────────────────────────
    embed_kw_sim = {}   # (keyword, url) -> {title_sim, snippet_sim, topical_comp}

    if not args.skip_embeddings:
        print("\nLoading sentence-transformer model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Collect all unique texts
        all_keywords = sorted(set(r["keyword"] for r in rows_data))
        all_titles = [r["title"] for r in rows_data]
        all_snippets = [r["snippet"] for r in rows_data]
        all_page_texts_for_embed = [page_texts.get(r["url"], "")[:2000] for r in rows_data]

        # Batch encode
        print("  Encoding keywords...")
        kw_embeddings = compute_embeddings(all_keywords, st_model)
        kw_emb_map = {kw: kw_embeddings[i] for i, kw in enumerate(all_keywords)}

        print("  Encoding titles...")
        title_embeddings = compute_embeddings(
            [t if t else "" for t in all_titles], st_model
        )

        print("  Encoding snippets...")
        snippet_embeddings = compute_embeddings(
            [s if s else "" for s in all_snippets], st_model
        )

        print("  Encoding page texts...")
        page_embeddings = compute_embeddings(
            [t if t else "" for t in all_page_texts_for_embed], st_model
        )

        # Compute similarities
        for i, row in enumerate(rows_data):
            kw_emb = kw_emb_map[row["keyword"]]
            title_sim = cosine_sim(kw_emb, title_embeddings[i]) if row["title"] else 0.0
            snippet_sim = cosine_sim(kw_emb, snippet_embeddings[i]) if row["snippet"] else 0.0
            topical_comp = cosine_sim(kw_emb, page_embeddings[i]) if page_texts.get(row["url"]) else None

            key = (row["keyword"], row["url"])
            embed_kw_sim[key] = {
                "conf_title_kw_sim": round(title_sim, 4),
                "conf_snippet_kw_sim": round(snippet_sim, 4),
                "treat_topical_comp": round(topical_comp, 4) if topical_comp is not None else None,
            }

        # Cache embeddings
        np.savez_compressed(
            EMBEDDINGS_CACHE,
            kw_embeddings=kw_embeddings,
            title_embeddings=title_embeddings,
            snippet_embeddings=snippet_embeddings,
            page_embeddings=page_embeddings,
        )
        print(f"  Cached embeddings -> {EMBEDDINGS_CACHE}")
    else:
        print("\nSkipping embeddings (--skip-embeddings)")

    # ── BM25 per-keyword ──────────────────────────────────────────────────
    print("\nComputing BM25 scores per keyword...")
    bm25_scores = {}  # (keyword, url) -> float

    # Group rows by keyword
    keyword_urls = {}
    for row in rows_data:
        kw = row["keyword"]
        if kw not in keyword_urls:
            keyword_urls[kw] = []
        keyword_urls[kw].append(row["url"])

    for kw, urls in keyword_urls.items():
        texts = [page_texts.get(u, "") for u in urls]
        scores = compute_bm25_scores(kw, texts)
        for url, score in zip(urls, scores):
            bm25_scores[(kw, url)] = round(score, 4)

    print(f"  BM25 scores computed for {len(bm25_scores)} (keyword, url) pairs")

    # ── Moz API ───────────────────────────────────────────────────────────
    moz_data = {}  # domain -> {domain_authority, backlinks, referring_domains}

    if args.moz_api_key:
        # Check cache
        if MOZ_CACHE.exists() and not args.refresh_moz:
            print(f"\nLoading Moz cache: {MOZ_CACHE}")
            moz_df = pd.read_csv(MOZ_CACHE)
            for _, row in moz_df.iterrows():
                d = str(row.get("domain", ""))
                moz_data[d] = {
                    "domain_authority": row.get("domain_authority"),
                    "backlinks": row.get("backlinks"),
                    "referring_domains": row.get("referring_domains"),
                }
            print(f"  Loaded {len(moz_data)} domains from cache")
        else:
            unique_domains = sorted(set(r["domain"] for r in rows_data))
            print(f"\nQuerying Moz API for {len(unique_domains)} domains...")

            # Test with ONE domain first
            test_domain = unique_domains[0]
            print(f"  Testing with: {test_domain}")
            test_result = fetch_moz_data([test_domain], args.moz_api_key)

            if test_result and test_result.get(test_domain, {}).get("domain_authority") is not None:
                print(f"  Test OK. Querying remaining domains...")
                moz_data = fetch_moz_data(unique_domains, args.moz_api_key)
            else:
                print(f"  Test failed or returned no data. Skipping Moz API.")

            # Cache results
            if moz_data:
                moz_rows = []
                for d, vals in moz_data.items():
                    moz_rows.append({"domain": d, **vals})
                pd.DataFrame(moz_rows).to_csv(MOZ_CACHE, index=False)
                print(f"  Cached Moz data -> {MOZ_CACHE}")
    else:
        # Try loading from cache even without API key
        if MOZ_CACHE.exists():
            moz_df = pd.read_csv(MOZ_CACHE)
            for _, row in moz_df.iterrows():
                d = str(row.get("domain", ""))
                moz_data[d] = {
                    "domain_authority": row.get("domain_authority"),
                    "backlinks": row.get("backlinks"),
                    "referring_domains": row.get("referring_domains"),
                }
            print(f"\nLoaded Moz cache (no API key): {len(moz_data)} domains")

    # ── T7: Domain classification ─────────────────────────────────────────
    unique_domains = sorted(set(r["domain"] for r in rows_data))
    print(f"\nDomain classification ({len(unique_domains)} unique domains):")

    brand_count = sum(1 for d in unique_domains if d in BRAND_DOMAINS)
    earned_count = sum(1 for d in unique_domains if d in EARNED_DOMAINS)
    other_count = len(unique_domains) - brand_count - earned_count
    print(f"  Brand:  {brand_count}")
    print(f"  Earned: {earned_count}")
    print(f"  Other:  {other_count}")

    # Print unclassified domains for review
    other_domains = [d for d in unique_domains if d not in BRAND_DOMAINS and d not in EARNED_DOMAINS]
    if other_domains:
        print(f"\n  Unclassified domains ({len(other_domains)}):")
        for d in other_domains:
            print(f"    {d}")

    # ── Assemble final dataframe ──────────────────────────────────────────
    print("\nAssembling final feature dataframe...")
    output_rows = []

    for row in rows_data:
        keyword = row["keyword"]
        url = row["url"]
        domain = row["domain"]
        key = (keyword, url)

        # HTML features
        hf = html_features.get(url, {})

        # Embedding features
        ef = embed_kw_sim.get(key, {})

        # BM25
        bm25_val = bm25_scores.get(key)

        # Carry forward
        cf = carry_forward.get(key, {})

        # T7 classification
        is_brand, is_earned, source_type = classify_source_type(domain)

        # Moz
        moz = moz_data.get(domain, {})

        out = {
            "keyword": keyword,
            "domain": domain,
            "url": url,

            # Treatments
            "treat_stats_present": hf.get("treat_stats_present"),
            "treat_stats_density": hf.get("treat_stats_density"),
            "treat_question_headings": hf.get("treat_question_headings"),
            "treat_structural_modularity": hf.get("treat_structural_modularity"),
            "treat_structured_data": hf.get("treat_structured_data"),
            "treat_ext_citations_any": hf.get("treat_ext_citations_any"),
            "treat_auth_citations": hf.get("treat_auth_citations"),
            "treat_topical_comp": ef.get("treat_topical_comp"),
            "treat_freshness": hf.get("treat_freshness"),
            "treat_source_brand": is_brand,
            "treat_source_earned": is_earned,
            "treat_source_type": source_type,

            # Confounders
            "conf_title_kw_sim": ef.get("conf_title_kw_sim"),
            "conf_snippet_kw_sim": ef.get("conf_snippet_kw_sim"),
            "conf_title_len": len(row.get("title", "") or ""),
            "conf_snippet_len": len(row.get("snippet", "") or ""),
            "conf_brand_recog": conf_brand_recog(domain),
            "conf_title_has_kw": conf_title_has_kw(row.get("title", ""), keyword),
            "conf_word_count": hf.get("conf_word_count") or cf.get("conf_word_count"),
            "conf_readability": hf.get("conf_readability") or cf.get("conf_readability"),
            "conf_internal_links": hf.get("conf_internal_links") or cf.get("conf_internal_links"),
            "conf_outbound_links": hf.get("conf_outbound_links") or cf.get("conf_outbound_links"),
            "conf_images_alt": hf.get("conf_images_alt") or cf.get("conf_images_alt"),
            "conf_bm25": bm25_val,
            "conf_https": hf.get("conf_https", 1 if url.lower().startswith("https://") else 0),
            "conf_domain_authority": moz.get("domain_authority") or cf.get("conf_domain_authority"),
            "conf_backlinks": moz.get("backlinks"),
            "conf_referring_domains": moz.get("referring_domains"),
            "conf_serp_position": row.get("pre_rank"),
        }

        output_rows.append(out)

    df = pd.DataFrame(output_rows)
    print(f"\nOutput dataframe: {df.shape}")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(output_csv_path, index=False)
    print(f"Saved -> {output_csv_path}")

    # ── Validation report ─────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("VALIDATION REPORT — extract_features.py")
    report_lines.append("=" * 70)
    report_lines.append(f"\nShape: {df.shape}")
    report_lines.append(f"Unique keywords: {df['keyword'].nunique()}")
    report_lines.append(f"Unique domains: {df['domain'].nunique()}")
    report_lines.append(f"Unique URLs: {df['url'].nunique()}")

    # Nulls per column
    report_lines.append("\nNulls per column:")
    for col in df.columns:
        n_null = df[col].isna().sum()
        pct = n_null / len(df) * 100
        report_lines.append(f"  {col:<30} {n_null:>4} / {len(df)} ({pct:.1f}%)")

    # Binary prevalence
    binary_cols = [
        "treat_stats_present", "treat_question_headings", "treat_structured_data",
        "treat_ext_citations_any", "treat_source_brand", "treat_source_earned",
        "conf_brand_recog", "conf_title_has_kw", "conf_https",
    ]
    report_lines.append("\nBinary prevalence:")
    for col in binary_cols:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                mean = valid.mean()
                report_lines.append(f"  {col:<30} {mean:.3f} ({valid.sum():.0f}/{len(valid)})")

    # Continuous stats
    continuous_cols = [
        "treat_stats_density", "treat_structural_modularity", "treat_auth_citations",
        "treat_topical_comp", "treat_freshness",
        "conf_title_kw_sim", "conf_snippet_kw_sim", "conf_title_len", "conf_snippet_len",
        "conf_word_count", "conf_readability", "conf_internal_links",
        "conf_outbound_links", "conf_images_alt", "conf_bm25",
        "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
        "conf_serp_position",
    ]
    report_lines.append("\nContinuous variable stats:")
    for col in continuous_cols:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                report_lines.append(
                    f"  {col:<30} n={len(valid):>4}  "
                    f"mean={valid.mean():>10.2f}  std={valid.std():>10.2f}  "
                    f"min={valid.min():>10.2f}  max={valid.max():>10.2f}"
                )

    # T7 distribution
    report_lines.append("\nT7 Source type distribution:")
    if "treat_source_type" in df.columns:
        for stype, count in df["treat_source_type"].value_counts().items():
            report_lines.append(f"  {stype:<15} {count:>4} ({count/len(df)*100:.1f}%)")

    # T6 freshness distribution
    report_lines.append("\nT6 Freshness distribution:")
    if "treat_freshness" in df.columns:
        for val in sorted(df["treat_freshness"].dropna().unique()):
            count = (df["treat_freshness"] == val).sum()
            report_lines.append(f"  Level {int(val)}: {count:>4} ({count/len(df)*100:.1f}%)")

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    with open(VALIDATION_REPORT, "w") as f:
        f.write(report_text)
    print(f"\nValidation report saved -> {VALIDATION_REPORT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
