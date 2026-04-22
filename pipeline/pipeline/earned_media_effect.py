#!/usr/bin/env python3
"""Test whether being mentioned in earned media boosts a brand's LLM ranking.

Hypothesis: The LLM reads G2/Capterra snippets in the SERP, learns which brands
are "top picks", demotes the earned media page itself, but PROMOTES the brands
it mentions. Earned media = signal amplifier that gets sacrificed.

Two approaches:
  A) Co-appearance proxy: brand pages in keywords WHERE earned media is present
     vs keywords where it is absent.
  B) Direct mention: fetch earned media HTML, parse which brands they link/mention,
     create a per-(keyword, brand) treatment.

Usage:
  python pipeline/earned_media_effect.py                     # Approach A only
  python pipeline/earned_media_effect.py --fetch-earned       # A + B (fetches HTML)
  python pipeline/earned_media_effect.py --dataset deepseek   # Use DeepSeek R1 data
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Experiment JSON files
EXPERIMENT_LLAMA = PROJECT_ROOT / "results" / "searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json"
EXPERIMENT_DEEPSEEK = PROJECT_ROOT / "output" / "deepseek-r1" / "experiment.json"

# Dataset files
DATASET_LLAMA = PROJECT_ROOT / "output" / "geodml_dataset.csv"
DATASET_DEEPSEEK = PROJECT_ROOT / "output" / "deepseek-r1" / "geodml_dataset.csv"

# Features
FEATURES_NEW = SCRIPT_DIR / "intermediate" / "features_new.csv"

# Output
OUTPUT_DIR = SCRIPT_DIR / "earned_media_results"

# ── Domain lists (same as extract_features.py) ───────────────────────────────

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

# Also build a brand name lookup (domain → short brand name for text matching)
BRAND_NAMES = {}
for d in BRAND_DOMAINS:
    name = d.split(".")[0]
    BRAND_NAMES[d] = name


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_domain(url: str) -> str:
    import tldextract
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def load_experiment_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── STEP 1: Parse raw SERP data ─────────────────────────────────────────────

def parse_serp_data(experiment: dict) -> dict:
    """Extract per-keyword SERP info: which domains appear, their snippets, etc.

    Returns dict[keyword] = {
        "earned_pages": [{domain, url, snippet, position}, ...],
        "brand_pages": [{domain, url, snippet, position}, ...],
        "all_pages": [{domain, url, snippet, position}, ...],
        "has_earned": bool,
        "earned_domains_in_serp": set,
    }
    """
    serp_data = {}

    # Handle both formats
    if "per_keyword_results" in experiment:
        items = experiment["per_keyword_results"]
    elif "queries" in experiment:
        items = experiment["queries"]
    else:
        items = experiment.get("results", [])

    if isinstance(items, dict):
        items = list(items.values())

    for item in items:
        keyword = item.get("keyword", item.get("query", ""))
        if not keyword:
            continue

        # Try multiple paths for raw results
        raw_results = []
        if "serp" in item and isinstance(item["serp"], dict):
            raw_results = item["serp"].get("raw_results", [])
        if not raw_results:
            raw_results = item.get("search_results", {}).get("raw_results", [])
        if not raw_results:
            raw_results = item.get("raw_results", [])

        earned_pages = []
        brand_pages = []
        all_pages = []
        earned_domains_in_serp = set()

        for r in raw_results:
            url = r.get("url", "")
            domain = extract_domain(url)
            snippet = r.get("snippet", "")
            title = r.get("title", "")
            position = r.get("position", 0)

            page_info = {
                "domain": domain,
                "url": url,
                "snippet": snippet,
                "title": title,
                "position": position,
            }
            all_pages.append(page_info)

            if domain in EARNED_DOMAINS:
                earned_pages.append(page_info)
                earned_domains_in_serp.add(domain)
            elif domain in BRAND_DOMAINS:
                brand_pages.append(page_info)

        serp_data[keyword] = {
            "earned_pages": earned_pages,
            "brand_pages": brand_pages,
            "all_pages": all_pages,
            "has_earned": len(earned_pages) > 0,
            "earned_domains_in_serp": earned_domains_in_serp,
            "n_earned": len(earned_pages),
        }

    return serp_data


# ── STEP 2: Extract brand mentions from earned media snippets ────────────────

def find_brand_mentions_in_snippets(serp_data: dict) -> dict:
    """For each keyword, check which brands are mentioned in earned media snippets/titles.

    Returns dict[keyword] = set(brand_domains mentioned in earned snippets)
    """
    mentions = {}

    for keyword, data in serp_data.items():
        mentioned_brands = set()
        for earned_page in data["earned_pages"]:
            text = (earned_page["snippet"] + " " + earned_page["title"]).lower()
            for brand_domain, brand_name in BRAND_NAMES.items():
                # Match brand name (at least 3 chars to avoid false positives)
                if len(brand_name) >= 3 and brand_name.lower() in text:
                    mentioned_brands.add(brand_domain)
                # Also match the full domain
                if brand_domain.lower() in text:
                    mentioned_brands.add(brand_domain)
        mentions[keyword] = mentioned_brands

    return mentions


# ── STEP 3: Fetch earned media HTML and extract brand mentions ───────────────

def fetch_earned_html(serp_data: dict, cache_dir: Path) -> dict:
    """Fetch HTML for earned media pages, cache locally.

    Returns dict[(keyword, earned_url)] = html_content
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    html_cache = {}
    urls_to_fetch = []

    for keyword, data in serp_data.items():
        for earned_page in data["earned_pages"]:
            url = earned_page["url"]
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = cache_dir / f"{cache_key}.html"

            if cache_file.exists():
                html_cache[(keyword, url)] = cache_file.read_text(errors="replace")
            else:
                urls_to_fetch.append((keyword, url, cache_file))

    print(f"  Earned media pages: {len(html_cache)} cached, {len(urls_to_fetch)} to fetch")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    for i, (keyword, url, cache_file) in enumerate(urls_to_fetch):
        print(f"  [{i+1}/{len(urls_to_fetch)}] Fetching {url[:80]}...")
        try:
            resp = session.get(url, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            html = resp.text[:5_000_000]  # 5MB max
            cache_file.write_text(html)
            html_cache[(keyword, url)] = html
        except Exception as e:
            print(f"    ERROR: {e}")
            html_cache[(keyword, url)] = ""

        # Polite delay
        time.sleep(random.uniform(2.0, 4.0))

    return html_cache


def find_brand_mentions_in_html(serp_data: dict, html_cache: dict) -> dict:
    """Parse earned media HTML to find which brands are mentioned/linked.

    Returns dict[keyword] = set(brand_domains found on earned pages)
    """
    from bs4 import BeautifulSoup

    mentions = {}

    for keyword, data in serp_data.items():
        mentioned_brands = set()

        for earned_page in data["earned_pages"]:
            url = earned_page["url"]
            html = html_cache.get((keyword, url), "")
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")

            # Method 1: Check all links on the page
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                link_domain = extract_domain(href)
                if link_domain in BRAND_DOMAINS:
                    mentioned_brands.add(link_domain)

            # Method 2: Check body text for brand names
            body = soup.find("body")
            if body:
                text = body.get_text(separator=" ", strip=True).lower()
                for brand_domain, brand_name in BRAND_NAMES.items():
                    if len(brand_name) >= 4 and brand_name.lower() in text:
                        mentioned_brands.add(brand_domain)

        mentions[keyword] = mentioned_brands

    return mentions


# ── STEP 4: Build treatment variables ────────────────────────────────────────

def build_treatments(dataset: pd.DataFrame, serp_data: dict,
                     snippet_mentions: dict, html_mentions: dict | None
                     ) -> pd.DataFrame:
    """Add new treatment columns to the dataset.

    New columns:
      - treat_earned_coappear: 1 if keyword has ANY earned media in SERP
      - treat_earned_n: count of earned media pages in same keyword SERP
      - treat_mentioned_snippet: 1 if brand appears in earned media snippet text
      - treat_mentioned_html: 1 if brand appears in earned media HTML (if available)
    """
    df = dataset.copy()

    # Approach A: co-appearance
    df["treat_earned_coappear"] = 0
    df["treat_earned_n"] = 0

    # Approach A+: snippet mentions
    df["treat_mentioned_snippet"] = 0

    # Approach B: HTML mentions
    if html_mentions is not None:
        df["treat_mentioned_html"] = 0

    for idx, row in df.iterrows():
        keyword = row.get("keyword", "")
        domain = row.get("domain", "")

        if keyword not in serp_data:
            continue

        kw_data = serp_data[keyword]

        # A: Co-appearance
        if kw_data["has_earned"]:
            df.at[idx, "treat_earned_coappear"] = 1
            df.at[idx, "treat_earned_n"] = kw_data["n_earned"]

        # A+: Snippet mentions
        if keyword in snippet_mentions and domain in snippet_mentions[keyword]:
            df.at[idx, "treat_mentioned_snippet"] = 1

        # B: HTML mentions
        if html_mentions is not None:
            if keyword in html_mentions and domain in html_mentions[keyword]:
                df.at[idx, "treat_mentioned_html"] = 1

    return df


# ── STEP 5: DML Analysis ────────────────────────────────────────────────────

CONFOUNDERS_NEW = [
    "conf_title_kw_sim", "conf_snippet_kw_sim", "conf_title_len",
    "conf_snippet_len", "conf_brand_recog", "conf_title_has_kw",
    "conf_word_count", "conf_readability", "conf_internal_links",
    "conf_outbound_links", "conf_images_alt", "conf_bm25",
    "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    "conf_serp_position",
]

CONFOUNDERS_LEGACY = [
    "X1_domain_authority", "X2_domain_age_years", "X3_word_count",
    "X6_readability", "X7_internal_links", "X7B_outbound_links",
    "X8_keyword_difficulty", "X9_images_with_alt",
]

OUTCOMES = ["rank_delta", "pre_rank", "post_rank"]


def pick_confounders(df: pd.DataFrame) -> list[str]:
    """Pick whichever confounder set has better coverage."""
    new_present = [c for c in CONFOUNDERS_NEW if c in df.columns]
    legacy_present = [c for c in CONFOUNDERS_LEGACY if c in df.columns]
    if len(new_present) >= 8:
        return new_present
    return legacy_present


def preprocess(df, treatment_col, outcome_col, confounders):
    """Impute, standardize, drop missing treatment/outcome rows."""
    cols = confounders + [treatment_col, outcome_col]
    sub = df[[c for c in cols if c in df.columns]].copy()
    sub = sub.dropna(subset=[treatment_col, outcome_col])
    n = len(sub)
    if n < 20:
        return None, None, None, n

    avail_conf = [c for c in confounders if c in sub.columns]
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(sub[avail_conf]),
                     columns=avail_conf, index=sub.index)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=avail_conf, index=sub.index)

    Y = sub[outcome_col].values
    D = sub[treatment_col].values
    return X, Y, D, n


def run_dml_analysis(df: pd.DataFrame, treatments: dict, confounders: list,
                     label: str) -> list[dict]:
    """Run DML for each treatment × outcome × learner."""
    from lightgbm import LGBMRegressor

    results = []

    for treat_name, treat_col in treatments.items():
        if treat_col not in df.columns:
            print(f"  SKIP {treat_name}: column {treat_col} not in dataset")
            continue

        n_treated = (df[treat_col] == 1).sum() if df[treat_col].dtype != float else (df[treat_col] > 0).sum()
        n_untreated = len(df) - n_treated
        print(f"\n  {treat_name} ({treat_col}): {n_treated} treated, {n_untreated} untreated")

        for outcome in OUTCOMES:
            if outcome not in df.columns:
                continue

            X, Y, D, n = preprocess(df, treat_col, outcome, confounders)
            if X is None:
                print(f"    {outcome}: skipped (n={n})")
                continue

            n_d1 = int(D.sum()) if D is not None else 0
            print(f"    {outcome}: n={n}, treated={n_d1}")

            for learner_name in ["lgbm", "rf"]:
                try:
                    if learner_name == "lgbm":
                        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=5, num_leaves=31,
                                             verbose=-1, random_state=42)
                        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=5, num_leaves=31,
                                             verbose=-1, random_state=42)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                                     random_state=42, n_jobs=-1)
                        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                                     random_state=42, n_jobs=-1)

                    dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)
                    model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                            n_folds=5, score="partialling out")
                    model.fit()

                    coef = model.coef[0]
                    se = model.se[0]
                    t_stat = model.t_stat[0]
                    p_val = model.pval[0]
                    ci = model.confint(level=0.95)

                    sig = ""
                    if p_val < 0.01:
                        sig = "***"
                    elif p_val < 0.05:
                        sig = "**"
                    elif p_val < 0.1:
                        sig = "*"

                    result = {
                        "experiment": label,
                        "treatment": treat_name,
                        "treatment_col": treat_col,
                        "outcome": outcome,
                        "learner": learner_name,
                        "n_obs": n,
                        "n_treated": n_d1,
                        "coef": coef,
                        "se": se,
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "ci_lower": ci.iloc[0, 0],
                        "ci_upper": ci.iloc[0, 1],
                        "significance": sig,
                    }
                    results.append(result)

                    marker = f" {sig}" if sig else ""
                    print(f"      {learner_name}: θ={coef:+.3f} (p={p_val:.4f}){marker}")

                except Exception as e:
                    print(f"      {learner_name}: ERROR {e}")

    return results


# ── STEP 6: Descriptive statistics ───────────────────────────────────────────

def print_descriptives(df, serp_data, snippet_mentions, html_mentions):
    """Print descriptive statistics about the earned media treatment."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    # Keywords with/without earned media
    kw_with = [k for k, v in serp_data.items() if v["has_earned"]]
    kw_without = [k for k, v in serp_data.items() if not v["has_earned"]]
    print(f"\nKeywords with earned media in SERP: {len(kw_with)}/{len(serp_data)}")
    print(f"Keywords without earned media:      {len(kw_without)}/{len(serp_data)}")

    # Earned media pages breakdown
    all_earned = []
    for k, v in serp_data.items():
        for ep in v["earned_pages"]:
            all_earned.append(ep["domain"])
    if all_earned:
        from collections import Counter
        print(f"\nEarned media pages in SERPs: {len(all_earned)} total")
        print("  Top earned domains:")
        for domain, count in Counter(all_earned).most_common(10):
            print(f"    {domain}: {count} appearances")

    # Treatment distribution
    if "treat_earned_coappear" in df.columns:
        n1 = (df["treat_earned_coappear"] == 1).sum()
        n0 = (df["treat_earned_coappear"] == 0).sum()
        print(f"\nApproach A — Co-appearance:")
        print(f"  treat_earned_coappear=1: {n1} rows ({100*n1/len(df):.1f}%)")
        print(f"  treat_earned_coappear=0: {n0} rows ({100*n0/len(df):.1f}%)")

    if "treat_mentioned_snippet" in df.columns:
        n1 = (df["treat_mentioned_snippet"] == 1).sum()
        print(f"\nApproach A+ — Snippet mentions:")
        print(f"  treat_mentioned_snippet=1: {n1} rows ({100*n1/len(df):.1f}%)")

        if n1 > 0:
            # Show which brands were mentioned
            mentioned = df[df["treat_mentioned_snippet"] == 1][["keyword", "domain"]].drop_duplicates()
            print(f"  Brands mentioned in earned snippets:")
            for _, row in mentioned.iterrows():
                print(f"    {row['keyword']}: {row['domain']}")

    if "treat_mentioned_html" in df.columns:
        n1 = (df["treat_mentioned_html"] == 1).sum()
        print(f"\nApproach B — HTML mentions:")
        print(f"  treat_mentioned_html=1: {n1} rows ({100*n1/len(df):.1f}%)")

    # Mean rank_delta by treatment
    print("\n--- Mean rank_delta by treatment ---")
    for col_name, col_label in [
        ("treat_earned_coappear", "Co-appear with earned"),
        ("treat_mentioned_snippet", "Mentioned in snippet"),
        ("treat_mentioned_html", "Mentioned in HTML"),
    ]:
        if col_name not in df.columns or "rank_delta" not in df.columns:
            continue
        sub = df.dropna(subset=["rank_delta", col_name])
        g = sub.groupby(col_name)["rank_delta"].agg(["mean", "std", "count"])
        print(f"\n  {col_label}:")
        for val, row in g.iterrows():
            label = "YES" if val == 1 else "NO"
            print(f"    {label}: mean={row['mean']:+.2f}, std={row['std']:.2f}, n={int(row['count'])}")


# ── STEP 7: Exclude earned media observations ───────────────────────────────

def filter_non_earned(df: pd.DataFrame) -> pd.DataFrame:
    """Remove earned media pages from the dataset (keep brand + other only).

    The hypothesis is about how earned media affects BRAND pages' rankings,
    so we exclude the earned pages themselves.
    """
    if "treat_source_earned" in df.columns:
        mask = df["treat_source_earned"] != 1
    else:
        # Fall back to domain matching
        mask = ~df["domain"].apply(lambda d: d in EARNED_DOMAINS if pd.notna(d) else False)

    n_removed = (~mask).sum()
    print(f"\n  Removed {n_removed} earned media observations (keeping brand + other)")
    return df[mask].copy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Earned media effect analysis")
    parser.add_argument("--fetch-earned", action="store_true",
                        help="Fetch earned media HTML for Approach B")
    parser.add_argument("--dataset", choices=["llama", "deepseek"], default="llama",
                        help="Which experiment dataset to use")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    if args.dataset == "deepseek":
        exp_path = EXPERIMENT_DEEPSEEK
        ds_path = DATASET_DEEPSEEK
        label = "deepseek-r1"
    else:
        exp_path = EXPERIMENT_LLAMA
        ds_path = DATASET_LLAMA
        label = "llama-3.3-70b"

    print(f"Loading experiment: {exp_path}")
    experiment = load_experiment_json(exp_path)

    print(f"Loading dataset: {ds_path}")
    dataset = pd.read_csv(ds_path)
    print(f"  Dataset shape: {dataset.shape}")

    # Try to merge new features if available
    if FEATURES_NEW.exists():
        print(f"  Merging new features from {FEATURES_NEW}")
        features = pd.read_csv(FEATURES_NEW)
        # Merge on keyword + domain (or keyword + url)
        merge_cols = []
        if "keyword" in features.columns and "keyword" in dataset.columns:
            merge_cols.append("keyword")
        if "url" in features.columns and "url" in dataset.columns:
            merge_cols.append("url")
        elif "domain" in features.columns and "domain" in dataset.columns:
            merge_cols.append("domain")

        if merge_cols:
            new_cols = [c for c in features.columns if c not in dataset.columns]
            if new_cols:
                features_subset = features[merge_cols + new_cols].drop_duplicates(subset=merge_cols)
                dataset = dataset.merge(features_subset, on=merge_cols, how="left")
                print(f"  Merged {len(new_cols)} new columns. Shape: {dataset.shape}")

    # ── Parse SERP data ──────────────────────────────────────────────────────
    print("\nParsing SERP data...")
    serp_data = parse_serp_data(experiment)
    print(f"  Parsed {len(serp_data)} keywords")

    # ── Extract snippet mentions ─────────────────────────────────────────────
    print("Extracting brand mentions from earned media snippets...")
    snippet_mentions = find_brand_mentions_in_snippets(serp_data)
    n_kw_with_mentions = sum(1 for v in snippet_mentions.values() if v)
    n_total_mentions = sum(len(v) for v in snippet_mentions.values())
    print(f"  {n_kw_with_mentions} keywords have snippet-level brand mentions")
    print(f"  {n_total_mentions} total brand mentions in earned snippets")

    # ── Approach B: fetch HTML ───────────────────────────────────────────────
    html_mentions = None
    if args.fetch_earned:
        print("\nFetching earned media HTML (Approach B)...")
        cache_dir = OUTPUT_DIR / "earned_html_cache"
        html_cache = fetch_earned_html(serp_data, cache_dir)
        print(f"  Fetched {len(html_cache)} pages")

        print("Extracting brand mentions from HTML...")
        html_mentions = find_brand_mentions_in_html(serp_data, html_cache)
        n_kw_with_html = sum(1 for v in html_mentions.values() if v)
        n_total_html = sum(len(v) for v in html_mentions.values())
        print(f"  {n_kw_with_html} keywords have HTML-level brand mentions")
        print(f"  {n_total_html} total brand mentions in earned HTML")

        # Show examples
        for kw, brands in sorted(html_mentions.items()):
            if brands:
                print(f"    {kw}: {', '.join(sorted(brands)[:5])}{'...' if len(brands) > 5 else ''}")

    # ── Build treatment variables ────────────────────────────────────────────
    print("\nBuilding treatment variables...")
    dataset = build_treatments(dataset, serp_data, snippet_mentions, html_mentions)

    # ── Remove earned media pages (we're testing effect ON brand pages) ──────
    dataset_filtered = filter_non_earned(dataset)

    # ── Descriptive statistics ───────────────────────────────────────────────
    print_descriptives(dataset_filtered, serp_data, snippet_mentions, html_mentions)

    # ── Pick confounders ─────────────────────────────────────────────────────
    confounders = pick_confounders(dataset_filtered)
    print(f"\nUsing {len(confounders)} confounders: {confounders[:5]}...")

    # ── Define treatments to test ────────────────────────────────────────────
    treatments = {
        "T8a_earned_coappear": "treat_earned_coappear",
        "T8b_earned_count": "treat_earned_n",
        "T8c_mentioned_snippet": "treat_mentioned_snippet",
    }
    if html_mentions is not None:
        treatments["T8d_mentioned_html"] = "treat_mentioned_html"

    # ── Run DML ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DML ANALYSIS — Earned Media Effect ({label})")
    print("=" * 70)

    results = run_dml_analysis(dataset_filtered, treatments, confounders, label)

    # ── Save results ─────────────────────────────────────────────────────────
    if results:
        results_df = pd.DataFrame(results)
        csv_path = OUTPUT_DIR / f"earned_media_dml_{label}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY — Significant results (p < 0.10)")
        print("=" * 70)
        sig_results = results_df[results_df["p_val"] < 0.10].sort_values("p_val")
        if len(sig_results) == 0:
            print("  No significant results at p < 0.10")
        else:
            for _, r in sig_results.iterrows():
                print(f"  {r['treatment']} on {r['outcome']} ({r['learner']}): "
                      f"θ={r['coef']:+.3f} (p={r['p_val']:.4f}) {r['significance']}  "
                      f"[n={r['n_obs']}, treated={r['n_treated']}]")

        # Save summary JSON
        summary = {
            "experiment": label,
            "hypothesis": "Being mentioned in earned media boosts brand page LLM ranking",
            "approaches": {
                "A_coappear": "Binary: keyword has any earned media in SERP",
                "A_count": "Count: number of earned media pages in keyword SERP",
                "A_snippet": "Binary: brand name appears in earned media snippet",
            },
            "n_keywords": len(serp_data),
            "n_keywords_with_earned": sum(1 for v in serp_data.values() if v["has_earned"]),
            "n_observations": len(dataset_filtered),
            "confounders": confounders,
            "results": results,
        }
        if html_mentions is not None:
            summary["approaches"]["B_html"] = "Binary: brand appears on earned media page HTML"
            summary["n_html_mentions"] = sum(len(v) for v in html_mentions.values())

        json_path = OUTPUT_DIR / f"earned_media_summary_{label}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
