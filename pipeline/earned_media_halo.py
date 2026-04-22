#!/usr/bin/env python3
"""Earned Media Halo Effect Analysis.

Tests whether brands/products mentioned in earned media pages (G2, Forbes,
Zapier, etc.) get a ranking boost from the LLM, even though the earned
media pages themselves are demoted.

Approach:
  1. For each keyword, identify earned media pages in the results
  2. Parse the earned media HTML to find which brand domains are mentioned
  3. Flag non-earned pages whose domain appears in an earned media article
     for the same keyword → treatment = "mentioned_in_earned_media"
  4. Run DML: does being mentioned in earned media cause a rank boost?

Usage:
  python pipeline/earned_media_halo.py \
    --run-dir paperSizeExperiment/output/duckduckgo_Llama-3.3-70B-Instruct_serp20_top10
"""

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore", category=UserWarning)

# ── Domain lists (same as extract_features.py) ──────────────────────────────

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

# Build brand name variants for matching in HTML
BRAND_NAMES = {}
for d in BRAND_DOMAINS:
    # "salesforce.com" -> ["salesforce"]
    name = d.split(".")[0].lower()
    # Handle special cases
    if d == "quickbooks.intuit.com":
        BRAND_NAMES[d] = ["quickbooks", "intuit"]
    elif d == "notion.so":
        BRAND_NAMES[d] = ["notion"]
    elif d == "zoom.us":
        BRAND_NAMES[d] = ["zoom"]
    elif d == "elastic.co":
        BRAND_NAMES[d] = ["elastic", "elasticsearch"]
    elif d == "gong.io":
        BRAND_NAMES[d] = ["gong"]
    elif d == "pendo.io":
        BRAND_NAMES[d] = ["pendo"]
    elif d == "confluent.io":
        BRAND_NAMES[d] = ["confluent", "kafka"]
    elif d == "lever.co":
        BRAND_NAMES[d] = ["lever"]
    elif d == "greenhouse.io":
        BRAND_NAMES[d] = ["greenhouse"]
    else:
        BRAND_NAMES[d] = [name]


def extract_text_from_html(html_path: Path) -> str:
    """Extract visible text from cached HTML file."""
    try:
        raw = html_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True).lower()
    except Exception:
        return ""


def find_brands_mentioned(text: str) -> set:
    """Find which brand domains are mentioned in the text."""
    mentioned = set()
    if not text:
        return mentioned
    for domain, names in BRAND_NAMES.items():
        for name in names:
            # Skip very short names that cause false positives
            if len(name) <= 3:
                continue
            # Word boundary match
            if re.search(r'\b' + re.escape(name) + r'\b', text):
                mentioned.add(domain)
                break
    return mentioned


def build_halo_dataset(run_dir: Path) -> pd.DataFrame:
    """Build the halo effect dataset.

    For each keyword:
      1. Find earned media pages and parse their HTML for brand mentions
      2. For all non-earned pages, set treatment = 1 if their domain is
         mentioned in any earned media page for the same keyword
    """
    keywords_path = run_dir / "keywords.jsonl"
    html_cache = run_dir / "html_cache"
    dataset_path = run_dir / "geodml_dataset.csv"

    # Load existing dataset for confounders
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} rows, {df['keyword'].nunique()} keywords")

    # Load keywords.jsonl for URL-to-file mapping
    kw_data = {}
    with open(keywords_path) as f:
        for line in f:
            kw = json.loads(line)
            kw_data[kw["query"]] = kw

    # Build URL -> HTML file mapping from html_cache
    url_to_file = {}
    for fp in html_cache.iterdir():
        if fp.suffix in (".html", ".htm", ""):
            url_to_file[fp.name] = fp

    # For each keyword, find brands mentioned in earned media
    keyword_earned_mentions = {}  # keyword -> set of brand domains mentioned
    keyword_has_earned = {}  # keyword -> bool

    earned_pages_parsed = 0
    earned_pages_with_brands = 0

    for keyword in df["keyword"].unique():
        kw_rows = df[df["keyword"] == keyword]
        earned_rows = kw_rows[kw_rows.get("treat_source_earned", 0) == 1]

        if earned_rows.empty:
            keyword_has_earned[keyword] = False
            keyword_earned_mentions[keyword] = set()
            continue

        keyword_has_earned[keyword] = True
        all_brands_mentioned = set()

        for _, er in earned_rows.iterrows():
            earned_url = er["url"]
            earned_domain = er["domain"]

            # Try to find cached HTML (sha256[:16] naming from gather_data.py)
            import hashlib
            url_hash = hashlib.sha256(earned_url.encode()).hexdigest()[:16]
            html_file = html_cache / f"{url_hash}.html"
            if not html_file.exists():
                html_file = None

            if html_file is None:
                continue

            text = extract_text_from_html(html_file)
            if not text:
                continue

            earned_pages_parsed += 1
            brands = find_brands_mentioned(text)
            if brands:
                earned_pages_with_brands += 1
            all_brands_mentioned.update(brands)

        keyword_earned_mentions[keyword] = all_brands_mentioned

    print(f"\nEarned media parsing:")
    print(f"  Keywords with earned media: {sum(keyword_has_earned.values())}")
    print(f"  Earned pages parsed: {earned_pages_parsed}")
    print(f"  Earned pages with brand mentions: {earned_pages_with_brands}")

    # Now build treatment columns
    mentioned_in_earned = []
    keyword_has_any_earned = []
    is_earned_page = []

    for _, row in df.iterrows():
        keyword = row["keyword"]
        domain = row["domain"]
        source_earned = row.get("treat_source_earned", 0)

        is_earned_page.append(int(source_earned == 1))
        keyword_has_any_earned.append(int(keyword_has_earned.get(keyword, False)))

        if source_earned == 1:
            # Earned media page itself — not in the treatment group
            mentioned_in_earned.append(np.nan)
        elif keyword in keyword_earned_mentions and domain in keyword_earned_mentions[keyword]:
            mentioned_in_earned.append(1)
        elif keyword_has_earned.get(keyword, False):
            # Same keyword has earned media, but this brand is NOT mentioned
            mentioned_in_earned.append(0)
        else:
            # No earned media for this keyword — exclude from analysis
            mentioned_in_earned.append(np.nan)

    df["mentioned_in_earned"] = mentioned_in_earned
    df["keyword_has_earned"] = keyword_has_any_earned
    df["is_earned_page"] = is_earned_page

    # Stats
    treatment = df["mentioned_in_earned"].dropna()
    print(f"\nHalo treatment variable:")
    print(f"  Total eligible rows (non-earned, keyword has earned): {len(treatment)}")
    print(f"  Mentioned in earned (treatment=1): {int(treatment.sum())}")
    print(f"  Not mentioned (treatment=0): {int((treatment == 0).sum())}")
    print(f"  Excluded (earned pages or no earned for keyword): {df['mentioned_in_earned'].isna().sum()}")

    # Show examples
    mentioned = df[df["mentioned_in_earned"] == 1].head(20)
    if not mentioned.empty:
        print(f"\nExample brand pages mentioned in earned media:")
        for _, row in mentioned.iterrows():
            brands_in_earned = keyword_earned_mentions.get(row["keyword"], set())
            print(f"  [{row['keyword']}] {row['domain']} "
                  f"(pre={row.get('pre_rank','?')} → post={row.get('post_rank','?')}, "
                  f"delta={row.get('rank_delta','?')})")

    return df


def run_dml_analysis(df: pd.DataFrame, output_dir: Path):
    """Run DML causal inference on the halo effect."""
    output_dir.mkdir(parents=True, exist_ok=True)

    treatment_col = "mentioned_in_earned"
    outcomes = ["rank_delta", "post_rank", "pre_rank"]

    # Use legacy confounders (higher coverage) + key new ones
    # Filter the eligible subset first to assess coverage
    eligible = df[df["mentioned_in_earned"].notna()].copy()

    all_confounders = [
        # Legacy (from gather_data — higher coverage ~80-85%)
        "X3_word_count", "X6_readability", "X7_internal_links",
        "X7B_outbound_links", "X9_images_with_alt", "X10_https",
        # New confounders (from extract_features)
        "conf_title_len", "conf_snippet_len", "conf_brand_recog",
        "conf_title_has_kw", "conf_bm25", "conf_serp_position",
        "conf_word_count", "conf_readability",
        "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    ]

    # Filter to available confounders with >30% coverage in the eligible subset
    confounders = []
    for c in all_confounders:
        if c not in df.columns:
            continue
        coverage = eligible[c].notna().mean()
        if coverage < 0.30:
            print(f"  Dropped confounder {c} (only {coverage:.1%} coverage in eligible rows)")
        else:
            confounders.append(c)

    methods = [("plr", "lgbm"), ("plr", "rf"), ("irm", "lgbm"), ("irm", "rf")]
    all_results = []

    print(f"\nConfounders ({len(confounders)}): {confounders}")
    print(f"\n{'='*80}")
    print(f"DML HALO EFFECT ANALYSIS")
    print(f"Treatment: mentioned_in_earned (brand page mentioned in earned media for same keyword)")
    print(f"{'='*80}\n")

    for outcome in outcomes:
        for method, learner in methods:
            # Only require treatment + outcome to be non-null; impute confounders
            sub = df[[treatment_col, outcome] + confounders].copy()
            sub = sub.dropna(subset=[treatment_col, outcome])

            if len(sub) < 20:
                print(f"  [{outcome} | {method}/{learner}] Skipped — only {len(sub)} rows")
                continue

            # Impute confounders
            imputer = SimpleImputer(strategy="median")
            X = pd.DataFrame(
                imputer.fit_transform(sub[confounders]),
                columns=confounders, index=sub.index
            )
            scaler = StandardScaler()
            X = pd.DataFrame(
                scaler.fit_transform(X),
                columns=confounders, index=sub.index
            )

            Y = sub[outcome].values
            D = sub[treatment_col].values

            # Get learners
            if learner == "lgbm":
                from lightgbm import LGBMRegressor, LGBMClassifier
                ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, verbose=-1, random_state=42)
                ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, verbose=-1, random_state=42)
                if method == "irm":
                    ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, verbose=-1, random_state=42)
            else:
                ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                            random_state=42, n_jobs=-1)
                ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                            random_state=42, n_jobs=-1)
                if method == "irm":
                    ml_m = RandomForestClassifier(n_estimators=200, max_depth=5,
                                                  random_state=42, n_jobs=-1)

            dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)

            try:
                if method == "plr":
                    model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                            n_folds=5, score="partialling out")
                else:
                    if len(np.unique(D)) > 2:
                        median_d = np.median(D)
                        D_binary = (D > median_d).astype(float)
                        dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D_binary)
                    model = dml.DoubleMLIRM(dml_data, ml_g=ml_l, ml_m=ml_m,
                                            n_folds=5, score="ATE")

                model.fit()

                coef = model.coef[0]
                se = model.se[0]
                p_val = model.pval[0]
                ci = model.confint(level=0.95)
                ci_lower, ci_upper = ci.iloc[0, 0], ci.iloc[0, 1]

                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""

                print(f"  [{outcome} | {method}/{learner}] n={len(sub)}  "
                      f"theta={coef:+.3f} SE={se:.3f} p={p_val:.4f}{sig} "
                      f"CI=[{ci_lower:.3f}, {ci_upper:.3f}]")

                all_results.append({
                    "treatment": "mentioned_in_earned",
                    "outcome": outcome,
                    "method": method,
                    "learner": learner,
                    "n": len(sub),
                    "theta": coef,
                    "se": se,
                    "p_val": p_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "sig": sig,
                })
            except Exception as e:
                print(f"  [{outcome} | {method}/{learner}] ERROR: {e}")

    # Also run comparison: earned pages themselves
    print(f"\n{'='*80}")
    print(f"COMPARISON: Earned media pages themselves (confirming demotion)")
    print(f"{'='*80}\n")

    df_with_earned = df[df["keyword_has_earned"] == 1].copy()
    treatment_col_earned = "is_earned_page"

    for outcome in outcomes:
        sub = df_with_earned[[treatment_col_earned, outcome] + confounders].copy()
        sub = sub.dropna(subset=[treatment_col_earned, outcome])

        if len(sub) < 20:
            continue

        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(
            imputer.fit_transform(sub[confounders]),
            columns=confounders, index=sub.index
        )
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=confounders, index=sub.index
        )
        Y = sub[outcome].values
        D = sub[treatment_col_earned].values

        from lightgbm import LGBMRegressor, LGBMClassifier
        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                             max_depth=5, verbose=-1, random_state=42)
        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                             max_depth=5, verbose=-1, random_state=42)
        dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)
        model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                n_folds=5, score="partialling out")
        model.fit()
        coef = model.coef[0]
        se = model.se[0]
        p_val = model.pval[0]
        ci = model.confint(level=0.95)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  [{outcome} | plr/lgbm] n={len(sub)}  "
              f"theta={coef:+.3f} SE={se:.3f} p={p_val:.4f}{sig} "
              f"CI=[{ci.iloc[0,0]:.3f}, {ci.iloc[0,1]:.3f}]")

        all_results.append({
            "treatment": "is_earned_page",
            "outcome": outcome,
            "method": "plr",
            "learner": "lgbm",
            "n": len(sub),
            "theta": coef,
            "se": se,
            "p_val": p_val,
            "ci_lower": ci.iloc[0, 0],
            "ci_upper": ci.iloc[0, 1],
            "sig": sig,
        })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "halo_effect_results.csv", index=False)
    print(f"\nSaved results -> {output_dir / 'halo_effect_results.csv'}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Treatment':<30} {'Outcome':<12} {'Method':<10} {'theta':>8} {'SE':>8} {'p-val':>8} {'Sig':>4}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['treatment']:<30} {r['outcome']:<12} {r['method']}/{r['learner']:<5} "
              f"{r['theta']:>+8.3f} {r['se']:>8.3f} {r['p_val']:>8.4f} {r['sig']:>4}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Earned Media Halo Effect Analysis")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to experiment run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = run_dir / "analysis_halo"

    print(f"Run directory: {run_dir}")
    print(f"Output: {output_dir}")

    df = build_halo_dataset(run_dir)
    run_dml_analysis(df, output_dir)


if __name__ == "__main__":
    main()
