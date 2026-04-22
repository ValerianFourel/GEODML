#!/usr/bin/env python3
"""
Domain classifier — 4-stage pipeline with checkpoint/resume.

Usage:
    python classify_domains.py --input domains.csv
    python classify_domains.py --input domains.csv --resume   # resume from last checkpoint
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHECKPOINT_FILE = "classify_checkpoint.json"
OUTPUT_FILE = "domains_classified.csv"
UNKNOWN_FILE = "domains_unknown.csv"

# ---------------------------------------------------------------------------
# STAGE 1 — Social blocklist
# ---------------------------------------------------------------------------
SOCIAL_BLOCKLIST = {
    "reddit.com", "x.com", "twitter.com", "facebook.com", "instagram.com",
    "tiktok.com", "youtube.com", "linkedin.com", "pinterest.com", "tumblr.com",
    "snapchat.com", "threads.net", "mastodon.social", "quora.com", "medium.com",
    "substack.com", "wordpress.com", "blogger.com", "weibo.com", "vk.com",
    "discord.com", "twitch.tv", "rumble.com", "gettr.com", "parler.com",
    "gab.com", "truth.social", "yelp.com", "tripadvisor.com", "trustpilot.com",
    "g2.com", "capterra.com", "producthunt.com", "stackexchange.com",
    "stackoverflow.com", "github.com", "dev.to", "hashnode.com",
    "ycombinator.com",  # hacker news
    "news.ycombinator.com",
    "flickr.com", "imgur.com", "tiktok.com", "whatsapp.com",
    "telegram.org", "signal.org", "slack.com", "reddit.old.com",
    "slideshare.net", "scribd.com", "goodreads.com",
    "gitlab.com", "bitbucket.org", "sourceforge.net",
    "wikia.com", "fandom.com",
}


def _strip_www(domain: str) -> str:
    return re.sub(r"^www\.", "", domain.lower().strip())


def _matches_blocklist(domain: str) -> bool:
    """Check if domain or any parent matches the blocklist."""
    d = _strip_www(domain)
    parts = d.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in SOCIAL_BLOCKLIST:
            return True
    return False


def stage1_blocklist(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STAGE 1 — Social blocklist")
    mask = df["label"] == "unknown"
    matched = 0
    for idx in df.index[mask]:
        if _matches_blocklist(df.at[idx, "domain"]):
            df.at[idx, "label"] = "social"
            df.at[idx, "label_source"] = "blocklist"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
    log.info(f"  Stage 1 classified {matched} domains as social")
    return df


# ---------------------------------------------------------------------------
# STAGE 2 — TLD / structural rules
# ---------------------------------------------------------------------------
def stage2_tld_rules(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STAGE 2 — TLD and structural rules")
    mask = df["label"] == "unknown"
    matched = 0
    for idx in df.index[mask]:
        d = _strip_www(df.at[idx, "domain"])
        parts = d.split(".")

        # .gov or .mil
        if parts[-1] in ("gov", "mil"):
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

        # country-level gov: .gov.XX
        if len(parts) >= 2 and parts[-2] == "gov":
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

        # .edu
        if parts[-1] == "edu":
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

        # country-level edu: .edu.XX
        if len(parts) >= 2 and parts[-2] == "edu":
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

        # .ac.XX pattern (academic)
        if len(parts) >= 2 and parts[-2] == "ac":
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

        # wikipedia
        if "wikipedia" in parts or d.endswith("wikipedia.org"):
            df.at[idx, "label"] = "earned"
            df.at[idx, "label_source"] = "tld_rule"
            df.at[idx, "label_confidence"] = 1.0
            matched += 1
            continue

    log.info(f"  Stage 2 classified {matched} domains as earned")
    return df


# ---------------------------------------------------------------------------
# STAGE 3 — Cloudflare Radar (async)
# ---------------------------------------------------------------------------
CF_SOCIAL_CATS = {"social networks", "social media", "forums", "message boards"}
CF_EARNED_CATS = {
    "news", "news and politics", "education", "government", "reference",
    "health", "science", "academic resources", "non-profit",
}
CF_BRAND_CATS = {
    "shopping", "business", "technology", "finance", "real estate",
    "ecommerce", "software", "saas", "marketing",
}


async def _cf_classify_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    domain: str,
    token: str,
) -> tuple[str, str | None, float]:
    """Returns (domain, label_or_None, confidence)."""
    url = f"https://api.cloudflare.com/client/v4/radar/categorize?domain={_strip_www(domain)}"
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(4):
        try:
            async with semaphore:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt
                        log.debug(f"  CF 429 for {domain}, retry in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        return (domain, None, 0.0)
                    data = await resp.json()
        except Exception as e:
            log.debug(f"  CF error for {domain}: {e}")
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)
                continue
            return (domain, None, 0.0)

        # Parse categories from response
        try:
            categories = []
            result = data.get("result", {})
            # Try multiple response shapes
            for key in ("categorizations", "categories", "content_categories"):
                items = result.get(key, [])
                if items:
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get("name", "") or item.get("label", "")
                            if name:
                                categories.append(name.lower())
                        elif isinstance(item, str):
                            categories.append(item.lower())
            if not categories:
                return (domain, None, 0.0)

            for cat in categories:
                if cat in CF_SOCIAL_CATS:
                    return (domain, "social", 0.85)
            for cat in categories:
                if cat in CF_EARNED_CATS:
                    return (domain, "earned", 0.85)
            for cat in categories:
                if cat in CF_BRAND_CATS:
                    return (domain, "brand", 0.85)
            return (domain, None, 0.0)
        except Exception:
            return (domain, None, 0.0)

    return (domain, None, 0.0)


async def stage3_cloudflare(df: pd.DataFrame, checkpoint: dict) -> pd.DataFrame:
    token = os.environ.get("CF_API_TOKEN", "")
    if not token:
        log.warning("STAGE 3 — CF_API_TOKEN not set, SKIPPING Cloudflare stage")
        return df

    log.info("STAGE 3 — Cloudflare Radar categorization")
    unknown_mask = df["label"] == "unknown"
    domains_todo = df.loc[unknown_mask, "domain"].tolist()

    # Filter out already-done domains from checkpoint
    cf_done = set(checkpoint.get("cf_done", []))
    cf_results = checkpoint.get("cf_results", {})
    domains_todo = [d for d in domains_todo if d not in cf_done]
    log.info(f"  {len(domains_todo)} domains to query ({len(cf_done)} already cached)")

    semaphore = asyncio.Semaphore(20)
    matched = 0
    batch_size = 500

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(domains_todo), batch_size):
            batch = domains_todo[batch_start : batch_start + batch_size]
            tasks = [
                _cf_classify_one(session, semaphore, d, token) for d in batch
            ]
            results = await asyncio.gather(*tasks)
            for domain, label, conf in results:
                cf_done.add(domain)
                if label:
                    cf_results[domain] = {"label": label, "confidence": conf}
                    matched += 1

            done_total = batch_start + len(batch)
            if done_total % 500 == 0 or done_total == len(domains_todo):
                log.info(f"  CF progress: {done_total}/{len(domains_todo)}")
                # Save checkpoint
                checkpoint["cf_done"] = list(cf_done)
                checkpoint["cf_results"] = cf_results
                _save_checkpoint(checkpoint)

    # Apply results to dataframe
    applied = 0
    for idx in df.index[df["label"] == "unknown"]:
        d = df.at[idx, "domain"]
        if d in cf_results:
            df.at[idx, "label"] = cf_results[d]["label"]
            df.at[idx, "label_source"] = "cloudflare"
            df.at[idx, "label_confidence"] = cf_results[d]["confidence"]
            applied += 1

    log.info(f"  Stage 3 classified {applied} domains via Cloudflare")
    return df


# ---------------------------------------------------------------------------
# STAGE 4 — Wikidata SPARQL (sync, rate-limited)
# ---------------------------------------------------------------------------
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

EARNED_INSTANCE_KEYWORDS = [
    "newspaper", "magazine", "news agency", "broadcaster", "television channel",
    "radio station", "news website", "online newspaper", "public broadcaster",
    "university", "college", "research institute", "government agency",
    "non-governmental organization", "wiki", "encyclopedia",
    "public university", "private university", "news media",
    "educational institution", "public library",
]

BRAND_INSTANCE_KEYWORDS = [
    "company", "corporation", "business", "startup", "enterprise",
    "software company", "technology company", "retail", "e-commerce",
    "public company", "private company",
]


def _wikidata_classify(domain: str) -> tuple[str | None, float]:
    d = _strip_www(domain)
    # Try multiple URL patterns for the official website property
    url_variants = [
        f"https://{d}",
        f"https://{d}/",
        f"https://www.{d}",
        f"https://www.{d}/",
        f"http://{d}",
        f"http://{d}/",
        f"http://www.{d}",
        f"http://www.{d}/",
    ]
    values_clause = " ".join(f'<{u}>' for u in url_variants)
    query = f"""
    SELECT ?item ?itemLabel ?instanceLabel WHERE {{
      VALUES ?website {{ {values_clause} }}
      ?item wdt:P856 ?website .
      ?item wdt:P31 ?instance .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    LIMIT 5
    """
    for attempt in range(4):
        try:
            resp = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "GEODML-DomainClassifier/1.0 (valerian.fourel@gmail.com)"},
                timeout=30,
            )
            if resp.status_code == 429:
                # Parse retry-after hint if available
                retry_text = resp.text
                wait = 2 ** (attempt + 1)
                match = re.search(r"retry in (\d+) seconds", retry_text)
                if match:
                    wait = int(match.group(1)) + 1
                log.info(f"  WD 429 for {domain}, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return (None, 0.0)
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            if not bindings:
                return (None, 0.0)

            instance_labels = []
            for b in bindings:
                lbl = b.get("instanceLabel", {}).get("value", "").lower()
                if lbl:
                    instance_labels.append(lbl)

            # Check earned first
            for lbl in instance_labels:
                for kw in EARNED_INSTANCE_KEYWORDS:
                    if kw in lbl:
                        return ("earned", 0.90)
            # Then brand
            for lbl in instance_labels:
                for kw in BRAND_INSTANCE_KEYWORDS:
                    if kw in lbl:
                        return ("brand", 0.90)

            return (None, 0.0)
        except Exception as e:
            log.debug(f"  Wikidata error for {domain}: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return (None, 0.0)
    return (None, 0.0)


def stage4_wikidata(df: pd.DataFrame, checkpoint: dict) -> pd.DataFrame:
    log.info("STAGE 4 — Wikidata SPARQL lookup")
    unknown_mask = df["label"] == "unknown"
    domains_todo = df.loc[unknown_mask, "domain"].tolist()

    wd_done = set(checkpoint.get("wd_done", []))
    wd_results = checkpoint.get("wd_results", {})
    domains_todo = [d for d in domains_todo if d not in wd_done]
    log.info(f"  {len(domains_todo)} domains to query ({len(wd_done)} already cached)")

    for i, domain in enumerate(domains_todo):
        label, conf = _wikidata_classify(domain)
        wd_done.add(domain)
        if label:
            wd_results[domain] = {"label": label, "confidence": conf}

        if (i + 1) % 500 == 0 or (i + 1) == len(domains_todo):
            log.info(f"  WD progress: {i+1}/{len(domains_todo)}")
            checkpoint["wd_done"] = list(wd_done)
            checkpoint["wd_results"] = wd_results
            _save_checkpoint(checkpoint)

        time.sleep(0.5)  # Wikidata rate-limits at ~5 req/s; 0.5s is safe

    # Apply
    applied = 0
    for idx in df.index[df["label"] == "unknown"]:
        d = df.at[idx, "domain"]
        if d in wd_results:
            df.at[idx, "label"] = wd_results[d]["label"]
            df.at[idx, "label_source"] = "wikidata"
            df.at[idx, "label_confidence"] = wd_results[d]["confidence"]
            applied += 1

    log.info(f"  Stage 4 classified {applied} domains via Wikidata")
    return df


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def _load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify domains into social/earned/brand/unknown")
    parser.add_argument("--input", required=True, help="Input CSV with 'domain' column")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert "domain" in df.columns, "Input CSV must have a 'domain' column"
    df["domain"] = df["domain"].str.lower().str.strip()
    df = df.drop_duplicates(subset="domain").reset_index(drop=True)
    log.info(f"Loaded {len(df)} unique domains from {args.input}")

    # Load checkpoint or init columns
    checkpoint = _load_checkpoint() if args.resume else {}

    if args.resume and Path(OUTPUT_FILE).exists():
        prev = pd.read_csv(OUTPUT_FILE)
        # Merge previous results into df
        prev_map = prev.set_index("domain")[["label", "label_source", "label_confidence"]].to_dict("index")
        df["label"] = "unknown"
        df["label_source"] = ""
        df["label_confidence"] = 0.0
        restored = 0
        for idx in df.index:
            d = df.at[idx, "domain"]
            if d in prev_map and prev_map[d]["label"] != "unknown":
                df.at[idx, "label"] = prev_map[d]["label"]
                df.at[idx, "label_source"] = prev_map[d]["label_source"]
                df.at[idx, "label_confidence"] = prev_map[d]["label_confidence"]
                restored += 1
        log.info(f"Resumed: restored {restored} previously classified domains")
    else:
        df["label"] = "unknown"
        df["label_source"] = ""
        df["label_confidence"] = 0.0

    # Run stages in order (skip already-classified domains via mask checks inside each stage)
    df = stage1_blocklist(df)
    df = stage2_tld_rules(df)
    # Stage 3 (Cloudflare) disabled — API returns no usable category data
    # df = asyncio.run(stage3_cloudflare(df, checkpoint))
    log.info("STAGE 3 — Cloudflare SKIPPED (disabled)")
    df = stage4_wikidata(df, checkpoint)

    # Save outputs
    out = df[["domain", "label", "label_source", "label_confidence"]]
    out.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Wrote {OUTPUT_FILE}")

    unknowns = out[out["label"] == "unknown"]
    unknowns.to_csv(UNKNOWN_FILE, index=False)
    log.info(f"Wrote {UNKNOWN_FILE} ({len(unknowns)} domains)")

    # Summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal domains: {len(out)}\n")
    print("By label:")
    for label, count in out["label"].value_counts().items():
        pct = 100 * count / len(out)
        print(f"  {label:10s}  {count:6d}  ({pct:.1f}%)")
    print("\nBy source:")
    for src, count in out["label_source"].value_counts().items():
        if src:
            print(f"  {src:15s}  {count:6d}")
    print(f"\nStill unknown: {len(unknowns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
