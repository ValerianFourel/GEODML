#!/usr/bin/env python3
"""Configuration for the paper-size experiment.

All experiment parameters in one place. Edit this file to configure
keywords, LLMs, pool sizes, and feature extraction settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env.local")

# Keywords file — one keyword per line, # for comments
KEYWORDS_FILE = SCRIPT_DIR / "keywords.txt"

# Master output directory — each (engine, model, pool_size) combo gets a subdir
OUTPUT_ROOT = SCRIPT_DIR / "output"

# ── API Keys ─────────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
OPENPAGERANK_KEY = os.getenv("OPENPAGERANK_KEY", "")
MOZ_API_KEY = os.getenv("MOZ_API_KEY", "")
KAGI_TOKEN = os.getenv("KAGI_TOKEN", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# ── Experiment Grid ──────────────────────────────────────────────────────────

# LLM models to use for re-ranking (HuggingFace model IDs)
LLM_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

# Pool sizes: (serp_results, llm_top_n)
# serp_results = how many results to fetch from search engine
# llm_top_n = how many the LLM re-ranks to
POOL_SIZES = [
    (20, 10),   # small pool: 20 SERP results -> top 10
    (50, 10),   # large pool: 50 SERP results -> top 10
]

# Search engine backend
# Options: searxng, duckduckgo, google_api, brave, serpapi
SEARCH_ENGINE = "duckduckgo"

# ── Feature Extraction Settings ──────────────────────────────────────────────

# Enable optional phases
ENABLE_LLM_FEATURES = True       # LLM-based T1-T4 extraction
ENABLE_PAGERANK = True            # Open PageRank API (X1)
ENABLE_WHOIS = False              # WHOIS domain age (X2) — slow, disable by default

# HTML fetching
FETCH_TIMEOUT = 30
MAX_HTML_SIZE = 5 * 1024 * 1024   # 5 MB
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

# LLM re-ranking parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

# ── DML Analysis Settings ────────────────────────────────────────────────────

# Methods to run
DML_METHODS = ["plr"]              # ["plr", "irm"]
DML_LEARNERS = ["lgbm", "rf"]     # sensitivity check
DML_N_FOLDS = 5
DML_OUTCOMES = ["rank_delta", "post_rank"]

# Treatment definitions — maps short name -> column name in merged dataset
# Code-based treatments (from HTML analysis)
TREATMENTS_CODE = {
    "T1_code": "T1_statistical_density_code",
    "T2_code": "T2_question_heading_code",
    "T3_code": "T3_structured_data_code",
    "T4_code": "T4_citation_authority_code",
}

# LLM-based treatments (from LLM reading page digest)
TREATMENTS_LLM = {
    "T1_llm": "T1_statistical_density_llm",
    "T2_llm": "T2_question_heading_llm",
    "T3_llm": "T3_structured_data_llm",
    "T4_llm": "T4_citation_authority_llm",
}

# New expanded treatments (from extract_features)
TREATMENTS_NEW = {
    "T1a_stats_present": "treat_stats_present",
    "T1b_stats_density": "treat_stats_density",
    "T2a_question_headings": "treat_question_headings",
    "T2b_structural_modularity": "treat_structural_modularity",
    "T3_structured_data_new": "treat_structured_data",
    "T4a_ext_citations": "treat_ext_citations_any",
    "T4b_auth_citations": "treat_auth_citations",
    "T5_topical_comp": "treat_topical_comp",
    "T6_freshness": "treat_freshness",
    "T7_source_earned": "treat_source_earned",
}

# All treatments combined
ALL_TREATMENTS = {**TREATMENTS_CODE, **TREATMENTS_LLM, **TREATMENTS_NEW}

# Treatment labels for plots and reports
TREATMENT_LABELS = {
    "T1_code": "T1 Statistical Density (code)",
    "T2_code": "T2 Question Headings (code)",
    "T3_code": "T3 Structured Data (code)",
    "T4_code": "T4 Citation Authority (code)",
    "T1_llm": "T1 Statistical Density (LLM)",
    "T2_llm": "T2 Question Headings (LLM)",
    "T3_llm": "T3 Structured Data (LLM)",
    "T4_llm": "T4 Citation Authority (LLM)",
    "T1a_stats_present": "T1a Stats Present (binary)",
    "T1b_stats_density": "T1b Stats Density (continuous)",
    "T2a_question_headings": "T2a Question Headings (binary)",
    "T2b_structural_modularity": "T2b Structural Modularity (count)",
    "T3_structured_data_new": "T3 Structured Data (expanded)",
    "T4a_ext_citations": "T4a External Citations (binary)",
    "T4b_auth_citations": "T4b Authority Citations (count)",
    "T5_topical_comp": "T5 Topical Competence (cosine)",
    "T6_freshness": "T6 Freshness (ordinal 0-4)",
    "T7_source_earned": "T7 Source: Earned",
}

# Confounders — new set with maximum coverage
CONFOUNDERS = [
    "conf_title_kw_sim",
    "conf_snippet_kw_sim",
    "conf_title_len",
    "conf_snippet_len",
    "conf_brand_recog",
    "conf_title_has_kw",
    "conf_word_count",
    "conf_readability",
    "conf_internal_links",
    "conf_outbound_links",
    "conf_images_alt",
    "conf_bm25",
    "conf_https",
    "conf_domain_authority",
    "conf_backlinks",
    "conf_referring_domains",
    "conf_serp_position",
]

# Legacy confounders (fallback if new ones unavailable)
CONFOUNDERS_LEGACY = [
    "X1_domain_authority",
    "X2_domain_age_years",
    "X3_word_count",
    "X6_readability",
    "X7_internal_links",
    "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt",
]


def run_label(engine: str, model_id: str, serp_n: int, llm_top_n: int) -> str:
    """Generate a short label for an experiment run."""
    model_short = model_id.split("/")[-1]
    return f"{engine}_{model_short}_serp{serp_n}_top{llm_top_n}"


def run_output_dir(engine: str, model_id: str, serp_n: int, llm_top_n: int) -> Path:
    """Return the output directory for a specific run configuration."""
    return OUTPUT_ROOT / run_label(engine, model_id, serp_n, llm_top_n)
