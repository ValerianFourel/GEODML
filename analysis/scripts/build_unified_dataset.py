#!/usr/bin/env python3
"""Build the single super-detailed unified parquet that bundles every
treatment, confounder, outcome, and metadata field across all experiments
and variants. Designed for reviewer distribution: one parquet + a sidecar
column dictionary describing the type, source, units, and notes for each
column.

Inputs (current on-disk state — POST DataForSEO merge):
  ~/geodml_data/data/main/full_experiment_data_{biased,neutral,
                                                  biased_rag,neutral_rag}.parquet
  ~/geodml_data/data/main/regression_dataset.parquet
  ~/geodml_data/data/dataforseo/domain_authority_dfs.parquet (provenance)

Outputs:
  ~/geodml_data/data/main/unified_2026-05-24.parquet
  ~/geodml_data/data/main/unified_2026-05-24_dictionary.csv
  ~/geodml_data/data/main/unified_2026-05-24_README.md

Run:
  python scripts/build_unified_dataset.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "geodml_data"
MAIN = ROOT / "data" / "main"
TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")

OUT_PARQUET = MAIN / f"unified_{TODAY}.parquet"
OUT_DICT_CSV = MAIN / f"unified_{TODAY}_dictionary.csv"
OUT_README = MAIN / f"unified_{TODAY}_README.md"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]


# ── Column registry: type, source, units, description, notes ──────────────


REGISTRY = [
    # (column, type, source, units, description, notes)
    # IDENTIFIERS
    ("keyword", "identifier", "SERP query", "text",
     "Search query string", "1011 distinct queries"),
    ("url", "identifier", "SERP result", "text",
     "Full URL of the candidate page", "Joined with domain"),
    ("domain", "identifier", "URL parse", "text",
     "Registered domain extracted from url", "~24k unique"),
    ("variant", "identifier", "experiment design", "category",
     "{biased, neutral, biased_rag, neutral_rag}",
     "Prompt × evidence cell"),
    ("search_engine", "identifier", "experiment design", "category",
     "{ddg, searxng}",
     "DuckDuckGo or SearXNG"),
    ("llm_model", "identifier", "experiment design", "category",
     "{Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct}",
     "70B-class instruction LLM"),
    ("pool_size", "identifier", "experiment design", "int",
     "{20, 50}", "SERP-pool size given to the LLM"),
    ("top_n", "identifier", "experiment design", "int",
     "10", "LLM asked to return top-N URLs"),
    ("run_id", "identifier", "pipeline run", "text",
     "Composite of engine_model_serpN_topK_variant",
     "Used to trace back to phase2 jsonl"),
    ("llm_backend", "identifier", "pipeline run", "category",
     "{local}", "Always local inference for the reported runs"),
    ("llm_precision", "identifier", "pipeline run", "category",
     "{bf16-full}", "bf16 full precision used across the campaign"),

    # OUTCOMES
    ("pre_rank", "outcome (input)", "SERP engine", "rank position (1-based)",
     "Original rank of the URL in the SERP",
     "NaN if URL was not in the SERP pool"),
    ("post_rank", "outcome (LLM)", "LLM rerank", "rank position (1-based)",
     "Rank the LLM assigned in its top-N output",
     "NaN if LLM did NOT include the URL → admission failure"),
    ("rank_delta", "outcome (derived)", "computation", "rank positions",
     "rank_delta = pre_rank − post_rank",
     "Positive = LLM moved doc UP; defined only for admitted URLs"),
    ("selected_by_llm", "outcome (derived)", "computation", "binary {0,1}",
     "1 if URL appears in the LLM's top-N output, else 0",
     "Binary admission outcome (Y_1 in dml_selected.py)"),

    # TREATMENTS — content/source flags
    ("treat_stats_present", "treatment (T1a)", "page-HTML parse",
     "binary {0,1}", "1 if page body contains numeric statistics",
     "Code-derived, features.py:extract_t1a_stats_present"),
    ("treat_stats_density", "treatment (T1b)", "page-HTML parse",
     "float (#stats per 1000 words)",
     "Density of numeric statistics in page body",
     "Code-derived, features.py:extract_t1b_stats_density"),
    ("treat_question_headings", "treatment (T2a)", "page-HTML parse",
     "binary {0,1}",
     "1 if page uses Q-format headings (e.g. 'What is X?')",
     "Code-derived"),
    ("treat_structural_modularity", "treatment (T2b)", "page-HTML parse",
     "int (count)", "Count of structural blocks (sections, lists, etc.)",
     "Code-derived"),
    ("treat_structured_data", "treatment (T3)", "page-HTML parse",
     "binary {0,1}",
     "1 if page contains JSON-LD structured data of recognized types",
     "Code-derived, narrower than T3_llm"),
    ("treat_ext_citations_any", "treatment (T4a)", "page-HTML parse",
     "binary {0,1}",
     "1 if page has any external citations / outbound to citation domains",
     "Code-derived"),
    ("treat_auth_citations", "treatment (T4b)", "page-HTML parse",
     "int (count)", "Count of citations to authoritative TLDs (.gov/.edu/.org)",
     "Code-derived"),
    ("treat_topical_comp", "treatment (T5)", "semantic embedding",
     "cosine similarity ∈ [0,1]",
     "Topical completeness — cosine(page body, keyword) via sentence-transformers",
     "Continuous"),
    ("treat_freshness", "treatment (T6)", "page-HTML parse",
     "ordinal 0–4",
     "Heavy-handed freshness signals (dates, '2024' boilerplate, etc.)",
     "Ordinal score"),
    ("treat_source_earned", "treatment (T7, descriptive)",
     "curated list lookup", "binary {0,1}",
     "1 if domain is on a curated earned-media list (~250 domains)",
     "List-membership flag, NOT a clean causal treatment"),
    ("treat_source_brand", "treatment (auxiliary)",
     "curated list lookup", "binary {0,1}",
     "1 if domain is on a curated brand list", "Auxiliary"),
    ("treat_source_type", "treatment (auxiliary)",
     "curated list lookup", "category",
     "Source-type classification", "Auxiliary"),
    ("treat_brand", "treatment (auxiliary)",
     "curated list lookup", "binary {0,1}",
     "Brand flag (collapsed)", "Auxiliary"),
    ("treat_earned", "treatment (auxiliary)",
     "curated list lookup", "binary {0,1}",
     "Earned-media flag (collapsed)", "Auxiliary"),

    # T_code / T_llm pair (from regression_dataset.parquet — code + LLM-coded)
    ("T1_statistical_density_code", "treatment (T1_code)", "page-HTML parse",
     "binary {0,1}", "Code-derived binary indicator of statistical content",
     "From regression_dataset.parquet, simpler than T1a/T1b"),
    ("T1_statistical_density_llm", "treatment (T1_llm)", "LLM annotator",
     "binary {0,1}", "LLM-judged statistical density",
     "From regression_dataset.parquet"),
    ("T2_question_heading_code", "treatment (T2_code)", "page-HTML parse",
     "binary {0,1}", "Code-derived Q-heading indicator",
     "From regression_dataset.parquet, paired with T2a"),
    ("T2_question_heading_llm", "treatment (T2_llm)", "LLM annotator",
     "binary {0,1}", "LLM-judged Q-heading presence",
     "From regression_dataset.parquet"),
    ("T3_structured_data_code", "treatment (T3_code)", "page-HTML parse",
     "binary {0,1}", "Code-derived schema-markup indicator",
     "Older/narrower than treat_structured_data (T3)"),
    ("T3_structured_data_llm", "treatment (T3_llm)", "LLM annotator",
     "binary {0,1}", "LLM-judged schema markup presence",
     "LLM is more permissive (~2× positive rate vs code)"),
    ("T4_citation_authority_code", "treatment (T4_code)",
     "page-HTML parse", "binary {0,1}",
     "Code-derived citation-authority flag",
     "Paired with T4a/T4b"),
    ("T4_citation_authority_llm", "treatment (T4_llm)", "LLM annotator",
     "int (count)", "LLM-judged citation-authority count",
     "Survives Romano-Wolf at rank-stage"),
    ("has_llms_txt", "treatment (T_llms_txt)", "/.well-known check",
     "binary {0,1}", "1 if domain serves an llms.txt file",
     "Single-version flag"),

    # CONFOUNDERS — page HTML
    ("conf_word_count", "confounder (page-HTML)", "page-HTML parse",
     "int (words)", "Body text word count",
     "Code-derived, BeautifulSoup"),
    ("conf_readability", "confounder (page-HTML)", "page-HTML parse",
     "float", "Flesch-style readability score",
     "Code-derived"),
    ("conf_internal_links", "confounder (page-HTML)", "page-HTML parse",
     "int", "Count of same-domain hyperlinks",
     "Code-derived"),
    ("conf_outbound_links", "confounder (page-HTML)", "page-HTML parse",
     "int", "Count of off-domain hyperlinks",
     "Code-derived"),
    ("conf_images_alt", "confounder (page-HTML)", "page-HTML parse",
     "int", "Count of <img> with non-empty alt text",
     "Code-derived"),
    ("conf_https", "confounder (URL)", "URL inspection",
     "binary {0,1}", "1 if URL begins with https://",
     "Computed at fetch time"),

    # CONFOUNDERS — SERP-derived
    ("conf_title_has_kw", "confounder (SERP)", "SERP snippet",
     "binary {0,1}", "1 if SERP title contains the keyword",
     "From features.py:conf_title_has_kw"),
    ("conf_title_len", "confounder (SERP)", "SERP snippet",
     "int (chars)", "Character length of the SERP title",
     "Computed from search engine output"),
    ("conf_snippet_len", "confounder (SERP)", "SERP snippet",
     "int (chars)", "Character length of the SERP snippet",
     "Computed from search engine output"),
    ("conf_serp_position", "confounder (SERP)", "SERP rank",
     "int (1-based)", "Original SERP rank position",
     "Mechanically explains most of rank_delta variance"),

    # CONFOUNDERS — semantic
    ("conf_title_kw_sim", "confounder (embedding)", "sentence-transformers",
     "cosine ∈ [-1,1]", "cos(title, keyword)",
     "Locally computed embeddings"),
    ("conf_snippet_kw_sim", "confounder (embedding)", "sentence-transformers",
     "cosine ∈ [-1,1]", "cos(snippet, keyword)",
     "Locally computed embeddings"),
    ("conf_bm25", "confounder (IR)", "BM25 algorithm",
     "float (BM25 score)", "BM25 score of page body vs keyword",
     "Classical IR measure, computed locally"),

    # CONFOUNDERS — Brand
    ("conf_brand_recog", "confounder (brand)", "DataForSEO Whois",
     "binary {0,1}",
     "1 if domain has brand-scale visibility "
     "(organic_count ≥ 100k OR pos_1 ≥ 500)",
     "REPLACED 2026-05-24: was a code heuristic on domain string; "
     "now derived from DataForSEO Whois Overview metrics"),

    # CONFOUNDERS — DataForSEO Whois Overview (replaces Moz)
    ("conf_domain_authority", "confounder (authority)",
     "DataForSEO Whois Overview", "float (log10 scale)",
     "log10(organic_count + 1) — proxy for domain authority",
     "REPLACED 2026-05-24: was Moz DA at 22% coverage; "
     "now log10(organic_count+1) at 100% coverage"),
    ("conf_backlinks", "confounder (authority)",
     "DataForSEO Whois Overview", "int (count)",
     "Total organic ranking positions across Google "
     "(proxy for external visibility)",
     "REPLACED 2026-05-24: was Moz backlinks at 11% coverage; "
     "now organic_count at 95% coverage"),
    ("conf_referring_domains", "confounder (authority)",
     "DataForSEO Whois Overview", "int (count)",
     "Number of #1 organic Google positions held by the domain",
     "REPLACED 2026-05-24: was Moz referring_domains at 11% coverage; "
     "now organic_pos_1 at 95% coverage"),
    ("conf_dfs_paid_count", "confounder (authority)",
     "DataForSEO Whois Overview", "int (count)",
     "Total paid-ad ranking positions across Google",
     "NEW 2026-05-24, no Moz analogue"),
    ("conf_dfs_etv", "confounder (authority)",
     "DataForSEO Whois Overview", "float (USD)",
     "Estimated traffic value (organic visits × CPC)",
     "NEW 2026-05-24, no Moz analogue"),
    ("conf_dfs_domain_age_years", "confounder (authority)",
     "DataForSEO Whois Overview", "float (years)",
     "Years since domain was first registered",
     "NEW 2026-05-24"),

    # CONFOUNDERS — DataForSEO keyword-level
    ("dfs_keyword_difficulty", "confounder (keyword)",
     "DataForSEO bulk keyword difficulty", "0–100",
     "Keyword ranking difficulty score", "Keyword-level"),
    ("dfs_search_volume", "confounder (keyword)",
     "DataForSEO keyword overview", "int (monthly searches)",
     "Estimated monthly Google search volume", "Keyword-level"),
    ("dfs_cpc", "confounder (keyword)",
     "DataForSEO keyword overview", "float (USD)",
     "Avg. Google Ads cost-per-click", "Keyword-level"),
    ("dfs_competition", "confounder (keyword)",
     "DataForSEO keyword overview", "float [0,1]",
     "Google Ads competition index", "Keyword-level"),
    ("dfs_intent_commercial", "confounder (keyword)",
     "DataForSEO search intent", "float [0,1]",
     "Probability the query has commercial intent", "Keyword-level"),
    ("dfs_intent_informational", "confounder (keyword)",
     "DataForSEO search intent", "float [0,1]",
     "Probability informational intent", "Keyword-level"),
    ("dfs_intent_navigational", "confounder (keyword)",
     "DataForSEO search intent", "float [0,1]",
     "Probability navigational intent", "Keyword-level"),
    ("dfs_intent_transactional", "confounder (keyword)",
     "DataForSEO search intent", "float [0,1]",
     "Probability transactional intent", "Keyword-level"),

    # METADATA
    ("position", "metadata", "SERP engine", "int",
     "Original position (alias of pre_rank, kept for traceability)",
     "Redundant with pre_rank"),
    ("title", "metadata", "SERP snippet", "text",
     "SERP-shown title of the result", "For inspection only"),
    ("snippet", "metadata", "SERP snippet", "text",
     "SERP-shown description snippet", "For inspection only"),
    ("html_present", "metadata", "pipeline run", "binary",
     "1 if the page HTML was successfully fetched",
     "Failures excluded from page-HTML confounders"),
    ("extract_error", "metadata", "pipeline run", "text",
     "Reason the page-HTML extraction failed, if any",
     "Diagnostic; null for successful rows"),
    ("used_fallback", "metadata", "pipeline run", "binary",
     "1 if a fallback content-extraction was used",
     "Diagnostic"),

    # Build-time metadata (added by this script)
    ("build_timestamp_utc", "build metadata", "build_unified_dataset.py",
     "ISO 8601 UTC",
     "Timestamp when this unified parquet was assembled",
     "Set by this script"),
    ("build_script_version", "build metadata", "build_unified_dataset.py",
     "text", "Identifies which version of this script built the row",
     "Set by this script"),
]


def build_dictionary_df():
    return pd.DataFrame(REGISTRY, columns=[
        "column", "type", "source", "units", "description", "notes",
    ])


# ── Main assembly ──────────────────────────────────────────────────────────


def main():
    print(f"[unified] timestamp = {datetime.now(timezone.utc).isoformat()}")

    # 1. Union the four per-variant parquets (these already have DFS-merged
    #    confounders since the merge script ran earlier today).
    print("\n[1/5] Loading 4 per-variant parquets …")
    parts = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        df = pd.read_parquet(p)
        df["variant"] = v  # already there but ensure
        # normalize engine name
        if "search_engine" in df.columns:
            df["search_engine"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        # rename pool_size → pool if needed (some parquets use pool)
        if "serp_pool_size" in df.columns and "pool_size" not in df.columns:
            df = df.rename(columns={"serp_pool_size": "pool_size"})
        if "pool" in df.columns and "pool_size" not in df.columns:
            df = df.rename(columns={"pool": "pool_size"})
        parts.append(df)
        print(f"   {v:12s} rows={len(df):>7,d}  cols={df.shape[1]}")
    big = pd.concat(parts, ignore_index=True)
    print(f"  → unioned: {len(big):,} rows × {big.shape[1]} cols")

    # 2. Merge T_code / T_llm / has_llms_txt from regression_dataset on (kw,url)
    print("\n[2/5] Merging T_code/T_llm + has_llms_txt from regression_dataset.parquet …")
    reg = pd.read_parquet(MAIN / "regression_dataset.parquet")
    code_llm_cols = [c for c in reg.columns
                     if c.startswith(("T1_", "T2_", "T3_", "T4_"))
                     and ("_code" in c or "_llm" in c)]
    code_llm_cols.append("has_llms_txt")
    code_llm_cols = [c for c in code_llm_cols if c in reg.columns]
    keep = ["keyword", "url"] + code_llm_cols
    reg_small = reg[keep].drop_duplicates(subset=["keyword", "url"], keep="first")
    print(f"   regression_dataset → keeping {len(reg_small):,} unique (kw,url)"
          f" × {len(code_llm_cols)} code/llm cols")
    big = big.merge(reg_small, on=["keyword", "url"], how="left")
    print(f"  → merged: {len(big):,} rows × {big.shape[1]} cols")

    # 3. Add derived columns
    print("\n[3/5] Adding derived columns …")
    big["selected_by_llm"] = big["post_rank"].notna().astype(int)
    big["build_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    big["build_script_version"] = "build_unified_dataset.py v1 (2026-05-24)"

    # build run_id if missing
    if "run_id" not in big.columns:
        big["run_id"] = (big["search_engine"].astype(str) + "_"
                        + big["llm_model"].astype(str) + "_serp"
                        + big["pool_size"].astype(str) + "_top"
                        + big.get("top_n", 10).astype(str) + "_"
                        + big["variant"].astype(str))

    # 4. Order columns: identifiers → outcomes → treatments → confounders → metadata
    print("\n[4/5] Re-ordering columns by category …")
    cat_order = {
        "identifier": 1, "outcome (input)": 2, "outcome (LLM)": 2,
        "outcome (derived)": 2,
        "treatment (T1a)": 3, "treatment (T1b)": 3, "treatment (T1_code)": 3,
        "treatment (T1_llm)": 3, "treatment (T2a)": 3, "treatment (T2b)": 3,
        "treatment (T2_code)": 3, "treatment (T2_llm)": 3,
        "treatment (T3)": 3, "treatment (T3_code)": 3, "treatment (T3_llm)": 3,
        "treatment (T4a)": 3, "treatment (T4b)": 3, "treatment (T4_code)": 3,
        "treatment (T4_llm)": 3, "treatment (T5)": 3, "treatment (T6)": 3,
        "treatment (T7, descriptive)": 3,
        "treatment (T_llms_txt)": 3, "treatment (auxiliary)": 3,
        "confounder (page-HTML)": 4, "confounder (URL)": 4,
        "confounder (SERP)": 4, "confounder (embedding)": 4,
        "confounder (IR)": 4, "confounder (brand)": 4,
        "confounder (authority)": 4, "confounder (keyword)": 4,
        "metadata": 5, "build metadata": 6,
    }
    col_to_cat = {r[0]: r[1] for r in REGISTRY}
    ordered = sorted(
        [c for c in big.columns if c in col_to_cat],
        key=lambda c: (cat_order.get(col_to_cat[c], 99), c),
    )
    unknown = [c for c in big.columns if c not in col_to_cat]
    if unknown:
        print(f"   ⚠ {len(unknown)} unregistered columns kept at end: "
              f"{unknown[:8]}{'…' if len(unknown) > 8 else ''}")
    big = big[ordered + unknown]

    # 5. Save parquet + dictionary + README
    print("\n[5/5] Saving …")
    big.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"   parquet → {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1024/1024:.1f} MB)")

    dict_df = build_dictionary_df()
    dict_df.to_csv(OUT_DICT_CSV, index=False)
    print(f"   dictionary → {OUT_DICT_CSV}  ({len(dict_df)} rows)")

    write_readme(big, dict_df)
    print(f"   README → {OUT_README}")

    print(f"\n[done] {len(big):,} rows × {len(big.columns)} cols")
    print(f"       coverage of selected_by_llm = 1: "
          f"{big['selected_by_llm'].mean()*100:.1f}%")


def write_readme(big, dict_df):
    rows_per_var = big["variant"].value_counts().to_dict()
    sel_rate = big["selected_by_llm"].mean() * 100
    n_kw = big["keyword"].nunique()
    n_dom = big["domain"].nunique() if "domain" in big.columns else "?"

    txt = f"""# Unified GEODML dataset — {TODAY}

Single super-detailed parquet bundling every treatment, confounder, outcome,
and metadata field across all experiments and variants, designed for reviewer
verification of the EMNLP 2026 submission.

## Quick stats

- **Rows:** {len(big):,}  (one row per (keyword × url × variant × LLM × engine × pool))
- **Columns:** {len(big.columns)}
- **Unique keywords:** {n_kw:,}
- **Unique domains:** {n_dom:,}
- **Selection rate** (selected_by_llm = 1): **{sel_rate:.1f}%** — see "Sample frame" below

### Rows per variant

| Variant | rows |
|---|---|
{chr(10).join(f"| {v} | {n:,} |" for v, n in rows_per_var.items())}

## ⚠ Sample frame (READ FIRST)

This parquet contains **every URL that the LLM included in its top-N output** —
i.e. `selected_by_llm = 1` for all rows. The complementary set (URLs the search
engine returned but the LLM dropped) is **not** in this file.

**What that means in practice:**

- ✅ Use this file for any analysis whose outcome is `rank_delta`, `post_rank`,
  or any continuous content-treatment effect on the admitted-URL subset.
  This is the right sample frame for the rank-effect DML (Study 1).
- ⚠ For binary-admission analyses (outcome = "did the LLM keep this URL"),
  the correct sample frame is the **full SERP pool** (admitted + rejected).
  Build it from `~/Hamburg/GEODML_Analysis/geodml_data/data/serp/phase0_top*.parquet`
  via `scripts/dml_selected.py::build_pool_table()`. The Study 2 results
  in the paper come from that pool, NOT from this file.

The per-variant LLM-output parquets only ever recorded URLs the LLM kept, so
admitted-only is the natural scope of THIS parquet.

## Files

- `unified_{TODAY}.parquet` — the data (this file)
- `unified_{TODAY}_dictionary.csv` — column-level metadata: type, source, units, description, notes
- `unified_{TODAY}_README.md` — this file

## Schema overview

The {len(dict_df)} documented columns fall into 5 categories:

| Category | Examples | Count |
|---|---|---|
| **Identifiers** | keyword, url, domain, variant, llm_model, run_id | {(dict_df['type']=='identifier').sum()} |
| **Outcomes** | pre_rank, post_rank, rank_delta, selected_by_llm | {(dict_df['type'].str.startswith('outcome')).sum()} |
| **Treatments** | treat_*, T*_code, T*_llm, has_llms_txt | {(dict_df['type'].str.startswith('treatment')).sum()} |
| **Confounders** | conf_*, dfs_* | {(dict_df['type'].str.startswith('confounder')).sum()} |
| **Metadata** | title, snippet, html_present, build_timestamp_utc | {(dict_df['type'].isin(['metadata', 'build metadata'])).sum()} |

## Provenance highlights

- **Page-HTML confounders** parsed from fetched HTML via
  `interpretability/pipeline/features.py` (`conf_word_count`, `conf_readability`,
  `conf_internal_links`, `conf_outbound_links`, `conf_images_alt`, `conf_https`).
- **SERP-derived confounders** computed from DuckDuckGo / SearXNG snippet output
  (`conf_title_has_kw`, `conf_title_len`, `conf_snippet_len`, `conf_serp_position`).
- **Semantic embeddings** computed locally via sentence-transformers
  (`conf_title_kw_sim`, `conf_snippet_kw_sim`, `conf_bm25`).
- **DataForSEO Whois Overview** signals fetched 2026-05-24 (POST replaces Moz):
  - `conf_domain_authority` = log10(organic_count+1) — 100 % coverage
  - `conf_backlinks` = organic_count (Google organic-position count)
  - `conf_referring_domains` = organic_pos_1 (# of #1 organic positions)
  - `conf_brand_recog` = brand-scale visibility binary (≥ 100k organic OR ≥ 500 pos-1)
  - `conf_dfs_paid_count`, `conf_dfs_etv`, `conf_dfs_domain_age_years` (NEW)
- **DataForSEO keyword-level** signals (`dfs_search_volume`, `dfs_cpc`,
  `dfs_competition`, `dfs_intent_*`) bulk-fetched during pipeline build.

## Treatment-coding redundancy

For T1–T4 treatment families, TWO codings are available side-by-side:

| Family | Code-derived (this file) | LLM-coded (this file) |
|---|---|---|
| T1 stats density | `T1_statistical_density_code` (or `treat_stats_density`, `treat_stats_present`) | `T1_statistical_density_llm` |
| T2 Q-headings   | `T2_question_heading_code` (or `treat_question_headings`) | `T2_question_heading_llm` |
| T3 schema       | `T3_structured_data_code` (or `treat_structured_data`)    | `T3_structured_data_llm` |
| T4 citations    | `T4_citation_authority_code` (or `treat_ext_citations_any`, `treat_auth_citations`) | `T4_citation_authority_llm` |

The code/LLM agreement audit (Cohen's κ, Pearson r) for T1–T4 is documented in
`docs/dml_survivors_2026-05-24.md` §"Notable from cross-referencing with
your code/LLM annotator audit".

## How to use this file

```python
import pandas as pd
df = pd.read_parquet("unified_{TODAY}.parquet")
dict_df = pd.read_csv("unified_{TODAY}_dictionary.csv")

# Filter to a single experiment cell
biased = df[df["variant"] == "biased"]

# Get treatment column names
treatment_cols = dict_df[dict_df["type"].str.startswith("treatment")]["column"].tolist()
print(treatment_cols)

# Get confounders with their source
print(dict_df[dict_df["type"].str.startswith("confounder")][["column", "source", "units"]])

# Re-verify a DML estimate: PLR Robinson-style on rank_delta for one treatment
# (Using the sklearn / scipy implementations referenced in scripts/full_paper_analysis.py)
```

## Reproducibility

To rebuild from current ~/geodml_data state:

```bash
python scripts/build_unified_dataset.py
```

Reads from `~/geodml_data/data/main/` (post-DataForSEO merge, dated 2026-05-24).
"""
    OUT_README.write_text(txt)


if __name__ == "__main__":
    sys.exit(main() or 0)
