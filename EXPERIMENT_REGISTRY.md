# GEODML Experiment Registry

> Complete mapping of scripts, hyperparameters, experiments, and output files.
> Last updated: 2026-03-24

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Scripts Reference](#scripts-reference)
4. [Experiment Registry](#experiment-registry)
5. [Output File Index](#output-file-index)
6. [Hyperparameter Reference](#hyperparameter-reference)

---

## Project Overview

**GEODML** uses Double Machine Learning (DML) to estimate the causal effect of on-page features on how LLMs re-rank search engine results for 50 B2B SaaS keywords.

**Core question**: When an LLM re-ranks SERP results, which page-level treatments causally affect whether a page gets promoted or demoted?

**Study period**: February 2026 (Feb 11 - Feb 24), Hamburg, Germany.

---

## Pipeline Architecture

```
keywords.txt (50 B2B SaaS keywords)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: Data Acquisition                          │
│  Scripts: run_ai_search.py, pipeline/gather_data.py │
│  Search engine (SearXNG/DDG/etc.) → raw SERP        │
│  LLM re-ranking → ranked domains                    │
│  Output: experiment.json, rankings.csv               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: Feature Extraction                        │
│  Scripts: run_page_scraper.py, extract_features.py  │
│  HTML fetch → code-based features (T1-T4, X1-X10)  │
│  Optional: LLM-based treatment eval, PageRank, etc. │
│  Output: features.csv, features_new.csv              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3: Data Assembly                             │
│  Scripts: clean_data.py, build_clean_dataset.py     │
│  Merge rankings + features + experiment metadata    │
│  Output: geodml_dataset.csv                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4: Causal Analysis                           │
│  Scripts: pipeline/analyze.py, run_dml_study.py     │
│  DML (PLR/IRM) with LGBM/RF nuisance learners      │
│  Output: all_experiments.csv, summary.json, plots   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5: Visualization                             │
│  Script: pipeline/visualize.py                      │
│  Publication-quality forest plots, heatmaps, etc.   │
│  Output: *.png figures                               │
└─────────────────────────────────────────────────────┘
```

---

## Scripts Reference

### Top-Level Entry Points

| Script | Purpose | Key Inputs | Key Outputs |
|--------|---------|------------|-------------|
| `run_ai_search.py` | SERP retrieval + LLM re-ranking | `keywords.txt`, `.env.local` | `results/{engine}_{model}_{date}.json/.csv` |
| `run_page_scraper.py` | HTML fetch + feature extraction | CSV from run_ai_search | `results/page_features_{tag}.csv`, `results/dml_dataset_{tag}.csv` |
| `build_clean_dataset.py` | Merge rankings + features (v1) | `results/dml_dataset_searxng.csv`, experiment JSON | `data/geodml_dataset.csv` |
| `run_dml_study.py` | DML analysis (v1, standalone) | `data/geodml_dataset.csv` | `results/dml_results.csv/.json`, `results/dml_coefficients.png` |

### Pipeline Scripts (`pipeline/`)

| Script | Purpose | Key Inputs | Key Outputs |
|--------|---------|------------|-------------|
| `gather_data.py` | End-to-end: SERP → LLM → HTML → features | `keywords.txt` | `output/experiment.json`, `rankings.csv`, `features.csv`, `html_cache/` |
| `extract_features.py` | Enrich features (new treatments T1a-T7, new confounders) | experiment JSON, `html_cache/`, `data/geodml_dataset.csv` | `pipeline/intermediate/features_new.csv`, `embeddings.npz` |
| `clean_data.py` | Merge all data into DML-ready CSV | `output/rankings.csv`, `features.csv`, `experiment.json`, `features_new.csv` | `output/geodml_dataset.csv` |
| `analyze.py` | DML causal inference (configurable) | `output/geodml_dataset.csv` | `output/results/all_experiments.csv`, `summary.json`, plots |
| `visualize.py` | Publication figures | `all_experiments.csv`, `confounder_importances.csv` | 6 PNG plots |
| `rebuild_features.py` | Rebuild features from cached HTML (recovery) | `html_cache/`, `rankings.csv` | `features.csv` |

### Source Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `config.py` | Environment vars (HF_TOKEN, SEARXNG_URL, API keys), constants (TOP_N=10) |
| `llm_ranker.py` | LLM re-ranking via HF Inference API (prompt building, domain parsing, fallback) |
| `searxng_client.py` | SearXNG client with DuckDuckGo fallback |
| `engine_scraper.py` | Multi-engine dispatcher (SearXNG, DDG, Google, Yahoo, Kagi, Brave, SerpAPI) |
| `page_features.py` | HTML feature extraction: T1-T4 treatments, X3/X6/X7/X9/X10 confounders, LLM digest |
| `experiment_context.py` | Provenance: IP, geolocation, machine info, library versions |
| `results_io.py` | JSON/CSV serialization for experiment results |
| `keywords.py` | Load keywords from `keywords.txt` |

---

## Experiment Registry

### Experiment 1: Original Small-Pool Study (Llama-3.3-70B)

**Date**: 2026-02-16
**Script chain**: `run_ai_search.py` → `run_page_scraper.py` → `build_clean_dataset.py` → `run_dml_study.py`

| Parameter | Value |
|-----------|-------|
| Keywords | 50 B2B SaaS |
| Search engine | SearXNG (Google + Bing + DDG + Brave + Startpage) |
| SERP results fetched | 20 per keyword |
| LLM re-ranks top | 10 |
| LLM model | `meta-llama/Llama-3.3-70B-Instruct` |
| LLM temperature | 0.1 |
| LLM max_tokens | 500 |
| Observations | 492 (355 with valid rank_delta) |
| DML method | PLR |
| Nuisance learners | LGBM (primary), RF (sensitivity) |
| LGBM params | n_estimators=200, lr=0.05, max_depth=5, leaves=31 |
| RF params | n_estimators=200, max_depth=5 |
| N folds | 5 |
| Treatments | T1-T4 code-based + T1-T4 LLM-based (8 total) |
| Confounders | X1 (domain auth), X2 (domain age), X3 (word count), X6 (readability), X7 (internal links), X7B (outbound links), X8 (kw difficulty), X9 (images alt) |
| Dropped confounders | X10 (zero variance), X4 (0% coverage) |
| Imputation | Median |
| Scaling | StandardScaler |

**Output files**:
| File | Description |
|------|-------------|
| `results/searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json` | Raw SERP + LLM re-ranking (1.1 MB) |
| `results/searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.csv` | Flattened rankings |
| `results/page_features_searxng.json` | Extracted features (458 entries) |
| `data/geodml_dataset.csv` | Clean DML dataset (492 rows, 27 cols) |
| `results/dml_results.json` | DML results with 8 treatments |
| `results/dml_results.csv` | Summary effects table |
| `results/dml_coefficients.png` | Coefficient forest plot |

**Key results** (rank_delta, PLR, LGBM):

| Treatment | Coef (θ) | p-value | Significant? |
|-----------|----------|---------|-------------|
| T1 Statistical Density (code) | +0.269 | 0.082 | marginal |
| T2 Question Headings (code) | +0.769 | 0.073 | marginal |
| T3 Structured Data (code) | +0.163 | 0.760 | no |
| T4 Citation Authority (code) | -0.440 | 0.487 | no |
| T1 Statistical Density (LLM) | +0.010 | 0.715 | no |
| T2 Question Headings (LLM) | +0.016 | 0.973 | no |
| T3 Structured Data (LLM) | +0.242 | 0.593 | no |
| T4 Citation Authority (LLM) | +0.112 | 0.674 | no |

---

### Experiment 2: Multi-Outcome Study with Extended Tests

**Date**: 2026-02-16 to 2026-02-18
**Script chain**: Same data as Exp 1, re-analyzed with 3 outcome specs

| Parameter | Value |
|-----------|-------|
| Same as Exp 1 but... | |
| Outcomes | `rank_delta`, `pre_rank`, `post_rank` |
| DML methods | PLR + IRM |
| Total experiments | 48 (3 outcomes × 4 treatments × 2 paths × 2 methods) |

**Output files**:
| File | Description |
|------|-------------|
| `test/results/all_experiments.csv` | 32 experiments (pre_rank + post_rank) |
| `test_diff/results/all_experiments.csv` | 16 experiments (rank_delta) |
| `test/results/heatmap_pvalues.png` | P-value heatmap |
| `test/results/coef_grid.png` | Coefficient grid |
| `test_full/results/full_diagnostics.csv` | Nuisance R², OLS coefficients, RMSE |
| `test_full_rf/results/full_diagnostics.csv` | Same with RF learner |
| `FINDINGS.md` | Narrative findings report |

**Key results** (cross-reference table, PLR, code-based):

| Treatment | pre_rank | post_rank | rank_delta |
|-----------|----------|-----------|------------|
| T1 Statistical Density | +0.315 (p=0.170) | **+0.101 (p=0.024)** | +0.186 (p=0.214) |
| T2 Question Headings | +0.909 (p=0.115) | -0.356 (p=0.233) | **+1.198 (p=0.009)** |
| T3 Structured Data | +0.145 (p=0.803) | **-0.719 (p=0.048)** | +0.812 (p=0.103) |
| T4 Citation Authority | -1.020 (p=0.219) | -0.740 (p=0.125) | -0.650 (p=0.311) |

---

### Experiment 3: Large-Pool Study (50 SERP / 20 LLM Re-rank)

**Date**: 2026-02-17
**Script chain**: `run_ai_search.py` (modified params) → `run_page_scraper.py` → `run_dml_study.py`

| Parameter | Value |
|-----------|-------|
| Keywords | 50 B2B SaaS (same) |
| Search engine | SearXNG (same) |
| **SERP results fetched** | **50 per keyword** |
| **LLM re-ranks top** | **20** |
| LLM model | `meta-llama/Llama-3.3-70B-Instruct` |
| Observations | 996 (374 with valid rank_delta) |
| DML/confounders | Same as Exp 1 |

**Output files** (all under `50_larger/`):
| File | Description |
|------|-------------|
| `50_larger/searxng_Llama-3.3-70B-Instruct_2026-02-17_2225.json/.csv` | Raw results |
| `50_larger/page_features_searxng.csv` | Features (944 rows) |
| `50_larger/data/geodml_dataset.csv` | Clean dataset (997 rows) |
| `50_larger/dml_results.csv/.json` | DML results |
| `50_larger/dml_coefficients.png` | Coefficient plot |
| `50_larger/test/`, `test_full/`, `test_full_rf/` | Extended analysis |
| `50_larger/figures/fig1-fig9.png` | Publication figures |

---

### Experiment 4: Comparative Analysis (Small vs Large Pool)

**Date**: 2026-02-18
**Script**: `both_analysis/run_comparative_dml.py`

| Parameter | Value |
|-----------|-------|
| Datasets compared | 20-SERP/10-rerank (492 rows) vs 50-SERP/20-rerank (996 rows) |
| Treatments | T1-T4 code-based + T1-T4 LLM-based |
| Outcomes | rank_delta, pre_rank, post_rank |
| Methods | PLR + IRM |
| Total experiments | 96 (48 per dataset) |
| Significant at p<0.05 | 7 findings |

**Output files**:
| File | Description |
|------|-------------|
| `both_analysis/results/all_experiments.csv` | 96 experiments |
| `both_analysis/results/summary.json` | Metadata |
| `both_analysis/results/descriptive_stats_20serp.csv` | Small pool descriptives |
| `both_analysis/results/descriptive_stats_50serp.csv` | Large pool descriptives |
| `both_analysis/figures/fig1_comparative_forest.png` | Side-by-side forest plots |
| `both_analysis/figures/fig2_coefficient_scatter.png` | 20-SERP vs 50-SERP scatter |
| `both_analysis/figures/fig3_pvalue_heatmap.png` | Full p-value heatmap |
| `both_analysis/figures/fig4_effect_comparison.png` | Grouped bar chart |
| `both_analysis/figures/fig5_dml_vs_ols.png` | DML vs OLS |
| `both_analysis/figures/fig6_plr_vs_irm.png` | Method sensitivity |
| `both_analysis/figures/fig7_multi_outcome_forest.png` | All outcomes |
| `both_analysis/figures/fig8_summary_table.png` | Summary table |
| `both_analysis/figures/fig9_dataset_descriptives.png` | Variable distributions |
| `both_analysis/COMPARATIVE_FINDINGS.md` | Narrative analysis |

**Key results** (rank_delta, PLR, code-based):

| Treatment | Small Pool (20/10) | Large Pool (50/20) | Direction consistent? |
|-----------|-------------------|-------------------|----------------------|
| T1 Statistical Density | **+0.39 (p=0.023)** | +0.03 (p=0.93) | Effect disappears |
| T2 Question Headings | **+1.07 (p=0.019)** | -0.99 (p=0.29) | **Reversal** |
| T3 Structured Data | **+1.10 (p=0.033)** | -1.45 (p=0.15) | **Reversal** |
| T4 Citation Authority | -1.12 (p=0.10) | -1.39 (p=0.55) | Consistent (both NS) |

**Headline finding**: LLM behavior changes qualitatively with pool size. In the small pool it promotes FAQ-style pages; in the large pool it penalizes them as generic SEO tactics.

---

### Experiment 5: DeepSeek R1 Pipeline (New Treatments)

**Date**: 2026-02-23 to 2026-02-24
**Script chain**: `pipeline/gather_data.py` → `pipeline/rebuild_features.py` → `pipeline/extract_features.py` → `pipeline/clean_data.py` → `pipeline/analyze.py`

| Parameter | Value |
|-----------|-------|
| Keywords | 50 B2B SaaS |
| Search engine | SearXNG |
| LLM model | DeepSeek R1 (via HF API) |
| SERP results | 20 per keyword |
| LLM re-ranks top | 10 |
| Observations | 416-446 (varies by treatment) |
| DML method | PLR |
| Learners | LGBM + RF |
| N folds | 5 |
| **Treatments (10 new)** | T1a (stats binary), T1b (stats density), T2a (question headings binary), T2b (structural modularity), T3 (structured data expanded), T4a (external citations binary), T4b (authority citations count), T5 (topical competence cosine), T6 (freshness ordinal 0-4), T7 (source earned vs brand) |
| **Confounders (16 new)** | conf_title_kw_sim, conf_snippet_kw_sim, conf_title_len, conf_snippet_len, conf_brand_recog, conf_title_has_kw, conf_word_count, conf_readability, conf_internal_links, conf_outbound_links, conf_images_alt, conf_bm25, conf_domain_authority, conf_backlinks, conf_referring_domains, conf_serp_position |
| Total experiments | 60 (10 treatments × 3 outcomes × 2 learners) |

**Output files**:
| File | Description |
|------|-------------|
| `output/deepseek-r1/experiment.json` | Raw SERP + LLM data |
| `output/deepseek-r1/rankings.csv` | Flattened rankings |
| `output/deepseek-r1/features.csv` | Code-based features |
| `output/deepseek-r1/html_cache/` | Cached HTML (171 MB) |
| `output/deepseek-r1/geodml_dataset.csv` | Clean dataset |
| `pipeline/intermediate/features_new.csv` | Enriched features (32 cols) |
| `pipeline/intermediate/embeddings.npz` | Cached embeddings |
| `pipeline/intermediate/validation_report.txt` | Data quality report |
| `pipeline/results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/all_experiments.csv` | All 60 DML results |
| `pipeline/results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/summary.json` | Metadata |
| `pipeline/results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/confounder_importances.csv` | Feature importance |
| `pipeline/results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/*.png` | 7 diagnostic plots |

**Key results** (rank_delta, PLR, LGBM):

| Treatment | Coef (θ) | p-value | Significant? |
|-----------|----------|---------|-------------|
| T1a Stats Present (binary) | +0.312 | 0.168 | no |
| T1b Stats Density (continuous) | +0.038 | 0.484 | no |
| **T2a Question Headings (binary)** | **+0.714** | **0.010** | **yes (\*\*\*)** |
| T2b Structural Modularity (count) | +0.002 | 0.691 | no |
| T3 Structured Data (expanded) | -0.054 | 0.793 | no |
| T4a External Citations (binary) | -0.250 | 0.600 | no |
| **T4b Authority Citations (count)** | **+0.392** | **0.038** | **yes (\*\*)** |
| T5 Topical Competence (cosine) | +0.427 | 0.558 | no |
| T6 Freshness (ordinal) | -0.032 | 0.552 | no |
| T7 Source Earned | -1.175 | 0.149 | no |

**post_rank significant results** (LGBM):
- T2a Question Headings: **-0.822 (p=0.0007)** — LLM places these higher
- T2b Structural Modularity (RF): **-0.010 (p=0.043)** — small but significant

---

### Experiment 6: Llama-3.3-70B with New Pipeline (New Treatments)

**Date**: 2026-02-24
**Script chain**: Same pipeline as Exp 5, different LLM

| Parameter | Value |
|-----------|-------|
| Same as Exp 5 but... | |
| LLM model | `meta-llama/Llama-3.3-70B-Instruct` |
| Observations | 355 (rank_delta) |
| Total experiments | 60 |

**Output files**:
| File | Description |
|------|-------------|
| `pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/all_experiments.csv` | All 60 results |
| `pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/summary.json` | Metadata |
| `pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/*.png` | Diagnostic plots |

---

### Exploratory Runs (Secondary Search Engines)

**Date**: 2026-02-11 to 2026-02-16

| Run | Engine | LLM | Output File |
|-----|--------|-----|-------------|
| DDG + Qwen | DuckDuckGo | Qwen2.5-72B-Instruct | `results/duckduckgo_Qwen2.5-72B-Instruct_2026-02-11_1709.json` |
| DDG + Qwen (v2) | DuckDuckGo | Qwen2.5-72B-Instruct | `results/duckduckgo_Qwen2.5-72B-Instruct_2026-02-11_1727.json` |
| Brave + Qwen | Brave | Qwen2.5-72B-Instruct | `results/brave_Qwen2.5-72B-Instruct_2026-02-11_1659.json` |
| Yahoo + Qwen | Yahoo | Qwen2.5-72B-Instruct | `results/yahoo_Qwen2.5-72B-Instruct_2026-02-11_1646.json` |
| DDG no LLM | DuckDuckGo | none | `results/duckduckgo_nollm_2026-02-16_0915.json` |
| Brave no LLM | Brave | none | `results/brave_nollm_2026-02-16_0917.json` |
| SerpAPI + Llama | SerpAPI | Llama-3.3-70B | `results/serpapi_Llama-3.3-70B-Instruct_*.json` (multiple) |
| SerpAPI no LLM | SerpAPI | none | `results/serpapi_nollm_*.json` |

---

## Output File Index

### By Directory

```
results/
├── searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json    # Exp 1: primary SERP data
├── dml_results.json                                         # Exp 1: DML results
├── dml_coefficients.png                                     # Exp 1: coefficient plot
├── page_features_searxng.json                               # Exp 1: extracted features
├── html_cache/                                              # Cached HTML (173 MB)
├── duckduckgo_*.json, brave_*.json, yahoo_*.json            # Exploratory runs
└── serpapi_*.json                                           # Exploratory runs

data/
├── geodml_dataset.csv                                       # Exp 1: clean dataset (492 rows)
├── url_mapping.csv                                          # URL hash → full URL
└── README.md                                                # Data dictionary

test/ test_diff/ test_full/ test_full_rf/                    # Exp 2: extended analysis
├── results/all_experiments.csv
├── results/full_diagnostics.csv
└── results/*.png

50_larger/                                                    # Exp 3: large pool
├── data/geodml_dataset.csv                                  # 997 rows
├── searxng_Llama-3.3-70B-Instruct_2026-02-17_2225.json
├── page_features_searxng.csv
├── dml_results.csv/.json
├── figures/fig1-fig9.png
└── test/ test_full/ test_full_rf/

both_analysis/                                                # Exp 4: comparative
├── results/all_experiments.csv                              # 96 experiments
├── results/summary.json
├── figures/fig1-fig9.png
└── COMPARATIVE_FINDINGS.md

output/
├── geodml_dataset.csv                                       # Pipeline output
└── deepseek-r1/                                             # Exp 5: DeepSeek R1
    ├── experiment.json, rankings.csv, features.csv
    ├── geodml_dataset.csv
    └── html_cache/

pipeline/
├── intermediate/
│   ├── features_new.csv                                     # Enriched features
│   ├── embeddings.npz                                       # Cached embeddings
│   └── validation_report.txt                                # Data quality
├── results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/ # Exp 5
│   ├── all_experiments.csv, summary.json
│   └── *.png (7 plots)
└── results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/ # Exp 6
    ├── all_experiments.csv, summary.json
    └── *.png (7 plots)
```

---

## Hyperparameter Reference

### LLM Re-Ranking

| Parameter | Value | Set In |
|-----------|-------|--------|
| Default model | `meta-llama/Llama-3.3-70B-Instruct` | `src/llm_ranker.py:15` |
| Temperature | 0.1 | `src/llm_ranker.py:145` |
| Max tokens | 500 | `src/llm_ranker.py:144` |
| Top-N domains | 10 | `src/config.py:18` |
| Prompt style | Bare keyword, no sentence wrapping | `src/llm_ranker.py:25` |
| DeepSeek R1 handling | Strip `<think>` tags via regex | `src/llm_ranker.py:150` |

### Search Engine

| Parameter | Value | Set In |
|-----------|-------|--------|
| Default engine | SearXNG on localhost:8888 | `src/config.py:10` |
| SERP results requested | 20 (default) or 50 (large pool) | `pipeline/gather_data.py` CLI arg |
| Rate limiting | 2-5s random sleep | `src/engine_scraper.py`, `src/searxng_client.py` |
| Fallback | DuckDuckGo via `ddgs` library | `src/searxng_client.py` |

### HTML Feature Extraction

| Parameter | Value | Set In |
|-----------|-------|--------|
| Fetch timeout | 30s | `pipeline/gather_data.py`, `run_page_scraper.py` |
| Max HTML size | 5 MB | `pipeline/gather_data.py`, `run_page_scraper.py` |
| User-Agent | Firefox 128.0 | `run_page_scraper.py` |
| Max body chars for LLM digest | 3000 | `src/page_features.py` |

### DML Analysis

| Parameter | Value | Set In |
|-----------|-------|--------|
| Method | PLR (default), IRM optional | `pipeline/analyze.py` CLI arg |
| LGBM: n_estimators | 200 | `pipeline/analyze.py`, `run_dml_study.py` |
| LGBM: learning_rate | 0.05 | `pipeline/analyze.py`, `run_dml_study.py` |
| LGBM: max_depth | 5 | `pipeline/analyze.py`, `run_dml_study.py` |
| LGBM: num_leaves | 31 | `pipeline/analyze.py`, `run_dml_study.py` |
| RF: n_estimators | 200 | `pipeline/analyze.py`, `run_dml_study.py` |
| RF: max_depth | 5 | `pipeline/analyze.py`, `run_dml_study.py` |
| N folds (cross-fitting) | 5 | `pipeline/analyze.py`, `run_dml_study.py` |
| Score function | "partialling out" | `run_dml_study.py` |
| Imputation | Median (sklearn SimpleImputer) | `pipeline/analyze.py`, `run_dml_study.py` |
| Scaling | StandardScaler | `pipeline/analyze.py`, `run_dml_study.py` |

### Treatment Definitions

**Legacy (4 treatments, code + LLM = 8 vars)**:

| ID | Name | Type | Measurement |
|----|------|------|-------------|
| T1 | Statistical Density | continuous | Unique numbers/percentages/dates per 500 words |
| T2 | Question Headings | binary | H2/H3 starting with What/How/Why/etc. |
| T3 | Structured Data | binary | JSON-LD @type in {faqpage, product, howto} |
| T4 | Citation Authority | count | Outbound links to .edu/.gov/academic |

**New (10 treatments, pipeline v2)**:

| ID | Name | Type | Measurement |
|----|------|------|-------------|
| T1a | Stats Present | binary | Any statistics present |
| T1b | Stats Density | continuous | Density per 500 words |
| T2a | Question Headings | binary | FAQ-style H2/H3 |
| T2b | Structural Modularity | count | Number of distinct heading sections |
| T3 | Structured Data (expanded) | binary | Expanded JSON-LD types |
| T4a | External Citations | binary | Any external citations |
| T4b | Authority Citations | count | Links to .edu/.gov/academic |
| T5 | Topical Competence | continuous (cosine) | Keyword-content semantic similarity |
| T6 | Freshness | ordinal 0-4 | Recency of dated content |
| T7 | Source Earned | binary | Earned media (G2, Capterra, etc.) vs brand |

### Confounder Definitions

**Legacy (8 confounders)**:

| ID | Name | Source |
|----|------|--------|
| X1 | Domain Authority | Open PageRank API |
| X2 | Domain Age (years) | WHOIS |
| X3 | Word Count | HTML parsing |
| X6 | Readability (Flesch-Kincaid) | textstat |
| X7 | Internal Links | HTML parsing |
| X7B | Outbound Links | HTML parsing |
| X8 | Keyword Difficulty | Mean domain authority of top-10 |
| X9 | Images with Alt Text | HTML parsing |

**New (16 confounders, pipeline v2)**:

| Name | Source |
|------|--------|
| conf_title_kw_sim | Sentence embedding cosine similarity |
| conf_snippet_kw_sim | Sentence embedding cosine similarity |
| conf_title_len | Character count |
| conf_snippet_len | Character count |
| conf_brand_recog | Lookup against 80+ known B2B SaaS domains |
| conf_title_has_kw | Binary keyword-in-title |
| conf_word_count | HTML parsing |
| conf_readability | Flesch-Kincaid grade |
| conf_internal_links | HTML parsing |
| conf_outbound_links | HTML parsing |
| conf_images_alt | HTML parsing |
| conf_bm25 | BM25 content relevance score |
| conf_domain_authority | Open PageRank / MOZ |
| conf_backlinks | MOZ API |
| conf_referring_domains | MOZ API |
| conf_serp_position | Original SERP position |

---

## Summary of Significant Findings Across All Experiments

| Finding | Experiment | Treatment | Outcome | θ | p-value |
|---------|-----------|-----------|---------|---|---------|
| Question headings promote (small pool) | Exp 2 | T2 code | rank_delta | +1.198 | 0.009 |
| Structured data improves LLM rank | Exp 2 | T3 code | post_rank | -0.719 | 0.048 |
| Stats density slightly penalizes | Exp 2 | T1 code | post_rank | +0.101 | 0.024 |
| Question headings promote (DeepSeek) | Exp 5 | T2a | rank_delta | +0.714 | 0.010 |
| Question headings improve LLM rank (DeepSeek) | Exp 5 | T2a | post_rank | -0.822 | 0.0007 |
| Authority citations promote (DeepSeek) | Exp 5 | T4b | rank_delta | +0.392 | 0.038 |
| Earned media worsens LLM rank (large pool, RF) | Exp 5 | T7 | post_rank | +1.645 | 0.028 |
| Question headings demote (large pool, LLM meas.) | Exp 4 | T2 LLM | rank_delta | -2.92 | 0.002 |
| Pool size reverses T2/T3 effects | Exp 4 | T2, T3 | rank_delta | see table | <0.05 |

**Cross-experiment consistency**: T2 (Question Headings) is the most robust finding — significant across Exp 2, Exp 4, and Exp 5, and across both Llama-3.3-70B and DeepSeek R1. Its effect reverses with pool size (Exp 4), suggesting LLM re-ranking behavior is context-dependent.

---

## LLM Models Tested

| Model | Used In | API |
|-------|---------|-----|
| `meta-llama/Llama-3.3-70B-Instruct` | Exp 1-4, 6 | HuggingFace Inference |
| `Qwen/Qwen2.5-72B-Instruct` | Exploratory runs | HuggingFace Inference |
| `deepseek-ai/DeepSeek-R1` | Exp 5 | HuggingFace Inference |

## Search Engines Tested

| Engine | Used In | Method |
|--------|---------|--------|
| SearXNG | Primary (all experiments) | Local Docker/Apptainer container |
| DuckDuckGo | Exploratory, fallback | `ddgs` Python library |
| Brave Search | Exploratory | Brave API |
| Yahoo | Exploratory | Web scraping |
| SerpAPI (Google) | Exploratory | SerpAPI |
