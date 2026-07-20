---
language:
- en
license: cc-by-4.0
pretty_name: "GEODML — Paper-Size Experiment (GEO Causal Inference)"
size_categories:
- 10K<n<100K
task_categories:
- tabular-regression
- tabular-classification
tags:
- causal-inference
- double-machine-learning
- seo
- generative-engine-optimization
- llm-reranking
- search
- b2b-saas
configs:
- config_name: main
  description: "Primary 65,203-row table: treatments + confounders + outcomes, all runs merged."
  data_files:
  - split: train
    path: data/main/full_experiment_data.parquet
- config_name: main_pre_dfs
  description: "Same as main but without the 8 DataForSEO confounder columns (pre-enrichment)."
  data_files:
  - split: train
    path: data/main/regression_dataset.parquet
- config_name: dml_results
  description: "Double/Debiased ML results (post-DataForSEO run). 570 fits × 2 outcomes × 19 treatments × 15 subsets."
  data_files:
  - split: long
    path: data/dml_results/dml_results_long.parquet
  - split: pivot_rank_delta
    path: data/dml_results/dml_pivot_rank_delta.parquet
  - split: pivot_post_rank
    path: data/dml_results/dml_pivot_post_rank.parquet
  - split: multi_treatment
    path: data/dml_results/dml_multi_treatment.parquet
  - split: multi_treatment_joint
    path: data/dml_results/dml_multi_treatment_study1_joint.parquet
  - split: multi_treatment_partial
    path: data/dml_results/dml_multi_treatment_study2_partial.parquet
  - split: confounder_audit
    path: data/dml_results/confounder_audit.parquet
  - split: confounder_loo_r2
    path: data/dml_results/confounder_loo_r2.parquet
  - split: confounder_ols_significance
    path: data/dml_results/confounder_ols_significance.parquet
  - split: nuisance_r2
    path: data/dml_results/nuisance_r2.parquet
  - split: variance_explained
    path: data/dml_results/variance_explained.parquet
- config_name: robust_winners
  description: "DML on the robust-winners frame: (keyword,url) pairs the LLM picked in its top-10 under both serp20 and serp50 pools. 152 fits across 4 (engine,model) categories × 19 treatments × 2 outcomes, with serp_pool_size partialled out as a confounder. See docs/robust-winners-analysis-2026-04-26.md."
  data_files:
  - split: long
    path: data/dml_results/dml_robust_winners.parquet
  - split: pivot
    path: data/dml_results/dml_robust_winners_pivot.parquet
- config_name: dml_results_pre_dfs
  description: "Baseline DML results without DataForSEO confounders — used for robustness comparison."
  data_files:
  - split: long
    path: data/dml_results_pre_dfs/dml_results_long.parquet
  - split: pivot_rank_delta
    path: data/dml_results_pre_dfs/dml_pivot_rank_delta.parquet
  - split: pivot_post_rank
    path: data/dml_results_pre_dfs/dml_pivot_post_rank.parquet
- config_name: serp
  description: "Phase-0 SERP snapshots — raw ranked URLs from SearXNG and DuckDuckGo before LLM re-ranking."
  data_files:
  - split: searxng_top20
    path: data/serp/phase0_top20_searxng.parquet
  - split: searxng_top50
    path: data/serp/phase0_top50_searxng.parquet
  - split: ddg_top20
    path: data/serp/phase0_top20_ddg.parquet
  - split: ddg_top50
    path: data/serp/phase0_top50_ddg.parquet
- config_name: dataforseo
  description: "DataForSEO enrichment: Google SERP, keyword_overview, bulk keyword difficulty, Google Ads search volume, search intent."
  data_files:
  - split: serp_google_organic
    path: data/dataforseo/serp_google_organic.parquet
  - split: keyword_overview
    path: data/dataforseo/keyword_overview.parquet
  - split: bulk_keyword_difficulty
    path: data/dataforseo/bulk_keyword_difficulty.parquet
  - split: google_ads_search_volume
    path: data/dataforseo/google_ads_search_volume.parquet
  - split: search_intent
    path: data/dataforseo/search_intent.parquet
- config_name: domains
  description: "Per-domain llms.txt classification (16,049 domains × has_llms_txt / class)."
  data_files:
  - split: train
    path: data/domains_llms_txt.parquet
---

# GEODML — Paper-Size Experiment

> A 65,203-row tabular dataset for causal identification of **Generative
> Engine Optimization** (GEO) — i.e., which page-level features cause a
> large language model to promote (or demote) a search result relative to
> its original search-engine rank.

This dataset was produced by running 1,011 B2B-SaaS keywords through two
search backends (SearXNG and DuckDuckGo) at two pool sizes (top 20, top 50),
then having two frontier LLMs (`Qwen/Qwen2.5-72B-Instruct`,
`meta-llama/Llama-3.3-70B-Instruct`) each re-rank the results. The eight
resulting run matrices (2 engines × 2 pool sizes × 2 LLMs) were merged
into a single long-format table and enriched with **DataForSEO** keyword
metadata (search volume, CPC, competition, keyword difficulty, search
intent) and **llms.txt** adoption signals. The outcome is the rank
change produced by the LLM; the treatments are 19 GEO-style content
features (statistical density, citations, structured data, freshness,
etc.). Estimation uses the **DoubleML PLR** estimator with LightGBM
nuisance models and 5-fold cross-fitting.

## Ranking convention (read this first)

| symbol | meaning |
|---|---|
| `pre_rank` | position given by the search backend (1 = top of SERP) |
| `post_rank` | position after the LLM re-ranks the pool (1 = LLM's top pick) |
| `rank_delta = pre_rank − post_rank` | positive = **LLM promoted the page** |

- **Lower rank number is better.** Rank 1 is the goal.
- On outcome `rank_delta`: **positive coefficient = treatment promotes the page (good)**.
- On outcome `post_rank`: **negative coefficient = treatment promotes the page (good)**.
- Both outcomes report the same causal effect with opposite sign — we keep
  both because reviewers disagree on which is more natural.

## Quick start

```python
from datasets import load_dataset

# Primary 65k-row table (all treatments, confounders, outcomes, DFS columns)
ds = load_dataset("<your-username>/geodml-papersize", "main")
df = ds["train"].to_pandas()

# 570 DML fits — subset × treatment × outcome with θ, SE, p-value
fits = load_dataset("<your-username>/geodml-papersize",
                    "dml_results", split="long").to_pandas()

# Raw phase-0 SERP results (before LLM re-ranking)
serp = load_dataset("<your-username>/geodml-papersize",
                    "serp", split="searxng_top50").to_pandas()
```

Or skip `datasets` entirely:

```python
import pandas as pd
df = pd.read_parquet("data/main/full_experiment_data.parquet")
```

## Directory layout

```
huggingface_bundle/
├── README.md                          this file
├── docs/                              written analysis + methodology
│   ├── proposition-2026-04-07.md          research question & hypotheses
│   ├── ROADMAP.md
│   ├── meta-analysis-report-2026-04-15.md pre-DFS narrative
│   ├── dataforseo-plan-2026-04-22.md      motivation for the DFS pull
│   ├── analysis-2026-04-23.md             ★ full post-DFS analysis
│   ├── DATAFORSEO_CATALOG.md              DFS endpoint catalog
│   └── treatment-confounder-dictionary.md column dictionary for treatments / confounders
│
├── project/                           what the experiment is
│   ├── CLAUDE.md                          project-wide context + ranking convention
│   ├── README.md                          experiment README (original)
│   └── keywords.txt                       the 1,011 B2B-SaaS keywords
│
├── manifests/                         how runs were produced
│   ├── experiment_manifest.json           engines × LLMs × pool sizes × runs
│   ├── consolidation_manifest.json        how the merged dataset was built
│   └── tracker.json                       per-run progress snapshot
│
├── data/
│   ├── main/
│   │   ├── full_experiment_data.{csv,parquet}     65,203 × 73 — ★ primary table
│   │   └── regression_dataset.{csv,parquet}       same rows, pre-DataForSEO (no dfs_* cols)
│   │
│   ├── dml_results/                   post-DFS DoubleML outputs (see Results section)
│   │   ├── dml_results_long.{csv,parquet}         570 fits (θ, SE, p)
│   │   ├── dml_pivot_rank_delta.{csv,parquet}     treatment × subset pivot, coef+stars
│   │   ├── dml_pivot_post_rank.{csv,parquet}
│   │   ├── dml_multi_treatment*.{csv,parquet}     joint-regression variants
│   │   ├── confounder_audit.{csv,parquet}         LGBM gain importance per confounder
│   │   ├── confounder_loo_r2.{csv,parquet}        per-confounder ΔR²
│   │   ├── confounder_ols_significance.{csv,parquet}
│   │   ├── variance_explained.{csv,parquet}       5-fold CV R² by confounder source
│   │   ├── nuisance_r2.{csv,parquet}              DML g₀ / m₀ / structural R²
│   │   ├── *.md                                   narrative summaries
│   │   └── *.log                                  raw run logs
│   │
│   ├── dml_results_pre_dfs/           baseline (same runs, 17 confounders, no DFS)
│   │
│   ├── serp/                          raw SERP snapshots
│   │   ├── phase0_top{20,50}_searxng.json + .parquet
│   │   └── phase0_top{20,50}_ddg.json + .parquet
│   │
│   ├── dataforseo/                    keyword metadata enrichment
│   │   ├── serp_google_organic.{csv,parquet}      18,969 rows
│   │   ├── keyword_overview.{csv,parquet}
│   │   ├── bulk_keyword_difficulty.{csv,parquet}
│   │   ├── google_ads_search_volume.{csv,parquet}
│   │   ├── search_intent.{csv,parquet}
│   │   ├── raw/                                   raw JSON responses, one per keyword
│   │   ├── README_pipeline.md                     pipeline overview
│   │   └── *.log, run_manifest*.json
│   │
│   ├── domains_llms_txt.{csv,parquet} 16,049 domains × llms.txt adoption + classification
│   │
│   ├── runs/                          per-run artifacts (8 runs)
│   │   └── <engine>_<llm>_serp<N>_top10/
│   │       ├── geodml_dataset.{csv,parquet}       merged per-run table
│   │       ├── phase2/
│   │       │   ├── experiment.json, progress.json
│   │       │   ├── features.{csv,parquet}         code + LLM feature extraction
│   │       │   ├── rankings.{csv,parquet}         LLM-produced ranking
│   │       │   ├── keywords.jsonl                 per-keyword raw SERP+LLM payload
│   │       │   └── html_cache.tar.gz              ★ gzipped raw HTML (see below)
│   │       └── phase3/
│   │           └── features_new.{csv,parquet}     expanded treatments
│   │
│   └── logs/                          top-level experiment logs
│
└── scripts/
    ├── convert_csv_to_parquet.py      regenerate .parquet from .csv
    ├── normalize_serp_json.py         regenerate serp/*.parquet from .json
    ├── extract_html_caches.sh         un-tar all html_cache.tar.gz in place
    ├── load_example.py                ready-to-run load_dataset() examples
    └── upload_to_hf.sh                huggingface-cli upload command
```

### About `html_cache.tar.gz`

Each of the 8 per-run directories contains a `phase2/html_cache.tar.gz` —
the compressed HTML Snapshot that the feature-extractor saw when it
computed the treatment variables. Each tarball is 340–465 MB (≈ 3.1 GB
total); unpacked, they total ≈ 28 GB (54k files). Third-party HTML is
included only for reproduction of the feature-extraction step. To
unpack everything:

```bash
bash scripts/extract_html_caches.sh
```

## The 8 runs

| run_id | engine | LLM | SERP pool size |
|---|---|---|---:|
| 1 | duckduckgo | Llama-3.3-70B-Instruct | 20 |
| 2 | duckduckgo | Llama-3.3-70B-Instruct | 50 |
| 3 | duckduckgo | Qwen2.5-72B-Instruct   | 20 |
| 4 | duckduckgo | Qwen2.5-72B-Instruct   | 50 |
| 5 | searxng    | Llama-3.3-70B-Instruct | 20 |
| 6 | searxng    | Llama-3.3-70B-Instruct | 50 |
| 7 | searxng    | Qwen2.5-72B-Instruct   | 20 |
| 8 | searxng    | Qwen2.5-72B-Instruct   | 50 |

## Main table schema (`data/main/full_experiment_data.parquet`)

65,203 rows × 73 columns.

| group | columns | description |
|---|---|---|
| **Keys** | `run_id`, `search_engine`, `llm_model`, `serp_pool_size`, `llm_pool_size`, `keyword`, `domain`, `url` | identifies one (keyword, url, run) observation |
| **Outcomes** | `pre_rank`, `post_rank`, `rank_delta` | rank before/after LLM, their difference |
| **Adoption signal** | `has_llms_txt` | 1 if domain publishes `/llms.txt` |
| **Treatments — code-extracted** | `T1_statistical_density_code`, `T2_question_heading_code`, `T3_structured_data_code`, `T4_citation_authority_code` | deterministic HTML parsers |
| **Treatments — LLM-extracted** | `T1_statistical_density_llm`, `T2_question_heading_llm`, `T3_structured_data_llm`, `T4_citation_authority_llm` | same constructs, LLM rater |
| **Treatments — expanded (phase 3)** | `treat_stats_present`, `treat_stats_density`, `treat_question_headings`, `treat_structural_modularity`, `treat_structured_data`, `treat_ext_citations_any`, `treat_auth_citations`, `treat_topical_comp`, `treat_freshness`, `treat_source_brand`, `treat_source_earned`, `treat_source_type` | finer-grained GEO features |
| **Confounders — original** | `conf_title_kw_sim`, `conf_snippet_kw_sim`, `conf_title_len`, `conf_snippet_len`, `conf_brand_recog`, `conf_title_has_kw`, `conf_word_count`, `conf_readability`, `conf_internal_links`, `conf_outbound_links`, `conf_images_alt`, `conf_bm25`, `conf_https`, `conf_domain_authority`, `conf_backlinks`, `conf_referring_domains`, `conf_serp_position` | original 17 confounders |
| **Confounders — DataForSEO** | `dfs_keyword_difficulty`, `dfs_search_volume`, `dfs_cpc`, `dfs_competition`, `dfs_competition_level`, `dfs_main_intent`, `dfs_foreign_intent`, `dfs_google_rank`, `dfs_google_rank_absolute`, `dfs_se_results_count`, `dfs_google_top_url`, `dfs_intent_commercial`, `dfs_intent_informational`, `dfs_intent_navigational`, `dfs_intent_transactional` | added April 2026 |
| **Legacy features** | `X1_domain_authority`, `X1_global_rank`, `X3_word_count`, `X6_readability`, `X7_internal_links`, `X7B_outbound_links`, `X8_keyword_difficulty`, `X9_images_with_alt`, `X10_https` | pre-refactor names; mostly superseded by `conf_*`/`dfs_*` |

See `docs/treatment-confounder-dictionary.md` for detailed definitions.

## Key findings (POOLED, outcome = `rank_delta`, n ≈ 65 k)

Columns in `dml_results_long.parquet`:
`subset`, `subset_type`, `treatment`, `treatment_col`, `treatment_label`,
`outcome`, `n`, **`coef`**, `se`, `t_stat`, **`p_val`**, `ci_lower`,
`ci_upper`, `stars`, `direction`, `note`.

| treatment | coef | p_val | direction |
|---|---:|---:|---|
| **T7_source_earned** | **−1.700** | 1.3e-144 *** | strongly demotes |
| T3_structured_data_new (expanded) | −0.140 | 3.4e-10 *** | demotes |
| T6_freshness | −0.060 | 7.1e-22 *** | demotes |
| T4b_auth_citations | −0.019 | 2.2e-03 *** | demotes |
| T1b_stats_density | −0.017 | 2.3e-14 *** | demotes |
| T_llms_txt | +0.094 | 1.8e-05 *** | promotes |
| T2a_question_headings | +0.103 | 3.9e-05 *** | promotes |
| T3_code | +0.127 | 1.3e-06 *** | promotes |
| **T5_topical_comp** | **+0.438** | 2.0e-05 *** | promotes |

Full 570-fit table in `data/dml_results/dml_results_long.parquet`.
Robustness vs. pre-DataForSEO baseline: **zero sign flips** across 570
paired fits (median |Δcoef| = 0.0057). See `docs/analysis-2026-04-23.md`
for the full post-DFS analysis, nuisance-model R² diagnostics,
leave-one-out ΔR², and OLS significance tables.

## Identification diagnostics (post-DFS)

From `data/dml_results/nuisance_r2.parquet`:

- R²(Y = rank_delta | X) = **78 %** — strong outcome residualisation
- R²(Y = post_rank | X) = **36 %**
- R²(D | X) across treatments = 20 – 58 % — all below the 95 % overlap-violation threshold
- R²(Ỹ | D̃) = 0.01 – 1.4 % — effects are small in variance terms but significance is driven by n ≈ 65 k

## How the 73 columns were produced

1. **Phase 0 — SERP pull.** Each of 1,011 keywords is queried against
   SearXNG (aggregating Brave, Startpage, Google, Bing) and DuckDuckGo
   (via `ddgs`). Top-20 and top-50 pools are stored.
   → `data/serp/phase0_top*.{json,parquet}`
2. **Phase 2 — HTML fetch + feature extraction.** For every URL the raw
   HTML is cached (`html_cache.tar.gz`) and parsed two ways:
   deterministically (regex / BeautifulSoup) and via a frontier LLM.
   → `data/runs/<run>/phase2/features.*`
3. **Phase 3 — expanded treatments.** The raw HTML is re-parsed to
   produce the finer-grained `treat_*` columns.
   → `data/runs/<run>/phase3/features_new.*`
4. **LLM re-ranking.** The LLM is given the keyword plus the top-N (20
   or 50) results and asked to return a top-10 ordered list. This
   defines `post_rank`.
   → `data/runs/<run>/phase2/rankings.*`
5. **Consolidation.** All 8 runs are merged into a long-format table
   keyed by `(run_id, keyword, url)`, with the LLM output joined on
   `(keyword, url)`. Result: `data/runs/<run>/geodml_dataset.parquet`,
   and after cross-run concat: `data/main/regression_dataset.parquet`.
6. **DataForSEO enrichment (April 2026).** Google SERP, search volume,
   CPC, competition, keyword difficulty, and search intent are joined
   on `keyword`. Result: `data/main/full_experiment_data.parquet`.
7. **DoubleML PLR with LightGBM.** 5-fold cross-fitting × 19 treatments
   × 2 outcomes × 15 subsets = 570 fits.
   → `data/dml_results/dml_results_long.parquet`

## Reading order for writing the paper

1. `project/CLAUDE.md` — context, ranking convention, architecture
2. `docs/proposition-2026-04-07.md` — research question
3. `docs/meta-analysis-report-2026-04-15.md` — pre-DFS narrative
4. `data/dml_results_pre_dfs/findings_report.md` — pre-DFS results
5. `docs/dataforseo-plan-2026-04-22.md` — why the enrichment was done
6. **`docs/analysis-2026-04-23.md`** — full post-DFS analysis ★
7. `data/dml_results/dml_summary.md` — top-line results
8. `data/dml_results/dml_results_long.parquet` — full 570-fit table

## Data provenance

Collection window: **March – April 2026**. All SERP and HTML snapshots
date from this period; re-running today will produce different raw
HTML. DataForSEO rows were pulled 22-23 April 2026.

Backlinks columns (`conf_domain_authority`, `conf_backlinks`,
`conf_referring_domains`, `X1_global_rank`) are **sparse (11–27 %)** —
the DataForSEO backlinks subscription was access-denied during this
pull. When unlocked, these columns will backfill to ≈ 100 %.

## License

**CC-BY-4.0** on all original content (the 65k-row table, DML results,
documentation, pipeline code).

**Third-party HTML** in `data/runs/*/phase2/html_cache.tar.gz` is
included under fair use for research reproducibility only. Do **not**
redistribute the raw HTML outside this research context. If you are a
rights-holder and would like content removed, open an issue.

## Citation

```bibtex
@misc{geodml_papersize_2026,
  author = {Valerian Fourel},
  title  = {GEODML — Paper-Size Experiment: Causal Identification of Generative-Engine-Optimization Features},
  year   = {2026},
  note   = {HuggingFace dataset},
}
```

## Contact

Open an issue on the dataset repo, or email
`valerian.fourel@gmail.com`.
