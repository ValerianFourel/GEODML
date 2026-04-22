# Paper-Size Experiment

Full pipeline for the scaled GEO causal inference study. Runs across multiple LLMs, pool sizes, and 1011 keywords.

## Quick Start

```bash
# Test with 3 keywords, 1 model, 1 pool size
python paperSizeExperiment/run_experiment.py --keywords 3 --models "meta-llama/Llama-3.3-70B-Instruct" --pool-sizes "20,10"

# Dry run — see the experiment plan
python paperSizeExperiment/run_experiment.py --dry-run

# Full experiment (all models x all pool sizes)
python paperSizeExperiment/run_experiment.py

# Full experiment with limited keywords (for initial testing)
python paperSizeExperiment/run_experiment.py --keywords 50
```

## Configuration

Edit `config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODELS` | Llama-3.3-70B, Qwen2.5-72B | Models for re-ranking |
| `POOL_SIZES` | (20,10), (50,10) | (serp_results, llm_top_n) pairs |
| `SEARCH_ENGINE` | searxng | Search backend |
| `ENABLE_LLM_FEATURES` | True | LLM-based T1-T4 extraction |
| `ENABLE_PAGERANK` | True | Domain authority via OpenPageRank |

## Pipeline Stages

```
keywords.txt
    |
    v
[1] gather_data.py  (per model x pool size)
    SERP search -> LLM re-ranking -> HTML fetch -> code-based features
    Output: experiment.json, rankings.csv, features.csv, html_cache/
    |
    v
[2] extract_features.py  (optional enhanced features)
    T1a-T7 treatments + expanded confounders from cached HTML
    Output: features_new.csv
    |
    v
[3] clean_data.py  (merge)
    Rankings + features + rank_delta -> single DML-ready CSV
    Output: geodml_dataset.csv
    |
    v
[4] analyze.py  (per-run DML)
    DoubleML PLR + sensitivity (LGBM, RF)
    Output: all_experiments.csv, plots
    |
    v
[5] analyze_cross_model.py  (cross-run comparison)
    Merged dataset -> per-model, per-pool, pooled analysis
    Output: robustness heatmap, cross-model coefficients
```

## Output Structure

```
paperSizeExperiment/output/
├── duckduckgo_Llama-3.3-70B-Instruct_serp20_top10/   # Run 1 — DONE
├── duckduckgo_Llama-3.3-70B-Instruct_serp50_top10/   # Run 2 — DONE
├── duckduckgo_Qwen2.5-72B-Instruct_serp20_top10/     # Run 3 — Phase 3 88.8%
├── duckduckgo_Qwen2.5-72B-Instruct_serp50_top10/     # Run 4 — Phase 3 33%
├── searxng_Llama-3.3-70B-Instruct_serp20_top10/      # Run 5 — DONE
│   ├── experiment.json
│   ├── keywords.jsonl
│   ├── rankings.csv
│   ├── features.csv
│   ├── features_new.csv
│   ├── geodml_dataset.csv
│   ├── html_cache/
│   ├── analysis_full/
│   └── analysis_halo/
├── merged_all_runs.csv
├── tracker.json
├── experiment_manifest.json
└── cross_model_analysis/
    ├── all_cross_model_results.csv
    ├── summary.json
    ├── cross_model_coefficients.png
    ├── robustness_heatmap.png
    └── pool_size_comparison.png
```

## Treatments Measured

| ID | Name | Type | Source |
|----|------|------|--------|
| T1 | Statistical Density | float | code + LLM |
| T2 | Question Headings | binary | code + LLM |
| T3 | Structured Data (JSON-LD) | binary | code + LLM |
| T4 | Citation Authority | int | code + LLM |
| T5 | Topical Competence | float | cosine similarity |
| T6 | Freshness | ordinal 0-4 | date extraction |
| T7 | Source: Earned Media | binary | domain classification |

## Confounders

Title/snippet similarity, brand recognition, word count, readability, link counts, BM25, SERP position, domain authority, backlinks, referring domains.

## Adding Keywords

Edit `keywords.txt` — one keyword per line. Lines starting with `#` are comments.
