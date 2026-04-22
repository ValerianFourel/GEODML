# Proposition — 2026-04-08 (updated)

## Context

The GEODML experiment measures the causal effect of page-level features on LLM re-ranking of search results. We use Double Machine Learning (DoubleML) to estimate how treatments (content features like statistical density, structured data, citation authority) affect rank changes, while controlling for confounders (domain authority, word count, readability, etc.).

The experiment follows a 2x2x2 factorial design: 2 search engines (DuckDuckGo, SearXNG) x 2 LLM models (Llama 3.3 70B, Qwen 2.5 72B) x 2 SERP pool sizes (20, 50) = 8 runs.

Each run goes through 4 phases:
1. **Phase 1** — SERP queries + LLM re-ranking (rankings.csv)
2. **Phase 2** — HTML fetch + code-based feature extraction (features.csv)
3. **Phase 3** — LLM-based feature scoring via HF API (T1-T4 LLM columns in features.csv)
4. **Phase 4** — Enriched features: Moz API (domain authority, backlinks), T5 topical competence, T6 freshness, T7 source classification (features_new.csv)

## Current State (2026-04-09)

### Completed runs with full analysis
- **Run 1** (DDG/Llama/s20): 7,890 rows, 204 DML experiments, **32 significant** (p<0.05)
- **Run 2** (DDG/Llama/s50): 8,088 rows, 204 DML experiments, **39 significant** (p<0.05)
- **Run 5** (SXG/Llama/s20): 8,197 rows, 216 DML experiments, **44 significant** (p<0.05)
- **Cross-model analysis**: 492 experiments, **165 significant** (p<0.05)

### In progress (stopped 2026-04-09, resume with commands below)
- **Run 3** (DDG/Qwen/s20): Phase 3 at **91.8%** (5422/5906, **484 remaining**, ~1.5h). Has geodml_dataset.csv with outcomes (77%) but missing features_new.csv.
- **Run 4** (DDG/Qwen/s50): Phase 3 at **35.6%** (2612/7347, **4735 remaining**, ~8-16h). No geodml_dataset.csv yet.

### Not started
- **Runs 6-8** (all SearXNG): require running SearXNG Docker container.

### Key findings from completed runs
- **T7 earned media pages are consistently demoted** by LLMs (rank_delta coef = -1.3 to -2.4, p ≈ 0)
- **Halo effect**: brands mentioned in earned media get better LLM rankings (post_rank improvement)
- **T4 citation authority** (LLM-measured) significant for SearXNG run (coef = -0.09, p = 0.004)
- **T1 statistical density** (LLM-measured) marginally significant for DDG/Llama/s20 (coef = -0.01, p = 0.015)
- Larger SERP pools (serp50) amplify T3 structured data and T6 freshness effects

### What was fixed
- **`clean_data.py` resume bug**: when `gather_data.py` resumes from checkpoints, `experiment.json` has empty `per_keyword_results`. Fixed to fall back to `keywords.jsonl`, restoring outcomes for Runs 3 and 5.

### What failed or is incomplete

**Moz API coverage is only 6-10%**
- Moz was queried for all domains but only ~6-10% returned data
- domain authority, backlinks, and referring domains are effectively unusable as confounders
- OpenPageRank ran for some runs but coverage improvement was marginal

**T5 topical competence at 0% for DDG runs**
- Embedding-based cosine similarity between page content and keyword query
- Works for SearXNG run (67.8% coverage) but 0% for DDG runs
- Depends on SearXNG-specific data fields not available from DuckDuckGo

**T6 freshness populated at ~64-70%**
- Previously reported as 0% — this was incorrect. Date extraction works and produces ~64-70% coverage for completed runs.

**Title/snippet keyword similarity at 0% for DDG runs**
- conf_title_kw_sim and conf_snippet_kw_sim rely on SERP snippet/title data that DuckDuckGo doesn't provide in the same format as SearXNG

**Several planned confounders never populated (0%)**
- X1_domain_authority (OpenPageRank) — ran for Run 5 but not reflected in geodml_dataset
- X2_domain_age (WHOIS) — never executed
- X4_lcp_ms (Core Web Vitals) — no API integration
- X8_keyword_difficulty — no API available

## What To Do Next

### Immediate priority — Resume commands

1. **Finish Phase 3 on Run 3** (484 URLs remaining, ~1.5h):
```bash
source venv312/bin/activate
python pipeline/gather_data.py --engine duckduckgo --serp-results 20 --llm-top-n 10 \
  --llm-model "Qwen/Qwen2.5-72B-Instruct" \
  --keywords-file paperSizeExperiment/keywords.txt \
  --output-dir paperSizeExperiment/output/duckduckgo_Qwen2.5-72B-Instruct_serp20_top10 \
  --llm-features --pagerank \
  --progress-file paperSizeExperiment/output/duckduckgo_Qwen2.5-72B-Instruct_serp20_top10/progress.json
```

2. **Finish Phase 3 on Run 4** (4735 URLs remaining, ~8-16h):
```bash
python pipeline/gather_data.py --engine duckduckgo --serp-results 50 --llm-top-n 10 \
  --llm-model "Qwen/Qwen2.5-72B-Instruct" \
  --keywords-file paperSizeExperiment/keywords.txt \
  --output-dir paperSizeExperiment/output/duckduckgo_Qwen2.5-72B-Instruct_serp50_top10 \
  --llm-features --pagerank \
  --progress-file paperSizeExperiment/output/duckduckgo_Qwen2.5-72B-Instruct_serp50_top10/progress.json
```

3. **After Phase 3 completes**, run the rest of the pipeline for each run:
```bash
python paperSizeExperiment/run_experiment.py --engine duckduckgo \
  --models "Qwen/Qwen2.5-72B-Instruct" --pool-sizes "20,10" \
  --skip-gather --skip-features
# Repeat with --pool-sizes "50,10" for Run 4
```

### Medium-term

4. **Start SearXNG container** and run Runs 6, 7, 8
5. **Full 8-run cross-model analysis** — test whether LLM choice, engine, and pool size affect treatment effects

### Optional improvements

6. **Improve domain authority coverage** — try OpenPageRank batch API (already coded), or Majestic ($50/mo) if coverage < 50%
7. **Fix T5 topical competence for DDG runs** — adapt embedding similarity to work without SearXNG data

### Dropped from scope

- **DeepSeek R1 32B** — not started
- **X2 domain age (WHOIS)** — unreliable and rate-limited
- **X4 Core Web Vitals** — would need CrUX API integration

## Priority Order

1. Finish Qwen Phase 3 (Runs 3-4) — enables model comparison
2. Run pipeline on Runs 3-4 (extract, clean, analyze)
3. Launch Runs 6-8 (SearXNG container needed)
4. Full 8-run cross-model analysis
5. Consider domain authority API alternatives if needed
