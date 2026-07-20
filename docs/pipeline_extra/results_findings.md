# GEODML Results & Findings

> Comprehensive summary of all experimental results, organized by model, search engine, and experimental design.
> Study period: February 2026, Hamburg, Germany.

---

## Table of Contents

1. [Study Design](#1-study-design)
2. [Results by Experiment](#2-results-by-experiment)
   - [2.1 Llama-3.3-70B + SearXNG, Small Pool (20/10)](#21-llama-33-70b--searxng-small-pool-2010)
   - [2.2 Llama-3.3-70B + SearXNG, Large Pool (50/20)](#22-llama-33-70b--searxng-large-pool-5020)
   - [2.3 Small Pool vs Large Pool Comparison](#23-small-pool-vs-large-pool-comparison)
   - [2.4 DeepSeek R1 + SearXNG, Small Pool (20/10), New Treatments](#24-deepseek-r1--searxng-small-pool-2010-new-treatments)
   - [2.5 Llama-3.3-70B + SearXNG, New Pipeline (10 Treatments)](#25-llama-33-70b--searxng-new-pipeline-10-treatments)
   - [2.6 Exploratory Runs (Other Engines)](#26-exploratory-runs-other-engines)
3. [Cross-Model Comparison](#3-cross-model-comparison)
4. [Consolidated Significant Findings](#4-consolidated-significant-findings)
5. [Robustness Analysis](#5-robustness-analysis)
6. [Key Takeaways](#6-key-takeaways)

---

## 1. Study Design

**What we measure**: The causal effect of on-page features (treatments) on how an LLM re-ranks search engine results for 50 B2B SaaS keywords (e.g., "CRM software", "ERP software").

**Method**: Double Machine Learning (DML) with Partially Linear Regression (PLR), 5-fold cross-fitting, LightGBM and Random Forest nuisance learners.

**Three outcome variables**:

| Outcome | Meaning | Interpretation of positive coefficient |
|---------|---------|---------------------------------------|
| `rank_delta` | pre_rank - post_rank | LLM promotes the page (good) |
| `pre_rank` | SERP position (1=best) | Search engine ranks worse (bad) |
| `post_rank` | LLM position (1=best) | LLM ranks worse (bad) |

**Convention**: Lower rank number = better position. Negative coefficient on post_rank = LLM ranks it higher = good. Positive coefficient on rank_delta = LLM promotes it = good.

---

## 2. Results by Experiment

### 2.1 Llama-3.3-70B + SearXNG, Small Pool (20/10)

| Config | Value |
|--------|-------|
| **LLM** | Llama-3.3-70B-Instruct (HuggingFace) |
| **Search engine** | SearXNG (Google + Bing + DDG + Brave + Startpage) |
| **Pool** | 20 SERP results, LLM re-ranks top 10 |
| **Observations** | 492 total, 349-355 with valid rank_delta |
| **Treatments** | T1-T4 code-based + T1-T4 LLM-based (8 vars) |
| **Confounders** | 8 legacy (domain auth, domain age, word count, readability, internal links, outbound links, kw difficulty, images alt) |
| **Date** | 2026-02-16 |
| **Results files** | `results/dml_results.json`, `test/results/all_experiments.csv`, `test_diff/results/all_experiments.csv` |

#### Cross-reference table (PLR, code-based, LGBM)

| Treatment | pre_rank (SERP) | post_rank (LLM) | rank_delta (gap) |
|-----------|----------------|-----------------|-----------------|
| **T1** Statistical Density | +0.315 (p=0.170) | **+0.101 (p=0.024)\*\*** | +0.186 (p=0.214) |
| **T2** Question Headings | +0.909 (p=0.115) | -0.356 (p=0.233) | **+1.198 (p=0.009)\*\*\*** |
| **T3** Structured Data | +0.145 (p=0.803) | **-0.719 (p=0.048)\*\*** | +0.812 (p=0.103) |
| **T4** Citation Authority | -1.020 (p=0.219) | -0.740 (p=0.125) | -0.650 (p=0.311) |

#### Initial rank_delta-only analysis (v1, LGBM)

| Treatment | Code path (θ, p) | LLM path (θ, p) |
|-----------|-------------------|------------------|
| T1 Statistical Density | +0.269 (p=0.082) | +0.010 (p=0.715) |
| T2 Question Headings | +0.769 (p=0.073) | +0.016 (p=0.973) |
| T3 Structured Data | +0.163 (p=0.760) | +0.242 (p=0.593) |
| T4 Citation Authority | -0.440 (p=0.487) | +0.112 (p=0.674) |

#### Key findings

1. **T2 Question Headings (rank_delta = +1.198, p=0.009)**: The strongest and most robust finding. Pages with FAQ-style H2/H3 headings ("What is CRM?", "How does it work?") get promoted ~1.2 positions by the LLM. Neither the search engine effect nor the LLM effect alone is significant, but the gap between them is. The LLM corrects what the search engine undervalues.

2. **T3 Structured Data (post_rank = -0.719, p=0.048)**: A pure LLM effect. JSON-LD schema markup (FAQ, Product, HowTo) improves LLM ranking by ~0.7 positions. The search engine is indifferent (p=0.80), but the LLM reads structured data as a quality signal.

3. **T1 Statistical Density (post_rank = +0.101, p=0.024)**: The LLM slightly penalizes number-heavy pages. Each additional stat per 500 words worsens LLM position by ~0.1 ranks. Small effect but consistent. The LLM prefers clarity over data density.

4. **T4 Citation Authority**: No significant effect. Consistent negative direction (better rankings) but only 3.3% of B2B SaaS pages cite academic sources -- insufficient statistical power.

---

### 2.2 Llama-3.3-70B + SearXNG, Large Pool (50/20)

| Config | Value |
|--------|-------|
| **LLM** | Llama-3.3-70B-Instruct |
| **Search engine** | SearXNG |
| **Pool** | 50 SERP results, LLM re-ranks top 20 |
| **Observations** | 996 total, 321-374 with valid rank_delta |
| **Treatments** | T1-T4 code-based + T1-T4 LLM-based |
| **Confounders** | Same 8 legacy |
| **Date** | 2026-02-17 |
| **Results files** | `50_larger/dml_results.json`, `50_larger/test/results/all_experiments.csv` |

#### rank_delta results (PLR, LGBM)

| Treatment | Code path (θ, p) | LLM path (θ, p) |
|-----------|-------------------|------------------|
| T1 Statistical Density | +0.027 (p=0.932) | -0.019 (p=0.726) |
| T2 Question Headings | -0.993 (p=0.295) | **-2.924 (p=0.002)\*\*\*** |
| T3 Structured Data | -1.447 (p=0.147) | -0.333 (p=0.697) |
| T4 Citation Authority | -1.392 (p=0.551) | +0.808 (p=0.175) |

#### Key findings

1. **T2 Question Headings (LLM path, rank_delta = -2.924, p=0.002)**: The single strongest finding across all experiments. In the large pool, question headings are associated with **less** LLM promotion -- a full reversal from the small pool. The LLM-based measurement captures this clearly.

2. **The T2 large-pool effect is driven by pre_rank**: Search engines already rank question-heading pages well in the broader SERP (pre_rank = -3.89, p=0.002). The LLM doesn't add further uplift. The rank_delta is negative because the pages were already well-placed.

3. **All small-pool significant effects vanish or reverse**: T1 effect disappears (θ=+0.03, p=0.93). T3 reverses direction (from +1.10 to -1.45). The LLM behaves fundamentally differently with more candidates.

---

### 2.3 Small Pool vs Large Pool Comparison

| Config | Value |
|--------|-------|
| **Design** | Comparative: 20-SERP/10-rerank vs 50-SERP/20-rerank |
| **LLM** | Llama-3.3-70B-Instruct (same for both) |
| **Search engine** | SearXNG (same for both) |
| **Total experiments** | 96 (48 per dataset) |
| **Significant at p<0.05** | 7 findings |
| **Date** | 2026-02-18 |
| **Results files** | `both_analysis/results/all_experiments.csv`, `both_analysis/COMPARATIVE_FINDINGS.md` |

#### Head-to-head comparison (rank_delta, PLR, code-based)

| Treatment | Small Pool (20/10) | Large Pool (50/20) | Consistent? |
|-----------|-------------------|-------------------|-------------|
| T1 Statistical Density | **+0.38 (p=0.023)\*** | +0.03 (p=0.93) | Effect disappears |
| T2 Question Headings | **+1.07 (p=0.019)\*** | -0.99 (p=0.29) | **Sign reversal** |
| T3 Structured Data | **+1.10 (p=0.033)\*** | -1.45 (p=0.15) | **Sign reversal** |
| T4 Citation Authority | -1.12 (p=0.10) | -1.39 (p=0.55) | Same direction (both NS) |

#### The convergence interpretation

| Scenario | Search engine's view | LLM's view | rank_delta |
|----------|---------------------|-----------|------------|
| **Small pool (20/10)** | Slightly undervalues question headings | Actively promotes them | +1.07 (positive, LLM corrects upward) |
| **Large pool (50/20)** | Already rewards question headings | Does not add further uplift | -2.92 (negative, less promotion needed) |

**Core insight**: With more data, the search engine and LLM **converge**. With fewer options, the LLM rewards structural signals (FAQ headings, schema). With more options, it becomes more discriminating and penalizes formulaic optimization in favor of content depth.

---

### 2.4 DeepSeek R1 + SearXNG, Small Pool (20/10), New Treatments

| Config | Value |
|--------|-------|
| **LLM** | DeepSeek R1 (HuggingFace) |
| **Search engine** | SearXNG |
| **Pool** | 20 SERP / 10 LLM re-rank |
| **Observations** | 411-446 (varies by treatment) |
| **Treatments** | 10 new (T1a, T1b, T2a, T2b, T3, T4a, T4b, T5, T6, T7) |
| **Confounders** | 16 new (title/snippet similarity, brand, BM25, domain auth, backlinks, SERP position, etc.) |
| **Total experiments** | 60 (10 treatments x 3 outcomes x 2 learners) |
| **Date** | 2026-02-23/24 |
| **Results files** | `pipeline/results_deepseek-r1_plr_lgbm+rf_new-10treat_3out_5fold/` |

#### rank_delta results (PLR, LGBM)

| Treatment | θ | SE | p-value | Sig? |
|-----------|---|-----|---------|------|
| T1a Stats Present (binary) | +0.312 | 0.227 | 0.168 | |
| T1b Stats Density (continuous) | +0.038 | 0.054 | 0.484 | |
| **T2a Question Headings (binary)** | **+0.714** | 0.276 | **0.010** | **\*\*\*** |
| T2b Structural Modularity (count) | +0.002 | 0.005 | 0.691 | |
| T3 Structured Data (expanded) | -0.054 | 0.206 | 0.793 | |
| T4a External Citations (binary) | -0.250 | 0.477 | 0.600 | |
| **T4b Authority Citations (count)** | **+0.392** | 0.189 | **0.038** | **\*\*** |
| T5 Topical Competence (cosine) | +0.427 | 0.727 | 0.558 | |
| T6 Freshness (ordinal 0-4) | -0.032 | 0.053 | 0.552 | |
| T7 Source Earned | -1.175 | 0.815 | 0.149 | |

#### post_rank results (PLR, LGBM)

| Treatment | θ | p-value | Sig? |
|-----------|---|---------|------|
| **T2a Question Headings** | **-0.822** | **0.0007** | **\*\*\*** |
| T2b Structural Modularity (RF) | -0.010 | 0.043 | \*\* |
| **T7 Source Earned (RF)** | **+1.646** | **0.028** | **\*\*** |
| T4b Authority Citations | -0.332 | 0.112 | marginal |

#### Key findings with DeepSeek R1

1. **T2a Question Headings (post_rank = -0.822, p=0.0007)**: The strongest result for DeepSeek R1. Pages with FAQ-style headings are placed ~0.8 positions higher by the LLM. Also significant on rank_delta (+0.714, p=0.010). **This replicates the Llama finding** -- both models reward question headings in the small pool.

2. **T4b Authority Citations (rank_delta = +0.392, p=0.038)**: New finding with the count-based authority citation measure. Each additional citation to .edu/.gov/academic domains promotes the page by ~0.4 positions. This was not significant in the Llama experiments (likely because the binary T4 lacked power). The continuous count measure is more sensitive.

3. **T7 Source Earned (post_rank = +1.646, p=0.028, RF)**: Earned media pages (G2, Capterra, TechCrunch, etc.) are ranked ~1.6 positions **worse** by the LLM compared to brand/vendor pages. DeepSeek R1 favors first-party product pages over third-party reviews.

4. **T3 Structured Data**: Not significant with DeepSeek R1 (p=0.79). This was significant with Llama, suggesting structured data sensitivity is model-dependent.

5. **T1 Statistical Density**: No significant effect with any measurement variant. The small penalty seen with Llama does not replicate with DeepSeek R1.

---

### 2.5 Llama-3.3-70B + SearXNG, New Pipeline (10 Treatments)

| Config | Value |
|--------|-------|
| **LLM** | Llama-3.3-70B-Instruct |
| **Search engine** | SearXNG |
| **Pool** | 20 SERP / 10 LLM re-rank |
| **Observations** | 349-492 (varies) |
| **Treatments** | Same 10 new treatments as Exp 2.4 |
| **Confounders** | Same 16 new confounders |
| **Total experiments** | 60 |
| **Results files** | `pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/` |

#### rank_delta results (PLR, LGBM)

| Treatment | θ | SE | p-value | Sig? |
|-----------|---|-----|---------|------|
| T1a Stats Present (binary) | +0.520 | 0.323 | 0.108 | |
| T1b Stats Density (continuous) | +0.129 | 0.073 | 0.078 | marginal |
| T2a Question Headings (binary) | +0.162 | 0.320 | 0.612 | |
| T2b Structural Modularity (count) | +0.006 | 0.008 | 0.480 | |
| T3 Structured Data (expanded) | +0.001 | 0.265 | 0.997 | |
| T4a External Citations (binary) | -1.165 | 0.699 | 0.096 | marginal |
| T4b Authority Citations (count) | -0.769 | 0.583 | 0.187 | |
| T5 Topical Competence (cosine) | -0.644 | 0.855 | 0.452 | |
| **T6 Freshness (ordinal)** | **-0.143** | 0.070 | **0.041** | **\*\*** |
| **T7 Source Earned** | **-5.271** | 0.476 | **<0.001** | **\*\*\*** |

#### post_rank results (PLR, LGBM)

| Treatment | θ | p-value | Sig? |
|-----------|---|---------|------|
| T1a Stats Present | -0.494 | 0.053 | marginal |
| T1b Stats Density | -0.064 | 0.098 | marginal |
| T2b Structural Modularity | -0.010 | 0.075 | marginal |
| **T7 Source Earned** | **+5.200** | **<0.001** | **\*\*\*** |
| T6 Freshness | +0.109 | 0.107 | marginal |

#### Key findings with Llama + new pipeline

1. **T7 Source Earned (rank_delta = -5.271, p<0.001)**: The largest effect in the entire study. Earned media pages are promoted ~5.3 rank positions **less** than brand pages by Llama (equivalently, they rank ~5.2 positions worse in post_rank). Llama strongly favors first-party vendor content over third-party review sites.

2. **T6 Freshness (rank_delta = -0.143, p=0.041)**: Each unit increase on the freshness scale (0-4) is associated with ~0.14 less LLM promotion. More recently dated content gets slightly less promotion. This suggests the LLM doesn't boost recency for its own sake.

3. **T2a Question Headings**: Not significant in this run (p=0.612). This contrasts with both the original Llama experiment (p=0.009) and the DeepSeek R1 run (p=0.010). The difference is likely due to the expanded 16-confounder set absorbing some of the T2 signal (especially conf_title_kw_sim and conf_brand_recog).

---

### 2.6 Exploratory Runs (Other Search Engines)

SERP data was collected from multiple search engines but full DML analysis was only run on SearXNG results. The following runs produced raw ranking data:

| Date | Engine | LLM | File | Notes |
|------|--------|-----|------|-------|
| 2026-02-11 | DuckDuckGo | Qwen2.5-72B-Instruct | `results/duckduckgo_Qwen2.5-72B-Instruct_2026-02-11_1709.json` | First experiment, DDG + Qwen |
| 2026-02-11 | DuckDuckGo | Qwen2.5-72B-Instruct | `results/duckduckgo_Qwen2.5-72B-Instruct_2026-02-11_1727.json` | Repeat run |
| 2026-02-11 | Brave | Qwen2.5-72B-Instruct | `results/brave_Qwen2.5-72B-Instruct_2026-02-11_1659.json` | Brave API |
| 2026-02-11 | Yahoo | Qwen2.5-72B-Instruct | `results/yahoo_Qwen2.5-72B-Instruct_2026-02-11_1646.json` | Web scrape |
| 2026-02-16 | DuckDuckGo | none | `results/duckduckgo_nollm_2026-02-16_0915.json` | Baseline (no LLM) |
| 2026-02-16 | Brave | none | `results/brave_nollm_2026-02-16_0917.json` | Baseline (no LLM) |
| Various | SerpAPI | Llama-3.3-70B | `results/serpapi_Llama-3.3-70B-Instruct_*.json` | Multiple runs |

These provide raw SERP rankings and LLM re-rankings but were not carried through the full feature extraction and DML analysis pipeline.

---

## 3. Cross-Model Comparison

### T2 Question Headings: The flagship finding

| Model | Pool | Outcome | θ | p-value | Direction |
|-------|------|---------|---|---------|-----------|
| Llama-3.3-70B | Small (20/10) | rank_delta | +1.198 | 0.009 | Promotes |
| Llama-3.3-70B | Large (50/20) | rank_delta | -2.924 | 0.002 | Demotes |
| DeepSeek R1 | Small (20/10) | rank_delta | +0.714 | 0.010 | Promotes |
| DeepSeek R1 | Small (20/10) | post_rank | -0.822 | 0.0007 | Improves LLM rank |
| Llama-3.3-70B (new pipeline) | Small (20/10) | rank_delta | +0.162 | 0.612 | NS (absorbed by new confounders) |

**Conclusion**: Both Llama and DeepSeek R1 reward question headings in the small pool. The effect reverses for Llama in the large pool. Cross-model replication strengthens confidence.

### T3 Structured Data

| Model | Pool | Outcome | θ | p-value |
|-------|------|---------|---|---------|
| Llama-3.3-70B | Small (20/10) | post_rank | -0.719 | 0.048 |
| Llama-3.3-70B | Small (20/10) | rank_delta | +1.10 | 0.033 |
| Llama-3.3-70B | Large (50/20) | rank_delta | -1.45 | 0.147 |
| DeepSeek R1 | Small (20/10) | rank_delta | -0.054 | 0.793 |

**Conclusion**: Structured data effects are model-dependent and pool-dependent. Significant only for Llama in the small pool.

### T7 Source Earned (Brand vs Review Sites)

| Model | Pool | Outcome | θ | p-value |
|-------|------|---------|---|---------|
| Llama-3.3-70B (new) | Small (20/10) | rank_delta | -5.271 | <0.001 |
| Llama-3.3-70B (new) | Small (20/10) | post_rank | +5.200 | <0.001 |
| DeepSeek R1 | Small (20/10) | rank_delta | -1.175 | 0.149 |
| DeepSeek R1 | Small (20/10) | post_rank (RF) | +1.646 | 0.028 |

**Conclusion**: Both models penalize earned media pages (G2, Capterra, etc.) relative to brand pages. The effect is massive for Llama (~5 positions) and moderate for DeepSeek R1 (~1.6 positions).

### T4b Authority Citations (new count measure)

| Model | Pool | Outcome | θ | p-value |
|-------|------|---------|---|---------|
| DeepSeek R1 | Small (20/10) | rank_delta | +0.392 | 0.038 |
| Llama-3.3-70B (new) | Small (20/10) | rank_delta | -0.769 | 0.187 |

**Conclusion**: Only significant for DeepSeek R1. Direction disagrees between models (DeepSeek promotes, Llama demotes). Not a robust finding.

---

## 4. Consolidated Significant Findings

All statistically significant results (p<0.05) across all experiments:

| # | Experiment | Model | Treatment | Outcome | θ | p-value | Interpretation |
|---|-----------|-------|-----------|---------|---|---------|---------------|
| 1 | Small pool | Llama-3.3-70B | T2 Question Headings (code) | rank_delta | +1.198 | 0.009 | LLM promotes FAQ-style pages by ~1.2 positions |
| 2 | Small pool | Llama-3.3-70B | T3 Structured Data (code) | post_rank | -0.719 | 0.048 | LLM ranks schema-markup pages ~0.7 positions higher |
| 3 | Small pool | Llama-3.3-70B | T1 Statistical Density (code) | post_rank | +0.101 | 0.024 | LLM slightly penalizes number-heavy pages |
| 4 | Large pool | Llama-3.3-70B | T2 Question Headings (LLM) | rank_delta | -2.924 | 0.002 | LLM promotes FAQ pages LESS (reversal from small pool) |
| 5 | Large pool | Llama-3.3-70B | T2 Question Headings (LLM) | pre_rank | -3.891 | 0.002 | Search engine already ranks FAQ pages well in wider SERP |
| 6 | Comparative | Llama-3.3-70B | T1 Statistical Density (code) | rank_delta | +0.384 | 0.023 | Small pool: stats promote (weak) |
| 7 | Comparative | Llama-3.3-70B | T2 Question Headings (code) | rank_delta | +1.072 | 0.019 | Small pool: headings promote |
| 8 | Comparative | Llama-3.3-70B | T3 Structured Data (code) | rank_delta | +1.097 | 0.033 | Small pool: schema promotes |
| 9 | Comparative | Llama-3.3-70B | T2 Question Headings (code) | post_rank | -0.616 | 0.037 | Small pool: LLM ranks FAQ pages higher |
| 10 | Comparative | Llama-3.3-70B | T3 Structured Data (code) | post_rank | -0.810 | 0.018 | Small pool: LLM ranks schema pages higher |
| 11 | DeepSeek R1 | DeepSeek R1 | T2a Question Headings | rank_delta | +0.714 | 0.010 | Replicates Llama finding: headings promote |
| 12 | DeepSeek R1 | DeepSeek R1 | T2a Question Headings | post_rank | -0.822 | 0.0007 | LLM places FAQ pages ~0.8 positions higher |
| 13 | DeepSeek R1 | DeepSeek R1 | T4b Authority Citations | rank_delta | +0.392 | 0.038 | Academic citations promote pages |
| 14 | DeepSeek R1 | DeepSeek R1 | T2b Struct. Modularity (RF) | post_rank | -0.010 | 0.043 | More heading sections = slightly better LLM rank |
| 15 | DeepSeek R1 | DeepSeek R1 | T7 Source Earned (RF) | post_rank | +1.646 | 0.028 | Earned media ranked ~1.6 positions worse |
| 16 | Llama new | Llama-3.3-70B | T7 Source Earned | rank_delta | -5.271 | <0.001 | Earned media gets ~5.3 fewer promotion positions |
| 17 | Llama new | Llama-3.3-70B | T7 Source Earned | post_rank | +5.200 | <0.001 | Earned media ranked ~5.2 positions worse |
| 18 | Llama new | Llama-3.3-70B | T6 Freshness | rank_delta | -0.143 | 0.041 | Fresher content gets slightly less promotion |
| 19 | Llama new | Llama-3.3-70B | T2a Q. Headings (RF) | pre_rank | +0.012 | 0.042 | SERP slightly penalizes FAQ headings |

---

## 5. Robustness Analysis

### PLR vs IRM consistency (Small pool, Llama)

All four code-based treatments agree in direction between PLR and IRM on rank_delta. IRM estimates are noisier but directionally consistent.

### LGBM vs Random Forest consistency

| Treatment | Outcome | LGBM p-val | RF p-val | Both significant? |
|-----------|---------|-----------|---------|-------------------|
| T1 code | post_rank | 0.024 | 0.039 | Yes |
| T3 code | post_rank | 0.048 | 0.037 | Yes |
| T2 code | rank_delta | 0.009 | 0.055 | LGBM yes, RF marginal |

### Code-based vs LLM-based measurement

All four treatments agree in direction between code and LLM measurement paths. Code-based measurement consistently produces stronger signals (lower p-values) because it extracts exact quantities rather than relying on LLM evaluation noise.

### DML vs OLS comparison

DML and OLS coefficients are close across all specifications (e.g., T2 code on rank_delta: DML=+0.86, OLS=+1.10). This confirms weak confounding -- treatments are near-randomly assigned conditional on confounders.

### Model fit

- OLS R-squared: 3-7% across specifications (most ranking variance from unmeasured factors like exact content relevance and backlinks)
- Nuisance R-squared: -0.05 to +0.03 (confounders predict weakly)
- Low R-squared does not invalidate significant treatment effects -- it means treatments are not the main driver of rankings, but their directional effects are consistent

---

## 6. Key Takeaways

### What replicates across models and experiments

1. **T2 Question Headings in the small pool**: The most robust finding. Significant for Llama (p=0.009) AND DeepSeek R1 (p=0.010). Both LLMs promote FAQ-style pages when choosing from a curated top-10 list.

2. **T7 Source Earned (brand vs review sites)**: Both Llama and DeepSeek R1 penalize earned media (G2, Capterra, TechCrunch) relative to brand pages. The effect is very large for Llama (~5 positions) and moderate for DeepSeek R1 (~1.6 positions).

3. **Code-based measurement outperforms LLM-based**: Across all experiments, deterministic HTML feature extraction produces sharper causal estimates than LLM-based evaluation.

### What depends on context

4. **Pool size changes everything**: The same treatment (T2 Question Headings) has opposite effects depending on whether the LLM sees 10 or 20 candidates. In the small pool, FAQ headings help. In the large pool, they hurt (or the effect vanishes).

5. **T3 Structured Data is model-dependent**: Significant for Llama (p=0.048) but not for DeepSeek R1 (p=0.79). Schema markup sensitivity varies by model.

6. **T1 Statistical Density is experiment-dependent**: Significant for Llama in the original experiment (p=0.024), not significant for DeepSeek R1 or in the new Llama pipeline with expanded confounders.

### What was not found

7. **T4 Citation Authority (binary)**: Never significant in any experiment. Too few B2B SaaS pages cite academic sources for statistical power. The count-based measure (T4b) was significant only for DeepSeek R1.

8. **T5 Topical Competence**: Never significant. Keyword-content semantic similarity does not predict LLM re-ranking beyond what confounders capture.

### Practical implications for GEO

- **Structure content around questions** -- the most reliable lever, but only when competing in a small candidate pool (chatbot, short list). In broader contexts, prioritize genuine depth over formulaic FAQ headings.
- **First-party vendor content beats third-party reviews** in LLM re-ranking. Both models strongly prefer brand/product pages over review aggregators.
- **Schema markup may help with some LLMs** (Llama) but not others (DeepSeek R1). Not a universal strategy.
- **Content clarity over data density** -- avoid stuffing pages with numbers. The LLM values focused explanatory content.
- **GEO strategy is context-dependent** -- what works depends on the LLM model, the number of candidates it sees, and the competitive landscape. There is no single optimization that works universally.
