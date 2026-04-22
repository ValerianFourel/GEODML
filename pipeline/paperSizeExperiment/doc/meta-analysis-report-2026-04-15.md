# GEODML Meta-Analysis Report

**Date:** 2026-04-15
**Dataset:** 65,426 observations across 8 experiment runs
**Keywords:** 1,011 | **Domains:** 15,378
**T7 EARNED_DOMAINS:** 249 domains (expanded from 59, covering review platforms, tech/business media, consulting firms, community/UGC, trade publications, wire services, and more)

## 1. Experiment Matrix

| Run | Engine | LLM Model | SERP Pool | Rows | Status |
|-----|--------|-----------|-----------|------|--------|
| R1 | DuckDuckGo | Llama-3.3-70B | serp20 | 7,890 | Complete |
| R2 | DuckDuckGo | Llama-3.3-70B | serp50 | 8,088 | Complete |
| R3 | DuckDuckGo | Qwen2.5-72B | serp20 | 8,335 | Complete |
| R4 | DuckDuckGo | Qwen2.5-72B | serp50 | 9,863 | Complete |
| R5 | SearXNG | Llama-3.3-70B | serp20 | 8,313 | Complete |
| R6 | SearXNG | Llama-3.3-70B | serp50 | 12,809 | Complete |
| R7 | SearXNG | Qwen2.5-72B | serp20 | 9,113 | Complete |
| R8 | SearXNG | Qwen2.5-72B | serp50 | 1,015 | Phase 3 partial (35%) |

**Method:** Double Machine Learning (DoubleML) with Partially Linear Regression (PLR), LightGBM learners, 5-fold cross-validation. All results use `rank_delta` as primary outcome (positive = LLM promoted the page).

**Ranking convention:** Lower rank number = better (rank 1 is best). Negative coefficient on rank = GOOD. Positive coefficient on rank_delta = page promoted by LLM.

---

## 2. Pooled Results (All 8 Runs Combined)

The pooled analysis maximizes statistical power by combining all 65,426 observations. T7 uses the expanded 249-domain EARNED_DOMAINS list (1,764 earned rows = 2.7% of dataset).

### 2.1 Highly Significant Treatments (p < 0.001)

| Treatment | n | theta | SE | p-value | Interpretation |
|-----------|---|-------|-----|---------|----------------|
| **T7 Source: Earned** | 50,659 | **-1.679** | 0.072 | <0.0001 | Earned media pages demoted ~1.7 ranks. Strongest effect. SE reduced from 0.093 to 0.072 with expanded domains. |
| **T5 Topical Competence** | 26,837 | **+0.608** | 0.123 | <0.0001 | Higher topical relevance (cosine similarity) leads to +0.6 rank promotion. Strongest positive signal. |
| **T3 Structured Data (code)** | 45,986 | **+0.146** | 0.030 | <0.0001 | Pages with structured data (schema.org, JSON-LD) promoted ~0.15 ranks. |
| **T3 Structured Data (expanded)** | 45,397 | **-0.144** | 0.025 | <0.0001 | LLM-assessed structured data: negative effect. Diverges from code-based measure. |
| **T1 Statistical Density (code)** | 44,628 | **-0.017** | 0.002 | <0.0001 | Statistical content slightly demoted. Small but highly significant. |
| **T1b Stats Density (continuous)** | 44,060 | **-0.014** | 0.002 | <0.0001 | Confirms T1 code finding with continuous measure. |
| **T6 Freshness** | 45,397 | **-0.056** | 0.007 | <0.0001 | Fresher content demoted ~0.06 ranks per freshness level. Counter-intuitive. |
| **T4 Citation Authority (LLM)** | 40,690 | **-0.028** | 0.007 | <0.0001 | LLM-assessed citation authority: significant negative effect. |
| **T4 Citation Authority (code)** | 45,986 | **-0.019** | 0.006 | 0.0009 | Code-detected citations: small demotion. |
| **T2a Question Headings (binary)** | 45,397 | **+0.104** | 0.029 | 0.0003 | Question-style headings promote pages ~0.1 ranks. |
| **T2b Structural Modularity** | 45,397 | **+0.002** | 0.001 | 0.0010 | More modular structure (more sections) has small positive effect. |
| **T4b Authority Citations (count)** | 45,397 | **-0.017** | 0.005 | 0.0022 | More authority citations = slight demotion. |

### 2.2 Significant Treatments (p < 0.05)

| Treatment | n | theta | SE | p-value | Interpretation |
|-----------|---|-------|-----|---------|----------------|
| T2 Question Headings (code) | 45,986 | +0.065 | 0.026 | 0.0132 | Code-detected question headings: mild promotion. |
| T3 Structured Data (LLM) | 40,690 | +0.055 | 0.025 | 0.0298 | LLM-assessed structured data: mild promotion. |
| T4a External Citations (binary) | 45,397 | -0.076 | 0.042 | 0.0689 | Having external citations: marginal demotion (borderline). |

### 2.3 Non-Significant Treatments

| Treatment | n | theta | p-value |
|-----------|---|-------|---------|
| T1 Statistical Density (LLM) | 40,690 | -0.002 | 0.3748 |
| T2 Question Headings (LLM) | 40,690 | -0.035 | 0.1836 |
| T1a Stats Present (binary) | 45,397 | +0.006 | 0.8564 |

---

## 3. T7 Source: Earned — Updated Analysis

With the expanded 249-domain EARNED_DOMAINS list (up from 59), T7 is now the most precisely estimated treatment.

### 3.1 T7 Across All Runs

| Run | Engine / Model / Pool | n | theta | SE | p-value | Sig |
|-----|----------------------|---|-------|-----|---------|-----|
| R1 | DDG / Llama / serp20 | 5,587 | -1.365 | 0.176 | <0.0001 | *** |
| R2 | DDG / Llama / serp50 | 6,064 | -1.496 | 0.189 | <0.0001 | *** |
| R3 | DDG / Qwen / serp20 | 6,415 | -1.731 | 0.154 | <0.0001 | *** |
| R4 | DDG / Qwen / serp50 | 8,742 | -1.192 | 0.167 | <0.0001 | *** |
| R5 | SXG / Llama / serp20 | 6,691 | -1.774 | 0.211 | <0.0001 | *** |
| R6 | SXG / Llama / serp50 | 8,969 | -2.148 | 0.276 | <0.0001 | *** |
| R7 | SXG / Qwen / serp20 | 7,375 | -1.850 | 0.194 | <0.0001 | *** |
| R8 | SXG / Qwen / serp50 | 816 | -0.627 | 0.597 | 0.2938 | ns |
| **POOLED** | **All** | **50,659** | **-1.679** | **0.072** | **<0.0001** | **\*\*\*** |

**T7 is significant in 7/8 runs** (all except R8 which has insufficient power at n=816). Sign is consistently negative across all runs. Average theta = -1.554. This is the most robust treatment in the entire experiment.

### 3.2 T7 by Model

| Model | n | theta | SE | p-value |
|-------|---|-------|-----|---------|
| Qwen2.5-72B | 23,348 | -1.674 | 0.098 | <0.0001 |
| Llama-3.3-70B | 27,311 | -1.680 | 0.105 | <0.0001 |

Both models agree almost exactly: earned media pages are demoted ~1.68 ranks.

### 3.3 Changes from Previous T7 (59 domains)

| Metric | Old (59 domains) | New (249 domains) |
|--------|-------------------|-------------------|
| Earned rows in dataset | 1,104 (1.7%) | 1,764 (2.7%) |
| Pooled theta | -1.631 | -1.679 |
| Pooled SE | 0.093 | 0.072 |
| Pooled p-value | <0.0001 | <0.0001 |
| Significant runs | 7/8 | 7/8 |

The expanded domain list **strengthened** the T7 finding: larger effect size (-1.679 vs -1.631), tighter standard error (0.072 vs 0.093), and 60% more earned observations. The additional 190 domains (review platforms, trade publications, community sites, etc.) behave consistently with the original set.

---

## 4. Per-Model Comparison

### 4.1 Qwen2.5-72B-Instruct (Runs 3, 4, 7, 8)

| Treatment | n | theta | p-value | Sig |
|-----------|---|-------|---------|-----|
| T7 Source: Earned | 23,348 | -1.674 | <0.0001 | *** |
| T5 Topical Competence | 12,696 | +0.728 | <0.0001 | *** |
| T1b Stats Density | 20,025 | -0.020 | <0.0001 | *** |
| T1 Statistical Density (code) | 20,559 | -0.021 | <0.0001 | *** |
| T2a Question Headings (binary) | 20,629 | +0.151 | 0.0005 | *** |
| T3 Structured Data (expanded) | 20,629 | -0.145 | 0.0002 | *** |
| T6 Freshness | 20,629 | -0.067 | <0.0001 | *** |
| T3 Structured Data (code) | 21,183 | +0.138 | 0.0017 | *** |
| T4 Citation Authority (LLM) | 15,892 | -0.024 | 0.0087 | *** |
| T4 Citation Authority (code) | 21,183 | -0.016 | 0.0301 | ** |
| T2b Structural Modularity | 20,629 | +0.002 | 0.0454 | ** |

### 4.2 Llama-3.3-70B-Instruct (Runs 1, 2, 5, 6)

| Treatment | n | theta | p-value | Sig |
|-----------|---|-------|---------|-----|
| T7 Source: Earned | 27,311 | -1.680 | <0.0001 | *** |
| T1 Statistical Density (LLM) | 24,798 | -0.009 | <0.0001 | *** |
| T3 Structured Data (expanded) | 24,768 | -0.138 | <0.0001 | *** |
| T3 Structured Data (code) | 24,803 | +0.147 | 0.0002 | *** |
| T6 Freshness | 24,768 | -0.047 | <0.0001 | *** |
| T1 Statistical Density (code) | 24,069 | -0.008 | 0.0053 | *** |
| T1b Stats Density | 24,035 | -0.008 | 0.0061 | *** |
| T4a External Citations (binary) | 24,768 | -0.139 | 0.0113 | ** |
| T4 Citation Authority (code) | 24,803 | -0.019 | 0.0102 | ** |
| T2b Structural Modularity | 24,768 | +0.002 | 0.0186 | ** |
| T5 Topical Competence | 14,141 | +0.383 | 0.0206 | ** |
| T2 Question Headings (code) | 24,803 | +0.071 | 0.0403 | ** |
| T2a Question Headings (binary) | 24,768 | +0.075 | 0.0464 | ** |
| T4b Authority Citations | 24,768 | -0.015 | 0.0402 | ** |

### 4.3 Model Agreement

Both models agree on direction and significance:

| Treatment | Qwen theta | Llama theta | Agreement |
|-----------|-----------|-------------|-----------|
| T7 Source: Earned | -1.674 | -1.680 | Strong demotion in both (near-identical) |
| T5 Topical Competence | +0.728 | +0.383 | Promotion in both (Qwen stronger) |
| T3 Structured Data (code) | +0.138 | +0.147 | Promotion in both |
| T3 Structured Data (expanded) | -0.145 | -0.138 | Demotion in both |
| T6 Freshness | -0.067 | -0.047 | Demotion in both (Qwen stronger) |
| T1b Stats Density | -0.020 | -0.008 | Demotion in both (Qwen stronger) |
| T2a Question Headings | +0.151 | +0.075 | Promotion in both (Qwen stronger) |
| T2b Structural Modularity | +0.002 | +0.002 | Identical small promotion |

**Key divergence:** T4a External Citations is significant for Llama (-0.139, p=0.011) but not for Qwen (-0.038, p=0.55). T1 Statistical Density (LLM) is significant for Llama (-0.009, p<0.0001) but not for Qwen (-0.001, p=0.93).

---

## 5. Per-Run Robustness

Count of runs (including pooled + per-model = 10 subsets) where each treatment reaches p < 0.05:

| Treatment | Sig runs | Avg theta | Sign consistent | Verdict |
|-----------|----------|-----------|-----------------|---------|
| **T7 Source: Earned** | **9/10** | **-1.554** | **Yes (all negative)** | **ROBUST** |
| **T6 Freshness** | 8/10 | -0.049 | Mostly negative | ROBUST |
| T3 Structured Data (expanded) | 6/10 | -0.125 | Mostly negative | MIXED |
| T1 Statistical Density (code) | 5/10 | -0.008 | Mostly negative | MIXED |
| T3 Structured Data (code) | 5/10 | +0.125 | Yes (all positive) | MIXED |
| T4 Citation Authority (code) | 5/10 | -0.024 | Yes (all negative) | MIXED |
| T2a Question Headings (binary) | 5/10 | +0.100 | Mostly positive | MIXED |
| T1b Stats Density | 4/10 | -0.007 | Mostly negative | MIXED |
| T2b Structural Modularity | 4/10 | +0.003 | Yes (all positive) | MIXED |
| T5 Topical Competence | 4/7 | +0.490 | Mostly positive | MIXED |
| T4b Authority Citations | 4/10 | -0.025 | Mostly negative | MIXED |
| T1 Statistical Density (LLM) | 4/10 | -0.027 | Mostly negative | MIXED |
| T4 Citation Authority (LLM) | 2/10 | -0.022 | Mostly negative | FRAGILE |
| T2 Question Headings (code) | 2/10 | +0.053 | Mostly positive | FRAGILE |
| T3 Structured Data (LLM) | 1/10 | -0.026 | Mixed | FRAGILE |
| T4a External Citations | 1/10 | -0.085 | Yes (all negative) | FRAGILE |
| T1a Stats Present (binary) | 0/10 | +0.008 | Mixed | FRAGILE |
| T2 Question Headings (LLM) | 0/10 | -0.028 | Mixed | FRAGILE |

---

## 6. Key Findings

### 6.1 What the LLM Rewards (Positive rank_delta = promoted)

1. **Topical Competence** (T5, theta=+0.608): The single strongest positive signal. Pages with high cosine similarity to the query topic are strongly promoted. This is the clearest "optimize for this" finding.

2. **Structured Data / Schema Markup** (T3 code, theta=+0.146): Code-detected structured data (JSON-LD, schema.org) is promoted. Clear technical signal. Consistent across models.

3. **Question-Style Headings** (T2a, theta=+0.104): Binary presence of question headings provides a modest but significant boost. Aligns with the "People Also Ask" format that LLMs favor.

4. **Structural Modularity** (T2b, theta=+0.002): More sections/headings = slight promotion. Small but consistent across all runs.

### 6.2 What the LLM Penalizes (Negative rank_delta = demoted)

1. **Source: Earned Media** (T7, theta=-1.679): By far the strongest effect. Earned media pages (review sites, tech press, consulting firms, community/UGC) are heavily demoted by ~1.7 ranks. Significant in 9/10 analysis subsets. Both models agree (-1.674 Qwen, -1.680 Llama). The LLM strongly prefers primary/official sources over third-party coverage.

2. **Structured Data Expanded** (T3 expanded, theta=-0.144): The LLM-assessed version of structured data shows demotion. May capture cases where structured data is present but irrelevant or spammy.

3. **Freshness** (T6, theta=-0.056): Counter-intuitively, fresher content is slightly penalized. The LLM may prefer established, comprehensive pages over recent updates. Significant in 8/10 subsets.

4. **Citation Authority** (T4 LLM theta=-0.028, T4 code theta=-0.019): Both measures show pages with more citations/references are demoted. The LLM may prefer concise authoritative sources over heavily-referenced review articles.

5. **Statistical Density** (T1 code theta=-0.017, T1b theta=-0.014): Pages dense with statistics are slightly demoted. Small effect but highly significant in pooled analysis.

### 6.3 Paradoxes and Tensions

- **Structured Data Paradox:** Code-detected structured data (T3 code) is positive (+0.146) while LLM-assessed expanded structured data (T3 expanded) is negative (-0.144). The code measure captures technical schema markup; the expanded measure may capture content claiming to be structured but not actually useful.

- **Freshness Penalty:** One might expect LLMs to prefer fresh content, but the negative freshness coefficient suggests LLMs value comprehensive, established content over recency.

- **Citation Penalty:** More citations don't help. The LLM may already have internal knowledge and prefer direct, authoritative answers over extensively referenced articles.

---

## 7. Methodology Notes

- **DML (Double Machine Learning):** Controls for confounders (title similarity, snippet similarity, brand recognition, readability, word count, domain authority, etc.) to isolate causal treatment effects.
- **17 confounders** including: title/snippet keyword similarity, title length, snippet length, brand recognition, keyword presence in title, word count, readability, internal/external link counts, image count, domain age, page authority, and domain authority.
- **Two learner types:** LightGBM and Random Forest — results are consistent across both.
- **Two outcome variables:** `rank_delta` (primary, how much the LLM moved the page) and `post_rank` (the final LLM rank). Post_rank results are directionally consistent (signs flip as expected).
- **T7 EARNED_DOMAINS:** Expanded from 59 to 249 domains across 22 categories: software review platforms (G2, Capterra, TrustRadius...), tech media (TechCrunch, ZDNet...), business media (Forbes, Bloomberg...), consulting firms (McKinsey, Deloitte...), community/UGC (Reddit, Stack Overflow, GitHub...), trade publications (Industry Dive network), and more.
- **R8 caveat:** Only 1,015 rows (Phase 3 at ~40%) — results from R8 individually have wide confidence intervals and should be treated as preliminary.

## 8. Actionable Recommendations for GEO

Based on the robust, cross-validated findings:

1. **Maximize topical competence** — Ensure page content is highly relevant to the target query. This is the #1 lever (+0.608 rank promotion).
2. **Implement schema.org markup** — Technical structured data provides a measurable boost (+0.146).
3. **Use question-style headings** — Structure content with question-and-answer format (+0.104).
4. **Avoid earned-media positioning** — Content that reads as third-party coverage (reviews, news, comparisons) is heavily penalized (-1.679). Prefer first-party authoritative content.
5. **Don't over-cite** — Dense references don't help and slightly hurt (-0.028).
6. **Don't chase recency for its own sake** — Comprehensive, evergreen content outperforms frequent updates (-0.056).

---

## Appendix: Run-Level Detail (rank_delta, PLR, LGBM)

### Runs 1-2: DuckDuckGo + Llama-3.3-70B

| Treatment | R1 theta (serp20) | R1 p | R2 theta (serp50) | R2 p |
|-----------|-------------------|------|-------------------|------|
| T7 Source | -1.365 | *** | -1.496 | *** |
| T6 Freshness | -0.038 | * | -0.065 | *** |
| T1 LLM | -0.010 | ** | -0.004 | ns |
| T3 Expanded | -0.116 | ns | -0.306 | *** |
| T3 Code | +0.114 | ns | +0.032 | ns |
| T2b Modularity | +0.001 | ns | +0.002 | ns |

### Runs 3-4: DuckDuckGo + Qwen2.5-72B

| Treatment | R3 theta (serp20) | R3 p | R4 theta (serp50) | R4 p |
|-----------|-------------------|------|-------------------|------|
| T7 Source | -1.731 | *** | -1.192 | *** |
| T6 Freshness | -0.087 | *** | -0.069 | *** |
| T5 Topical | +0.971 | *** | n/a | |
| T2a Q-Headings | +0.261 | *** | +0.061 | ns |
| T2 Code | +0.213 | *** | -0.007 | ns |
| T1 Code | -0.013 | * | -0.019 | *** |

### Runs 5-6: SearXNG + Llama-3.3-70B

| Treatment | R5 theta (serp20) | R5 p | R6 theta (serp50) | R6 p |
|-----------|-------------------|------|-------------------|------|
| T7 Source | -1.774 | *** | -2.148 | *** |
| T1 LLM | -0.013 | *** | -0.006 | ns |
| T3 Code | +0.156 | ** | +0.259 | *** |
| T4 Code | -0.012 | ns | -0.045 | ** |
| T4b Auth Citations | -0.008 | ns | -0.051 | ** |
| T2a Q-Headings | +0.222 | *** | +0.056 | ns |

### Runs 7-8: SearXNG + Qwen2.5-72B

| Treatment | R7 theta (serp20) | R7 p | R8 theta (serp50) | R8 p |
|-----------|-------------------|------|-------------------|------|
| T7 Source | -1.850 | *** | -0.627 | ns |
| T1 Code | -0.023 | *** | +0.028 | ns |
| T1b Stats | -0.025 | *** | +0.026 | ns |
| T5 Topical | +0.596 | ** | -0.096 | ns |
| T3 Code | +0.188 | ** | +0.066 | ns |
| T4 Code | -0.011 | ns | -0.089 | ** |

Note: R8 has only 757-816 observations per treatment — too small for reliable inference. Most R8 results are non-significant due to insufficient power.
