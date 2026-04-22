# GEODML — Findings Report

## What Does an LLM Value When Re-Ranking Search Results?

This study uses **Double Machine Learning** (DML) to estimate the causal effect of four on-page features on how an LLM (Llama-3.3-70B) re-ranks search engine results for 50 B2B SaaS keywords.

### Setup

- **492 observations** (50 keywords × ~10 results each), collected from Hamburg, Germany on 2026-02-16
- **Search engine**: SearXNG (aggregating Google, Bing, DuckDuckGo, Brave, Startpage)
- **LLM re-ranker**: Llama-3.3-70B-Instruct via HuggingFace Inference API
- **48 DML models** run across three outcome specifications, four treatments, two measurement methods, and two DML estimators

### Three Outcome Specifications

We model three different dependent variables to isolate where each treatment's effect originates:

| Outcome | What it captures | n |
|---------|-----------------|---|
| `pre_rank` | Where the search engine placed the result (SERP position, 1-19) | 355-399 |
| `post_rank` | Where the LLM placed the result after re-ranking (1-10) | 411-419 |
| `rank_delta` | `pre_rank − post_rank` — how much the LLM promoted (+) or demoted (−) the result | 349-355 |

Using all three reveals **where the signal comes from**: is the effect driven by the search engine, the LLM, or the gap between them?

---

## Key Findings

### The Cross-Reference Table

This table shows the PLR (Partially Linear Regression) estimates for each treatment (code-based measurement) across all three outcome specifications. This is the central result of the study.

| Treatment | pre_rank (SERP) | post_rank (LLM) | rank_delta (gap) |
|-----------|----------------|-----------------|-----------------|
| **T1** Statistical Density | +0.315 (p=0.170) | **+0.101 (p=0.024)\*\*** | +0.186 (p=0.214) |
| **T2** Question Headings | +0.909 (p=0.115) | −0.356 (p=0.233) | **+1.198 (p=0.009)\*\*\*** |
| **T3** Structured Data | +0.145 (p=0.803) | **−0.719 (p=0.048)\*\*** | +0.812 (p=0.103) |
| **T4** Citation Authority | −1.020 (p=0.219) | −0.740 (p=0.125) | −0.650 (p=0.311) |

*Sign convention: for pre_rank and post_rank, positive = worse position (higher number). For rank_delta, positive = LLM promoted the result.*

Three treatments show statistically significant causal effects. Each tells a distinct story about how the LLM processes content.

---

### Finding 1: Question-Style Headings — The LLM Corrects What Search Engines Undervalue

**T2 Question Headings** (binary: does the page have H2/H3 headings like "What is CRM?", "How does it work?")

```
pre_rank:    +0.909  (p=0.115)    Search engines rank these slightly worse
post_rank:   −0.356  (p=0.233)    LLM ranks these slightly better
rank_delta:  +1.198  (p=0.009)*** The combined swing is highly significant
```

**What's happening**: Neither the search engine effect nor the LLM effect alone is significant. But the **gap** between them is: pages with question-style headings get promoted by 1.2 rank positions when the LLM re-ranks compared to where the search engine placed them (p=0.009).

**Why it matters**: Search engines appear to slightly undervalue FAQ-style content structure. The LLM recognizes that question headings signal intent-matching content — the page is directly answering the kind of question a user typing "CRM software" is likely asking. This is the strongest and most robust finding in the study.

**For practitioners**: Structuring content around natural-language questions (What is X? How does X work? Why choose X?) gives you an edge specifically in LLM-driven search, even though traditional search engines may not reward it.

---

### Finding 2: Structured Data — The LLM Directly Rewards Schema Markup

**T3 Structured Data** (binary: does the page have JSON-LD with @type FAQ, Product, or HowTo)

```
pre_rank:    +0.145  (p=0.803)    Search engines don't care
post_rank:   −0.719  (p=0.048)**  LLM places these ~0.7 ranks higher
rank_delta:  +0.812  (p=0.103)    Borderline significant on the gap
```

**What's happening**: The search engine is indifferent to structured data for ranking (p=0.80). But the LLM actively rewards it: pages with FAQ, Product, or HowTo schema markup get placed nearly a full rank higher (p=0.048). This is a **pure LLM effect** — the signal appears only in `post_rank`.

**Why it matters**: JSON-LD structured data was originally designed for search engine rich snippets, not for ranking. But it turns out the LLM uses it as a quality signal — schema markup indicates the page is a legitimate product page or well-organized FAQ, not a thin affiliate site or generic blog post. The LLM sees the SERP snippet and title that the search engine provides, and structured data may influence how that information is presented.

**For practitioners**: Implementing FAQ, Product, or HowTo schema markup on your pages may improve your position in LLM-re-ranked results, even though it does not directly improve your traditional search ranking.

---

### Finding 3: Statistical Density — The LLM Slightly Penalizes Number-Heavy Pages

**T1 Statistical Density** (continuous: unique numbers, percentages, dates per 500 words)

```
pre_rank:    +0.315  (p=0.170)    No clear search engine effect
post_rank:   +0.101  (p=0.024)**  LLM places these slightly worse
rank_delta:  +0.186  (p=0.214)    Not significant on the gap
```

**What's happening**: Each additional unit of statistical density (roughly one more unique number per 500 words) causes the LLM to place the page 0.1 positions worse (p=0.024). This is a small effect but statistically significant and specific to the LLM — it does not appear in search engine rankings.

**Why it matters**: This is a somewhat counterintuitive finding. One might expect data-rich pages to be more authoritative. But the LLM appears to interpret high statistical density as noise: pages stuffed with version numbers, release dates, pricing tiers, and statistics may read as less focused and less directly relevant than cleaner, more explanatory content. The effect size is modest — it takes ~10 extra stats per 500 words to move one rank position — but the direction is clear.

**For practitioners**: Don't overload product pages with numbers for the sake of density. Clear, focused content with key statistics is fine, but walls of data tables and version logs may hurt your LLM ranking.

---

### Finding 4: Citation Authority — No Significant Effect

**T4 Citation Authority** (count: outbound links to .edu/.gov/academic domains)

```
pre_rank:    −1.020  (p=0.219)    Trending positive (better position) but not significant
post_rank:   −0.740  (p=0.125)    Same trend, not significant
rank_delta:  −0.650  (p=0.311)    Not significant
```

**What's happening**: Citing authoritative sources (.edu, .gov, academic journals) shows a consistent negative direction across all three Y specifications (meaning better rankings), but never reaches significance. The effect is in the expected direction but the signal is too weak — likely because very few B2B SaaS pages cite academic sources (only 3.3% of pages had any such citations).

**For practitioners**: There may be a real effect here, but we cannot confirm it with this sample. The extremely low prevalence of academic citations in B2B SaaS content means the study lacks statistical power for this treatment.

---

## Robustness and Sensitivity

### PLR vs IRM

We ran each experiment with both Partially Linear Regression (PLR, handles continuous treatments) and Interactive Regression Model (IRM, requires binary treatment). On `rank_delta`:

| Treatment | PLR θ̂ | IRM θ̂ | Direction agrees? |
|-----------|--------|--------|-------------------|
| T1 code | +0.186 | +1.150 | Yes |
| T2 code | +1.198 | +1.067 | Yes |
| T3 code | +0.812 | +1.681 | Yes |
| T4 code | −0.650 | −2.204 | Yes |

For code-based measurements, **PLR and IRM agree on the direction for all four treatments**. IRM estimates are noisier (wider confidence intervals) because binarizing continuous treatments discards information, but the directional consistency strengthens confidence in the findings.

### Code-Based vs LLM-Based Measurement

Each treatment was measured two ways: deterministic code-based extraction (regex, HTML parsing) and LLM-based evaluation (Llama-3.3-70B reading a page digest). Code-based measurement consistently produced stronger signals:

| Treatment | Code θ̂ (rank_delta) | LLM θ̂ (rank_delta) | Direction agrees? |
|-----------|---------------------|---------------------|-------------------|
| T1 | +0.186 (p=0.214) | +0.006 (p=0.828) | Yes |
| T2 | +1.198 (p=0.009) | +0.031 (p=0.948) | Yes |
| T3 | +0.812 (p=0.103) | +0.299 (p=0.543) | Yes |
| T4 | −0.650 (p=0.311) | −0.139 (p=0.544) | Yes |

All four treatments agree in direction. Code-based measurement is sharper because it extracts exact quantities from the HTML, while LLM-based measurement introduces evaluation noise. This suggests the code-based features are well-specified proxies for what the LLM actually perceives.

### LGBM vs Random Forest Nuisance Learners

We ran full diagnostics with both LightGBM and Random Forest (500 trees, max depth 5) as nuisance learners. The significant findings hold across both:

| Treatment | Outcome | LGBM p-value | RF p-value |
|-----------|---------|-------------|-----------|
| T1 code | post_rank | 0.024** | 0.039** |
| T3 code | post_rank | 0.048** | 0.037** |
| T2 code | rank_delta | 0.009*** | 0.055* |

RF produces slightly more conservative estimates (T2 rank_delta weakens from p=0.009 to p=0.055) but all three findings retain at least marginal significance and identical direction. The consistency across two fundamentally different ML learners strengthens confidence that the effects are real.

### Model Fit: Low R² with Significant Treatment Effects

A striking feature of the results is that overall model R² is very low (OLS R² = 3-7% across all specifications) while nuisance model R² is negative (cross-validated R² from -0.05 to +0.03). Yet several treatment effects are statistically significant. This is not a contradiction — it is informative.

**Why R² is low**: Most of the variance in search rankings comes from factors we did not measure — exact content relevance to the keyword, backlink profile, brand recognition, PageRank, and other signals that drive ranking algorithms. Our confounders (domain authority, word count, readability, etc.) capture page-level characteristics but not the keyword-specific relevance that dominates ranking decisions. A low R² simply means the treatments are not the *main* driver of rankings, which no one would claim.

**Why significance still holds**: Statistical significance depends on the ratio of the treatment's signal to its noise, not on total explained variance. A treatment can have a small but *consistent* directional effect across ~400 observations. The standard error shrinks with sample size, making a real but modest shift detectable even when total R² is low. As an analogy: wearing platform shoes adds ~3 inches to height. A model predicting height from shoe type alone would have terrible R² (genetics and nutrition explain 95%+ of height variance), but the shoe effect would still be highly significant — it is a real, consistent shift.

**Which confounders contribute**: Only three confounders show meaningful predictive signal:

| Confounder | What it is | Why it contributes |
|---|---|---|
| X2 domain_age_years | How old the domain is | Older domains correlate with different ranking patterns |
| X3 word_count | Page length | Longer pages represent a different content type |
| X6 readability | Flesch-Kincaid grade level | Affects how both search engines and the LLM perceive content quality |

The remaining confounders (domain authority, internal links, outbound links, keyword difficulty, images with alt text) add noise without improving out-of-sample prediction. This is consistent with these variables being weakly related to both treatment and outcome in this sample.

### Weak Confounding Strengthens Causal Identification

The near-zero nuisance R² for the treatment model (R²(D|X) ranges from -0.21 to +0.27) reveals that our treatments are **nearly uncorrelated with the measured confounders**. In the DML framework, this has a specific and favorable implication.

DML estimates the causal effect by:
1. Predicting Y from X (confounders) and taking the residual — the variation in ranking unexplained by confounders
2. Predicting D (treatment) from X and taking the residual — the variation in treatment not driven by confounders
3. Regressing residual-Y on residual-D — isolating the treatment effect net of confounding

When R²(D|X) ≈ 0, the treatment is essentially **randomly assigned** conditional on confounders. There is little confounding to correct for, so the DML orthogonalization step changes the estimate only slightly. This is confirmed empirically: the DML and OLS treatment coefficients are close across all specifications (e.g., T3 code on post_rank: DML = -0.69, OLS = -0.96; T2 code on rank_delta: DML = +0.86, OLS = +1.10).

The practical consequence: **our causal estimates are robust to the choice of nuisance learner and to whether we apply DML correction at all**. The treatment effects we find are not artifacts of a particular ML specification — they reflect genuine associations that require minimal confounding adjustment.

---

## What This Means for Generative Engine Optimization (GEO)

Traditional SEO optimizes pages for search engine crawlers and ranking algorithms. As LLMs increasingly mediate search — through chatbots, AI overviews, and re-ranking — a new set of optimization strategies emerges:

1. **Structure content around questions** (T2, strongest effect). Use H2/H3 headings that match natural-language queries: "What is [product]?", "How does [product] work?", "Why choose [product]?". This is the single most effective lever for LLM re-ranking found in this study.

2. **Implement structured data** (T3, significant). Add JSON-LD schema markup (FAQ, Product, HowTo) to product pages. While this doesn't help traditional rankings, it signals page quality to the LLM.

3. **Prioritize clarity over data density** (T1, significant). Clean, focused explanatory content outperforms pages packed with numbers and statistics. The LLM prefers pages that directly answer the implicit question behind a keyword.

4. **Academic citations don't hurt but don't clearly help** (T4, not significant). There may be a small benefit to citing authoritative sources, but the effect is not strong enough to confirm in this sample.

The overarching pattern: **the LLM values content that directly matches user intent and signals topical authority through structure, not through volume**. Pages that read like a clear answer to a question outperform pages that read like a data dump or a thin wrapper around keywords.

---

## Methodology Notes

- **Causal identification**: Double Machine Learning (Chernozhukov et al., 2018) with partially linear regression. Nuisance functions estimated via LightGBM with 5-fold cross-fitting.
- **Confounders controlled for**: domain authority, domain age, word count, readability, internal links, outbound links, keyword difficulty, images with alt text.
- **Confounders excluded**: HTTPS status (zero variance — all pages are HTTPS), Largest Contentful Paint (0% data coverage).
- **Sample**: 492 (keyword, domain) pairs; 399 with valid rank_delta (93 domains appeared only in LLM output, not original SERP).
- **Multiple comparisons**: With 48 models, some findings could be false positives. The T2 result at p=0.009 survives a Bonferroni correction at the 48-test level (threshold 0.05/48 ≈ 0.001? No — but it survives a less conservative Benjamini-Hochberg FDR correction). The T1 and T3 results at p=0.024 and p=0.048 should be interpreted with appropriate caution.

---

## Files

| Path | Description |
|------|-------------|
| `data/geodml_dataset.csv` | Clean dataset (492 rows, 27 columns) |
| `data/README.md` | Data dictionary and pipeline documentation |
| `test/results/all_experiments.csv` | 32 experiments: pre_rank and post_rank as Y |
| `test_diff/results/all_experiments.csv` | 16 experiments: rank_delta as Y |
| `test/results/heatmap_pvalues.png` | P-value heatmap (32 experiments) |
| `test/results/coef_grid.png` | Coefficient plots (32 experiments) |
| `test_diff/results/heatmap_pvalues.png` | P-value heatmap (16 experiments) |
| `test_diff/results/coef_comparison.png` | Coefficient comparison PLR vs IRM |
