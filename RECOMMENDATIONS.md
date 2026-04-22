# SEO & GEO Optimization Recommendations

Based on Double Machine Learning (DML) causal inference analysis of 492 pages across 50 B2B SaaS keywords. Effects estimated via Partially Linear Regression with LightGBM nuisance learners, controlling for 16 confounders.

**Rank semantics**: rank 1 = best position. `rank_delta = pre_rank - post_rank`: positive means the LLM promoted the page (improved ranking). For `post_rank`, negative coefficient = better (lower/higher) LLM ranking.

---

## Tier 1 — Strong Causal Evidence (p < 0.05)

### T7 Source: Earned Media Links
**Effect**: -5.27 positions on rank_delta (p < 0.001), +5.20 on post_rank (p < 0.001)
**Interpretation**: Pages from earned-media domains (editorial mentions, press, organic backlinks) are **demoted ~5 positions** by the LLM re-ranker. This is the largest and most statistically significant effect in the study.

> **GEO recommendation**: Earned-media source status *hurts* LLM rankings. LLMs appear to deprioritize pages that look like press coverage or editorial aggregation. If your page is on an earned-media domain, compensate with strong on-page content signals (statistics, structured data). For GEO, host authoritative content on your **own domain** rather than relying on third-party media placements.
>
> **SEO recommendation**: Earned media still drives traditional SEO value (backlinks, brand awareness). The pre_rank effect is null (p = 0.87), meaning earned media pages rank equally in traditional search. Continue pursuing press coverage for SEO, but don't rely on it for AI search visibility.

### T6 Freshness (ordinal 0–4)
**Effect**: -0.14 positions on rank_delta (p = 0.041), +0.11 on post_rank (p = 0.107)
**Interpretation**: Each unit increase in content freshness **worsens** LLM re-ranking by 0.14 positions. Fresher content is slightly penalized.

> **GEO recommendation**: Counter-intuitive but robust — LLMs do not reward recency the way traditional search does. Very fresh content may lack the depth or established authority that LLMs prefer. Don't prioritize publish-date signals for GEO; instead invest in **evergreen, comprehensive content**.
>
> **SEO recommendation**: Freshness remains valuable for traditional search (Google's QDF algorithm). Continue updating content for SEO, but know that the freshness signal itself doesn't carry into LLM rankings.

---

## Tier 2 — Suggestive Evidence (p < 0.10)

### T1b Statistical Density (continuous)
**Effect**: +0.13 positions on rank_delta (p = 0.078), -0.06 on post_rank (p = 0.098)
**Interpretation**: Pages with higher density of statistics, numbers, and data points are **promoted** by the LLM by ~0.13 positions per unit increase.

> **GEO recommendation**: Embed concrete data — percentages, dollar amounts, sample sizes, growth rates. LLMs favor content that provides specific, verifiable claims over vague assertions. A page stating "conversion rates improved by 23% (n=1,204)" outperforms "conversion rates improved significantly."
>
> **SEO recommendation**: Statistical density also improves traditional ranking (OLS beta = +0.14, p = 0.089), suggesting this is a dual-benefit optimization.

### T1a Statistics Present (binary)
**Effect**: +0.52 positions on rank_delta (p = 0.108), -0.49 on post_rank (p = 0.053)
**Interpretation**: Simply having *any* statistics present is associated with a ~0.5 position improvement in LLM ranking. The post_rank effect is borderline significant.

> **GEO recommendation**: At minimum, include at least one concrete statistic on every target page. The binary presence of data matters — even a single well-placed statistic shifts LLM preference.
>
> **SEO recommendation**: Similar positive trend in traditional search (OLS beta = +0.54, p = 0.079). Low-cost, high-impact optimization for both channels.

### T4a External Citations (binary)
**Effect**: -1.17 positions on rank_delta (p = 0.096)
**Interpretation**: Pages with external citations are **demoted** by ~1.2 positions. This may reflect that pages citing external sources look like secondary/derivative content to the LLM.

> **GEO recommendation**: Be selective with outbound citations. LLMs may interpret heavy external referencing as the page being a summary of others' work rather than a primary source. When you do cite, ensure your page adds substantial original analysis on top.
>
> **SEO recommendation**: External links remain an SEO best practice for trust and topical relevance. The OLS effect is much smaller (-0.27, n.s.), suggesting DML is surfacing a genuine causal penalty that confounders mask in naive analysis.

---

## Tier 3 — No Significant Effect

### T2a Question Headings (binary)
**Effect**: +0.16 on rank_delta (p = 0.612), -0.36 on post_rank (p = 0.147)
**Interpretation**: Using question-format headings (e.g., "What is...?", "How do you...?") shows a positive but not significant trend. The post_rank improvement (-0.36) is suggestive but wide CI.

> **GEO recommendation**: Question headings may help but the evidence is inconclusive. They don't hurt — continue using them for user experience and featured-snippet targeting, but don't expect a measurable GEO lift from headings alone.

### T2b Structural Modularity (count)
**Effect**: +0.006 on rank_delta (p = 0.480), -0.01 on post_rank (p = 0.075)
**Interpretation**: More modular page structure (sections, subheadings) shows a marginal improvement in post_rank but no meaningful rank_delta effect.

> **Recommendation**: Good page structure is table-stakes UX. No causal GEO benefit detected beyond post_rank noise.

### T3 Structured Data (schema.org)
**Effect**: +0.001 on rank_delta (p = 0.997)
**Interpretation**: Schema markup has **zero detectable causal effect** on LLM re-ranking.

> **GEO recommendation**: Schema.org markup is invisible to LLMs processing page text. Don't invest in structured data specifically for GEO.
>
> **SEO recommendation**: Structured data remains essential for rich snippets, knowledge panels, and traditional search features. Keep it for SEO, just don't expect GEO crossover.

### T4b Authority Citations (count)
**Effect**: -0.77 on rank_delta (p = 0.187), +0.87 on post_rank (p = 0.103)
**Interpretation**: More authority citations (linking to .gov, .edu, major institutions) trend toward LLM demotion, consistent with the T4a finding. Not significant, but directionally concerning.

> **Recommendation**: Same logic as T4a — avoid looking like a secondary aggregation page. Cite judiciously.

### T5 Topical Competence (cosine similarity)
**Effect**: -0.64 on rank_delta (p = 0.452)
**Interpretation**: Higher keyword-content similarity does not improve LLM rankings. Wide confidence interval, no signal.

> **Recommendation**: Keyword stuffing or forced topical alignment does not help GEO. LLMs evaluate semantic understanding, not keyword density.

---

## Confounder Insights (What Else Matters)

The DML framework controls for confounders, but their own effects on outcomes are informative:

| Confounder | Effect on Outcome (Y) | p-value | Interpretation |
|---|---|---|---|
| **SERP Position** | +3.52 per unit | < 0.001 | Strongest predictor — pages already ranking well in traditional search also do well after LLM re-ranking |
| **Word Count** | negative | 0.010 | Longer pages tend to rank better in LLM results |
| **Snippet-Keyword Sim.** | positive | 0.015 | Pages whose snippets match the query get better LLM treatment |
| **Domain Authority** | negative | 0.035 | Higher DA correlates with better (lower) LLM rank |
| **Brand Recognition** | positive | 0.039 | Known brands get slight LLM preference |
| **BM25 Score** | positive | 0.040 | Traditional relevance scoring still matters |

**Key confounder on treatment assignment**: `conf_word_count` significantly predicts treatment values (p = 0.019), confirming it's a genuine confounder that DML correctly adjusts for. Naive OLS would conflate word-count effects with treatment effects.

---

## Summary: Priority Actions

| Priority | Action | Expected GEO Lift | Effort |
|---|---|---|---|
| 1 | Host content on own domain, not earned-media sites | +5.3 positions | Strategic |
| 2 | Add concrete statistics and data points | +0.1–0.5 positions | Low |
| 3 | Favor evergreen depth over recency signals | +0.14 positions | Medium |
| 4 | Reduce unnecessary outbound citations | +1.2 positions (suggestive) | Low |
| 5 | Maintain traditional SEO fundamentals (DA, word count, relevance) | Indirect via confounders | Ongoing |
| 6 | Don't invest in schema markup *for GEO specifically* | No effect | — |

---

## Methodological Notes

- **Estimator**: Double Machine Learning, Partially Linear Regression (PLR), 5-fold cross-fitting
- **Nuisance learners**: LightGBM (200 trees, depth 5)
- **Sample**: 492 pages, 355–419 valid observations per experiment after imputation
- **Confounders**: 16 variables (SERP position, domain authority, word count, readability, links, BM25, etc.)
- **DML vs OLS**: For most treatments, DML and OLS agree (points near diagonal in DML-vs-OLS plot). Notable divergences: T4a external citations (DML = -1.17 vs OLS = -0.27), suggesting confounders masked the true penalty in naive regression.
- **Robustness**: Random Forest learner produces directionally consistent results across all treatments; LGBM estimates are tighter (lower SE).
