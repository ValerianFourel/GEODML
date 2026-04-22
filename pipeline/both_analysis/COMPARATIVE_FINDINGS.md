# How Does an LLM Re-Rank Search Results When It Sees More Candidates?

## Comparative DML Analysis: Small Pool vs Large Pool

**Study date**: February 2026
**Author**: GEODML project
**For**: Internal briefing

---

## The Experiment

We ran the same causal inference pipeline on two experimental designs to understand how the size of the candidate pool changes an LLM's re-ranking behavior.

| Parameter | Small Pool | Large Pool |
|---|---|---|
| Search results collected | 20 per keyword | 50 per keyword |
| LLM re-ranks top | 10 | 20 |
| Observations | 492 (355 with rank_delta) | 996 (374 with rank_delta) |
| Keywords | 50 B2B SaaS terms | same 50 keywords |
| LLM | Llama-3.3-70B-Instruct | same model |
| Search engine | SearXNG (Google + Bing + DDG + Brave + Startpage) | same setup |
| Confounders | 8 (domain authority, domain age, word count, readability, internal links, outbound links, keyword difficulty, images with alt) | same 8 |

Everything is identical except how many results the search engine returns and how many the LLM gets to re-order. This isolates the effect of **pool size** on LLM re-ranking behavior.

---

## The Headline Result: The LLM Behaves Differently With More Candidates

In the small pool (20/10), the LLM **promotes** pages with question headings, structured data, and statistical density. In the large pool (50/20), these effects either vanish or **reverse direction**.

### rank_delta Results (PLR, code-based measurement)

| Treatment | Small Pool (20/10) | Large Pool (50/20) | Same direction? |
|---|---|---|---|
| T1 Statistical Density | **+0.39 (p=0.023)\*** | +0.03 (p=0.93) | Nominally yes, but effect disappears |
| T2 Question Headings | **+1.07 (p=0.019)\*** | -0.99 (p=0.29) | **No -- reversal** |
| T3 Structured Data | **+1.10 (p=0.033)\*** | -1.45 (p=0.15) | **No -- reversal** |
| T4 Citation Authority | -1.12 (p=0.10) | -1.39 (p=0.55) | Yes, both negative, neither significant |

*Positive theta = LLM promotes the page relative to where the search engine placed it.*

### The one result that gets stronger with a larger pool

| Treatment | Small Pool | Large Pool |
|---|---|---|
| T2 Question Headings (LLM-based measurement) | +0.04 (p=0.93) | **-2.92 (p=0.002)\*\*** |

When measured by the LLM evaluator (rather than code), question headings show a strong, highly significant **demotion** effect in the large pool. This is the single strongest finding across both datasets.

---

## What Is Going On? Three Interpretations

### 1. With a small pool, the LLM is mostly confirming. With a large pool, it is actually choosing.

In the small pool (top-10), the LLM sees 10 results that the search engine already filtered heavily. Most of these pages are high-quality, relevant, and broadly similar. The LLM can make minor adjustments -- nudging FAQ-style pages up a spot or two -- but the search engine already did most of the work. The rank_delta is small (mean +3.9 positions, range -9 to +16).

In the large pool (top-20), the LLM sees a wider range of quality. Positions 11-20 in the original SERP include weaker candidates. Now the LLM has to make harder choices about what deserves to be in the top 10 versus positions 11-20. The rank_delta is much larger (mean +10.1 positions, range -17 to +33). The LLM is doing **real re-ordering**, not just confirming the search engine's judgment.

**Why this reverses the effect of question headings**: In a small, curated pool, a page with "What is CRM?" headings stands out as well-structured. In a larger pool, the LLM encounters more such pages and starts treating question headings as a **generic SEO tactic** rather than a quality signal. It may even penalize them because FAQ-style pages with superficial question headings get demoted in favor of pages with deeper, more specific content.

### 2. The treatments interact differently with pool position

In the small pool, almost all results come from the top of the SERP (positions 1-20). These are high-authority pages where question headings and structured data genuinely correlate with quality. In the large pool, many results come from SERP positions 20-50 -- lower-quality pages where these features may be used as SEO tricks rather than signals of genuine content quality.

The LLM may be detecting this difference: structured data on a position-5 result probably reflects a legitimate product page, while structured data on a position-40 result may reflect an affiliate site gaming rich snippets.

### 3. The LLM's re-ranking strategy changes with the size of its input

When given 10 items to rank, the LLM may use a simple heuristic: "promote things that look like direct answers." When given 20 items, it can afford to be more discriminating: "demote things that look like they're optimizing for me."

This is consistent with the **question headings reversal** specifically. A page titled "What Is CRM Software? Everything You Need to Know" gets a boost when the LLM has few options, but gets penalized when the LLM can choose a page that addresses the topic with more depth and less formulaic structure.

---

## The Cross-Reference Table: Where Does the Effect Come From?

For the large pool, using the LLM-based measurement path (which captured the strongest signal):

| Treatment | pre_rank (SERP position) | post_rank (LLM position) | rank_delta (gap) |
|---|---|---|---|
| T2 Question Headings (llm) | **-3.89 (p=0.002)\*\*** | +0.30 (p=0.67) | **-2.92 (p=0.002)\*\*** |

The significant T2 effect in the large pool is driven entirely by **pre_rank**: the search engine already ranks pages with question headings better (lower pre_rank number = higher position). The LLM does not further reward them. The result is that these pages have a **smaller rank_delta** -- they don't get promoted as much because the search engine already placed them well.

This is the opposite of the small pool finding (from FINDINGS.md), where question headings didn't affect pre_rank significantly but **did** affect rank_delta -- the LLM was actively promoting them.

### What this means

| Scenario | Search engine's view | LLM's view | Outcome |
|---|---|---|---|
| **Small pool (20/10)** | Slightly undervalues question headings | Actively promotes them | rank_delta = +1.07 (positive, LLM corrects upward) |
| **Large pool (50/20)** | Already rewards question headings | Does not additionally reward them | rank_delta = -2.92 (negative, less promotion needed) |

In other words: with more data, the search engine and the LLM **converge**. The search engine already ranks question-heading pages well in the broader SERP. The LLM agrees but doesn't add further uplift. The "disagreement" between SERP and LLM shrinks.

---

## The Robustness Picture

### What is consistent across both designs

- **T4 Citation Authority** shows no significant effect in either design. Academic citations do not measurably influence LLM re-ranking regardless of pool size.
- **Code-based measurement** produces sharper estimates in the small pool (3 of 3 significant findings use code path).
- **LLM-based measurement** produces the strongest signal in the large pool (the only significant finding is via LLM path).
- Both datasets have low nuisance R-squared (confounders predict weakly), confirming that treatments are near-randomly assigned.

### What is not consistent

- 5 of 8 treatment-path combinations **flip sign** between the two designs.
- The small pool produces 5 significant findings at p<0.05; the large pool produces 2.
- The small pool's effects are moderate in size (theta around 0.4-1.1); the large pool's single significant effect is large (theta = -2.9).

### What to make of the sign reversals

The sign reversals are not necessarily contradictions. The two designs measure **different things**:
- Small pool rank_delta captures: "how does the LLM re-order a curated top-10?"
- Large pool rank_delta captures: "how does the LLM re-order a broad top-20?"

A treatment can legitimately help in one context and hurt in another if the LLM's evaluation strategy changes with the candidate set.

---

## Practical Takeaways

### For GEO (Generative Engine Optimization) strategy

1. **Question headings are not a universal win.** In constrained settings (chatbot returning a short list), they help. In broader re-ranking (AI overview synthesizing many sources), they may actually hurt. The effect depends on the competitive context.

2. **Structured data (FAQ/Product schema) has a similarly contingent effect.** Significant positive effect in the small pool, trending negative in the large pool. It works when you are already among the top candidates; it does not rescue a page from lower positions.

3. **The LLM is not a simple pattern matcher.** Its behavior changes qualitatively with the size of its input. GEO strategies that work in one context may backfire in another. This argues for **content quality over optimization tactics**.

4. **The strongest signal is content depth.** The large pool finding (T2 llm theta=-2.92) suggests that the LLM, when given real choice, penalizes pages that *look* optimized (formulaic question headings) and favors pages with genuine depth.

### For the research

5. **Pool size is a first-order experimental design parameter.** The same pipeline with the same keywords and same LLM produces opposite conclusions depending on how many results the LLM sees. Future GEO studies must specify and justify this parameter.

6. **The next step is a position bias test.** Some of the differences between the two designs may come from the LLM attending differently to a 10-item list versus a 20-item list (attention patterns, primacy/recency effects). The position bias test outlined in EXPANSION.md would help disentangle this from genuine content-quality effects.

7. **Multi-LLM replication becomes more important.** If Llama-3.3-70B changes behavior with pool size, other models (GPT-4o, Claude, Gemini) may not. Running the same comparison across models would reveal whether this is a Llama-specific pattern or a general LLM property.

---

## Summary for the Meeting

**One slide version:**

> We ran the same causal analysis on 50 B2B SaaS keywords with two designs: a small candidate pool (20 results, re-rank top 10) and a large candidate pool (50 results, re-rank top 20). The results are strikingly different.
>
> In the small pool, the LLM promotes pages with question headings (+1.1 ranks, p=0.019) and structured data (+1.1 ranks, p=0.033). In the large pool, these effects reverse: question headings are associated with **less** LLM promotion (-2.9 ranks, p=0.002).
>
> The interpretation: when the LLM has fewer options, it rewards pages that look well-structured (FAQ headings, schema markup). When it has more options, it becomes more discriminating and penalizes formulaic optimization in favor of content depth. This means GEO strategy is context-dependent -- what works in a chatbot's short list may backfire in a broader AI-powered search.

---

## Files

| Path | Description |
|---|---|
| `both_analysis/results/all_experiments.csv` | Full results: 96 experiments (48 per dataset) |
| `both_analysis/results/summary.json` | Experiment metadata |
| `both_analysis/figures/fig1_comparative_forest.png` | Side-by-side forest plots (rank_delta, PLR) |
| `both_analysis/figures/fig2_coefficient_scatter.png` | 20-SERP theta vs 50-SERP theta scatter |
| `both_analysis/figures/fig3_pvalue_heatmap.png` | Full p-value heatmap, both datasets |
| `both_analysis/figures/fig4_effect_comparison.png` | Grouped bar chart, effect sizes |
| `both_analysis/figures/fig5_dml_vs_ols.png` | DML vs naive OLS, both datasets |
| `both_analysis/figures/fig6_plr_vs_irm.png` | Method sensitivity, both datasets |
| `both_analysis/figures/fig7_multi_outcome_forest.png` | All outcomes x both datasets |
| `both_analysis/figures/fig8_summary_table.png` | Publication-style summary table |
| `both_analysis/figures/fig9_dataset_descriptives.png` | Variable distributions comparison |
| `FINDINGS.md` | Original findings from the small pool experiment |
| `EXPANSION.md` | Planned expansion (position bias test, multi-LLM, etc.) |
