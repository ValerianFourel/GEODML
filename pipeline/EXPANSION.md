# GEODML — Expansion Plan

## The Core Problem: What Explains Ranking Variability?

Our current confounders (X1-X10) produce **negative cross-validated R²** — they predict ranking outcomes *worse than the mean*. This means the variables we control for explain essentially none of the variance in where pages rank, either in the SERP or in the LLM's re-ranking. The treatments are near-randomly assigned conditional on these confounders, which makes causal identification easy but reveals a gap: **we don't yet know what actually drives rankings in this setting**.

This document outlines what confounders are missing, what new treatments to explore, and how to scale the experiment to get a fuller picture.

---

## Part 1: Missing Confounders

### 1A. What the LLM Actually Sees

This is the most important gap. When re-ranking, the LLM receives **only this** per result:

```
3. [hubspot.com] Streamline Your Entire Business With a Free CRM — HubSpot's free CRM powers your customer support, sales, and marketing...
```

It sees: **(1) position number, (2) domain name, (3) title, (4) snippet (truncated to 150 chars)**. It does **not** see the full page HTML. So the confounders that drive `post_rank` must come from these four elements, not from deep page analysis.

| Missing Confounder | What it captures | How to extract | Expected impact |
|---|---|---|---|
| **S1** Title-keyword semantic similarity | How well the title matches the query | Sentence embeddings (e.g., `all-MiniLM-L6-v2`) cosine similarity between keyword and title | High — the LLM likely promotes titles that directly address the query |
| **S2** Snippet-keyword semantic similarity | How well the snippet matches the query | Same embedding approach on snippet text | High — snippet is the main content the LLM reads |
| **S3** Title length (chars) | How much info the title conveys | `len(title)` | Moderate — longer titles may contain more persuasive detail |
| **S4** Snippet length (chars) | How much context the LLM gets | `len(snippet)` | Moderate — truncated snippets give less signal |
| **S5** Domain brand recognition | Whether the LLM "knows" the brand | LLM evaluation: "Is [domain] a well-known software company?" or proxy via Wikipedia page existence | High — LLMs have strong brand priors from pretraining |
| **S6** Title contains keyword | Exact keyword match in title | Simple substring check | Moderate — keyword stuffing vs. relevance signal |
| **S7** Domain TLD | .com vs .io vs .de vs ccTLD | Parse from URL | Low-moderate — .com may signal legitimacy to the LLM |
| **S8** SERP position (input rank) | Where the result appears in the prompt | Already have as `pre_rank` | High for `post_rank` — position bias in LLM attention |

**Priority**: S1, S2, S5 are the most likely to explain `post_rank` variance. These capture what the LLM actually processes.

**Implementation**: All can be extracted from the existing JSON results file — no new data collection needed. The SERP results already contain title and snippet for every result. Sentence embeddings can be computed locally with a small model.

### 1B. What Drives Search Engine Rankings (pre_rank)

For `pre_rank`, our confounders miss the major SEO signals. These require external tools or APIs.

| Missing Confounder | What it captures | How to get it | Expected impact | Difficulty |
|---|---|---|---|---|
| **B1** Backlink count | Number of external sites linking to the page | Ahrefs/Moz/Majestic API, or Common Crawl | Very high — the #1 ranking factor | Paid API ($99+/mo) or Common Crawl processing |
| **B2** Referring domains | Number of unique domains linking | Same as B1 | Very high — more important than raw count | Same |
| **B3** Keyword-in-URL | Whether the keyword appears in the URL path | Parse URL, fuzzy match against keyword | Moderate | Free — already have URLs |
| **B4** URL depth | How many path segments (`/a/b/c` = depth 3) | Parse URL | Low-moderate — homepage vs deep page | Free |
| **B5** Content freshness | When the page was last modified | HTTP `Last-Modified` header, or `<meta>` tags, or dates in content | Moderate | Free — can extract from cached HTML |
| **B6** Mobile-friendliness | Whether the page is mobile-optimized | Google PageSpeed API (already partially wired for X4) | Moderate | Free API |
| **B7** Core Web Vitals (CLS, FID, LCP) | Page experience signals | Google PageSpeed API | Moderate — but X4 LCP had 0% coverage | Free API, but coverage issues |
| **B8** Title tag length | SEO best practice: 50-60 chars | Parse `<title>` from HTML | Low | Free — from cached HTML |
| **B9** Meta description match | Whether the snippet comes from meta description | Compare SERP snippet to `<meta name="description">` | Low | Free — from cached HTML + JSON |
| **B10** Content-keyword relevance (BM25/TF-IDF) | How topically relevant the page body is to the keyword | Compute BM25 or TF-IDF score between page text and keyword | High — fundamental ranking signal | Free — from cached HTML |

**Priority**: B1/B2 (backlinks) are the elephant in the room — they likely explain the majority of `pre_rank` variance. B10 (content relevance) is the other major factor and is free to compute.

**Practical path**: Start with what's free (B3, B4, B5, B8, B9, B10 from cached HTML), then invest in backlink data (B1, B2) via a paid API or Common Crawl if the free confounders are still insufficient.

### 1C. What Might Differ Between SERP and LLM (rank_delta drivers)

For `rank_delta`, the interesting confounders are things that search engines value but LLMs don't (or vice versa):

| Confounder | Hypothesis | How to test |
|---|---|---|
| **D1** Backlinks without content relevance | High backlinks + low keyword relevance = SERP ranks it high, LLM demotes it | Interaction term B1 × B10 |
| **D2** Brand without product match | Famous brand but page isn't about the keyword = LLM demotes | S5 (brand) × S2 (snippet relevance) interaction |
| **D3** Position bias | LLM may attend more to results shown first in the prompt | Test by randomizing SERP order in prompt |

---

## Part 2: New Treatments to Explore

The current T1-T4 focus on content structure and credibility signals. Several other page features could causally affect LLM re-ranking.

### From the Page Content (extractable from cached HTML)

| Treatment | Type | What it captures | Extraction method |
|---|---|---|---|
| **T5** Direct answer in first paragraph | Binary | Does the page immediately answer the implied query? | LLM evaluation of first 200 words, or heuristic: does first `<p>` contain the keyword + a verb? |
| **T6** Comparison/vs content | Binary | Does the page compare products (e.g., "X vs Y")? | Regex for "vs", "versus", "compared to", "alternative" in headings |
| **T7** Pricing transparency | Binary | Does the page show pricing info? | Regex for "$", "pricing", "per month", "free trial", "/mo" in body text |
| **T8** Social proof signals | Count | Testimonials, customer counts, logos | Regex for "customers", "trusted by", "companies use", review widgets |
| **T9** Content recency signals | Binary | Does the page mention the current year? | Regex for "2026", "2025" in body text |
| **T10** List/step structure | Count | Number of ordered/unordered lists | Count `<ol>`, `<ul>` elements |
| **T11** Video embed | Binary | Does the page embed video? | Detect `<video>`, `<iframe>` with youtube/vimeo |

### From the SERP Snippet (what the LLM actually reads)

These are potentially more powerful because they affect the LLM's input directly:

| Treatment | Type | What it captures | Extraction |
|---|---|---|---|
| **T12** Snippet contains a number/stat | Binary | Does the snippet include a specific claim? | Regex on snippet text from JSON |
| **T13** Snippet tone (promotional vs informational) | Categorical | Sales language vs neutral description | LLM classification of snippet |
| **T14** Title contains brand name | Binary | "HubSpot CRM" vs "Best CRM Software" | Regex: does title contain the domain's brand? |
| **T15** Snippet contains call-to-action | Binary | "Try free", "Get started", "Sign up" | Regex on snippet text |

**Key insight**: T12-T15 operate on what the LLM actually sees (title + snippet), not on the full page. If the LLM's re-ranking is driven primarily by the snippet, these snippet-level treatments may show stronger effects than page-level treatments like T1-T4.

---

## Part 3: Scaling the Experiment

### 3A. More Keywords (50 → 500+)

**Why**: 50 keywords gives ~400 observations. This is enough to detect medium effects (rank shift > 0.5 positions) but not small ones. With 500 keywords we'd have ~4,000 observations, enough to detect subtle effects and run subgroup analyses.

**How**:
1. Expand keyword list across B2B SaaS subcategories (CRM, ERP, HR, marketing, analytics, security, DevOps, etc.)
2. Include long-tail variants ("best CRM for small business", "CRM software pricing comparison")
3. Include non-English keywords (German, French) to test language effects
4. Use SEO keyword tools (Ahrefs, SEMrush) to get search volume and competition metrics per keyword

**Cost**: Mainly compute time. SearXNG is free. HF Inference API has rate limits but 500 keywords × 1 LLM call each is feasible in a few hours.

**Implementation**:
```bash
# Generate expanded keyword list
python scripts/expand_keywords.py --categories b2b_saas --count 500

# Run with batching to respect rate limits
python run_ai_search.py --keywords-file keywords_expanded.txt --batch-size 10 --delay 5
```

### 3B. Multiple LLMs

**Why**: Our findings are about Llama-3.3-70B specifically. Do GPT-4, Claude, Qwen, Mistral, Gemini re-rank differently? This is the most important robustness check.

**How**:

| LLM | API | Cost estimate (500 keywords) |
|---|---|---|
| Llama-3.3-70B | HuggingFace (current) | Free tier / ~$2 |
| Qwen2.5-72B | HuggingFace (already wired) | Free tier / ~$2 |
| GPT-4o | OpenAI API | ~$5-10 |
| Claude 3.5 Sonnet | Anthropic API | ~$5-10 |
| Gemini 1.5 Pro | Google API | ~$5-10 |
| Mistral Large | Mistral API | ~$5-10 |

**Implementation**: Modify `src/llm_ranker.py` to accept a `provider` parameter (huggingface, openai, anthropic, google, mistral). The re-ranking prompt stays identical — only the model changes.

**Analysis**: Run the same DML models per LLM, then compare treatment effects across LLMs. This answers: "Is question-heading promotion a Llama-specific behavior, or do all LLMs do it?"

### 3C. Multiple Search Engines Separately

**Why**: SearXNG aggregates multiple engines, which is good for robustness but bad for understanding which engine's rankings drive what. Separating them lets us test whether the LLM corrects Google differently than Bing.

**How**:
1. Query each engine separately (Google via SerpAPI/ScraperAPI, Bing via Bing Web Search API, DuckDuckGo via `ddgs`)
2. Run the same LLM re-ranking on each engine's SERP independently
3. Compare `rank_delta` distributions across engines

**Cost**: API costs for Google/Bing ($50-100 for 500 queries each). DuckDuckGo is free.

### 3D. Temporal Replication

**Why**: Our data is a single snapshot (2026-02-16). SERPs change daily. Repeating the experiment weekly for 4-8 weeks lets us:
1. Estimate within-keyword variance (how stable are rankings?)
2. Test whether treatment effects are stable over time
3. Use panel data methods (fixed effects) for stronger identification

**How**: Cron job running the full pipeline weekly:
```bash
# Weekly cron (Sunday 10:00 UTC)
0 10 * * 0 cd /path/to/GEODML && bash run_weekly.sh
```

**Analysis**: Panel DML with keyword fixed effects. This absorbs all time-invariant keyword-level confounding.

### 3E. Prompt Variation (Position Bias Test)

**Why**: The LLM sees results in SERP order. It may have position bias (attending more to results listed first). This confounds the treatment effect if high-ranked SERP results also have different treatment values.

**How**:
1. For each keyword, run re-ranking 3 times:
   - Original SERP order
   - Reversed order
   - Random shuffle
2. Compare `post_rank` across the three orderings
3. If `post_rank` is stable across orderings → no position bias
4. If not → add SERP position as a confounder or use the shuffled version as the primary specification

**Cost**: 3× the LLM API calls. ~$6-30 depending on model.

### 3F. Beyond B2B SaaS

**Why**: Our findings are specific to B2B SaaS keywords. Do they generalize to:
- B2C products (electronics, fashion, food)?
- Informational queries ("how to fix a leaky faucet")?
- Medical/health queries (YMYL — Your Money Your Life)?
- Local queries ("plumber near me")?

**How**: Create keyword lists for 4-5 verticals, run the full pipeline on each, compare treatment effects across verticals.

**Expected insight**: The LLM's preference for question headings (T2) may be strongest for informational queries and weakest for transactional ones.

---

## Part 4: Implementation Roadmap

### Phase 1: Free Improvements (no new data collection)

Everything below uses existing cached HTML and JSON files.

| Task | Effort | Expected impact on R² |
|---|---|---|
| Extract S1/S2 (title/snippet-keyword similarity) from JSON | 2-3 hours | High — likely explains most `post_rank` variance |
| Extract S5 (brand recognition) via LLM batch eval | 1-2 hours | High — LLMs have strong brand priors |
| Extract B3, B4 (keyword-in-URL, URL depth) from URLs | 30 min | Low-moderate |
| Extract B5 (content freshness) from cached HTML | 1 hour | Moderate |
| Extract B10 (BM25 content-keyword relevance) from cached HTML | 2 hours | High — fundamental relevance signal |
| Extract T5-T11 (new page-level treatments) from cached HTML | 3-4 hours | Moderate — new causal questions |
| Extract T12-T15 (snippet-level treatments) from JSON | 1-2 hours | Potentially high — directly in LLM's input |
| Re-run DML with expanded confounders | 1 hour | Should substantially improve nuisance R² |

**Total**: ~12-15 hours of work, no API costs, no new data needed.

**Expected outcome**: Adding S1, S2, S5, and B10 as confounders should push nuisance R² from negative into the 0.1-0.3 range for `post_rank` models. This would mean confounders explain 10-30% of ranking variance instead of 0%.

### Phase 2: Backlink Data ($100-200 budget)

| Task | Effort | Cost |
|---|---|---|
| Get Ahrefs/Moz backlink data for ~400 domains | 2 hours | $99/mo subscription |
| Extract referring domains, domain rating, backlink count | 1 hour | Included |
| Re-run DML with backlink confounders | 1 hour | — |

**Expected outcome**: Backlink data should push `pre_rank` nuisance R² from negative into 0.3-0.5 range. This is the single biggest confounder gap.

### Phase 3: Scale Up (1-2 week project)

| Task | Effort | Cost |
|---|---|---|
| Expand to 500 keywords | 1 day | ~$5-20 in API calls |
| Add 3-4 more LLMs | 1 day | ~$20-40 in API calls |
| Position bias test (3 orderings) | 1 day | ~$15-60 in API calls |
| Re-run full DML battery | 1 day | Compute only |
| Write up comparative results | 2-3 days | — |

### Phase 4: Full Study (1-2 month project)

| Task | Effort | Cost |
|---|---|---|
| 4-8 weekly replications | 8 weeks (automated) | ~$50-100 total |
| Multiple verticals (4-5 categories) | 1 week | ~$50-100 |
| Panel DML with keyword fixed effects | 3-4 days | — |
| Separate search engines (Google, Bing, DDG) | 1 week | ~$100-200 |
| Full paper write-up | 2-3 weeks | — |

---

## Part 5: Analysis Upgrades

### 5A. Keyword Fixed Effects

Currently, keyword-level variation (some keywords have inherently higher rank_delta because the SERP and LLM disagree more) is uncontrolled noise. Adding keyword fixed effects (or keyword dummies) absorbs this and isolates within-keyword treatment effects.

**Implementation**: Add `pd.get_dummies(df['keyword'])` to confounders. With 50 keywords and ~400 observations this is tight on degrees of freedom, but feasible with regularized nuisance learners (LGBM/RF handle high-dimensional X well).

**Alternative**: Use keyword-level random effects or cluster-robust standard errors at the keyword level.

### 5B. Domain Clustering

Some domains appear across multiple keywords (e.g., hubspot.com appears for "CRM software", "marketing automation", "sales tools"). Treatment values are constant within a domain, so observations are not independent.

**Fix**: Cluster standard errors at the domain level. DoubleML supports this via `DoubleMLClusterData`.

### 5C. Causal Forest (Heterogeneous Effects)

DML gives an average treatment effect. But the effect may vary: question headings may matter more for informational keywords than transactional ones, or more for low-authority domains than high-authority ones.

**Implementation**: Use `econml.dml.CausalForestDML` to estimate conditional average treatment effects (CATE). This answers: "For which types of pages/keywords does T2 matter most?"

### 5D. Sensitivity Analysis for Unobserved Confounding

Even with better confounders, there may be unobserved confounders we miss. Use the Cinelli & Hazlett (2020) partial R² framework or Oster (2019) coefficient stability test to bound how much an unobserved confounder would need to explain to overturn the findings.

---

## Summary: Priority Order

1. **S1/S2 (title/snippet similarity)** — free, likely biggest R² improvement, directly addresses what the LLM sees
2. **B10 (content-keyword BM25)** — free, addresses `pre_rank` variance
3. **S5 (brand recognition)** — free, addresses LLM brand bias
4. **Keyword fixed effects** — free, absorbs keyword-level noise
5. **Domain clustering** — free, corrects standard errors
6. **B1/B2 (backlinks)** — paid, but likely explains most remaining `pre_rank` variance
7. **Position bias test** — cheap, validates experimental design
8. **Multiple LLMs** — cheap, validates generalizability
9. **Scale to 500 keywords** — cheap, increases power for all analyses
10. **Temporal replication** — cheap per run, requires patience
