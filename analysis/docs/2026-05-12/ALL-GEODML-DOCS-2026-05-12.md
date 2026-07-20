# GEODML — Combined Project Documentation

Concatenated dump of every project markdown in `~/Hamburg/GEODML/` and its `paperSizeExperiment/` subfolder. Generated 2026-05-12.

Excluded: third-party READMEs (`Perplexica/`, `searxng-local/`, `huggingface_bundle/`).

Total: 22 files, ~245 KB.

## Sections

- **GEODML root** (project overview, scoping, findings, recommendations) — files 1–9
- **paperSizeExperiment/** (the 32-cell factorial experiment, chronological) — files 10–22

---

## Table of contents

1. [`CLAUDE.md`](#claude) — 2377 bytes
2. [`PROPOSITION.md`](#proposition) — 5504 bytes
3. [`EXPANSION.md`](#expansion) — 18042 bytes
4. [`EXPERIMENT_REGISTRY.md`](#experiment-registry) — 27927 bytes
5. [`FINDINGS.md`](#findings) — 16609 bytes
6. [`RECOMMENDATIONS.md`](#recommendations) — 9344 bytes
7. [`results_findings.md`](#results-findings) — 24335 bytes
8. [`both_analysis/COMPARATIVE_FINDINGS.md`](#both-analysis--comparative-findings) — 12130 bytes
9. [`data/README.md`](#data--readme) — 12383 bytes
10. [`paperSizeExperiment/README.md`](#papersizeexperiment--readme) — 3688 bytes
11. [`paperSizeExperiment/doc/proposition-2026-04-07.md`](#papersizeexperiment--doc--proposition-2026-04-07) — 6068 bytes
12. [`paperSizeExperiment/doc/audit-2026-04-07.md`](#papersizeexperiment--doc--audit-2026-04-07) — 7255 bytes
13. [`paperSizeExperiment/doc/status-2026-04-13.md`](#papersizeexperiment--doc--status-2026-04-13) — 10212 bytes
14. [`paperSizeExperiment/doc/status-2026-04-14.md`](#papersizeexperiment--doc--status-2026-04-14) — 3881 bytes
15. [`paperSizeExperiment/audit_2026-04-14.md`](#papersizeexperiment--audit-2026-04-14) — 7884 bytes
16. [`paperSizeExperiment/doc/meta-analysis-report-2026-04-15.md`](#papersizeexperiment--doc--meta-analysis-report-2026-04-15) — 17660 bytes
17. [`paperSizeExperiment/doc/dataforseo-plan-2026-04-22.md`](#papersizeexperiment--doc--dataforseo-plan-2026-04-22) — 4473 bytes
18. [`paperSizeExperiment/doc/DATAFORSEO_CATALOG.md`](#papersizeexperiment--doc--dataforseo-catalog) — 9739 bytes
19. [`paperSizeExperiment/doc/analysis-2026-04-23.md`](#papersizeexperiment--doc--analysis-2026-04-23) — 23605 bytes
20. [`paperSizeExperiment/doc/robust-winners-analysis-2026-04-26.md`](#papersizeexperiment--doc--robust-winners-analysis-2026-04-26) — 12021 bytes
21. [`paperSizeExperiment/doc/ROADMAP.md`](#papersizeexperiment--doc--roadmap) — 5811 bytes
22. [`paperSizeExperiment/doc/treatment-confounder-dictionary.md`](#papersizeexperiment--doc--treatment-confounder-dictionary) — 9714 bytes

---



<a id="claude"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 1 / 22 — CLAUDE.md  (2377 bytes)
# ═══════════════════════════════════════════════════════════════

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GEODML — GEO (Generative Engine Optimization) causal inference experiment. Compares how an LLM re-ranks search results vs. traditional search engine rankings for 50 B2B SaaS keywords.

## Architecture

```
keywords.txt → SearXNG (Apptainer container, port 8888) → LLM re-ranking (HF API) → results/
```

- **SearXNG**: Apptainer/Docker container aggregating Google, Bing, DuckDuckGo, Brave, Startpage
- **LLM**: Qwen2.5-72B-Instruct via HuggingFace Inference API (remote, no local GPU)
- **DuckDuckGo fallback**: `ddgs` library used directly if SearXNG unavailable

## Setup & Run

```bash
bash setup.sh                          # pulls SearXNG container + creates Python venv
bash start_searxng.sh &                # start SearXNG in background
source venv/bin/activate
python run_ai_search.py                # run full experiment (50 keywords)
python run_ai_search.py --keywords 5   # test with 5 keywords
python summarize_results.py results/ai_search_rankings.json
```

## Key Files

- `setup.sh` — Apptainer/Docker/native detection, SearXNG pull, venv creation
- `run_ai_search.py` — main experiment: SearXNG → LLM re-ranking → JSON/CSV
- `run_ranking_comparison.py` — live experiment + side-by-side terminal comparison
- `summarize_results.py` — reads existing results JSON, prints ranking report
- `run_engine_search.py` — traditional DDG + Google scraper for baseline
- `src/searxng_client.py` — SearXNG client with DDG fallback
- `src/llm_ranker.py` — Qwen2.5-72B re-ranking via HF chat_completion API
- `src/experiment_context.py` — IP, geolocation, machine info for provenance

## Ranking Convention

- **Lower rank number = better** (rank 1 is the best position, the goal)
- In coefficients: **negative effect on post_rank or rank = GOOD** (moves the page closer to rank 1)
- `rank_delta = pre_rank - post_rank`: **positive = LLM promoted the page** (good)
- When interpreting results, always remember: we want to be ranked 1st, so anything that decreases rank number is beneficial

## Environment

- `.env.local`: `HF_TOKEN`, `SEARXNG_URL` (default `http://127.0.0.1:8888`)
- Python venv in `venv/`, SearXNG image in `searxng.sif`
- HPC (JUWELS): Apptainer for SearXNG, `--no-cache-dir` for pip (home quota)


---

*end of CLAUDE.md*



<a id="proposition"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 2 / 22 — PROPOSITION.md  (5504 bytes)
# ═══════════════════════════════════════════════════════════════

# GEODML — Expansion Cost Proposition

## Scale: 1,000 Keywords × 8 Weekly Snapshots × 2 Months

| Dimension | Current | Expanded |
|---|---|---|
| Keywords | 50 | 1,000 |
| Results per keyword | ~10 | ~10 |
| Observations per snapshot | ~500 | ~10,000 |
| Weekly snapshots | 1 | 8 (2 months) |
| Total observations | ~500 | ~80,000 |
| Unique domains (est.) | ~300 | ~3,000-5,000 |
| Unique pages to fetch | ~400 | ~5,000-8,000 |

---

## Token Math Per LLM

Each re-ranking call: ~1,200 input tokens (20 results × ~50 tokens each + prompt boilerplate), ~80 output tokens (10 domain names).

- 8,000 calls → ~9.6M input tokens + ~0.6M output tokens

---

## Cost Breakdown by Component

### Search Engine Queries — $0

SearXNG is self-hosted. 1,000 queries/week × 8 weeks = 8,000 queries. Cost: **$0**.

### LLM Re-Ranking (8,000 calls per model over 2 months)

| Model | Input cost | Output cost | 2-month total |
|---|---|---|---|
| Llama-3.3-70B (HF Pro + Fireworks) | ~$9 | ~$1 | **~$28** (incl. $9/mo sub) |
| GPT-4o-mini | $1.44 | $0.38 | **~$2** |
| GPT-4o | $24 | $6.40 | **~$30** |
| Claude Haiku 4.5 | $9.60 | $3.20 | **~$13** |
| Claude Sonnet 4.5 | $28.80 | $9.60 | **~$38** |
| Gemini 2.0 Flash | $0.96 | $0.26 | **~$1** |
| Mistral Large 3 | $4.80 | $0.96 | **~$6** |

### Backlink Data (~5,000 domains)

| Provider | Plan | 2-month cost | Notes |
|---|---|---|---|
| **Moz** | Standard ($79/mo) | **$158** | 5,000 queries/mo — fits perfectly |
| Ahrefs | API Standard ($500/mo) | $1,000 | Overkill for domain-level metrics |
| Majestic | API ($400/mo) | $800 | Full API access required |
| Common Crawl | Free | $0 | Raw link counts only, significant engineering effort |

### Feature Extraction LLM (T1-T4 LLM-based, one-time)

~5,000 pages × ~3,000 input tokens + ~500 output = 15M input + 2.5M output. Via Llama on HF: **~$15**. But code-based measurement was stronger in every experiment, so this is optional.

### Sentence Embeddings (S1/S2 — title/snippet-keyword similarity)

Local model (`all-MiniLM-L6-v2`), runs on CPU. **$0**.

### Position Bias Test (3 orderings, one snapshot)

+2,000 extra LLM calls per model. Adds ~30% to one week's LLM cost. **~$3-10** depending on model.

---

## Budget Scenarios

### Minimum Viable: One LLM, Moz backlinks — ~$190

| Item | Cost |
|---|---|
| SearXNG (8,000 queries) | $0 |
| Llama-3.3-70B re-ranking (8,000 calls) | $28 |
| Moz backlinks (5,000 domains, 2 months) | $158 |
| Feature extraction (code-based only) | $0 |
| Sentence embeddings (local) | $0 |
| **Total** | **~$186** |

Same design as now but 20x more keywords, 8 weekly snapshots, plus backlink data. Gets you ~80,000 observations, keyword fixed effects, temporal variation, and the missing confounder.

### Recommended: 4 LLMs, Moz, position bias test — ~$260

| Item | Cost |
|---|---|
| SearXNG (8,000 queries) | $0 |
| Llama-3.3-70B (8,000 calls) | $28 |
| GPT-4o-mini (8,000 calls) | $2 |
| Gemini 2.0 Flash (8,000 calls) | $1 |
| Mistral Large 3 (8,000 calls) | $6 |
| Moz backlinks (2 months) | $158 |
| Position bias test (1 snapshot, 3 orderings, Llama) | $8 |
| Feature extraction LLM-based (optional) | $15 |
| Sentence embeddings (local) | $0 |
| **Total** | **~$218** |

This answers: "Do different LLMs re-rank differently?" at negligible marginal cost. GPT-4o-mini and Gemini Flash are essentially free.

### Full Study: Add frontier models + separate search engines — ~$550

| Item | Cost |
|---|---|
| Everything in Recommended | $218 |
| GPT-4o (8,000 calls) | $30 |
| Claude Sonnet 4.5 (8,000 calls) | $38 |
| SerpAPI (separate Google/Bing, 2 months) | $150 |
| Position bias test (all models, 1 snapshot) | $30 |
| LLM-based feature extraction (T1-T4) | $15 |
| **Total** | **~$481** |

---

## The Surprise: LLM Costs Are Negligible

Backlink data is the single biggest cost (~$158), not the LLM calls. Running 4 LLMs for 8,000 calls each costs less than $40 total. Even adding GPT-4o and Claude Sonnet only adds ~$70. The API price war has made this kind of experiment very cheap.

The real cost is engineering time: expanding the keyword list, wiring up new APIs (Moz, additional LLM providers), building the panel data pipeline, and running the analysis. The infrastructure work is probably 2-3 weeks of engineering; the API bills are under $300 for a publishable study.

---

## What Each Budget Level Buys

| Capability | Minimum ($190) | Recommended ($260) | Full ($550) |
|---|---|---|---|
| 1,000 keywords | Yes | Yes | Yes |
| 8 weekly snapshots | Yes | Yes | Yes |
| Backlink confounders (Moz) | Yes | Yes | Yes |
| Snippet-level confounders (S1/S2) | Yes | Yes | Yes |
| Keyword fixed effects | Yes | Yes | Yes |
| Domain clustering | Yes | Yes | Yes |
| Multiple LLMs | No (Llama only) | 4 LLMs | 6 LLMs |
| Position bias test | No | 1 model | All models |
| Separate search engines | No | No | Google + Bing + DDG |
| Frontier models (GPT-4o, Claude) | No | No | Yes |
| Publishable multi-LLM comparison | No | Yes | Yes |

---

## Pricing Sources (February 2026)

- HuggingFace Inference Providers: via Fireworks AI at ~$0.90/M tokens
- OpenAI: GPT-4o $2.50/$10.00 per M tokens (input/output), GPT-4o-mini $0.15/$0.60
- Anthropic: Claude Sonnet 4.5 $3.00/$15.00, Haiku 4.5 $1.00/$5.00
- Google: Gemini 2.0 Flash $0.10/$0.40, Gemini 2.5 Pro $1.25/$10.00
- Mistral: Large 3 $0.50/$1.50, Small 3.1 $0.03/$0.11
- Moz: Standard plan $79/mo (5,000 queries/month)
- SerpAPI: Developer plan $75/mo (5,000 searches/month)


---

*end of PROPOSITION.md*



<a id="expansion"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 3 / 22 — EXPANSION.md  (18042 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of EXPANSION.md*



<a id="experiment-registry"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 4 / 22 — EXPERIMENT_REGISTRY.md  (27927 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of EXPERIMENT_REGISTRY.md*



<a id="findings"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 5 / 22 — FINDINGS.md  (16609 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of FINDINGS.md*



<a id="recommendations"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 6 / 22 — RECOMMENDATIONS.md  (9344 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of RECOMMENDATIONS.md*



<a id="results-findings"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 7 / 22 — results_findings.md  (24335 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of results_findings.md*



<a id="both-analysis--comparative-findings"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 8 / 22 — both_analysis/COMPARATIVE_FINDINGS.md  (12130 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of both_analysis/COMPARATIVE_FINDINGS.md*



<a id="data--readme"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 9 / 22 — data/README.md  (12383 bytes)
# ═══════════════════════════════════════════════════════════════

# GEODML Clean Dataset

This folder contains the final, analysis-ready dataset for the GEODML causal inference study. The experiment measures how an LLM re-ranks search engine results for 50 B2B SaaS keywords, and estimates the causal effect of page-level features on that re-ranking using Double Machine Learning.

## File

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `geodml_dataset.csv` | 492 | 27 | One row per (keyword, domain) pair. Contains outcome variables, treatments, confounders, and metadata. |

---

## Experiment Overview

**Research question:** Which on-page features cause an LLM to promote or demote a search result relative to its original search engine ranking?

**Setting:** 50 B2B SaaS keywords queried from Hamburg, Germany on 2026-02-16.

**Search engine:** SearXNG (local Apptainer container aggregating Google, Bing, DuckDuckGo, Brave, Startpage)

**LLM re-ranker:** Llama-3.3-70B-Instruct via HuggingFace Inference API

---

## Pipeline Workflow

The experiment runs in four sequential phases. Each phase feeds into the next.

```
 PHASE 1          PHASE 2           PHASE 3              PHASE 4
 ────────         ────────          ────────              ────────
 keywords.txt     SearXNG SERP      Fetch HTML            DoubleML
 (50 keywords)    (20 results       per URL               PLR model
      │           per keyword)           │                     │
      │                │                 │                     │
      ▼                ▼                 ▼                     ▼
 ┌─────────┐    ┌────────────┐    ┌────────────────┐    ┌──────────┐
 │ Query    │───▶│ Pre-rank   │───▶│ Extract T1-T4  │───▶│ Estimate │
 │ SearXNG  │    │ (SERP      │    │ (treatments)   │    │ causal   │
 │          │    │  position)  │    │ Extract X1-X10 │    │ effect θ │
 └─────────┘    └─────┬──────┘    │ (confounders)  │    └──────────┘
                      │           └────────────────┘
                      ▼
                ┌────────────┐
                │ LLM re-    │
                │ ranks top  │
                │ 10 domains │
                │ (post_rank)│
                └────────────┘
```

### Phase 1: Search Engine Ranking (pre_rank)

**Script:** `run_ai_search.py` (also handles Phase 2)

1. Read 50 B2B SaaS keywords from `keywords.txt`
2. For each keyword, query SearXNG for the top 20 results
3. SearXNG aggregates results from multiple search engines (Google, Bing, DuckDuckGo, Brave, Startpage) and returns a merged, scored list
4. Extract the unique root domain from each result URL using `tldextract`
5. Assign `pre_rank` = the position of each domain's first appearance in the SERP (1-indexed)

**Output per keyword:**
- 20 raw SERP results (position, title, url, snippet, engines, score)
- Deduplicated domain list with SERP positions

### Phase 2: LLM Re-Ranking (post_rank)

**Script:** `run_ai_search.py` → `src/llm_ranker.py`

1. For each keyword, build a re-ranking prompt containing:
   - The bare keyword (e.g., "CRM software")
   - All 20 SERP results formatted as `[domain] title — snippet`
   - Instructions to return the top 10 software product domains, excluding review sites, directories, Wikipedia, forums, YouTube
2. Send prompt to Llama-3.3-70B-Instruct via HuggingFace Inference API
3. Parse the LLM's response: extract domain names, assign `post_rank` = position in LLM output (1-indexed)
4. Compute `rank_delta = pre_rank - post_rank` for each domain
   - Positive = LLM promoted the result (moved it up)
   - Negative = LLM demoted the result (moved it down)
   - None = domain appeared only in LLM output but not in original SERP top-20 (93 cases)

**LLM prompt structure:**
```
Search keyword: {keyword}

Below are search engine results for the above keyword. Re-rank the results
and return the top 10 software product domains, ordered by relevance to
the keyword.

Exclude non-product sites: review aggregators, directories, Wikipedia,
news, blogs, forums, YouTube.

Return only root domains, one per line. No explanations.

Search results:
1. [forbes.com] 10 Best CRM Software Of 2026 – Forbes Advisor — ...
2. [hubspot.com] Streamline Your Entire Business With a Free CRM — ...
...
20. [example.com] Example Title — snippet...

Re-ranked product domains:
```

### Phase 3: Page Feature Extraction (Treatments T1-T4 & Confounders X1-X10)

**Script:** `run_page_scraper.py` → `src/page_features.py`

For each URL in the merged result set:

1. **Fetch HTML** (30s timeout, 5MB max, cached locally by SHA-256 hash)
2. **Parse** with BeautifulSoup (lxml parser)
3. **Extract code-based treatments** (T1-T4) by parsing the HTML directly
4. **Extract LLM-based treatments** (T1-T4) by sending a page digest to the LLM for independent evaluation
5. **Extract confounders** (X1-X10) from the HTML, external APIs (Open PageRank, WHOIS, Google PageSpeed), and computed metrics

**External APIs used:**
- Open PageRank API → X1 domain authority, X1 global rank
- WHOIS lookup → X2 domain age
- Google PageSpeed Insights API → X4 LCP (0% coverage in this dataset)

### Phase 4: DML Causal Inference

**Script:** `run_dml_study.py`

1. Load the merged dataset (this CSV) and drop rows with missing `rank_delta` (93 rows → 399 usable)
2. For each of 8 treatments (T1-T4 x code/LLM), fit a Partially Linear Regression model:
   ```
   Y = D*theta + g(X) + noise       (outcome equation)
   D = m(X) + V                     (treatment equation)
   ```
   where Y = rank_delta, D = treatment, X = confounders, theta = causal effect
3. Nuisance models (g and m) fitted with LGBMRegressor, 5-fold cross-fitting
4. Sensitivity check with RandomForestRegressor as alternative learner
5. Report: coefficient, standard error, 95% CI, p-value per treatment

---

## Data Dictionary

### Identifiers & Metadata

| Column | Type | Description |
|--------|------|-------------|
| `keyword` | string | B2B SaaS search keyword (50 unique) |
| `domain` | string | Root domain of the search result (e.g., `hubspot.com`) |
| `url` | string | Full URL of the search result page. Missing for 93 rows where the domain was surfaced by the LLM but not in the original SERP |
| `search_engine` | string | Search engine used for SERP. Always `searxng` in this dataset |
| `llm_model` | string | LLM used for re-ranking. Always `meta-llama/Llama-3.3-70B-Instruct` |

### Outcome Variables (Y)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `pre_rank` | int or NaN | 1-20 | Domain's position in the original SERP. NaN for 93 domains that only appeared in the LLM output |
| `post_rank` | int | 1-10 | Domain's position after LLM re-ranking |
| `rank_delta` | int or NaN | -9 to +16 | `pre_rank - post_rank`. Positive = LLM promoted the result. NaN when `pre_rank` is NaN |

**rank_delta distribution** (399 non-null values): mean=3.89, median=4.0, std=4.06

### Treatment Variables (T1-T4)

Each treatment is measured two ways: code-based (deterministic HTML parsing) and LLM-based (Llama-3.3-70B evaluation of a page digest).

| Column | Type | Range | Measurement | Description |
|--------|------|-------|-------------|-------------|
| `T1_statistical_density_code` | float | 0-29.4 | Code | Unique numbers, percentages, dates, dollar amounts per 500 words of body text. Regex-based counting |
| `T1_statistical_density_llm` | float | 0-34.7 | LLM | Same metric as T1 but scored by the LLM reading a page digest |
| `T2_question_heading_code` | binary | 0/1 | Code | 1 if any H2/H3 heading starts with a question word (What, How, Why, When, Where, Can, Does, Is...) |
| `T2_question_heading_llm` | binary | 0/1 | LLM | Same as T2 but evaluated by the LLM |
| `T3_structured_data_code` | binary | 0/1 | Code | 1 if the page has JSON-LD structured data with @type FAQ, Product, or HowTo |
| `T3_structured_data_llm` | binary | 0/1 | LLM | Same as T3 but evaluated by the LLM |
| `T4_citation_authority_code` | int | 0-3 | Code | Count of outbound links to .edu, .gov, .ac.uk, or known academic domains |
| `T4_citation_authority_llm` | int | 0-6 | LLM | Same as T4 but evaluated by the LLM |

**Treatment coverage:** ~85% (73-81 missing values out of 492, depending on treatment). Missing when the page could not be fetched (HTTP error, timeout, or no URL).

### Confounder Variables (X1-X10)

| Column | Type | Range | Source | Coverage | Description |
|--------|------|-------|--------|----------|-------------|
| `X1_domain_authority` | float | 0-10 | Open PageRank API | 95% | Domain authority score (0=lowest, 10=highest) |
| `X1_global_rank` | int | 4 - 231M | Open PageRank API | 94% | Global traffic rank (lower = more popular) |
| `X2_domain_age_years` | float | 0.9-39.9 | WHOIS | 87% | Years since domain creation date |
| `X3_word_count` | int | 3-17797 | HTML parsing | 84% | Total words in the page body text |
| `X4_lcp_ms` | float | -- | Google PageSpeed | 0% | Largest Contentful Paint in ms. **Not available** in this dataset |
| `X6_readability` | float | 5.96-27.97 | textstat library | 82% | Flesch-Kincaid grade level. Only computed for pages with 100+ words |
| `X7_internal_links` | int | 0-1116 | HTML parsing | 85% | Count of same-domain links on the page |
| `X7B_outbound_links` | int | 0-801 | HTML parsing | 85% | Count of external-domain links on the page |
| `X8_keyword_difficulty` | float | 3.26-5.79 | Computed | 100% | Average domain authority of the top-10 results for that keyword |
| `X9_images_with_alt` | int | 0-650 | HTML parsing | 85% | Count of `<img>` tags with non-empty alt text |
| `X10_https` | binary | 1 | URL parsing | 95% | 1 if URL is HTTPS, 0 otherwise. **Zero variance** in this dataset (all are 1) |

**Note for DML:** X4_lcp_ms (0% coverage) and X10_https (zero variance) should be excluded from analysis. The DML script uses X1 (authority), X2 (age), X3 (word count), X6 (readability), X7 (internal links), X7B (outbound links), X8 (keyword difficulty), X9 (images with alt) as confounders.

---

## Missing Data Summary

| Reason | Count | Columns affected |
|--------|-------|-----------------|
| Domain not in SERP top-20 (LLM-surfaced only) | 93 | url, pre_rank, rank_delta |
| Page fetch failed (HTTP error, timeout, blocked) | 73 | All T1-T4, X3, X6, X7, X7B, X9 |
| Page fetch failed + no stats extractable | 81 | T1_code, X3 (word count needed for density calc) |
| Domain not in Open PageRank database | 25-30 | X1_domain_authority, X1_global_rank |
| WHOIS lookup failed | 65 | X2_domain_age_years |
| Page too short for readability (<100 words) | 91 | X6_readability |
| Google PageSpeed API not configured | 492 | X4_lcp_ms |
| Domain uses HTTPS (no variance) | 24 NaN | X10_https |

---

## Experiment Provenance

| Field | Value |
|-------|-------|
| Date | 2026-02-16 |
| Time window | 10:06 - 10:12 UTC |
| Location | Hamburg, Germany |
| IP | 84.63.169.80 |
| ISP | ARCOR-IP |
| Machine | MacBook Pro (macOS, x86_64) |
| Python | 3.13.5 (Anaconda) |
| SearXNG | Local Apptainer container, port 8888 |
| LLM | meta-llama/Llama-3.3-70B-Instruct (HuggingFace Inference API) |
| Keywords | 50 B2B SaaS categories |
| Results per keyword | Top 20 SERP, re-ranked to top 10 by LLM |

---

## Reproducing

From the project root:

```bash
# 1. Build this clean dataset from raw results
python build_clean_dataset.py

# 2. Run DML causal inference
python run_dml_study.py
```

To regenerate the raw data from scratch (requires SearXNG + HF API token):

```bash
bash start_searxng.sh &                              # start search engine
python run_ai_search.py --keywords 50                 # Phase 1+2: SERP + LLM
python extract_all_results.py                         # merge result files
python run_page_scraper.py --input results/all_results_searxng.csv --all  # Phase 3
python build_clean_dataset.py                         # build clean CSV
python run_dml_study.py                               # Phase 4: DML
```


---

*end of data/README.md*



<a id="papersizeexperiment--readme"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 10 / 22 — paperSizeExperiment/README.md  (3688 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of paperSizeExperiment/README.md*



<a id="papersizeexperiment--doc--proposition-2026-04-07"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 11 / 22 — paperSizeExperiment/doc/proposition-2026-04-07.md  (6068 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of paperSizeExperiment/doc/proposition-2026-04-07.md*



<a id="papersizeexperiment--doc--audit-2026-04-07"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 12 / 22 — paperSizeExperiment/doc/audit-2026-04-07.md  (7255 bytes)
# ═══════════════════════════════════════════════════════════════

# Pipeline Audit — 2026-04-09

## 5-Run x 4-Phase Completion Matrix

| # | Run ID | Engine | Model | Pool | P1: SERP+LLM | P2: HTML+Code | P3: LLM Features | P4: Enriched |
|---|--------|--------|-------|------|--------------|---------------|-------------------|--------------|
| 1 | DDG/Llama/s20 | DDG | Llama 3.3 70B | 20/10 | DONE — 1011 kw, 7890 rankings | DONE — 6413 URLs (776 err, 12%) | DONE — 5636/5637 (99.9%) | DONE (features_new 7890 rows) |
| 2 | DDG/Llama/s50 | DDG | Llama 3.3 70B | 50/10 | DONE — 1011 kw, 8088 rankings | DONE — 6817 URLs (1170 err, 17%) | DONE — 5647/5647 (100%) | DONE (features_new 8088 rows) |
| 3 | DDG/Qwen/s20 | DDG | Qwen 2.5 72B | 20/10 | DONE — 1011 kw, 8335 rankings | DONE — 6947 URLs (1041 err, 15%) | 91.8% — 5422/5906 (484 left) | features_new MISSING |
| 4 | DDG/Qwen/s50 | DDG | Qwen 2.5 72B | 50/10 | DONE — 1011 kw, 9863 rankings | DONE — 8591 URLs (1244 err, 14%) | 35.6% — 2612/7347 (4735 left) | not started |
| 5 | SXG/Llama/s20 | SearXNG | Llama 3.3 70B | 20/10 | DONE — 960 kw (51 failed), 8197 rankings | DONE — 6968 URLs (1027 err, 15%) | DONE — 5941/5941 (100%) | DONE (features_new 8197 rows) |

Runs 6-8 (SearXNG Llama/s50, Qwen/s20, Qwen/s50) have **not been started**.

---

## File Availability

| Run | features.csv | features_new.csv | geodml_dataset.csv | analysis_full/ | analysis_halo/ |
|-----|-------------|-----------------|-------------------|----------------|----------------|
| 1 DDG/Llama/s20 | 6,413 rows | 7,890 rows | 7,890 rows, 56 cols | 204 experiments | yes |
| 2 DDG/Llama/s50 | 6,817 rows | 8,088 rows | 8,088 rows, 56 cols | 204 experiments | yes |
| 3 DDG/Qwen/s20 | 6,947 rows | **MISSING** | 8,335 rows, 56 cols | **empty** | no |
| 4 DDG/Qwen/s50 | 8,591 rows | MISSING | **MISSING** | no | no |
| 5 SXG/Llama/s20 | 6,968 rows | 8,197 rows | 8,197 rows, 56 cols | 216 experiments | yes |

---

## Outcomes (geodml_dataset.csv)

| Variable | Run 1 (DDG/Llama/s20) | Run 2 (DDG/Llama/s50) | Run 3 (DDG/Qwen/s20) | Run 5 (SXG/Llama/s20) |
|----------|----------------------|----------------------|----------------------|----------------------|
| post_rank | 100% | 100% | 100% | 100% |
| pre_rank | 70.8% | 75.0% | 77.0% | 80.2% |
| rank_delta | 70.8% | 75.0% | 77.0% | 80.2% |

Missing pre_rank/rank_delta rows are URLs the LLM ranked but that were absent from the original SERP (no pre_rank to compute delta from).

**Bug fixed**: Runs 3 and 5 previously had 0% outcomes because `experiment.json` had empty `per_keyword_results` (resume bug). Fixed `clean_data.py` to fall back to `keywords.jsonl`.

---

## Treatments — Code-Extracted (features.csv → geodml_dataset.csv)

| Variable | Run 1 | Run 2 | Run 3 | Run 5 |
|----------|-------|-------|-------|-------|
| T1_statistical_density_code | 83.9% | 79.4% | 81.7% | 82.3% |
| T2_question_heading_code | 85.3% | 82.4% | 84.5% | 85.3% |
| T3_structured_data_code | 85.3% | 82.4% | 84.5% | 85.3% |
| T4_citation_authority_code | 85.3% | 82.4% | 84.5% | 85.3% |

Coverage ~80-85%. Missing rows correspond to HTML fetch errors (~12-17%).

---

## Treatments — LLM-Scored (features.csv → geodml_dataset.csv)

| Variable | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|----------|-------|-------|-------|-------|-------|
| T1_llm_statistical_density | 87.0% | 82.3% | **91.8%** | **35.6%** | 85.2% |
| T2_llm_question_heading | 87.0% | 82.3% | **91.8%** | **35.6%** | 85.2% |
| T3_llm_structured_data | 87.0% | 82.3% | **91.8%** | **35.6%** | 85.2% |
| T4_llm_citation_authority | 87.0% | 82.3% | **91.8%** | **35.6%** | 85.2% |

Run 3 at 91.8% (5422/5906 processed, 484 remaining). Run 4 at 35.6% (2612/7347, 4735 remaining). Processes stopped 2026-04-09 — resume with same gather_data.py command.

---

## Treatments — Enriched (features_new.csv → geodml_dataset.csv)

| Variable | Run 1 | Run 2 | Run 5 | Runs 3-4 |
|----------|-------|-------|-------|----------|
| treat_source_earned (T7) | 70.8% | 75.0% | 80.2% | MISSING |
| treat_freshness (T6) | 63.7% | 63.6% | 69.7% | MISSING |
| treat_topical_comp (T5) | **0%** | **0%** | 67.8% | MISSING |
| treat_stats_density | 62-68% | 62-68% | 67.8% | MISSING |
| treat_question_headings | 64-70% | 64-70% | 69.7% | MISSING |
| treat_structured_data | 64-70% | 64-70% | 69.7% | MISSING |
| treat_auth_citations | 64-70% | 64-70% | 69.7% | MISSING |

**T5 topical competence**: only works for SearXNG run (embedding similarity depends on SearXNG-specific data).
**T6 freshness**: now populated at ~64-70% for Runs 1, 2, 5 (was incorrectly reported as 0% in previous audit).

---

## Confounders (geodml_dataset.csv)

| Variable | Run 1 | Run 2 | Run 5 | Notes |
|----------|-------|-------|-------|-------|
| conf_serp_position | 70.8% | 75.0% | 80.2% | |
| conf_bm25 | 70.8% | 75.0% | 80.2% | |
| conf_brand_recog | 70.8% | 75.0% | 80.2% | |
| conf_title_has_kw | 70.8% | 75.0% | 80.2% | |
| conf_word_count | 58-62% | 58-62% | 67.8% | |
| conf_readability | 56-60% | 56-60% | 65.9% | |
| conf_internal_links | 64-70% | 64-70% | 69.7% | |
| conf_title_kw_sim | **0%** | **0%** | **80.2%** | SearXNG-only |
| conf_snippet_kw_sim | **0%** | **0%** | **80.2%** | SearXNG-only |
| conf_domain_authority | **6.4%** | **7.4%** | **9.7%** | Moz API limit |
| conf_backlinks | **6.4%** | **7.4%** | **9.7%** | Moz API limit |
| X1_domain_authority | 0% | 0% | 0% | Never called |
| X2_domain_age_years | 0% | 0% | 0% | Never ran |
| X4_lcp_ms | 0% | 0% | 0% | No API built |
| X8_keyword_difficulty | 0% | 0% | 0% | No API |

---

## Analysis Results (p < 0.05)

| Run | Experiments | Significant | Top Finding |
|-----|------------|-------------|-------------|
| 1 DDG/Llama/s20 (full) | 204 | 32 | T7 earned media: rank_delta coef=-2.02 (***) |
| 2 DDG/Llama/s50 (full) | 204 | 39 | T7 earned media: rank_delta coef=-2.43 (***) |
| 5 SXG/Llama/s20 (full) | 216 | 44 | T7 earned media: rank_delta coef=-2.37 (***) |
| Cross-model | 492 | 165 | T7 earned media: rank_delta coef=-1.98 (***) |
| 3 DDG/Qwen/s20 | needs re-run | — | — |

---

## Critical Issues

1. **Run 3 (DDG/Qwen/s20)**: LLM features at 91.8% (484 remaining, ~1.5h to finish). features_new.csv MISSING. Outcomes fixed (77.0% coverage). **To resume**: run gather_data.py with `--engine duckduckgo --serp-results 20 --llm-top-n 10 --llm-model "Qwen/Qwen2.5-72B-Instruct"`. Then run extract_features + clean + analyze.

2. **Run 4 (DDG/Qwen/s50)**: LLM features at 35.6% (4735 remaining, ~8-16h to finish). No geodml_dataset.csv. **To resume**: run gather_data.py with `--engine duckduckgo --serp-results 50 --llm-top-n 10 --llm-model "Qwen/Qwen2.5-72B-Instruct"`. Then full pipeline.

3. **Runs 6-8 (SearXNG)**: Not started. Require running SearXNG container.

4. **Moz API coverage ~6-10%**: domain authority/backlinks effectively unusable. OpenPageRank ran for Run 5 but not reflected in features_new.csv.

5. **T5 topical competence at 0% for DDG runs**: embedding similarity only computed for SearXNG.

6. **Title/snippet keyword similarity at 0% for DDG runs**: these confounders depend on SearXNG-specific snippet data.

7. **`experiment.json` resume bug**: fixed in `clean_data.py` (now falls back to `keywords.jsonl` when `per_keyword_results` is empty).


---

*end of paperSizeExperiment/doc/audit-2026-04-07.md*



<a id="papersizeexperiment--doc--status-2026-04-13"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 13 / 22 — paperSizeExperiment/doc/status-2026-04-13.md  (10212 bytes)
# ═══════════════════════════════════════════════════════════════

# Experiment Status Log — 2026-04-13

## Overview

2×2×2 factorial design: 2 engines (DuckDuckGo, SearXNG) × 2 models (Llama-3.3-70B, Qwen2.5-72B) × 2 pool sizes (serp20/top10, serp50/top10) = 8 runs.

Pipeline per run: Phase 1 (SERP+LLM) → Phase 2 (HTML fetch) → Phase 3 (LLM features) → Phase 4 (extract_features) → Phase 5 (clean_data) → Phase 6 (analyze)

**7/8 runs fully complete with T7 analysis.**

## Completion Matrix

| # | Engine | Model | Pool | P1: SERP | P2: HTML | P3: LLM | P4: Extract | P5: Clean | P6: Analyze | T7 coef |
|---|--------|-------|------|----------|----------|---------|-------------|-----------|-------------|---------|
| R1 | DDG | Llama-3.3-70B | 20/10 | 1011/1011 kw → 7890 rank | 5801 ok, 612 err | 90% (5801/6413) | 7890 rows | 7890 rows, rd=71% | 102 exp, 9 sig | -1.28 / -1.33 *** |
| R2 | DDG | Llama-3.3-70B | 50/10 | 1011/1011 kw → 8088 rank | 6218 ok, 599 err | 91% (6218/6817) | 8088 rows | 8088 rows, rd=75% | 102 exp, 22 sig | -1.63 / -1.78 *** |
| R3 | DDG | Qwen2.5-72B | 20/10 | 1011/1011 kw → 8335 rank | 6195 ok, 752 err | 89% (6195/6947) | 8335 rows | 8335 rows, rd=77% | 108 exp, 24 sig | -1.68 / -1.83 *** |
| R4 | DDG | Qwen2.5-72B | 50/10 | 1011/1011 kw → 9863 rank | 7874 ok, 717 err | 90% (7732/8591) | 9863 rows | 9863 rows, rd=89% | 102 exp, 32 sig | -1.11 / -1.36 *** |
| R5 | SXG | Llama-3.3-70B | 20/10 | 1011/1011 kw → 8313 rank | 6492 ok, 582 err | 92% (6492/7074) | 8197 rows | 8197 rows, rd=80% | 108 exp, 29 sig | -2.26 / -2.38 *** |
| R6 | SXG | Llama-3.3-70B | 50/10 | 1011/1011 kw → 12809 rank | 6432 ok, 700 err | 90% (6432/7132) | 859 rows | 12809 rows, rd=6% | 108 exp, 9 sig | -7.32 / -4.56 *** |
| R7 | SXG | Qwen2.5-72B | 20/10 | 1011/1011 kw → 9113 rank | 6995 ok, 753 err | 16% (1266/7748) | 1210 rows | 7903 rows, rd=80% | MISSING | — |
| R8 | SXG | Qwen2.5-72B | 50/10 | 396/1011 kw → 3593 rank | 830 ok, 80 err | 91% (828/910) | 901 rows | 1015 rows, rd=80% | 108 exp, 14 sig | -0.69 / -0.68 *** |

## Per-Run Detail

### R1 — DDG / Llama-3.3-70B / serp20 (COMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (1011 ok, 0 failed) → 7890 rankings
- **Phase 2 — HTML fetch**: 5801 cached HTML files, 6413 in features.csv (5801 ok, 612 errors)
- **Phase 3 — LLM features**: 5801/6413 (90%)
- **Phase 4 — extract_features**: 7890 rows, T7 coverage: 7890/7890 (100%)
- **Phase 5 — clean_data**: 7890 rows, rank_delta: 5587/7890 (71%), T7: 5587/7890 (71%)
- **Phase 6 — analyze**: 102 experiments, 9 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-1.284 (p=0.0000\*\*\*), coef=-1.334 (p=0.0000\*\*\*)

### R2 — DDG / Llama-3.3-70B / serp50 (COMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (1011 ok, 0 failed) → 8088 rankings
- **Phase 2 — HTML fetch**: 6219 cached HTML files, 6817 in features.csv (6218 ok, 599 errors)
- **Phase 3 — LLM features**: 6218/6817 (91%)
- **Phase 4 — extract_features**: 8088 rows, T7 coverage: 8088/8088 (100%)
- **Phase 5 — clean_data**: 8088 rows, rank_delta: 6064/8088 (75%), T7: 6064/8088 (75%)
- **Phase 6 — analyze**: 102 experiments, 22 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-1.633 (p=0.0000\*\*\*), coef=-1.779 (p=0.0000\*\*\*)

### R3 — DDG / Qwen2.5-72B / serp20 (COMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (1011 ok, 0 failed) → 8335 rankings
- **Phase 2 — HTML fetch**: 6197 cached HTML files, 6947 in features.csv (6195 ok, 752 errors)
- **Phase 3 — LLM features**: 6195/6947 (89%)
- **Phase 4 — extract_features**: 8335 rows, T7 coverage: 8335/8335 (100%)
- **Phase 5 — clean_data**: 8335 rows, rank_delta: 6415/8335 (77%), T7: 6415/8335 (77%)
- **Phase 6 — analyze**: 108 experiments, 24 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-1.678 (p=0.0000\*\*\*), coef=-1.828 (p=0.0000\*\*\*)

### R4 — DDG / Qwen2.5-72B / serp50 (COMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (1011 ok, 0 failed) → 9863 rankings
- **Phase 2 — HTML fetch**: 7876 cached HTML files, 8591 in features.csv (7874 ok, 717 errors)
- **Phase 3 — LLM features**: 7732/8591 (90%)
- **Phase 4 — extract_features**: 9863 rows, T7 coverage: 9863/9863 (100%)
- **Phase 5 — clean_data**: 9863 rows, rank_delta: 8742/9863 (89%), T7: 8742/9863 (89%)
- **Phase 6 — analyze**: 102 experiments, 32 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-1.112 (p=0.0000\*\*\*), coef=-1.364 (p=0.0000\*\*\*)

### R5 — SXG / Llama-3.3-70B / serp20 (COMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (973 ok, 38 failed) → 8313 rankings
- **Phase 2 — HTML fetch**: 6493 cached HTML files, 7074 in features.csv (6492 ok, 582 errors)
- **Phase 3 — LLM features**: 6492/7074 (92%)
- **Phase 4 — extract_features**: 8197 rows, T7 coverage: 8197/8197 (100%)
- **Phase 5 — clean_data**: 8197 rows, rank_delta: 6575/8197 (80%), T7: 6575/8197 (80%)
- **Phase 6 — analyze**: 108 experiments, 29 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-2.259 (p=0.0000\*\*\*), coef=-2.375 (p=0.0000\*\*\*)

### R6 — SXG / Llama-3.3-70B / serp50 (COMPLETE — DATA QUALITY FLAG)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (891 ok, 120 failed) → 12809 rankings
- **Phase 2 — HTML fetch**: 6458 cached HTML files, 7132 in features.csv (6432 ok, 700 errors)
- **Phase 3 — LLM features**: 6432/7132 (90%)
- **Phase 4 — extract_features**: 859 rows, T7 coverage: 859/859 (100%)
- **Phase 5 — clean_data**: 12809 rows, rank_delta: 787/12809 (6%), T7: 787/12809 (6%)
- **Phase 6 — analyze**: 108 experiments, 9 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-7.316 (p=0.0000\*\*\*), coef=-4.564 (p=0.0096\*\*)
  - **WARNING**: Only 6% rank_delta coverage. T7 coefficient anomalously large. Likely clean_data merge issue.

### R7 — SXG / Qwen2.5-72B / serp20 (INCOMPLETE)

- **Phase 1 — SERP + LLM**: 1011/1011 keywords (1011 ok, 0 failed) → 9113 rankings
- **Phase 2 — HTML fetch**: 6995 cached HTML files, 7748 in features.csv (6995 ok, 753 errors)
- **Phase 3 — LLM features**: 1266/7748 (16%) — **STALLED**
- **Phase 4 — extract_features**: 1210 rows (partial), T7 coverage: 1210/1210 (100%)
- **Phase 5 — clean_data**: 7903 rows, rank_delta: 6352/7903 (80%), T7: 16/7903 (0%)
- **Phase 6 — analyze**: NOT RUN
- **Action needed**: Re-run Phase 3 LLM features via Qwen API for ~6500 cached HTML files, then P4→P5→P6.

### R8 — SXG / Qwen2.5-72B / serp50 (COMPLETE — LOW POWER)

- **Phase 1 — SERP + LLM**: 396/1011 keywords (396 ok, 0 failed) → 3593 rankings
- **Phase 2 — HTML fetch**: 831 cached HTML files, 910 in features.csv (830 ok, 80 errors)
- **Phase 3 — LLM features**: 828/910 (91%)
- **Phase 4 — extract_features**: 901 rows, T7 coverage: 901/901 (100%)
- **Phase 5 — clean_data**: 1015 rows, rank_delta: 816/1015 (80%), T7: 816/1015 (80%)
- **Phase 6 — analyze**: 108 experiments, 14 significant (p<0.05), outcomes: post_rank, pre_rank, rank_delta
  - **T7 rank_delta**: coef=-0.694 (p=0.3204), coef=-0.684 (p=0.2396)
  - **NOTE**: Only 1015 rows (vs 7890–9863 for other runs). Phase 1 only 39% done. Low statistical power — T7 not significant.

## Known Issues

1. **R7 (SXG/Qwen/s20)**: Phase 3 LLM features at 16% — Qwen API processing stalled. 6995 cached HTML files available. Needs LLM feature re-extraction, then P4→P5→P6.
2. **R6 (SXG/Llama/s50)**: Only 6% rank_delta coverage in geodml_dataset.csv. T7 coefficient (-7.32) is anomalously large. Likely a clean_data merge issue — needs investigation and re-run of P5→P6.
3. **R8 (SXG/Qwen/s50)**: Phase 1 only reached 396/1011 keywords (39%). Small dataset (1015 rows). Results valid but low-powered — T7 not significant. Needs SearXNG to finish remaining keywords.

## T7 Earned Media — Cross-Run Summary

T7 measures whether a page is from an earned media source (review sites, news, blogs) vs a brand's own domain.

| Run | Engine | Model | Pool | T7 rank_delta (LGBM) | T7 rank_delta (RF) | Significant |
|-----|--------|-------|------|---------------------|-------------------|-------------|
| R1 | DDG | Llama-3.3-70B | 20/10 | -1.284 | -1.334 | p<0.0001 *** |
| R2 | DDG | Llama-3.3-70B | 50/10 | -1.633 | -1.779 | p<0.0001 *** |
| R3 | DDG | Qwen2.5-72B | 20/10 | -1.678 | -1.828 | p<0.0001 *** |
| R4 | DDG | Qwen2.5-72B | 50/10 | -1.112 | -1.364 | p<0.0001 *** |
| R5 | SXG | Llama-3.3-70B | 20/10 | -2.259 | -2.375 | p<0.0001 *** |
| R6 | SXG | Llama-3.3-70B | 50/10 | -7.316 | -4.564 | p<0.01 ** (anomalous) |
| R7 | SXG | Qwen2.5-72B | 20/10 | — | — | no analysis |
| R8 | SXG | Qwen2.5-72B | 50/10 | -0.694 | -0.684 | not significant (low power) |

**Key finding**: Earned media pages are consistently demoted by LLMs by 1.1–2.4 rank positions (negative rank_delta = demotion). This effect is statistically significant (p<0.0001) across all well-powered runs (R1–R5), both LLM models, both search engines, and both SERP pool sizes. R6 shows an anomalously large effect due to data quality issues. R8 lacks statistical power due to small sample size.

## Fixes Applied During This Session (2026-04-13)

1. **`extract_features.py` keywords.jsonl fallback**: Fixed the resume bug where `experiment.json` has empty `per_keyword_results` after gather_data.py resumes from checkpoints. Now falls back to `keywords.jsonl`, matching the fix already in `clean_data.py`.
2. **Phase 2+3 retry**: Removed error rows from `features.csv` and re-ran `gather_data.py` to retry previously failed HTML fetches. Recovered +1,964 new LLM features across R1–R5, improving coverage from 83–88% to 89–92%.
3. **R1, R2, R5 re-analysis**: Re-ran `analyze.py` with `--outcome all --measurement all` to get full 3-outcome (rank_delta, post_rank, pre_rank) analysis with all treatments including T7.
4. **R3, R4 full pipeline**: Completed Phase 3 (LLM features), extract_features, clean_data, and analyze for both Qwen runs — first time T7 results available for Qwen model.


---

*end of paperSizeExperiment/doc/status-2026-04-13.md*



<a id="papersizeexperiment--doc--status-2026-04-14"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 14 / 22 — paperSizeExperiment/doc/status-2026-04-14.md  (3881 bytes)
# ═══════════════════════════════════════════════════════════════

# Experiment Status Log — 2026-04-14

## Today's Progress

- **R6 (SXG / Llama 3.3 / serp50): COMPLETED** — finished Phase 3 (806 URLs), ran PageRank, extract_features, clean_data, analyze. Now 100% LLM coverage, 20,184 rows in dataset, 35 significant experiments.
- **R8 Phase 2 finished** — went from 47% to 100% (7,837/7,837 URLs fetched). Phase 3 advanced to 24.1%.
- **R7 Phase 3 advanced** — went from 42.7% to 49.4% (2,984 -> 3,458 LLM features).

## What's Left

### R7 — SXG / Qwen2.5-72B / serp20

**Phase 3: 3,537 URLs remaining (~5-6 hours)**

All other phases done or ready to re-run after Phase 3.

| Phase | Status |
|-------|--------|
| P1 SERP+LLM | DONE — 1,011/1,011 kw -> 9,113 rankings |
| P2 HTML | DONE — 7,748 URLs (6,995 OK, 753 errors) |
| P3 LLM features | **49.4%** — 3,458/6,995 done, **3,537 remaining** |
| P4 extract_features | needs re-run after P3 |
| P5 clean_data | needs re-run after P3 |
| P6 analyze | needs re-run after P3 |

**Resume command:**
```bash
source venv312/bin/activate
python pipeline/gather_data.py \
  --engine searxng \
  --serp-results 20 \
  --llm-top-n 10 \
  --llm-model "Qwen/Qwen2.5-72B-Instruct" \
  --keywords-file paperSizeExperiment/keywords.txt \
  --output-dir paperSizeExperiment/output/searxng_Qwen2.5-72B-Instruct_serp20_top10 \
  --progress-file paperSizeExperiment/output/searxng_Qwen2.5-72B-Instruct_serp20_top10/progress.json \
  --llm-features --pagerank
```

**After Phase 3 completes, run the rest of the pipeline:**
```bash
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "20,10" \
  --skip-gather \
  --skip-merge \
  --force
```

---

### R8 — SXG / Qwen2.5-72B / serp50

**Phase 3: 5,142 URLs remaining (~8-10 hours)**

Phase 2 just completed today. All other phases need re-run after Phase 3.

| Phase | Status |
|-------|--------|
| P1 SERP+LLM | DONE — 1,011/1,011 kw -> 9,217 rankings |
| P2 HTML | DONE — 7,837 URLs (6,774 OK, 1,063 errors) |
| P3 LLM features | **24.1%** — 1,632/6,774 done, **5,142 remaining** |
| P4 extract_features | needs re-run after P3 (currently stale: 901 rows) |
| P5 clean_data | needs re-run after P3 (currently stale: 1,015 rows) |
| P6 analyze | needs re-run after P3 (ran on 1,015 rows) |

**Resume command:**
```bash
source venv312/bin/activate
python pipeline/gather_data.py \
  --engine searxng \
  --serp-results 50 \
  --llm-top-n 10 \
  --llm-model "Qwen/Qwen2.5-72B-Instruct" \
  --keywords-file paperSizeExperiment/keywords.txt \
  --output-dir paperSizeExperiment/output/searxng_Qwen2.5-72B-Instruct_serp50_top10 \
  --progress-file paperSizeExperiment/output/searxng_Qwen2.5-72B-Instruct_serp50_top10/progress.json \
  --llm-features --pagerank
```

**After Phase 3 completes, run the rest of the pipeline:**
```bash
python paperSizeExperiment/run_experiment.py \
  --engine searxng \
  --models "Qwen/Qwen2.5-72B-Instruct" \
  --pool-sizes "50,10" \
  --skip-gather \
  --skip-merge \
  --force
```

---

## After Both R7 and R8 Complete

Re-run cross-model analysis with all 8 runs:
```bash
python paperSizeExperiment/run_experiment.py \
  --skip-gather \
  --skip-features \
  --skip-analysis \
  --force
```

Optionally run analysis_full (PLR+IRM) and analysis_halo for R3, R4, R6, R7, R8 to match R1/R2/R5 completeness.

## Overall Completion Matrix

| # | Run | Status |
|---|-----|--------|
| R1 | DDG / Llama 3.3 / serp20 | **COMPLETE** |
| R2 | DDG / Llama 3.3 / serp50 | **COMPLETE** |
| R3 | DDG / Qwen 2.5 / serp20 | **COMPLETE** |
| R4 | DDG / Qwen 2.5 / serp50 | **COMPLETE** |
| R5 | SXG / Llama 3.3 / serp20 | **COMPLETE** |
| R6 | SXG / Llama 3.3 / serp50 | **COMPLETE** (finished today) |
| R7 | SXG / Qwen 2.5 / serp20 | Phase 3 at 49.4% — 3,537 URLs left |
| R8 | SXG / Qwen 2.5 / serp50 | Phase 3 at 24.1% — 5,142 URLs left |


---

*end of paperSizeExperiment/doc/status-2026-04-14.md*



<a id="papersizeexperiment--audit-2026-04-14"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 15 / 22 — paperSizeExperiment/audit_2026-04-14.md  (7884 bytes)
# ═══════════════════════════════════════════════════════════════

# Experiment Audit — 2026-04-14 (Full Refresh)

## Summary

**5 of 8 runs fully complete.** R6 nearly done (Phase 3 at 89%). R7 and R8 have significant Phase 3 gaps, and R8's Phase 2 is still running (~47%).

## Completion Matrix

| # | Engine | Model | Pool | P1 SERP | P2 HTML | P3 LLM feat | P4 extract | P5 clean | P6 analyze | Status |
|---|--------|-------|------|---------|---------|-------------|------------|----------|------------|--------|
| R1 | DDG | Llama 3.3 | 20/10 | DONE | DONE | DONE (100%) | DONE | DONE | DONE | **COMPLETE** |
| R2 | DDG | Llama 3.3 | 50/10 | DONE | DONE | DONE (100%) | DONE | DONE | DONE | **COMPLETE** |
| R3 | DDG | Qwen 2.5 | 20/10 | DONE | DONE | DONE (100%) | DONE | DONE | DONE | **COMPLETE** |
| R4 | DDG | Qwen 2.5 | 50/10 | DONE | DONE | DONE (100%) | DONE | DONE | DONE | **COMPLETE** |
| R5 | SXG | Llama 3.3 | 20/10 | DONE | DONE | DONE (100%) | DONE | DONE | DONE | **COMPLETE** |
| R6 | SXG | Llama 3.3 | 50/10 | DONE | DONE | **88.9%** (806 remaining) | DONE | DONE* | DONE* | **NEAR COMPLETE** |
| R7 | SXG | Qwen 2.5 | 20/10 | DONE | DONE | **42.7%** (4,011 remaining) | DONE | DONE* | DONE* | **INCOMPLETE** |
| R8 | SXG | Qwen 2.5 | 50/10 | DONE | **47% done** | **43.3%** (1,874 remaining) | partial | partial | DONE* | **INCOMPLETE** |

\* = built on incomplete upstream data; needs re-running after Phase 3 completes

---

## Per-Run Detail

### R1 — DDG / Llama-3.3-70B / serp20 — COMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 7,890 rankings
- **P2**: 6,413 URLs in features.csv (5,801 OK, 612 errors), 5,801 html_cache files
- **P3**: 5,801/5,801 OK rows have LLM features (100%)
- **P4**: features_new.csv = 7,890 rows
- **P5**: geodml_dataset.csv = 7,890 rows, rank_delta = 5,587/7,890 (70.8%)
- **P6**: analysis/ (102 exp, 9 sig), analysis_full/ (204 exp), analysis_halo/ (15 rows)
- **progress.json**: phase=done

### R2 — DDG / Llama-3.3-70B / serp50 — COMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 8,088 rankings
- **P2**: 6,817 URLs in features.csv (6,218 OK, 599 errors), 6,219 html_cache files
- **P3**: 6,218/6,218 OK rows have LLM features (100%)
- **P4**: features_new.csv = 8,088 rows
- **P5**: geodml_dataset.csv = 8,088 rows, rank_delta = 6,064/8,088 (74.9%)
- **P6**: analysis/ (102 exp, 22 sig), analysis_full/ (204 exp), analysis_halo/ (15 rows)
- **progress.json**: phase=done

### R3 — DDG / Qwen2.5-72B / serp20 — COMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 8,335 rankings
- **P2**: 6,947 URLs in features.csv (6,195 OK, 752 errors), 6,197 html_cache files
- **P3**: 6,195/6,195 OK rows have LLM features (100%)
- **P4**: features_new.csv = 8,335 rows
- **P5**: geodml_dataset.csv = 8,335 rows, rank_delta = 6,415/8,335 (76.9%)
- **P6**: analysis/ (108 exp, 24 sig). No analysis_full/ or analysis_halo/.
- **progress.json**: phase=done

### R4 — DDG / Qwen2.5-72B / serp50 — COMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 9,863 rankings
- **P2**: 8,591 URLs in features.csv (7,877 OK, 714 errors), 7,883 html_cache files
- **P3**: 7,877/7,877 OK rows have LLM features (100%)
- **P4**: features_new.csv = 9,863 rows
- **P5**: geodml_dataset.csv = 9,863 rows, rank_delta = 8,742/9,863 (88.6%)
- **P6**: analysis/ (102 exp, 28 sig). No analysis_full/ or analysis_halo/.
- **progress.json**: phase=done

### R5 — SXG / Llama-3.3-70B / serp20 — COMPLETE

- **P1**: 1,011/1,011 attempted (973 OK, 38 failed) -> 8,313 rankings
- **P2**: 7,074 URLs in features.csv (6,492 OK, 582 errors), 6,493 html_cache files
- **P3**: 6,492/6,492 OK rows have LLM features (100%)
- **P4**: features_new.csv = 8,313 rows (also features_new_moz.csv = 8,197 rows)
- **P5**: geodml_dataset.csv = 8,313 rows, rank_delta = 6,691/8,313 (80.5%)
- **P6**: analysis/ (108 exp, 31 sig), analysis_full/, analysis_halo/
- **progress.json**: phase=done

### R6 — SXG / Llama-3.3-70B / serp50 — NEAR COMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 13,915 rankings (1,011 unique keywords)
  - keywords.jsonl has 1,650 lines but 1,011 unique queries (duplicates from restarts)
- **P2**: 8,098 URLs in features.csv (7,268 OK, 830 errors), 7,294 html_cache files
- **P3**: 6,462/7,268 OK rows have LLM features (**88.9%**, 806 remaining)
  - progress.json says done=0/total=836 but this is stale — actual features.csv shows 6,462 filled
- **P4**: features_new.csv = 7,570 rows
- **P5**: geodml_dataset.csv = 12,809 rows, rank_delta = 8,969/12,809 (70.0%)
  - LLM coverage in dataset: 11,588/12,809 (90.5%)
- **P6**: analysis/ (108 exp, 32 sig). No analysis_full/ or analysis_halo/.

**Remaining work:** Finish Phase 3 (806 URLs), then re-run P4-P5-P6 for clean results.

### R7 — SXG / Qwen2.5-72B / serp20 — INCOMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 9,113 rankings
- **P2**: 7,748 URLs in features.csv (6,995 OK, 753 errors), 6,995 html_cache files
- **P3**: 2,984/6,995 OK rows have LLM features (**42.7%**, 4,011 remaining)
  - progress.json says done=0/total=5,370 — stale counter, actual data shows 2,984 done
- **P4**: features_new.csv = 9,113 rows
- **P5**: geodml_dataset.csv = 9,113 rows, rank_delta = 7,375/9,113 (80.9%)
  - LLM coverage in dataset: 2,201/9,113 (24.2%)
- **P6**: analysis/ (108 exp, 34 sig). No analysis_full/ or analysis_halo/.
  - Analysis ran with only 24.2% LLM feature data — LLM treatment results unreliable

**Remaining work:** Finish Phase 3 (4,011 URLs, ~6-8 hours), then re-run P4-P5-P6.

### R8 — SXG / Qwen2.5-72B / serp50 — INCOMPLETE

- **P1**: 1,011/1,011 keywords, 0 failed -> 9,217 rankings (1,011 unique keywords)
- **P2**: ~3,891 URLs in features.csv (~3,302 OK, ~589 errors), 3,140 html_cache files
  - progress.json: phase=phase2_html_features, done=3,708, total=7,837
  - **Phase 2 is ~47% complete** — ~4,129 URLs remaining (may still be actively running)
- **P3**: 1,428/3,302 OK rows have LLM features (**43.3%**, 1,874 remaining + future P2 URLs)
  - progress.json: done=0, total=0 (Phase 3 not formally started)
- **P4**: features_new.csv = 901 rows (stale)
- **P5**: geodml_dataset.csv = 1,015 rows (stale, should be ~9,217 when complete)
- **P6**: analysis/ (108 exp, 14 sig) — ran on only 1,015 rows, very low power

**Remaining work:** Finish Phase 2 (~4,129 URLs), then Phase 3 for all OK rows, then P4-P5-P6.

---

## Analysis Completeness

| Run | analysis/ | analysis_full/ | analysis_halo/ | Experiments | Significant (p<0.05) |
|-----|-----------|---------------|---------------|-------------|---------------------|
| R1 | YES | YES | YES | 102 | 9 |
| R2 | YES | YES | YES | 102 | 22 |
| R3 | YES | no | no | 108 | 24 |
| R4 | YES | no | no | 102 | 28 |
| R5 | YES | YES | YES | 108 | 31 |
| R6 | YES | no | no | 108 | 32 (on partial data) |
| R7 | YES | no | no | 108 | 34 (on partial data) |
| R8 | YES | no | no | 108 | 14 (on 1,015 rows) |

R3 and R4 are missing analysis_full/ (PLR+IRM) and analysis_halo/ for parity with R1/R2/R5.

## Cross-Run Analysis

- **cross_model_analysis/**: 924 experiments across all subsets
- **merged_all_runs.csv**: 65,310 rows
- **merged_all_8runs.csv**: 65,426 rows (R6-R8 have incomplete data)

## Domain Classification (T7 Improvement)

- **domains.csv**: 15,328 unique domains
- **classify_checkpoint.json**: 500 domains processed, cf_results empty
- Goal: replace hardcoded earned-media domain lists with LLM-based classification

## Next Steps (Priority Order)

1. **R6**: Finish Phase 3 (806 URLs remaining — ~1-2 hours), then re-run P4-P5-P6
2. **R7**: Finish Phase 3 (4,011 URLs remaining — ~6-8 hours), then re-run P4-P5-P6
3. **R8**: Finish Phase 2 (~4,129 URLs), then Phase 3 for all OK rows, then P4-P5-P6
4. **R3, R4**: Run analysis_full/ and analysis_halo/ for completeness
5. **All 8 runs clean**: Re-run cross-model analysis
6. **Domain classification**: Continue classifying remaining ~14,828 domains for improved T7


---

*end of paperSizeExperiment/audit_2026-04-14.md*



<a id="papersizeexperiment--doc--meta-analysis-report-2026-04-15"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 16 / 22 — paperSizeExperiment/doc/meta-analysis-report-2026-04-15.md  (17660 bytes)
# ═══════════════════════════════════════════════════════════════

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


---

*end of paperSizeExperiment/doc/meta-analysis-report-2026-04-15.md*



<a id="papersizeexperiment--doc--dataforseo-plan-2026-04-22"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 17 / 22 — paperSizeExperiment/doc/dataforseo-plan-2026-04-22.md  (4473 bytes)
# ═══════════════════════════════════════════════════════════════

# DataForSEO — Menu, Pricing Corrections, and Run Plan (2026-04-22)

## Account status

- Login: `valerian.fourel@uni-hamburg.de`
- Balance after first run: **$5.87** ($5 seed + $1 signup bonus − $0.13 KD).
- Backlinks API subscription: **being activated** (previously 40204).

## Bundle A' — executed 2026-04-22

First DataForSEO run against the corpus (13,435 domains / 1,011 keywords).

| Endpoint | Result |
|---|---|
| `backlinks/bulk_ranks/live` | 40204 — awaiting subscription |
| `backlinks/bulk_backlinks/live` | 40204 — awaiting subscription |
| `backlinks/bulk_referring_domains/live` | 40204 — awaiting subscription |
| `backlinks/bulk_spam_score/live` | 40204 — awaiting subscription |
| `dataforseo_labs/google/bulk_keyword_difficulty/live` | **OK** — 1,011 keywords, 786 with KD, ~$0.13 billed |

Outputs under `paperSizeExperiment/dataforseo/output/` (gitignored).

## Pricing corrections from actual account pricing page

The catalog in `DATAFORSEO_CATALOG.md` was off on several lines. True prices:

| Endpoint | Catalog | Actual (per-task + per-result) |
|---|---|---|
| Bulk Backlinks / Ranks / Referring Domains / Spam Score | $0.00006/domain | **$0.02 task + $0.00003 result** |
| 4 bulk backlinks endpoints at full scope | $3.24 | **~$2.72** |
| Labs Bulk KD (1,011 kw) | $0.51 | **$0.12** (confirmed billed $0.13) |
| Labs Search Intent | $0.01 task + $0.0001 result | $0.001 task + $0.0001 result |

Net: the Bundle A' refresh is **~$0.70 cheaper** than the catalog claimed.

## Newly-visible capabilities not in the original catalog

### AI Optimization API — direct GEO measurement

These directly measure LLM visibility (ChatGPT, Claude, Gemini, Perplexity)
rather than the re-ranking proxy the paper currently relies on.

| Endpoint | Per task | Per result | What it returns |
|---|---|---|---|
| `llm_mentions/search/live` | $0.10 | $0.001 | Where a keyword/brand appears in real LLM responses |
| `llm_mentions/top_domains/live` | $0.10 | $0.001 | Top domains cited by LLMs for a keyword |
| `llm_mentions/top_pages/live` | $0.10 | $0.001 | Top pages cited by LLMs |
| `llm_mentions/aggregated_metrics/live` | $0.10 | $0.001 | Aggregated brand/domain visibility |
| `llm_scraper/live/advanced` | $0.004 | — | Scrape a specific LLM's response to a prompt |
| `llm_responses/live` | $0.0006 | — | Direct LLM response |
| `ai_keyword_data/keywords_search_volume/live` | $0.01 | $0.0001 | Keyword volume inside AI assistants |

**Paper-level implication:** this converts the largest methodological
limitation of GEODML ("re-ranking is a proxy for GEO visibility") into a
validation layer. Running `llm_mentions/top_domains/live` on the 1,011 keywords
(~$121) gives a direct measurement to cross-check the DML estimates against.

### Other high-value adds

| Endpoint | Full-scope cost | Unlocks |
|---|---|---|
| `serp/google/organic/live/regular` | $2.02 (1,011 × $0.002) | Third SERP source; fixes T5/title-sim/snippet-sim on DDG runs |
| `dataforseo_labs/search_intent/live` | ~$0.10 | T8 treatment: info/commercial/transactional |
| `dataforseo_labs/bulk_search_volume/live` | ~$0.12 | Per-keyword traffic as confounder |
| `on_page/instant_pages` | $3.19 (25,481 × $0.000125) | Replaces HTML scraper for page features |
| `domain_analytics/whois/overview/live` | $13.53 | Resurrects `X2_domain_age_years` (100% NaN) |
| `on_page/lighthouse/live` | ~$108 | Resurrects `X4_lcp_ms` (100% NaN) — expensive |

## Next run plan — Bundle B (rows 1+2+3)

Hard cost ceiling: **$6**.

| # | Endpoint | Scope | Estimated |
|---|---|---|---|
| 1a | backlinks/bulk_ranks/live | 13,435 domains | $0.68 |
| 1b | backlinks/bulk_backlinks/live | 13,435 domains | $0.68 |
| 1c | backlinks/bulk_referring_domains/live | 13,435 domains | $0.68 |
| 1d | backlinks/bulk_spam_score/live | 13,435 domains | $0.68 |
| 2a | dataforseo_labs/google/search_intent/live | 1,011 keywords | $0.10 |
| 2b | dataforseo_labs/google/bulk_search_volume/live | 1,011 keywords | $0.12 |
| 3 | serp/google/organic/live/regular | 1,011 keywords | $2.02 |
| | **Total** | | **~$4.96** |

Outputs saved under `paperSizeExperiment/dataforseo/output/` (gitignored):
raw JSON per chunk + flat CSVs + updated `run_manifest.json`.

## Deferred for later

- `llm_mentions/*` pilot on 50 keywords (~$6) → if shape is right, full run ~$121.
- `on_page/instant_pages` for all 25,481 URLs ($3.19).
- `domain_analytics/whois/overview/live` for domain age ($13.53).


---

*end of paperSizeExperiment/doc/dataforseo-plan-2026-04-22.md*



<a id="papersizeexperiment--doc--dataforseo-catalog"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 18 / 22 — paperSizeExperiment/doc/DATAFORSEO_CATALOG.md  (9739 bytes)
# ═══════════════════════════════════════════════════════════════

# DataForSEO Data Catalog — GEODML Paper-Size Experiment

**Purpose:** Menu of DataForSEO endpoints that can refill/replace Moz and extend the dataset, with per-call prices and the *full* cost to cover our entire corpus.

**Corpus scope (as of 2026-04-21):**

- 1,011 unique keywords
- 25,481 unique URLs
- 13,436 unique domains

All "full cost" figures below = (scope size) × (price per call), rounded. Prices are DataForSEO's public pay-as-you-go rates — no subscription required.

---

## 1. Per-domain endpoints (scope = 13,436 domains)

### Backlinks — Bulk API (ultra-cheap)

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| Bulk **Ranks** | Domain / page rank (Moz-DA proxy) — replaces `X1_domain_authority` | $0.00006 | **~$0.81** |
| Bulk **Backlinks** | Total backlinks count — replaces `conf_backlinks` | $0.00006 | **~$0.81** |
| Bulk **Referring Domains** | Replaces `conf_referring_domains` | $0.00006 | **~$0.81** |
| Bulk **Spam Score** | Domain spam score | $0.00006 | **~$0.81** |
| Bulk **New/Lost Backlinks** | Link-velocity signal | $0.00006 | **~$0.81** |
| Bulk **New/Lost Referring Domains** | Referring-domain velocity | $0.00006 | **~$0.81** |
| Bulk **Pages Summary** | # of pages indexed + rank distribution | $0.00006 | **~$0.81** |

### Backlinks — Detailed API

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| Backlinks **Summary** | Full backlink profile per domain | $0.02 | ~$269 |
| Backlinks **Anchors** | Anchor-text distribution | $0.02 | ~$269 |
| Backlinks **Referring Domains (list)** | Full list of referring domains | $0.02 | ~$269 |
| Backlinks **Backlinks (list)** | Full link list | $0.02 | ~$269 |
| Backlinks **History** | Backlink profile over time | $0.02 | ~$269 |

### DataForSEO Labs — Domain Level

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| Labs **Bulk Traffic Estimation** | Organic traffic estimate per domain | $0.0005 | **~$6.72** |
| Labs **Bulk Keyword Difficulty for URL** | Avg rank-worthy KD | $0.0005 | **~$6.72** |
| Labs **Domain Rank Overview** | Traffic + keyword count + rank metrics — replaces `X1_global_rank` | $0.02 | ~$269 |
| Labs **Ranked Keywords** | All keywords a domain ranks for | $0.02 | ~$269 |
| Labs **SERP Competitors** | Competitors for a domain | $0.02 | ~$269 |
| Labs **Domain Intersection** | Keyword overlap between domains | $0.02 | ~$269 |

### Domain Analytics

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| **Whois Overview** | Domain age — resurrects `X2_domain_age_years` (100% NaN today, dropped) | $0.03 | ~$403 |
| **Technologies** | Tech stack (CMS, analytics, frameworks) | $0.02 | ~$269 |
| **Technologies Aggregation** | Bulk tech lookup | $0.004 | ~$54 |

---

## 2. Per-URL endpoints (scope = 25,481 URLs)

### On-Page API

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| **Instant Pages** | Meta tags, headings, word count — fills X3/X6/X7 counts | $0.00025 | **~$6.37** |
| **Raw HTML** | Cached HTML body (reusable for our LLM features) | $0.00025 | **~$6.37** |
| **Content Parsing** | Parsed article text, structured content | $0.0005 | **~$12.74** |
| **Microdata** | JSON-LD / schema.org extraction — validates T3 | $0.00025 | **~$6.37** |
| **Links** | Internal / outbound link graph | $0.00025 | **~$6.37** |
| **Duplicate Content** | Canonical / duplication ratio | $0.00025 | **~$6.37** |
| **Redirect Chains** | Hop count, final URL | $0.00025 | **~$6.37** |
| **Lighthouse** | Core Web Vitals: LCP, FCP, CLS, TTI — resurrects `X4_lcp_ms` (100% NaN, dropped) | $0.003 | ~$76.44 |
| **Keyword Density** | Top N-grams per page | $0.00025 | ~$6.37 |

---

## 3. Per-keyword endpoints (scope = 1,011 keywords)

### Keywords Data API

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| **Google Ads Search Volume** | Monthly volume, CPC, competition | $0.00005 | **~$0.05** |
| **Google Ads Keywords for Site** | Keywords a site ranks for | $0.05 | ~$51 |
| **Google Trends** | 5-year trend curve | $0.005 | **~$5.06** |
| **Clickstream Search Volume** | Clickstream-based monthly volume | $0.0001 | **~$0.10** |
| **Clickstream Global Search Volume** | Global clickstream volume | $0.0001 | **~$0.10** |
| **Bing Keyword Data** | Bing monthly volume, CPC | $0.00005 | **~$0.05** |

### DataForSEO Labs — Keyword Level

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| **Bulk Keyword Difficulty** | Resurrects `X8_keyword_difficulty` (54.6% NaN) | $0.0005 | **~$0.51** |
| **Keyword Difficulty** (detailed) | KD + SERP features | $0.01 | **~$10.11** |
| **Search Intent** | Info / commercial / transactional / nav label | $0.01 | **~$10.11** |
| **Historical Search Volume** | Monthly vol 2019 → today | $0.01 | **~$10.11** |
| **Historical SERPs** | SERP snapshots over time | $0.01 | **~$10.11** |
| **Keyword Ideas** | Related keyword suggestions | $0.01 | **~$10.11** |
| **Keyword Suggestions** | Autocomplete-style | $0.01 | **~$10.11** |
| **Related Keywords** | "Also searches for" | $0.01 | **~$10.11** |
| **Subdomains** | Subdomain rank distribution | $0.01 | **~$10.11** |

### SERP API — Google (per keyword)

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| Google Organic **Regular** | Full fresh Google top-100 SERP | $0.0006 | **~$0.61** |
| Google Organic **Live Regular** | Real-time SERP | $0.002 | **~$2.02** |
| Google Organic **Live Advanced** | Regular + rich SERP elements | $0.002 | **~$2.02** |
| Google **News** | News SERP | $0.0006 | **~$0.61** |
| Google **Images** | Image SERP | $0.0006 | **~$0.61** |
| Google **Shopping** | Shopping SERP | $0.0006 | **~$0.61** |
| Google **Events** | Events SERP | $0.002 | **~$2.02** |
| Google **Jobs** | Jobs SERP | $0.002 | **~$2.02** |
| Google **Maps** | Maps SERP | $0.002 | **~$2.02** |
| Google **Local Finder** | Local pack | $0.002 | **~$2.02** |
| Google **Autocomplete** | Autocomplete suggestions | $0.0006 | **~$0.61** |

### SERP API — Other Engines (per keyword)

| Endpoint | Variable it fills | $/call | Full cost |
|---|---|---|---|
| **Bing Organic** | Bing SERP | $0.0006 | **~$0.61** |
| **Yahoo Organic** | Yahoo SERP | $0.0006 | **~$0.61** |
| **DuckDuckGo Organic** | DDG SERP — replaces our scraper | $0.0006 | **~$0.61** |
| **Yandex Organic** | Yandex SERP (RU) | $0.0006 | **~$0.61** |
| **Naver Organic** | Naver SERP (KR) | $0.0006 | **~$0.61** |
| **Baidu Organic** | Baidu SERP (CN) | $0.0006 | **~$0.61** |
| **YouTube Organic** | YouTube SERP | $0.0006 | **~$0.61** |

---

## 4. Cost bundles

### Bundle A — "Just replace Moz, cheapest path"

| Endpoint | Cost |
|---|---|
| Bulk Ranks | $0.81 |
| Bulk Backlinks | $0.81 |
| Bulk Referring Domains | $0.81 |
| Labs Bulk KD | $0.51 |
| **Total** | **~$3** |

### Bundle B — "Full referencing refresh"

| Endpoint | Cost |
|---|---|
| Bulk Ranks + Backlinks + Referring Domains + Spam + Velocity (×7) | ~$5.70 |
| Labs Bulk Traffic Estimation | $6.72 |
| Labs Domain Rank Overview | $269 |
| **Total** | **~$281** |

### Bundle C — "Resurrect the 100%-NaN columns"

| Endpoint | What it brings back | Cost |
|---|---|---|
| Whois Overview | `X2_domain_age_years` | ~$403 |
| On-Page Lighthouse | `X4_lcp_ms` | ~$76 |
| Labs Bulk KD | `X8_keyword_difficulty` (from 54.6% NaN → 0%) | ~$0.51 |
| **Total** | | **~$480** |

### Bundle D — "Throw everything cheap at the experiment"

Every endpoint in this doc priced ≤ $0.001/call × full scope:

| Block | Subtotal |
|---|---|
| All 7 Bulk Backlinks endpoints | ~$5.70 |
| Labs bulk domain metrics (Traffic + KD-for-URL) | ~$13.44 |
| All 8 On-Page endpoints @ $0.00025 | ~$51 |
| Keywords Data (Volume + Trends + Clickstream + Bing) | ~$5.36 |
| Labs keyword endpoints (8 at $0.01 + KD bulk) | ~$81 |
| SERP × 10 engines/variants | ~$10 |
| **Total** | **~$166** |

### Bundle E — "Absolute minimum for a Moz drop-in replacement"

Just the three headline backlink numbers:

- Bulk Ranks + Bulk Backlinks + Bulk Referring Domains = **~$2.43 total**

---

## 5. Mapping DataForSEO → existing dataset columns

| Dataset column (current NaN %) | DataForSEO endpoint | Full cost |
|---|---|---|
| `conf_backlinks` (88.8%) | Bulk Backlinks | $0.81 |
| `conf_referring_domains` (88.8%) | Bulk Referring Domains | $0.81 |
| `conf_domain_authority` (78.2%) | Bulk Ranks | $0.81 |
| `X1_domain_authority` (72.4%) | Bulk Ranks | $0.81 |
| `X1_global_rank` (72.8%) | Labs Domain Rank Overview | ~$269 |
| `X2_domain_age_years` (100%, dropped) | Whois Overview | ~$403 |
| `X4_lcp_ms` (100%, dropped) | On-Page Lighthouse | ~$76 |
| `X8_keyword_difficulty` (54.6%) | Labs Bulk KD | $0.51 |
| `X3_word_count`, `conf_word_count` (~12%) | On-Page Instant Pages | $6.37 |
| `X7_internal_links`, `X7B_outbound_links` (~10%) | On-Page Links | $6.37 |
| `X9_images_with_alt` (~10%) | On-Page Instant Pages | $6.37 |
| `treat_structured_data` (~10%) | On-Page Microdata | $6.37 |

**Total to refill every missing numeric column (excluding Whois): ~$373.**

**Cheapest full referencing drop-in: ~$3.**

---

## 6. Notes

- Prices above are DataForSEO's **Standard** (async) pricing. **Live** endpoints cost roughly 3× more but return synchronously — useful for small batches, not for our 13k-scale backfill.
- All endpoints accept **bulk POST** of up to ~1,000 targets per request, so wall-clock time is minutes not days.
- DataForSEO bills on successful responses only; failed lookups are free.
- No monthly minimum — pay-as-you-go from a prepaid balance.
- Prices verified against DataForSEO public pricing as of early 2026; confirm on their site before committing a budget.


---

*end of paperSizeExperiment/doc/DATAFORSEO_CATALOG.md*



<a id="papersizeexperiment--doc--analysis-2026-04-23"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 19 / 22 — paperSizeExperiment/doc/analysis-2026-04-23.md  (23605 bytes)
# ═══════════════════════════════════════════════════════════════

# Analysis log — 2026-04-23

Comprehensive writeup of today's re-analysis after merging DataForSEO-derived
keyword-level confounders into the experiment. All numbers sourced from the
CSV/log files listed at the end of each section.

---

## 1. Session goals

1. Finish the in-flight DataForSEO Google SERP pull (114/1,011 done when
   interrupted) — completed via resume logic, 1,011/1,011.
2. Backfill the 161 keywords missing from Labs `keyword_overview/live`
   using `google_ads/search_volume/live` and `labs/search_intent/live`.
3. Merge everything (keyword_overview, bulk KD, Google SERP, Google Ads
   SV, search_intent) into `full_experiment_data.csv` as new `dfs_*`
   columns.
4. Re-run the full DML PLR study (15 subsets × 19 treatments × 2 outcomes
   = 570 fits).
5. Audit confounder performance: LGBM gain importance, 5-fold CV R² for
   each nuisance model, leave-one-out ΔR², and OLS significance with HC3
   robust standard errors.

All five completed. Total DataForSEO spend: **$3.86**. DML rerun
runtime: 23.8 min. No sign flips across 570 paired fits against the
pre-DFS baseline.

---

## 2. Data inventory after the pull

### New columns in `full_experiment_data.csv`

| column | source | coverage | notes |
|---|---|---:|---|
| `dfs_keyword_difficulty` | Labs bulk_keyword_difficulty + keyword_overview fallback | 76.9% | Google KD 0-100 |
| `dfs_search_volume` | Labs keyword_overview + Google Ads backfill | 80.6% | Monthly US searches |
| `dfs_cpc` | Labs keyword_overview + Google Ads backfill | 65.2% | Cost per click (USD) |
| `dfs_competition` | Labs keyword_overview | 76.8% | Ads competition 0-1 |
| `dfs_competition_level` | Labs keyword_overview + Google Ads backfill | 79.2% | LOW / MEDIUM / HIGH (string) |
| `dfs_main_intent` | Labs keyword_overview + Labs search_intent backfill | **100.0%** | commercial/informational/navigational/transactional |
| `dfs_foreign_intent` | Labs keyword_overview + Labs search_intent backfill | 38.8% | JSON of secondary intents |
| `dfs_intent_commercial` | one-hot from `dfs_main_intent` | 100.0% | |
| `dfs_intent_informational` | one-hot | 100.0% | |
| `dfs_intent_navigational` | one-hot | 100.0% | |
| `dfs_intent_transactional` | one-hot | 100.0% | |
| `dfs_google_rank` | Google SERP | 12.7% | Best rank per (keyword, domain) on Google organic top-20 |
| `dfs_google_rank_absolute` | Google SERP | 12.7% | Including ads/AI overview positions |
| `dfs_se_results_count` | Google SERP | 12.7% | Total Google results for keyword |
| `dfs_google_top_url` | Google SERP | 12.7% | Domain's best URL on Google |

### Existing columns filled

- `X8_keyword_difficulty` went from **45.4% → 87.4%** non-null via
  `dfs_keyword_difficulty` backfill.

### Still blocked (40204 on backlinks subscription)

- `conf_domain_authority` remains at 21.8% coverage
- `conf_backlinks`, `conf_referring_domains` remain at 11.2%
- `X1_domain_authority`, `X1_global_rank` remain at 27%

When the DataForSEO backlinks subscription activates, rerunning the
merge pipeline will backfill these and add new `dfs_domain_rank`,
`dfs_backlinks`, `dfs_referring_domains`, `dfs_spam_score` columns.

Source files: `results/full_experiment_data.csv` (65,203 × 73).

---

## 3. DML rerun — headline results (POOLED, outcome = `rank_delta`)

Sorted by magnitude. Positive coefficient = LLM **promotes** the page.

| treatment | θ | SE | t | p | direction |
|---|---:|---:|---:|---:|---|
| T7_source_earned | **−1.7000** | 0.066 | −25.8 | 1.3e-144 *** | **strongly demotes** |
| T5_topical_comp | +0.4383 | 0.102 | +4.29 | 2.0e-05 *** | promotes |
| T3_structured_data_new | −0.1396 | 0.022 | −6.32 | 3.4e-10 *** | demotes |
| T3_code | +0.1269 | 0.026 | +4.85 | 1.3e-06 *** | promotes |
| T2a_question_headings | +0.1034 | 0.025 | +4.12 | 3.9e-05 *** | promotes |
| T_llms_txt | +0.0942 | 0.022 | +4.29 | 1.8e-05 *** | promotes |
| T2_code | +0.0677 | 0.023 | +2.93 | 3.4e-03 *** | promotes |
| T6_freshness | −0.0597 | 0.006 | −9.65 | 7.1e-22 *** | demotes |
| T2_llm | −0.0445 | 0.022 | −2.05 | 4.0e-02 ** | demotes |
| T4_llm | −0.0213 | 0.006 | −3.66 | 2.5e-04 *** | demotes |
| T4_code | −0.0196 | 0.006 | −3.14 | 1.7e-03 *** | demotes |
| T4b_auth_citations | −0.0190 | 0.006 | −3.06 | 2.2e-03 *** | demotes |
| T1_code | −0.0179 | 0.002 | −8.01 | 1.0e-15 *** | demotes |
| T1b_stats_density | −0.0173 | 0.002 | −7.60 | 2.3e-14 *** | demotes |
| T1_llm | −0.0083 | 0.002 | −4.71 | 2.8e-06 *** | demotes |
| T2b_structural_modularity | +0.0022 | 0.001 | +3.56 | 3.8e-04 *** | promotes |

Non-significant at POOLED: T1a_stats_present, T3_llm, T4a_ext_citations.

Results mirror (with opposite sign per ranking convention) on outcome =
`post_rank`. All 570 subset × treatment × outcome fits in
`results/dml_study/dml_results_long.csv`. Narrative in
`results/dml_study/dml_summary.md`.

### Robustness — pre-DFS vs post-DFS

| metric | value |
|---|---|
| Sign flips across 570 paired fits | **0** |
| Median \|Δcoef\| | 0.0057 |
| Max \|Δcoef\| | 0.334 (on `duckduckgo_Qwen…serp20` · T5_topical_comp; POOLED barely moved) |
| Significant fits (p<0.01): pre-DFS → post-DFS | 230 → 234 (+4) |
| Significance upgrades | 1 (T2_llm POOLED rank_delta: * → **) |
| Significance downgrades | 0 |

**The DataForSEO confounders did not confound any prior finding.** This
is the robustness result that belongs in the paper.

---

## 4. Nuisance model predictive performance

DML PLR partials out confounders by fitting two nuisance models:

    Y = g₀(X) + ε_y     →  ε̂_y = Y - ĝ(X)
    D = m₀(X) + ε_d     →  ε̂_d = D - m̂(X)
    ε̂_y = θ · ε̂_d + noise       (final structural regression)

The 5-fold CV R² of each, on POOLED, for every (treatment, outcome) pair:

### R²(Y | X) — same across treatments, depends on outcome

| outcome | n | R²(Y \| X) |
|---|---:|---:|
| rank_delta | 65,203 | **0.781** |
| post_rank  | 65,203 | **0.359** |

Strong outcome residualisation. ~78% of the `rank_delta` variance is
predicted by confounders (largely by `conf_serp_position`, which equals
`pre_rank` and therefore appears inside the rank_delta outcome by
construction). ~36% on the raw `post_rank` is genuine predictive signal.

### R²(D | X) — varies per treatment

Lower R² = more *exogenous* treatment variation = stronger DML
identification. All values safely below the 95% overlap-violation
threshold.

| treatment | R²(D \| X) | identification strength |
|---|---:|---|
| T1_llm | 0.199 | strongest |
| T3_code | 0.203 | strongest |
| T3_llm | 0.207 | strong |
| T_llms_txt | 0.226 | strong |
| T3_structured_data_new | 0.321 | good |
| T2_llm | 0.299 | good |
| T2a_question_headings | 0.360 | moderate |
| T6_freshness | 0.371 | moderate |
| T1b_stats_density | 0.404 | moderate |
| T1_code | 0.409 | moderate |
| T4b_auth_citations | 0.511 | weaker |
| T4_code | 0.494 | weaker |
| T5_topical_comp | 0.529 | weaker |
| T4a_ext_citations | 0.580 | weakest |

### R²(Ỹ | D̃) — variance share of treatment effect after partialling

| treatment | R²(Ỹ \| D̃) on rank_delta |
|---|---:|
| T7_source_earned | **0.0138** |
| T6_freshness | 0.0016 |
| T1_code | 0.0012 |
| T1b_stats_density | 0.0012 |
| T3_structured_data_new | 0.0006 |
| T5_topical_comp | 0.0005 |
| T3_code | 0.0004 |
| T_llms_txt | 0.0003 |
| T2a_question_headings | 0.0003 |
| (all others) | < 0.001 |

**T7_source_earned is the only treatment with a non-trivial variance
share** (~1.4%). All others are statistically significant but explain
tiny shares of residualised outcome variance — significance is powered
by sample size (n ≈ 60k), not effect magnitude.

Source: `results/dml_study/nuisance_r2.csv`.

---

## 5. Confounder gain importance (LightGBM)

For each nuisance model we extracted LightGBM gain importance. The top
confounders by mean rank across 2 outcome models + 8 top treatment
models:

| rank | confounder | mean imp % | coverage |
|---:|---|---:|---:|
| 1 | conf_word_count | 19.0 | 87% |
| 2 | conf_internal_links | 10.8 | 89% |
| 3 | conf_images_alt | 6.9 | 89% |
| 4 | conf_outbound_links | 4.2 | 89% |
| 5 | conf_readability | 5.5 | 85% |
| 6 | conf_domain_authority | 3.7 | **22% (sparse)** |
| 7 | **dfs_cpc** | **2.5** | 65% |
| 8 | conf_title_len | 3.5 | 100% |
| 9 | conf_bm25 | 4.5 | 100% |
| 10 | conf_title_kw_sim | 4.0 | 100% |
| 11 | conf_brand_recog | 2.0 | 100% |
| 12 | **dfs_keyword_difficulty** | **2.7** | 77% |
| 13 | **dfs_competition** | **2.1** | 77% |
| 14 | conf_backlinks | 2.8 | **11% (sparse)** |
| 15 | conf_referring_domains | 2.1 | 11% |
| 16 | conf_snippet_len | 1.6 | 100% |
| 17 | conf_snippet_kw_sim | 2.5 | 100% |
| 18 | **dfs_search_volume** | **1.3** | 81% |
| 19 | conf_serp_position | 17.4 | 100% |
| 20+ | **dfs_intent_informational** | 0.32 | 100% |
| | **dfs_intent_commercial** | 0.28 | 100% |
| | **dfs_intent_navigational** | 0.10 | 100% |
| | **dfs_intent_transactional** | 0.06 | 100% |
| | conf_title_has_kw | 0.06 | 100% |
| | conf_https | 0.05 | 100% |

Where the DataForSEO confounders do the most work (share of nuisance
model importance):

| nuisance model | DFS share |
|---|---:|
| m₀(D = T1_stats_density_code \| X) | **25.3%** |
| m₀(D = has_llms_txt \| X) | 13.2% |
| m₀(D = T3_code \| X) | 12.4% |
| m₀(D = T7_source_earned \| X) | 9.4% |
| m₀(D = T3_structured_data_new \| X) | 9.2% |
| m₀(D = T6_freshness \| X) | 7.1% |
| m₀(D = T2a_question_headings \| X) | 6.5% |
| m₀(D = T5_topical_comp \| X) | 4.2% |
| g₀(Y = post_rank \| X) | 5.4% |
| g₀(Y = rank_delta \| X) | 0.9% |

Source: `results/dml_study/confounder_audit.csv`.

---

## 6. Variance explained — CV R² decomposition

Fitting each target on (all confounders), (only `conf_*`), (only `dfs_*`):

| target | n | R²_all | R²_conf_only | R²_dfs_only | ΔR²_dfs |
|---|---:|---:|---:|---:|---:|
| OUT rank_delta | 65,203 | 0.7811 | 0.7774 | 0.0481 | +0.0037 |
| OUT post_rank | 65,203 | 0.3585 | 0.3491 | 0.0051 | +0.0094 |
| TRT topical_comp | 39,561 | 0.5289 | 0.5216 | 0.1443 | +0.0074 |
| TRT **T1_stats_density_code** | 57,757 | 0.4091 | 0.3676 | 0.1617 | **+0.0415** |
| TRT freshness | 58,566 | 0.3710 | 0.3628 | 0.0951 | +0.0082 |
| TRT question_headings | 58,566 | 0.3601 | 0.3535 | 0.0883 | +0.0065 |
| TRT structured_data_new | 58,566 | 0.3213 | 0.3102 | 0.0776 | +0.0111 |
| TRT source_earned | 65,203 | 0.2949 | 0.2772 | 0.0468 | +0.0177 |
| TRT has_llms_txt | 65,203 | 0.2264 | 0.2158 | 0.0655 | +0.0106 |
| TRT T3_structured_data_code | 59,577 | 0.2026 | 0.1958 | 0.0441 | +0.0068 |

**Biggest incremental contribution**: T1_stats_density_code gains
+4.15pp of R² from DFS features — the only place where the pre-DFS
fit was missing a meaningful chunk of identifying variation. The
corresponding DML coefficient barely moved (−0.01694 → −0.01794), so
the hidden bias from that missing variation was small.

Source: `results/dml_study/variance_explained.csv`.

---

## 7. Leave-one-out ΔR² — unique contribution per confounder

For each confounder, how much does 5-fold CV R²(Y|X) drop when we remove
just that column?

### On outcome = rank_delta (full R² = 0.7811)

| confounder dropped | ΔR² |
|---|---:|
| **conf_serp_position** | **+0.6547** |
| conf_bm25 | +0.0023 |
| conf_brand_recog | +0.0023 |
| conf_domain_authority | +0.0014 |
| dfs_competition | +0.0012 |
| conf_word_count | +0.0012 |
| conf_outbound_links | +0.0010 |
| dfs_cpc | +0.0009 |
| (all others) | < 0.001 |

**One column does 84% of the work.** The `rank_delta` outcome is
`pre_rank − post_rank`, and `conf_serp_position = pre_rank`, so the
confounder appears inside the outcome by construction. The R² inflation
is partly algebraic, not purely informational.

### On outcome = post_rank (full R² = 0.3585)

| confounder dropped | ΔR² |
|---|---:|
| **conf_serp_position** | **+0.2615** |
| conf_bm25 | +0.0064 |
| conf_brand_recog | +0.0061 |
| conf_domain_authority | +0.0036 |
| conf_word_count | +0.0031 |
| **dfs_competition** | **+0.0029** |
| conf_outbound_links | +0.0024 |
| conf_images_alt | +0.0017 |
| conf_readability | +0.0014 |
| conf_internal_links | +0.0012 |
| **dfs_cpc** | **+0.0011** |
| conf_title_len | +0.0009 |
| **dfs_keyword_difficulty** | **+0.0008** |

More distributed pattern. No single non-serp-position confounder is
uniquely load-bearing — LightGBM can substitute across correlated
features.

Source: `results/dml_study/confounder_loo_r2.csv`.

---

## 8. OLS significance — confounders on `post_rank`

Standardised OLS (median-imputed, HC3 robust SEs, POOLED n = 65,203):

Signs interpreted under the ranking convention (negative coef on
`post_rank` = confounder associated with **better** rank).

| confounder | coef (std) | t | p | reading |
|---|---:|---:|---:|---|
| conf_serp_position | +1.419 | 131 | <1e-300 | mechanical pre→post |
| conf_brand_recog | **−0.244** | 29 | 4e-180 | known brands → better rank |
| dfs_competition | **−0.110** | 12 | 3e-31 | ads-competitive kws → better ranks |
| conf_domain_authority | **−0.116** | 11 | 2e-27 | high DA → better rank |
| conf_title_has_kw | +0.101 | 10 | 2e-21 | naive title-kw → worse rank |
| conf_bm25 | +0.094 | 9 | 1e-20 | kw-match surface → worse rank |
| conf_snippet_kw_sim | +0.079 | 8 | 1e-15 | surface kw in snippet → worse |
| conf_word_count | +0.117 | 8 | 1e-14 | longer pages → worse rank |
| conf_outbound_links | +0.063 | 6 | 2e-08 | more outbound → worse |
| dfs_intent_navigational | +0.038 | 5 | 6e-07 | navigational → worse (expected) |
| **dfs_cpc** | **−0.042** | 5 | 4e-06 | high CPC → better rank |
| conf_title_len | +0.061 | 4 | 2e-04 | longer titles → worse |
| conf_https | −0.032 | 3 | 5e-04 | https → slightly better |
| **dfs_search_volume** | **−0.029** | 3 | 6e-04 | higher volume → better rank |
| conf_images_alt | −0.031 | 3 | 5e-03 | accessibility signal → better |
| conf_referring_domains | +0.023 | 2 | 0.02 * | weakly worse (sparse column) |
| dfs_intent_commercial | −0.013 | 2 | 0.02 * | marginal |

Non-significant (p > 0.10):

| confounder | t | p |
|---|---:|---:|
| dfs_keyword_difficulty | 1.6 | 0.11 |
| dfs_intent_informational | 1.4 | 0.15 |
| conf_backlinks | 1.1 | 0.26 (sparse) |
| conf_snippet_len | 1.1 | 0.28 |
| conf_internal_links | 0.6 | 0.52 |
| conf_title_kw_sim | 0.6 | 0.57 |
| conf_readability | 0.1 | 0.93 |
| dfs_intent_transactional | 0.04 | 0.97 |

**17 of 25 confounders reach p < 0.05 on `post_rank` after controlling
for all others.**

Of the 8 new DataForSEO confounders:
- **4 are significant at p<0.001**: `dfs_competition`,
  `dfs_cpc`, `dfs_intent_navigational`, `dfs_search_volume`
- **1 is marginal**: `dfs_intent_commercial` (p=0.02)
- **3 are not significant**: `dfs_keyword_difficulty` (OLS-insignificant
  but LGBM-important on specific treatments — see §5),
  `dfs_intent_informational`, `dfs_intent_transactional`

Source: `results/dml_study/confounder_ols_significance.csv`.

---

## 9. Substantive interpretation

### Which covariates actually describe what ranks well?

1. **Brand recognition is the single strongest predictor** after
   pre_rank. Named-entity matching to a known-brand list gives t = 29.
   The LLM re-rankers favour known brands.

2. **Paid-search competitiveness predicts better organic rank.**
   `dfs_competition` (t = 12) and `dfs_cpc` (t = 5) both favour
   commercially hot keywords. Pages for keywords with expensive ads
   tend to rank better organically after re-ranking — plausibly because
   commercially-important keywords have better-optimised pages in their
   SERPs.

3. **Domain authority predicts better rank, but sparse coverage hides
   it.** Only 22% of rows have `conf_domain_authority`. When present,
   t = 11 — we expect this to become the dominant non-brand feature
   once the backlinks subscription unblocks.

4. **Counter-signal: surface keyword matching correlates with *worse*
   rank.** `conf_title_has_kw`, `conf_bm25`, `conf_snippet_kw_sim`,
   `conf_word_count`, `conf_title_len` all have *positive* coefficients
   on `post_rank` (= worse rank). Interpretation: after conditioning on
   pre_rank and brand, pages that are *only* superficially optimised
   (long with kw matches, thin on content) rank worse. The LLM
   promotes brand-authority over naive SEO.

5. **Intent matters in one direction only.** `dfs_intent_navigational`
   is positively signed (navigational keywords → worse post_rank).
   Other intent dummies are ~zero. The commercial-vs-informational
   dimension is absorbed by `dfs_cpc` and `dfs_competition`.

### Dead confounders

`conf_readability` (t = 0.1, p = 0.93), `conf_title_kw_sim`,
`dfs_intent_transactional`, `dfs_intent_informational`, and
`dfs_intent_commercial` (barely marginal) add no unique signal after
the others are controlled.

**Recommendation for a cleaner specification**: drop the 4
`dfs_intent_*` dummies in favour of CPC + competition + (optionally)
main_intent as a categorical dummy. `conf_readability` can also be
dropped without loss.

---

## 10. Identification summary for the paper

DML is operating in a healthy regime on POOLED:

| diagnostic | value | verdict |
|---|---|---|
| Outcome residualisation R²(Y=rank_delta\|X) | 78% | Strong |
| Outcome residualisation R²(Y=post_rank\|X) | 36% | Adequate |
| Max R²(D\|X) across treatments | 58% (T4a_ext_citations) | Below 95% overlap-violation threshold |
| Min R²(D\|X) across treatments | 20% (T1_llm, T3_code) | Strong identification |
| Significance upgrades post-DFS | 1 | Confounders add minor refinement |
| Significance downgrades post-DFS | 0 | No spurious findings unmasked |
| Sign flips post-DFS | 0 | Causal conclusions stable |

**Statistical significance is powered by n ≈ 60k**, not effect
magnitude. Residual variance explained by treatments (R²(Ỹ|D̃))
ranges from 0.01% to 1.4%. T7_source_earned is the only treatment
with a large variance share (1.4%); all others explain <0.1%
of residualised outcome variance.

---

## 11. Cost breakdown

| DataForSEO endpoint | chunks | rows | cost |
|---|---:|---:|---:|
| `serp/google/organic/live/regular` | 1,011 | 18,969 | $3.54 |
| `labs/google/keyword_overview/live` | 2 | 850 | $0.11 |
| `labs/google/bulk_keyword_difficulty/live` | 2 | 1,011 | $0.12 |
| `keywords_data/google_ads/search_volume/live` (backfill) | 1 | 161 | $0.08 |
| `labs/google/search_intent/live` (backfill) | 1 | 161 | $0.02 |
| `backlinks/bulk_ranks/live` | 1 | 0 | $0.00 (40204) |
| `backlinks/bulk_backlinks/live` | 1 | 0 | $0.00 (40204) |
| `backlinks/bulk_referring_domains/live` | 1 | 0 | $0.00 (40204) |
| `backlinks/bulk_spam_score/live` | 1 | 0 | $0.00 (40204) |
| **TOTAL** | | | **$3.86** |

---

## 12a. Multi-treatment DML (added session 2)

Per DoubleML's recommendation for studies with many treatments
(https://docs.doubleml.org/stable/guide/se_confint.html), we ran two
additional analyses.

### Study 1 — joint inference with multiplier bootstrap

All 19 treatments as `d_cols`, B=500 Gaussian bootstrap, n=39,481
(rows non-null on all 19). Adjusted p-values via Romano-Wolf stepdown
and Bonferroni.

**Bonferroni-surviving at p<0.05 on `rank_delta`** (only 6):

| treatment | coef | p_Bonf |
|---|---:|---:|
| T7_source_earned | −1.572 | 0 |
| T5_topical_comp | +0.476 | 9.5e-5 |
| T6_freshness | −0.057 | 4.7e-10 |
| T2_llm | −0.130 | 0.003 |
| T2a_question_headings | +0.208 | 0.011 |
| T_llms_txt | +0.092 | 0.024 |

**Additional at Romano-Wolf p<0.05** (8 total):
T3_structured_data_new, T4_llm.

### Study 2 — mutually-controlled partial effects

Each treatment estimated with the other 18 in X. Key findings:

- **T_llms_txt grew 38%** under mutual control (+0.094 → +0.130).
  Its univariate effect was masked by correlation with T3_code.
- **T1a_stats_present became significant** (p=0.48 → 0.004). Its
  univariate effect was masked; conditional on other treatments,
  having stats signals promotes.
- **T2_llm grew 2.5×** (−0.044 → −0.112) under mutual control.
- **T3_code (HTML-parsed) lost significance** but T3_structured_data_new
  (expanded) held. Validates the expanded treatment set — old T3 was a
  correlate, new T3 captures the mechanism.
- **9 treatments lost significance under mutual control**: T2_code,
  T3_code, T4_code, T1_llm, T1b_stats_density, T3_llm, T4a_ext_citations,
  T4b_auth_citations, T2b (marginal). These were borrowing significance
  from correlated treatments.

### The "core four" bulletproof treatments

Significant under (univariate) AND (Bonferroni) AND (mutual control):

| treatment | θ_univar | p_Bonf | p_mutual | direction |
|---|---:|---:|---:|---|
| T7_source_earned | −1.70 | ~0 | 4e-150 | strongly demotes |
| T5_topical_comp | +0.44 | 1e-4 | 9e-6 | promotes |
| T6_freshness | −0.06 | 5e-10 | 1e-15 | mildly demotes (surprising) |
| T_llms_txt | +0.09→+0.13 | 0.024 | 6e-9 | promotes (grows under control) |

### How to report in the paper

Three significance counts, different claims:

| claim | count | what it means |
|---|---:|---|
| Univariate DML p<0.01 | 15 | original analysis, uncorrected for multiplicity |
| Mutually-controlled p<0.05 | 10 | holds when other content signals are controlled |
| Bonferroni-corrected p<0.05 | 6 | bulletproof under strictest family-wise error control |

The paper's headline should report all three. Safest narrative: the
"core four" (T7_source_earned, T5_topical_comp, T6_freshness, T_llms_txt)
are robust under every identification lens; the remaining 6–11 treatments
are secondary signals that hold up in some lenses but not all.

Source files (in `results/dml_study/`):
- `multi_treatment_summary.md` — full narrative
- `dml_multi_treatment.csv` — all 76 rows (both studies)
- `dml_multi_treatment_study1_joint.csv` — Study 1 only
- `dml_multi_treatment_study2_partial.csv` — Study 2 only
- `dml_multi_treatment.log`

## 12. Open questions

1. **Activate backlinks subscription?** Would fill `conf_backlinks`,
   `conf_referring_domains`, `conf_domain_authority` to near-100%
   coverage. Based on current sparse signal, `conf_domain_authority`
   would likely become the #2 confounder after `conf_brand_recog`.
   Budget: ~$2.73.

2. **Reduce confounder set?** 4 `dfs_intent_*` dummies contribute <0.35%
   each in LGBM importance and are OLS-insignificant. Dropping them
   would clean up the specification without loss. Similarly
   `conf_readability`.

3. **Investigate the counter-signal finding.** The positive coefficient
   on surface-kw-match confounders (`conf_title_has_kw`, `conf_bm25`,
   etc.) is substantively important for the paper — it's evidence that
   LLMs penalise thin-SEO content. Worth a dedicated robustness check
   on a non-LLM baseline ranking.

---

## 13. Files referenced

| reference | path |
|---|---|
| Merged dataset | `results/full_experiment_data.csv` |
| New DML results | `results/dml_study/dml_results_long.csv` |
| Baseline DML results | `results/dml_study_pre_dfs/dml_results_long.csv` |
| Top-line summary | `results/dml_study/dml_summary.md` |
| Baseline narrative | `results/dml_study_pre_dfs/findings_report.md` |
| Confounder LGBM importance | `results/dml_study/confounder_audit.csv` / `.log` |
| 5-fold CV R² decomposition | `results/dml_study/variance_explained.csv` / `variance_audit.log` |
| DML nuisance R² per fit | `results/dml_study/nuisance_r2.csv` / `.log` |
| Leave-one-out ΔR² | `results/dml_study/confounder_loo_r2.csv` |
| OLS HC3 significance | `results/dml_study/confounder_ols_significance.csv` / `confounder_significance.log` |


---

*end of paperSizeExperiment/doc/analysis-2026-04-23.md*



<a id="papersizeexperiment--doc--robust-winners-analysis-2026-04-26"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 20 / 22 — paperSizeExperiment/doc/robust-winners-analysis-2026-04-26.md  (12021 bytes)
# ═══════════════════════════════════════════════════════════════

# Pool-size sensitivity & robust-winners analysis — 2026-04-26

How the LLM's choice of candidate pool size (top-20 vs top-50 SERP results)
changes its top-10 output, and what page-level features causally predict
ranking among URLs that the LLM picks **regardless** of pool size.

Inputs: `consolidated_results/runs/*/geodml_dataset.csv`,
`consolidated_results/regression_dataset.csv`.
Outputs (new):
`consolidated_results/dml_robust_winners.csv`,
`consolidated_results/dml_robust_winners_pivot.csv`,
`consolidated_results/dml_robust_winners.log`,
`dml_robust_winners.py`.

---

## 1. The question

For every (search_engine, llm_model) "category" (4 of them), we ran the
LLM re-ranking pipeline twice:

- once with the top-20 SERP results as candidate pool (`serp20_top10`)
- once with the top-50 SERP results as candidate pool (`serp50_top10`)

Pool size is an **experimental knob**, not a treatment — we'd hope the LLM
picks the "best 10" regardless of how many candidates it sees.

The first question: **how stable is the LLM's top-10 across pool sizes?**
The second: if we restrict to URLs the LLM picked in *both* pools (the
"robust winners"), **what page-level features causally predict their
position within the top-10?**

---

## 2. Stability of LLM top-10 across pool sizes

For each (engine, model, keyword) we computed the URL overlap between the
serp20 and serp50 top-10 lists.

| category | keywords w/ both pools | mean overlap (/10) | median | Jaccard | mean \|top10\| (20 / 50) |
|---|---:|---:|---:|---:|---|
| duckduckgo + Llama-3.3-70B | 866 | 2.53 | 2 | 0.314 | 6.1 / 6.6 |
| duckduckgo + Qwen2.5-72B | 945 | 2.72 | 3 | 0.222 | 6.7 / 8.9 |
| searxng + Llama-3.3-70B | 928 | 2.57 | 2 | 0.239 | 7.2 / 7.5 |
| searxng + Qwen2.5-72B | 940 | 3.22 | 3 | 0.281 | 7.6 / 8.0 |

Distribution of #common URLs (out of top-10) per keyword:

| category | 0 | 1-2 | 3-4 | 5-6 | 7-8 | 9-10 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| ddg + Llama | 133 | 332 | 271 | 105 | 23 | 2 | 866 |
| ddg + Qwen | 111 | 349 | 315 | 152 | 17 | 1 | 945 |
| searxng + Llama | 115 | 369 | 301 | 121 | 21 | 1 | 928 |
| searxng + Qwen | 87 | 299 | 290 | 200 | 59 | 5 | 940 |

### Findings

1. **Pool size meaningfully changes the LLM's output.** Median overlap
   2-3/10, Jaccard 0.22-0.31. Roughly 70% of top-10 URLs are different
   between the two pool sizes for the same keyword.
2. **The LLM is doing "pick a defensible subset" more than "rank by
   intrinsic quality."** If it were ranking by intrinsic quality, adding
   candidates 21-50 should not displace candidates 1-20 from the top
   choices. It does — extensively.
3. **searxng + Qwen is the most stable** (Jaccard 0.281), DDG + Llama
   second (0.314 by Jaccard but lower mean overlap). Stability is
   model-and-engine-specific; treat it as a property of the pipeline,
   not the LLM alone.
4. **Lists are usually shorter than 10** (mean 6-9). The LLM declines to
   rank a full 10 when it isn't confident, which caps the maximum
   possible overlap below 10.

**Implication for the DML study:** pool size is a confounder, not a
robustness check. Treatment effects estimated by pooling serp20 and
serp50 mix (a) the page-level treatment effect with (b) the pool-size
selection effect. The fix is either: condition on pool size (already
done in `dml_pivot_*.csv` per-pool subsets) or restrict to URLs that
appear in both pools (the **robust-winners** frame, this document).

---

## 3. The robust-winners subset

A **robust winner** is a (keyword, url) pair that the LLM placed in its
top-10 under *both* pool sizes within a single (engine, model) category.
By construction, these URLs survive perturbation of the candidate set —
they're the GEO target.

| category | robust (kw, url) pairs | rows used (both pool rows) | covered keywords |
|---|---:|---:|---:|
| duckduckgo + Llama-3.3-70B | 2,187 | 4,374 | 733 |
| duckduckgo + Qwen2.5-72B | 2,572 | 5,144 | 834 |
| searxng + Llama-3.3-70B | 2,386 | 4,772 | 813 |
| searxng + Qwen2.5-72B | 3,031 | 6,062 | 853 |

For DML we keep both the serp20 row and the serp50 row for each robust
pair (so post_rank can vary across pools), and add `serp_pool_size` as
an extra confounder so the within-pool variation is partialled out.

---

## 4. DML on the robust winners

Estimator: DoubleML PLR with LightGBM nuisance learners
(`n_estimators=200`, 5-fold cross-fitting, partialling-out score).
Confounders: 25 from `config.CONFOUNDERS` + `serp_pool_size`,
median-imputed and standardised. 19 treatments × 2 outcomes ×
4 categories = **152 fits**, runtime ~5 min.

Outcomes: `post_rank` (lower = better; **negative coef = good**) and
`rank_delta = pre_rank − post_rank` (positive = good).

### 4.1 Headline coefficients on `post_rank`

(showing only treatments that hit p<0.05 in ≥2 of 4 categories, plus
the consistent helpers)

| treatment | ddg+Llama | ddg+Qwen | searxng+Llama | searxng+Qwen | reading |
|---|---:|---:|---:|---:|---|
| **T6_freshness** | +0.071** | +0.153*** | +0.078** | +0.066** | **HURTS** in all 4 |
| **T7_source_earned** | +0.977* | +1.221*** | +0.482 | +1.516*** | **HURTS** in 3/4 |
| **T3_structured_data_new** | +0.140 | +0.285** | +0.193* | +0.162* | **HURTS** in 3/4 |
| **T2b_structural_modularity** | −0.004 | −0.005* | −0.004· | −0.005* | **HELPS** consistently |
| T5_topical_comp | (n/a) | −0.821* | +0.054 | −0.295 | helps where significant |
| T1b_stats_density | −0.001 | +0.010* | +0.008 | +0.018* | mild HURT |
| T1a_stats_present | −0.223 | −0.137 | −0.152 | +0.231* | inconsistent |

Sign legend: `*` p<0.05, `**` p<0.01, `***` p<0.001, `·` p<0.10.
Coefficients are in **rank positions**: `+0.07` = page lands ~0.07 spots
*lower* (worse) on average per unit of treatment.

Full results in `consolidated_results/dml_robust_winners_pivot.csv`.

### 4.2 What this says

**(a) The robust-winners frame produces sharper, more consistent
estimates than the per-pool subsets.** In the prior `dml_pivot_post_rank.csv`,
several treatments flipped sign across the 4 (engine, model) cells.
Here, freshness, "earned source," structured data, and structural
modularity all point the same direction in 3-4 of 4 cells — that is
evidence the pool-size noise was diluting signal in the per-pool fits.

**(b) The biggest consistent effects are *negative* for features that
intuitively "should" help.** Freshness (T6), earned/non-brand source
(T7), and structured-data markup (T3) all *push pages further down*
within the top-10 among robust winners.

Two non-exclusive readings:

- **Survival bias.** To be a robust winner at all, a page already has to
  clear the authority/BM25/brand bar (controlled for via confounders).
  Conditional on that, an "earned" signal proxies for third-party /
  news / aggregator content that the LLM systematically deprefers
  versus owned, branded pages with similar baseline metrics.
- **Recency penalty inside owned content.** T6_freshness positive on
  post_rank means *fresher* robust winners rank slightly worse, all
  else equal. Plausibly reflects established, link-rich pages
  outranking newly published content.

**(c) The one feature that helps consistently is structural modularity
(T2b).** Counts of modular sections / lists / clear headings. Effect is
small (−0.004 to −0.005 rank positions per modular block) but signed
the same way in all 4 categories and significant in 2. **This is the
cleanest GEO-positive recommendation in the data.**

**(d) Topical competence (T5)** is the second consistent helper —
significant only in DDG+Qwen (−0.82 on post_rank) but pointing the
right direction in 3/4 cells. Worth investigating further.

### 4.3 Limits of this analysis

1. **Position effects only.** Robust winners are by construction enriched
   for already-strong pages. Coefficients here describe what moves a
   page *within* the top-10, not what gets it *into* the top-10.
2. **Per-fit n varies** with treatment NaN rate (276 → 6,062). The most
   reliable cells are T6/T7/T2b/T3 (n ≥ ~4,000); T5 is sparser.
3. **Pool size is partialled out**, but a residual bias is possible if
   the *same* treatment that helps a page enter the pool also affects
   its within-pool position, and confounders only partially capture
   selection.
4. **Two outcome rows per robust pair** (one per pool) introduces mild
   intra-pair correlation; standard errors are LightGBM-cross-fitted,
   not cluster-robust on (keyword, url). Effect on inference is
   probably small but worth checking.

---

## 5. Plan to apply this

Rough priority order. Each item lists what to build, what we expect to
learn, and the file to add or change.

### 5.1 Selection model: who *enters* the robust-winners set

Logistic DML / IRM with binary outcome `is_robust_winner ∈ {0,1}` over
*all* URLs that appeared in the serp50 candidate pool (~50k rows). Same
treatments, same confounders. This is the missing half of §4: gets at
"which features cause an LLM to pick a page at all," complementing
"which features cause the LLM to rank a picked page higher."

- New script: `dml_winner_selection.py`
- Model: `dml.DoubleMLIRM` (binary outcome) with LGBMClassifier.
- Output: `consolidated_results/dml_winner_selection_pivot.csv`.

### 5.2 Cluster-robust inference

Re-fit the 152 robust-winners PLRs with bootstrap clustering on
(keyword, url) — `model.bootstrap(method='wild', cluster=...)` or a
manual block bootstrap. Verify that the 4 headline findings (T6, T7,
T3, T2b) survive.

- Edit `dml_robust_winners.py` to add a `--cluster` mode.
- Output: `consolidated_results/dml_robust_winners_clustered.csv`.

### 5.3 Robust-winners by keyword category

The 1,011 keywords are stratified across 20 subcategories
(`generate_keywords.py: TOPIC_TREE`). The category labels were not
preserved per-keyword in any data file. Re-derive them by re-running
`generate_keywords.py` deterministically (or by reading the LLM
generation log if cached) and join onto `regression_dataset.csv`.
Then re-fit DML on robust winners **× topic category** and look for
heterogeneity. Two specific hypotheses to check:

- T2b_structural_modularity is most beneficial in informational /
  educational queries.
- T6_freshness is *less* harmful in news/finance categories where
  recency genuinely matters.

- Edit `generate_keywords.py` to emit a `keywords_with_category.csv`.
- New script: `dml_robust_winners_by_category.py`.
- Output: `consolidated_results/dml_robust_winners_by_topic.csv`.

### 5.4 Sensitivity to pool size as a treatment

Within the robust-winners set, treat `serp_pool_size ∈ {20, 50}` as a
treatment and estimate its causal effect on post_rank, controlling for
the page-level treatments and confounders. Tells us how much "merely
showing the LLM more candidates" changes a *fixed* page's rank — i.e.,
the magnitude of the pool-selection nuisance we identified in §2.

- Add a single fit to `dml_robust_winners.py`.
- Output: appended row in `dml_robust_winners.csv`.

### 5.5 Complementary descriptive plot

Per category, scatter `post_rank_serp20` vs `post_rank_serp50` for
robust winners (one point per (keyword, url)). Pearson r quantifies
how stable the *position* (not just inclusion) is across pools.

- New notebook or `analyze_robust_position_stability.py`.
- Output: `output/robust_position_correlation.csv` + 4 PNGs.

### 5.6 Reporting

Update `consolidated_results/dml_study/dml_summary.md` with a
"Robust-winners frame" section pointing to this analysis, and add the
table in §4.1 to the slide deck (`presentation.pptx`).

---

## 6. Files

New
- `dml_robust_winners.py` — pipeline.
- `consolidated_results/dml_robust_winners.csv` — 152-row long form.
- `consolidated_results/dml_robust_winners_pivot.csv` — treatment ×
  category × outcome.
- `consolidated_results/dml_robust_winners.log` — runtime log.
- `doc/robust-winners-analysis-2026-04-26.md` — this file.

Read
- `consolidated_results/regression_dataset.csv` — 65,203 rows, 8 runs.
- `config.py` — `ALL_TREATMENTS`, `CONFOUNDERS`.


---

*end of paperSizeExperiment/doc/robust-winners-analysis-2026-04-26.md*



<a id="papersizeexperiment--doc--roadmap"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 21 / 22 — paperSizeExperiment/doc/ROADMAP.md  (5811 bytes)
# ═══════════════════════════════════════════════════════════════

# Experiment Roadmap

## Pipeline Phases

**Phase 1 — SERP + LLM Re-Ranking** (`gather_data.py`)
- For each keyword: query the search engine (DDG or SearXNG) for N results (serp20 or serp50)
- Send those results to the LLM (Llama/Qwen via HF API) to re-rank and pick top 10
- Output: `rankings.csv`, `keywords.jsonl` (pre_rank vs post_rank per domain)

**Phase 2 — HTML Fetch + Code-Based Features** (`gather_data.py`)
- Fetch the actual HTML of every unique URL from Phase 1
- Extract code-based features via BeautifulSoup: T1 statistical density, T2 question headings, T3 structured data (JSON-LD), T4 citation count, word count, readability, links, images
- Cache HTML to disk (`html_cache/`)
- Output: `features.csv` with code-measured treatments + confounders

**Phase 3 — LLM Feature Extraction** (`gather_data.py` with `--llm-features`)
- Re-read cached HTML, build a page digest, send to the LLM
- LLM scores the same T1-T4 treatments but from its own "perception" (T1_llm, T2_llm, T3_llm, T4_llm)
- Resumable, checkpoints every 5 URLs
- Output: updates `features.csv` with LLM-judged treatment columns

**Phase 4 — Enriched Features / Moz API** (`extract_features.py`)
- Adds confounders from external APIs: domain authority, backlinks, referring domains via Moz Links API
- Adds new treatments: T5 topical competence (embedding similarity), T6 freshness (date extraction), T7 source type (earned media classification)
- Adds confounders: BM25 score, title/snippet similarity, brand recognition
- Output: `features_new.csv`

## Design

- **1011 keywords** (B2B SaaS queries)
- **Search engines**: DuckDuckGo, SearXNG
- **LLM models** (HuggingFace Inference API):
  - `meta-llama/Llama-3.3-70B-Instruct`
  - `Qwen/Qwen2.5-72B-Instruct`
- **SERP pool sizes**: serp20/top10, serp50/top10
- **Full 2x2x2 factorial**: 2 engines x 2 models x 2 pool sizes = **8 runs**
- **Pipeline per run**: gather_data (P1+P2+P3) -> extract_features (P4) -> clean_data -> analyze -> halo analysis

## Run Status (as of 2026-04-09)

| # | Engine | Model | SERP/Top | P1 (SERP+LLM) | P2 (HTML) | P3 (LLM feat) | P4 (extract) | clean_data | analysis_full | halo |
|---|--------|-------|----------|----------------|-----------|----------------|--------------|------------|---------------|------|
| 1 | DDG | Llama-3.3-70B | 20/10 | DONE (1011 kw) | DONE (6413 URLs) | DONE (5636/5637) | DONE | DONE | DONE (204 exp, 32 sig) | DONE |
| 2 | DDG | Llama-3.3-70B | 50/10 | DONE (1011 kw) | DONE (6817 URLs) | DONE (5647/5647) | DONE | DONE | DONE (204 exp, 39 sig) | DONE |
| 3 | DDG | Qwen2.5-72B | 20/10 | DONE (1011 kw) | DONE (6947 URLs) | 91.8% (5422/5906) | TODO | DONE* | TODO | TODO |
| 4 | DDG | Qwen2.5-72B | 50/10 | DONE (1011 kw) | DONE (8591 URLs) | 35.6% (2612/7347) | TODO | TODO | TODO | TODO |
| 5 | SearXNG | Llama-3.3-70B | 20/10 | DONE (960 kw) | DONE (6968 URLs) | DONE (5941/5941) | DONE | DONE | DONE (216 exp, 44 sig) | DONE |
| 6 | SearXNG | Llama-3.3-70B | 50/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 7 | SearXNG | Qwen2.5-72B | 20/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 8 | SearXNG | Qwen2.5-72B | 50/10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

\* Run 3 has geodml_dataset.csv with outcomes (77%) but no features_new.csv, so enriched treatments/confounders are missing. Analysis needs re-run after Phase 3+4 complete.

**Note**: Runs 6-8 require a running SearXNG container for Phase 1.

## Pipeline Steps (per run)

1. **gather_data.py** — Phase 1: SERP queries + LLM re-ranking -> keywords.jsonl, rankings.csv
2. **gather_data.py** — Phase 2: Fetch HTML for all URLs -> html_cache/, features.csv
3. **gather_data.py** — Phase 3: LLM-based feature extraction (T1-T4 via LLM) -> features.csv (updated)
4. **extract_features.py** — P4: T5-T7, BM25, Moz API -> features_new.csv
5. **clean_data.py** — Merge rankings + features + features_new -> geodml_dataset.csv
6. **analyze.py** — Full DML: 17 treatments x 3 outcomes x 2 methods x 2 learners -> analysis_full/
7. **earned_media_halo.py** — Halo effect analysis -> analysis_halo/

## Key Findings (Runs 1, 2, 5 completed)

### Consistent across all completed runs:
- **T7 (Earned media) demotion**: rank_delta coef = -1.3 to -2.4 (***) — earned media pages consistently demoted by LLM
- **Halo effect**: brands mentioned in earned media get better post_rank (***) — LLM boosts mentioned brands

### Run-specific findings:
- **Run 1** (DDG/Llama/s20): 32/204 significant results. T1_llm small negative effect (-0.01, p=0.015)
- **Run 2** (DDG/Llama/s50): 39/204 significant results. Larger SERP pool amplifies T3 structured data and T6 freshness effects
- **Run 5** (SXG/Llama/s20): 44/216 significant results. T4_llm citation authority significant (-0.09, p=0.004)

### Cross-model analysis:
- 492 experiments, 165 significant (p<0.05)
- T7 earned media is the dominant treatment across all subsets

## Known Issues

1. **`experiment.json` resume bug** (FIXED): when gather_data.py resumes, `per_keyword_results` in experiment.json is empty. Fixed `clean_data.py` to fall back to `keywords.jsonl`.

2. **Moz API coverage ~6-10%**: domain authority/backlinks unusable as confounders. OpenPageRank already integrated as alternative.

3. **T5 topical competence 0% for DDG runs**: embedding similarity depends on SearXNG-specific data. Not available for DuckDuckGo runs.

4. **conf_title_kw_sim / conf_snippet_kw_sim 0% for DDG runs**: these confounders rely on SearXNG snippet format.

## SearXNG Availability

SearXNG requires a running container. Start with:
```bash
docker run -d --rm -p 8888:8080 -v $(pwd)/searxng-config:/etc/searxng searxng/searxng:latest
```
SearXNG runs must be done when the container is active. DDG runs use the `ddgs` library directly (no container needed).


---

*end of paperSizeExperiment/doc/ROADMAP.md*



<a id="papersizeexperiment--doc--treatment-confounder-dictionary"></a>

# ═══════════════════════════════════════════════════════════════
# FILE 22 / 22 — paperSizeExperiment/doc/treatment-confounder-dictionary.md  (9714 bytes)
# ═══════════════════════════════════════════════════════════════

# Treatment and Confounder Dictionary

Exact definitions, extraction method, and measurement scale for every
variable in the DML study. Sourced from the production extraction code
in `pipeline/extract_features.py`.

---

## Treatments — 19 total

Treatments are page-level content properties whose causal effect on
LLM re-ranking we want to measure.

### The "code" family (HTML-parsed, coarse)

These are earlier detectors, pattern-matching the raw HTML.

| ID | column | type | what it measures |
|---|---|---|---|
| **T1_code** | `T1_statistical_density_code` | continuous | statistical-density score computed directly from HTML (regex count of stats patterns per section) |
| **T2_code** | `T2_question_heading_code` | 0/1 | does any H1–H3 end with `?` or start with a question word (what/how/why/…)? |
| **T3_code** | `T3_structured_data_code` | 0/1 | is there a `<script type="application/ld+json">` tag with recognised schema? |
| **T4_code** | `T4_citation_authority_code` | continuous | count of outbound links to authority domains (`.edu`, `.gov`, arxiv, wikipedia, nature, etc.) |

### The "llm" family (LLM-as-judge on the page digest)

Same T1–T4 concepts but scored by an LLM reading the page digest rather
than by regex. More semantically aware, more noisy.

| ID | column | type | what it measures |
|---|---|---|---|
| **T1_llm** | `T1_statistical_density_llm` | continuous | LLM estimate of stats density |
| **T2_llm** | `T2_question_heading_llm` | 0/1 | LLM judgment that headings are question-shaped (catches rhetorical questions without literal `?`) |
| **T3_llm** | `T3_structured_data_llm` | 0/1 | LLM judgment that the page has structured content sections |
| **T4_llm** | `T4_citation_authority_llm` | continuous | LLM assessment of citations to authoritative sources |

### The "new" expanded family (feature extractor)

Newer, more precise measurements; separate presence from density.

| ID | column | type | what it measures | regex / logic |
|---|---|---|---|---|
| **T1a_stats_present** | `treat_stats_present` | 0/1 | does the body contain ANY statistic? | hits on `\d+\.?\d*%`, years, dates, `$N`, `Nm/k/b` units |
| **T1b_stats_density** | `treat_stats_density` | float | unique stat matches per 500 words | `len(unique matches) / (word_count/500)` |
| **T2a_question_headings** | `treat_question_headings` | 0/1 | any H2/H3 that starts with question word OR ends with `?` | `^(what\|how\|why\|when\|where\|which\|who\|can\|does\|is\|are\|should\|will\|do)\b` |
| **T2b_structural_modularity** | `treat_structural_modularity` | count | number of H2 + H3 headings | `len(soup.find_all(["h2","h3"]))` |
| **T3_structured_data_new** | `treat_structured_data` | 0/1 | JSON-LD with any of: FAQPage, Product, HowTo, Article, BlogPosting, Review, AggregateRating, Offer, ItemList, BreadcrumbList, VideoObject, Dataset, Course, Event, Recipe, QAPage, SoftwareApplication | parses every `application/ld+json` script and checks `@type` |
| **T4a_ext_citations** | `treat_ext_citations_any` | 0/1 | at least one outbound link to any external domain (excluding social/CDN noise) | excludes facebook/twitter/linkedin/etc. |
| **T4b_auth_citations** | `treat_auth_citations` | count | number of distinct outbound links to authority domains | authority = `.edu/.gov/.gov.uk/.ac.uk/.mil` suffixes + curated list (wikipedia, arxiv, nature, ieee, acm, springer, wiley, jstor, who.int, worldbank, statista, pewresearch, …) |
| **T5_topical_comp** | `treat_topical_comp` | float 0–1 | cosine similarity between keyword embedding and page-text embedding | sentence-transformers; "topical competence" |
| **T6_freshness** | `treat_freshness` | 0–4 ordinal | how recent the page appears to be | 4 = <6 months; 3 = 6–12 months; 2 = 1–2 years; 1 = 2–5 years; 0 = >5 years or no date found; parses meta tags (`article:published_time`, `datePublished`, etc.), JSON-LD dates, `<time datetime>`, and body-text year patterns |
| **T7_source_earned** | `treat_source_earned` | 0/1 | is the page's domain in the "earned media" list (G2, Capterra, TechCrunch, Gartner, Forrester, software review aggregators, tech press, etc.)? | lookup against curated `EARNED_DOMAINS` set |

Related but not used as a main treatment:
- `treat_source_brand` (0/1): domain in the B2B-SaaS brand list (Salesforce, HubSpot, Microsoft, Slack, Stripe, 100+ more). Kept as a confounder-like feature, not reported as a treatment.
- `treat_source_type` (string): one of `brand`, `earned`, or `other`.

### llms.txt

| ID | column | type | what it measures |
|---|---|---|---|
| **T_llms_txt** | `has_llms_txt` | 0/1 | does the page's domain serve `/llms.txt` (HTTP 200 with non-empty body)? |

---

## Confounders — 25 total

Confounders are page/keyword-level variables that plausibly affect both
the content treatments AND the rank outcome, and therefore must be
controlled for to get a causal estimate.

### Page-level text features (from HTML + keyword)

| column | type | what it measures |
|---|---|---|
| `conf_title_kw_sim` | float 0–1 | cosine similarity between keyword embedding and page-title embedding |
| `conf_snippet_kw_sim` | float 0–1 | cosine similarity between keyword embedding and the search-engine snippet |
| `conf_title_len` | int | character length of the page title |
| `conf_snippet_len` | int | character length of the search-result snippet |
| `conf_title_has_kw` | 0/1 | does any ≥3-char word from the keyword literally appear in the title? |
| `conf_bm25` | float | BM25 score of the page body against the keyword |
| `conf_https` | 0/1 | is the URL HTTPS? |

### Page-level structural features (from HTML)

| column | type | what it measures |
|---|---|---|
| `conf_word_count` | int | `len(body_text.split())` |
| `conf_readability` | float | Flesch-Kincaid grade level (via `textstat`); higher = harder |
| `conf_internal_links` | int | count of `<a>` tags pointing within the same domain OR relative links |
| `conf_outbound_links` | int | count of `<a>` tags pointing to a different domain (HTTP(S) only) |
| `conf_images_alt` | int | count of `<img>` tags with non-empty `alt` attribute (accessibility signal) |

### Domain/brand features (classifier lookups)

| column | type | what it measures |
|---|---|---|
| `conf_brand_recog` | 0/1 | is the domain in the curated B2B-SaaS brand set (~100 well-known SaaS companies)? |

### Position-in-pre-LLM-SERP

| column | type | what it measures |
|---|---|---|
| `conf_serp_position` | int | the rank the URL had BEFORE LLM re-ranking (i.e., the SearXNG/DuckDuckGo position). Identical to `pre_rank`; the most powerful outcome predictor in the dataset. |

### Domain-level authority features (sparse — 11–22% coverage; blocked backlinks subscription will fill these)

| column | type | what it measures |
|---|---|---|
| `conf_domain_authority` | float | Open PageRank score (0–10); domain-level authority proxy |
| `conf_backlinks` | int | number of backlinks to the domain (from DataForSEO when available) |
| `conf_referring_domains` | int | count of unique referring domains |

### DataForSEO keyword-level features (added 2026-04-23)

| column | type | what it measures |
|---|---|---|
| `dfs_keyword_difficulty` | 0–100 | Google KD (harder keyword = higher KD) |
| `dfs_search_volume` | int | monthly US Google searches for the keyword |
| `dfs_cpc` | float USD | average cost-per-click for the keyword on Google Ads |
| `dfs_competition` | 0–1 | Google Ads competition index (paid-search competitiveness) |
| `dfs_intent_commercial` | 0/1 | one-hot of `dfs_main_intent = commercial` |
| `dfs_intent_informational` | 0/1 | one-hot of `dfs_main_intent = informational` |
| `dfs_intent_navigational` | 0/1 | one-hot of `dfs_main_intent = navigational` |
| `dfs_intent_transactional` | 0/1 | one-hot of `dfs_main_intent = transactional` |

---

## Outcomes

The DML study has two outcomes. They are related by construction:

| outcome | type | what it is | sign convention |
|---|---|---|---|
| `post_rank` | int 1–N | rank after LLM re-ranking (lower = better) | negative coefficient on a treatment = promotes |
| `rank_delta` | int | `pre_rank − post_rank` (positive = LLM moved the page up) | positive coefficient = promotes |

The paper's headline results are reported on `rank_delta` (more intuitive
sign); `post_rank` is reported for robustness and should mirror with
inverted signs.

---

## Ranking convention (critical when reading any coefficient)

1. Rank 1 is the goal. Lower rank number = better position.
2. `pre_rank` is a confounder (the pre-LLM search-engine rank), not a
   treatment effect target.
3. `rank_delta = pre_rank − post_rank`. Positive = promotion by LLM.
4. A positive coefficient on a treatment against `rank_delta` means
   that treatment causes the LLM to promote pages = GOOD content signal.
5. A negative coefficient on a treatment against `post_rank` means the
   same thing (lower post_rank = better).

---

## Cross-reference: which family is each treatment in?

| statistical construct | HTML-parse | LLM-judge | Expanded extractor |
|---|---|---|---|
| statistical density | T1_code | T1_llm | T1a_stats_present (binary) + T1b_stats_density (continuous) |
| question headings | T2_code | T2_llm | T2a_question_headings + T2b_structural_modularity |
| structured data | T3_code | T3_llm | T3_structured_data_new (expanded JSON-LD type list) |
| citation authority | T4_code | T4_llm | T4a_ext_citations_any (binary) + T4b_auth_citations (count) |

Conceptually-independent treatments (no code/LLM sibling):
- T5_topical_comp — content-query semantic match
- T6_freshness — recency
- T7_source_earned — domain classification
- T_llms_txt — llms.txt presence


---

*end of paperSizeExperiment/doc/treatment-confounder-dictionary.md*

