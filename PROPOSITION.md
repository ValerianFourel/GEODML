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
