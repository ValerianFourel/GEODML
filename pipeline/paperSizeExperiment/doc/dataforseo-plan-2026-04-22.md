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
