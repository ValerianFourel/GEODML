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
