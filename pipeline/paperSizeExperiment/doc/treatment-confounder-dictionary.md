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
