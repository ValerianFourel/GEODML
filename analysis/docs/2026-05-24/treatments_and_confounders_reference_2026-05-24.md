# Treatments and confounders — full reference (2026-05-24)

Complete documentation of every treatment and confounder used in the GEODML
EMNLP-2026 pipeline. Cross-references the actual extractor source code in
`interpretability/pipeline/features.py` and the upstream pipeline
`~/Hamburg/GEODML/pipeline/`, plus the data parquets in `~/geodml_data/`.

For each variable: **what it measures**, **how it's computed**, **what data it depends on**, **scale**, and any **known caveats**.

---

## OUTCOMES

| Column | What it is | Computed from | Scale | Notes |
|---|---|---|---|---|
| `pre_rank` | Original SERP rank position | DuckDuckGo / SearXNG raw output | int 1–N | Before LLM rerank; "input" rank |
| `post_rank` | LLM-assigned rank in its top-N output | The LLM rerank step | int 1–10 | NaN if LLM did NOT include the URL |
| `rank_delta` | `pre_rank − post_rank` | computation | rank positions | Positive = LLM moved doc UP. Defined only for admitted URLs |
| `selected_by_llm` | Binary admission indicator | `post_rank.notna()` | 0 / 1 | 1 if URL is in the LLM's top-N; 0 if rejected. **Requires the SERP pool sample frame** (see unified README) |

---

## TREATMENTS — content/structural signals (extracted from page HTML)

All defined in `interpretability/pipeline/features.py`; the upstream version is `~/Hamburg/GEODML/pipeline/extract_features.py`.

### T1 — Statistical content

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_stats_present` (T1a) | 1 if page contains ANY statistic | Regex match against page body text | binary |
| `treat_stats_density` (T1b) | Statistics per 500 words | (# unique stat matches) ÷ (word_count ÷ 500) | float (rounded to 2 d.p.) |

**`_STAT_PATTERNS`** (regex defs in features.py:203):
```python
re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")   # 1,234 or 1,234,567.89
re.compile(r"\b\d+\.?\d*%")                        # 50% or 12.5%
re.compile(r"\b(?:19|20)\d{2}\b")                  # years 1900-2099
re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b") # 12/31/2023 or 12-31-23
re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")         # $100 or $1,234.56
re.compile(r"\b\d+(?:\.\d+)?[BMKbmk]\b")             # 1B, 500M, 2.5K
```

### T2 — Structural headings

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_question_headings` (T2a) | 1 if any `<h2>`/`<h3>` is a question | Heading text matches `_QUESTION_RE` or ends in "?" | binary |
| `treat_structural_modularity` (T2b) | Count of `<h2>` + `<h3>` tags | `len(soup.find_all(['h2','h3']))` | int (count) |

**`_QUESTION_RE`** (features.py:212): `^\s*(?:what|how|why|when|where|which|who|can|does|is|are|should|will|do)\b` (case-insensitive)

### T3 — Structured data / schema markup

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_structured_data` (T3) | 1 if page has JSON-LD with a recognized `@type` | parse `<script type="application/ld+json">`; recursive `_check_ld_type` | binary |

**Recognized `@type` values** (`STRUCTURED_DATA_TYPES`, 18 of them):
`faqpage`, `faq`, `product`, `howto`, `softwareapplication`, `article`, `blogposting`, `review`, `aggregaterating`, `offer`, `itemlist`, `breadcrumblist`, `videoobject`, `dataset`, `course`, `event`, `recipe`, `qapage`.

### T4 — Citations

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_ext_citations_any` (T4a) | 1 if page has any outbound link to a non-social-media, non-CDN domain | `<a href>` loop, excluding `page_domain` and `LINK_FILTER_DOMAINS` | binary |
| `treat_auth_citations` (T4b) | Count of outbound links to authoritative TLDs or curated authority list | `<a href>` loop, hostname's suffix in `AUTHORITY_SUFFIXES` OR registrable domain in `AUTHORITY_DOMAINS` | int (count) |

**`AUTHORITY_SUFFIXES`**: `{"edu", "gov", "gov.uk", "ac.uk", "mil"}`

**`AUTHORITY_DOMAINS`** (22 curated): wikipedia.org, scholar.google.com, ncbi.nlm.nih.gov, arxiv.org, nature.com, sciencedirect.com, ieee.org, acm.org, researchgate.net, pubmed.ncbi.nlm.nih.gov, springer.com, wiley.com, jstor.org, ssrn.com, nber.org, who.int, un.org, worldbank.org, statista.com, pewresearch.org, gallup.com

**`LINK_FILTER_DOMAINS`** (excluded from "outbound" counting): facebook.com, twitter.com, x.com, linkedin.com, instagram.com, pinterest.com, tiktok.com, youtube.com, apple.com, play.google.com, apps.apple.com, plus 5 CDN domains.

### T5 — Topical completeness

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_topical_comp` (T5) | Cosine similarity between page body and search keyword | `cosine_sim(embed(body), embed(keyword))` via sentence-transformers (`features.py:533`) | float ∈ [−1, 1] |

Model: sentence-transformers `all-MiniLM-L6-v2` (or pipeline-default). The page body is the BeautifulSoup-extracted text; the keyword is the literal SERP query.

### T6 — Freshness

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_freshness` (T6) | Ordinal 0–4 based on most-recent dated signal on the page | Parse `<meta name=date|published|modified|time content=…>`, JSON-LD `datePublished/dateModified/dateCreated`, `<time datetime=…>`, fallback regex over first 5k chars of body | int 0–4 |

**Bucket thresholds** (features.py:441–449):
- `age_days ≤ 180` → **4** (freshest)
- `≤ 365` → 3
- `≤ 730` → 2
- `≤ 1825` → 1
- otherwise → **0** (stale or no date found)

### T7 — Source: earned-media list (DESCRIPTIVE, NOT CAUSAL)

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `treat_source_earned` (T7) | 1 if domain is in `EARNED_DOMAINS` hardcoded set | Set membership lookup | binary |

**`EARNED_DOMAINS`** (~250 hand-curated domains) — categories:
- SaaS review sites: g2.com, capterra.com, trustradius.com, softwareadvice.com, getapp.com, gartner.com, forrester.com, idc.com, peerspot.com, sourceforge.net, crozdesk.com, financesonline.com, goodfirms.co, trustpilot.com, alternativeto.net, softwaresuggest.com, technologyadvice.com, saashub.com, clutch.co, stackshare.io, etc.
- Tech media: techcrunch.com, venturebeat.com, zdnet.com, techradar.com, pcmag.com, cnet.com, tomsguide.com, theverge.com, wired.com, arstechnica.com, infoworld.com, computerworld.com, engadget.com, gizmodo.com, mashable.com, thenextweb.com, digitaltrends.com, fastcompany.com, etc.
- Business media: forbes.com, businessinsider.com, entrepreneur.com, inc.com, nytimes.com, wsj.com, bloomberg.com, reuters.com, fortune.com, economist.com, etc.
- Marketing media: adweek.com, marketwatch.com, cnbc.com, ft.com, adage.com, digiday.com, marketingdive.com, etc.
- Security media: darkreading.com, securityweek.com, thehackernews.com, etc.
- Analyst/consulting: hbr.org, mckinsey.com, bain.com, bcg.com, deloitte.com, accenture.com, pwc.com, kpmg.com, ey.com, etc.
- Wikipedia, reddit, quora, stackexchange, etc.

**⚠ Why T7 is descriptive, not causal:** it's a *list-membership flag*, not a content feature you can manipulate. A reviewer can rightly ask "what content property is T7 measuring?"; the honest answer is "this domain is on our curated earned-media list" — circular. Used for descriptive reporting (the LLM systematically excludes domains on this list), not as a clean causal treatment.

### T_llms_txt — agent-readable site declaration

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `has_llms_txt` (T_llms_txt) | 1 if domain serves a `/llms.txt` or `/.well-known/llms.txt` file | HTTP HEAD/GET probe at fetch time | binary |

Sourced from `regression_dataset.parquet` (joined into the unified file).

### T_code / T_llm — parallel re-codings (for inter-annotator audit)

For T1–T4, two parallel coders exist:

| Family | Code-derived (`*_code`) | LLM-coded (`*_llm`) | Code/LLM agreement |
|---|---|---|---|
| T1 stats density | `T1_statistical_density_code` | `T1_statistical_density_llm` | Pearson r=0.19, weak |
| T2 Q-headings | `T2_question_heading_code` | `T2_question_heading_llm` | Cohen's κ=0.665, substantial |
| T3 structured data | `T3_structured_data_code` | `T3_structured_data_llm` | κ=0.535, moderate; LLM ≈ 2× more permissive |
| T4 citation authority | `T4_citation_authority_code` | `T4_citation_authority_llm` | Pearson r=0.56, moderate |

The `*_code` versions use simpler regex/rule logic than the refined `treat_*` versions above (which are the "new code" coding). The `*_llm` versions are LLM-judged (a separate annotator step).

DML survivor analysis (`docs/dml_survivors_2026-05-24.md`) found that:
- T2a (code) and T2_llm BOTH survive Romano-Wolf with OPPOSITE signs — they extract orthogonal information.
- T3_structured_data_new (code) survives RW; T3_llm is only marginal — the code version wins despite the LLM finding more positives.
- T4_llm survives RW; T4a/T4b code variants do NOT — LLM coding wins here.

---

## CONFOUNDERS — page-HTML extracted

All defined in `features.py`; the page body comes from BeautifulSoup-parsed downloaded HTML.

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `conf_word_count` | Word count of page body | `len(body_text.split())` | int |
| `conf_readability` | Flesch readability score (or similar) | `extract_readability(body_text)` | float |
| `conf_internal_links` | Count of same-domain `<a>` tags | `<a href>` loop, hostname matches page_domain | int |
| `conf_outbound_links` | Count of different-domain `<a>` tags | `<a href>` loop, hostname ≠ page_domain | int |
| `conf_images_alt` | Count of `<img>` with non-empty `alt=` | `<img>` loop with alt-text filter | int |
| `conf_https` | 1 if URL starts with `https://` | `url.startswith("https://")` | binary |

---

## CONFOUNDERS — SERP-snippet derived

Computed from search-engine output (title + snippet + position fields).

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `conf_title_has_kw` | 1 if SERP title contains any keyword word ≥ 3 chars | `features.py:conf_title_has_kw` (case-insensitive) | binary |
| `conf_title_len` | Character length of SERP title | `len(title)` | int (chars) |
| `conf_snippet_len` | Character length of SERP snippet | `len(snippet)` | int (chars) |
| `conf_serp_position` | Original SERP rank position | from DDG/SearXNG raw output | int 1–N |

**`conf_serp_position` is mechanically explanatory** for `rank_delta` (since rank_delta = pre_rank − post_rank), so it's typically excluded from rank-side confounder rankings (e.g., fig 13 panel b).

---

## CONFOUNDERS — semantic embeddings (locally computed)

| Column | What it is | Computation | Scale |
|---|---|---|---|
| `conf_title_kw_sim` | Cosine similarity (title, keyword) | `cosine_sim(embed(title), embed(keyword))` via sentence-transformers | float ∈ [−1, 1] |
| `conf_snippet_kw_sim` | Cosine similarity (snippet, keyword) | Same, with snippet instead of title | float ∈ [−1, 1] |
| `conf_bm25` | BM25 score (body text vs keyword) | `compute_bm25_scores(keyword, page_texts)` in features.py:550 — classical IR formula | float ≥ 0 |

---

## CONFOUNDERS — DataForSEO Whois Overview (REPLACED Moz, 2026-05-24)

All fetched via DataForSEO `/v3/domain_analytics/whois/overview/live` on 2026-05-24. Replaces the older Moz columns (which had only 22 % coverage).

| Column | What it is | DataForSEO field | Scale | Coverage |
|---|---|---|---|---|
| `conf_domain_authority` | log10(organic_count + 1) — clean DA proxy | `metrics.organic.count`, transformed | float | **~100 %** |
| `conf_backlinks` | Total organic-ranking positions across Google (visibility proxy) | `metrics.organic.count` | int | **94–97 %** |
| `conf_referring_domains` | Number of #1 organic Google positions held by the domain | `metrics.organic.pos_1` | int | **94–97 %** |
| `conf_brand_recog` | Binary: brand-scale visibility | derived: `organic_count ≥ 100k OR pos_1 ≥ 500` | binary | **~100 %** |
| `conf_dfs_paid_count` | Total paid-ad ranking positions | `metrics.paid.count` | int | **94–97 %** |
| `conf_dfs_etv` | Estimated traffic value | `metrics.organic.etv` | float USD | **94–97 %** |
| `conf_dfs_domain_age_years` | Years since domain creation | `created_datetime` parsed | float | **88–96 %** |

**Important provenance note**:
- `conf_brand_recog` was originally a hardcoded set of ~100 SaaS/tech brand domains (`BRAND_DOMAINS` in features.py) — see "Brand heuristic — the old version" appendix below.
- `conf_domain_authority`, `conf_backlinks`, `conf_referring_domains` were originally from Moz API (separate paid product), cached as a parquet that only covered 24 % of unique domains.
- All four columns were OVERWRITTEN on 2026-05-24 via `scripts/merge_dfs_domain_authority.py`. Originals backed up at `*.bak-pre-dfs.parquet`.

---

## CONFOUNDERS — DataForSEO keyword-level (bulk-fetched once)

These describe the **search query** itself, not individual pages. Same value for every URL within a given keyword.

| Column | What it is | DataForSEO field | Scale |
|---|---|---|---|
| `dfs_keyword_difficulty` | Keyword ranking-difficulty score | `bulk_keyword_difficulty` | int 0–100 |
| `dfs_search_volume` | Estimated monthly Google search volume | `keyword_info.search_volume` | int |
| `dfs_cpc` | Avg. Google Ads cost-per-click | `keyword_info.cpc` | float USD |
| `dfs_competition` | Google Ads competition index | `keyword_info.competition` | float [0, 1] |
| `dfs_intent_commercial` | Probability of commercial intent | `search_intent.commercial` | float [0, 1] |
| `dfs_intent_informational` | Probability of informational intent | `search_intent.informational` | float [0, 1] |
| `dfs_intent_navigational` | Probability of navigational intent | `search_intent.navigational` | float [0, 1] |
| `dfs_intent_transactional` | Probability of transactional intent | `search_intent.transactional` | float [0, 1] |

Source parquets in `~/Hamburg/GEODML_Analysis/geodml_data/data/dataforseo/`:
- `keyword_overview.parquet`
- `bulk_keyword_difficulty.parquet`
- `search_intent.parquet`
- `google_ads_search_volume.parquet`

---

## Appendix A — Brand heuristic (the OLD version of `conf_brand_recog`)

Before the 2026-05-24 replacement, `conf_brand_recog` was a hand-curated set membership:

```python
BRAND_DOMAINS = {  # ~100 SaaS/tech brands
    "salesforce.com", "hubspot.com", "microsoft.com", "oracle.com",
    "sap.com", "adobe.com", "google.com", "ibm.com", "cisco.com",
    "servicenow.com", "workday.com", "zendesk.com", "atlassian.com",
    "slack.com", "zoom.us", "dropbox.com", "shopify.com", "twilio.com",
    "datadog.com", "snowflake.com", "cloudflare.com", "okta.com",
    "pagerduty.com", "elastic.co", "mongodb.com", "confluent.io",
    "hashicorp.com", "databricks.com", "stripe.com", "brevo.com",
    "mailchimp.com", "intercom.com", "freshworks.com", "zoho.com",
    "monday.com", "asana.com", "notion.so", "airtable.com",
    "clickup.com", "smartsheet.com", "wix.com", "squarespace.com",
    "bigcommerce.com", "klaviyo.com", "semrush.com", "ahrefs.com",
    "moz.com", "hootsuite.com", "buffer.com", "sproutsocial.com",
    "canva.com", "figma.com", "webflow.com", "unbounce.com",
    # … and ~50 more SaaS / vertical-software domains
}

def conf_brand_recog(domain): return 1 if domain in BRAND_DOMAINS else 0
```

This is a B2B-SaaS-biased list (the original corpus was a SaaS-marketing keyword set), which is why it had ~100 % coverage but very low precision/recall for "brand" in any general sense. The DataForSEO-derived replacement is empirically grounded in actual Google visibility.

---

## Appendix B — Source code anchors

- **All `extract_*` and `conf_*` extractors**: `interpretability/pipeline/features.py`
  - lines 251–449: per-treatment `extract_*` functions
  - lines 463–530: per-confounder `conf_*` functions
  - lines 533–574: embedding + BM25 helpers
  - lines 583–714: `extract_one_page` orchestrator + semantic-column attachment
- **Static domain sets / regexes**: `features.py:60–225` (BRAND_DOMAINS, EARNED_DOMAINS, AUTHORITY_DOMAINS, LINK_FILTER_DOMAINS, STRUCTURED_DATA_TYPES, _STAT_PATTERNS, _QUESTION_RE, _DATE_PATTERNS)
- **DataForSEO keyword-level pipeline**: `scripts/build_features_from_legacy.py:74–113`
- **DataForSEO Whois Overview fetch (NEW)**: `scripts/fetch_dfs_domain_authority.py`
- **DataForSEO → parquet merge (NEW)**: `scripts/merge_dfs_domain_authority.py`
- **Confounder list in DML pipeline**: `interpretability/pipeline/config.py:133–160`
- **Upstream legacy code**: `~/Hamburg/GEODML/pipeline/extract_features.py` (verbatim source of patterns and domain sets)

---

## Reproducibility

The unified parquet (`~/geodml_data/data/main/unified_2026-05-24.parquet`) bundles all of the above per-row. The CSV dictionary
(`unified_2026-05-24_dictionary.csv`) gives per-column type / source / units / description / notes.

To verify any single coefficient:

```python
import pandas as pd
df = pd.read_parquet("~/geodml_data/data/main/unified_2026-05-24.parquet")
# e.g. average T6 freshness per variant
df.groupby("variant")["treat_freshness"].mean()
```

To rebuild from the per-variant + regression sources after a change:

```bash
python scripts/build_unified_dataset.py
```
