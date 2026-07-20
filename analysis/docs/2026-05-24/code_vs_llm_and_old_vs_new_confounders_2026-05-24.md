# Deep-dive: code-vs-LLM treatment codings, and old-vs-new confounder versions

Reviewer-grade reference detailing **the precise difference between the
code-derived and LLM-derived versions of each T1–T4 treatment**, and the
**before/after for every confounder that was replaced on 2026-05-24**.

For the per-row dataset and column-level dictionary, see
`unified_2026-05-24.parquet` + `unified_2026-05-24_dictionary.csv`. For each
column's general definition, see
`treatments_and_confounders_reference_2026-05-24.md`.

---

## Part A — Code vs LLM treatment codings

For T1 / T2 / T3 / T4 we ran TWO independent annotators against the same web
page and recorded both. The "code" version is a deterministic Python extractor
(regex + DOM parsing); the "LLM" version is the same prompt run against a
condensed page digest by Llama-3.3-70B-Instruct.

### Where the codings live

| Coding side | File | Function |
|---|---|---|
| code (older, narrow definitions) | `~/Hamburg/GEODML/src/page_features.py` | `t1_statistical_density`, `t2_question_heading_match`, `t3_structured_data_presence`, `t4_external_citation_authority` |
| code (newer, broader; THE one used in `treat_*`) | `interpretability/pipeline/features.py` | `extract_t1a_stats_present`, `extract_t1b_stats_density`, `extract_t2a_question_headings`, `extract_t2b_structural_modularity`, `extract_t3_structured_data`, `extract_t4a_ext_citations_any`, `extract_t4b_auth_citations` |
| LLM (single shared definition) | `~/Hamburg/GEODML/src/page_features.py` lines 372–388 | `llm_extract_treatments()` (HuggingFace Inference API, Llama-3.3-70B) |

So **THREE codings actually exist in the data** for T1/T2/T3/T4:

1. **`T*_code`** — old-style code, narrower definitions (`src/page_features.py`)
2. **`T*_llm`** — Llama-3.3-70B-Instruct judging the same metric (`src/page_features.py`)
3. **`treat_*` (a.k.a. T1a / T1b / T2a / T2b / T3 / T4a / T4b)** — newer, broader code from `features.py` in the analysis pipeline

### The LLM prompt (verbatim)

From `~/Hamburg/GEODML/src/page_features.py:372–388`:

> You are analyzing a web page to measure 4 treatment variables for a causal
> inference experiment on search engine optimization. Given the page digest
> below, evaluate each treatment and return ONLY a JSON object with these
> exact keys:
>
> - `"T1_llm_statistical_density"`: (float) Count of unique statistics, numbers,
>   percentages, dollar amounts, or dates per 500 words of body text. **Be precise.**
> - `"T2_llm_question_heading"`: (0 or 1) Does the page contain H2 or H3
>   headings that **closely match natural language questions** (e.g. "What
>   is…", "How to…", "Why should…")? 1 if yes, 0 if no.
> - `"T3_llm_structured_data"`: (0 or 1) Does the page have JSON-LD
>   structured data of type **FAQ, Product, or HowTo**? 1 if yes, 0 if no.
> - `"T4_llm_citation_authority"`: (integer) Count of outbound links to
>   authoritative sources (`.edu`, `.gov`, academic journals, Wikipedia,
>   government sites). **Only count genuinely authoritative citations, not
>   marketing links.**
> - `"T*_reasoning"` (string per T*): brief explanation
>
> Return ONLY valid JSON. No markdown, no extra text.
>
> PAGE DIGEST:
> {digest}

LLM call: `client.chat_completion(model='meta-llama/Llama-3.3-70B-Instruct',
max_tokens=800, temperature=0.1)`. Digest = first 3,000 chars of body text +
headings + up to 50 outbound link domains, annotated with `[.gov]` / `[academic]`.

### T1 — Statistical density

| Aspect | `T1_code` (`T1_statistical_density_code`) | `T1_llm` (`T1_statistical_density_llm`) | `treat_stats_density` (T1b) — the newer code |
|---|---|---|---|
| **Output type** | float (count per 500 words) | float (LLM's own count per 500 words) | float (count per 500 words) |
| **Source** | `src/page_features.py:64` `t1_statistical_density` | LLM via prompt above | `interpretability/pipeline/features.py:257` `extract_t1b_stats_density` |
| **Mechanism** | 6 regex patterns; counts UNIQUE matches | LLM reads digest, returns its own count | Same 6 regex patterns as T1_code; UNIQUE matches |
| **Regex patterns** | thousands-comma, percent, year (1900-2099), date `m/d/y`, dollar amount, abbreviated count `\b\d[BMK]\b` | not exposed to LLM | same as T1_code |
| **Coverage** | 100% (deterministic) | ~99% (LLM might return None on parse error) | 100% |
| **Code vs LLM agreement** | reference | Pearson r=0.19, Spearman ρ=0.36 — **weak** | almost identical to T1_code (same patterns) |
| **Means** | mean ≈ 2.56 | mean ≈ 4.41 (LLM more liberal) | ≈ T1_code |
| **DML signal** | NS after multi-test correction | NS after multi-test correction | NS on rank but significant on binary admission (Bonferroni ✓, β=-0.078) |

**Why they differ**: the LLM eyeballs the page and decides "what counts as a statistic" — it includes things like "the past 5 years" or "Q3 2024" that the regex misses, but it also miscounts duplicates. They're estimating the same underlying quantity through two very different lenses.

### T2 — Question headings

| Aspect | `T2_code` (`T2_question_heading_code`) | `T2_llm` (`T2_question_heading_llm`) | `treat_question_headings` (T2a) |
|---|---|---|---|
| **Output type** | binary (0 / 1) | binary (0 / 1) | binary (0 / 1) |
| **Source** | `src/page_features.py:87` `t2_question_heading_match` | LLM via prompt above | `interpretability/pipeline/features.py:271` `extract_t2a_question_headings` |
| **Rule** | regex `^\s*(what|how|why|when|where|which|who|can|does|is|are|should|will|do)\b` on `<h2>` and `<h3>` text | "closely matches natural language questions" | same regex as T2_code, **plus** any heading ending in `?` |
| **Coverage** | 100% | ~99% | 100% |
| **Code vs LLM agreement** | reference | **κ=0.665, raw agreement 83.7%** — substantial |  |
| **Pos rate** | ~64% (code more liberal) | ~54% (LLM more conservative on "closely match") | similar to T2_code |
| **Disagreement breakdown** | — | 2,983 cells where code=yes / LLM=no  vs  791 where LLM=yes / code=no | — |
| **DML signal** | NS after RW | **survives RW** (β=−0.13 on rank_delta, demoter) | **survives RW** (β=+0.21 on rank_delta, promoter — opposite sign!) |

**Why same metric → opposite DML signs**: the question-style coding picks up *templated SEO FAQs* (which the LLM penalises) versus *genuinely Q-formatted instructional content* (which the LLM rewards). Code and LLM tag slightly different subsets, and the difference is causally meaningful.

### T3 — Structured data (schema markup)

| Aspect | `T3_code` (`T3_structured_data_code`) | `T3_llm` (`T3_structured_data_llm`) | `treat_structured_data` (T3_structured_data_new) |
|---|---|---|---|
| **Output type** | binary | binary | binary |
| **Source** | `src/page_features.py:95` `t3_structured_data_presence` | LLM via prompt above | `interpretability/pipeline/features.py:298` `extract_t3_structured_data` |
| **JSON-LD `@type` accepted** | **4 types**: FAQPage, FAQ, Product, HowTo | **3 types**: FAQ, Product, HowTo | **18 types** (FAQPage, FAQ, Product, HowTo, SoftwareApplication, Article, BlogPosting, Review, AggregateRating, Offer, ItemList, BreadcrumbList, VideoObject, Dataset, Course, Event, Recipe, QAPage) |
| **Recurses into `@graph`** | yes | implicit (LLM does whatever) | yes |
| **Coverage** | 100% | ~99% | 100% |
| **Code vs LLM agreement** | reference (narrow set) | **κ=0.535** — moderate; LLM finds ~2× more positives (32% vs 16% positive rate) | broader than both, not directly comparable |
| **Disagreement** | — | 3,934 "LLM yes / code no" vs only 125 "code yes / LLM no" — LLM catches JSON-LD variants the strict 4-type code misses | — |
| **DML signal** | NS | marginal (RW p≈0.078) | **survives RW** (β=−0.10 on rank_delta, demoter) |

**Why the LLM is more permissive**: the prompt says "Does the page have JSON-LD structured data of type FAQ, Product, or HowTo?" — the LLM also picks up visible FAQ sections (even without JSON-LD), or schema using non-standard `@type` strings. The code requires an exact lowercased `@type` match in a recognised list.

### T4 — Citation authority

| Aspect | `T4_code` (`T4_citation_authority_code`) | `T4_llm` (`T4_citation_authority_llm`) | `treat_auth_citations` (T4b) | `treat_ext_citations_any` (T4a) |
|---|---|---|---|---|
| **Output type** | integer (count) | integer (count) | integer (count) | binary (0/1) |
| **Source** | `src/page_features.py:132` `t4_external_citation_authority` | LLM via prompt above | `interpretability/pipeline/features.py:354` `extract_t4b_auth_citations` | `interpretability/pipeline/features.py:339` `extract_t4a_ext_citations_any` |
| **Authority suffixes** | `.edu`, `.gov`, `.gov.uk`, `.ac.uk`, `.mil` | LLM judges; prompt lists "`.edu`, `.gov`, academic journals, Wikipedia, government sites" | same as T4_code | n/a (counts ANY outbound) |
| **Authority domains** | **9 hardcoded**: wikipedia.org, scholar.google.com, ncbi.nlm.nih.gov, arxiv.org, nature.com, sciencedirect.com, ieee.org, acm.org, researchgate.net | LLM uses its own training-time judgment of "authoritative" | **22 hardcoded** (T4_code's 9 + pubmed, springer, wiley, jstor, ssrn, nber, who, un, worldbank, statista, pewresearch, gallup) | n/a |
| **Dedup of duplicate links** | yes (via `seen` set on href) | the LLM gets at most 50 unique outbound domains in the digest | yes | no |
| **Coverage** | 100% | ~99% | 100% | 100% |
| **Code vs LLM agreement** | reference | **r=0.56, ρ=0.50** — moderate; LLM gives higher counts (mean 0.88 vs 0.37 — likely from broader "authoritative" judgement) | extends T4_code; correlates strongly with T4_code | same target, different operationalisation |
| **DML signal** | NS | **survives RW** marginal (β=−0.024 on rank_delta) | NS after RW | NS after RW |

**Why means differ (LLM=0.88 vs code=0.37)**: the prompt lets the LLM judge what's "authoritative" more liberally — it counts links to .gov, .edu, Wikipedia (matching code), but ALSO things like nytimes.com, reuters.com, scientificamerican.com that aren't on the code's authority list. The code is rule-tight; the LLM is judgement-tight.

---

## Part B — Old vs new confounders (the 2026-05-24 DataForSEO replacement)

On 2026-05-24 we ran `scripts/fetch_dfs_domain_authority.py` →
`scripts/merge_dfs_domain_authority.py` to overwrite four canonical confounder
slots with DataForSEO-derived values. The originals are preserved at
`*.bak-pre-dfs.parquet`.

### `conf_domain_authority`

| | OLD (Moz) | NEW (DataForSEO Whois Overview) |
|---|---|---|
| **Backing field** | Moz `domain_authority` (proprietary API) | `metrics.organic.count` from `/v3/domain_analytics/whois/overview/live`, transformed by `log10(count + 1)` |
| **Scale** | int 0–100 (Moz DA convention) | float; log10 of organic-ranking count; typical range 0–9 |
| **Coverage** | **22 %** (Moz free-tier cache only covered ~3.2k of 13.4k domains) | **~100 %** (DFS Whois indexed virtually all domains we queried) |
| **What "high" means** | Moz's proprietary aggregate of link profile / ranking history | Wide footprint in Google organic results across the entire web |
| **Continuity** | continuous within [0, 100] | continuous on log scale — differentiates wikipedia (8.3) from a niche blog (1–2) by orders of magnitude |
| **Failure mode** | missing for 78 % of pages | a handful of brand-new or geofenced domains return None |

### `conf_backlinks`

| | OLD (Moz) | NEW (DataForSEO Whois Overview) |
|---|---|---|
| **Backing field** | Moz `backlinks` count | `metrics.organic.count` from DFS |
| **What it measures** | Count of inbound hyperlinks to the domain (actual backlinks) | Count of pages on the domain that rank in Google organic results |
| **Coverage** | **5–11 %** depending on parquet | **94–97 %** |
| **Semantic note** | This is a **different signal**: DFS gives you "how much organic SEO real estate does this domain hold" rather than "how many other sites link to this domain". Both correlate strongly with "is this a big authoritative site" but they're not the same number. The column name is preserved for compatibility with downstream scripts. | |

### `conf_referring_domains`

| | OLD (Moz) | NEW (DataForSEO Whois Overview) |
|---|---|---|
| **Backing field** | Moz `referring_domains` (number of distinct domains linking to this domain) | `metrics.organic.pos_1` (number of #1 organic positions held by this domain across Google) |
| **What it measures** | Inbound-link diversity | Elite-rank visibility (how many queries the domain wins outright) |
| **Coverage** | **5–11 %** | **94–97 %** |
| **Semantic note** | Even more different from the Moz semantics than `conf_backlinks` was. We preserved the column name for downstream compatibility, but the *signal* is now "how many keywords this domain ranks #1 for". |  |

### `conf_brand_recog`

| | OLD (heuristic) | NEW (DataForSEO Whois Overview, derived) |
|---|---|---|
| **Backing source** | Hardcoded set `BRAND_DOMAINS` in `interpretability/pipeline/features.py:65–95` (~100 SaaS / tech / vertical-software brand domains) | Binary derived from DFS metrics: `(organic_count ≥ 100,000) OR (pos_1 ≥ 500)` |
| **Method** | Set-membership lookup on the domain string | Empirical Google footprint at brand scale |
| **Output** | binary 0 / 1 | binary 0 / 1 |
| **Coverage** | 100 % (heuristic always returns 0 or 1) | ~100 % (NaN-fills to 0) |
| **Positive rate** | extremely high in our SaaS-marketing corpus (~77–100 % per variant — many of our domains hit the hardcoded list) | **13.8 % overall** — much more conservative and empirically grounded |
| **Domains classified positive (sample)** | salesforce.com, hubspot.com, microsoft.com, oracle.com, sap.com, adobe.com, slack.com, zoom.us, shopify.com, mongodb.com, snowflake.com, stripe.com, figma.com, etc. (~100 hand-picked) | youtube.com, facebook.com, instagram.com, reddit.com, wikipedia.org, tiktok.com, pinterest.com, quora.com, amazon.com, google.com (the actual most-visited domains by organic-Google footprint, top 3,315 of 24k) |
| **Failure mode** | B2B-SaaS-biased; ignores anything off the curated list | A small site that legitimately ranks well for many queries (e.g., a 6-year-old niche blog with 200k organic positions) would be classified as "brand" |

**Why the change**: the heuristic was reviewer-vulnerable ("why did you classify these specific 100 domains as brands?"). The DFS-derived version is empirically defensible ("brand = ranks for many things on Google"). And the lower positive rate (13.8 % vs ~85 %) means the DML can actually use the signal to discriminate — when nearly everything was tagged "brand" the variable carried very little information.

### NEW DataForSEO columns (no prior analogue)

These have **no Moz / heuristic counterpart** — they're entirely new signals added 2026-05-24:

| New column | DFS source field | What it measures | Coverage |
|---|---|---|---|
| `conf_dfs_paid_count` | `metrics.paid.count` | Total Google paid-ad positions (domain's paid SEO scale) | 94–97 % |
| `conf_dfs_etv` | `metrics.organic.etv` | Estimated traffic value (USD): organic visits × CPC. A revenue-weighted authority proxy | 94–97 % |
| `conf_dfs_domain_age_years` | parsed from `created_datetime` | Years since domain registration | 88–96 % |

---

## Part C — Where the NEW versions are stored, and how to verify

After the 2026-05-24 merge:

```
~/geodml_data/data/main/
├── full_experiment_data_biased.parquet           ← NEW values (DFS-backed)
├── full_experiment_data_biased.bak-pre-dfs.parquet  ← OLD values (Moz + heuristic)
├── full_experiment_data_neutral.parquet           ← NEW
├── full_experiment_data_neutral.bak-pre-dfs.parquet  ← OLD
├── full_experiment_data_biased_rag.parquet       ← NEW
├── full_experiment_data_biased_rag.bak-pre-dfs.parquet  ← OLD
├── full_experiment_data_neutral_rag.parquet      ← NEW
├── full_experiment_data_neutral_rag.bak-pre-dfs.parquet  ← OLD
├── regression_dataset.parquet                    ← NEW
└── regression_dataset.bak-pre-dfs.parquet        ← OLD

~/geodml_data/data/dataforseo/
├── domain_authority_dfs.parquet         ← the raw DFS pull (24k domain rows)
└── .checkpoints/whois/chunk_0001.json ... 0024.json   ← per-batch API responses
```

To restore the old (Moz + heuristic) version:

```bash
python scripts/merge_dfs_domain_authority.py --restore
```

To verify any single value (e.g. wikipedia.org's domain authority):

```python
import pandas as pd
df = pd.read_parquet("~/geodml_data/data/dataforseo/domain_authority_dfs.parquet")
print(df[df["domain"] == "wikipedia.org"].T)
# dfs_organic_count = 213,098,793  → log10 ≈ 8.33 → conf_domain_authority
# dfs_organic_pos_1 = 52,721,271   → conf_referring_domains
# dfs_brand_proxy   = 1            → conf_brand_recog
# dfs_domain_age_years = 25.36     → conf_dfs_domain_age_years
```

To verify the raw DataForSEO response for a chunk:

```bash
cat ~/geodml_data/data/dataforseo/.checkpoints/whois/chunk_0001.json | head -50
```

---

## Part D — Impact on fig 13 (top confounders)

The 2026-05-24 swap caused **structural shifts in fig 13 panel (a) — top
admission confounders**:

| Confounder | OLD t-stat | NEW t-stat | Notes |
|---|---|---|---|
| `conf_brand_recog` | **+81.1** (heuristic) | **+61.5** (DFS) | Heuristic was overfit; empirical version more conservative |
| `conf_domain_authority` | barely measurable (22 % coverage) | **−30.9** | NEGATIVE coefficient at admission — unexpected |
| `conf_backlinks` | barely measurable | **−55.7** | NEGATIVE — LLM penalises domains with many organic rankings? |
| `conf_referring_domains` | barely measurable | **−59.7** | TOP NEGATIVE confounder (replaces Moz with DFS pos_1 metric) |
| `conf_dfs_etv` | NEW | **−53.2** | NEW — estimated traffic value also negative |

**The negative authority coefficients are a real finding**: the LLM appears to
*penalise* mega-domains at the admission stage, plausibly to avoid the
"Wikipedia + Reddit + YouTube" boilerplate over-representation that purely
authority-based ranking would produce. Combined with the rank-side panel (where
authority is positive), this paints a picture of a **two-tier admission policy**:
the LLM is more selective with big-brand domains at the gate, but rewards them
once admitted.

Verifying this finding against the OLD data (Moz at 22 % coverage) was
basically impossible because the signal didn't reach significance with such
sparse data. The DFS pull is what enables this part of the paper.

---

## Part E — Verification checklist for reviewers

1. **Code-side definitions verifiable**: every regex pattern, set, and parsing rule is documented above with file:line anchors in `interpretability/pipeline/features.py` and `~/Hamburg/GEODML/src/page_features.py`.
2. **LLM prompt verbatim**: see Part A "The LLM prompt (verbatim)" — exact string used for all T*_llm scores.
3. **DataForSEO fetch reproducible**: `scripts/fetch_dfs_domain_authority.py` + per-chunk JSON checkpoints in `~/geodml_data/data/dataforseo/.checkpoints/whois/` (24 chunks of up to 1000 domains each, cost $48 total documented in `logs/fetch_dfs_*.log`).
4. **Merge reproducible**: `scripts/merge_dfs_domain_authority.py --restore` reverts to the pre-2026-05-24 state; `--dry-run` previews changes.
5. **Old values preserved**: every modified parquet has a `*.bak-pre-dfs.parquet` sibling.
6. **Old vs new measured**: `docs/dml_survivors_2026-05-24.md` reports DML coefficients on the new data; the OLD coefficients are in `docs/archive_2026-05-24-am/` (pre-merge analysis runs).
