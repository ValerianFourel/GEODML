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
