# Canonical treatments and confounders — locked 2026-05-24

This document pins down the **final treatment set** and **final confounder set**
to be used by every DML analysis going forward (admission, rank_delta,
rank_post). It supersedes earlier "everything tested" lists.

Frozen so subsequent analyses cannot accidentally swap codings or include the
list-membership flag that biases interpretation.

---

## TREATMENTS (7) — canonical set

| # | Column | Family | Type / units | Source | Why this is the canonical choice |
|---|---|---|---|---|---|
| 1 | `treat_stats_density` | T1b | float (unique regex matches per 500 words) | `interpretability/pipeline/features.py:extract_t1b_stats_density` | Continuous, more informative than the binary T1a; 6-pattern regex set is broader/cleaner; LLM version has Pearson r = 0.19 vs code — they measure different things and the LLM version doesn't survive correction |
| 2 | `treat_question_headings` | T2a | binary {0, 1} | `interpretability/pipeline/features.py:extract_t2a_question_headings` | Same 14 question-word prefix regex as T2_code PLUS catches "ends-with-?"; survives Romano-Wolf (+0.21***); broader than strict-lexical, less noisy than loose-semantic LLM version |
| 3 | `treat_structured_data` | T3 (new) | binary {0, 1} | `interpretability/pipeline/features.py:extract_t3_structured_data` | 18 JSON-LD `@type` values accepted (FAQPage, Product, HowTo, Article, BlogPosting, Review, AggregateRating, Offer, ItemList, BreadcrumbList, VideoObject, Dataset, Course, Event, Recipe, QAPage, FAQ, SoftwareApplication) — vs 4 in T3_code and 3 in T3_llm prompt; survives RW (−0.10*) |
| 4 | `T4_citation_authority_code` | T4_code | int (count) | `~/Hamburg/GEODML/src/page_features.py:t4_external_citation_authority` | Deterministic count of UNIQUE outbound links whose host suffix is `.edu`, `.gov`, `.gov.uk`, `.ac.uk`, or `.mil`, OR whose registrable domain is in a curated 9-domain authority list (wikipedia.org, scholar.google.com, ncbi.nlm.nih.gov, arxiv.org, nature.com, sciencedirect.com, ieee.org, acm.org, researchgate.net). Chosen over T4_llm for full reproducibility — deterministic regex/list lookup, no LLM-judgement noise. Doesn't survive Romano-Wolf on its own (NS), but kept for methodological completeness as the cleanest T4 coding. |
| 5 | `treat_topical_comp` | T5 | float (cosine ∈ [−1, 1]) | `interpretability/pipeline/features.py:extract_one_page` semantic step | Only version exists. Survives RW with largest content-effect magnitude (+0.48***). Cosine sim of (body, keyword) via sentence-transformers. |
| 6 | `treat_freshness` | T6 | ordinal 0–4 | `interpretability/pipeline/features.py:extract_t6_freshness` | Only version exists. Survives RW on BOTH stages — the dual-stage finding (admission AND rank). Score buckets: ≤ 180 days → 4, ≤ 365 → 3, ≤ 730 → 2, ≤ 1825 → 1, else → 0 |
| 7 | `has_llms_txt` | T_llms_txt | binary {0, 1} | HTTP probe on `/llms.txt` and `/.well-known/llms.txt`; column in `regression_dataset.parquet` | Only version. Survives RW (+0.09*). Clean operational definition (file exists or not). |

---

## TREATMENTS — DROPPED (and why)

### Hard drops (don't include in any analysis)

| Column(s) | Reason |
|---|---|
| **`treat_source_earned` (T7)** | List-membership flag for ~250 curated earned-media domains. Not a content treatment — circular as a causal estimand. **Explicitly excluded by analyst decision** since it dominates other effects and clouds interpretation. Can be reported descriptively in §4 of the paper as a side finding, but never enters the DML treatment set. |
| `treat_source_brand`, `treat_source_type`, `treat_brand`, `treat_earned` | Auxiliary T7-family list flags with the same defect |
| `treat_ext_citations_any` (T4a) | Binary version of T4; doesn't survive Romano-Wolf correction; binary throws away count information |
| `treat_auth_citations` (T4b) | Count to 22-domain authority list (broader than T4_code); doesn't survive RW; broader list felt less defensible than T4_code's tighter 9-domain set |
| `T4_citation_authority_llm` (T4_llm) | LLM-judged; marginal RW survival (β=−0.024, p=0.040) but introduces LLM-judgement noise into the X-set — chose deterministic T4_code over reproducibility concerns |
| `treat_stats_present` (T1a) | Binary version of T1b — redundant; the continuous T1b carries more signal |
| `T1_stats_regex_count` (was `T1_statistical_density_code`) | Same regex idea as T1b but from `src/page_features.py` (older, narrower); T1b in `interpretability/pipeline/features.py` is the canonical code coding |
| `T1_stats_llm_count` (was `T1_statistical_density_llm`) | LLM-judged float count; Pearson r = 0.19 vs code — measures something different but doesn't survive correction |
| `treat_structural_modularity` (T2b) | Count of `<h2>` + `<h3>` tags — structural-complexity metric, not a content treatment; doesn't survive correction |
| `T2_qhead_strict_lexical` (was `T2_question_heading_code`) | Strict-prefix regex on 14 question words; same family as `treat_question_headings` (T2a) which extends it; the broader T2a is canonical |
| `T2_qhead_loose_semantic` (was `T2_question_heading_llm`) | LLM-coded; **survives RW with opposite sign to T2a** (−0.13 demoter vs T2a +0.21 promoter) — keeping both invites reviewer questions about coding-method-dependent findings; we lock T2a as canonical |
| `T3_structured_data_code` | Only 4 schema `@type` values accepted vs 18 in canonical T3; redundant and narrower |
| `T3_structured_data_llm` | LLM-coded; marginal (RW p ≈ 0.078); broader-coverage canonical T3 wins |

---

## CONFOUNDERS (28) — canonical set

### Tier 1 — page-HTML extracted (6)

`conf_word_count`, `conf_readability`, `conf_internal_links`, `conf_outbound_links`, `conf_images_alt`, `conf_https`

All deterministic, locally parsed from the downloaded HTML via BeautifulSoup. Defined in `interpretability/pipeline/features.py`.

### Tier 2 — SERP-derived (4)

`conf_title_has_kw`, `conf_title_len`, `conf_snippet_len`, `conf_serp_position`

Computed from DDG / SearXNG raw output.

**Note**: `conf_serp_position` (= pre_rank) mechanically dominates rank_delta variance (since rank_delta = pre_rank − post_rank). Including it as a confounder is **required** for valid causal interpretation of T-effects on rank_delta — otherwise pre_rank confounds everything. In fig 13 panel (b) it is excluded only because it would visually swamp the other confounders' t-stats; in the DML itself it stays in the X-set.

### Tier 3 — semantic / IR (3)

`conf_title_kw_sim`, `conf_snippet_kw_sim`, `conf_bm25`

Computed locally via sentence-transformers (cosine similarities) and the classical BM25 formula.

### Tier 4 — DataForSEO Whois Overview (7) — REPLACED Moz on 2026-05-24

These four canonical slots now contain DataForSEO-derived values (not Moz):

| Column slot | Now contains | Coverage |
|---|---|---|
| `conf_domain_authority` | `log10(dfs_organic_count + 1)` — clean domain-authority proxy | ~100% |
| `conf_backlinks` | `dfs_organic_count` — total Google organic-ranking positions (visibility proxy) | 94–97% |
| `conf_referring_domains` | `dfs_organic_pos_1` — number of #1 organic positions | 94–97% |
| `conf_brand_recog` | binary: `(organic_count ≥ 100k) OR (pos_1 ≥ 500)` | ~100% |

Plus three **new** DFS columns with no prior analogue:

| Column | What it is |
|---|---|
| `conf_dfs_paid_count` | Total paid-ad ranking positions across Google |
| `conf_dfs_etv` | Estimated traffic value (USD): organic visits × CPC |
| `conf_dfs_domain_age_years` | Years since domain creation |

The OLD Moz columns and the OLD `BRAND_DOMAINS` heuristic are preserved only in `*.bak-pre-dfs.parquet` backups; **not used in any analysis going forward**.

### Tier 5 — DataForSEO keyword-level (8)

`dfs_keyword_difficulty`, `dfs_search_volume`, `dfs_cpc`, `dfs_competition`, `dfs_intent_commercial`, `dfs_intent_informational`, `dfs_intent_navigational`, `dfs_intent_transactional`

Bulk-fetched once during the pipeline build; same value for every URL within a given keyword (since they describe the search query, not the page).

---

## CONFOUNDERS — IGNORED / RETIRED

| Source / column | Reason |
|---|---|
| **Moz `conf_domain_authority`** (pre-2026-05-24) | 22% coverage; slot overwritten with DFS-derived log10(organic_count). Original in `*.bak-pre-dfs.parquet` only. |
| **Moz `conf_backlinks`** (pre-2026-05-24) | 11% coverage; slot overwritten with DFS organic_count. |
| **Moz `conf_referring_domains`** (pre-2026-05-24) | 11% coverage; slot overwritten with DFS organic_pos_1. |
| **Heuristic `conf_brand_recog`** (`BRAND_DOMAINS` set in `features.py:65`) | ~100 hardcoded SaaS brand domains; slot overwritten with empirical DFS brand_proxy. |

---

## What the DML re-run will fit

Using the canonical 7 treatments and 28 confounders above:

| Outcome | Spec | Fitted by |
|---|---|---|
| **Y₁ = `selected_by_llm`** (binary admission) | Spec A (per-variant slices + per-engine + per-model + per-pool), Spec B (joint mutually-controlled) | new script that consumes `unified_2026-05-24.parquet` |
| **Y₂ = `rank_delta`** | Spec A per-variant, Spec B joint, Spec C joint with Romano-Wolf | same |
| **Y₃ = `rank_post`** | Spec A per-variant, Spec B joint, Spec C joint with Romano-Wolf | same |

Sample frames:
- For Y₁: full SERP pool (admitted + rejected URLs) — built from `phase0_top*.parquet` + per-variant LLM outputs, as in `dml_selected.py::build_pool_table()`.
- For Y₂, Y₃: admitted URLs only (`selected_by_llm = 1`).

---

## Decisions confirmed by analyst (2026-05-24)

1. **T7 excluded** from all DML treatment sets. Reported descriptively only.
2. **T4 family**: drop T4a, T4b, T4_llm. **Keep T4_code** as the canonical T4 — chosen for full determinism (no LLM-judgement noise) even though it doesn't survive RW alone.
3. **T5 (`treat_topical_comp`)**: keep as a treatment (not a confounder). The literature on LLM ranking treats topical relevance as a manipulable lever.
4. **Final treatment count**: 7 (T1b, T2a, T3, T4_code, T5, T6, T_llms_txt).
5. **Moz columns and heuristic brand_recog**: retired. The four canonical slots now contain DataForSEO-derived values exclusively.
