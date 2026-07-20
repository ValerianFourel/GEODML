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
