# Funnel study: admission into the pool and ranking within it

Protocol fixed on 2026-10-07, before any result below was computed. Companion to
`geo_drivers_study.md` and `intent_surfacing_study.md`; same testbed, units of inference and
observational wording. Code: `analysis/interpretability/pipeline/funnel_rows.py`,
`funnel_features.py`, `funnel_models.py`; drivers `analysis/scripts/funnel_features.py` (Mac,
feature table) and `analysis/scripts/funnel_study.py` (HoreKa: extract, assemble, analyze, report).

## 1. Question and wording

Where does a page win or lose on its way into a generated answer, and how do prompt intent, page
intent, topic, and the SEO and page data collected in experiment 1 relate to each step? All
results are associations. The prompt position `x ∈ [0, 1]` is a measured property of prompt
text, not a randomized treatment. Page, domain and SEO variables are observational. Text
embeddings describe text; they are never called confounders.

## 2. Already seen (exploratory for this study)

GEO drivers E1/E2 (job 5180401) and the page-readiness llama fit; the cited-source slope by
model × engine and by model × method from the published generation rows (2026-10-06); the
snapshot audit (glued-title rows); the feature-table coverage counts and the experiment-1
validation of the re-extracted page features; experiment 1's DML results (a different testbed).
**Not yet seen:** any association between a funnel stage and any feature below.

## 3. Testbed facts used

The V2 search snapshots are byte-identical to experiment 1's `phase0_top20_{ddg,searxng}.parquet`
(sha256 `894130b7…`, `a693dd56…`); usable rows 10,227 and 13,540 (adapter rule). Search scores
every row of one engine snapshot (exact keyword, then 4 × keyword-word overlap + title/snippet
overlap, then stored position, then a hash) and returns 20 rows. Parallel Expansion: 3 searches,
union deduplicated by URL, scored by the cross-encoder against the user prompt, top 7 shown.
Reactive Loop: up to 3 searches, each scored against the AI's own query, top 3 each. The
cross-encoder sees only `title + "\n" + text`; the generator also sees the URL.

## 4. Units and sets

Answer `i` (natural condition primary): model, method, engine, prompt, keyword `k`, position `x`.
Item = snapshot row (engine, keyword, stored position, URL, title, snippet). Per answer:
`U_i` the usable rows of its engine for keyword `k` (≤ 20); `R_i` rows any of its searches
returned; `C_i` rows the reranker scored; `P_i` rows presented; `K_i` rows ranked. On `U_i`,
`r ≥ c ≥ p ≥ k` is required (extraction stops otherwise).

## 5. Feature blocks and visibility

| Block | Features | Retriever | Reranker | Generator |
|---|---|---|---|---|
| A1 intent | page intent z; z·(x−½); alignment −\|u−x\| | text | text | text |
| A2 topic | intent-free prompt–page topic similarity | text | text | text |
| A3 snippet surface | title/snippet length, digits, %, year, currency, question title, listicle title, domain named in text, glued title | text | text | text |
| A4 URL string | https, path depth, length, query, subdomain, .com/.org/.edu-gov, user-content platform, Wikipedia, ad redirect | — | — | yes |
| B search | stored engine position; SearXNG score and engine count | yes | — | — |
| C1 off-page | DataForSEO domain data (organic count, top-1 count, traffic value, paid count, age), Open PageRank, llms.txt, brand and earned lists, Google rank of the URL and of the domain for the keyword | — | — | only via brand knowledge |
| C2 page body | stats density, question headings, modularity, JSON-LD, external and authority citations, word count, readability, internal/outbound links, images with alt, freshness at 2026-04-15 | — | — | — |

C2 is seen by no component and C1/A4 not by the reranker: these are negative controls. Each
block with incomplete coverage gets one missing indicator (main specification: missing values set
to the mean, indicator included). Continuous counts are log1p-transformed before standardisation.
Standardisation (mean, SD) is computed once over the `U` population and used at every stage,
stratum and specification. Keyword-level DataForSEO variables enter only as two moderators:
commercial-or-transactional keyword intent, difficulty tercile.

## 6. Stage models

* **R|U** retrieval admission: Chamberlain conditional logit over `U_i` (answers with none or all
  items admitted carry no information and are counted). **R0|U**: same model where admission
  means "among the frozen search's top 20 for the prompt text itself" (exact replay).
  Query-rewriting contrast θ_R − θ_R0.
* **C|R**: rates only (Parallel URL deduplication; ablation).
* **P|C** reranker selection: top-k Plackett–Luce over each reranker event's candidates (k = 7
  Parallel, 3 Reactive), no position effects; plus within-event least squares of the reranker
  score on the same features.
* **K|P** generator ranking: top-k Plackett–Luce over presented items with presented-slot
  effects (capped at 20).

Reported per feature: coefficient and odds ratio per SD, 95% keyword-bootstrap interval (200
draws shared by all stages and strata), within-keyword permutation p (200 shuffles of x) for
x-dependent features; per block: share of fit (drop-block increase in negative log-likelihood).
Replicates that fail to converge are counted and reported.

## 7. Exact funnel decomposition

For a feature, items in the top versus bottom quartile within each (engine, keyword) cell (for
alignment: within each answer; binary features: 1 versus 0), each answer giving each group equal
weight: log RR(K|U) = log RR(R|U) + log RR(C|R) + log RR(P|C) + log RR(K|P), exactly, in every
bootstrap replicate. Admission into the pool = R + C + P; ranking within the pool = K.

## 8. Strata, replication and the confirmatory family

Primary strata: model (llama4, qwen38) × method (Parallel, Reactive); engine is a secondary
breakdown. A result replicates if, in all four strata, it has the same sign, its 95% interval
excludes 0 and, for x-dependent terms, permutation p < 0.05 (`geo_drivers.replication`).
Confirmatory family (each must replicate):

* **P1** intent alignment at R|U (the AI's searches admit pages that match the prompt's intent).
* **P2** alignment R − R0 (query rewriting adds intent matching beyond the prompt's own words).
* **P3** domain authority (C1 summary: DataForSEO organic count) generator − reranker, on items
  whose title and snippet do not name the domain (a brand preference the reranker cannot have),
  and the generator coefficient must exceed the 95th percentile of the C2 empirical null.
* **P4** z·(x−½) at R|U.

Everything else is secondary (Benjamini–Hochberg within stage × stratum, plus replication) or
exploratory (individual C1/C2 features, moderators, engine breakdown).

## 9. Specifications

Main (all blocks); visible-only (A + B); complete-case (items with every block observed);
all conditions; engine breakdown. A specification with rich text controls is deferred.

## 10. Off-topic audit (descriptive)

Per answer and stage: share of items outside `U_i`, of glued-title rows and of ad-redirect rows,
by x bin, model, method and engine (keyword-clustered standard errors); hub table of rows
presented under the most foreign keywords.

## 11. Not claimed

Causal effects of SEO or page features; effects of keyword-level variables; any statement about
live search engines or other models. Coverage limits: ~21k Qwen cells only on Hugging Face;
2 SearXNG keywords without rows.

## 12. Deviations

Dated addenda only, written before any result they concern.

## Addendum A1 (2026-10-07, before any funnel result): exploration and confirmation keywords

Valerian asked for a full exploratory review on the Mac (no GPU) before the HoreKa run. To keep the
confirmatory family untouched, keywords are split once, by a seeded hash fixed now:
a keyword is an **exploration** keyword if `int(sha256("funnel-exploration-v1:" + keyword)[:8], 16) / 2**32 < 0.30`
(278 of 1,011 keywords), otherwise a **confirmation** keyword (733).

* Local exploratory analyses may use the exploration keywords for every stage model, the funnel
  decomposition and the off-topic audit (trace sample downloaded from Hugging Face), and all keywords
  for analyses of the published generation rows that do not estimate P1–P4 (cited-source intent,
  ranking length, search count, answer length, cross-model agreement, ranking change, SEO and page
  features of the cited sources, keyword moderators of the cited-source slope).
* The confirmatory family P1–P4 is evaluated on HoreKa on the confirmation keywords only
  (`funnel_study.py analyze/report --split confirmation`). Everything computed on the Mac is
  labelled exploratory.
