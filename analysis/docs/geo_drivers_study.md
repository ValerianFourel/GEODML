# GEO drivers study: prompt intent, page intent, keyword and ranking

Protocol fixed on 2026-10-04 before the analysis below was run. Already seen at that
time, and therefore exploratory: the llama-only intent model (`analysis-llama/`, consensus
point estimates) and the within-keyword ranking-change correlations. Not yet seen: Qwen,
every engine-specific estimate, the topic and keyword drivers, and E1.

All results are observational. The prompt's axis position is a measured property of its
text, not a randomized treatment. Page positions and similarities describe evidence text;
they are not treatments or confounders. The testbed is a closed, frozen corpus (21,384
distinct snippets, two engines) searched by a corpus-wide lexical retriever.

## Units and data

- Answer `i`: one generation by model `m ∈ {llama4, qwen38}` for prompt `p` (keyword `k`),
  engine `e ∈ {duckduckgo, searxng}`, method `s`, condition `c`. Presented set `P_i` in its
  presented order; ranking `R_i`, an ordered subset of `P_i` (`extract-all-v1`).
- Prompt position `x_p ∈ [0, 1]`: archived axis-1 percentile (`final-axis-map.jsonl`).
- Page position `z_d`: consensus axis-1 z; `u_d`: its percentile on the prompt scale
  (`snippet-embeddings-corpus-v1`).
- Topic similarity `t_pd`: cosine between prompt and page after L2 normalization,
  centering with the frozen map's `embedding_mean`, and removing the span of the map's two
  supervised axes; averaged over the Qwen and Mistral views. It measures topical closeness
  with the intent subspace taken out.
- On-keyword `o_pd`: 1 if the page's snapshot row belongs to the prompt's own keyword.

## E1. Prompt position → permutation position

`y_i` = rank-weighted mean of `z_d` over `R_i`, weights `1/log2(r + 1)` for rank `r`.
`y0_i` = mean `z_d` over `P_i` (the pool the model saw). `Δ_i = y_i − y0_i` (ranking stage).
Estimand: the within-keyword slope of `y`, `y0` and `Δ` on `x` (keyword-demeaned least
squares), i.e. the change in ranking, pool and reordering intent from the most informational
to the most action-ready prompt of a keyword. Also reported: 10 prompt-position bins.

## E2. Drivers of rank

Top-k Plackett–Luce over each ranked prefix among presented pages, with presented-position
fixed effects (capped at 20) and four standardized drivers:
`o_pd` (on-keyword), `t_pd` (topic), `z_d` (page intent), `−|u_d − x_p|` (intent alignment).
Reported per driver: coefficient per standard deviation, its odds ratio, and its share of
fit (increase in mean negative log-likelihood when the driver is dropped).
Robustness specification: the earlier form `z_d` and `z_d·(x_p − 0.5)` plus the two keyword
drivers.

## Strata, inference and replication

Each estimate is fitted separately in the four strata model × engine, and pooled per model.
Uncertainty: keyword-cluster bootstrap, 200 replicates (95% percentile intervals).
Null for every term involving `x` (E1 slopes, intent alignment, interaction): 200 shuffles of
`x` among the prompts of each keyword (two-sided p, `(1 + #|null| ≥ |obs|) / 201`).

A result **replicates** if, in all four strata, it has the same sign, its 95% interval
excludes 0, and (for terms involving `x`) its permutation p is below 0.05. Anything else is
reported as not replicated, with the per-stratum estimates. Thresholds are not changed after
results are seen. Methods and conditions are reported as secondary breakdowns only.

## Not in this study

Retrieval selection from the 20 returned rows (needs search-event extraction), answer-text
analysis (`answer_readiness.py`), page-content interventions, and judge-based relevance.
