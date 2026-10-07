# Steelman pre-registration: does prompt intent reach the cited sources at the shortlist?

Fixed on 2026-10-07, before any analysis in `analysis/steelman/` ran. Companion to
`analysis/docs/funnel_study.md` (addendum A1 split) and `analysis/docs/intent_surfacing_study.md`.
Observational throughout: the prompt position x is a measured text property, not a randomized treatment.

## Scope and what was already seen

Qwen (`qwen38`), natural condition, exploration keywords of addendum A1 that have traces on the Mac
(`~/Hamburg/geodml-inputs/funnel-assembled-exploration-v1`, 237 keywords). Both search methods are strata;
Reactive Loop is primary for C1 because its reranker scores snippets against the agent's own query, never
the user prompt (`agentic_search.py:759-764`); Parallel Expansion scores against the full user prompt.

Already seen before this file: the per-metric stage slopes of `funnel-explore-v1` (deck values); point
estimates of the identity baseline on the common sample (Parallel β_P .0507, β_I .0546, β_K .0597;
Reactive .0574, .0611, .0681); the keep/order coefficients of `funnel-generator-decisions-v1`; within-answer
correlations of shown slot with intent and topic; the slope of the cited count on x. Not seen: every
interval, permutation p, controls and lexical variants, Δ_gen, fixed-effects and score-matched results.
The same rules apply unchanged to the confirmation keywords if they are analysed later.

## Scale, sample, inference

* Page intent: `u`, the page's percentile on the prompt scale (`row_features.npz`).
* Stage values per answer: R0 = mean u over the 20 rows the frozen search returns for the prompt text
  (`funnel-replay-v1/replay.npz`); R, C, P = mean u over retrieved, reranker-scored and shown rows;
  I = rank-weighted mean (w_r = 1/log2(r + 2)) of the first L shown rows in shown order, L = cited count;
  K = rank-weighted mean of the cited rows in cited order.
* Common sample: answers with R0, R, C, P, I and K all finite (answers citing nothing drop out).
* β_S: within-keyword least-squares slope of S on x (`geo_drivers.within_keyword_slope`).
* 200 keyword-bootstrap draws (`intent_stages.keyword_draws`, seed 20261007) shared by every quantity;
  200 within-keyword shuffles of x (`intent_stages.shuffle_draws`, seed 20261008). 95% intervals are the
  2.5/97.5 percentiles, 90% intervals the 5/95 percentiles; shares are ratios computed in each replicate.
  Permutation p = (1 + #|null| ≥ |observed|)/(n + 1); permutation p is reported, never decisional.
  Model-based generator quantities use the first 100 of the same draws.

## Estimands

* Chain: β_K = β_R0 + (β_R − β_R0) + (β_C − β_R) + (β_P − β_C) + (β_I − β_P) + (β_K − β_I), exact by linearity.
  R0 is labelled mechanical (lexical replay). β_R − β_R0 is query rewriting, β_P − β_C the reranker,
  β_I − β_P the shown-order weighting, β_K − β_I the generator's own choices. s_gen = (β_K − β_I)/β_K.
* Null selectors: identity (I), random (E[K] = P), lexical selector (top-k by BM25 of the prompt over the
  reranker's candidates, Parallel).
* Δ_gen = β_x(E_full[K]) − β_x(E_blind[K]): expected rank-weighted cited intent under the fitted keep model
  (Chamberlain conditional logit given the cited count) and order model (Plackett–Luce with shown-slot
  effects), 200 Monte-Carlo draws per answer with common random numbers; "blind" refits both models without
  the intent block A1 (primary) or zeroes the A1 coefficients (secondary). Parallel: keep fixed at the observed
  set (99% of shown links are kept), order only. Model check: β_x(E_full[K]) − β_K with its interval.
* SESOI = 0.015 u per unit x (about 25% of the pooled cited-intent slope 0.06), fixed in absolute terms.

## Decision rules

**C1 (admission).** Primary Qwen · Reactive, secondary Qwen · Parallel.
* Supported if (a) the 95% interval of s_gen lies below 1/3 and (b) β_R − β_R0 and β_P − β_C both have
  95% intervals above 0; "supported" also requires (a) in the controls variant and in each engine.
* Narrowed if (a) holds at 1/2 but not at 1/3, or (b) holds for the reranker only ("mainly the reranker").
* Failed if the upper bound of s_gen is at least 1/2.

**C2 (selection is intent-insensitive in its effect on the cited-intent slope), Qwen only.**
* Supported if the 90% interval of Δ_gen (drop-A1 refit) lies inside [−0.015, 0.015] in both methods (TOST).
* Narrowed if that fails but the 95% upper bound of |Δ_gen| is below 0.030 in both methods: the claim becomes
  "the generator contributes X% [CI] of the shift, less than half".
* Failed if the 95% lower bound of |Δ_gen| exceeds 0.015 in either method: the paper keeps C1 and reports the
  generator's intent sensitivity as a finding.
* Coefficient-level nullity is not claimed (Reactive keep has non-zero intent coefficients of opposite sign).
  Reported, not decisional: zeroed-coefficient Δ_gen, s_gen, two-way fixed-effects coefficients, no-slot and
  score-control variants, score-matched adjacent pairs (|Δ logit score| < 0.1; 0.25 sensitivity), the slope of
  the cited count on x.

## Robustness reported (not decisional unless named above)

Controls variant (log prompt words, keyword-in-prompt, candidate slot of the prompt generator); a separate
"given intended position" variant with the lattice target; engine strata; leave-one-keyword-out; noise floor
within (keyword, target); lexical-overlap control in the reranker selection model; Llama R0 and K slopes from
the published answers (Llama traces are not on the Mac).

No variants are added after the first run; anything added later is labelled exploratory.

## Addendum B1 (2026-10-08, before any Llama-trace or confirmation-keyword result of this study)

The HoreKa run applies the rules above unchanged to both models and to the held-out confirmation keywords
(`--split confirmation`, `analysis/docs/horeka-steelman.sbatch`). Choices that the Mac run did not need, fixed now:

* **Keep model.** The keep decision is modelled when at least 25% of a stratum's answers keep some and drop some
  of their shown links; otherwise the cited set is held at its observed value and only the order is modelled.
  This reproduces the Mac run (Qwen · Parallel 1.5% informative: fixed; Qwen · Reactive 41%: modelled) and models
  Llama's keep decision wherever it drops links.
* **Strata and verdicts.** C1 is judged per model × method, Reactive Loop primary for each model. C2 is judged per
  model across its two methods, with the same SESOI (0.015 u per unit x) and the same thresholds.
* **Lexicon.** The action-word list is frozen as computed by the Mac run (`analysis/steelman/lexicon.json`, from
  confirmation-keyword prompt texts and x only), so both runs use the same instrument.
* **Lexical shortlisting refits.** Skipped on HoreKa (`--selection-draws 0`): reported-only, and about 20 minutes per
  refit at that size. The BM25 selector comparison is kept.
* **Agent queries.** Read from the intent trace extract, joined on the cell fingerprint (the funnel extract does
  not store them).
* **Prompt fields.** Read from the registration's `population-prompts.jsonl`. A control a file lacks is left out
  and recorded (`controls_used`); the lattice-target variant and the noise floor are skipped if the target is absent.
* **Published-answer rows.** Not used on HoreKa: both models are analysed from their own traces.

## Addendum B2 (2026-10-08, after the Mac lexical result; before any HoreKa result)

The lexical selector was pre-registered with BM25. V2 does not use BM25 and the paper does not mention it: the
retriever that produced the logged results is the frozen search's own word-overlap rule (exact keyword match;
then 4 × shared keyword words + shared title/snippet words; then stored position; then a hash). The selector is
therefore switched to that rule, applied to each reranker event's candidates against the text the reranker scored
against. This change was made after the BM25 result was seen (92% / 119% of the reranker's increment). Both
versions are reported; the BM25 one is labelled as the pre-registered record and is not used in the paper text.
Neither is decisional for C1 or C2.

## Addendum B3 (2026-10-08, before any full-run result; plan in `analysis/docs/handoff/2026-10-08_full-run-plan_handoff.md`)

The full run covers both models, all 1,011 keywords and all three conditions from the existing traces. Every part is
computed separately on the exploration keywords and on the held-out confirmation keywords (`funnel_study.exploration_keyword`).
**The confirmatory verdicts are those of the confirmation keywords.** Already seen: the Qwen exploration results of this
study; Llama's prompt-text replay and cited-intent slopes from the published answers; the quick supply look below. Not
seen: anything from Llama traces, anything on the confirmation keywords, and every interval and test of the parts below.

**Census (descriptive).** Cells per model × method × engine × condition × split: planned (26,000 prompts × 12 cells per
model), completed on the Hub, present in the traces, passing the extraction checks, and in the analysis sample; failures
by reason; duplicates dropped; keyword coverage.

**Supply of action-ready pages (claim: "bounded by how few action-ready pages exist").** Page intent u is the page's
prompt-scale percentile. Per keyword × engine: number of usable rows, share with u ≥ .6, .7 and .8, maximum u and SD of u.
Oracle K for answer i over a pool S: the L_i rows of S closest to x_i on u (ties by row id), rank-weighted as K. Pools:
the keyword's own usable rows (U), the retrieved rows (R), the reranker's candidates (C), the shortlist (P) and the whole
engine snapshot. Reported: oracle slopes β_oracle(S); utilisation β_K / β_oracle(S); mean x, mean K and mean oracle(U) by
x decile; β_K within terciles of keyword supply (share of the keyword's rows with u ≥ .6, both engines pooled; tercile
cut points from the keyword table, fixed before results) and the top − bottom tercile contrast.
* *Supported as "the shift is bounded by scarcity"* only if, in every model × method stratum, (a) the ceiling gap
  mean(x) − mean(oracle(U)) over answers with x ≥ .9 has a 95% interval above 0, (b) the top − bottom tercile contrast of
  β_K has a 95% interval above 0, and (c) utilisation β_K / β_oracle(U) has a 95% lower bound of at least 0.5.
* *Narrowed to "few action-ready pages exist, which caps how close cited sources can come to the most action-ready
  prompts"* if (a) holds in every stratum but (b) or (c) does not.
* *Failed* if (a) does not hold: the paper drops the scarcity sentence.
Quick look already seen (Mac, Qwen exploration and published answers, no intervals): 0.76% of rows with u ≥ .8; 7.6% of
keywords with any such row; own-row oracle slopes .23–.37 against observed .06–.09; utilisation of the shortlist oracle
about 60% and of the retrieved-pool oracle about 20–25%.

**Ablated condition (reported, not decisional).** Pairs of natural and ablated cells with the same model, prompt, engine
and method (the target URL is removed before the rerank). Per pair ΔP and ΔK (ablated − natural); reported: their means,
the within-keyword slope of ΔK on ΔP (pass-through) and the slope of ΔK on x, with keyword-bootstrap intervals, by model ×
method. The shuffled condition acts before the rerank and leaves the shown order unchanged; it is not used as a slot test.

**Keyword share of queries among prompts that name their keyword (reported).** The share of the agent's queries containing
every keyword token, at x < .2 and x ≥ .8 and its within-keyword slope, restricted to prompts containing every keyword token.

**Rules carried over unchanged:** C1 and C2 as above (C2 per model, Reactive primary for C1), addenda B1 and B2, seeds,
draws, SESOI, and the exploratory label for the keyword-mention follow-up.

*Note to B3 (2026-10-08, after a 30-draw smoke run of the new parts on the Mac exploration data, before any HoreKa
result).* About 40% of keywords have no row with u ≥ .6, so tercile cut points on that share alone leave the lowest
tercile empty. Terciles are therefore cut on the rank of (share of rows with u ≥ .6, then mean u) over all 1,011
keywords. Seen in that smoke run (Qwen, exploration, exploratory): supply verdict "narrowed" (own-row oracle utilisation
0.19–0.26); ablation pass-through 0.69–0.86; among prompts that name their keyword, the keyword share of queries does not
fall with x (Reactive 0.62 → 0.55, Parallel 0.27 → 0.31).

## Computational note C1 (2026-10-08, before any GPU result on HoreKa data): an optional PyTorch backend

A computational change only: estimands, features, standardisation, ridge (1e-4), seeds (20261007, 20261008), draw counts,
SESOI, Monte-Carlo random numbers and every decision rule are unchanged. The heavy fits (the funnel stage models, the
generator decisions, the steelman generator refits and Monte Carlo, the two-way fixed effects) may run on
`analysis/interpretability/pipeline/torch_fits.py` (float64; damped Newton with exact gradients and Hessians on the same
objectives) with `--backend cuda` on HoreKa `accelerated` nodes. The numpy/scipy path stays the default and the reference.
The Monte Carlo uses the same numpy random numbers on both backends; caches carry the backend so CPU and GPU units of a
task are never mixed.

An estimator is routed to the GPU only if it passes these gates, recorded in `analysis/fullrun/validation/REPORT.md`:
* synthetic problems (`analysis/tests/test_torch_fits.py`): coefficients within 1e-7 and objective within 1e-10 of a
  tight-tolerance CPU optimum; inclusion probabilities within 1e-12; Monte-Carlo twins identical to the numpy code; batched
  fixed effects within 1e-8 of the CPU fits; unit ranges assemble identically;
* real data (Mac exploration tables): funnel coefficients within 1e-4 of the cached CPU fits with an objective no worse
  than the CPU optimum; steelman Δ_gen within 1e-4 and its interval endpoints within 2e-4 with the same TOST verdict;
  fixed-effects coefficients and interval endpoints within 1e-6.
Near-separable fits are not gated (float noise of 1e-16 moves their optimum by about 1e-3 even on the CPU). If a gate
fails, that estimator stays on the CPU path and the task list routes it there.
