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
