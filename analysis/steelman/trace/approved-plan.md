# Steelman plan: "intent works at the shortlist" from existing data only

Work happens on `codex/acl-figures-20261006` in `.worktrees/acl-figures` (clean at a7da82d). New code goes in `analysis/steelman/`; nothing in `analysis/scripts/`, `analysis/interpretability/pipeline/`, the frozen pools or the existing result folders is modified. No model inference, no reranker rescoring, no embeddings. Everything below is re-analysis of the Qwen exploration traces and the published answer rows already on the Mac. Outputs go to `~/Hamburg/geodml-inputs/steelman-v1/`.

## 1. Context

The ACL ARR draft is being rewritten around one claim: intent gets a page shortlisted (C1), topic and shown order decide its rank once shown (C2). The supporting deck numbers are exploratory (237 exploration keywords, Qwen traces) and several are ratios of point estimates without intervals. Six objections (O1–O6) can be raised. This plan says which existing tables answer each, what each result would change, and pre-registers the decision rules before anything runs.

Four facts found during exploration shape the plan:

- **"Intent ≈ 0%" at the generator is a Pratt-share statement, not a coefficient statement.** Qwen · Reactive keep: intent alignment +0.32 per SD, CI [0.19, 0.46], permutation p = 0.05; page intent × (x − ½) −0.55, CI [−0.81, −0.35], p = 0.02. Their Pratt shares (+0.009, −0.009) cancel. C2 as "direct intent effect indistinguishable from zero" cannot be defended on the coefficient scale with the existing fits. It is pre-registered instead as a materiality claim on the slope scale (§5).
- **Slot carries topic, not intent.** Within-answer correlation of shown slot with intent alignment is −0.03 (Parallel) / −0.02 (Reactive); with topic similarity −0.28 / −0.22. O2 ("slot absorbs intent") is weak on its face; A4 stays cheap and A5 becomes the coefficient-level steelman.
- **The generator's own share of the slope is small at the point estimate** (seen during the design review, declared in PREREG): Parallel β_P .0507, β_I .0546, β_K .0597 → (K − I)/K = 8.4%; Reactive .0574 / .0611 / .0681 → 10.3%. Intervals, controls and every model-based quantity are unseen.
- **Llama traces are not on the Mac** (0 bundles; the Oct 7 download died on the fourth object). Llama keeps 40% of shown links under Parallel versus Qwen's 99%, so the keep decision with real room is Llama's. A14's trace part is a HoreKa CPU parsing job (no inference), outside the Mac scope; C2 is scoped to Qwen unless it runs.

Also found: the cited count falls with x under Reactive (−0.73 links per unit x; Parallel −0.05). Every estimand that conditions on the cited count is silent about this channel; it is reported as its own generator behaviour.

## 2. Repo map

| Stage | Produced by | Logged where (local) | Fields |
|---|---|---|---|
| Queries | `analysis/interpretability/pipeline/agentic_search.py` — Parallel: 3 queries in one call (`:610-693`); Reactive: 1–3 sequential queries (`:696-789`); observed Reactive search counts 1: 29%, 2: 69%, 3: 1.4% | `funnel-extract-exploration-v1/queries.jsonl.gz`; `items.npz: search, rank` (first search returning each row) | query strings |
| R0 | not in production; analysis-time replay with the frozen search's rule (`funnel_rows.LexicalIndex`, `funnel_study.py:177-197`) | `funnel-replay-v1/replay.npz` (20 rows per prompt × engine) — use this one, not `u.npz["replay"]` (keyword rows only) | row ids |
| R | frozen snapshot adapter, top 20 per query by (exact keyword, overlap, stored position, hash) | `items.npz` per answer (all rows incl. off-keyword) | row, search, rank, scored, presented, ranked |
| C, scores | `bge-reranker-v2-m3` on `(query, title\ntext)` (`agentic_search.py:244`); top-k by (−score, source_index) (`:186`), 0 of 88,756 events deviate. **Parallel: one rerank of the deduplicated union against the full user prompt, k = 7 (`:668-673`). Reactive: one rerank per search against the agent's own query, k = 3 (`:759-764`).** Scores are sigmoid-scale (10% exact zeros, median 5e-4); use logits. | `funnel-assembled-exploration-v1/candidates.npz` | event, answer, row, **score for every candidate**, selected, topic, on_keyword |
| P, slot | Parallel: score order. Reactive: blocks in search order, each in score order, URL-deduplicated (mean shown 4.1) | `presented.npz` | answer, row, slot, rank, topic, on_keyword |
| K, order | generator returns JSON `{"ranking": [...]}` of shown ids (`agentic_search.py:1160-1175`); nothing parsed from answer text | `presented.npz: rank` (−1 = dropped) | kept = rank ≥ 0; order = rank |
| Page features | `row_features.npz` (23,767 snapshot rows) | | `page_intent_z`, `u`, blocks A3/A4/B/C1/C2, `corpus_row` |
| Row text | `funnel-features-v1/rows.parquet` | | title, snippet, url, domain, doc_id |
| Prompt text | `ARR_ACL_CycleOct2026/hf-min/snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz` (26,008) | | `question`, `keyword`, `target_normalized_axis_1`, `target_index`, `candidate_slot`, `round_index`, `generation_seed`; 8–41 words |
| Answer rows (both models, all keywords) | `funnel-review-v1/answers.parquet` (621,003) | | x, ranking_len, search_count, shown, cited_u (K), r0_u (R0) |
| Existing fits | `funnel-report-exploration-v2/results.json`, `funnel-generator-decisions-v1/decisions.json`, `funnel-explore-v1/explore.json`, `funnel-contrasts-v1`, `funnel-keywords-v2` | | bootstrap coefficient vectors of keep/order fits are **not** stored; Δ_gen needs refits |

Reused code (no new frameworks):
- `analysis/scripts/funnel_study.py:330-480`: `load_assembled`, `answer_arrays`, `strata`, `split_mask`, `exploration_keyword`, `_features`, `standardisation`, `stage_task`.
- `analysis/interpretability/pipeline/funnel_models.py`: `Feature`, `Design` (`matrix(drop_block=)`, `refresh`), `Stage`, `admission_data`, `fit_admission`, `_inclusion` (elementary-symmetric-polynomial recursion, `:87-109`), `estimate_blocks(parts=, deadline=)`, `contrast_from_replicates`, `rr_decomposition`.
- `analysis/interpretability/pipeline/page_readiness_ordering.py:187-229`: `fit_choice_model` (PL with position effects; `levels = 1` removes them).
- `analysis/interpretability/pipeline/intent_stages.py`: `Stages`, `stage_values`, `top_weighted`, `oracle_values` (its `presented` entry is the identity selector), `keyword_draws`, `shuffle_draws`, `mediation`, `analyse_stratum` (cross-check only).
- `analysis/interpretability/pipeline/geo_drivers.py`: `within_keyword_slope`, `_interval`, `_p_value`.
- `analysis/scripts/intent_stages_study.py:511-531`: `ResultCache`; `funnel_importance.py:89-144`: `decisions()` design-row builder (re-implemented with variant knobs, equality-tested against it); `funnel_explore.py:38-53`: `tokens`, `ACTION` lexicon.

Held-out results: none exist anywhere under `~/Hamburg` (no `funnel-report-v1`, `intent-stages-v1`, confirmation or holdout folder; the deck's todo slide says nothing confirmatory has run). Split rule: `funnel_study.exploration_keyword` (278 exploration keywords by rule, 237 with traces locally, 19,944 natural Qwen answers).

## 3. Corrections to Section 1 and O1

1. **Reactive shows up to 9 links, not 7**, and in practice 4.1 on average (3 per search, 1–3 searches, URL dedup). "Keeps all 7" is Parallel only.
2. **Stage shares (18–20 / 18–34 / 35–47 / 13–15%) are hand ratios of point estimates** from `funnel_explore` slopes on the u scale, each slope on its own finite subset, no intervals; "reranker" there is P − R and includes URL dedup (Parallel). Superseded by the common-sample chain with intervals (A1).
3. **"Intent's share ≈ 0%" is a Pratt share**; Reactive keep intent coefficients are non-zero with opposite signs (§1). Parallel keep (148 informative answers) separates: its CIs do not contain the point estimates; report it as "no decision to model".
4. **Off-topic share of the shortlist:** 0.29 → 0.49 (Parallel), 0.27 → 0.48 (Reactive). Keyword-in-query 27.2% → 12.5% and 61.4% → 21.0% are correct.
5. **97% / 83%** is the share of the within-U log-RR of intent alignment arising at step "shortlist | C" (reranker selection among its candidates, `rr_decomposition`), not a share of the slope; the paper must keep the two decompositions apart.
6. **Two keep rates for Qwen · Reactive:** 77.4% (traces, deduplicated shown) versus 54.9% (published rows, where `shown` = 3 × search_count before dedup). Use the trace figure. Llama from published rows: 39.6% kept (Parallel, 7 shown), 7.5% of answers keep all 7; Reactive 2.45 of ≤ 7.4.
7. **No style seed exists in V2.** Prompts carry `generation_seed` (a random 64-bit integer, unusable as a regressor), `candidate_slot`, `round_index` and a lattice target; ≈ 2 realizations per (keyword, target); the target explains 78% of x's within-keyword variance, so it is not a control but a separate "given intended position" variant.
8. **O1 is correct as stated** for both methods (verified in code). Addition: Reactive's slot = (search block, rank against the agent's query); no score against the user prompt exists there.
9. **R0** is a replay (analysis time); its rule is lexical, so "mechanical" is right. It covers all 20 rows for the prompt, including off-keyword rows, which matches R (37–40% of shown rows are off-keyword).
10. The shuffled and ablated conditions run **before** the rerank (`agentic_search.py:662-672`, `:754-763`), so shown order is score order in every condition: there is no slot randomization anywhere in the data.
11. "About 621k answers" (621,003) and the four cited-intent slopes are confirmed.

## 4. Feasibility table

Effort = coding; run time on the Mac in parentheses. Priority 1 = changes what the paper can claim.

| # | Analysis | Needs | Exists? | Effort | Priority |
|---|---|---|---|---|---|
| A1 | Chain on one common sample (R0, R, C, P, I, K all finite), shared draws and shuffles, per method; Reactive primary | items, presented, row_features, replay | yes | 3 h (< 1 min) | 1 |
| A3 | R0 share labelled mechanical; query rewriting = β_R − β_R0 | same | with A1 | — | 1 |
| A10 | Null selectors: identity I (first L shown in shown order), random (E[K] = P exactly), lexical selector from A2(ii); exact split K − P = (I − P) + (K − I) | same | `oracle_values["presented"]` | 1 h | 1 |
| A9 | Permutation null (200 within-keyword shuffles) for every slope, increment and share; reported, not decisional | `shuffle_draws` | yes | with A1 | 2 |
| A8 | Intent-blind generator Δ_gen and TOST (§5) | fitted keep/order models, MC, bootstrap refits | no | 7 h (≈ 15 min per stratum, 8 workers) | 1 |
| A5 | Snapshot-row fixed effects: two-way FE linear probability model (answer + row, alternating projections) for keep and for "credit" y = w_r/W_L; informative answers, rows shown ≥ 2 times | presented, row_features | feasible (Reactive: 2,668 rows both kept and dropped, 24,539 of 40,997 shown rows) | 3 h (seconds) | 1 |
| A14 | Llama: R0 and K rows from published answers with keyword draws; keep rates; trace-based cells blocked | `funnel-review-v1` | partial | 1 h; HoreKa CPU extraction (rerun of job 5185751 with 40 workers) written up as the blocked branch | 1 (published rows) |
| A12 | Controls in the stage regressions: log prompt words, keyword-in-prompt (79.8% yes), candidate_slot; chain identity holds by linearity. Separate variant: given `target_index` | prompts | yes | 1.5 h | 2 |
| A13 | Noise floor: within-(keyword, target, method, engine) SD of stage values and within-cell slopes on x (1,186 of 2,620 natural cells have ≥ 2 prompts) | prompts | yes | 1 h | 2 |
| A15 | Engine × method strata for every closed-form quantity; leave-one-keyword-out and jackknife influence for chain shares; heavy fits keep the bootstrap | all | no held-out results → all exploratory | 1.5 h | 2 |
| A2 | Lexical mediator for the Parallel reranker: (i) overlap of (title + snippet) with prompt tokens minus keyword tokens, log1p, added to the P|C design; paired β_A1 difference via `contrast_from_replicates`; Reactive: overlap with the agent's query, prompt overlap as a null control; (ii) BM25(prompt) top-7 selector as a reranker null; action lexicon fixed from confirmation-keyword prompt text (text only, no outcomes) | rows.parquet, prompts, queries, candidates | text yes | 5 h (≈ 3 min per P|C refit) | 2 |
| A4 | Keep/order without slot, and with the reranker logit score as a control (score is absent from every existing design) | assembled, `decisions()` re-implementation | fits exist with slot only | 1 h (10 min) | 3 |
| A6 | Within-answer correlation of slot with u, alignment, z·(x − ½), topic, by method, keyword-bootstrap CI | presented | computed once in review | 0.5 h | 3 |
| A11 | Replaced: conditioning on "dropped ≥ 1" conditions on a generator outcome that depends on x. Report instead the slope of L and L/n on x per cell, plus the droppers-only chain as a labelled descriptive | presented | yes | 0.5 h | 3 |
| A7 | Reframed: not an RD (assignment is deterministic in text; the out-of-shortlist item has no outcome). Score-matched adjacent shown slots (|Δlogit| < 0.1 primary, 0.25 sensitivity; exact ties dropped; Reactive within-block only): pair-level conditional logit of "later slot ranked higher / kept while earlier dropped" on Δalignment, Δz·(x − ½), Δtopic | candidates scores + slots | yes (21.7% / 13.9% of adjacent pairs) | 3 h (minutes) | 3 |

Drop order if time runs out: A7, A6, A4, A2(ii), A13, A15 (LOKO part).

## 5. Pre-registration (`analysis/steelman/PREREG.md`, committed before the first run; its sha256 recorded in results)

Scope. Qwen, natural condition, exploration keywords with traces (237). Already seen: the deck's per-metric stage slopes; the identity point estimates in §1; the existing keep/order coefficients. Unseen: every interval, permutation p, controls and lexical variants, Δ_gen, FE and pair results. The same rules apply unchanged to the confirmation split if HoreKa confirmation lands.

Scale and inference. u scale (prompt-scale percentile). Common sample = answers with finite R0, R, C, P, I, K. 200 keyword-bootstrap draws (`keyword_draws`, seed 20261007) shared by every quantity; 200 within-keyword shuffles of x (`shuffle_draws`, seed 20261008); 95% intervals = 2.5/97.5 percentiles, 90% = 5/95; permutation p = (1 + #|null| ≥ |obs|)/(n + 1); shares are ratio replicates.

Definitions. β_S = within-keyword LS slope of stage value S on x. I_i = rank-weighted mean (w_r = 1/log2(r + 2)) of the first L_i shown links in shown order, L_i = observed cited count. Chain: β_K = β_R0 + (β_R − β_R0) + (β_C − β_R) + (β_P − β_C) + (β_I − β_P) + (β_K − β_I). s_gen = (β_K − β_I)/β_K.
Δ_gen = β_x(E_full[K_i]) − β_x(E_blind[K_i]): expected rank-weighted cited intent under the fitted keep model (Chamberlain given L_i, sequential sampling with the `_inclusion` recursion) and order model (PL with slot effects, Gumbel-max), M = 200 draws per answer with common random numbers; "blind" = both models refitted without the A1 block (primary) or A1 coefficients zeroed (secondary). Parallel: keep fixed at the observed set, order only. Model check: β_x(E_full[K]) versus β_K with CI of the difference. Bootstrap refits both models per keyword draw (100 draws) through a thin loop over `fm._fit` keeping the full coefficient vector (slot effects included), checkpointed with `ResultCache`.
SESOI = 0.015 u per unit x, fixed in absolute terms (≈ 25% of the pooled β_K ≈ 0.06 already seen). ε = 0.1 on the reranker logit scale.

Decision rules (the request's literal C1 rule compares β_P/β_K with β_P/β_I, which is sign-inverted because β_K > β_I; it is restated below).
- **C1 supported** (primary Qwen · Reactive, secondary Qwen · Parallel) if (a) the 95% CI of s_gen lies below 1/3, and (b) the query-rewriting increment β_R − β_R0 and the reranker increment β_P − β_C both have 95% CIs excluding 0 on the positive side; "supported" also requires (a) in the controls variant and in each engine. **Narrowed** if (a) holds at 1/2 but not 1/3, or (b) holds for the reranker only (claim becomes "mainly the reranker"). **Failed** if the upper bound of s_gen ≥ 1/2. Parallel is reported with its reranker column flagged "scored against the user prompt" and next to the lexical-selector null.
- **C2 supported (Qwen)** if the 90% CI of Δ_gen (drop-A1 refit) lies inside [−0.015, 0.015] in both methods (TOST). **Narrowed** if TOST fails but the 95% upper bound of |Δ_gen| < 0.030 (restated as "the generator contributes X% [CI] of the shift, less than half"). **Failed** if the 95% lower bound of |Δ_gen| > 0.015: the paper keeps C1 and reports the generator's intent sensitivity (both signs) as a finding. Reported, not decisional: the zeroed-coefficient variant, s_gen, the 2WFE coefficients, the no-slot and score-control variants, the score-matched pairs, the slope of L on x. Coefficient-level nullity is not claimed.
- No variants are added after the first run; anything further is labelled exploratory.

## 6. Execution order (after approval)

1. `analysis/steelman/PREREG.md` and `__init__.py`; commit (sha256 of PREREG recorded by every run).
2. `tables.py`: one loader joining answers, items (all rows), presented, candidates (logit score, Reactive block index), row_features, replay, prompts (text, words, keyword-in-prompt, candidate_slot, target_index) into a `Stages`-like structure on the u scale with R0 and I; common-sample masks. Fixture-based tests.
3. `chain.py`: `within_keyword_ols` (generalises `mediation`), chain terms with CI and permutation p, null-selector variants, controls variant, target variant, engine strata, LOKO/jackknife, side-by-side table builder; cross-check against `analyse_stratum` (`share_of_ranking:reordering` equal to 1e-12). Run.
4. `generator.py`: keep/order builders with knobs (`include_slot`, `score_control`, `drop_blocks`) equality-tested against `funnel_importance.decisions`; `sample_admitted`, `sample_order`, `expected_cited_intent` (CRN MC), `delta_gen` with keyword bootstrap and TOST; `two_way_fe` (LPM, alternating projections, weighted for keyword draws); L-on-x slopes; A4 and A6. Run (longest step).
5. `lexical.py`: tokens, overlap features (prompt \ keyword; agent query), BM25 selector, lexicon from confirmation prompt text; P|C refit variant wrapping `stage_task` + `estimate_blocks`. Run.
6. `pairs.py`: score-matched adjacent pairs (A7), if time remains.
7. `decide.py`: pure verdict functions with module-constant thresholds; `__main__.py` subcommands `chain | generator | fe | lexical | pairs | report` (`python -m analysis.steelman ...`), common flags `--input-root --prompts --output --seed --bootstrap --permutations --workers --stop-after-minutes` (exit 4 on deadline), atomic output folders, refuse existing.
8. `report.py`: `RESULTS.md` (objection → analysis → result with CIs → verdict), the side-by-side table, draft paper text, `manifest.json` (commit, seeds, input digests, PREREG sha256).
9. Llama published-row rows and keep rates into RESULTS as supporting evidence; HoreKa CPU extraction commands for the Llama traces into the handoff as the blocked branch (not run here).
10. Tests (`analysis/tests/test_steelman_*.py`), dated handoff + index line, commit.

## 7. Deliverables

- `analysis/steelman/{PREREG.md, RESULTS.md}`; `~/Hamburg/geodml-inputs/steelman-v1/{results.json, stage_shares.csv, generator.csv, manifest.json}`.
- Side-by-side stage-share table per method: rows R0, R − R0, C − R, P − C, I − P, K − I, K − P, β_K; columns deck original, common sample, + controls, lexical-selector reranker (Parallel), identity baseline, random baseline; cells share [CI].
- Draft paper text: the claim sentence as supported; a limitations paragraph naming the fixed-set test as the decisive experiment not run, plus the cited-count channel and the Qwen-only scope of C2; one sentence positioning against Tannenbaum (2026) and Martinez (2026).
- Handoff in `analysis/docs/handoff/` with the index line.

## 8. Out of scope (one decisive run)

**Fixed-set test.** Show the same shortlist (same links, same order) to the generator under low-x and high-x prompts of the same keyword; the within-keyword slope of cited intent on x then measures the generator's intent sensitivity with P fixed by design. About 237 keywords × 6 prompt pairs × 2 methods × 2 models ≈ 11k generations; cost on this testbed unknown, order of a few four-GPU hours [H].

## 9. Verification

- `.venv/bin/python -m pytest -q analysis/tests/test_steelman_*.py` plus the existing funnel, intent-stages and geo-drivers suites (94 pass today).
- Planted-math fixtures (patterns of `test_intent_stages.py:36-70`, `test_funnel_models.py:15-54`): identity ranking gives K − I = 0 exactly and increments sum to β_K; controls keep the exact sum (Frisch–Waugh); `within_keyword_ols` reduces to `within_keyword_slope` and `mediation`; random selector E[K] = P; `sample_admitted` frequencies match `_inclusion`; MC expected K matches exact enumeration for n ≤ 4; planted θ_A1 = 0 gives Δ_gen ≈ 0 and planted θ_A1 ≠ 0 is recovered; CRN variance check; default generator variant reproduces `funnel_importance.decisions` rows and column names on the pipeline fixture (`test_funnel_study.py:61-78`); 2WFE recovers planted coefficients and equals dummy OLS on a small problem; pair extraction respects ties and Reactive blocks; verdict functions on boundary cases; CLI smoke with refuse-overwrite.
- Every number in RESULTS.md regenerated by `python -m analysis.steelman report` from the JSON outputs.
