# Steelman trace: everything done, run and decided (2026-10-07 to 2026-10-08)

A complete record of the steelman work on the claim "prompt intent reaches the cited sources at the
shortlist, not in the generator's final ranking". Times are local (UTC+9), from commit times, file
modification times and the `seconds` field each part records. Branch `codex/acl-figures-20261006`,
worktree `.worktrees/acl-figures`. Nothing in this trace ran on a cluster.

Contents of this folder:

| Path | What it is |
|---|---|
| `TRACE.md` | this record |
| `approved-plan.md` | the plan Valerian approved before any code was written |
| `outputs/mac-exploration-v1/` | every result file of the Mac run (copies of `~/Hamburg/geodml-inputs/steelman-v1/`), `files.sha256`, `caches.sha256`, and `checkpoint-caches.tar.gz` (all bootstrap replicates of the generator refits and the lexical shortlisting refits) |
| `logs/` | stdout/stderr of every background part run |

## 1. Request

Valerian asked for the strongest honest case, from existing data only, for two claims:

* **C1 (admission).** Intent reaches the cited sources mainly by changing which documents enter the shortlist
  (queries and reranker), not through the generator's choice among shown documents.
* **C2 (selection).** Given the shortlist and its order, the generator's keep and order decisions follow topic
  and slot; the direct intent effect is indistinguishable from zero.

Constraints: no new inference of any kind; new code only in `analysis/steelman/`; fixed seeds, keyword-clustered
intervals, every number regenerable; a pre-registration committed before analysis; objections O1–O6 answered by
analyses A1–A15 in value-per-hour order. Follow-up requests (2026-10-08): make the code run on HoreKa, give the
HoreKa commands, and save a full trace here.

## 2. Exploration findings (read-only, before the plan)

Three read-only searches mapped the pipeline code, the analysis code and the data on disk; a design review
stress-tested the estimands. Facts that shaped the plan:

* **Reranker inputs, verified in code** (`analysis/interpretability/pipeline/agentic_search.py`): the same
  `BAAI/bge-reranker-v2-m3` scores `(query, title + "\n" + snippet)`, top k by (−score, source index).
  Parallel Expansion: one rerank of the deduplicated union of three searches against the **full user prompt**,
  k = 7, shown in score order. Reactive Loop: one rerank per search against **the agent's own query**, k = 3 per
  search, shown as search blocks in score order, URL-deduplicated (mean 4.1 shown, at most 9).
* **Scores are logged for every candidate** (`candidates.npz`); stable snapshot row ids exist (`corpus_row`).
* **R0 is not in production traces**; it is an analysis-time replay of the frozen search on the prompt text
  (`funnel-replay-v1`, 20 rows per prompt × engine).
* **The shuffled and ablated conditions act before the rerank**, so shown order is score order in every condition;
  no slot randomisation exists.
* **The deck's stage shares (18–20 / 18–34 / 35–47 / 13–15%) were hand ratios of per-metric slopes** from
  `funnel_explore.py`, each on its own subset, without intervals, "reranker" including deduplication.
* **"Intent ≈ 0%" at the generator was a Pratt share.** The Reactive keep coefficients are non-zero and of
  opposite sign (alignment +0.32 per SD, interaction −0.55), so C2 cannot hold on the coefficient scale.
* **Llama traces were not on the Mac** (the 2026-10-07 download wrote 0 bundles). Llama keeps about 40% of shown
  links under Parallel versus Qwen's 99%.
* **No held-out (confirmation-keyword) result existed anywhere.** Split rule: `funnel_study.exploration_keyword`
  (278 exploration keywords by rule, 237 with traces on the Mac).
* **No style seed exists in V2 prompts**; prompts carry a lattice target, candidate slot and generation seed.
* **Design-review diagnostics** (seen before the pre-registration and listed in it): identity-selector point
  estimates; shown slot correlates with topic (−0.22/−0.28) and barely with intent (−0.02/−0.03); the cited count
  falls with x under Reactive (−0.73 links); the lattice target explains 78% of x within keyword.

Deck corrections found: off-topic shortlist share 0.29→0.49 / 0.27→0.48 (not 0.28→0.48); Reactive shows ~4 links
(not 7); Qwen Reactive keeps 77.4% in traces versus 54.9% in published rows (published `shown` counts before URL
deduplication); 97%/83% is a log-RR decomposition, not a share of the slope.

## 3. Pre-registration

* `5458494` (2026-10-07 22:22) `analysis/steelman/PREREG.md`, sha256 prefix `ff9e04839a29`, recorded by every
  Mac part. Estimands: the common-sample chain β_K = β_R0 + (β_R − β_R0) + (β_C − β_R) + (β_P − β_C) + (β_I − β_P)
  + (β_K − β_I); Δ_gen = slope of expected cited intent under the fitted keep/order models minus the same with
  the intent block removed; SESOI 0.015 u per unit x. Decision rules for C1 (generator share upper bound below
  1/3, query and reranker increments above 0) and C2 (TOST of Δ_gen inside ±0.015 in both methods).
* The literal C1 rule in the request compared shares that are sign-inverted by construction; the pre-registration
  restates it (stated in the plan and the PREREG).
* `8977cd2` (2026-10-08 00:37) addendum B1 for the HoreKa run, before any Llama-trace or confirmation result:
  keep model when ≥ 25% of answers are informative, C2 per model, frozen lexicon, lexical shortlisting refits
  skipped, queries from the intent trace extract, prompt fields from `population-prompts.jsonl`.

## 4. Code

`analysis/steelman/` — `tables.py` (loader: stage values R0, R, C, P, I, K per answer; prompt metadata; shown-row
reranker logits and search blocks; agent queries), `chain.py` (chain with shared keyword draws and shuffles,
controls, lattice-target variant, engine strata, leave-one-keyword-out, noise floor, published Llama/Qwen rows,
cross-check against `intent_stages.analyse_stratum`, exploratory follow-up), `generator.py` (keep/order refits,
exact conditional-logit subset sampling and Gumbel Plackett–Luce with common random numbers, Δ_gen, variants
without slot and with the reranker score, two-way fixed effects, slot correlations), `lexical.py` (word overlap in
the shortlisting model, BM25 selector, action-word lexicon), `pairs.py` (score-matched adjacent shown links),
`decide.py` (verdicts), `report.py` (RESULTS.md, stage_shares.csv, manifest), `__main__.py` (CLI),
`lexicon.json` (frozen list). Tests: `analysis/tests/test_steelman_{chain,generator,lexical,decide,horeka}.py`.
Existing pipeline and analysis code was not modified.

Commits, in order: `5458494` PREREG · `8f3ef31` loader + chain · `daad5ab` generator · `c3c5004` integer fix ·
`e46850b` lexical, pairs, fixed effects as a parallel part · `e45260d` verdicts + report · `e8b0075` follow-up ·
`bb1dd29` paper text, lexical draws capped · `7d6e1f5` results + handoff · `8977cd2` HoreKa inputs + addendum B1 ·
this trace commit.

## 5. Runs on the Mac

Environment: Python 3.13.5, numpy 2.4.2, scipy 1.17.0, pandas 3.0.1 (miniconda `python3`; the worktree has no
`.venv`), macOS x86_64, Intel i7-1068NG7 (8 threads), 32 GB. BLAS threads pinned to 1, forked workers.
Inputs (read-only): `~/Hamburg/geodml-inputs/{funnel-assembled-exploration-v1, funnel-extract-exploration-v1,
funnel-replay-v1, funnel-features-v1, funnel-review-v1}` and the archived prompt snapshot
`ARR_ACL_CycleOct2026/hf-min/snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz`.
Settings: 200 keyword draws (seed 20261007), 200 within-keyword shuffles (seed 20261008), generator refits on the
first 100 draws with 200 Monte-Carlo draws per answer; output `~/Hamburg/geodml-inputs/steelman-v1/`.

| Part | Command | Code (recorded commit) | Start → end | Wall | Exit | Notes |
|---|---|---|---|---|---|---|
| chain | `python3 -m analysis.steelman chain` | `8f3ef31` | 22:27:37 → 22:28:09 | 32 s | 0 | cross-check with `analyse_stratum`: difference 1e-16 |
| generator, attempt 1 | `… generator --workers 7` | `daad5ab` | ~22:34 → ~22:36 | — | 1 | `IndexError`: float array used as index in the fixed-effects credit outcome; no fit had run. Fixed in `c3c5004` |
| generator, attempt 2 | same | `c3c5004` | ~22:36 → ~22:38 | — | stopped | stopped by me: the fixed-effects bootstrap ran serially (~5 s per fit × 200 × 4) before any refit; moved to its own parallel part in `e46850b`. No checkpoint existed |
| generator | `… generator --workers 7` | `e46850b` | 22:38:57 → 23:18:16 | 2,360 s | 0 | 2 × (1 full fit + 100 refits); 0 failed replicates |
| fe | `… fe --workers 7` | recorded `e45260d`; fe code identical to `e46850b` | 23:18:16 → 23:34:49 | 993 s | 0 | the runner reads HEAD when each part starts; `e45260d` only added `decide.py`/`report.py` |
| lexical, attempt 1 | `… lexical --workers 7` | `e46850b` code | 23:34:49 → ~23:50 | — | stopped | stopped by me after 50 of ~440 checkpointed units (projected ~5 h); refit draws capped at 30 in `bb1dd29`, units reused |
| pairs | `… pairs --workers 1` | recorded `e45260d` | 23:47:46 → 23:48:31 | 45 s | 0 | run directly alongside lexical; the sequential runner's later `pairs` call refused to overwrite (`logs/steelman-pairs.log`) |
| followup | `… followup` | `e8b0075` | 23:49:25 → 23:50:18 | 53 s | 0 | exploratory, added after the first run |
| lexical | `… lexical --workers 7` | `bb1dd29` | 23:52:10 → 00:14:58 | 1,368 s | 0 | 30 draws for the shortlisting refits (reported only); BM25 selector with all 200 draws |
| report | `… report` | `7d6e1f5` | 00:17:09 | seconds | 0 | writes RESULTS.md here and in the output folder |

Bugs found while running and fixed in code (each covered by a test or by the end-to-end test): float index in the
credit outcome; contrast key name in the report; NaN-safe ratios and formatters; empty subsets in the follow-up.

## 6. Results (exploratory; Qwen, 237 keywords, natural condition)

Full numbers with intervals: `outputs/mac-exploration-v1/RESULTS.md`.

* **C1 supported** in both methods under the pre-registered rule. Reactive: prompt-text replay 17%, query rewriting
  32%, reranker 35%, shown order 5%, generator's own choices 10% [5, 17] of the cited-intent slope. Parallel: 20%,
  18%, 47%, 7%, 8% [5, 12]. Held in each engine, in the controls variant and leave-one-keyword-out.
* **C2 supported (Qwen).** Δ_gen −0.0001 [−0.0022, +0.0017] Reactive, −0.0017 [−0.0028, −0.0003] Parallel; 90%
  intervals inside ±0.015. Same with zeroed coefficients, without slot and with the reranker score as control.
* Two-way fixed effects (the same snippet under different prompts): intent terms' intervals include 0 for Reactive
  keep and for rank credit in both methods; topic strongly positive.
* Score-matched adjacent shown links: the earlier slot wins the order 88% (Reactive) and 63% (Parallel).
* Lexical: BM25 against the reranker's own text reaches 92% [72, 113] of the Reactive reranker's intent increment
  and 119% [101, 142] of the Parallel one; overlap controls leave the reranker's intent coefficients unchanged.
* Caveats: with prompt controls the query-rewriting share falls (Reactive 6% [−3, 18]); the exploratory follow-up
  traces this to whether the prompt names its keyword (falls ~0.74 across x); Reactive keeps 15% [7, 26] with that
  control alone. The cited count falls with x under Reactive (−0.73 links). Parallel model check: fitted slope
  0.0033 below observed.

## 7. Deviations from the approved plan

* Plan step "default variant reproduces `funnel_importance.decisions` design rows" was not written as a test; the
  generator builds the same features and slot dummies and its main-variant coefficients reproduce the existing
  `funnel-generator-decisions-v1` values (e.g. Reactive keep alignment +0.320, interaction −0.550).
* Lexical shortlisting refits used 30 draws instead of 100 (runtime); reported only.
* A13/A15 were folded into the chain part; A11 was replaced by the cited-count slopes plus a droppers-only chain,
  as the plan stated.
* The keyword-mention follow-up was added after the first run and is labelled exploratory everywhere.

## 8. Verification

* Tests: `python3 -m pytest -q analysis/tests/test_steelman_*.py` → 20 passed (planted-math checks of the loader,
  chain identity with and without controls, subset sampler against exact inclusion probabilities, Monte Carlo
  against exact enumeration, two-way FE against dummy OLS, verdict boundaries, BM25 ties, pair extraction, and an
  end-to-end run of every part on HoreKa-shaped inputs with two models). With the existing funnel, intent-stages
  and geo-drivers suites: 64 passed before the HoreKa changes.
* Reproduction: after the HoreKa changes (`8977cd2`) the chain part was rerun on the Mac data into a scratch
  folder; every number equals the published `chain.json` (0 differences above 1e-12).
* Checksums of every output and cache: `outputs/mac-exploration-v1/{files,caches}.sha256`.

## 9. HoreKa preparation (2026-10-08)

Code changes: input-folder options and `--split`; both prompt formats; agent queries from the intent trace extract
joined on fingerprint; keep-model rule; fixed-effects part checkpointed per stratum with a deadline; report for any
models and split; runner `analysis/docs/horeka-steelman.sbatch` (cpuonly, 32 CPUs, 128 GB, 1 h, no requeue;
builds missing prerequisites with the existing runners' commands; exit 4 = deadline checkpoint; ends with a
tarball in `$W/reviews/`). Not executed anywhere yet; the commands and estimates are in the session handoff
`analysis/docs/handoff/2026-10-07_steelman-shortlist-claim_handoff.md`.

## 10. Open

Llama keep/order and the held-out keywords need the HoreKa run. The ablated condition (a forced change of the
shortlist for the same prompt) is unanalysed. The decisive fixed-shortlist test needs new generations and is out
of scope.
