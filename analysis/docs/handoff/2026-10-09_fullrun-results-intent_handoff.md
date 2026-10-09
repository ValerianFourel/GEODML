# Full run finished: held-out results for the intent study, and what the paper must change

Written 2026-10-09 for the session that updates the ACL paper (`ARR_ACL_CycleOct2026/paper_v2`). Start here, then read
`ARR_ACL_CycleOct2026/paper_v2/HANDOFF.md` (how the paper is built) and `analysis/steelman/PREREG.md` (the rules).

Evidence labels: **F** = fact read from a result file or log; **P** = plan; **A** = assumption; **H** = estimate.
Every number in §3–§5 is **F**, from `steelman-confirmation/` unless a line says otherwise.

## 1. Where the data is [F]

| What | Where |
|---|---|
| Archive on the Hub | `ValerianFourel/geodml-experiment-v2-paper-private`, `derived/fullrun-run2/fullrun-run2-20261009T050224Z.tar.gz`, revision `96d0e9dd7958a134a7477dfea4a9cc0b2fb9e8e9`, sha256 `87c9c224b1af0da08f1c1e6e2df007290f967a227f0cdacfee110e3241aaedb8` (checked equal on HoreKa, the Hub and the Mac) |
| Unpacked on the Mac | `~/Hamburg/geodml-inputs/fullrun-run2/` |
| Steelman, held-out (confirmatory) | `steelman-confirmation/RESULTS.md`, `chain.json`, `generator.json`, `fe.json`, `pairs.json`, `supply.json`, `ablation.json`, `queries.json`, `followup.json`, `lexical.json`, `census.json`, `stage_shares.csv` |
| Steelman, exploration (exploratory) | `steelman-exploration/` (same files) |
| Funnel stage models (paper Tables 3, A3) | `paper-results/funnel-{confirmation,exploration}/results.json` |
| Generator keep/order decisions | `paper-results/decisions-{confirmation,exploration}/decisions.json` |
| Intent in text (queries, answers on the axis) | `paper-results/intent-stages/results.json`, `paper-results/answers-analysis/{report.md,answer_coordinates.jsonl.gz}` |
| One-page summary of the bundle | `paper-results/report.md` |
| Checker output | `verdicts.json` (from `python3 -m analysis.steelman.check_fullrun ~/Hamburg/geodml-inputs/fullrun-run2 --status .../final-status.json --out .../verdicts.json`) |
| HoreKa copy | `$W/reviews/page-readiness-20261004/fullrun-v1/run2` and `$W/reviews/fullrun-run2-20261009T050224Z.tar.gz` |

Scope: both generators (Qwen3.8-27B `qwen38`, Llama-4-Scout `llama4`), 624,132 answers (312,042 Qwen, 312,090 Llama),
1,011 keywords: 278 exploration, 733 held-out confirmation. Natural condition unless stated.

## 2. How the run went [F]

- Ledger: 300/300 tasks done, 0 failed. Done records by commit: `fdba5e2` 75 (extraction, 2026-10-08), `cf5c452` 46
  (prep, steelman parts, embeddings, decisions), `de5d8fb` 177 (every funnel task and the reports).
- Checker: no missing parts; one commit per steelman split (`cf5c452`); PREREG sha256 `b0e41e43…`; settings
  200/200/100/200, seeds 20261007 / 20261008; generator and fe on CUDA, the rest on CPU.
- Jobs on the `casualnet` reservation (account hk-project-p0026831, 16 A100 nodes, until 2026-10-31): 5187589,
  5187091–5187122 (2026-10-08, ended at the 60-min idle limit or, 5187094, TIMEOUT); 5188024–5188044 (2026-10-09 morning);
  5188186–5188201 and 5188206 (final). The jobs that "stopped after one hour" had nothing left to start; not crashes.
- Three code bugs found and fixed on branch `codex/acl-figures-20261006` (all with regression tests that fail on the old code):
  1. `cf5c452`: `funnel_study.py assemble` re-read whole npz arrays per answer (~570 h at full size; job 5187094 timed
     out on it). `answers-export` called `git` inside the container (now `GEODML_GIT_COMMIT`).
  2. `de5d8fb`: `intent_stages.selection_rows` assumed shortlist picks 0..k-1 per event. The **complete-case**
     specification drops rows, which left gaps in 48.8% of shortlist events (Mac exploration tables) and pushed later
     picks into the next event's choice sets. Picks are now renumbered per event (main/visible specifications unchanged:
     exact identity checked on real data). New `fullrun redo --stage funnel` reran all 176 funnel fits on `de5d8fb`.
- Consequence for the paper: **the complete-case shortlist odds ratios quoted in Appendix A3 (5.29 / 4.80) came from the
  buggy fits** and must be replaced from `paper-results/funnel-*/results.json`. Complete-case retrieval numbers (domain
  size 1.38 / 1.67) do not use this code.

## 3. Pre-registered verdicts, held-out keywords [F]

| Claim | qwen38 · Reactive | qwen38 · Parallel | llama4 · Reactive | llama4 · Parallel |
|---|---|---|---|---|
| C1 admission (intent enters before the generator) | supported | supported | supported | **narrowed** (generator share bound fails with prompt controls and in one engine) |
| C2 selection (per model, both methods) | supported | supported | **narrowed** | **narrowed** |
| Supply | narrowed (all strata) | | | |

Exploration split gives the same verdicts. Per PREREG the confirmatory verdict is the reported one.

## 4. Where intent shapes the result: shares of β_K, held-out, common sample [F]

β_K = within-keyword slope of the cited sources' rank-weighted page intent on x (x = 0 → 1).

| Step | Qwen · Reactive | Qwen · Parallel | Llama · Reactive | Llama · Parallel |
|---|---|---|---|---|
| β_K | +0.072 [+0.067, +0.076] | +0.058 [+0.055, +0.062] | +0.076 [+0.071, +0.081] | +0.094 [+0.088, +0.099] |
| R0 prompt words (mechanical replay) | 23% | 29% | 23% | 15% |
| R − R0 query rewriting | 27% (11% with prompt controls) | 13% (2%) | 22% (4%) | 22% (12%) |
| P − C reranker shortlist | 38% | 47% | 33% | 22% |
| I − P shown order | 3% | 6% | 10% | 12% |
| K − I generator's own choices | 8% | 6% | 12% | 30% |
| **P: in place before the generator** | **89% [86, 92]** | **88% [86, 90]** | **78% [75, 81]** | **58% [56, 61]** |

Mechanisms (for the text):
- Reranker = BAAI/bge-reranker-v2-m3 cross-encoder. Parallel: scores pooled rows against the **user prompt**, top 7.
  Reactive: scores each search's rows against **the agent's query**, top 3 per search, duplicate URLs merged.
- Under Parallel the frozen-search word-overlap rule explains none of the reranker's intent increment (−15% Qwen,
  −47% Llama of the step): the reranker matches meaning. Under Reactive word overlap reproduces 43–45%: intent reaches
  the reranker through the agent's queries.
- **Keyword finding reverses the old sentence.** Prompts stop naming their keyword along x (share −0.74 to −0.79,
  within keyword); with prompt controls the query-rewriting share falls to 2–12%. The drop is the prompt's, not the
  agent's ("action-ready prompts stop naming the topic keyword and the agent's queries follow").

## 5. Generator, supply, manipulated contrast [F]

- C2 Δ_gen (drop-intent refit), 90% interval vs SESOI ±0.015: Qwen Reactive −0.0003 [−0.0014, +0.0008]; Qwen Parallel
  −0.0026 [−0.0031, −0.0022]; Llama Reactive +0.0033 [+0.0017, +0.0049]; Llama Parallel +0.0141 [+0.0113, +0.0159]
  (15% of β_K). Llama keep decision modelled (Parallel: 27,965 informative answers; 91% of answers drop a shown link).
- Topic decides rank everywhere: keep/order per-SD coefficients, topic similarity +0.49 to +2.07 vs intent alignment
  +0.01 to +0.20; within-snippet two-way fixed effects agree.
- Cited count on x: Qwen Reactive −0.79 [−0.85, −0.73] links; Llama Parallel −1.96 [−2.06, −1.86] (of 7 shown);
  Llama Reactive +0.06 [+0.01, +0.11]; Qwen Parallel −0.13.
- Supply (narrowed): ceiling gap at x ≥ .9 is 0.37–0.43 (all > 0); utilisation of the own-row oracle (β_K / oracle_U)
  ≈ 0.20–0.28, below the 0.5 rule. Rows with u ≥ .8: 0.76%; keywords with any such row: 7.6%. Write: scarcity caps
  how close citations get at the action end, not the slope.
- Ablated condition (the only manipulated contrast): 73–88% of a forced change in shortlist intent reaches the cited
  sources (pass-through dK/dP: Qwen Reactive 0.74, Qwen Parallel 0.88, Llama Reactive 0.73, Llama Parallel 0.72).
- Stability: leave-one-keyword-out moves every share by under one point.

## 6. What the paper must change (P)

Headline numbers move from Qwen exploration to **both models, held-out keywords** (PREREG B3). Edit numbers only in
`paper_v2/numbers.tex`, keep `% src:` lines pointing at the files in §1.
1. Abstract and §1: "85% to 87% of the final shift" → Qwen 88–89%, Llama 78% (Reactive) and 58% (Parallel).
   Stage shares "18–20 / 18–34 / 35–47 / 13–15%" → §4 table.
2. Table 1: fill the 7 red `[pending: Llama]` cells; replace Qwen cells by held-out values. Table 2 (stage slopes) from
   `chain.json`. Table 3 (Pratt shares) from `paper-results/funnel-confirmation/results.json` and `decisions-confirmation`.
3. Generator claim (abstract, §5.3, §6, §7): scope "selection among shown sources does not respond to intent" to Qwen;
   add Llama narrowed (Δ_gen +0.014 under Parallel, 15% of β_K; 30% generator share).
4. Keyword sentence (§1 lines ~108–111, §5.2): "the agent drops the keyword" → the prompts stop naming it (§4).
5. Supply sentence: narrowed wording (§5).
6. Appendix A3: complete-case shortlist odds ratios from the fixed fits; recheck every A3 table against
   `funnel-confirmation` / `funnel-exploration`.
7. Limitations: held-out confirmation and Llama traces are no longer pending; keep "associations", add the reranker
   and corpus scope, and that page effects are observational (we varied prompts, not pages).
8. Main claim stays: "intent gets a page shortlisted; topic decides its rank" (topic dominates in all four cells);
   the "generator ignores intent" half is Qwen-only.

## 7. Known flaws and open items

- `analysis/steelman/report.py` template text is wrong for the held-out split (F): "No held-out result exists …
  everything here is exploratory" (§O6) and "Supporting numbers (Qwen, exploration keywords)" under the llama4 heading.
  Numbers are right; do not quote these lines. Fix the template (P).
- Earlier handoff `2026-10-08_fullrun-results-analysis_handoff.md` §1 calls qwen38 "Qwen3-8B"; the generator is
  Qwen3.8-27B (Qwen3-8B is an LLM2Vec encoder base).
- Gemma SI-v4 (development material in the paper), 2026-10-09 04:38 UTC audit: Llama 200/200 shards finished and on the
  Hub; Qwen main run 135 finished (134 published), 11 `budget_exhausted` (re-run needs Valerian's budget approval),
  47 not started; Qwen top-up 0/16. hkn0807 also fails the startup user lookup (exit 1 in ~33 s) and was not yet in
  `$W/control/gemma-exclude-nodes`. "finished_with_failures" = some answer maps failed after retry (blocked tasks:
  Llama 7.6%, Qwen so far 2.5%); no saved judgment was lost.
- Generation: Llama 312,090 done + 6 terminal failures; Qwen 312,042 done, 54 checkpointed (stuck claims; sync `apply`
  then one Qwen bout). Hub Qwen count matches HoreKa exactly; Llama Hub count 310,374 is a lower bound (registered hours).
- Optional exploratory additions (not pre-registered, label as such): (A) refit keep/order on the shuffled condition to
  separate shown slot from topic (existing data, ~1 CPU node-hour); (B) axis battery bootstrap clustered by source.
- A small human rating of the axis (≈200 prompts, agreement reported) is the gap reviewers will see first.
