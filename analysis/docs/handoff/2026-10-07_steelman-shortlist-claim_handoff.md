# Steelman of "intent works at the shortlist" from existing data only

Valerian asked (2026-10-07) for the strongest honest case for C1 (intent reaches the cited sources by changing the
shortlist) and C2 (the generator's choice among shown links is intent-insensitive), using only existing tables:
no generator reruns, no reranker rescoring, no embeddings. Plan approved; pre-registration committed before any run.

## State

Branch `codex/acl-figures-20261006` (not pushed; Valerian pushes). Commits this session, in order:
`5458494` PREREG.md · `8f3ef31` loader + chain · `daad5ab` generator · `c3c5004` fix · `e46850b` lexical, pairs, fe part ·
`e45260d` verdicts + report · follow-up commit · `bb1dd29` paper text, lexical draws capped · final commit with RESULTS.md.
Code: `analysis/steelman/` (`tables.py`, `chain.py`, `generator.py`, `lexical.py`, `pairs.py`, `decide.py`, `report.py`,
`__main__.py`), tests `analysis/tests/test_steelman_{chain,generator,lexical,decide}.py`. Run on the Mac with
`python3` (miniconda; the worktree has no `.venv`): `python3 -m analysis.steelman <chain|generator|fe|lexical|pairs|followup|report>`.
Outputs: `~/Hamburg/geodml-inputs/steelman-v1/` (one JSON per part, `stage_shares.csv`, `RESULTS.md`, `manifest.json`);
the same RESULTS.md is committed at `analysis/steelman/RESULTS.md`. Nothing ran on a cluster.

## Results [F], exploratory (Qwen, 237 exploration keywords, natural)

- C1 supported in both methods under the pre-registered rule. Reactive shares of the cited-intent slope: prompt-text
  replay 17%, query rewriting 32%, reranker 35%, shown order 5%, generator's own choices 10% [5%, 17%].
  Parallel: 20%, 18%, 47%, 7%, 8% [5%, 12%].
- C2 supported (TOST, ±0.015): Δ_gen (intent-blind generator refit, Monte Carlo with common random numbers)
  −0.0001 [−0.0022, +0.0017] Reactive, −0.0017 [−0.0028, −0.0003] Parallel. Same with zeroed coefficients, without
  slot, and with the reranker score as control. Coefficients are not zero (Reactive keep alignment +0.32, interaction
  −0.55 per SD): they offset on the slope scale.
- Two-way FE (same snippet under different prompts): intent terms' intervals include 0 for Reactive keep and for
  rank credit in both methods; topic strongly positive.
- Shown slot correlates with topic (−0.22/−0.28) and the reranker score, barely with intent (−0.02/−0.03).
  Score-matched adjacent pairs: the earlier slot wins the order 88% (Reactive) / 63% (Parallel) at equal scores.
- Caveats for the paper: (1) with prompt controls the query-rewriting share drops (Reactive 6% [−3, 18], Parallel
  −3%); exploratory follow-up: whether the prompt names its keyword (falls 0.73–0.75 across x) absorbs it; Reactive
  keeps 15% [7, 26]. Do not write "chiefly through the queries". (2) The cited count falls with x under Reactive
  (−0.73 links), a channel the counterfactual conditions away. (3) Parallel model check: fitted E[K] slope 0.0033
  below observed.
- Lexical: a BM25 selector against the reranker's own text reaches 92% [72, 113] (Reactive, agent query) and 119%
  [101, 142] (Parallel, prompt) of the reranker's intent increment; overlap controls leave the reranker's intent
  coefficients unchanged (30 draws, reported only).
- Llama: R0 and K from published rows only (R0/K 15–22%); its traces (51 GB) are still not on the Mac.

## Next

Push the branch. Llama keep/order needs the HoreKa CPU trace extract (job 5185751 rerun with 40 workers, prepared
earlier); then rerun `generator`/`fe` with Llama strata. Confirmation keywords: apply PREREG.md unchanged.
Out of scope, decisive: the fixed-set test (same shortlist, prompts at different x; ≈ 11k generations).
