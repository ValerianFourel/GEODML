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
- Lexical (addendum B2; V2 and the paper do not use BM25): the frozen search's own word-overlap rule applied to the
  reranker's candidates reproduces 52% [33, 72] of the Reactive reranker's intent increment and −27% [−57, −2] of
  the Parallel one; the cross-encoder adds intent beyond word overlap. The pre-registered BM25 version (92% / 119%)
  is kept as a record only.
- Llama: R0 and K from published rows only (R0/K 15–22%); its traces (51 GB) are still not on the Mac.

## Next

Push the branch. Llama keep/order needs the HoreKa CPU trace extract (job 5185751 rerun with 40 workers, prepared
earlier); then rerun `generator`/`fe` with Llama strata. Confirmation keywords: apply PREREG.md unchanged.
Out of scope, decisive: the fixed-set test (same shortlist, prompts at different x; ≈ 11k generations).

## Addendum (2026-10-08): HoreKa-ready code, full trace

Commits `8977cd2` (HoreKa inputs, keep-model rule, checkpointed fixed effects, `analysis/docs/horeka-steelman.sbatch`,
PREREG addendum B1), `eb2f2a2` (frozen-search selector, addendum B2), then the trace commit. Full record of the work:
`analysis/steelman/trace/` (TRACE.md, approved plan, every output with checksums, checkpoint caches, logs).
Tests: 21 steelman tests (incl. an end-to-end run of every part on HoreKa-shaped inputs with two models); 66 with
the funnel, intent-stages and geo-drivers suites. The Mac chain reproduces exactly with the new code.

HoreKa run (not submitted; Valerian runs it). One job at a time, each 1 node, 32 CPUs, 128 GB, 01:00:00, cpuonly.
Estimates [H, scaled from the Mac run by answer count ×7.4 and two models]: prerequisites if missing: funnel
extract + replay + assemble 25–70 min, intent trace-extract 15–35 min; steelman: chain/followup/pairs 10–20 min,
generator ~40 CPU-h (16 workers: ~2.5–3 h wall), fixed effects ~14 CPU-h (32 workers: ~30 min), lexical 10–20 min.
Expected 4–7 allocations (128–224 allocated CPU-hours); declared budget 8 allocations (256 CPU-hours). Cheaper
fallback: `GEN_WORKERS=24` if memory allows (check MaxRSS of the first generator job), or skip `fe`.
Exit 4 = deadline checkpoint: resubmit the same command after checking the queue; 0 = done; the job prints
`STEELMAN DONE` and the tarball path. Bring back `$W/reviews/steelman-confirmation-<sha7>-*.tar.gz`.
