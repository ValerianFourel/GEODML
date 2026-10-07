# Funnel study: exploratory results on the Mac, HoreKa confirmation prepared

Valerian's goal (2026-10-07): a full exploratory analysis of how the prompt's axis position x relates to
every step of the V2 search pipeline (SERP/SEO data, DataForSEO, page features, Llama and Qwen
generations, Gemma), a by-keyword table, a published report, and HoreKa commands for the confirmatory
run (addendum A1: Mac on exploration keywords, P1–P4 on the held-out confirmation keywords).

## State

Branch `codex/acl-figures-20261006`, code pin for HoreKa `dbda07a` (full SHA below).
**Not pushed**: the push from the Mac was refused by the session's permission check. Valerian pushes:
`git -C .worktrees/acl-figures push origin codex/acl-figures-20261006`. Nothing ran on a cluster.

Commits this session (after 4146073):
- `638e229` fork-safe analysis on macOS (Accelerate is not fork-safe: the first exploration run died with
  SIGSEGV in a forked worker, crash report "crashed on child side of fork pre-exec"); drop-block fits in
  the pool; `funnel_local.py extract-hf --model`, `merge`.
- `31d0ec1` keyword shrinkage centres on the random-effects mean (the fixed-effect mean, 0.042, is pulled
  down by precise small-slope keywords; RE mean 0.064). `funnel-keywords-v1` kept; v2 is current.
- `362a9e9` resumable fits: every fit is a checkpointed unit (`<task>.parts.jsonl`), a deadline stops new
  fits (analyze exit 4), column-major designs (no per-worker matrix copies); runner `MODE=prep|analyze|all`,
  deadline from Slurm's end time minus `MARGIN_MIN`.
- `ceaa255` figures fig5–fig8 (`acl_figures.py render --funnel-results --explore`).
- `f38f221` keyword table reads the HoreKa `gemma-extract` folder (join on the generation fingerprint inside
  `record_id`, join rate reported; key equality is an assumption until real grades arrive).
- `dee07fd` `funnel_contrasts.py`: model, method and engine slope contrasts with shared keyword draws.
- `50151dd` report generator in the goal's section order, embedded figures, CSV tables; fixed coverage
  columns (values were under the wrong headers) and an unbound variable that broke every earlier run.
- `dbda07a` stage contrasts pair bootstrap replicates by design column order (protocol addendum A2): the
  cache writes sorted JSON keys, so P2/P3 paired wrong columns (estimate −0.172 outside its own interval).
  Only P2/P3 were affected; the exploratory report was rebuilt from the cached fits of 362a9e9.
Tests: 94 pass (funnel suites, intent stages, GEO drivers, figures, page readiness, report).

The Llama traces (56 bundles, 51 GB on HF) could not be downloaded: the connection ran at about 0.05–0.1 MB/s.
The Mac stage models therefore cover Qwen only; Llama enters through the published generation rows.

## Exploratory results (addendum A1; nothing here is confirmatory)

Report: https://claude.ai/artifact/FfAx8BttD7f1XrWRxC2LaB (private), built from
`~/Hamburg/geodml-inputs/funnel-local-report-v1/` (page, tables/*.csv, results bundle, figures, the
findings script `narrative.py`). Inputs, all under `~/Hamburg/geodml-inputs/`: `funnel-review-v1`
(621,003 answers, 200 draws), `funnel-explore-v1` (Qwen traces, 237 exploration keywords, 19,944 natural
answers, 200 draws), `funnel-report-exploration-v2` (stage models, 100 draws / 100 shuffles, analysis
commit 362a9e9, report commit dbda07a; v1 has the wrong P2/P3 intervals and is kept as history),
`funnel-contrasts-v1` (200 shared draws), `funnel-keywords-v2` (1,011 keywords).
- [F] Cited-source intent slope on x: Llama · Parallel +0.091 [0.087, 0.098], Llama · Reactive +0.074
  [0.070, 0.079], Qwen · Parallel +0.057 [0.054, 0.061], Qwen · Reactive +0.068 [0.064, 0.072].
- [F] Qwen stage slopes (Parallel / Reactive): literal prompt search R0 +0.012 / +0.012, retrieved R
  +0.023 / +0.035, shortlist P +0.051 / +0.059, ranked K +0.060 / +0.068. [A] R/K 39% / 51%.
- [F] Stage models: topic similarity dominates (OR per SD 3.0–6.5; 35–85% of the fit); intent block
  0.2–2.1%; page intent × (x − ½) at the shortlist 0.77 / 0.69, permutation p = 0.010 (all 100 shuffles);
  alignment alone p = 0.16–0.58 at shortlist and ranking. Domain organic keywords: retrieval 1.46 / 1.54
  (above the C2 bar), ranking 0.95 / 0.97. P2 and P3 intervals include 0.
- [F] Decomposition within the keyword's own rows: intent alignment log RR +0.135 / +0.214, 97% / 83% of it
  at the shortlist [A]; topic similarity +1.120 / +1.576.
- [F] Contrasts of the cited-intent slope: Llama − Qwen +0.034 [0.031, 0.038] (Parallel), +0.006
  [0.003, 0.009] (Reactive); SearXNG − DuckDuckGo +0.024 (Llama), +0.021 (Qwen).
- [F] By keyword: random-effects mean +0.064 [0.061, 0.066], τ = 0.040 [A], Q 6,462 on 1,010 df above all
  200 shuffles (max 1,240); 76% of shrunken intervals above 0, 0.4% below. Gemma column empty.
- [F] Cross-model top-1 agreement 0.578 (lowest x decile) to 0.451 (highest), 92,073 cells.

## HoreKa confirmation (prepared, not submitted)

Commands: the session reply of 2026-10-07 (blocks 0–6), repeated in short here. Pin
`dbda07a509a918e3fa80986b3d50d7065eeb2353`, checkout `$W/checkouts/funnel-<pin>`. Feature table from HF
`ValerianFourel/geodml-experiment-v2-paper-private@5f89299a81ae015d43d7e22afe78f76764ac9630`,
`derived/funnel-features-v1` (manifest sha256 `ddf78ad8…`). Order: status → checkout + features (login
node) → `MODE=prep` → intent stages with `GEMMA_RUN` (≥ 10 min after the prep start) → `MODE=analyze`
shards 3/4, 1/4, 4/4, 2/4, one at a time, each ≥ 10 min after the previous start and with fewer than five
jobs queued. Exit 4 = deadline checkpoint (fits saved): resubmit the same shard.

Resources and estimates (every job: 1 node, 32 CPUs, 128 GB, 01:00:00, `--no-requeue`):
- prep: extract + replay + assemble, 25–70 min [H: scaled from the Mac extract rate, ≥ 7 MB/s per
  process over ~110 GB of traces with 32 workers]; 1–2 allocations (stage-level resume; an interrupted
  extract restarts).
- analysis: about 1,700 scored-candidate fits at ~250 s each plus the smaller stages ≈ 120–190 CPU-hours
  [A: Mac measurement ~80 s of core time per fit at 3.07M rows, × 3.1 rows on the confirmation split];
  shard 3/4 needs 2–3 allocations, the others 1–2.
- intent stages / Gemma extract: 1–2 allocations, length unknown [H].
- Expected 8–12 allocations (256–384 allocated CPU-hours) within the declared budget of 14 (448).
  Cheaper fallbacks: secondary specifications at 50 draws, drop `complete`, main specification only.

Results to bring back: `$R/funnel-report-v1/results.json` (+ `final-report.html`),
`$R/intent-gemma-extract-v1/{grades.jsonl.gz,manifest.json}`, `$R/intent-stages-v1/results.json`.
Then on the Mac: `funnel_keywords.py --gemma <extract> --output funnel-keywords-v3` fills the Gemma column
(check `gemma.graded_answers_joined` in the summary first).

Note: AGENTS.md still names `.worktrees/threehour-relaunch-fix` as the active checkout; this line of work
continues on `codex/acl-figures-20261006` (branched from its last commit, a7fc602).

## Addendum (same day): the generator's own decisions

Valerian set the priority (now in the root AGENTS.md, not under git): study the LLM's behaviour first,
i.e. which shown sources an answer keeps (0/1) and how it orders the kept ones; the reranker and the
frozen search come second. Engine position reaches V2 only through the frozen search's tie-break
(`funnel_rows.LexicalIndex`: exact keyword, overlap, stored position, hash).

`funnel_importance.py` (commits c0a36b6, 228a3c2): keep = conditional logit within each answer given how
many shown snippets it kept (shown-slot dummies); order = Plackett–Luce over the kept snippets with
shown-slot effects; McFadden pseudo-R² and Pratt shares; 100 draws / 100 shuffles. Output
`~/Hamburg/geodml-inputs/funnel-generator-decisions-v1/` (Qwen, exploration keywords, natural):
- [F] Qwen · Parallel keeps 99.1% of shown links (98.3% of answers keep all 7): no keep decision to model
  (148 informative answers; the fit separates). Qwen · Reactive keeps 77.4% (4,057 informative answers).
- [F] Keep, Reactive: pseudo-R² 0.46; topic block 80% (topic similarity OR 6.89 [6.26, 8.02] per SD),
  shown slot 14%, URL 2.7%, off-page 1.5%, page body 1.0%, snippet 0.3%, intent 0.0% (alignment and the
  interaction offset each other).
- [F] Order: pseudo-R² 0.19 Parallel, 0.35 Reactive; shown slot 40% / 88%, topic 53% / 9%, every other block
  ≤ 3.4%, intent ≈ 0%.
Llama's keep decision (it keeps far fewer of the 7) needs its traces: HoreKa.

## Addendum (same day): one interactive runner for everything the paper still needs

`analysis/docs/horeka-paper-interactive.sh SLURM_JOB_ID` (commit below; branch also carries the
other session's fig9 commit 26a7c5a and the cherry-picked answer-text export 9f66a4f). Run it from a
second login shell against a four-A100 `salloc`; rerun the same command in each later allocation until
it prints `BUNDLE`. Stages: funnel confirmation (extract → replay → assemble → analyze main, then
secondary → report), generator keep/order decisions (both models, 733 confirmation keywords),
intent-stages (trace extract, replay, Gemma extract when `$W/reviews/gemma-si-v4-llama-reuse-5h-20261004`
exists), query and answer embeddings on the GPUs, answers on the axis, intent-stages analysis, and
`paper_results.py` → `$R/paper-results-<commit>/report.md` + `$W/reviews/paper-results-<commit>-<time>.tar.gz`.
Exit 4 = deadline checkpoint (fits saved). Estimate [H]: 6–10 one-hour allocations in total; the
confirmation analysis is the long part (120–190 CPU-hours on 32 cores, less with the GPU node's cores).
Bring back the tarball; `paper_results.py` also runs on the Mac to merge in the exploratory folders.
