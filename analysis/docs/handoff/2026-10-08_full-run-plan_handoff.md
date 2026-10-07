# START HERE: full run to finish the paper's math (plan approved 2026-10-08)

**Read `analysis/steelman/IMPORTANT.html` first.** It lists every abstract claim, its current evidence, what is
overstated, and what the full run must compute. This is the next job for whoever picks the project up.

Decisions by Valerian (2026-10-08): run on HoreKa with approved longer jobs (6-hour CPU allocations, cap 8, at most
4 concurrent, 10-minute start gaps; up to 3 GPU allocations of 2 hours); include the LLM2Vec encoder embeddings of
queries and answers; cover both models, all 1,011 keywords and all three conditions, reporting exploration and
held-out keywords separately. No new text generation.

State at writing: branch `codex/acl-figures-20261006`, 13 commits ahead of origin (Valerian pushes). Nothing of the
full run exists yet; this handoff is updated as each step lands.

## The approved plan (verbatim copy)

### Full run: every trace, both models, all keywords — finishing the math behind the abstract

Worktree `.worktrees/acl-figures`, branch `codex/acl-figures-20261006` (pushed state unknown; last local commit `37f783d`).
Decisions taken by Valerian (2026-10-08): **HoreKa with approved longer jobs**; **include the GPU encoder embeddings**
of queries and answers; **everything** (both models, 1,011 keywords, three conditions) with exploration and
held-out confirmation keywords **reported separately**. No new text generation of any kind.

## 1. Context

The paper's abstract and contribution list are being finalised around: a validated continuous intent measure;
a stage-wise trace of how intent moves the results through a frozen, fully logged LLM search pipeline; findings
on queries, the generator, and scarcity of action-ready pages. The steelman so far covers Qwen on 237
exploration keywords only. To finish, every claim needs its number from the full data (both models, held-out
keywords confirmatory), and three claims need analyses that do not exist yet (supply bound, ablated shortlist,
full census). The Mac cannot hold the 114 GB of traces; one-hour HoreKa jobs make the run slow and fragile.
Outcome: one reproducible, sharded, resumable run on HoreKa that produces every number the paper cites,
an `IMPORTANT.html` that says exactly what remains and why, and a handoff pointing to it.

## 2. Where the abstract stands today (feeds IMPORTANT.html)

| Claim in the draft abstract | Evidence now | Verdict | Corrected wording / what closes it |
|---|---|---|---|
| Continuous, validated intent measure; topic held fixed; prompts and documents on one scale | Battery 11/11; two encoders agree (axis-1 ρ .866); fresh re-embedding ρ .99997; 98.3% of x variance within keyword; pages placed by `PromptScale.percentile` | Supported, with care | "topic held fixed" → "compared within the same topic (keyword), with an intent-free topic control". Labels are LLM-judge consensus; intervals row-bootstrap; page positions out-of-domain. Close: source-clustered battery bootstrap (existing data) |
| "known pipeline" | — | Change | "frozen, fully logged pipeline"; R0 is an analysis-time replay |
| Intent shifts what gets cited mainly through the queries and the shortlist | Qwen: 84–85% of the cited-intent slope is in place before the generator; query rewriting 32%/18%, reranker 35%/47% | Supported (Qwen, exploration) | Never "chiefly through the queries". Close: both models, held-out keywords |
| Generator's selection does not respond to intent | Δ_gen −0.0001 (Reactive), −0.0017 (Parallel), inside ±0.015; coefficients non-zero and offsetting; cited count falls with x (Reactive −0.73) | Overstated | "adds almost nothing to the shift (Qwen)". Close: Llama keep/order (its keep decision has room) |
| Size of the shift bounded by scarcity of action-ready pages | No saved analysis. Quick look: 0.76% of rows have u ≥ .8; 7.6% of keywords have any; own-row oracle slope .23–.37 vs observed .06–.09 | Not supported as written | "few action-ready pages exist, which caps how close cited sources can come to the most action-ready prompts". Close: supply part (below) |
| Stage shares 18–20 / 18–34 / 35–47 / 13–15% | Old hand ratios | Replace | Steelman common-sample shares with intervals, then full-run numbers |
| Agent drops the keyword from queries 61% → 21% | Qwen, 237 keywords | Partly prompt-driven | Prompts themselves stop naming the keyword (−0.73 across x). Close: report it among prompts that name the keyword |

Also flagged: glued-title rows still mentioned in `manuscript_draft/sections/methods.tex:60` and `introduction.tex:6`
(memory rule says never); "Tannenbaum (2026)" and "Martinez (2026)" have no source anywhere (find or drop);
BM25 stays out of the paper.

## 3. Data facts the run is built on

- Hub `ValerianFourel/geodml-experiment-v2-paper-private`: 985 bundle descriptors (`exchange/bundles/`), content-addressed
  objects (`exchange/objects-v2/<aa>/<sha>`, legacy `exchange/objects/`). Traces ≈ 104 GB unique (Qwen 48.5, Llama 51.2,
  shared 4.3). Qwen 312,042 completed on the live index (Mac mirror 310,623, stale); Llama 312,096 incl. 1,716 only in the
  frozen-inputs bundle. Design: 26,000 prompts × 2 methods × 3 conditions × 2 engines per model.
- HoreKa already holds most cells (`$W/shared-hours/dataset`, `$W/llama-hf/dataset`; Qwen local 287,944). About 21–24k Qwen
  cells and the 1,716 Llama cells exist only on the Hub; `Exchange.download` (`agentic_hour_sync.py:426-480`) imports a
  bundle into a dataset root with hash checks.
- Gemma SI-v4 judgments for Llama: `reviews/gemma-si-v4/gemma-v4-8941…` (200 shards, ~1.6 GB); none for Qwen.
- Compute profile (from the scripts and handoffs): funnel analyze 120–190 CPU-h; steelman generator ~45 CPU-h and fe ~14
  CPU-h at full size; intent analyze ~1 h; extraction 40–105 min; GPU embeddings 1–2 h on 4 A100s. Total ≈ 250–330 CPU-h
  plus one or two GPU allocations.
- Mergeability: keyword draws and shuffles are deterministic and prefix-stable (seed 20261007); `.parts.jsonl` units are
  keyed (`full-0`, `drop-i`, `bootstrap-i`, `permutation-i`, `draw-i`), so partial runs on different nodes concatenate.
  Extraction (`funnel_study.py extract`, `intent_stages_study.py trace-extract`) is not resumable and not splittable today.

## 4. Approach

### 4.1 Container (CPU stages)
- `analysis/container/Dockerfile` (python:3.12-slim, pinned `requirements-cpu.lock`: numpy, scipy, pandas, pyarrow,
  scikit-learn, huggingface_hub, pytest) and `analysis/container/geodml-cpu.def` (Apptainer, `Bootstrap: docker-archive`
  from the Docker image, or `Bootstrap: docker` from the same base so HoreKa can build it without Docker).
- The code is **not baked in**: the container mounts the pinned clean checkout (`CODE`), so git stays authoritative and
  the image only changes when dependencies change. `GEODML_GIT_COMMIT` is passed in and `page_readiness_ordering.git_commit`
  reads it first (one-line change + test), so caches are keyed even without `.git` inside the container.
- GPU embedding keeps the **validated HoreKa venvs** (`$W/environment/llm2vec-{qwen,mistral}`, pinned model revisions,
  `horeka-page-readiness-lib.sh:embed_view`) — AGENTS.md requires unchanged validated four-A100 profiles; no container there.
- Smoke test on the Mac: `docker build` + `docker run … pytest analysis/tests/test_steelman_* test_fullrun_*`.

### 4.2 A finite task ledger (divisible as far as the math allows)
New package `analysis/fullrun/` (reuses existing estimators; no rewrite):
- `plan.py` writes `tasks.jsonl` once, deterministically: one line per unit of work with `id`, `stage`, `args`,
  `cost_estimate`, `depends_on`. Units:
  - fetch: one per missing Hub bundle (login node only);
  - extract: (model source × keyword-hash shard k/32) for funnel extract and intent trace-extract (new shard-and-merge
    wrappers around the existing `_extract_chunk` / `_trace_chunk`, outputs merged in shard order, duplicates dropped on
    (model, prompt, engine, method, condition) like `funnel_local.merge_chunks`);
  - funnel analyze: one per (stratum × stage × spec) task, the 12 heavy `P|C` tasks split into bootstrap ranges
    (new `--units START:END` option in `funnel_models._run` callers via a thin wrapper; `full-0` computed first and copied);
  - funnel_importance: 8 tasks; intent analyze: 12 reranker strata; steelman generator: 4 strata × 10 draw-ranges;
    steelman fe: 4 strata; chain/followup/pairs/lexical/supply/ablation/census: one each per split;
  - GPU: (view × 20k-text shard) for queries and answers.
- `worker.py`: runs inside one allocation, claims tasks with an atomic `O_EXCL` claim file on the shared filesystem,
  runs them with all allocated cores, writes `done/<id>.json` (commit, seconds, output digest), stops claiming at
  Slurm end − margin (exit 4). Ownership never expires on its own; a task claimed by an ended allocation is released only
  by `ledger reconcile` after `sacct` shows the allocation terminal (AGENTS.md).
- `status.py`: counts by state, CPU-hours done/remaining from measured throughput, storage (`df`, inode count).
- Many small units, few allocations: each allocation is one node working through the ledger, so the work divides
  across any number of nodes and any job length without changing the math.

### 4.3 New analyses (pre-registered in `PREREG.md` addendum B3 before any HoreKa run)
- **supply** (claim 5): per keyword × engine supply table (share of rows with u ≥ .6/.7/.8, max u, SD of u, rows);
  oracle slopes over own rows U, retrieved R, candidates C, shortlist P and the whole snapshot (`intent_stages.oracle_values`
  logic, closest-on-axis first, observed cited count L, rank weights); utilization β_K / β_oracle; ceiling by x decile;
  slope moderated by supply tercile. Decision rule fixed in B3 for the wording "bounded by scarcity" (e.g. supported only
  if the own-row oracle's mean at the top x decile is below the prompt position and the observed slope rises with supply
  across keyword terciles with an interval excluding 0). The Mac quick-look numbers are declared as already seen.
- **ablation**: natural vs ablated cells of the same prompt × engine × method (target URL removed before the rerank):
  change in shortlist intent ΔP and in cited intent ΔK, pass-through ΔK/ΔP with keyword-bootstrap intervals, by model ×
  method. This is the only experimentally manipulated part of the data.
- **census**: the full count — cells planned / completed / with valid trace / passed extraction checks, by model × method
  × engine × condition × split; failures by reason; keyword coverage; duplicates dropped; this table becomes the paper's
  data appendix.
- **queries among keyword-naming prompts**: keyword share of queries restricted to prompts that name their keyword
  (closes the 61% → 21% caveat), added to `followup`.
- Existing parts rerun on both models and both splits: chain, followup, pairs, generator (Llama keep modelled by the
  B1 rule), fe, lexical (frozen-search rule), plus funnel P1–P4, generator decisions, intent stages with Q, A and Gemma G.

### 4.4 Bugs fixed on the way (found by exploration)
`answer_readiness.py export` without `.partial` (blocks reruns); answers-analysis gate and missing from
`paper_results.FILES`; steelman `generator.cache` not keyed by commit/settings; funnel analyze shard imbalance (replaced by
the ledger); `funnel_local` majority-model labelling (1,716 Llama answers tagged Qwen).

## 5. HoreKa resources and budget (to approve)

- CPU: cpuonly nodes, whole node per allocation, **6-hour allocations**, at most 4 concurrent, ≥ 10 minutes between
  observed starts. Expected 3–5 node-allocations (≈ 18–30 node-hours); **cap 8 node-allocations (48 node-hours)**.
- GPU: one or two 4-A100 allocations of 2 hours for the embeddings (validated profile); cap 3.
- Login/transfer node: finite fetch helper for the missing bundles (≈ 5–15 GB) and the Gemma Llama shards if absent.
- Storage: ≈ 20–40 GB new (imports, extracts, caches); check `df`/quota and inodes before each admission; stop on failure.
- Cheaper fallbacks: secondary specifications at 50 draws; skip `fe`; one CPU node at a time.

## 6. Deliverables

- `analysis/steelman/IMPORTANT.html` (local, self-contained): what must be done to finish, the corrected abstract and
  contribution list with each claim's status, the run plan with commands, the checklist; linked from a new handoff
  `analysis/docs/handoff/2026-10-08_full-run-plan_handoff.md` (index line) and from AGENTS-style "start here" note in it.
- Code: `analysis/fullrun/`, `analysis/container/`, steelman parts `supply`, `ablation`, `census`, PREREG addendum B3, tests.
- HoreKa runbook `analysis/docs/horeka-fullrun.md` + `horeka-fullrun-cpu.sbatch` / `horeka-fullrun-gpu.sh`.
- After the run: one tarball → Mac → `RESULTS.md` (both models, both splits), updated `IMPORTANT.html`, draft
  abstract/contributions with final numbers, the census table, trace folder update.

## 7. Execution order

1. Mac, docs first: `IMPORTANT.html` + handoff (status table above, corrected abstract, open questions on citations and
   glued-title lines). Commit.
2. PREREG addendum B3 (supply, ablation, census, keyword-naming queries, full-run reporting by split). Commit before code
   that computes them runs on HoreKa data.
3. Code + tests on the Mac: container files, `git_commit` env fallback, `fullrun` plan/worker/status/reconcile, sharded
   extract wrappers, unit-range funnel analyze wrapper, new parts, bug fixes. Synthetic fixtures (two-model pipeline from
   `test_funnel_study.pipeline`, `test_steelman_horeka`). Docker smoke run. Commit; Valerian pushes.
4. HoreKa (Valerian runs, I prepare commands): status, quota, `apptainer --version`; pinned checkout; build SIF; reconcile
   local cells against the Hub; fetch missing bundles on the login node; `fullrun plan`.
5. CPU wave 1 (extraction shards, merge, assemble, replay) → GPU allocation (queries, answers) → CPU waves (analysis units),
   reconciling and re-estimating after each allocation.
6. Merge, report, tarball; Mac: final RESULTS, IMPORTANT.html update, paper text, trace, handoff.

## 8. Help needed from Valerian

Push the branch; approve the 6-hour CPU allocations and the caps above; confirm Apptainer is available on HoreKa
(`apptainer --version`) and the workspace quota; ensure the login node has Hub read access (token) for the fetch helper;
run the commands and bring back logs and the final tarball. Also: find or drop the Tannenbaum/Martinez references.

## 9. Verification

- Mac: `python3 -m pytest -q analysis/tests/test_steelman_* analysis/tests/test_fullrun_*` plus funnel, intent-stages,
  geo-drivers suites; Docker image runs the same tests; a two-model fixture runs plan → worker → merge → report end to end,
  including an interrupted worker resumed by a second one and a merged split of bootstrap ranges equal to the unsplit run.
- Reproduction gate: the full-run code on the Mac exploration tables reproduces the published steelman numbers exactly.
- HoreKa: `status.py` shows every task done; census counts match the Hub index; every result carries commit, PREREG
  sha256, seeds and input digests.

## 10. Out of scope

Fixed-shortlist test (needs new generations, ≈ 11k answers); human or judge validation of page positions (new judge
inference); Qwen Gemma judgments (none exist).
