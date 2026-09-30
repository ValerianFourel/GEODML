# Gemma first replay results and fresh twenty-cell diagnostic

Valerian returned both complete summaries for job 5171305, then requested another
20-cell SI-v3 test with Gemma and full output review. This authorizes a finite
diagnostic in the existing allocation; no new allocation or extension approved.
No remote command was executed by Codex. All cluster facts below are pasted
evidence, not current live state.

## First run evidence

- Code ddc93fe5342152d7c33f2abfc0658a48714d7dcc; run root
  `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/reviews/gemma4-si-v3-ddc93fe`.
- Job 5171305, hkn0401, dev_accelerated; start 2026-09-30T08:19:29,
  scheduled end 09:19:29 in cluster time. One exclusive node, four A100 GPUs,
  32 requested / 152 allocated CPUs, 497500M memory, one approved hour.
- Gemma `google/gemma-4-31B-it`, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, shared qwen-runtime.
- Original frozen inputs SHA256
  `122d810824b1073ff04d7f566ed9fce41fb864f52328d1b1e7a728d1bbc01276`.
- `attempts/job5171305/trial-result.json`: completed, returncode 0,
  elapsed_seconds 464.0727119445801. Trial passed; both pass exit codes zero;
  console GEMMA_EXIT=0. scientific_result=false.
- Each pass: 20 complete cells, 20 successful J1 and 106 successful SI requests,
  all finish reasons stop, no reported execution errors. Seed 2026093010,
  concurrency 4, SI max_tokens 640, temperature 0, thinking off.
- Pass times 47.4 and 44.9 seconds. 252 judgments / 92.3 seconds = 2.73/s,
  approximately 9,829 judgments/hour or 1,560 cell evaluations/hour during
  judging. Including reported stage elapsed gives about 1,955 judgments/hour.
  The 464-second measure excludes earlier preparation/allocation overhead.
  Two passes repeat the same 20 cells: not 40 unique cells. Not production rates.
- Server log shows 66.1–70.4 generated tokens/s in the displayed intervals.
  SIGTERM, forced worker teardown and EngineDeadError followed successful
  requests during shutdown. These messages do not invalidate completed passes.
  Earlier prolog warning resolved before the successful run.
- Both grade histograms: 0:58, 1:16, 2:21, 3:7, 4:3, 5:1.
  Histograms conceal two changes: Winninghunter in 9d3e114a47ba goes 2 to 1;
  UPS in ccf34042bf62 goes 1 to 2. Therefore 104/106 identical source scores,
  18/20 identical source-grade vectors, all 20 J1 scores identical.
- Compliance 9407a920b4e1 has six tied grade-1 sources; Trello 2dd070ba00e1
  has three tied grade-2 sources. Top alignment true does not imply unique
  ranking accuracy. Four all-zero cells and many zeros need source-level review.
  Repeatability alone does not validate semantic support or centrality grades.

## Prepared next test

Code commit `779c897cae570949d09e804d1e2f550995af6cc3` adds a finite fresh
sample and saved-output reader, reusing the existing SI builder and two-pass
replay. No new inference engine or modified rubric.

`fresh_gemma_si_review.py freeze` selects ten Qwen and ten Llama cells, seed
2026093011, across twenty distinct prompt IDs. It excludes all prompt IDs in the
original run inputs. Reads sealed task metadata/latest ledger, then verifies
selected generation and trace records. Selection is deterministic for the same
completed dataset snapshot. Invalid selected evidence fails without replacement.
Frozen bundle records exclusions, dataset roots, task hashes and boundary.
Existing matching bundle can be reused; conflicts and prior-prompt overlap fail.

Unchanged SI-v3: complete citation-masked answer, one source at a time, passage
matches, 0–5 ordinal importance, J1 separately. SI cap 640, J1 cap 64, concurrency
4, temperature 0, thinking off. Same Gemma revision, BF16, TP4, context 73728,
GPU fraction .85 and eager serving. Two passes share one server. No fabricated
Nemotron baseline for the fresh sample.

The new config binds execution to existing job 5171305 and requires RUNNING plus
at least 1200 seconds remaining, before an attempt directory or model startup.
Preparation also enforces this. Estimated stage runtime 10–15 minutes after
input freezing, with a 20-minute minimum remaining window. Estimated work
0.67–1.0 GPU-hours inside the already approved job, no extra allocation budget.
A distinct serving cache avoids changing the earlier run's cache/artifacts.
Run directory is `reviews/gemma4-si-v3-fresh20-2026093011`; frozen input bundle
is its sibling `reviews/gemma4-si-v3-fresh20-inputs-2026093011.json`.

Updated `analysis/docs/horeka-gemma-si-v3.html` and identical convenient root
copy. Eight copy blocks: exact-pin checkout in separate login shell; fresh freeze
and prepare inside compute shell; quota check and run; summaries and score
differences; four batches of five full cell/source judgments. No salloc, sbatch,
scancel, release, hold, requeue or queue mutations. Prior allocation workflow
remains in Git history. User must not repeat old allocation commands.

## Verification and next step

58 focused tests passed with Miniconda Python: test_horeka_gemma_si,
test_fresh_gemma_si_review, test_horeka_nemotron, test_source_importance_pipeline.
New contracts cover prompt exclusions/distinctness, immutable deterministic
freezing on real sealed fixtures, corrupt selected evidence rejection, replay
without baseline, full judgment joins, config binding and rejection of wrong,
ended or short allocations before an attempt. These are CPU/mocked diagnostics,
not scientific results. Eight extracted Bash blocks and embedded Python parse;
HTML copies match; diff whitespace check passed. No GPU run performed locally.

Next: Valerian executes page steps if job 5171305 is still running with enough
time, returns summaries then four output batches. If expired/insufficient time,
stop execution and get fresh wall-time approval before supplying any replacement
allocation command, with recalculated cost/runtime and retained scheduling and
storage guards. Do not release the existing allocation. Judge quality remains
unresolved until full inputs and source-level judgments are reviewed.
