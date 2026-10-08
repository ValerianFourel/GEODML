# Full run of the intent-in-the-pipeline study: what was launched, and how to analyse its results

Written 2026-10-08 for the session that analyses the results, about 48 hours later. This is the single entry point.
Earlier context: `2026-10-08_full-run-plan_handoff.md` (plan), `2026-10-08_gpu-stats_handoff.md` (GPU backend),
`2026-10-07_steelman-shortlist-claim_handoff.md` (the Mac steelman). What the results are for:
`analysis/steelman/IMPORTANT.html`. The rules: `analysis/steelman/PREREG.md`.

Evidence labels: **F** = fact seen in output or code; **P** = plan; **A** = assumption; **H** = estimate.

## 1. The study in one paragraph

Experiment V2 has 26,000 prompts, each with a measured position x on a 0–1 axis from information seeking to action
readiness. x is a property of the prompt text, not a randomized treatment, so every result is an association.

Each cell runs an agentic search pipeline and a generator LLM (Qwen3-8B "qwen38", Llama 4 "llama4"). The pipeline
steps are:
1. R0 is what the prompt's words alone retrieve, a mechanical lexical replay.
2. R is what the agent's own queries retrieve.
3. C is the reranker's candidates.
4. P is the shortlist shown to the generator.
5. I is the shown order.
6. K is what the answer cites, in order.

The page intent u is a page's percentile on the prompt scale. The study asks where the slope of cited intent on x comes
from, and splits β_K into exact increments, one per step.

The paper's main claim (memory "paper-main-claim"): **intent gets a page shortlisted; topic decides its rank**.

The pre-registered claims:
- **C1, admission.** Intent reaches the cited sources mainly through the shortlist, via query rewriting and the
  reranker. The generator's own share s_gen = (β_K − β_I)/β_K is small.
- **C2, selection.** Given the shortlist, the generator's keep and order decisions are intent-insensitive in their
  effect on the cited-intent slope. Δ_gen sits within a smallest effect of interest (SESOI) of 0.015 by a TOST.
- **Supply (addendum B3).** The shift is bounded by how few action-ready pages exist.

Per AGENTS.md priority: study the generator's own behaviour first (C2, keep 0/1 and order of the kept), then the
reranker and search mechanics.

## 2. State at the time of writing [F]

| Item | Value |
|---|---|
| Branch | `codex/acl-figures-20261006`, worktree `.worktrees/acl-figures` |
| Run commit (task code) | `fdba5e20d624d5f460ca4c12bcac67117b779e30` (later commits are docs, this handoff and the checker only) |
| HoreKa checkout | `$W/checkouts/fullrun-fdba5e20d624d5f460ca4c12bcac67117b779e30` |
| Run folder | `$FR/run2` with `$FR=$W/reviews/page-readiness-20261004/fullrun-v1`, `$W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen` |
| Ledger | `$FR/run2/ledger`: 300 tasks, all pending at submission (68 ready, 232 waiting) |
| PREREG sha256 at the run commit | `b0e41e437465eed2…`; every part file records it |
| Jobs | 20 jobs on `accelerated`, 4 A100 + 152 CPUs, 8 h each, all mixed (CPU_STEPS=1): 5187091–5187102 and 5187115–5187122 |
| Start estimates at submission | first nodes about 2026-10-10 17:00; the rest 17:10–18:40 the same day |
| No cpuonly jobs | Valerian's explicit instruction; the 8 cpuonly jobs 5187103–5187110 were cancelled |
| `run1` | abandoned; never ran a task (its config included the empty `hub-import` source) |
| Hub check | 0 missing cells: completed unique qwen38 312,042 and llama4 312,090; all local |

**Not yet verified:**
- no task of `run2` has run;
- no result exists;
- the start times are estimates only.

## 3. What the 300 tasks compute [F, from `analysis/fullrun/plan.py` at the run commit]

| Tasks | Stage | What they do | Output under `$FR/run2/` |
|---|---|---|---|
| 32 `extract-funnel-NN` + `merge-funnel` | extract, merge | per-answer funnel rows from every completed cell, sharded by keyword hash, exact merge | `funnel-extract/` |
| 32 `extract-trace-NN` + `merge-trace` | extract, merge | stage traces, agent queries, answers | `trace-extract/` |
| `funnel-replay`, `funnel-assemble` | prepare | R0 prompt-words replay (frozen search rule), joined table | `funnel-replay/`, `funnel-assembled/` |
| `intent-replay`, `gemma-extract`, `answers-export` | prepare | prompt-text replay for intent stages, Gemma judge records (Llama reuse run), answer texts | `intent-replay/`, `gemma-extract/`, `answers-export/` |
| `relocation-sample`, `embed-sample-{qwen,mistral}`, `relocation` | prepare, gpu | 512 archived prompts re-embedded; must reproduce the archived axis before any new embedding | `relocation-fresh.json` |
| `embed-queries-{qwen,mistral}`, `embed-answers-{qwen,mistral}` | gpu | LLM2Vec embeddings of agent queries and answers (validated 4-A100 profile) | `embed-*.merged/` |
| `answers-analyze`, `intent-analyze` | analysis | answers on the axis; intent through the stages in text | `answers-analysis/`, `intent-stages/` |
| 2 splits × 4 strata × 4 stages × 3 specs funnel tasks (176 tasks incl. bootstrap/permutation units of P\|C main) | funnel | stage selection models (CPU estimator) | `funnel-analysis-{split}/`, `funnel-report-{split}/` |
| `decisions-{split}` | analysis | generator keep/order decision models | `decisions-{split}/` |
| per split: `chain`, `followup`, `pairs`, `census`, `supply`, `ablation`, `queries`, `lexical` | steelman | the steelman parts | `steelman-{split}/<part>.json` |
| per split: `generator` and `fe` per stratum (8 GPU tasks) + aggregation | steelman | Δ_gen (C2) and two-way fixed effects, PyTorch on CUDA | `steelman-{split}/generator.json`, `fe.json` |
| `steelman-{split}-report` | report | RESULTS.md, verdicts, manifest | `steelman-{split}/RESULTS.md` |
| `bundle` | report | every result in one tarball | `paper-results/`, `paper-results.tar.gz` |

**Splits:**
- `exploration` is the keywords already seen in the Mac work.
- `confirmation` is the held-out keywords. **Confirmatory verdicts are those of the confirmation split** (B3).

**Strata:** `llama4 · Parallel`, `llama4 · Reactive`, `qwen38 · Parallel`, `qwen38 · Reactive`. Reactive Loop is
primary for C1.

**Draws:** 200 keyword-bootstrap draws (seed 20261007), 200 shuffles (seed 20261008), generator models on the first
100 draws, 200 Monte Carlo draws per answer.

**Deliberately left out:**
- The lexical shortlisting refits (`--selection-draws 0`, addendum B1). They are reported-only; the lexical part itself
  and the BM25 record do run.
- Gemma judgements exist only for the Llama reuse run (282,478 of 312,052 cells). The Qwen Gemma bouts were still queued
  on 2026-10-08. Gemma feeds the "which sources should have mattered" analysis, not C1, C2 or supply.

**Backends:**
- Generator and fixed effects run on the GPU (PyTorch float64 Newton). They passed their gates
  (`analysis/fullrun/validation/REPORT.md`); `generator.json` and `fe.json` record `"backend": "cuda"`.
- Funnel and decisions run on the CPU numpy/scipy estimator, as before. Their GPU gate was never re-referenced.

## 4. Decision rules to apply, exactly as pre-registered [F, `PREREG.md`, `analysis/steelman/decide.py`]

**C1, per model × method, Reactive primary:**
- *Supported* needs (a) and (b):
  - (a) the 95% upper bound of s_gen is below 1/3 on the common sample, in the controls variant and in each engine;
  - (b) the query-rewriting increment β_R − β_R0 and the reranker increment β_P − β_C both have 95% intervals above 0.
- *Narrowed* if the upper bound is between 1/3 and 1/2, or only the reranker increment is above 0.
- *Failed* if the upper bound is at least 1/2.

**C2, per model, both methods jointly:**
- *Supported* if the 90% interval of Δ_gen (drop-A1 refit) lies inside ±0.015 in both methods.
- *Narrowed* if not, but the 95% upper bound of |Δ_gen| is below 0.030 in both.
- *Failed* if the 95% lower bound of |Δ_gen| exceeds 0.015 in either.
- *Undetermined* otherwise.
- Coefficient-level nullity is never claimed.

**Supply, every stratum:**
- (a) the ceiling gap at x ≥ .9 is above 0;
- (b) the top − bottom supply-tercile contrast of β_K is above 0;
- (c) utilisation of the own-row oracle is at least 0.5 (95% lower bound).

*Supported* if all three hold, *narrowed* if only (a) holds, *failed* if (a) fails. Then the paper drops the scarcity
sentence.

**Reported, never decisional:**
- permutation p;
- zeroed-coefficient Δ_gen, fixed effects, score-matched pairs;
- the lexical selector;
- ablation pass-through;
- keyword-naming queries;
- the census;
- the exploratory keyword-mention follow-up.

**What the Mac already showed** (Qwen, exploration, 237 keywords; exploratory, not to be re-derived):
- C1 supported in both methods;
- C2 supported;
- Reactive shares of β_K: R0 17%, query rewriting 32%, reranker 35%, shown order 5%, generator 10%;
- Δ_gen −0.0001 Reactive, −0.0017 Parallel;
- β_K Reactive +0.068 [+0.059, +0.079];
- supply smoke "narrowed" (utilisation 0.19–0.26);
- ablation pass-through 0.69–0.86.

The full run's exploration split should reproduce the Qwen numbers closely, because it uses the same keywords with all
cells. A large difference there signals a pipeline problem before any science.

## 5. In 48 hours: step by step

### 5.1 HoreKa: is it done? (login shell)

```bash
W=/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen; FR=$W/reviews/page-readiness-20261004/fullrun-v1; source $W/geodml-nemotron-env.sh
CODE=$W/checkouts/fullrun-fdba5e20d624d5f460ca4c12bcac67117b779e30; RUN=$FR/run2; LEDGER=$RUN/ledger
squeue -u $USER -n geodml-fullrun-gpustats -o '%.10i %.8T %.10M %.20S'
(cd $CODE && $RT/bin/python -m analysis.fullrun reconcile --ledger $LEDGER && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER) | head -n 40
sacct -X -u $USER -S 2026-10-08 --name=geodml-fullrun-gpustats -o JobID,State,Start,End,Elapsed,ExitCode
```

**Reading the status:**
- `"done": 300` means complete.
- `"running"` lists tasks claimed by live jobs.
- `failed_tasks` lists tasks that failed twice.
- `checkpoints` counts deadline checkpoints. These are normal: the next node resumes them.

### 5.2 If tasks failed

```bash
CODE=$CODE LEDGER=$LEDGER bash $CODE/analysis/docs/horeka-fullrun-relaunch.sh
```

It prints each failed task with the end of its log. Then:
- **Transient cause** (node fault, storage, timeout): run the same script with `--retry-all`. The work resumes as long
  as at least one of the 20 jobs is still pending or running. If none is left, submit a few more mixed jobs with the
  submission loop in §5.6.
- **Code bug:**
  1. Fix it on the Mac, test, commit and push.
  2. Make a new clean checkout on HoreKa.
  3. Run the relaunch script with `--repin <new checkout> --retry-all`. Repin is refused while any task is claimed.
  4. From then on, use the new checkout as `CODE`.
  5. Record the second commit: parts done before the fix keep their own commit in their headers.
- Failed outputs leave `<output>.partial.interrupted-<time>` folders for inspection. The worker sets them aside
  automatically before each new attempt.

### 5.3 Bring the results to the Mac

On HoreKa:
```bash
cd $RUN && (cd $CODE && $RT/bin/python -m analysis.fullrun status --ledger $LEDGER) > $RUN/final-status.json
T=$W/reviews/fullrun-run2-$(date -u +%Y%m%dT%H%M%SZ).tar.gz
tar -C $RUN -czf $T paper-results.tar.gz final-status.json config.json ledger/tasks.jsonl ledger/done ledger/failed ledger/checkpoint slurm launch
ls -la $T
```

On the Mac, with `<horeka-login>` being the usual HoreKa login host:
```bash
mkdir -p ~/Hamburg/geodml-inputs/fullrun-run2 && cd ~/Hamburg/geodml-inputs/fullrun-run2
scp <horeka-login>:<path printed by ls -la above> .
tar -xzf fullrun-run2-*.tar.gz && tar -xzf paper-results.tar.gz
ls paper-results steelman-exploration steelman-confirmation
```

### 5.4 First look: completeness, provenance, verdicts (Mac)

```bash
cd ~/Hamburg/GEODML_Unified/.worktrees/acl-figures
python3 -m analysis.steelman.check_fullrun ~/Hamburg/geodml-inputs/fullrun-run2 --status ~/Hamburg/geodml-inputs/fullrun-run2/final-status.json --out ~/Hamburg/geodml-inputs/fullrun-run2/verdicts.json
```

The checker (`analysis/steelman/check_fullrun.py`) applies the same rules in `decide.py` to the stored part files and
computes nothing new. It reports:
- missing parts per split;
- that each split has exactly one code commit and one PREREG hash;
- the backends and settings;
- C1 per stratum, C2 per model and the supply verdict;
- β_K, answers per stratum, whether the generator's keep model was used, and failed bootstrap replicates;
- the census.

It exits 1 if anything is missing.

**Checks before believing any number:**
1. `"done": 300`, an empty `failed` list, and `bundle_manifest: true`.
2. Every part in both splits; one commit, `fdba5e2`, unless a repin happened (then two, recorded); PREREG
   `b0e41e43…`; settings 200 / 200 / 100 / 200 with seeds 20261007 and 20261008.
3. `generator.json` and `fe.json` say `"backend": "cuda"`. The chain and every other part say `cpu`.
4. Census: completed cells per model near 312,042 (Qwen) and 312,090 (Llama). Each drop from planned to analysed
   needs a reason in the census.
5. `generator_failed_replicates` near 0. Many failures mean a fit problem, so read them before the verdict.
6. The exploration split reproduces the Mac Qwen numbers in §4 to within bootstrap noise.

### 5.5 Analysis order (AGENTS.md priority)

1. **C2 and the generator's own behaviour:**
   - `steelman-{split}/generator.json`: Δ_gen per stratum, keep and order coefficients, the slot correlations;
   - `fe.json`;
   - `decisions-{split}/decisions.json`: keep 0/1 and order of the kept, per model.
   Llama is new here: its keep decision is modelled wherever it drops links.
2. **C1:** `chain.json` (shares and increments, controls, engines, given lattice target), `followup.json`,
   `lexical.json` (selector comparison, frozen-search rule), `pairs.json` (score-matched).
3. **Supply:** `supply.json` (oracle slopes, utilisation per pool, terciles, ceiling gap).
4. **Reported extras:** `ablation.json`, `queries.json`, `census.json`.
5. **Funnel stage models:** `funnel-report-{split}/results.json` and `final-report.html`, for the stage-by-stage
   coefficients behind "intent gets a page shortlisted; topic decides its rank".
6. **Intent in text:**
   - `intent-stages/results.json` and `final-report.html` (queries and answers on the axis, with the relocation check);
   - `answers-analysis/report.md` and `answer_coordinates.jsonl.gz`.

Rebuild a split's RESULTS.md at any time with:
```bash
python3 -m analysis.steelman report --output <steelman-split folder>
```

### 5.6 Submitting more nodes, if needed

Use only mixed accelerated jobs (memory "cpu-work-on-gpu-nodes"):
```bash
for i in $(seq 1 4); do sbatch --parsable --time=08:00:00 --export=ALL,CODE=$CODE,LEDGER=$LEDGER,CPU_STEPS=1 \
  --output=$RUN/slurm/gpustats-%j.out $CODE/analysis/docs/horeka-fullrun-gpu-stats.sbatch; done
```

## 6. Then update the paper material

- `analysis/steelman/RESULTS.md`: replace it with the confirmation-split results, plus a table of both splits.
- `analysis/steelman/IMPORTANT.html`: claim by claim, set the status from the verdicts and finalise the abstract and
  contributions. Tick the checklist.
- The paper text:
  - observational wording;
  - every number labelled F/P/A/H in plans;
  - no mention of glued DuckDuckGo hub rows or the contamination artifact (memory);
  - concise slides.
- Still to fix in the manuscript:
  - the glued-title mentions in `manuscript_draft/sections/methods.tex:60` and `introduction.tex:6`;
  - find or drop the Tannenbaum and Martinez references.
- If a confirmatory verdict differs from the Mac exploration verdict, the confirmatory one is reported. The paper
  wording follows the narrowed or failed branch that PREREG fixes, with no new variants. Anything added after seeing
  the results is labelled exploratory.

## 7. How we got here (2026-10-08) [F]

| Step | What happened |
|---|---|
| Plan | Full run of every analysis, split reporting, GPU embeddings; PREREG addendum B3 and computational note C1 |
| GPU backend | PyTorch float64 Newton for generator, fixed effects, funnel and decisions. Generator and fe gates passed. Funnel/decisions failed against early-stopped CPU fits (tight CPU optimum agrees to 4e-8), so they stay on CPU |
| Smokes on HoreKa | CPU smoke, GPU smoke 7/7, CUDA equivalence (cuda vs cpu within 1.1e-5), real 1-of-500 CPU chain 32/32 |
| Launch tooling | `fullrun retry`, `repin`, `mixed`; `horeka-fullrun-launch.sh` (finite admission, fill mode); `horeka-fullrun-relaunch.sh` (report, repin, retry) |
| Queue reality | cpuonly estimated no start before 2026-10-12 14:27. Valerian chose to run all CPU work on the CPU cores of accelerated GPU nodes and to submit every job at once, explicitly overriding the 5-allocation limit and the 10-minute start gap |
| Bugs found before any real job ran | (1) independent CPU and GPU workers could both give up before GPU inputs existed; fixed by a supervisor. (2) The empty `hub-import` source failed every extraction; planning now refuses sources without sealed tables. (3) A failed attempt's `.partial` folder blocked retries; set aside automatically. (4) Two worker-exit races. (5) The launcher counted pending Gemma jobs toward the limit |
| Development-node tests | Job 5187074 found bug (2). Job 5187088 on `dev_accelerated` ran the production batch script in mixed mode: 38/38 done, exit 0. It covered real 1-of-500 extraction to assembly, funnel, decisions, every steelman part, the 8 generator/fe strata on CUDA with one GPU each, the device slots and a real 4-GPU Qwen embedding |
| Submission | 12 + 8 mixed accelerated jobs of 8 h; the cpuonly jobs cancelled |
| HoreKa limits seen | `scontrol top` is denied to users. `scancel --name` with a comma list matches nothing, so cancel by ID. A fresh login shell loses the variables; set them first |

**Queue note.** Valerian's 157 pending Gemma v4 bouts (job names `geodml-gemma-v4-bout`) compete with the full run on
`accelerated`. Raising their `Nice` would let the full run start first. That is Valerian's call; it was proposed, not
done.

## 8. Files

- Code:
  - `analysis/fullrun/`: `plan.py`, `ledger.py` (worker, supervisor, retry, repin), `smoke.py`, `hub.py`, `merge.py`,
    `gpu_validation.py`, `validation/REPORT.md`;
  - `analysis/steelman/`: parts, `decide.py`, `report.py`, `check_fullrun.py`, `PREREG.md`, `RESULTS.md`,
    `IMPORTANT.html`, `trace/TRACE.md`.
- Batch scripts: `analysis/docs/horeka-fullrun-gpu-stats.sbatch`, the one used. `horeka-fullrun-cpu.sbatch` is not to
  be used for this run. Also `horeka-fullrun-embed.sh`, `horeka-fullrun-launch.sh` and `horeka-fullrun-relaunch.sh`.
- Runbook: `analysis/docs/horeka-fullrun.md`.
- Tests: `analysis/tests/test_fullrun_*.py`, `test_steelman_*.py`, `test_torch_fits.py`.
