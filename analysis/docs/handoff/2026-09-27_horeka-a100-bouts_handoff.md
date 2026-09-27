# HoreKa A100 compatibility and the first 30 Qwen bouts — 2026-09-27

Operator: Valerian runs every cluster command in his own shells and pastes the
output back; the agent prepares paste-ready blocks (local HTML pages) and never
fabricates cluster output. Cluster facts below come from pasted outputs of
27 Sep (HoreKa local time); recheck before acting. Continues
[2026-09-27_llama-round6-qwen-published_handoff.md](2026-09-27_llama-round6-qwen-published_handoff.md)
and [2026-09-27_llama-qwen-horeka_session_handoff.md](2026-09-27_llama-qwen-horeka_session_handoff.md)
(their JUPITER sections still stand; this document supersedes their HoreKa sections).

## 1. Code state

Worktree `.worktrees/threehour-relaunch-fix`, branch `threehour-relaunch-fix`,
pushed to `origin/codex/pilot-continuation`. **Current HoreKa pin:
`184d03859cb924307a1c262e5978c258cc3accd1`** (set in the HoreKa setup file).

| Commit | Change |
|---|---|
| `ab263af` | HoreKa whole-node exclusivity: `horeka.json` `full_node_representation` (152 CPUs, 4 GPUs) + `shaped_full_node_allocation` rule; job 5162814's record now passes |
| `0264ad8` | `horeka_qwen_probe.py submit --no-submit`: prepares an attempt and writes `interactive.json` (salloc request + unchanged `execute`) without sbatch |
| `07cf679` | Earlier 27 Sep handoff committed |
| `184d038` | `analysis/scripts/horeka_qwen_bouts.py` (`divide`, `submit`, `execute`, `status`) + `analysis/tests/test_horeka_qwen_bouts.py` (10 tests) |

Focused suites (bouts, probe, boundary, stage runner, publish, HoreKa prep):
72 passed on the Mac with conda `python3` (the worktree has no `.venv`).

## 2. Operator pages (local, not in git)

In `/Users/valerianfourel/Hamburg/GEODML_Unified/`:

| Page | Purpose |
|---|---|
| `horeka-qwen-bouts.html` | **Current runbook**: steps 0–10 (running?, setup pin, preflight, Hub state, divide, review, dry run, submit 1–30, watch, stop rule, totals) |
| `horeka-interactive-run.html` | Interactive A100 compatibility run in plain terminals A/B/C (12 steps) |
| `horeka-status.html` | Read-only status blocks and the older interactive runbook |

HoreKa setup file: `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-horeka-env.sh`
(`PIN=184d038…`; sourcing it creates a clean checkout under `~/geodml-checkouts/`).

## 3. HoreKa facts (verified from pasted output)

- Account `hk-project-p0026831`, association QoS `normal` with no MaxJobs/MaxSubmit/MaxWall.
- `accelerated`: 162 nodes, MaxTime **2 days**. `dev_accelerated`: 3 nodes, 1 h, QoS `dev` = 1 running / 4 submitted per user.
- `casualnet`: ACTIVE until 31 Oct 18:00, 16 nodes, our account only; no per-job time limit.
- Workspace `geodml-qwen` expires 23 Nov. Node driver 595.71.05 (CUDA 13.2); runtime Python 3.12.9, vLLM 0.28.0, torch 2.13.0+cu130.
- HF token on HoreKa is **read-only**; publishing from HoreKa needs a write token Valerian adds himself.
- `salloc` opens its shell on the compute node (`use_interactive_step`). `srun --jobid … --overlap --pty bash` works for monitoring, but `execute` fails there (`requires an allocated node count`: no `SLURM_JOB_NUM_NODES`), so run `execute` only in the salloc shell.
- `vllm --version` fails on login nodes (`Failed to infer device type`: no GPU); harmless.

## 4. A100 compatibility (done)

- Job 5166810 (dev_accelerated, hkn0401): stages 0–2 passed, vLLM start failed at 14:22: FlashInfer JIT-compiled `sampling.cu` with `nvcc` 12.4 and `-ccbin` Intel `icx` (from `CC`/`CXX` of the default modules), which nvcc rejects. Allocation ended with an SSH disconnect after 29 min.
- Job 5167484 (dev_accelerated, hkn0401): with `CC/CXX/CUDAHOSTCXX/NVCC_CCBIN` = system GCC 11, **`compatible`, 12/12 cells, 915.9 s**. Weights 20.2 s (12.89 GiB/GPU), engine init 189.5 s (31.9 s compile), server ready ≈5.2 min after start, 12 parallel cells ≈10.5 min. Controller runs `--request-concurrency 1 --cell-concurrency 12` (frozen reference runtime), so GPU use is bursty.
- `horeka_qwen_bouts.execute` sets the GCC variables itself. The shared-hour runtime (`agentic_hour_runtime`) does **not** yet.
- Lesson: an interrupted run was suspended with Ctrl-Z, leaving its vLLM server (own session) holding the GPUs. Stop runs with Ctrl-C once and wait; never Ctrl-Z.

## 5. Division of all remaining Qwen work

Division `…/qwen-bouts/division-20260927-1813` (path also in `…/qwen-bouts/CURRENT`),
Hub revision `859f469`:

| Item | Value |
|---|---|
| Registered Qwen cells | 312,096 |
| Published on the Hub, excluded | 24,218 |
| Remaining | **287,878** |
| Bouts (5 h, one A100 node) | **893** (4,465 node-hours, 17,860 GPU-hours) |
| Per bout | 318–324 primary cells (whole prompts, ≤328 = 52.5 s/cell after 6 min start, 5 min admission stop, 2 min cleanup) + 82 spill-over cells from the next bout |

All bouts write to the HoreKa dataset root (`$DS`) in direct dataset mode with
a unique writer per job; the striped task ledger admits each cell once.
Decisions (Valerian, 27 Sep): **HoreKa owns all remaining Qwen** (JUPITER runs
no Qwen until HoreKa results are published), overbook 1.25 × with the shared
ledger, results stay on HoreKa and are published later.

## 6. Submitted: bouts 1–30

- Approved: 30 independent jobs, 1 node / 4 × A100 / 05:00:00 each in `casualnet`, ≤600 GPU-hours, overriding the five-allocation cap and ten-minute start gap for this finite test; no retries.
- Jobs **5167563–5167592** (bout 1 = 5167563 … bout 30 = 5167592), receipt `$DIV/submissions/submitted-0001-0030.json`.
- At 18:19: all 30 PENDING (`Resources`/`Priority`), start estimates `N/A`. **Not yet started at handoff.**
- Direct dataset mode has **not yet run on HoreKa**: the first started bout is the real test.
- The 4 older diagnostic jobs (5164061, 5164064, 5164070, 5164075) are still held (`JobHeldUser`); release or cancel is Valerian's decision.

## 7. Next actions, in order

1. Watch the first bout start (`horeka-qwen-bouts.html` step 8): `Application startup complete`, no `FAILED`, then rising `completed`. If the first 2–3 bouts fail identically before any cell, step 9 cancels only pending bouts; then diagnose from `server.log` and `bout-result.json`.
2. After bouts finish: `status`, measured seconds per cell from real 5 h bouts, and re-size the remaining 863 bouts only if the rate differs materially (new division; never edit submitted bouts).
3. Not yet implemented: a leftover sweep list (cells still missing in the ledger after their bout), and publishing HoreKa results (write token + `publish_qwen_results.py` against `$DS`).
4. Next ranges (bouts 31+) need a fresh estimate and approval.
5. Put the GCC setting into the shared-hour runtime path as well; optionally let the boundary check accept `srun --overlap` shells.
6. Decide on the 4 held jobs.

Standing rules unchanged: fresh approval with estimate for every allocation;
never cancel live jobs without explicit permission; git is the handoff
boundary; no scientific-setting changes to make a model fit.
