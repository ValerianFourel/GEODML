# GPU backend for the heavy statistics of the full run (+ relocation check before embedding)

Valerian asked (2026-10-08) to adapt the CPU jobs of the full run to a GPU approach (brief pasted in the session) and to
keep the relocation check as the first GPU step. Plan approved; CPU numpy/scipy stays the default and the reference.

## What exists now (branch `codex/acl-figures-20261006`, not pushed)

- `analysis/interpretability/pipeline/torch_fits.py`: float64 damped Newton with exact gradients and Hessians on the same
  objectives as the CPU fits (choice/Plackett–Luce with position effects; Chamberlain conditional logit with autograd
  Hessians of the elementary symmetric polynomials); resident design, `free` masks for drop-one-block fits, chunking;
  Monte-Carlo twins fed the same numpy random numbers; two-way fixed effects for all bootstrap weights at once.
- Backend switch `--backend {cpu,torch-cpu,cuda}` in `funnel_study.py analyze/report`, `funnel_importance.py` and the
  steelman parts; `funnel_models.set_backend`; the torch path of `estimate_blocks` writes the same unit keys and values;
  the backend enters cache keys only when not cpu (existing CPU caches stay valid); assembly needs no GPU.
- Full run: ledger tasks carry `gpus`; the GPU worker hands each 1-GPU task its own device and gives embedding tasks all
  four; `--python` lets tasks run in the GPU container; planner option `gpu_stats` (true or a list of
  funnel/decisions/generator/fe); relocation tasks (sample 512 archived prompts → embed both views → `relocate`) gate every
  query and answer embedding and feed `--relocation` to the intent analysis.
- Containers: `analysis/container/Dockerfile.gpu` / `geodml-gpu.def` (CPU lock + torch 2.5.1, CUDA 12.1 wheel; CPU wheel
  for Mac tests). Batch script `analysis/docs/horeka-fullrun-gpu-stats.sbatch` (accelerated, 4 A100, 4 h placeholder).
  Runbook section 5b in `analysis/docs/horeka-fullrun.md`. PREREG computational note C1 (gates) committed first.
- Tests: `test_torch_fits.py` (13 in the torch image), `test_fullrun_gpu_slots.py`, planner tests; 104 Mac tests pass.

## Gates (analysis/fullrun/validation/REPORT.md)

| Gate | Result | Route |
|---|---|---|
| Synthetic (`test_torch_fits.py`, torch image) | all pass: coefficients within 1e-7 and objective within 1e-10 of a tight CPU optimum; inclusion probabilities 1e-12; Monte Carlo identical to numpy; FE within 1e-8; unit ranges identical | — |
| Steelman generator, Qwen · Reactive, 100 draws | **pass** under note C1: Δ_gen within 4.1e-6, interval endpoints within 1.5e-4, same TOST verdict, 0 failed replicates (coefficients differ by up to 9.1e-4: CPU early stopping) | GPU |
| Steelman fixed effects, both Qwen strata, 200 draws | **pass**: 36 of 36 values within 1e-6 (in fact about 1e-15), after fixing the batched solve to fail only singular draws | GPU |
| Funnel `P|C main`, Qwen · Reactive, full fit + 3 draws vs cached CPU | **fail** at 1e-4 (up to 1.4e-3); diagnosis: a tightly converged CPU fit has the same objective as the torch fit (2.2348226179799) and differs from it by 4.3e-8, while the cached CPU fit stopped early (objective 2.2348226394673) | CPU until Valerian decides |
| Generator decisions (same estimator as the funnel) | not run separately; same status as the funnel | CPU until Valerian decides |

Mac timings (torch on CPU, not the GPU): funnel shortlisting fit, 1.1M rows × 57 features, 14.5 s full fit and about
10 s per bootstrap unit; generator, one stratum, 3,247 s; fixed effects, two strata, 6,147 s. On A100s the fits should
take a small fraction of that [H: to be measured in the first allocation].

## Needs Valerian

Push; approve (or not) the GPU-statistics allocations (up to 3 × 4 h on accelerated, 4 A100, ≤ 48 GPU-hours) and a
15-minute dev_accelerated check; decide whether the funnel/decisions gate reference may be the tightly converged CPU
optimum (see the funnel diagnosis in REPORT.md). Until then plan with `"gpu_stats": ["generator", "fe"]` (the
estimators that passed; about 130 of the 630 CPU-hours move to the GPU) or leave it off. If the tight CPU optimum is
accepted as the reference, `"gpu_stats": true` moves about 570 CPU-hours.
