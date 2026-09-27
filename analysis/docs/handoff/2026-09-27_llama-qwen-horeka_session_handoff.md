# Llama waves, Qwen publication and HoreKa A100 session handoff — 2026-09-27

Operator: Valerian runs every cluster command in his own open login shells and
pastes the output back. The agent prepares paste-ready blocks (mostly on local
HTML pages), never runs cluster commands itself, and never fabricates output.
Supersedes the operational status in
[2026-09-25_threehour_wave_session_handoff.md](2026-09-25_threehour_wave_session_handoff.md).
Cluster facts below come from Valerian's pasted outputs; recheck before acting.

## 1. Code state

| Item | State |
|---|---|
| Branch / worktree | `threehour-relaunch-fix` in `.worktrees/threehour-relaunch-fix`, pushed to `origin/codex/pilot-continuation` |
| Pushed pin | **`836ca8608c44267405f25edb8a8632c21c4804f3`** (used on JUPITER and HoreKa) |
| `eba0cbf` | Wave-level coordination: registry bytes cached by commit revision; plan/bundle verification once per staging session; `upload_many`; `sync_wave` (one scheduler capture, one reconcile, one registry commit per finished wave); dispatcher `sync-llama` mode (no allocation, no approval). Docs: `analysis/docs/agentic_shared_hours.md` "Wave-level registry cost". |
| `836ca86` | `analysis/scripts/check_axis_ranking_change.py`: read-only real-data checker for the axis → top-3/top-5 analysis |
| **Uncommitted, tested, not pushed** | HoreKa whole-node exclusivity fix: `analysis/config/cluster_execution/horeka.json` (`full_node_representation` 152 CPUs / 4 GPUs with job 5162814 evidence), `inference_network_namespace.py`, `verify_inference_allocation.py`, tests in `test_jupiter_inference_boundary.py`. 124 related tests pass. **Needs Valerian's OK to commit and push**, then a new pin on HoreKa. |

Full test suite at `eba0cbf`: 1660 passed; 12 failures (missing `torch` locally,
prompt-manifold tests) are identical on the unmodified tree. One audit-progress
test fails only in a specific combined ordering, also on the unmodified tree.

## 2. Operator pages (local, not in git)

In `/Users/valerianfourel/Hamburg/GEODML_Unified/`:

| Page | Purpose |
|---|---|
| `llama-next-runs.html` | JUPITER Llama runbook: setup, status, discover, sync, registry check, orphan release, dispatch form, round-5/round-6 launch blocks, monitor |
| `qwen-results-to-hf.html` | JUPITER: publish verified Qwen results to HF in tmux (one command at top) |
| `horeka-a100-checks.html` | HoreKa login-node tests 0–9 plus the diagnosis section |
| `horeka-qwen-quick-call.html` | HoreKa 15-minute vLLM/Qwen call check; section 4 = `dev_accelerated` submission |

JUPITER setup file (instead of full `$HOME`): `/e/fscratch/scifi/fourel1/geodml/audits/llama-prep/env.sh`.
HoreKa setup file: `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-horeka-env.sh`.

## 3. JUPITER Llama

- Registry after cleanup (26 Sep): 73,838 Llama cells completed; 774 eligible
  packages / 236,541 cells before round 5. Remaining work ≈ 150 node-hours at the
  realized ~1,620 cells/node-hour (215 at the worst one-hour rate).
- Round 4 (`audits/threehour-wave-round4/llama`, pin 2e37d3a) died after staging 4
  of 10 members; its dispatcher (PID 2418749) had been suspended with Ctrl-Z and
  held `$MEAS/expansion.lock`. The 4 attempts (109 packages) were released with
  `release-unstarted --apply` after a clean dry run; the process was killed.
- **Round 5 running**: 10 × 5 h, jobs 2069428, 2069430, 2069431, 2069433, 2069435–2069440,
  output `audits/llama-round5`, approved 200 GPU-hours, cap 10, simultaneous starts,
  stale-quota exception, FPH 1600.
- **Round 6 prepared, not submitted**: 20 × 5 h, FPH 1450, cap 20, 400 GPU-hours
  (approval text in the page's R6 block). Its attempt was refused by the
  concurrency gate because round 5 is live; the refusal left only the guard
  `expansion-llama-r6.json` → `audits/llama-round6` (reused by the same block).
  Launch after round 5 ends: page step 6 (all final) → step 2 (`NEXT_ROUND=6`)
  → R6. Running both concurrently needs a code change plus approval of 30 nodes.
- JUPITER home (`/e/home/jusers/fourel1/jupiter`) is **over quota**. Nothing in the
  runbook writes there any more; the diagnosis commands (du of home, caches) were
  given but not run. Check that jobs do not write caches to home.

## 4. Qwen

- HF registry (read 26 Sep): **0 Qwen packages**; all 309,924 Qwen cells deferred
  `awaiting_calibration`; plan `plan-f0bb16b258434743c7952ae2` calibrates Llama only.
- JUPITER Qwen results live in the local ledger of
  `datasets/experiment-v2-incremental-v1` (24,218 verified at the checker run);
  only ≈2,172 are on HF. **Publication started** via tmux session `qwen-publish`
  on login node `jpbl-s02-03` (first attempt stopped at the check while Llama ran;
  the second version accepts running jobs only when each maps to a saved Llama
  attempt writing another dataset root and records them in the snapshot).
  Outputs: `audits/qwen-publish/{run.log,publish.log,scheduler-snapshot.json,input-bundle.txt}`.
  **Outcome not yet reported.** Success = `QWEN_RESULTS_ON_HF=yes`, `ALL_DONE`.
  Do not start Qwen jobs on that dataset until then.
- Next milestone after publication: build/publish Qwen packages with
  `prepare_jupiter_qwen_hours.py` from the member-one GH200 measurement
  (`audits/qwen-threehour-wave-member1-6909506`). Needs (a) an `--input-bundle`
  option (it currently reuses the old plan bundle and would refuse the unpublished
  completions otherwise), (b) downloading the new bundle into the planning root
  (Llama site dataset `audits/llama-five-872875d/dataset`), (c) no registry change
  during planning (window between Llama waves).
- Decision still open with Valerian: after Qwen packages exist, **all** Qwen work
  (JUPITER too) must go through HF packages; the local-ledger continuation waves
  stop. Then generalize `dispatch_threehour_wave.py` (hard-wired to JUPITER Llama
  in `groups`, `admit`, `llama`) to (model, cluster).

## 5. HoreKa (A100)

- Account `hk-project-p0026831`, partition `accelerated`, reservation `casualnet`
  ACTIVE until 31 Oct (16 nodes). `sinfo` is denied (normal).
- Workspace `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen`, expires 23 Nov;
  quota ample. Runtime `environment/qwen-runtime`: Python 3.12.9, vLLM 0.28.0,
  torch 2.13.0+**cu130**, `pip check` clean. Weights 52 GB (HF cache symlinks).
  Pinned checkout `~/geodml-checkouts/geodml-836ca86…`.
- HF login on HoreKa is **read-only**; production publication needs a write token.
- Job **5162814** (25 Sep compatibility probe) **FAILED after 31 s** at the
  whole-node exclusivity check: HoreKa reports `OverSubscribe=NO`, no Exclusive/
  Shared field, all 152 CPUs, `gres/gpu=4`, the same shape JUPITER uses, but the
  JUPITER full-node rule was disabled for HoreKa. Fixed locally (section 1).
- Frozen serving settings (profile
  `shared-hours/dataset/artifacts/shared-inputs/qwen38/d8d4866f…/qwen38.json`):
  TP 4, bf16, max_model_len 41,984, gpu_memory_utilization 0.9, concurrency 4,
  prefix caching, xgrammar, language-model-only. Estimate: KV need ≈11 GB vs
  ≈76 GB free after weights (16 of 64 layers use full attention) → fits.
- `casualnet`/`accelerated` start estimates were 27–30 Sep; `dev_accelerated`
  starts within minutes. The three pending casualnet jobs (5164061 sanity,
  5164064 and 5164070 quick calls) are cancelled and one **15-minute quick-call
  job in `dev_accelerated`** is submitted by page section 4 (approved: 1 node,
  4 × A100, ≤1 GPU-hour, diagnostic only). **Result not yet reported.**
  Success = `SERVER_READY`, three `CALL` lines, `CONCURRENT … tokens/s`,
  `QWEN_CALLS_OK`. Unknown until then: whether the node driver supports CUDA 13.0.

## 6. Axis → top-k ranking analysis

- Checker on `experiment-v2-incremental-v1`: **PASS, 0 failures**, 627 keywords;
  report code matches independent recomputation for every pair and correlation.
- Coverage is thin: median 1 prompt per comparison group; ~80–100 Qwen keywords
  usable; Reactive-Snippet-Loop top-5 mostly short lists (3 URLs). Empty-ranking
  pairs (up to 970 per Qwen group) and tied pairs are silently dropped by the
  report, contrary to the spec's exclusion reporting. Pooled Qwen association
  ρ ≈ +0.09…+0.14 (descriptive only). URL spelling irrelevant.
- The Llama wave dataset lacks `local-only/population-registration-v1/`, so the
  report cannot read it. Proposed (awaiting OK): `--registration DIR` option in
  report and checker, and counting empty-ranking pairs as excluded.

## 7. Next actions, in order

1. HoreKa: read the `dev_accelerated` quick-call result (page section 4, second block).
2. Ask Valerian to approve commit + push of the HoreKa exclusivity fix; then update
   the HoreKa page's setup pin and run the one-hour probe (`horeka_qwen_probe.py
   submit`, already approved: 1 node, 1 h, 4 GPU-hours) — in `dev_accelerated` if
   its limits allow, otherwise casualnet.
3. JUPITER: read `audits/qwen-publish/run.log`; expect `QWEN_RESULTS_ON_HF=yes`.
4. After round 5 ends: launch round 6 from the page (step 6 → step 2 → R6).
5. Qwen packages milestone (section 4), then the (model, cluster) dispatcher,
   HoreKa HF write token, and a small first HoreKa wave to measure A100 speed.
6. Optional: `--registration` for the axis report; JUPITER home cleanup.

Standing rules unchanged: fresh approval with estimate for every allocation;
never cancel live jobs without explicit permission; git is the handoff boundary;
no scientific-setting changes to make a model fit.
