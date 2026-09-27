# Llama round 6 and Qwen publication handoff — 2026-09-27

Operator: Valerian runs every cluster command in his own open JUPITER login
shells and pastes the output back. The agent prepares paste-ready blocks on
local HTML pages, never runs cluster commands itself, and never fabricates
output. Cluster facts below come from Valerian's pasted outputs; recheck before
acting.

Supersedes the Llama and Qwen status in
[2026-09-27_llama-qwen-horeka_session_handoff.md](2026-09-27_llama-qwen-horeka_session_handoff.md)
(written at pin `836ca86`, before the fixes below). Its HoreKa sections still
stand, including the uncommitted HoreKa exclusivity change in this worktree,
which this session did not touch, commit or push.

## 1. Code state

Branch `threehour-relaunch-fix` (worktree `.worktrees/threehour-relaunch-fix`),
pushed to `origin/codex/pilot-continuation`. **Current pin: `a55f38264d9701dd83bc285cadaec9ba794eb2fa`.**

| Pin | Change |
|---|---|
| `eba0cbf` | Wave coordination: registry cached by commit revision; plan/bundle verified once per staging session; `upload_many`; `sync_wave` (one scheduler capture, one reconcile, one registry commit per wave); `dispatch_threehour_wave.py sync-llama` |
| `836ca86` | `check_axis_ranking_change.py`: real-data checker for the axis → top-3/top-5 analysis |
| `032e4bf` | New Hub objects go to `exchange/objects-v2/<2 hex>/<sha>`; the flat `exchange/objects/` hit the Hub's 10,000-files-per-directory limit. Reads fall back to the flat path |
| `8b77f4a` | `publish_qwen_results.py`: publish verified Qwen cells from the shared ledger (never touches the hour registry) |
| `bb2e747` | Hub read timeout 120 s; re-upload checks presence instead of downloading objects |
| `5f16dca` | Every Hub call (repo_info, head, reads, commits) retried on timeouts and dropped connections |
| `a828247` | Upload verification uses the Hub's own content hashes (git blob id or LFS SHA-256) instead of re-downloading; checked live against the private repo |
| `a55f382` | `verify_record_reference` verifies each sealed shard once per process; `import_events` reads the ledger once per bundle. These two made Llama wave preparation read 366 GB in 11 min and stalled the first Qwen publish |

Full suite at `a55f382`: 1680 passed; the 12 failures are the known
missing-`torch` prompt-manifold tests, identical on the unmodified tree.

## 2. Operator pages (local, not in git)

In `/Users/valerianfourel/Hamburg/GEODML_Unified/`:

| Page | Purpose |
|---|---|
| `llama-next-runs.html` | Llama runbook at pin `a55f382`: 0 is-a-launch-running check, 1 setup, 2 Hub link check, 3 launch round 6 (filled, approved), 4 monitor, 5 Qwen (superseded by the page below), 6 results after round 6 |
| `qwen-publish.html` | Qwen publication in four steps (stop old publish, setup, publish, confirm) |

JUPITER setup files (home is over quota, so nothing is written to `$HOME`):
`/e/fscratch/scifi/fourel1/geodml/audits/llama-prep/env.sh` (Llama) and
`…/llama-prep/qwen-env.sh` (Qwen). Step 1 of each page rewrites them for the
current pin. Scratch lists (`llama-sites.txt`, `live-waves.txt`,
`next-round.txt` = 6, `last-wave.txt`) live in `…/audits/llama-prep/`.

## 3. Qwen: published

- Hub index `coordination/qwen-results.json`: **26 bundles, 24,218 completed,
  0 failed**, equal to the verified count in
  `datasets/experiment-v2-incremental-v1` on fscratch. Before, the Hub showed
  only the 2,172 bootstrap cells.
- Future Qwen results: `qwen-publish.html` step 3 again; only new cells upload.
- `coordination/progress.json` is stale for both models and was not refreshed.

## 4. Llama

| Round | Shape | State |
|---|---|---|
| 1, 2 and the one-hour waves | — | Published earlier (`ALREADY SYNCED`) |
| 4 | 10 × 5 h, pin 2e37d3a | Dispatcher was suspended with Ctrl-Z after staging 4 members; killed; its 109 reserved packages released with `release-unstarted` after a clean dry run. Never submitted |
| 5 | 10 × 5 h, jobs 2069428–2069440 | Ran 26 Sep 15:47–20:43. 9 COMPLETED, **2069428 FAILED after 2h35 (cause not yet read)**. All 10 attempts published (4,132–8,319 cells each; members 9 and 10: 8,264 and 6,794) |
| 6 | **20 × 5 h approved** (400 GPU-hours, budget 1,450 fingerprints per member-hour, cap 20, simultaneous starts, stale-quota exception) | Dispatch started on pin `5f16dca`, PID 3829560 on login node **jpbl-s01-04**, output `audits/llama-round6`. It published round-5 members 9 and 10, then entered the slow preparation that `a55f382` fixes. It was once suspended by Ctrl-Z and resumed with `fg`. **Whether it submitted is unknown at handoff** |

Registry before round 5: 73,838 Llama cells completed, 774 eligible packages
(236,541 cells). After round 5 roughly 160,000 remain; round 6 covers most of
it. Realized throughput is about 1,600 cells per node-hour.

## 5. Next actions, in order

1. **Round 6 status**: Llama page step 0 (on jpbl-s01-04 for the process view;
   the file checks work anywhere). If `SUBMITTED`, monitor with step 4. If the
   old process is still preparing and the Llama ledger has not been written for
   more than 2 minutes, Ctrl-C it (it is before reservation), rerun step 1
   (want `READY …a55f382…`), then step 3. Do not rerun the discover step: it
   would advance `next-round.txt` to 7.
2. Read `audits/llama-round5/llama-5hour-171f96ac7925-1/slurm-2069428.err`
   (`tail -40`) before trusting round 6's twenty jobs.
3. After round 6 ends: Llama page step 6 (publishes it, prints what is left);
   then size and get fresh approval for the last wave.
4. Diagnose the JUPITER home quota (`jutil user dataquota`, `du` of `~/.cache`
   etc.); do not delete anything unseen.
5. Axis analysis: the checker passed on the Qwen root (627 keywords). The Llama
   wave dataset lacks `local-only/population-registration-v1`; proposed
   `--registration DIR` for the report and checker, and counting empty-ranking
   pairs as excluded. Not started.
6. Optional: refresh `coordination/progress.json`; consider moving
   `coordination/operations/` receipts to prefix folders before they reach the
   same 10,000-file limit.

## 6. Operational lessons

- **Ctrl-Z suspends** a dispatcher (state `T`) and keeps its lock and any
  ledger stripe locks. Use `fg` to resume or Ctrl-C to stop; never Ctrl-Z.
- One dispatcher at a time: `$MEAS/expansion.lock`. A second launch fails with
  `BlockingIOError`; the Llama page now says so plainly.
- Processes are only visible on their own login node; file-based checks
  (journal, staged attempt directories, `submitted.json`) work from any node.
- `HF WRITE token (hidden):` waits for input; setup step 1 exports `HF_TOKEN`.
- Hub transfers from JUPITER ran at 0.2–4.5 MB/s with dropped connections;
  every call now retries up to five times.
- Allocations still need Valerian's explicit approval per shape; the pages hold
  commands only for approved shapes.
