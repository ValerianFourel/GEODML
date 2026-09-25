# Three-hour/four-hour wave session handoff — 2026-09-25 (JUPITER, GEODML Experiment V2)

Operator: Valerian (runs every cluster command himself in an open JUPITER login
shell; agent prepares paste-ready chat blocks, never fabricates cluster output).
Branch: `threehour-relaunch-fix` → pushed to `origin/codex/pilot-continuation`.
All times below are cluster-local (CEST) on 2026-09-25 unless marked.

## 1. Live state at session end (last verified squeue ≈ 16:4x)

| Job(s) | What | State | Notes |
|---|---|---|---|
| 2014264 | Qwen r2 (relaunch of failed qwen-5 r1), 3 h | RUNNING since 14:05:32 on jpbo-073-48 | ends ≈ 17:05; receipt `$WQ/relaunch-2.json` |
| 2018545–2018566 (20 ids; 2018555/2018561 belong to other users) | **Qwen continuation round 2: 20 × 4 h = 320 GPU-hours**, names `geodml-qwen-4hour-{1..20}-c2` | 2018545 RUNNING on jpbo-104-39; 19 PD (Priority) | receipt `$R2/qwen/submitted.json`; guard `$MEAS/expansion-qwen-r2.json`; each job's 4 h clock starts at its own allocation start |
| 2017345–2017353 | **Llama round 2: 9 held members, 108 GPU-hours**, names `geodml-hours-llama-threehour-a8892c11e5d3-{1..9}` | PD (JobHeldUser) — **STILL HELD** | receipt `$R2/llama/queued.json`; release blocked twice by tooling bugs (§4); corrected release command in §5 **not yet confirmed run** |

Aggregate approved today: up to **30 concurrent allocations (120 GH200)** =
20 Qwen + 9 Llama + 1 r2, simultaneous starts, stale-quota exception, **no
automatic retries or replacement allocations**.

## 2. Approvals granted today (exact scope)

1. Qwen continuation round 2: 20 allocations × 4 h × 4 GH200 = **320 new
   GPU-hours** (80 node-hours); each job = four consecutive one-hour bouts
   against the frozen member-one ledger (untouched cells only, per-cell
   checkpoints, every hour boundary a valid stop point); concurrent with the
   Llama nine and r2 → 30-concurrent aggregate cap; simultaneous starts;
   stale-quota exception. Approval text is embedded verbatim in the dispatch
   command that ran.
2. Immediate release (not cancellation) of the nine held Llama members within
   the 30-concurrent aggregate — "released immediately along the 20 qwen jobs".
3. Earlier today (pre-checkpoint): the 9-job Llama held queue itself
   (108 GPU-hours, cap-10 shape) and the qwen-5 r2 unchanged retry.

Standing rules re-confirmed: fresh approval + estimate before ANY new
allocation shape; default wall-time 1 h and 5 concurrent unless explicitly
overridden; never cancel live jobs; git is the handoff boundary (no
cluster-only source edits).

## 3. Git pins (chronological, all pushed to codex/pilot-continuation)

| Pin | Content |
|---|---|
| `7551630eb2e62fbc937398102e4fecad4b98ad86` | localize_reference + relaunch-qwen mode |
| `9fdcc69cabe7dbeaa395a1d233f139ffb70fc97f` | round-2 relaunch accepts prior-round receipts (wave_jobs union) |
| `d05ac17bb4cca235983f12f8ca5d9c8895566499` | --round / --members / qwen continuation rounds |
| `dd842199849239f636db0746d31fe7e7f46bd23a` | held queue (--hold-queue) + release-queue poller |
| `3d6b566617cd355d8573ebedde2fa19c8df06383` | pre-gates running-only under hold-queue |
| `4b825c0f961ddd6f40f9d68ba86f6732de281530` | **4-hour wall-time parameterization, members ≤ 20, caps {5,10,20,30}, release-queue --once** — the pin checked out and used on the cluster today |
| `29f525772896736d3491efa97370916a68cfc780` | CLI: release-queue no longer demands --qwen-reference/--account/--partition (per-mode required map; releaser runs before the expansion lock) |
| `3cbc86aca62cdc4d07a8b45e50412c16d7f4b076` | test-audit batch (tests only, +34/−25): folded 4-hour staging replay into a table row; new main()-level CLI regression test. **Current tip.** |

Cluster checkouts live at `$BASE/src/threehour-wave-<pin>`. Only `4b825c0`
exists on the cluster; it is sufficient for every pending command (the CLI
relaxation just removes three now-optional flags; pending commands pass them
explicitly). Dispatchers never import tests, so `3cbc86a` changes nothing at runtime.

Test suite state: `analysis/tests/test_dispatch_threehour_wave.py` +
`analysis/tests/test_dispatch_llama_five.py` = **55 passed** via
`cd <checkout> && PYTHONPATH="$PWD" UV_CACHE_DIR=/tmp/geodml-uv-cache uv run --with pytest pytest -q analysis/tests/test_dispatch_threehour_wave.py analysis/tests/test_dispatch_llama_five.py`.

## 4. Incidents and fixes today

1. **Original releaser (PID 2411420) died instantly at 16:08** — argparse
   globally required `--qwen-reference/--account/--partition`, which the
   release-queue command never passed. Evidence preserved in
   `$R2/release/nohup.log` (usage error). Fixed twice: operationally by
   passing the three args; structurally in pin `29f5257` (per-mode required
   map) + regression test in `3cbc86a`. The old PID is gone (confirmed
   "poller 2411420 not running"); no receipts were flipped, nothing released.
2. **First release-now (--once) run failed CLOSED by design** —
   `blocked_unexpected_running` listing 2018545–66: the Qwen wave had already
   been dispatched, so its 20 jobs were unheld but absent from the command's
   allowed set (and cap 10 was stale under the approved 30 aggregate). Zero
   side effects; `$R2/release-now/` contains only the blocked log line; no
   `queue-released.json`, member receipts still `submitted_held` → the
   directory is safe to reuse.
3. Historical context (pre-checkpoint, all resolved): Qwen r0 mass failure
   (external frozen paths) → localize_reference; qwen-5 r1 2012327 vLLM CUDA
   OOM at ~9 min → approved unchanged retry r2 2014264 (running fine; if a
   4-hour job ever OOMs the same way, lowering gpu_memory_utilization/
   max_num_seqs is a scientific-settings change needing explicit approval).

**Known limitation (flagged, unimplemented):** `relaunch-qwen` cannot relaunch
a failed *continuation* member — its guard resolves `expansion-qwen.json`
(round 1) in `$MEAS`, which points at the original `$WQ` output, so a
`--output "$R2/qwen"` relaunch is refused as "originally dispatched"
mismatch. The correct recovery for a failed 4-hour member is a **fresh
continuation round** (`--round 3`, new output dir, live-waves including
`$R2/qwen/submitted.json`); the ledger guarantees only missing cells are
admitted. Extending relaunch-qwen to continuation waves is an optional future
milestone.

## 5. PENDING — the one command that still must run (then paste output)

Release the nine Llama jobs under the approved 30-cap (runs on the existing
`4b825c0` checkout; reuses `$R2/release-now` safely):

```bash
(
  set -eo pipefail
  source "$HOME/geodml-acl-arr-pilot.env"
  if ! type module >/dev/null 2>&1; then source /etc/profile; fi
  module load Stages/2026 GCC Python CUDA git
  source "${ACL_ARR_VENV:?}/bin/activate"
  export PYTHONDONTWRITEBYTECODE=1
  BASE=/e/fscratch/scifi/fourel1/geodml
  PIN=4b825c0f961ddd6f40f9d68ba86f6732de281530
  REPO="$BASE/src/threehour-wave-$PIN"
  test "$(git -C "$REPO" rev-parse HEAD)" = "$PIN"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  WQ="$BASE/audits/threehour-approved-wave/qwen"
  R2="$BASE/audits/threehour-wave-round2"

  python3 -u "$REPO/analysis/scripts/dispatch_threehour_wave.py" release-queue --once \
    --output "$R2/release-now" \
    --qwen-reference "$BASE/audits/qwen-threehour-wave-member1-6909506" \
    --account scifi --partition booster \
    --queue "$R2/llama/queued.json" \
    --live-wave "$WQ/relaunch-1.json" --live-wave "$WQ/relaunch-2.json" \
    --live-wave "$R2/qwen/submitted.json" \
    --since 2026-09-01 \
    --maximum-concurrent 30 --simultaneous-starts --start-gap-seconds 0 \
    --approval 'Valerian approved immediate release of the nine held Llama round-2 members (jobs 2017345-2017353; 108 GPU-hours approved at queue time) alongside the twenty-job four-hour Qwen continuation wave (320 GPU-hours) and Qwen r2 2014264: up to thirty concurrent allocations (120 GH200) for this finite aggregate. No automatic retries.'

  squeue --me --sort=i --format='%.10i %.60j %.10T %.20S %.10M %.8l %.16R'
)
```

Expected: 9 × `RELEASED_JOB=201734x`; `$R2/release-now/queue-released.json`
written; squeue shows the nine as PD (Priority), names
`geodml-hours-llama-threehour-a8892c11e5d3-N`, alongside 20 Qwen PD/R jobs
and 2014264 R. If it raises `release-now stopped...`, paste the reason —
fail-closed, zero side effects, rerun-safe.

## 6. Path map (cluster)

```
BASE=/e/fscratch/scifi/fourel1/geodml
WQ=$BASE/audits/threehour-approved-wave/qwen      # qwen-1..5; submitted.json (2011080-84, failed r0);
                                                  # relaunch-1.json (2012323-27); relaunch-2.json (2014264);
                                                  # rotated *-failed-<job> receipts under qwen-5
WL=$BASE/audits/threehour-approved-wave/llama     # Llama round 1 (2011639-43), site.json
R2=$BASE/audits/threehour-wave-round2             # llama/ (queued.json 2017345-53, site.json, 9 member dirs)
                                                  # qwen/ (submitted.json 20-job round 2, qwen-1..20)
                                                  # release/ (dead poller nohup.log), release-now/ (blocked log; reuse)
MEAS=$BASE/audits/qwen-threehour-wave-member1-6909506   # frozen member-1 reference; expansion-qwen*.json guards;
                                                  # dataset_root holds the shared atomic ledger (309,924 tasks)
Llama frozen runtime: $BASE/audits/llama-first-hour-0f7ef5f/runtime.json
Env file: $HOME/geodml-acl-arr-pilot.env    HF repo: ValerianFourel/geodml-experiment-v2-paper-private
Qwen-5 r1 OOM log (historical): $WQ/qwen-5/logs/qwen38-2012327-0.server.log
```

## 7. Monitoring / next actions after the release

1. Confirm the §5 paste (9 RELEASED_JOB lines + squeue).
2. r2 2014264 ends ≈ 17:05 → check `sacct -X -j 2014264 --format=JobID,State,Elapsed`;
   COMPLETED vs DEADLINE distinction; its results land in `$WQ/qwen-5` + shared ledger.
3. Also still unverified: final sacct states of r1 members 2012323–26 (ran ≈ 11:57–14:57).
4. 4-hour Qwen jobs end at own-start + 4 h; Llama nine at own-start + 3 h.
   PD (Priority) is normal — starts stagger as booster nodes free.
5. Next morning: measure throughput from receipts (cells committed per member
   per hour, from the ledger attempt dirs) to size future waves; queue total
   is 309,924 tasks, r0–r2 + round-2 have consumed an unmeasured share.
6. Future rounds: Qwen → `--round 3` fresh output dir, `--walltime` per fresh
   approval, live-waves must include every receipt whose jobs may still be
   live (`$R2/qwen/submitted.json` etc.). Llama → round 3 dispatch with
   `--llama-site "$R2/llama/site.json"` so prior attempts sync (ALREADY
   SYNCED / terminal-state reconciliation releases finished ownerships)
   before regrouping; expect ~10–20 min of registry commits (23.5 MB
   `coordination/hours.json` monolith, full-state per commit).
7. **Deferred milestone (ordered earlier, then deprioritized with "forget the
   scheduler"):** registry split / scheduler-file — small
   `coordination/scheduler.json` status + per-hour shards so commits cost
   O(change), lightweight wave-status, heavy audit only on explicit resync.
   Do not start without Valerian re-confirming.

## 8. Tooling notes for the next session

- Dispatcher modes: qwen (round 1 / continuation `--round N`), llama
  (HF shared-hours), relaunch-qwen (Slurm-FAILED members of the ORIGINAL wave
  only, see §4 limitation), release-queue (`--once` = single fail-loud pass;
  default = finite poller with `--queue-timeout-seconds` 21600).
- Key flags: `--walltime {03:00:00,04:00:00}` (04 only for qwen continuation
  rounds; llama/relaunch/round-1 refuse changes), `--members` 1–20,
  `--maximum-concurrent {5,10,20,30}`, `--hold-queue`, `--live-wave`
  (repeatable), `--queue` (repeatable), `--start-gap-seconds` (0 =
  simultaneous backfill), `--simultaneous-starts` + non-empty `--approval`
  required for ALL modes.
- Wall-time plumbing: member `preparation.json['approved_walltime']` is what
  the compute-node execute verifier compares against scontrol TimeLimit;
  continuation staging writes it from `--walltime`; member-1's frozen
  reference stays 03:00:00 forever.
- Standard env preamble for dispatch cells: §5 block minus the python line.
- Local (Mac): `test-audit` skill vendored at `.agents/skills/test-audit/`
  (from openclaw main; NOT published on ClawHub — `npx clawhub install`
  fails with "Skill not found"). npm cache `~/.npm` has root-owned files
  (npm suggests `sudo chown -R 501:20 ~/.npm` — not done). Harness
  web_search has no DEEPSEEK_API_KEY configured.
- Git writes from the agent need one-shot sandbox escalation (metadata lives
  at `~/Hamburg/geodml-mono/.git`, outside the workspace); push flow =
  commit → fetch → verify origin tip == HEAD~1 → push HEAD:codex/pilot-continuation.
