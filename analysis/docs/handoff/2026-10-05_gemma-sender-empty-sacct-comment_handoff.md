# Gemma v4 sender stopped on empty sacct comments; 57 bouts FAILED

Read the last three handoffs. Evidence below was pasted by Valerian; not live.
Run root `$W/reviews/gemma-si-v4-llama-reuse-5h-20261004`, plan pin 493d06e.

Pasted status: 57 bouts FAILED, 50 RUNNING, 93 PENDING (all 200 attempt-1
bouts). The sampled failed bouts ran 3:06–3:54 and ended with ExitCode 2:0.
Their .err files have no Python exception, only `srun: ... exit code 2`. The
judge returns 2 whenever its report status is not `finished`, i.e.
`incomplete` or `finished_with_failures`. The cause has not been established yet;
the per-shard report summaries were requested.

Sender stopped at 2026-10-04T19:11:56Z with "known allocation no longer matches
its submission comment". sacct shows an empty Comment for finished bouts:
HoreKa accounting does not store job comments. identify() compared the receipt's
job ID against that empty comment once a job left squeue. status.json still
says all 200 shards are `live`; it is stale. No resubmission happened since.

Fix aead85c (branch codex/gemma-v4-sender-empty-sacct-comment-20261005):
a known job ID with an empty accounting comment is accepted; a different
comment, or a live squeue mismatch, still stops. checked_plan now lets a newer
clean committed sender supervise a plan pinned to an older checkout: it verifies
the pinned checkout is clean and at the plan commit. Bouts still run the pinned
493d06e code. No scientific setting, budget (400 bouts), max_inflight (200) or
plan file changed. Tests: new regressions fail on the old code; 61 Gemma v4,
sender and prequeue tests pass.

Queue: the authorized ceiling is 200 in flight and 400 bouts total. The
restarted sender refills only eligible shards (terminal with pending work and
budget left). Shards whose report is `finished_with_failures` are exhausted and
are not retried. Going beyond 200 in flight needs an explicit plan change.

Next: Valerian runs the shard-report diagnostic, then restarts the sender from
a checkout of aead85c in tmux `gemma-v4-bouts`. Codex/Claude ran no cluster
commands.

## Update: the whole llama plan is done (pasted, 2026-10-05 ~14:50 UTC)

squeue is empty. The read-only count over all 200 shard databases shows every
cell closed: 312,052 of 312,052, 0 missing, all 200 shards finished. Tasks:
answer_map 208,002 done; source_dependency 1,004,308 done and 99,816 blocked;
fulfilment 208,132 not_requested (source-importance-only run). 27,074 cells
include a blocked task. A source task is blocked when its parent answer map is
not eligible, failed or quarantined. The per-status breakdown has not been
returned yet. The Slurm "FAILED" bouts were shards that finished early with
judge status other than `finished`, which exits 2. 200 of 400 bouts were used.

The fixed sender (aead85c) restarted at 14:50 UTC on hkn1990 and is reconciling
terminal bouts. With no pending work it should end as finished or
finished_with_failures without new submissions. Semantic acceptance of the
Gemma judgments remains not established. Next: return the blocked-status
breakdown and the sender's final status.

## Update: blocked tasks classified (pasted, 14:54 UTC)

Answer maps (208,002): 194,178 ok and eligible; 11,958 ok with
global_absence_only; 21 no_substantive_content; 10 map_unusable; 1,835 failed
(ok=0). Blocked source tasks (99,816): global_absence_only 85,894,
map_failed 9,804, map_quarantined 3,946, no_substantive_content 110,
map_unusable 62. Only the 1,835 failed maps and their 9,804 source tasks are
inference failures; the rest are protocol outcomes. The plan never retries known
failures, so a targeted retry is new scope and needs explicit approval and a
small code path. Failure reasons were requested first. Sender was still
reconciling (shard-0027 at 14:54 UTC).

## Update: failed maps are output-validation failures (pasted)

All 1,835 failed answer maps, spread over all 200 shards, are JudgeOutputError:
mostly "exclusion overlaps claim or exclusion", some "map leaves answer words
uncovered". The judge runs at temperature zero with task-derived seeds; the
frozen tasks carry the v4 corrective-retry contract (si-map-corrective-retry-v4-r2)
and the deterministic overlap repair. An identical rerun is expected to
reproduce the same outputs, so no retry was recommended. A different
correction protocol would change scientific settings and is Valerian's call.
Failure rate: 1,835 of 208,002 maps. Attempt counts on failures not yet checked.

Confirmed (pasted): failed maps by validation attempts: 2 attempts 1,831,
1 attempt 3, none 1. The corrective retry ran for nearly all failures; they are
final under the frozen protocol. The 4 maps without a second attempt are not
diagnosed and are too few to justify an allocation. Sender at shard-0090 at
15:05 UTC, still reconciling. No further Gemma llama launches are needed.

## Update: Qwen pass launcher; failed-map relaunch declined (2026-10-06)

squeue empty. Valerian asked to relaunch the failed maps and to start Gemma on
the Qwen answers saved so far. Relaunching the 1,835 failed maps was not built:
they already had the corrective retry at temperature zero, the protocol has no
retry path for terminal failures, and a third attempt is a protocol change for
Valerian to decide.

Added `analysis/docs/horeka-gemma-v4-qwen.sh`: `start --source QWEN:qwen38
--prepare-on-cpu` with the llama run's account and exclusions, in tmux
`gemma-v4-qwen`. The HF write token is typed hidden inside the session (needed
for the new reservation commit), never on disk or argv. It refuses to start
while the `gemma-v4-bouts` session exists, because one sender lock is shared per
workspace. The freeze takes the Qwen answers complete at that moment; later
completions need a later pass. Preparation is one cpuonly 4 CPUs/32G
allocation with a four-hour limit, above the one-hour default; running the
command is Valerian's approval. Llama measured about 3 h freeze + 13 min
partition for 312k cells. Inference uses the same frozen settings, 200 in flight,
five-hour bouts and per-shard ceilings from the plan. This is within the
original whole-dataset Gemma authorization (qwen38 + llama4).
Stand-in shell test passed (command, session, no token on disk).

## Update: Qwen pass submitted (pasted, 2026-10-05 15:25–15:35 UTC)

Llama sender ended `finished_with_failures`, no extra submissions. Qwen sender
started 15:25:43 UTC in tmux `gemma-v4-qwen` on hkn1990 (PID 50912) from
469b786, root `$W/reviews/gemma-si-v4-qwen-cpu-5h-20261006`, Qwen source
`$W/shared-hours/dataset`, account hk-project-p0026831, same exclusions as llama.
Startup checks took about 7 minutes. sbatch rc 0: preparation job 5183428
(cpuonly, 4:00:00) PENDING (Priority). Next: confirm it runs, writes plan.json,
and that the sender reserves on HF and submits bouts.

## Update: queue blocked; move preparation to dev_cpuonly (pasted, 2026-10-06)

Job 5183428 (cpuonly, 4 h) estimated start 2026-10-12T03:00, priority from age
only; it was the only pending cpuonly job. Test-only probes: cpuonly 1 h and
4 h start 2026-10-10 19:18; accelerated 5 h 2026-10-10 20:12; no maintenance
reservation. dev_cpuonly (MaxTime 04:00:00, 12 nodes) would start a 1 h job
2026-10-06 11:55; dev_accelerated (1 h max) 14:49. CPU preparation checks only
that it runs inside a Slurm job, not its partition, so moving the pending job
with `scontrol update JobId=5183428 Partition=dev_cpuonly` keeps the same job ID
for the sender. preparation-submission.json still says cpuonly; the actual
partition is in sacct. The four-hour limit is unchanged and has no extra margin:
llama needed about 3 h 15 min. Judging bouts still face the accelerated queue
(estimated 10 October) after the plan exists.

## Update: fire-all script (2026-10-06)

Valerian asked for one script that sends all missing work now. Added
`analysis/docs/horeka-fire-all.sh` (b8643ef). Rerunnable, starts only missing
senders: (1) Gemma-on-Qwen sender, restarted on the same root if its tmux is gone;
(2) second Qwen recovery sweep at 405b408, `--walltime 05:00:00 --gpu-all-at-once
--previous-recovery qwen-recovery-20261003`, root `$W/reviews/qwen-recovery-20261006`,
tmux qwen-recovery-2; (3) tmux dev-mover moving our pending cpuonly preparation
jobs to dev_cpuonly every 2 min for 6 h, logged to `$W/reviews/fire-all.log`.
Syntax checked only; not run on the cluster by Claude. The Qwen sweep stops with
an error if the 2026-10-03 sweep is unfinished. It covers ~456 cells in the
current division; ~1,000 missing Qwen cells outside it still have no route.
GPU bouts still face the accelerated queue (estimated 2026-10-10).

## Update: Qwen sweep blocked by the start-gap gate (pasted, 2026-10-06 09:22 UTC)

`$W/reviews/qwen-recovery-20261006` already existed (pin 405b408, previous
`qwen-recovery-5h-20261003`, 5 h); its sender has polled since 2026-10-05 15:31 UTC
with "another released allocation has not started". admit() defers while any
of the user's jobs is PENDING and not held. It was blocked by Gemma prep 5183428,
now running, and is now blocked by 5184545 geodml-intent-stages (cpuonly) and 5184546
geodml-gemma-si-v4 (dev_accelerated, QOSMaxJobsPerUserLimit), neither from this work.
No Qwen preparation or GPU bout has been submitted. Restarted in tmux qwen-recovery-2
with the saved settings. fire-all script previous path corrected to
qwen-recovery-5h-20261003. Holding the two unrelated jobs would unblock admission;
Valerian's decision.

## Update: direct Qwen division without the recovery check (2026-10-06)

Valerian cancelled 5184572 (never-started Qwen recovery preparation) and 5184545
(intent-stages), held 5184546, and asked twice to send the missing Qwen cells as
GPU jobs now without the recovery CPU check. Given route: stop tmux qwen-recovery-2,
then `horeka_qwen_bouts.py divide` (pin 405b408) on the login host into a new
`$W/qwen-bouts/division-direct-20261006`, using dataset, plan, seconds-per-cell
and startup from the CURRENT division, then `submit` all bouts (05:00:00) at once.
No admission gate applies (no recovery.json in its parent). It skips Slurm
reconciliation of writers that died: their cells are in the division, but the
ledger may still refuse them, so they can stay missing. The inventory and HF
read run on the login node, against the usual rule, at Valerian's request.
qwen-recovery-20261006 is left as is (preparation renamed only if Valerian
ran that block).

Direct Qwen division result (pasted): `$W/qwen-bouts/division-direct-20261006`,
registered 312,096 Qwen cells, 310,623 published on HF, 402 remaining and runnable,
2 bouts (328 primary cells each, 52.5 s/cell measured in division-20260927-1813),
10 node-hours. Submitted 5184598 and 5184599 (accelerated, 5 h), PENDING Priority.
The other ~1,071 unpublished cells were excluded by the ledger as completed-unpublished,
claimed/running by ended writers, saved or terminal failures; they need publication
or Slurm reconciliation, not new inference. Gemma Qwen prep 5183428 at 52 min.

## Update: Qwen Gemma plan and llama top-up pass (2026-10-06)

Qwen Gemma plan (pasted): 288,531 cells_ok, 193 shards, max 386 bouts; 26 cells
share a prior map, 40 excluded diagnostic cells; 127,715 unique answer maps,
720,973 source tasks. 193 bouts submitted (fewer than 200 because the partition
targets at least 1,500 cells per shard).

Valerian asked to finish llama Gemma coverage. Added
`analysis/docs/horeka-gemma-v4-topup.sh` (any model, one or more `--exclude-inputs`),
for a llama pass excluding the original diagnostics and the llama run's `frozen/`
(protocol + file hashes verified by `exclusions()`), so none of the 312,052 judged
cells repeat. `start` now waits (600 s polls, up to the plan deadline) when another
Gemma sender holds the workspace lock, via `SenderBusy`, instead of failing; the
one-sender-per-workspace rule and the 200 in-flight cap are unchanged. The top-up's
bouts therefore go out only after the Qwen Gemma sender ends. Its freeze re-reads all
llama answers (about 2–3 h on dev_cpuonly); whether new llama answers exist is
unknown until then. The 1,835 failed llama maps are not retried (protocol decision).
Tests: 62 Gemma v4/sender/prequeue pass, including the new wait test; stand-in shell test.

## Update: labelled map recovery (2026-10-07, Valerian's decision)

Valerian asked to retry the failed Gemma maps. Added a labelled recovery path; the
original run is never changed. `select_gemma_v4_failed_maps.py` lists the cells whose
answer map has ok = false (llama: 1,835 maps). `start` accepts `--cells` (exact
freeze selection), `--map-validation-attempts N` (2–6, judge config
`map_validation_attempts`, recorded in the plan and the execution hash) and
`--recovery-of PLAN_ID` (the HF reservation may re-claim only that plan's maps).
Attempts beyond two use new seeds with corrective feedback (existing loop). A recovery
with at most 10 shards uses its own sender lock, so it runs beside the main Gemma
sender (193 + at most 10 jobs, within the 200 authorization). Launcher:
`analysis/docs/horeka-gemma-v4-map-recovery.sh`. Report recovered maps separately as
"recovered with up to N corrective attempts". 119 tests pass across the Gemma v4,
sender, prequeue, SI-v4, publish and selector suites. Not yet run on HoreKa.

## Update: 27 Qwen Gemma bouts failed at startup (pasted, 2026-10-07 16:52–17:17 UTC)

All failed in about 35 s, exit 1, in the startup quota check: `mmlsquota -u uhh_bbf7367`
returned "no such user" (status 22) on those compute nodes; a transient user-lookup
outage, no judging done. Later bouts run normally (24 running at 15:17 UTC). The sender
had marked 24 shards blocked. Sender change: a terminal allocation that FAILED (or
NODE_FAIL/BOOT_FAIL) within 300 s with no results database is eligible again, within the
shard's existing allocation budget (never beyond it). Restart only the login-side sender
from the new commit; bouts keep their pinned code 469b786.

## Correction: one broken node, not a transient outage (pasted, 2026-10-07)

All 98 startup failures since 12:30 ran on hkn0515 (mmlsquota "no such user"): a node
that cannot resolve the user. It frees within ~35 s, so pending bouts keep landing on
it. Sender now adds `--exclude=<list>` from `$W/control/gemma-exclude-nodes` at every
submission. Pending bouts get `scontrol update ExcNodeList=hkn0515`. The pinned bout
code (469b786) refuses a shard's third submission (bind_submission budget), so shards
whose two attempts both died on hkn0515 need a labelled re-run of their cells.
Report hkn0515 to HoreKa support.

## Update: per-run senders under one global ceiling (2026-10-08)

Valerian wants the JUPITER Qwen answers judged without waiting days for the main Qwen
Gemma sender. Each run now has its own sender lock (`gemma-v4-sender-<plan_id>.lock`;
plans without plan_id keep the old shared lock). Before every submission a sender takes
`control/gemma-v4-submit.lock`, counts all `geodml-gemma-v4-bout` jobs in squeue and
submits at most 200 minus that count. So the Qwen top-up and the llama map recovery fill
slots as main-run bouts finish, never exceeding Valerian's 200. The small-recovery lock
exception is removed. 123 tests pass (Gemma v4, sender, prequeue, SI-v4, selector,
publish).

## Update: Qwen top-up submitting (pasted, 2026-10-07 17:01 UTC)

Top-up plan: 23,387 cells, 16 shards (JUPITER and late Qwen answers), preparation job
5186411 on dev_cpuonly. Relaunching through `start` with newer code was refused (pinned
preparation), as designed; the sender alone was restarted from d621fee. It submitted
7 bouts (attempt 1) under the shared 200 ceiling; 9 shards eligible, waiting for room.
Gemma bouts in queue: 199. Main + top-up plans cover 311,918 of 312,042 generated Qwen
cells; the rest are excluded diagnostics.
