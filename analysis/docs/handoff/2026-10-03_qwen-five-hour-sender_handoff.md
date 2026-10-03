# Qwen five-hour recovery sender

Valerian requested one paste-ready shell block that reconciles the existing
Qwen cells, divides missing work into roughly five-hour bouts and sends the
jobs automatically. This explicitly authorizes five-hour GPU bouts for this
recovery sweep. It does not revive the original first pass's 50-running-job
or 295-submitted-job scheduling exceptions.

Read the three latest indexed handoffs. No live cluster query, quota check,
inference, allocation or HF publication ran here. Previous cluster counts and
timings remain historical. Fetched the current HoreKa batch, hardware and
filesystem documentation. The operator command performs live checks on HoreKa.

Implementation commit: `32eb54d`. Published executable SHA:
`76622b72dd4346669ccad9bbd10b8a98d27c5f5d`, on
`origin/codex/qwen-recovery-20261003`, confirmed by `git ls-remote`.
Only the implementation was cherry-picked onto the isolated release checkout
at `.worktrees/qwen-recovery-release`; unrelated active-checkout history was
not pushed.

## Behavior

Extended `analysis/scripts/horeka_qwen_recovery.py` with `send`,
`--walltime 05:00:00` and `--previous-recovery`. The default remains one hour.
The sender performs CPU preparation, submits each frozen GPU bout once, waits
for terminal accounting, then performs the final CPU reconciliation/coverage
audit. It has a durable seven-day deadline and polls every minute. Restarting
preserves that deadline, receipts and finite allocation count. It does not
create replacement waves or retry refused, ambiguous, failed or timed-out jobs.

Capacity and start-spacing delays now have a distinct `AdmissionDeferred`
exception in the existing admission owner. Storage, ownership, integrity,
network and uncertain-submission errors stop the sender. The existing Qwen
bout submission now times out after sixty seconds, retaining its durable
submission marker if the response is uncertain.

A workspace sender lock prevents duplicate new senders. The shorter operator
lock serializes submissions and CPU audits. A CPU audit may hold this lock
for minutes; the sender waits, including after a restart, rather than failing
or interfering. Existing immutable contexts can be read while that lock is held.

Original allocations must have terminal accounting before reconciliation.
The supplied command checks the earlier one-hour recovery directory too.
If that directory has live jobs, the sender waits. If it has unsubmitted frozen
work, failed preparation or uncertain receipts, the sender stops and names the
directory. It never silently converts that division to five hours.

Existing saved answers are reconciled before remaining-cell selection. Existing
model settings, task identities, seeds, ledger and inference executor remain
unchanged. Completed cells, terminal failures and HF-owned cells are excluded.
Every submission preserves the five-allocation cluster cap and ten-minute
observed-start gap, counting interactive allocations. Fresh quota and storage
checks still gate admission. HF ownership/publication changes stop overlapping
legacy admission. No live allocation is cancelled, extended or released.

## Resource estimate and bounds

The preparation audit measures throughput from successful terminal original
bouts. Division sizing uses the 90th-percentile seconds per newly committed,
unreused cell, ten minutes startup, five minutes drain and two minutes cleanup.
Each full GPU bout requests one node, four A100 GPUs, 32 CPUs, whole-node memory
and five hours: at most five node-hours and twenty GPU-hours. A backlog smaller
than one full bout gets a shorter allocation. The fixed bout count and total
GPU/node-hour budget are written to `preparation/ready.json` before submission,
alongside the measured primary-runtime range and cheaper alternative. Historical
30–53 seconds/cell is not substituted for current measured remaining work.

CPU preparation/finalization retain two allocations maximum, each one node,
four requested CPUs, 16 GiB and one hour in `cpuonly`, provisionally 5–45 minutes
per phase. These are requested resources; actual exclusive-node allocation and
charged CPU-hours must be read from Slurm accounting.

## Operator block and verification

The exact chat block is saved in
`analysis/docs/horeka-qwen-recovery-send-5h.sh`. It fetches the published branch,
uses a clean detached checkout at the exact executable SHA, and starts a finite
login-side sender with `nohup`. Environment and cached HF authentication are
inherited from the existing workspace setup; no token is embedded or requested.

Output: `$W/reviews/qwen-recovery-5h-20261003`.
Log: `sender.log`. Current state: `sender-status.json`.
Final coverage: `final/coverage.json`. Existing one-hour run guard:
`$W/reviews/qwen-recovery-20261003`.

55 focused tests passed in both the active and exact release checkouts:

```bash
/Users/valerianfourel/miniconda3/bin/python -m pytest -q \
  analysis/tests/test_horeka_qwen_recovery.py \
  analysis/tests/test_capture_agentic_scheduler_snapshot.py \
  analysis/tests/test_horeka_qwen_bouts.py \
  analysis/tests/test_reconcile_agentic_dataset.py \
  analysis/tests/test_shared_hour_scheduler_owners.py \
  analysis/tests/test_agentic_admission.py \
  analysis/tests/test_publish_qwen_results.py \
  analysis/tests/test_shared_hour_reservation.py
```

The added lifecycle proof runs real CPU reconciliation, division, submission
receipts and final coverage, with external Slurm/Hub/model-file boundaries
substituted. It exercises scheduling waits, timeout preservation and restart
without duplicate allocation. Other cases cover five-hour sizing, immutable
budgets, submission refusal/ambiguity, durable sender expiration, storage
failure and previous-division preservation. The operator-lock regression first
failed with `BlockingIOError`, then passed after the lock handling was fixed.
No scientific result is established by these tests. Bash/Zsh syntax checks and
`git diff --check` pass for the operator block and implementation.

Next action: Valerian pastes the supplied block into an already-open HoreKa
login shell. Submission and sender survival are not proof of completed cells.
Inspect its log and final coverage, which reports eligible missing cells,
terminal failures and blocked/unverified work separately. A sender deadline or
failed GPU bout can leave work; no extra sweep is inferred or launched.
