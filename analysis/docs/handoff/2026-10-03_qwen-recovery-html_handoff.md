# Qwen missing-cell recovery HTML

Valerian requested an HTML page to finalize missing Qwen jobs and relaunch
missed cells. Read the last three indexed handoffs and the prior Qwen operator
records. No live cluster state, quota, remaining-cell count or HF dataset
revision was checked. All previous pasted cluster counts remain historical.

## Delivered page and executable code

Created `analysis/docs/horeka-qwen-recovery.html` and six adjacent shell
downloads for preparation, next submission, status, logs, finalization and
publication. Matching exports are in the workspace root and Downloads.
Opened `/Users/valerianfourel/Downloads/horeka-qwen-recovery.html` in Safari.

Active implementation commit: `f5b138e`. Published executable commit:
`4a668d72bb2727518e9aa30ee08442d848fbc254` on
`origin/codex/qwen-recovery-20261003`, confirmed with `git ls-remote`.
An isolated release checkout at `.worktrees/qwen-recovery-release` starts
from the previously published `caf8a7d` and contains only this recovery
implementation on top. No unrelated active-checkout history was pushed.

Added `analysis/scripts/horeka_qwen_recovery.py`. It reuses existing dataset
reconciliation, verified inventories, bout splitting/execution, quota checks,
scheduler admission and HF publication. Extracted the existing division writer
in `horeka_qwen_bouts.py`; historical CLI behavior remains available. Recovery
submissions add a final admission check immediately before submission.

The scheduler capture now recognizes `horeka-boutNNNN-jobID` writers and can
include ordinary-named interactive allocations. A reproduced prior bug allowed
stale terminal accounting to describe an allocation still present in squeue.
Live evidence now overrides that terminal owner and completed-segment evidence.
Array inventory is expanded and scheduler subprocesses have a finite timeout.

## Recovery scope and budgets

This is one finite recovery sweep over the existing legacy HoreKa Qwen dataset,
not a replacement HF hour plan. The original division and sender are preserved.
Preparation requires all original bout submissions to have confirmed terminal
accounting. Live or unknown owners, unsent original bouts, unavailable throughput
and storage failures stop preparation. The observed ledger must have 256 stripes,
matching the existing Qwen executor. HF-registered overlapping work is refused.

Login operations fetch verified HF metadata. The two full reconciliation/audit
phases run through CPU Slurm jobs, with at most two allocations total: one node,
four CPUs, 16 GiB and one hour each, at most eight requested CPU-hours. The
provisional 5–45-minute estimate depends on artifact volume and verification
cache; it has not been measured for this cluster's current state.

The prepared GPU division has a fixed allocation count and total budget derived
from verified missing cells and successful terminal bouts with new, unreused
cells. Sizing uses the observed 90th-percentile rate, ten minutes startup,
five minutes admission drain and two minutes cleanup. Each job requests one
node, four A100s, 32 CPUs and whole-node memory for 20–60 minutes. Small complete
backlogs receive shorter allocations. At most one job is submitted per command.
No automatic requeue, replacement wave or scope expansion exists.

Every GPU submission checks fresh storage, all user allocations, the default
five-allocation limit, pending starts, ten-minute observed-start spacing,
previous submission receipts and HF overlap/newly published selected outcomes.
Confirmed sbatch refusals may be retried; uncertain submissions block further
admission. Original model settings, seeds and task identities are unchanged.

Valid saved responses are finalized before selecting missing cells. Completed
and terminal-failed cells are not resubmitted. Unknown ownership and invalid
evidence remain visible. Execution uses the existing shared ledger and actual
Slurm deadline, preserving completed results and active allocations.

Final reconciliation writes `final/coverage.json`, separating verified completion,
eligible missing cells, terminal failures and blocked/unverified evidence. A
failed or timed-out recovery bout can leave missing cells for a later explicitly
bounded action. The page does not label such a sweep complete. Publication uses
the existing private dataset and a hidden write-token prompt; it preserves
partial progress and does not establish scientific acceptance.

## Verification and next action

46 focused tests passed in both active and exact release checkouts. They cover
real dataset/ledger recovery, saved-result preservation, remaining-cell selection,
HF overlap rejection, active/unknown original owners, duplicate/ambiguous
submission evidence, admission, legacy bouts, reconciliation and publication.
The live-owner regression demonstrably fails against the previous implementation.

All six copied commands match their shell downloads; Bash, embedded Python and
JavaScript syntax pass; anchors and download links resolve. Browser verification
using installed Chrome and Playwright exercised all six clipboard buttons,
desktop 1280x900 and mobile 390x844, with no page errors or horizontal overflow.
Proof is in `output/playwright/qwen-recovery/` under the workspace root. The
Playwright CLI setup was slow, so the completed check used the installed library.
The setup process later exited normally; no browser/server test process remains.

HoreKa batch, hardware and filesystem documentation was fetched and checked.
No Slurm job, inference, HF publication or cluster shell command ran here.
Next: Valerian runs page Step 1 in a HoreKa login shell and returns the prepared
counts/budget or any blocking message. Continue with next-bout submissions only
when preparation succeeds and the fresh admission checks permit them.
