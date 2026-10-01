# Qwen bout counts and queue target 294, 1 October 2026

Valerian requested counts for `geodml-qwen-bout-*`, then pasted the SI-v4
admission failure at the account queue threshold of 295 and requested a Qwen
refill target of 294. Updated `analysis/docs/horeka-si-v4.html` and its identical
workspace-root copy. Existing admission, allocation and execution commands
remain unchanged.

Step 7 counts only the user's Qwen bout job names. It combines `sacct -X` since
25 September with current `squeue`, deduplicating job IDs and letting the live
queue override accounting. It prints completed, failed, running and pending,
plus other states separately. Accounting failures produce unavailable output,
not misleading partial totals. Different allocation IDs count separately.

Step 1 updates existing shared-workspace `send-all-bouts.sh`, `send-salvo2.sh`
and `send-what-fits.sh` to literal `CAP=294`. This is necessary because the old
tmux loops explicitly supply `CAP=295` on every invocation; changing a default
alone would not work. The command validates all existing CAP assignments before
editing, checks Bash syntax, saves original-byte backups, preserves permissions
and replaces scripts atomically. It refuses unexpected assignments and symlinks,
and is idempotent. It starts no sender and submits or cancels no jobs. Approval
ranges, durations, scientific pins and existing receipts are unchanged.

The update applies on the next invocation of an existing sender, not a currently
running invocation. Existing 295-job queues must drain naturally. Other account
submissions can consume a slot, so repeat SI-v4 admission before allocation.
Old sender-creation pages can overwrite the modified scripts; the new page
explicitly says not to rerun them. No sender was restarted or extended here.

Local verification executed all three actual sender scripts with substituted
site paths and stubbed scheduler/submission boundaries. Before the patch, the
sender tries to fill slot 295. Afterward all three stop at 294 despite inherited
CAP=295 and admit exactly one job at 293. Verified backups, modes, idempotence,
unknown/symlink refusal, Qwen filtering, deduplication, missing/failed scheduler
output, and fail-closed queue checks. All ten HTML command blocks pass Bash
syntax; existing eight command blocks are unchanged. Browser navigation, actual
clipboard text for the two new blocks, mobile width and page-error checks pass.
Proof script: `/private/tmp/verify-qwen-counts-cap294.py`.

The latest pasted admission failure is evidence of a full account queue at that
check, not a current scheduler snapshot. No live SSH, cluster command, allocation
or inference ran in this turn. Next: Valerian pastes step 1 in a HoreKa login
shell, waits for natural room if needed, and repeats step 2 admission.
