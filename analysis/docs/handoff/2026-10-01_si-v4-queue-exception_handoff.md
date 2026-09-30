# SI-v4 scoped queue exception, 1 October 2026

Valerian pasted successful preparation on HoreKa hkn1993, followed by admission
failure because at least five allocations were active. No allocation was
submitted by those commands. The frozen inventory is 40 corpus cells, 214
source tasks, 40 maps and 40 J1 tasks, plus 48 constructed cases deduplicated to
54 tasks. Preparation passed cached model, runtime, template and schema checks.
These facts are pasted evidence, not an independent live scheduler inspection.

Valerian then explicitly approved the proposed one-run scheduling exception to
allow the one-hour v4 test alongside the existing queue, retaining quota, account
queue limit and ten-minute observed-start checks. No cancellation, hold, retry
allocation or change to other jobs is authorized.

Admission helper pin `f256b68809628998e7ee29b9d022b03591db1b3f` adds the explicit
`--approved-existing-queue-exception` flag. It waives five-active/no-pending guards
only for `$W/reviews/gemma-si-v4-development-20261001`, with the original execution
pin `9aa5d90e0b310f3eea587a60787727787ab92757`, Gemma model/revision and one-hour
configuration. Existing attempts, duplicate queued/live v4 jobs, stale scheduler
evidence, recent starts, 295 account jobs and unsafe storage still block. Default
admission without the flag is unchanged. Admission receipt records the approval,
configuration hash and helper file hash.

The HTML now begins with preparation-complete status and step 2. That block
fetches the new helper into a separate worktree and rechecks the existing run.
It does not repeat preparation or modify its config, inputs or run.sh. Subsequent
salloc/compute/run/results blocks use the original execution checkout. Tracked
page is `analysis/docs/horeka-si-v4.html`, with an identical workspace-root copy.

19 focused tests passed. The new public check tests exercise 28 active plus
pending fixture jobs, scoped approval recording, and each retained block. All
five remaining HTML Bash blocks pass `bash -n`; page copies match and whitespace
checks pass. No cluster command or GPU inference was executed by Codex.

Push the helper and updated page to `origin/codex/pilot-continuation`. Valerian
starts at updated step 2, then runs the separate approved salloc only after CHECK
PASSED. A recent-start block still means waiting on the login host and repeating
only admission. Preserve all existing allocations and prepared artifacts.
