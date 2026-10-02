# Rebuilt SI-v4 recovery page

Valerian requested rewriting the HTML from scratch after pasting the original
9aa5d90 launch again, an admission-check refusal, and a second original launch
ending in FileExistsError for `20261001/attempts/job5175111`. This new error is
the intentional exclusive attempt-directory creation, not another source bug.
The old run.sh was still being executed. His later account snapshot was 208
jobs with CAP=294 already set. No HASH_REPAIR_VERIFIED output was supplied.
Allocation 5175111's remaining time and live state are unverified; the latest
paste still shows the hkn0402 prompt. Existing 5175056 status remains unknown.

Read the last three indexed handoffs. Traced the attempt guard in
horeka_nemotron.execute and admission guard in horeka_si_v4.check, and reused
the verified source-hash recovery from the previous turn. No scientific or
execution-helper code changed. Applied surgical-patch and test-audit guidance.

Rebuilt `analysis/docs/horeka-si-v4.html` as one three-step workflow:
1. Login shell: verify the published repair checkout and the existing job;
   prepare or verify/reuse the corrected, unattempted copy.
2. Compute shell for 5175111: validate receipt, exact corrected launcher,
   remaining time and fresh storage evidence, then execute the corrected copy.
3. Login shell: read current job/accounting, repair receipt, corrected summaries,
   bounded console log and per-pass task counts.

Removed allocating/admission/queue-cap blocks from this active page. The old
three-hour page remains in Git history at bfb76db. The former standalone source
`horeka-si-v4-existing-job.html` is now a simple link to the canonical page.
Updated matching main and former-entry-point exports in the workspace root and
Downloads. Added the distinctly named `horeka-si-v4-recovery.html` export in both
locations so Valerian can open a fresh tab. Other status and dataset pages are
unchanged. No unmatched export content required a backup during this update.

Preparation now reuses an already verified, unattempted corrected directory
without rewriting it, after checking configuration, source provenance, frozen
file hashes, manifests, launcher and semantic record preservation. Partial or
mismatched preparations stop explicitly. Any corrected attempt stops both setup
and launch with CORRECTED_RUN_ALREADY_ATTEMPTED and directs the user to step 3.
Original attempts and task state are preserved. No marker deletion or alternate
attempt name bypasses existing protections. The original Python bug is already
fixed in published pin 1bf21cb0e37374634be30a07bca9079ec18d6f87.

The page retains the job-5175111 binding, one-hour identity, 20-minute remaining
floor, storage checks and actual deadline. The recorded 20–50 minute estimate
and startup uncertainty remain explicit. No new job, retry allocation, time
extension, cancellation, cluster edit, model download or inference was executed.

Verification: exact updated setup/run programs passed
`/tmp/verify-si-v4-existing-job.py`. This reproduces the original source-hash
failure with the old freeze, then checks preserved original bytes, identical
task text, all three Coordinator imports, unchanged verified reuse, rejection
of a corrected duplicate attempt with its files preserved, wrong allocation,
insufficient time, changed receipt/configuration/launcher, prior task state,
and unsafe storage. Runtime/model/quota/scheduler boundaries use fixtures;
actual input, config and SQLite I/O runs. These are not scientific findings.

All 44 SI-v4/Gemma tests passed, including the retained integer-versus-mapping
inventory monitor test. All three copied shell blocks and embedded Python
programs parse; JavaScript syntax, IDs, navigation and ARIA references pass.
Exports match. Browser verification remains unavailable from the preceding
browser discovery, and no interactive rendering/copy proof is claimed.

Commit locally with this index entry. Next: Valerian opens the new recovery
export, starts at step 1, returns HASH_REPAIR_VERIFIED or its error, then uses
the existing compute shell only if the same job still has enough time. If a
corrected attempt already exists or the job ended, inspect step 3 output.
