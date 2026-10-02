# Update the Qwen dataset and inspect failures

Valerian requested an HTML page to update the dataset and check what failed.
This adds a publication workflow to the preceding audit-only request. Stated
the interpretation to him: publish verified saved Qwen outcomes to the existing
private `ValerianFourel/geodml-experiment-v2-paper-private` dataset. The HTML
prepares the commands; no publication or remote operation ran in this turn.

Read the last three indexed handoffs before work. No new cluster evidence was
supplied. The latest pasted snapshot remains 2026-10-02T03:52Z, Qwen 647
completed, 25 failed, 41 running and 180 pending; Gemma 5175056 was pending.
These allocation counts do not establish cell completion. Preserve that Gemma
request and all existing jobs. Do not treat its forecast as a guaranteed start.

Updated `analysis/docs/horeka-full-audit.html` and its matching workspace-root
and Downloads copies. The four copyable steps are:

1. Existing self-contained saved-progress/failure audit, byte-for-byte unchanged.
2. Create or verify an isolated publisher checkout at published commit
   `1bf21cb0e37374634be30a07bca9079ec18d6f87`, using the existing HoreKa runtime.
   Derive the dataset from the current Qwen division and check its contract.
3. Run the existing `publish_qwen_results.publish(..., apply=True)` once. Prompt
   for a write token locally with echo disabled, check the existing private
   destination through `HubStore`, prevent concurrent invocations from this page
   with a workspace file lock, and save reports in a unique
   `reviews/qwen-hf-update-*` directory. Preserve completed and terminal_failed
   outcomes separately. Validate returned bundle receipts and their state counts
   against a fixed remote revision after the existing publisher's remote hash
   verification. Save context, publish report, remote index and verification
   receipt, or a credential-redacted error. Return nonzero for blocked writers
   or verification errors.
4. Read the latest update directory and audit summary. Choose the newest attempt
   even when it failed before writing context, so an older success cannot hide
   a newer error. Missing reports stay explicitly unfinished.

The publisher and its dataset/ledger/transfer dependencies are unchanged between
the execution pin and the current checkout. No production implementation changed.
Publication metadata identifies the code pin as the publication helper, not the
generator execution commit. The page explains that existing indexed cells are
skipped, saved successful cells from failed allocations remain publishable,
terminal failures remain failures, and blocked writers are publication problems.
It distinguishes index totals, snapshot completion and full corpus completion.

The upload snapshots the ledger and therefore holds its locks during that read;
the page says active writers wait during this operation. It uploads only verified
terminal outcomes using existing sealed references. It does not seal unfinished
files, release claims, retry inference, alter the hour registry, allocate GPUs,
cancel jobs or update running checkouts. New worker results can appear after the
snapshot. Rerunning a finished/interrupted transfer reuses immutable objects;
do not launch a duplicate while a publisher remains active.

Applied the technical-writing guidance to the how-to and the existing HoreKa,
test-audit and unslop guidance. All four decoded shell blocks pass `bash -n` and
their Python programs parse. Anchors resolve and exports match. The unchanged
publisher and audit suites passed 5 tests. Additional proof executes the exact
copied publication program through the real publisher against `MemoryHub` and
actual temporary dataset/ledger files: successful and failed outcomes remain
distinct; a second run publishes nothing already indexed; a corrupted remote
index receipt fails and writes no success receipt; token contents are absent
from output/reports. The exact status program selects a newer attempt with only
an error record. Only remote storage and terminal credential input are replaced
at their boundaries. Proof script: `/tmp/verify-horeka-publish-page.py`.

No cluster execution, live token check, network upload, inference, allocation or
browser interaction is claimed. `git diff --check` passes. Commit this page and
handoff locally; no push needed because the execution helper is already published.
Next: Valerian runs the page in order and returns the failed-job audit and update
verification receipt. Inspect returned failures before any reconciliation/retry.
