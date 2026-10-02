# HoreKa saved progress and failure audit command

Valerian requested a full situation audit and the extent of work retained by
failed Qwen allocations. He clarified that "send the existing pieces" means
include saved results and logs in the audit, not upload outputs to Hugging Face.

Read the last three indexed handoffs first and rechecked their historical facts
against his new pasted evidence. At 2026-10-02T03:52:26Z, Qwen showed 893
allocations: 647 completed, 25 failed, 41 running, 180 pending. The forecast at
03:52:53Z additionally showed Gemma 5175056 pending with an estimated start of
2026-10-04T00:20+02:00. This supersedes the earlier absence of a live Gemma
request. Preserve that request; its forecast is not a guaranteed start. These
are user-provided observations, not live remote tool checks or scientific counts.

Inspected the existing bout execution/status code, generator checkpoint schema,
dataset audit, publication helper, forensic collector and striped ledger. Existing
bout `status` sums `completed_count` across overlapping primary/spill lists and
reused cells, so that sum is not distinct corpus completion. The ledger snapshot
acquires all stripe locks; the full forensic collector copies raw artifacts.
Neither is needed for this login-shell operational snapshot.

Added standard-library-only `analysis/scripts/audit_horeka_saved_progress.py`.
It accepts workspace and optional account arguments; imports no inference stack.
It joins accounting since 25 September with the live queue, reads CURRENT and
the saved bout configs/receipts/attempts, and reports per-job completed, remaining,
newly committed, reused, explicit failed IDs, primary/spill sizes, execution pin
and stop/error evidence. Missing/corrupt counts stay unknown. It falls back to
the saved bout result where a run manifest is absent, with provenance. Failed
allocations are grouped into partial, zero, all requested, or unknown saved
progress. Completed jobs with unresolved work or validation errors also appear.

The report includes all discovered SI-v4 preparations and receipts, per-pass
SQLite task states read in read-only mode, live Gemma job details, failed job
steps/accounting, filesystem byte/inode output, work quota and optional account
queue/home quota. Selected logs preserve up to 64 KiB each; full saved small JSON
metadata is captured with hashes, sizes and timestamps. Files changing during
capture are unavailable. Each external command has a 30-second timeout. Only a
new uniquely named `reviews/horeka-audit-*` directory is written, containing
`summary.txt`, `audit.json` and `audit.tar.gz`. Original runs and claims remain
untouched. No sender changes, retries, uploads or scheduler mutations exist.

Scope is explicit: this does not read/hash/copy the full generator dataset or
validate record references. Per-attempt checkpoint counts can lag current shared
state, overlap with other attempts, and include reuse. Remaining is not equivalent
to failed. Exact distinct remaining/lost-cell counts require later ledger/artifact
reconciliation; the audit does not claim those counts or fabricate zeros.

The complete self-contained Python heredoc is in the single Copy block of
`analysis/docs/horeka-full-audit.html`. Matching exports:
`/Users/valerianfourel/Hamburg/GEODML_Unified/horeka-full-audit.html` and
`/Users/valerianfourel/Downloads/horeka-full-audit.html`.
Paste-ready `.sh` exports sit beside these pages. No cluster checkout, download,
model, environment source or new allocation is needed to run the command.

Verification follows the previously read test-audit skill. Two CLI fixture cases
passed: accounting available/unavailable, stale failed accounting overridden by
live RUNNING, partial failed output, corrupt checkpoint, summary fallback, mixed
integer/mapping Gemma inventories, pending Gemma, unchanged original source/index
bytes and file inventory, and report/archive creation. Only scheduler/storage
subprocess boundaries are stubbed. Actual JSON/SQLite/report I/O is exercised.
The decoded page command matches the standalone source exactly, Bash syntax and
embedded Python AST pass, and workspace/Downloads exports match. Command SHA256:
`ee38dd53c1c897f40be9a34b03a2549ae9e46e048006ad239dc081a86f15a2f2`.

No remote operation, actual cluster audit, job or inference ran here. No push was
requested; commit locally. Next: Valerian pastes the page command in the existing
HoreKa login shell and returns the printed summary or audit archive. Interpret
returned failed-job progress and error evidence before planning any reconciliation
or retry. Existing three-hour Gemma approval is not authorization for another job.
