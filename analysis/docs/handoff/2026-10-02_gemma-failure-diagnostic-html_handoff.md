# Gemma failure diagnostic HTML

Valerian requested the preceding failure diagnostic as HTML. Read the last
three indexed handoffs. No new cluster evidence was provided; job 5175121's
reported finished_with_failures and 17 failed inference tasks remain the latest
pasted evidence. Current allocation state and exact failure causes are unknown.

Added `analysis/docs/horeka-gemma-failures.html` with the unchanged read-only
command from the preceding diagnostic handoff, a copy button, the reported
counts and instructions to return task errors. It distinguishes durable done
states from successful results and blocked sources from failed inference tasks.
Existing results, launch pages and scientific settings are unchanged.

Matching exports: workspace root and Downloads `horeka-gemma-failures.html`.
Verified exact command preservation, Bash/Python/JavaScript syntax, unique HTML
IDs, resolved ARIA references and byte-identical exports. The diagnostic's
SQLite behavior was already verified in the preceding turn. Browser verification
remains unavailable from earlier discovery; no visual or clipboard interaction
check is claimed. No cluster command, inference, retry or allocation executed.

Commit locally with index. Next: Valerian opens the HTML, copies the diagnostic
into an existing HoreKa login shell and returns its output.
