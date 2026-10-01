# JUPITER completion and publication verification page, 1 October 2026

Valerian requested an HTML page to verify that JUPITER work is finished, pushed
and ready. This turn prepared checks only. No JUPITER SSH command, allocation,
upload, cancellation, registry mutation or deletion ran.

## Files and pins

Verifier commit `adb2586e5f8566437f9ee0916ebf82bdb3ebb15b` adds
`analysis/scripts/verify_jupiter_exports.py` and focused tests. The subsequent
documentation commit adds this handoff and
[jupiter-verify-ready.html](../jupiter-verify-ready.html). Identical convenient
copy is at workspace root `jupiter-verify-ready.html`.

The five copy blocks prepare the pinned verifier in project storage, inspect
Slurm/local processes/scheduled entries/Git, check local Llama sync receipts and
the current Hub registry plus Qwen publisher dry run, verify four export scopes,
and read the latest transfer logs. Existing HF authentication is reused or a
hidden read-token prompt is used locally. No token is requested in chat.

Reports and metadata cache use `/e/project1/scifi/$USER/geodml/jupiter-verification`.
Code checks fetch/prune origin tracking refs and inspect dirty/unpushed work;
they do not reset checkouts or push cluster changes. The preparation reuses the
known sweep environment and stops if that file is missing. It does not invent
replacement paths after scratch deletion.

## Verification semantics

The export helper pins each repository's current revision. General-archive
checks compare saved unit receipts with remote manifests, verify archive part
sizes/hashes and the archived plan/index, and require a matching COMPLETE
receipt. GeoAxis checks use saved MANIFEST.tsv plus README/manifest files,
remote LFS SHA-256 and local Git-blob comparisons. Missing files or unavailable
hash evidence remain attention/unknown. No empty manifest can pass.

Activation verification inventories all local regular files except `.cache`,
including `t7_chunks_full`, and compares current remote names and sizes.
Symlinks/unreadable paths do not silently pass. This deliberately avoids reading
551 GB of activation contents on a login host. It reports `inventory_match`,
never content verification or a successful full restore. Full report always
states that all-content verification has not been established.

Historical facts are displayed only as comparison points: 310,374 completed
Llama cells, six accepted terminal failures, 24,218 published JUPITER Qwen cells,
and completed GeoAxis uploads. The general archive and activation upload needed
confirmation in the prior JUPITER handoff. No current completion claim is made.

The page distinguishes Llama accounted-for status from all-successful status.
Qwen dry run covers terminal outcomes with valid sealed references, not a fresh
whole-corpus audit. Local process/tmux checks cannot inspect other login nodes.
Archived filtered scopes do not establish that every arbitrary JUPITER file was
selected. None of the checks authorizes cleanup or transfer of plan ownership.

## Tests and next action

14 tests passed across `test_verify_jupiter_exports.py` and
`test_archive_jupiter_to_hub.py`. Tests include absent/corrupt/unverifiable files,
Git-blob evidence, empty manifests, missing full-scope activations and local
completion receipts whose remote archive part is absent. All five Bash blocks,
four embedded Python programs and copy-button JavaScript parse. Page copies
match and `git diff --check` passes.

Push helper/page to `origin/codex/pilot-continuation`, open the root HTML, and have
Valerian run it from an already-open JUPITER login shell. Inspect returned
CODE_STATE, LLAMA_COUNTS, LLAMA_LOCAL_RECEIPTS, QWEN_PUBLICATION_DRY_RUN and export
results before making a readiness conclusion or proposing repairs.
