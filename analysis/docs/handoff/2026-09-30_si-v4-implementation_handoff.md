# SI-v4 implementation, 30 September 2026

Valerian authorized implementation now, retaining SI-v3 and allowing up to three
times the warm GPU-hours per completed eligible cell. Mapping and retries count;
J1 is separate. This is development authorization, not approval for a cluster
allocation or production promotion.

## Code state

Active checkout `.worktrees/threehour-relaunch-fix`, branch
`threehour-relaunch-fix`, upstream `origin/codex/pilot-continuation`. Prior HEAD
`1fcb5fd` contains the Gemma audit. This handoff is committed with the SI-v4
implementation. No push or remote execution was performed.

Operator/protocol guide: [source-importance-v4.md](../source-importance-v4.md).

- New `source_importance_v4.py` owns exact prompts, schemas, original-span answer
  maps, independent source judgments, validators, map hashes and task identities.
  Six v3 anchors are preserved, including grade 5's most essential content.
- Existing preparer defaults to SI-v3; `--protocol si-v4` freezes map dependencies
  and source tasks. Exact cell selection fails on missing/duplicate selections.
  Original/recovered answers, serialized final requests and provenance are retained.
- New `run_source_importance_judge.py` uses an existing authenticated server,
  disk-backed queues, the existing ledger and sealed records. It supports bounded
  retries, checkpoint/resume, durable-response recovery and fixed-map repeats.
  Map issues quarantine dependent grades. Failed judgments never become zero.
- New constructed-diagnostics and comparison scripts support separate development
  and held-out controls, frozen reference ranges, repeats, cluster uncertainty and
  cost reporting. Configurations pin the Gemma candidate and acceptance design.
- Original `source_importance.py` remains unchanged. J1, generators, frozen
  evidence and historical results are unchanged. Mask-v2 and the metric correction
  use separate versions. V4 failures never silently fall back to v3.

## Verification

Local Python: `/Users/valerianfourel/miniconda3/bin/python`.

The broader relevant suite passed 175 tests and five subtests, with one optional
JSON Schema test skipped because `jsonschema` is unavailable. Its one sandbox-
blocked loopback security test passed on an approved escalated rerun, for 176
passing tests across that verification. The suite covered SI-v3/v4 preparation,
transcripts, Gemma preparation/export, inference security, allocation budgets
and context budgets.

After the final dependency-validation/helper changes, the focused v4 and task-
preparation suite passed 74 tests. This includes four added exact-selection
cases covering successful selection, unavailable cells, duplicate IDs and empty
selection. It also exercises map coverage, partial support, N/A, identities,
failure/resume behavior, map quarantine, context admission and comparisons.

`git diff --check` passed. Runner/diagnostic help entrypoints were exercised.
No ruff or optional JSON Schema validation is claimed. No live Gemma/vLLM/
xgrammar inference or GPU benchmark was performed. CPU fixtures establish
engineering behavior only.

## Scientific and operational limits

Historical cluster facts come from the verified Gemma archive and prior handoffs.
They were not rechecked against live schedulers in this local-only milestone.
No cluster command, model download, generator rerun or production mutation ran.

The 40 audited Gemma cells are development evidence. Fresh selection, independent
map/support review and actual 3x cost compliance remain unmeasured. A source-blind
map may reduce inconsistent centrality judgments but cannot establish semantic
correctness. Reference-model agreement is not human gold.

The client verifies local template bytes and context budgets; pinned server
receipts must establish loaded weight revision, runtime and hardware. Refresh
quota evidence after model loading. An unresolved source provenance check remains
unresolved; interpretation is attribution to the supplied record.

## Next milestone

Push this commit through the usual operator workflow, then freeze matched v3/v4
development inputs and constructed controls from a clean pinned checkout. Review
actual unique task counts and estimate startup, runtime, GPU-hours and wall-time
before requesting allocation approval. Do not launch based on the 3x ceiling.
Use the same judge configuration for the architecture comparison. Fresh
acceptance follows development; production promotion requires semantic and cost
evidence. Rollback selects retained v3 artifacts and protocol explicitly.
