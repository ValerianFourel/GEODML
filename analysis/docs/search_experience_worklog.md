# Search experience implementation worklog

Baseline: `ce86d7e46a0fbaf5163fe8324f34b3114153f2b2`.
Scope: a separately versioned, captured-evidence pilot. No production launch,
no allocation changes, no replacement of the frozen ACL ARR protocol.

## Workflow

- [x] Read Principles and actual Poteto runtime instructions.
- [x] Phase A: Frame. Trace retrieval and independent generation/judging.
- [x] Phase B: Design workflow. Ground, sketch two independent candidates,
  cross-judge, pick and graft. Agree using the requested implementation scope.
- [x] Phase C: Run loop. Implement the smallest complete pilot adapter and tests.
- [x] Phase D: Audit trail. Record provenance, limitations and decisions.
- [x] Phase E: Verify and hand back. Tests, independent Interrogate review,
  reproducible commands and explicit unavailable measurements.

## Throughput checkpoint

Blocking first steps: establish frozen input contracts and select the versioned
extension boundary. Independent workstreams: two read-only execution traces,
two isolated architecture candidates, official documentation review.
Shared mutable state: none in exploration. One owner per implementation file.
Smallest safe decomposition: contract and replay implementation separate from
documentation and independent review. No shared output directories between runs.

## Evidence and decisions

The HOW explorers found keyword-only capture, offline historical HTML selection,
strict citation-index validation and a judge that cannot represent zero support.
No actual prompt/capture/page records or model snapshots exist locally.
Tests may use explicitly synthetic records, never present them as search results.

Official source consulted 2026-09-07:
https://help.openai.com/en/articles/9237897-chatgpt-search
It documents targeted query rewriting, possible follow-up searches, and source
links. It does not publish internal ranking algorithms or prompts.

Poteto sticky activation and distinct served model profiles are unverified.
Independent agents use their inherited model configuration. Their reports are
independent reviews, not evidence of model diversity.

## Implemented files and verification

New code: `analysis/interpretability/pipeline/search_experience.py` and
`analysis/scripts/run_search_experience.py`. New tests:
`analysis/tests/test_search_experience.py` and
`analysis/tests/test_run_search_experience.py`. No legacy source files changed.

The end-to-end synthetic test has three cases, 11 documents per case and all
three conditions. It validates 18 primary requests and nine judgments. The long
case retains 12,000 characters per document. Every ranking contains ten valid
IDs. The insufficient-evidence case has empty claims and empty assessments.
These are injected responses, not measured LLM behavior.

Focused verification: 19 contract, runtime and legacy experiment tests passed.
Adjacent runner reliability and pipeline CLI verification: 13 tests passed.
CLI help also runs. Python's pytest package is absent; tests use unittest.

## Independent review verdict

Intent: add a versioned captured-evidence pilot without changing legacy
treatments, ranking requests or analysis eligibility.

Two inherited reviewer lanes found no remaining blockers after fixes.
Acted on: exhausted capture iteration, mutable primary-manifest judge dependency,
exact model-configuration lookup, pre-client restart validation, source hashes,
explicit synthetic provenance and the complete three-case integration test.
The human packet manifest now retains synthetic and fake-backend flags.

Dismissed: rejecting later duplicate URL positions would contradict the
first-occurrence deduplication policy. Later search hits remain in the trace;
the URL audit now names selected and excluded URLs and duplicate counts.
Comment review removed one descriptive module docstring. Public API contract
docstrings remain. No suppression or constraint encoding was required.

## Handoff

This milestone is prepared for GitHub deployment on the stated baseline.
Unrelated pre-existing untracked files were preserved. The command page is
`/tmp/geodml-jupiter-commands.html`; its prior contents were preserved at
`/tmp/geodml-jupiter-commands-before-search-pilot.html`.

Next runnable local step: the unittest command in `search_experience_pilot.md`.
Next cluster prerequisite: commit/deploy this code and supply three actual prompt
IDs with their capture checkpoint and frozen plan. Verify model-specific context
capacity before execution. No allocation, endpoint inference, download or SSH
operation was performed. All-model quality, human calibration, actual search
examples and GPU performance remain unmeasured.
