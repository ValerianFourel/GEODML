# Nemotron SI-v3 passage selection, 2026-09-29

## Decision and evidence

Valerian chose numbered passage selection over model-written quotations and
approved the resulting plan with "Implement the plan." This implements that
local milestone, including publication and the operator page. GPU execution is
a separate milestone; do not treat the previous SI-v2 replay approval as approval
for another replay or a replacement allocation.

The user-pasted SI-v2 replay ran in job 5169986 on hkn0403, pin
`1722926cf1eda4b2f568491e77c7307e2112cc2b`. Native xgrammar 0.2.7 accepted its
schema. Run path:
`/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/runs/nemotron-si-v2-replay-job5169986-20260929-094837`.
It finished with exit 1 / status failed: SI 21/29 valid, J1 5/5, 2/5 complete
cells, 76.2 seconds of judging. Eight terminal semantic failures remained; the
first source quotation blended answer wording with source wording. This improved
on SI-v1's 10/29, 0/5 and 138.4 seconds but did not pass. The selector's count of
60 is from filtering to saved prompt IDs, not a new corpus total. These are pasted
observations, not live cluster state. The earlier job 5169925 expired naturally.

## Code and contracts

The implementation is the commit containing this handoff, on branch
`threehour-relaunch-fix`. Scientific rubric and serving configuration are
unchanged. Main owner: `source_importance.py`; shared runner retains historical
retry contracts. Diagnostic preparation/launcher/preflight use the new protocol.

- Protocol `agentic-source-importance-v3`, task `source-importance-task-v3`, retry
  `si-selection-retry-v3`.
- Answer units remain `a1..aN`. Source title and text use the existing deterministic
  sentence splitter independently, with IDs `title1..` and `text1..`. No clipping.
- Raw matches contain only `answer_unit_id` and `evidence_unit_id`, each constrained
  to supplied IDs. Grade 0 requires no matches; grades 1–5 require 1–3 pairs.
  Unknown/non-string IDs, duplicate pairs, extra keys and invalid grades fail.
- Valid selections resolve to whole exact passages. Parsed matches retain
  `answer_quote`, unit-relative `answer_start/end`, `evidence_quote`, source-field
  `evidence_start/end`, and `evidence_field`, adding `evidence_unit_id`. There is
  no 320-character cap on program-copied text. This is explicitly sentence-level
  provenance, not a claim that every clause is supported.
- Frozen tasks retain the masked answer, answer map, original title/text and source
  map. Preparation and replay share one reconstruction path. Task identity binds
  prompt/schema, original source text, masked answer, passage maps, cap and retry
  contract; whitespace/offset changes cannot silently reuse identities. Maps,
  ID, seed and hashes must reproduce at load time.
- Keep one corrective retry, both attempts and error categories. SI-v3 feedback
  addresses passage IDs; SI-v2 still gets quotation feedback; SI-v1/claims-v3 keep
  identical prompts. Old SI task records require their original pinned checkout.
- Smoke examples now print both raw IDs and `EXAMPLE_RESOLVED_OUTPUT`, and keep
  both in example.json. Existing exact-cell replay, J1, saved diagnostics and
  nonzero exits on incompleteness remain.

No additional models, dependencies, frameworks, GPUs, token caps or retries.
No J3 or production-bout integration. Grouped citation masking and full-answer
recovery remain as before. Valid passage IDs establish provenance, not entailment.

## Proof

Three new selection/offset regression cases failed on SI-v2 for the expected
schema mismatch, then passed after implementation. Final focused run:

```text
/Users/valerianfourel/miniconda3/bin/python -m pytest -q \
  analysis/tests/test_source_importance.py \
  analysis/tests/test_source_importance_pipeline.py \
  analysis/tests/test_horeka_nemotron.py \
  analysis/tests/test_acl_arr_runner_reliability.py \
  analysis/tests/test_try_claims_v3_judge.py \
  analysis/tests/test_agentic_judging.py
100 passed, 1 skipped, 6 subtests passed in 7.19s
```

Coverage includes long Unicode/Markdown, qualifications/negation, repeated text
at distinct offsets, title-only/text-only inputs, unknown/duplicate IDs, version
and map tampering, old retry behavior, exact replay and smoke failure diagnostics.
The skip is conda's missing jsonschema. System Python independently validated the
actual SI-v3 schema and passed nine acceptance/rejection cases. Native xgrammar
is absent locally; the prior SI-v2 native success is not SI-v3 validation.
`git diff --check` passed. All local inference responses are mocked.

## Operator page and next milestone

Update root `testnemotron.html` to the final SI-v3 commit and reopen in Safari.
Keep login/node commands separate. Login setup fetches the new pin into its own
checkout, checks the native schema, and reports allocation state. The node replay
must remain explicitly gated until authorization and remaining time are confirmed.
Do not overwrite the old environment pin, checkout, or SI-v1/SI-v2 outputs.

Next live test is the original five cells via saved
`nemotron-si-smoke-20260929-0829/attempts/job5169925/trial/cells.jsonl`, in a new
SI-v3 run directory. Budget estimate remains 10–25 minutes including startup,
using the unchanged exclusive four-A100/32-CPU/all-memory profile. Startup and
actual remaining allocation time need checking; no automatic extension or retry.
Require 29/29 valid SI, 5/5 valid J1 and five complete cells, then inspect the
resolved pairs for support quality. Follow with five fresh Qwen and five Llama
cells only as a separately authorized step. Preserve model-based reference
validation as later scientific validation. Do not promise a valid-corpus runtime
or production readiness from these small structural smokes.
