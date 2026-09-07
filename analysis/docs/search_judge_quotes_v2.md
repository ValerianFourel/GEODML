# Quote-only judge contract

Valerian, this opt-in pilot contract removes character counting from the model's task. It does not relax evidence matching or revise historical judgments.

## Problem and decision

The reported v1 failures include incorrect Unicode offsets and claim IDs that do not belong to the answer. The existing validator correctly rejects both. Repeating the same request did not address its underconstrained schema or its demand for model-generated offsets.

The selected design keeps one request per answer. The model supplies scores, claim assessments, document IDs, and verbatim quotes. Code resolves each quote to exactly one span in that document's text and then invokes the original strict validator. The generation schema restricts claim IDs and document IDs to the public judge input. An answer with no claims requires an empty assessment list.

A competing design used one request per claim plus a whole-answer scoring request. That would remove claim selection from generation, but multiply long-evidence prefills and add orchestration. It is not included in this bounded fix.

## Contract and stored records

Select `search-experience-judge-quotes-v2` with `run-judge --judge-contract`. The default remains the original v1 contract. Existing callers and v1 results keep their identities.

The v2 evidence entry has `document_id` and `quote`, without `start` or `end`. The stored `raw_output` remains exactly what the model returned. The validated `parsed_output` includes deterministic start and end offsets. Existing resume validation recomputes the parsed result from the raw result, rather than trusting saved offsets.

The selected contract changes task identity and therefore task-derived seeds. Model revision, temperature, output allowance, evidence text, judge order, score rubric, and retry settings otherwise remain unchanged. Comparisons with v1 are protocol comparisons, not a claim of bitwise equivalence. Use a separate output directory. Never mix v1 and v2 successes as if they shared a frozen contract.

Quotes must occur exactly once in the specified document's text. Matching is case-sensitive and Unicode-exact. No whitespace repair, accent normalization, fuzzy matching, nearest-span choice, or first-occurrence fallback is permitted. Overlapping duplicate matches are ambiguous. A title absent from document text is not evidence text. A title present in document text can pass the mechanical check but may still fail semantic assessment.

Unknown or duplicate claim IDs, missing claims, unknown document IDs, and unsupported schema fields remain errors. V1 offset-bearing responses are not automatically accepted under v2. This change does not recover the three historical Qwen failures or alter their files.

## Verification and remaining boundary

Synthetic tests cover exact Unicode offsets, absent and repeated quotes, overlapping matches, unknown IDs, duplicate and missing claims, abstention, immutable inputs, v1 strictness, protocol isolation, checkpoint resume, and report reconstruction.

The installed-backend grammar check includes the nonempty-claim and empty-claim v2 schemas. Passing local contract tests is not proof that the cluster's xgrammar build accepts those schemas. Run the backend check on the compute node before loading a model.

No GPU success rate is measured for v2 yet. Mechanical quote presence does not prove factual support, citation coverage, or appropriate uncertainty. Human calibration remains necessary. This contract remains pilot-only and does not authorize production launch.

Local verification on 7 September ran 39 tests: 38 passed and one installed-vLLM check was skipped. An independent read-only review found no material correctness or comment issues. The review used an inherited agent, not verified multi-model diversity. CLI help exposes the new opt-in flag. No code was pushed or deployed as part of this change.
