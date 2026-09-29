# Nemotron SI-v2 quotation fix, 2026-09-29

## Authorization and scope

Supersedes the no-edit instruction recorded in
[the SI-v1 review](2026-09-29_nemotron-si-review_handoff.md). After inspecting the
failed responses, Valerian approved the inference-fix plan and said "Implement
the plan." This milestone implements and tests it locally. No cluster command,
allocation, inference, model download or publication was executed. Preserve the
existing scientific design and serving profile; production bouts are separate work.

Active branch: `threehour-relaunch-fix`, in `.worktrees/threehour-relaunch-fix`.
The implementation is the commit containing this handoff. The previous historical
SI-v1 pin remains `55b7a4a36d090176a5129c50b564b23d6959b420`.

## Evidence and diagnosis

User-pasted output from HoreKa job 5169925, node hkn0403, reports 5 Qwen cells,
J1 5/5, SI 10/29, and no complete cells in 138.4 seconds. All final responses
stopped normally, below the 640-token cap. Terminal failures were 5 answer-quote
mismatches, 8 source-quote mismatches, and 6 zero grades with nonempty matches.
One answer quote rewrote "Look for capabilities in forecasting future demand"
as "It should support forecasting future demand". This is paraphrasing, not
punctuation that the validator should forgive. The identical-prompt retry rescued
none of those failures. Compound S-ID citations also escaped masking.

Run: `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/runs/nemotron-si-smoke-20260929-0829`.
Its replay selection is `attempts/job5169925/trial/cells.jsonl`.
The user subsequently started an `0850` trial and suspended it with Ctrl-Z.
These are pasted historical facts, not a live cluster check. Do not resume or
terminate it automatically, start a competing server, or close the allocation shell.

## Implemented behavior

- `source_importance.py`: SI-v2 prompt asks for short, contiguous exact quotes
  with qualifications preserved. The schema requires grade 0 with zero matches,
  or grades 1–5 with 1–3 matches. The existing strict quotation validator and its
  limited typography normalization remain. No fuzzy matching or automatic score repair.
- `run_acl_arr_vllm.py`: SI-v2 gets one corrective retry containing the rejected
  match number, quote and target field/unit. Both response bodies, usage, errors,
  seeds and prompt hashes are retained. Transport limits, truncation failures,
  legacy SI-v1 and claims-v3 retry behavior stay unchanged.
- Task identity now binds the actual prompt/schema hashes and retry contract.
  Frozen records must reproduce the protocol, version, ID, seed and hashes.
  SI-v1 records are refused by SI-v2; replay them with their original checkout.
- Grouped citations such as `(S1, S2)` and `[S3, S7]` are masked. Cell records
  preserve reversible mask spans and original/masked answer hashes.
- `try_source_importance_judge.py`: `--cells-from` replays exact cell fingerprints
  and generation/trace IDs, with an optional selection-file SHA256. No missing-cell
  substitutions. Tasks and cell provenance are written before connecting to the
  server. Results are saved as each request completes, using the existing bounded
  scheduler. Incomplete selection, invalid SI/J1, or execution failure returns 1
  with diagnostics. Valid all-zero source grades can pass.
- `horeka_nemotron.py`: exposes `--model qwen38|llama4`, carries replay identity
  and checksum, and runs the native `--check-schema` preflight before preparation
  or model startup. Saves `schema-check.json`; rejection stops execution.

The 0–5 rubric, one-answer/one-source design, separate J1, full-answer recovery,
all observed sources, and the model/revision stay unchanged. No J3 added. BF16,
TP4, four A100, context 73728, eager mode, memory utilization 0.85, concurrency 4,
temperature 0, thinking disabled, SI cap 640 and J1 cap 64 stay unchanged.
This remains a diagnostic driver, not a resumable five-hour production worker.

## Verification

Focused CPU run with the installed conda Python:

```text
python -m pytest -q \
  analysis/tests/test_source_importance.py \
  analysis/tests/test_source_importance_pipeline.py \
  analysis/tests/test_horeka_nemotron.py \
  analysis/tests/test_acl_arr_runner_reliability.py \
  analysis/tests/test_try_claims_v3_judge.py \
  analysis/tests/test_agentic_judging.py
90 passed, 1 skipped, 6 subtests passed in 8.96s
```

The skip is `jsonschema` missing from conda. System Python has it: independent
Draft 2020-12 schema validation passed six acceptance/rejection checks, including
both valid branches, contradictory grade/match combinations, excessive matches,
and unknown unit IDs. No dependencies installed. Native `--check-schema` exits
nonzero locally because `xgrammar` is absent. That is not proof of compatibility;
the installed HoreKa runtime must accept it before inference. Tests also exercise
launcher refusal before model startup when native preflight returns failure.
`git diff --check` passed. All test model responses are mocked, not scientific evidence.

## Next milestone

Valerian normally publishes and checks out the recorded implementation commit.
Do not point a running pinned job at edited files. Once that exact checkout is
available on HoreKa, run this CPU-only command with `REPO` referring to the new
checkout and `RT` to the existing HoreKa runtime:

```bash
"$RT/bin/python" "$REPO/analysis/scripts/try_source_importance_judge.py" --check-schema
```

Expect `status: accepted` and `protocol: agentic-source-importance-v2` plus the
installed xgrammar version and schema hash. A successful grammar check does not
yet prove live constrained decoding or semantic quality.

After fresh runtime/resource estimates and allocation approval, validate the
original five cells using `--cells-from`, then five fresh Qwen and five Llama
cells. Preserve the original 0829 artifacts and use new output directories.
Check exit status, `summary.json` status, requested/complete cell counts, J1/SI
successes, retry histories and actual quotations. Even a zero-failure 15-cell
diagnostic is not production validation. Measure sustained valid throughput
before revising the five-hour bout estimate. The old estimate describes failed
processing costs, not time to a valid complete corpus. No new GPU allocation or
five-hour bout is authorized by this implementation milestone.
