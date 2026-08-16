# Semantic-readiness super prompt: Phase 1 implementation record

Date: 2026-08-16

## Scope

This document records work performed for **Phase 1 — Freeze semantic
corpora**, corresponding to **Subprompt 1 — Corpus freeze** in:

`GEODML GEO Master Codex Prompt v2 — Continuous Semantic Readiness, Four-Model Behavioral Panel, and Hierarchical Search-RAG.md`

Only Phase 1 was executed. No readiness annotation, LLM2Vec embedding, GEO
outcome inspection, production model inference, or cluster job was run.

The implementation was made in `geodml-mono`, which contains the active
analysis code and data pipeline. The master prompt itself is stored in the
separate `GEODML_Unified` working directory.

## Super-prompt requirement mapping

| Phase-1 requirement | Work completed | Status |
|---|---|---|
| Validate the 5,091-prompt Layer-1 corpus | Rebuilt it from the recorded surface and web inputs and compared every resulting record | Passed |
| Confirm exact counts and unique texts | Verified 5,091 unique item IDs and 5,091 unique normalized-text hashes | Passed |
| Preserve development/locked-confirmation separation | Verified 3,808 development and 1,283 confirmation records, with zero cross-split groups | Passed |
| Verify revisions, licenses, and hashes | Audited the Layer-1 provenance manifests, raw source revisions, per-record licenses, and file hashes | Passed |
| Freeze the Layer-2 registry before judging | Verified the versioned eight-source registry and its four-development/four-confirmation assignment | Passed |
| Acquire Layer-2 data only where legally and technically permitted | Acquired the five plainly open sources at pinned immutable revisions | Partial by design |
| Do not replace unavailable or gated sources | MS MARCO, WildChat, and LMSYS remain explicitly pending | Passed |
| Use deterministic filtering and sampling | Used seed `20260817`, a maximum of 1,000 exact-unique prompts per source, and stable bottom-hash sampling | Passed |
| Confirm deterministic artifact hashes | Repeated the complete five-source collection; the JSONL outputs matched byte-for-byte | Passed |
| Produce manifests and tests | Produced Layer-1/Layer-2 freeze artifacts, transfer acquisition metadata, and focused tests | Passed |
| Stop before readiness inference | The phase gate prevents full Phase 2 while transfer sources remain pending | Passed |

## Layer-1 freeze result

The existing Layer-1 corpus passed all freeze checks:

- records: `5,091`;
- development: `3,808`;
- locked confirmation: `1,283`;
- unique item IDs: `5,091`;
- unique exact-text hashes: `5,091`;
- normalized-text hash mismatches: `0`;
- cross-split groups: `0`;
- included records with unknown licenses: `0`;
- exact reconstruction from recorded surface and web inputs: `true`;
- corpus SHA-256:
  `c851c31f99bdfecd31238f36d6fe24d1e379a4ebf48c8dadc98ebfb2af1b26a8`.

The Stack Exchange acquisition audit also confirmed:

- 48 frozen retrieval probes;
- 1,380 unique acquired records;
- 1,091 licensed records included in Layer 1;
- 289 records without an API `content_license` retained in the raw audit but
  excluded from the corpus;
- raw-response directory SHA-256:
  `dc994a5cbd161b800d3abe5698b425a8dcedd3b02ce65dd7f3f3ddbd73d05561`.

## Layer-2 acquisition and deterministic sample

The source-to-split assignment was kept exactly as declared in
`semantic_readiness_transfer_sources_v1.json`. Sampling roles were retained as
acquisition metadata and were not treated as readiness labels.

| Split | Source | License/access | Pinned revision | Selected prompts | Status |
|---|---|---|---|---:|---|
| Development | OpenAssistant OASST1 | Apache-2.0, open | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | 1,000 | Frozen |
| Development | Google CCPE-M | CC-BY-4.0, open | `2c9cd30f33f3a154b5a27d015333679262ff36f5` | 370 | Frozen |
| Development | Google Taskmaster-1 | CC-BY-4.0, open | `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` | 1,000 | Frozen |
| Development | Microsoft MS MARCO v1 | research terms, local-only | Pending | — | Not acquired |
| Locked confirmation | AllenAI WildChat-1M | terms, sensitive content, local-only | Pending | — | Not acquired |
| Locked confirmation | Google Schema-Guided Dialogue | CC-BY-SA-4.0, open | `e852981ae34990f4358979625854259302feaa78` | 1,000 | Frozen |
| Locked confirmation | Amazon Shopping Queries | Apache-2.0, open | `7916cdf6ab75a462e77f20ab40428a10923998d5` | 1,000 | Frozen |
| Locked confirmation | LMSYS-Chat-1M | explicit license acceptance required, local-only | Pending | — | Not acquired |

CCPE-M produced 445 eligible rows but only 370 exact-unique eligible prompt
texts, so all 370 were retained. The approximately 1,000-per-source target is
an upper bound and no duplicate texts were added merely to reach it.

The five acquired sources produced:

- 4,370 transfer prompts;
- 2,370 development prompts;
- 2,000 locked-confirmation prompts;
- 4,370 unique transfer record IDs;
- 4,370 unique exact-text hashes;
- zero text-hash mismatches;
- zero exact-text overlap across the two splits;
- transfer-record SHA-256:
  `f5f98202dbec61e23589b07ac9627ba1edba616f4f69cdd43f31b8da7ffd163a`.

An independent replay with the same inputs, revisions, eligibility rules, and
seed produced the identical transfer-record SHA-256.

## Code added

- `analysis/interpretability/pipeline/semantic_readiness_phase1.py`
  implements the read-only Phase-1 audit and phase gate.
- `analysis/scripts/freeze_semantic_readiness_phase1.py` provides the CLI that
  writes the JSON manifest and Markdown report atomically.
- `analysis/tests/test_semantic_readiness_phase1.py` tests exact rebuilding,
  missing transfer snapshots, revision-pinned snapshots, empty snapshot
  rejection, cross-split leakage blocking, and portable raw-snapshot overrides.
- `analysis/docs/semantic_readiness_phase1_jupyter_jsc_runbook.md` gives the
  CPU-only Jupyter-JSC staging, rehearsal, final-freeze, and acceptance path.

The audit rejects count changes, duplicate IDs, duplicate Layer-1 text hashes,
text-hash mismatches, out-of-range texts, unknown included licenses,
development/confirmation group leakage, changed provenance files, malformed
Stack Exchange snapshots, unpinned transfer snapshots, and empty transfer
snapshots.

## Generated local artifacts

The following artifacts were generated under the ignored `analysis/output/`
tree:

- `semantic_readiness_phase1_freeze_v1/phase1_corpus_freeze_manifest.json`;
- `semantic_readiness_phase1_freeze_v1/phase1_corpus_freeze_report.md`;
- `semantic_readiness_transfer_records_open_v1/run_manifest.json`;
- `semantic_readiness_transfer_records_open_v1/transfer_source_diagnostics.json`;
- `semantic_readiness_transfer_records_open_v1/semantic_readiness_transfer_records.jsonl`.

The manifests record input paths, content hashes, licenses, access policies,
source revisions, split assignments, sampling seed, maximum sample size, and
the absence of scientific/model-inference results.

The five-open-source Jupyter-JSC rehearsal inputs were also packaged as a
curated 301-entry archive, excluding nested upstream Git metadata and the three
restricted/pending sources. Its SHA-256 is
`f0422481b094da8f8e4db5e0d4ed7668fed603e7857db0b53069deac0387d344`.
The JSC runbook requires download from an immutable Hugging Face commit and
revalidates this digest before extraction.

## Verification

The focused Phase-1 and existing semantic-corpus tests were run with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  pytest -p no:cacheprovider -q \
  analysis/tests/test_semantic_readiness_phase1.py \
  analysis/tests/test_semantic_readiness_transfer.py \
  analysis/tests/test_semantic_readiness_dataset.py \
  analysis/tests/test_query_free_surface_corpus.py \
  analysis/tests/test_horeka_semantic_readiness_job.py
```

Result: `21 passed`.

Syntax compilation, CLI help, whitespace checks, and `git diff --check` also
passed.

The real 5,091-row audit was also rerun with explicit
`--surface-source-input SOURCE_ID=PATH` overrides for the Dolly and Anthropic
raw snapshots. This simulates staging the same bytes under `$PROJECT` on JSC,
instead of following the Mac absolute paths retained for provenance. Both
overrides verified against the frozen hashes, Layer 1 remained frozen, and the
gate remained `phase1-transfer-snapshots-pending` with exactly MS MARCO,
WildChat, and LMSYS pending.

## Current phase gate and unresolved items

The current machine-readable gate is:

- Layer 1 frozen: `true`;
- Layer-2 registry frozen: `true`;
- all Layer-2 snapshots frozen: `false`;
- phase status: `phase1-transfer-snapshots-pending`;
- safe to begin full Phase 2: `false`.

The three remaining acquisition blockers are intentional:

1. MS MARCO requires compliance with its research-use terms.
2. WildChat requires license, privacy, and sensitive-content review before a
   local-only snapshot is admitted.
3. LMSYS-Chat-1M requires explicit acceptance through its gated dataset page.

No replacement datasets should be introduced. Once access is documented and
the exact three snapshots are acquired, rerun the transfer collection and
Phase-1 freeze over all eight sources. Do not start Phase 2 until the full gate
passes.

Finally, the implementation and generated manifests currently reflect the
local working tree. Before treating them as final scientific artifacts, commit
the Phase-1 code, regenerate the manifests from that exact commit, and record
the resulting Git SHA according to the local–GitHub–HoreKa workflow.
