# GEODML Experiment V2 paper dataset, version 1

Status: final-paper export contract only. The separately versioned recovery
export is implemented in `build_agentic_recovery_dataset.py`; see
`agentic_recovery_operations.md`. No cluster dataset or HF upload has been
verified in this session. Full-population scientific completion remains unaudited.

## Scope and packaging

Use one private Hugging Face dataset containing immutable snapshots of the
Experiment V2 prompt population, plans, outcomes, evidence, execution records,
and coverage. Keep the original 500-prompt pilot identifiable within the larger
population. Earlier fixed-prompt experiments and readiness annotation studies
retain their own dataset contracts; link their verified exports in the catalog.

```text
README.md
catalog.json
snapshots/<snapshot_id>/
  manifest.json
  schemas/
  summary.json
  summary.md
  data/
    prompts/part-*.parquet
    cohorts/part-*.parquet
    protocols/part-*.parquet
    coverage/part-*.parquet
    generations/part-*.parquet
    judgments/part-*.parquet
    evidence/part-*.parquet
    attempts/part-*.parquet
    calls/part-*.parquet
    artifacts/part-*.parquet
    exclusions/part-*.parquet
  raw/
    traces/part-*.jsonl.gz
    claims/part-*.jsonl.gz
    manifests/part-*.jsonl.gz
    logs/part-*.jsonl.gz
  provenance/
    source-commits.json
    export-config.json
    validation-report.json
    checksums.sha256
```

Use typed Parquet tables with declared schemas and stable column names. Preserve
arrays as arrays. Partition initially by table and then, where useful, by model;
never by individual prompt, cell, retry, or job. Target roughly 128 MiB
uncompressed shards, splitting only between complete rows. This target is a
packaging choice, not a scientific parameter. Keep every table's schema present
even when its row count is zero.

Raw JSONL shard rows contain `artifact_id`, `source_sha256`, and `raw_text`.
For UTF-8 source artifacts, encoding `raw_text` back to UTF-8 must reproduce the
original bytes and checksum, including whitespace and final newline. Binary
inputs such as frozen search Parquet files use separate content-addressed
artifact files referenced by the artifact table. Never silently reserialize a
source artifact and retain its old checksum.

Snapshot IDs are immutable export identifiers, separate from run IDs. The catalog
lists snapshots, their manifest checksums, validation states, and related dataset
revisions. Updating a catalog must not overwrite an earlier snapshot.

## Table contracts

All exported paths are dataset-relative. Original cluster paths belong in
`artifacts.source_path`, where they are provenance rather than required joins.
All timestamps include their timezone; normalize to UTC only when the source
timezone is known. Unknown values are null, never fabricated zeros.

| Table | Row represents | Required identity and principal fields |
| --- | --- | --- |
| `prompts` | One population prompt | `population_id`, `prompt_id`, `prompt_sha256`, `prompt_text`, `keyword`, original axis measurements and bin, source artifact reference |
| `cohorts` | One prompt's membership in a frozen cohort | `cohort_id`, `prompt_id`, selection manifest hash, selection seed, original-pilot flag |
| `protocols` | One explicit execution specification | `protocol_id`, model ID/revision, generator or judge role, native protocol version, configuration hash, source commit, seeds, search snapshot hashes, compactor revision, thinking/token settings |
| `coverage` | One target generation or judging slot | `population_id`, `cell_id`, generator alias, stage, selected protocol reference, plan status, coverage status, accepted outcome reference, exclusion reason if applicable |
| `generations` | One distinct validated generator outcome | `generation_id`, `cell_id`, `prompt_id`, prompt hash, method, engine, condition, model ID/revision, protocol, native claim fingerprint when available, request hash when available, answer, ordered ranking URLs, trace reference/hash, diagnostics reference |
| `judgments` | One distinct validated judge outcome | `judgment_id`, native judge task ID, exact `generation_id`, source result/trace hashes, judge model ID/revision, protocol, claim fingerprint when available, structured judgment fields, evidence mapping reference |
| `evidence` | One evidence item in a particular recorded context | `evidence_context_id`, local `evidence_id`, generation/trace reference, step, URL, title, snippet text, presented position, compactor score when recorded |
| `attempts` | One allocation task or recorded cell attempt | `attempt_id`, `attempt_kind`, parent attempt reference, run ID, Slurm array/task/raw IDs, node, start/end, elapsed time, scheduler state, exit code, failure type, source log reference |
| `calls` | One independently identifiable recorded model request | `call_id`, producer attempt reference, task/outcome reference, model ID/revision, call kind, request ID or original event locator, response status, latency and token usage when recorded, evidence level |
| `artifacts` | One original artifact location | `artifact_id`, source path, source checksum, byte count, artifact kind, packed destination and record locator, duplicate-content reference, export eligibility |
| `exclusions` | One omitted, invalid, conflicting, or unmappable item | artifact/task reference, reason code, audit impact, transfer eligibility and provenance |

The three judge score fields `request_fulfillment`, `evidence_grounding`, and
`judge_confidence` are integers from 1 through 5. Preserve
`unsupported_claim_count`, `ideal_relevance_ranking`, and
`realized_support_ranking` exactly under the existing validated judge schema.
Do not substitute a new success metric or flatten rankings into unordered sets.

Local evidence IDs such as `S1` are scoped to their recorded context. URL equality
alone does not prove that snippet content or presented order was the same.

## Scientific identity and reconciliation

Preserve native `cell_id`, `judge_task_id`, prompt IDs, and immutable revisions.
A cell ID does not include the generator model; it is not by itself a unique
generation key.

1. Verify population source hashes, prompt text hashes, and task membership.
2. Resolve claim roots from all manifests, including nested adaptive and judge
   references. Traverse each physical registry once. The reported 70 top-level
   registry references are an inventory count, not proof of exhaustive coverage.
3. For native claims, preserve all five identity fields: `task_id`, `model_id`,
   `model_revision`, `protocol`, and `request_sha256`. Verify identity and outcome
   hashes and validate the scientific payload using the pinned source validators.
4. Record all original artifact locations, but deduplicate identical scientific
   outcomes. Shared claims and materialized worker copies are references to the
   same outcome, not additional completed cells or calls.
5. Assign export outcome IDs as SHA-256 of the canonical JSON object containing
   the native identity fingerprint and validated scientific payload digest.
   Preserve the native fingerprint separately. Exclude machine paths and
   exporter timestamps from the scientific payload digest.
6. For legacy outcomes without claims, retain the validated task identity,
   original configuration/provenance, and payload digest. Use an explicit
   `legacy` identity namespace; leave unknown native fields null. Merge with a
   native outcome only when validation establishes equivalence. Record unresolved
   alias relationships instead of counting both toward coverage.
7. Preserve different request/protocol variants separately. Never select an
   outcome because its job ID or modification time is newest. A versioned
   acceptance policy in `export-config.json` determines coverage. Unresolved
   competing outcomes or same-identity payload conflicts make the slot ambiguous.
8. Bind each accepted Nemotron judgment to the exact accepted generation and
   evidence/trace payload. A judgment of an older answer does not complete a
   replacement answer's judging slot. Keep recorded-conversation V2 and blinded
   V1 judgments distinct.

The result of a validated durable claim counts even if a later local result or
checkpoint write failed. A failed Slurm allocation does not invalidate earlier
committed outcomes. Mere file existence, `status=submitted`, and manifest totals
are insufficient completion evidence.

## Coverage and calls

The population manifest supplied by Valerian reports 26,009 prompts. An export
must verify both referenced files and their row identities before asserting that
population as validated:

| Source | SHA-256 |
| --- | --- |
| `compliant-candidates.jsonl` | `1718321f8fc86f63d00aab30e87991ada9b62a59c5ef4ce99b5acef0df4d32e9` |
| `final-axis-map.jsonl` | `43189f68bcafc77f9dceb7a1a8d993251d4c2a739b401ef4fd24cb64e292682e` |

Extending the pilot factorial to this population gives 12 cells per prompt per
generator: two methods, two engines, and three evidence conditions.

| Coverage stage | Full-population target |
| --- | ---: |
| Qwen generation | 312,108 |
| Llama generation | 312,108 |
| Nemotron bulk judging of both generators | 624,216 |

These are coverage targets, not a claim that all queues have been frozen or
approved for execution. `plan_status` is `unplanned` or `frozen`; deriving an
expected slot does not manufacture a native task ID for an unprepared judge case.

Each target slot has exactly one `coverage_status`: `complete`, `missing`,
`failed`, `busy`, `invalid`, `ambiguous`, `blocked_on_generation`, or `unverified`.
`busy` requires observed ownership at capture time. `blocked_on_generation`
applies only to judging. `failed` means a validated terminal task failure, not
merely a failed allocation. If no task was frozen, record `unplanned` in the plan
field and `missing` in coverage only after verifying that no accepted outcome
exists. Otherwise use `unverified`.

Report accepted completion as `100 * complete / target`. Keep the full target
fixed when a prompt is excluded from the Hub; exclusions affect transfer scope,
not research completion. Publish project coverage and uploaded coverage
separately. A snapshot with unaudited artifacts cannot label its counts exact;
report validated lower bounds and null exact percentages instead.

Also report complete/partial/untouched prompts per generator, paired generation,
and fully generated-and-judged prompts. Publish breakdowns by cohort, method,
engine, condition, model, and protocol. Cohorts may overlap; their counts must
not be summed as a population total.

Report model calls separately from coverage. Two to four logical generator
steps per cell do not determine actual HTTP calls, schema retries, transport
retries, or failed/unpersisted requests. Reused traces do not create new calls.
Count physical requests only when producer/request identity establishes a unique
execution; otherwise report recorded logical events separately and mark the
physical-call total unknown. Retain failures and retries as execution history.

## Compact output and publication

The default terminal output contains one row per stage for the original pilot
and full population, then artifact validation totals, call-count observability,
failure summary, snapshot path, and upload state. Write verbose manifests,
per-cohort tables, and error details to files. Never dump every manifest to stdout.

`summary.json` carries scope, target, validated completed count or null, missing
count or null, exact percentage or null, audit status, and evidence references.
`summary.md` renders those same values. The initial known status is recorded in
`agentic_paper_dataset_observed_status.json`; it is user-supplied audit evidence,
not a newly executed audit or a prepared upload bundle.

Before publishing, validate unique keys, foreign keys, checksums, model/protocol
identity, evidence order, and count reconciliation. Fail publication of a snapshot
advertised as complete when these checks fail. A partial snapshot must explicitly
list its coverage and exclusions. Record exporter Git SHA, configuration hash,
validator Git SHA, source commits, capture interval, and the resulting Hub commit.
If writers are active, record that the capture is non-atomic and do not assert
a single consistent point-in-time state without an immutable capture procedure.

Apply the existing source-transfer rules, including exclusion of restricted-local
and WildChat-derived material. Missing provenance is an export exclusion requiring
review, not permission to transfer. Do not include model weights, virtual
environments, caches, credentials, or live lock files. Retain original data on
JUPITER. The repository must be verified private before transfer; creating a repo
with `private=True` does not establish an existing repo's visibility.

No allocation, resubmission, deletion, data conversion, or upload is authorized
by the dataset contract itself. The next implementation milestone is a resumable
exporter and validator that implement this contract against the inventoried roots.
