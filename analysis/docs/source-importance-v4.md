# SI-v4 implementation and evaluation

SI-v4 creates a source-blind answer map once, then judges each observed source
against that map and the complete answer. It retains the SI-v3 ordinal anchors,
including grade 5's **most essential content**. It measures observable textual
support and importance within the completed answer, not causal reliance or
unique contribution. J1 remains a separate, unchanged task.

This is an implemented development protocol, not a validated production judge.
No Gemma/v4 semantic result follows from CPU tests or constructed fixtures.
SI-v3 remains the default of the existing preparer and its historical artifacts
are unchanged. The original forty audited Gemma cells are development cases.

## Protocol and contracts

The authoritative exact prompts are `MAP_INSTRUCTIONS` and `SOURCE_INSTRUCTIONS`
in `analysis/interpretability/pipeline/source_importance_v4.py`. Its schema
builders expand valid word, claim and source passage enums for each input.
There is no free-form claim paraphrase, model-written offset, claim cap,
joint-source attribution, ideal-relevance task or confidence score.

Stage A receives the request and citation-masked complete answer only. The request
resolves references, not the ideal answer. Existing answer units become `a1`,
`a2`, etc.; non-whitespace words become `a1w1`, `a1w2`, etc. Python resolves exact
masked-answer offsets, preserves Unicode and assigns canonical `c1` claim IDs.
Full original units remain visible even when a claim has several spans.

Claims have an assertion/recommendation/attributed-report/global-absence kind and
one importance role, or two adjacent roles when centrality is ambiguous. Every
word must occur in a claim or explicit exclusion. Shared context is allowed;
exclusions cannot overlap claims. Coverage and valid spans do not prove that the
map preserved meaning. Independent map-fidelity review remains mandatory.

Stage B receives the same request, full answer, fixed map, and one unchanged
title/text record. It returns sparse full/partial/contradicted/uncertain findings,
affected answer spans, 1–3 passage witnesses per finding, and an ordinal grade.
Omitted claims become unsupported only after a valid `scored` completion.
Witness counts do not determine grades. Partial support receives credit only for
the supported content; shared topic words and mismatched entities do not suffice.
No other source, generator identity, ranking, J1, readiness or treatment metadata
enters either prompt. Experimental metadata remain in the frozen cell records.

Positive grades require supporting findings. Zero can retain contradiction
findings but no positive matches. `uncertain` and `map_issue` have null grades.
An unusable/failed map blocks source calls. A Stage B `map_issue` quarantines all
scores sharing that map, including already completed scores; pending dependent
calls stop. Raw outcomes remain auditable. There is no silent v3 fallback.

Pure abstention, non-substantive answers and global evidence-absence-only answers
receive separate N/A reasons. A one-source judge cannot verify a global absence
claim. Mixed absence/factual answers retain both in the map and grade only the
supported factual content within the complete answer. Narration and off-target
answers remain eligible. These eligibility changes are separately versioned.

Example answer:

```text
[a1] Choose Acme for offline CSV export.
[a2] It exports CSV without an internet connection.
[a3] Its interface is blue.
```

The first sentence has six words. A faithful map links the central recommendation
to `a1w1..a1w6`, its major justification to `a2w1..a2w7`, and peripheral colour to
`a3w1..a3w4`. A source explicitly recommending Acme for that use and establishing
offline export can receive 5. Colour alone receives 1. Unrelated material receives
0. The worked JSON contracts are exercised in `test_source_importance_v4.py`.

## Preparation and comparison arms

Use a committed checkout and new output directories. These are CPU preparation
commands for an already-open shell; substitute the existing runtime and paths.
They neither start a server nor allocate a GPU.

```bash
"$RT/bin/python" analysis/scripts/prepare_source_importance_tasks.py \
  --source "$QWEN_DATA:qwen38" --source "$LLAMA_DATA:llama4" \
  --cell-fingerprints "$CELL_SELECTION" \
  --protocol si-v4 --max-tokens 4096 --map-max-tokens 4096 \
  --truncation-sensitivity-fraction 0 \
  --output "$V4_INPUTS"

"$RT/bin/python" analysis/scripts/prepare_source_importance_tasks.py \
  --source "$QWEN_DATA:qwen38" --source "$LLAMA_DATA:llama4" \
  --cell-fingerprints "$CELL_SELECTION" \
  --protocol si-v3 --max-tokens 4096 \
  --truncation-sensitivity-fraction 0 \
  --output "$V3_INPUTS"
```

`CELL_SELECTION` contains exactly the selected verified cell fingerprints, one
per line. Missing selections fail rather than silently shrink a held-out batch.
Use matched generator pairs where available. `--prompt-ids` alone selects every
available cell for those prompts and therefore does not guarantee 120 cells.

Preparation preserves stored/recovered answers, reversible masks, raw final calls,
trace hashes, repeated Reactive observations, prompt metadata, task metadata and
keyword memberships. It compares actual serialized final evidence and the final
raw answer with judge inputs. An unresolved check stays `provenance_unresolved`,
not verified. Interpretation is attribution to a supplied record unless webpage
provenance is independently established. Do not fetch replacement pages or clean
mixed records. Missing metadata remain missing, especially the high-readiness
generation regime; do not infer it from the readiness value.

Freeze both arms with `--preprocessing si-passages-mask-v2` to investigate prose
markers such as "Snippet S1". V3 then uses the distinct
`agentic-source-importance-v3-mask-v2` namespace with the original v3 rubric and
schema. Do not mix that result with historical v3. The first development
comparison uses the original masking version; the held-out freeze must select
one common validated version. Request removal is not part of this implementation.

The historical 640-token run is not the 4096-token bridge baseline. Preserve its
artifacts; compare architecture with identical source budgets and judge settings.

## Execution, identity and resume

`analysis/config/si_v4_gemma.template.json` pins the previously used Gemma model
and runtime profile. Replace only the local tokenizer path and repetition ID for
the intended run, and verify the actual serving receipt. The template's official
chat-template hash was inspected previously; it does not attest the historical
server's loaded template. The new client hashes its actual local template and
checks the model name exposed by the server. It cannot remotely prove a weight
revision or hardware identity: retain the pinned server launch/config/runtime
receipts with the experiment.

`run_source_importance_judge.py` consumes an already-running authenticated vLLM
server. Its required arguments are `--inputs`, `--output`, `--config`, `--base-url`,
`--workspace` and `--quota-evidence`. It requires a clean committed checkout,
Slurm job/end-time metadata and the existing managed authentication boundary.
This command is not a model launcher or Slurm submission command. Use the
existing validated cluster boundary; no namespace, precision or model fallback
has been added. Shared-hour reservations are not silently created by this driver.

The controller checks quota freshness, free bytes/inodes and storage incidents
before admission. Refresh quota evidence through the existing authorized finite
helper if it expires; a stale snapshot stops new calls. Live allocations remain
untouched. Context overflow is a failure, never truncation of frozen input.
Capture fresh quota evidence after model loading and before starting the client;
a snapshot taken before a long cold start may already be stale.

SQLite indexes the finite frozen backlog on disk. One coordinator holds an
exclusive output lock. The existing striped ledger owns inference tasks, and
the existing writer fsyncs raw inputs/responses and seals shards. Mapping releases
dependent source tasks incrementally through indexed queues. J1 executes in a
separate measured phase. The entire corpus is not loaded into the controller.
Existing corpus-selection helpers still index the prompt/verified-cell inventory
in memory, as before; this is not an assertion of a production-scale benchmark.

Semantic task identity covers text, prompt/schema versions, preprocessing,
eligibility, token budget and retry contract. Source identity includes the
canonical map hash. Execution identity additionally binds configuration, code
revision, input manifest and repetition. The original v3 request hash is unchanged;
the new execution identity includes settings that its old hash omitted.

Source judgments receive one structural corrective retry and at most three
transport attempts per request. Truncation is terminal for v4 at the frozen
budget. Explicit uncertainty or unusability is not retried until a preferred
answer appears. Raw attempts and finish reasons remain available.

Normal cancellation checkpoints unfinished work and seals completed results.
Resume uses the same output/configuration and does not repeat completed calls.
After abrupt process death, `--recovery-evidence` must name JSON containing
`writer_id`, `slurm_job_id`, `owner_terminal: true`, and an `evidence` description
backed by actual scheduler accounting. Do not assert terminal ownership while an
allocation can still write. Recovery verifies sealed records and reconciles a
durable response even if interruption occurred before its index commit. Partial
file tails fail closed; they are preserved for diagnosis.

Each report directory contains cells, a summary and reusable `maps.jsonl`. For
fixed-map Stage B repeats, use a new repetition ID/output and `--fixed-maps` with
exactly the selected input maps. End-to-end repeats use new repetition IDs/outputs
and omit that argument. Identical semantic tasks still deduplicate within a run.

## Diagnostics and acceptance

```bash
"$RT/bin/python" analysis/scripts/prepare_si_v4_diagnostics.py \
  --split development --output "$V4_CONSTRUCTED_DEV"
"$RT/bin/python" analysis/scripts/prepare_si_v4_diagnostics.py \
  --split heldout --output "$V4_CONSTRUCTED_HELDOUT"
```

Each split has 48 source-answer pairs. Expectations are separate from model
inputs. These template-related constructed examples cover exact/paraphrased
support, topics/entities, contradictions, qualifications, lists, unsupported
advice, equivalent sources, narration, pure/mixed absence, distractors, deletion
and an actual bijective passage-ID rename. They are diagnostic assertions, not
human annotations, and do not replace fresh corpus validation.
Constructed source calls use the same explicitly recorded seed, so ID changes
do not also change the inference seed. This override is refused for corpus
freezes. Identical-answer controls reuse the same source-blind map.

The bounded design and thresholds are in `analysis/config/si_v4_evaluation.json`.
Use the forty development cells first. Freeze a credible candidate before the
120 fresh cells: 96 core cells from 48 paired prompts, plus 24 stress cells from
12 paired prompts. Balance marginal distributions, preserve generator pairing
and separate prompt/keyword groups; report unavailable strata rather than
pretending to cross every factor. Repeat 24 cells three times in both modes.
Reserve another 120 fresh cells after revisions. Actual fresh selection and
semantic inference are later experiments, not completed by this code change.

Targets remain: no unexplained integrity errors; 100% retained structural
validity; 99% eligible completion; 95% resolved witness agreement with at most
5% ambiguity; 90% grade-range agreement; 95% exact repeats, 99% within one,
95% nonempty top-group agreement; 95% map fidelity with no essential omissions
or decisive reversals; no constructed unsupported 4/5; and a 3× warm compute
ceiling. Promotion also requires a 10-point paired grade-range improvement with
a keyword-cluster interval excluding zero and no material support/coverage
regression. Persistent entity/qualification errors block promotion independently
of averages. These are working screening criteria, not universal standards.

Use independently frozen model-reference ranges and separate witness/map audits.
Record exact GPT-6 Astra/Sol identities, disagreement and unresolved cases.
Model consensus is not human gold. Style is descriptive, not a causal mediator.
The readiness coordinate remains observational; preserve the looser-generation
metadata above 0.70. Judge changes do not repair erased shuffles or unseen
ablation targets.

Compare reports with `compare_source_importance_runs.py --baseline ...
--candidate ... --references ... --repeat ... --repeat ... --output ...`.
It rejects mismatched inputs/configurations and repeated execution identities.
References are JSONL, one record per cell/source, with:

```json
{
  "cell_id": "frozen cell ID",
  "url": "external join URL",
  "source_sha256": "hash from report",
  "masked_answer_sha256": "hash from report",
  "reviewer_models": ["exact logged reference model identity"],
  "frozen_before_grade_review": true,
  "acceptable_grade_range": [3, 4],
  "keyword_id": "frozen keyword ID"
}
```

Use a null range for unresolved centrality. Optional witness assessments require
`candidate_raw_output_sha256` and `candidate_pair_support`, whose entries are
supported/unsupported/ambiguous. Optional `map_sha256` and `map_assessment` bind
boolean faithful/essential_omission/meaning_reversal assessments to the actual
map. Optional style/readiness_stratum/source_length_stratum fields support
descriptive subgroup reporting. Preserve the separate reviewers' original
records; this tool does not adjudicate them. Missing labels remain missing.
The keyword bootstrap runs only with keyword IDs, never substitutes source-level
independence, and belongs on the cluster for substantial evaluations.

## Cost and reporting

For actual frozen unique counts U and P, v4 adds U mapping calls to P source
calls, with different input/output token lengths. J1 is separate. Reports retain
phase wall time, node-hours, GPU-hours, request times/tokens, failures and N/A
coverage. Request latencies overlap and must not be summed as node-hours.
One four-GPU node-hour is four GPU-hours. Fixed-map repeats cannot establish the
end-to-end cost ratio. Report startup, drain, total allocation occupancy and
reasoning tokens separately if a later reasoning configuration is introduced.

The 3× ceiling is an acceptance constraint, not a measured forecast or allocation
approval. Before any allocation, measure/estimate the actual backlog, give a
runtime range and resource-hours, then obtain explicit wall-time approval.
There is no automatic launch, extension or production-scale rollout.

Metrics retain ties, all observed sources and missing grades. Positive flat
vectors have trivial top alignment and undefined tau. All-zero vectors have no
positive top group; no grades 4/5 means undefined omission. Report top-group size,
generator-list coverage and candidate-set chance baselines. The new metrics
version fixes presentation alignment for an empty generator ranking without
rewriting old results. Ordinal grades are not percentages or equally spaced use.

Publish version-specific prompts, configuration, post hoc development history,
dated freeze, recovery/masking, eligibility, failures, validation limitations and
selection metadata. A later protocol note is not preregistration of observed
experiments. Rollback selects the retained v3 implementation and artifacts;
it never replaces v4 failures with v3 scores inside one result set.
