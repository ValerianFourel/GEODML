# Nemotron v4 diagnostic against the intended construct

The tested Nemotron Nano configuration does not meet v4's intended standard and
is not a viable replacement for Gemma in this workflow. It returned no numeric
source grades and no complete cells. Its main problem is semantic: it repeatedly
rejects ordinary facts, recommendations and procedures as having no substantive
content. The one answer that reaches source judging produces unsupported
findings and seven unjustified grade abstentions.

This is a diagnostic of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, with thinking
disabled and the existing v4 r2 instructions. It is not a result about every
Nemotron model or configuration. The matched Gemma outputs also contain semantic
errors and are not a reference standard.

## Evidence and comparison

The private dataset is `ValerianFourel/geodml-experiment-v2-paper-private`.
Nemotron evidence is pinned to revision
`99d32ee66bc9fd35c4d2b8f099c996c13958babf`, file
`reviews/nemotron-si-v4-20cells-20261004-job5177346/assessment-nemotron-wwd7dud2/all-inference.jsonl`.
The downloaded bytes match the supplied SHA-256:
`9d940509b83c40e19959e1ab122ca96f523746fb61719caa20e6201b60b50170`.

All 20 frozen cell objects and all 147 task input records exactly match their
counterparts in Gemma `candidate-e2e-1`. There are 10 Qwen answers, 10 Llama
answers and 107 complete supplied source records. The 147 input records contain
20 map tasks, 107 source dependencies and 20 fulfilment tasks. Fulfilment was
explicitly not requested and is not counted as a failure.

The two saved judge configurations differ only in model identity/revision,
tokenizer path and chat-template hash. The model revision is
`bf77c3174f68ad409e1c2aa60daeb46e32d1c606`; execution code is
`f57435f425bd8bbffc8263f1f2af2da3c078e686`. The v4 protocol, task executor and
vLLM client source files are byte-identical to Gemma's scientific commit
`caf8a7d3fbe69736789767efcc4551d0e0347c11`. Precision, context, token cap,
temperature, semantic-task seed policy and concurrency were preserved.
All 27 request fingerprints and all 29 validation-attempt prompt hashes
reproduce. The twenty-cell selection reproduces seed 20261004 on the sorted
forty-cell development pool.

This compares complete workflows on identical inputs. Stage B receives each
model's own map, so it is not a fixed-map isolation of the source judge.
Stored source provenance remains unresolved: judgments concern the supplied
title/text record, not an independently verified webpage. No replacement pages
were fetched.

## What the models actually returned

| Observable outcome on the same 20 cells | Nemotron | Gemma first pass |
|---|---:|---:|
| Maps returned as `ready` | 1/20 | 20/20 |
| Maps returned as `unusable` | 18/20 | 0/20 |
| Maps failing final validation | 1/20 | 0/20 |
| Numeric source grades | 0/107 | 99/107 |
| Sources blocked by unusable or failed maps | 100 | 0 |
| Source results marked `uncertain` | 7 | 0 |
| Source inference failures | 0 | 1 |
| Sources excluded as `global_absence_only` | 0 | 7 |
| Cells with complete numeric grading | 0/20 | 18/20 |

These are output and completion counts, not semantic accuracy. Gemma's 20 ready
maps include substantive omissions, and its seven global-absence exclusions
include a mixed answer whose factual narration should remain assessable.
Nemotron's seven source calls ran; the other 100 source dependencies did not.
Their null grades are not zeros.

The reported `done: 27` means 20 terminal map tasks plus seven terminal source
tasks. It does not mean 27 successful judgments. There are 29 saved model
attempts after validation retries, all with `finish_reason: stop`. This run
reports no deadline admission stop. The failure is not explained by running out
of output tokens or by an unfinished queue.

## Failures against v4's goals

V4 measures the importance of content supported by one supplied source within
the answer as written. It deliberately separates that quantity from request
fulfilment, source reputation, ideal relevance and causal reliance. Its map must
preserve factual descriptions, procedures, recommendations and qualified claims,
including off-target content. Stage B must credit genuinely supported portions,
without inventing support for the rest.

### Descriptive facts are incorrectly treated as non-substantive

Cell `ab28bea4028bc50c3380`, about AliExpress wholesale pricing, says:

> lower per-unit prices unlocked at higher quantities.

Nemotron returns an empty unusable map and explains:

> The content consists solely of descriptive statements about AliExpress wholesale
> pricing without any evaluative, prescriptive, or absence-based claims.

A descriptive statement is an assertion under v4. Neither evaluation nor a
recommendation is required. The model substitutes a different definition of
substantive content for the frozen task.

### Procedures and recommendations disappear for being procedures

Cell `ff36fa6c503732eccfcc` supplies six care-coordination setup steps, including
creating an organization profile, importing patients, configuring access controls,
defining care plans and training staff. Nemotron calls this:

> only procedural steps and a note about verifying platform types

and claims no substantive assertion or recommendation can be mapped. These steps
are precisely the content v4's procedure and recommendation roles cover. Whether
the sources establish them is a later, separate judgment.

The same pattern affects investment-transfer instructions, governance policies,
Trello-selection criteria and pharmacy-software features. In pharmacy cell
`166ffa4bb37c4f872e9a`, Gemma previously omitted the second feature recommendation;
Nemotron discards the whole answer. Neither outcome satisfies map fidelity.

There is one local improvement: Nemotron retains both sentences of customer-service
cell `288e52a73496c7f39141`, while Gemma excludes the substantive planning sentence
as `non_substantive`. That improvement does not extend to Nemotron's source
judgments or the rest of the sample.

### Request fulfilment improperly controls map eligibility

Cell `1a3e9fa35cb066d65473` describes English-course providers and scheduling.
Nemotron says the answer:

> fails to address the request's core requirement of comparison

That may be relevant to the separate fulfilment task, but it cannot erase
assertions about the British Council, Coursera or tutoring platforms. V4
explicitly retains content in imperfect and off-target answers.

Mixed absence/narration answers also remain mappable. For example,
`f9c5c4e1940622aa3acb` both denies the presence of deployment instructions and
describes price-optimization models. Nemotron's empty map loses the distinction.

### Partial support becomes an unjustified reason to withhold all grades

The only ready map belongs to customer-service cell `288e52a73496c7f39141`.
Its answer recommends considering Zendesk, Freshdesk or DevRev and planning
through research, workflow assessment and a trial/demo. All seven source calls
return `uncertain`, all 14 findings are labeled `partial`, and no grade is given.

One explanation states:

> Both central claims are only partially supported; no claim is fully supported.
> ... the support distinction prevents a defensible grade.

V4 explicitly permits grading partial support according to the importance of
the supported portion. Missing full support is not by itself unresolved
uncertainty. A source can also receive zero when it supports no substantive
content. The model never makes either decision in this cell.

### Some source findings invent content or contradict their own notes

For dependency `si-v4-dependency-7a757f2a4866a7ca95d2abba`, the ProProfs record
names SolarWinds Service Desk. Nemotron says:

> The source lists Zendesk, Freshdesk, and DevRev as examples

None of those three names occurs in the complete supplied title/text. The
PressOne record similarly contains none of the recommended tool names, while
Nemotron claims it lists Zendesk and Freshdesk. Leafworks mentions Freshdesk;
the model also attributes Zendesk and DevRev to it. These are errors even though
the final grade remains null.

All seven findings about the research/workflow/trial procedure claim partial
support, although their supplied records do not establish that procedure.
The PressOne finding itself says:

> It supports no part of the procedural recommendation.

That explanation contradicts its `partial` label. Twelve findings cite only
`title1`; two cite only `text1`. Valid passage IDs establish structural validity,
not that the cited passage supports the proposition or explanation.

Several complete source records do support narrower tool-choice content. That
does not validate invented additional tool names or unsupported procedural
findings. Even where a partial relation is defensible using the full source,
the selected witness and explanation must still support that finding. These
fourteen findings come from one answer and are not independent cells.

### Validation catches one defect, but accepted abstention hides another

Social-API cell `ebd04d863dcd705a9689` repeats identical claim spans in both map
attempts and is correctly rejected. Its seven source calls are blocked.

Threat-intelligence cell `779fdfc78b624ede0b8e` initially misses the final word
`accuracy.` in a span and produces many tiny overlapping claims. On feedback,
Nemotron returns `unusable`, saying the missing word prevents faithful mapping,
instead of repairing its selections. The resulting three blocked source tasks
do not count as an inference failure because the abstention satisfies the schema.

The local validator reproduces all 19 retained map parses and all seven source
parses, and reproduces the rejected map's duplicate-span error. Thus the export
and parser are consistent with the saved responses. Semantic review is still
needed for the accepted responses, particularly empty unusable maps and null
grades.

The protocol auditor additionally reproduces all three failed validation
attempts and both corrective prompt hashes. The longest saved completion is
2,811 tokens, below the 4,096-token cap. Increasing that cap is not supported as
an explanation or remedy for the observed refusals.

## Timing and resource use

| Measurement | Observed |
|---|---:|
| Warm map/source phase | 187.02 seconds, 3.12 minutes |
| Warm node-hours / GPU-hours | 0.05195 / 0.20780 |
| Serving/trial wrapper elapsed | 610.09 seconds, 10.17 minutes |
| Slurm allocation elapsed | 946 seconds, 15 minutes 46 seconds |
| Allocated node-hours / GPU-hours | 0.26278 / 1.05111 |

The exported accounting record marks job 5177346 `FAILED`, with one node and
four GPUs. These are returned historical accounting records. Wrapper elapsed
includes serving startup and the trial; allocation time additionally includes
time outside that measured wrapper. Overlapping request durations must not be
summed into node-hours.

The short warm time mostly reflects empty map responses and 100 skipped source
calls. It cannot establish a speed advantage per useful cell. Cost per completed
cell is undefined at zero completions, and no credible full-corpus runtime can
be projected from this run. The matched Gemma subset shared its warm phase with
other cells, so an exact wall-clock ratio for these 20 cells is unavailable.

## Acceptance and next decision

Eligible-source completion is 0/107, against a target of at least 99%. The
observed map and source defects independently defeat the substantive goals.
Ordinal grade accuracy cannot be estimated because no numeric grades were
returned. This single pass supplies no Nemotron repeats, constructed controls,
list rotations or fixed-map source comparison. Those unperformed checks are
unestablished, not measured zero accuracy or stability.

Keep this Nemotron configuration out of production. The evidence does not
justify replacing Gemma to save time, and Gemma remains unaccepted on semantic
grounds documented in the earlier review. Any later Nemotron repair should first
demonstrate faithful mapping of descriptions and procedures, grounded source
findings and correct treatment of partial support on a bounded matched sample.
This review changes no prompts, results or model settings.

## Reviewer artifacts

Three GPT-6 Astra agents completed the requested review at maximum effort. Two
reviewed disjoint ten-cell sets with 56 and 51 sources, freezing initial map
feasibility and source-support expectations before seeing either model's
outputs. Both sets contain only mappable answers under the intended construct.
A third audited all twenty map outcomes, all seven executed source calls,
validation behavior and cost. Each initial source expectation has one blind
reviewer, who also sees the other supplied sources for that cell.

The parent verified all twenty cell identities, all 107 source identities and
content hashes, 254 compared model-outcome records and 147 non-null raw-output
hashes against the saved evidence. Both blind freezes remain byte-identical.
The hashes, coverage and input checks are recorded in
[the validation receipt](si-v4-nemotron-semantic-review-20261004.validation.json).

Per-cell/source reports, the blind freezes and the independent protocol audit
are in `/Users/valerianfourel/Downloads/nemotron-v4-review-20261004`.
This is an exploratory model-assisted diagnosis, not the formal independent
Astra/Sol acceptance process or human gold. No scientific outputs were changed,
and no inference or new allocation was run for this review.
