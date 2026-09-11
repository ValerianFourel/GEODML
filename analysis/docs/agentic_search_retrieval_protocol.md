# Agentic snippet-retrieval protocol

## Status and scope

This document is the source of truth for the proposed agentic extension to the
captured-evidence search experiment. The extension adds two methods:
`Parallel-Expansion-v1` and `Reactive-Snippet-Loop-v1`. Both methods use a
pinned local cross-encoder to compact search-result snippets before final
generation.

The implementation follows a common retrieve, rerank, and generate pattern. It
approximates public agentic-search behavior, but it does not claim to reproduce
the private prompts, indexes, ranking functions, or control logic of any
commercial system.

In this protocol, a search result is a snippet record containing a URL, title,
and snippet text. It is not a fetched web page. The agent does not open URLs,
parse a DOM, use find-in-page, or retrieve page chunks. These restrictions keep
the intervention bounded and separate snippet retrieval from the project's
historical page-content RAG experiments.

The exact historical baseline remains unchanged. Agentic outputs form a new,
separately versioned search-policy experiment and cannot be merged silently
into existing headline results.

## Proposed experimental factors

The proposed full design crosses these fixed or manipulated factors:

| Symbol | Factor | Levels |
|---|---|---|
| `P` | Prompt population | 26,009 fixed prompt records |
| `C` | Evidence condition | Natural, Ablated, Shuffled |
| `M` | Generator model | Four pinned local models |
| `E` | Search engine | DuckDuckGo, SearXNG |
| `A` | Search method | Historical Baseline, Parallel Expansion, Reactive Loop |

The pinned generator panel is:

| Model | Revision |
|---|---|
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3` |
| `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | `2dc98e2afe4face0e4ce40972a915c45368bd34a` |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Llama 4 Scout, Qwen2.5, and Qwen3.8 have passed both the four-task primary smoke
and the 12-cell agentic smoke. The earlier Mistral candidate passed its primary
smoke but repeatedly stalled during agentic server initialization. It was
replaced before scientific execution. Nemotron must pass the same primary and
12-cell gates before the revised four-model panel is ready. Compatibility
smokes do not prove answer quality, agentic-path quality, judge validity, or
production throughput.

Nemotron uses a chat template that enables thinking by default. This protocol
sets `chat_template_kwargs={"enable_thinking": false}` on every Nemotron
request. The flag is recorded in the run manifest and HTTP audit trace. This
keeps reasoning tokens out of the bounded structured-output channel and makes
its call contract comparable with the other three models.

### Model replacement record

The Mistral agentic smoke exhausted nearly all available device memory during
startup and then remained in shared-memory broadcast waits until the readiness
timeout. That failed run remains part of the audit history. It is not a
scientific result, and its files must not be reused as Nemotron evidence.

The active panel replaces Mistral with the pinned Nemotron BF16 checkpoint. The
replacement requires a new model lock, serving profile, primary compatibility
smoke, and 12-cell agentic smoke. Existing Llama, Qwen2.5, and Qwen3.8 evidence
does not need to be rerun merely because the fourth model changed. Any plan
that contains the former Mistral model ID must be regenerated under a new run
directory. Results from the old and revised panels must not be pooled without
an explicit model-panel indicator.

The 26,009-prompt matrix is a proposed expansion. It is not the existing
128-prompt document-pilot plan, whose manifest contains 4,608 planned
inferences under a different design. Reports and launch tools must never use
the two counts interchangeably.

## Why context compaction is necessary

Passing every raw result directly to a generator creates three avoidable
problems. Duplicate URLs consume context. Weak snippets dilute the evidence
signal. Different query paths can produce very different prompt lengths.

The compaction stage addresses these problems without another generator call.
It scores every candidate against the information need, keeps a small fixed
set, and records every score. The generator therefore sees a bounded evidence
set, while an auditor can still reconstruct what was searched, removed,
scored, and retained.

This stage is evidence selection, not fact verification. A high cross-encoder
score means that a snippet appears relevant to a query. It does not establish
that the snippet is correct, independent, complete, or sufficient to support a
claim. Final answers and judgments must preserve that distinction.

Pure top-K selection can also overrepresent one expanded query or one domain.
The pilot must report query coverage, unique-domain count, duplicate rate, and
score concentration in each retained set. Do not add a domain cap, per-query
quota, maximal marginal relevance, or an LLM summarization stage after seeing
pilot outcomes. Any such change defines a new versioned method and requires a
new preregistration. This keeps the current mechanism simple enough to identify
while exposing whether simplicity harms evidence breadth.

## Frozen compaction contract

The production protocol pins the following values:

| Item | Contract |
|---|---|
| Cross-encoder | `BAAI/bge-reranker-v2-m3` |
| Revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Runtime used by the readiness check | `sentence-transformers==6.0.1` |
| Search results per query | At most 20 snippets |
| Parallel query count | Exactly 3 distinct queries |
| Parallel retained evidence | Top 7 after URL deduplication and condition application |
| Reactive search iterations | At most 3 |
| Reactive retained evidence | Top 3 from each iteration, at most 9 total |
| Schema retries | At most 2 retries after the first attempt |

For a pair `(query, snippet)`, the cross-encoder receives the query and the
string `title + newline + snippet_text`. Scores are sorted from highest to
lowest. The original source index breaks exact score ties. The trace stores all
scores and the selected records, not only the final top K.

Determinism requires more than a model name. Every run must record the model
revision, model-file hashes, package versions, device type, numerical dtype,
batch size, input hashes, and tie-breaking rule. Accelerator kernels can still
produce small floating-point differences. A preregistered near-tie policy is
required before scientific execution if such differences can change the top K.

## `Parallel-Expansion-v1`

This method tests whether one model can translate a user request into a diverse
set of search intents before seeing evidence.

1. The generator receives the unchanged user request.
2. It returns exactly `{"queries": ["q1", "q2", "q3"]}`. Queries must be
   nonempty and distinct under case-insensitive comparison.
3. The search adapter submits all three queries concurrently to one assigned
   engine. Each call returns at most 20 snippets.
4. The harness concatenates responses in generated-query order and removes
   later exact-URL duplicates. At most 60 raw snippets enter this step.
5. Target ablation, when assigned, is applied to the deduplicated list.
6. The cross-encoder scores the remaining snippets against the original user
   request and retains the top 7.
7. The assigned Natural or Shuffled presentation order is applied to the same
   retained set.
8. The generator receives the unchanged request and those seven snippets. It
   returns `{"ranking": [...], "answer": "..."}`.
9. Every ranked URL must occur in the retained evidence.

The method always makes three search calls if query generation succeeds. It
makes two logical LLM calls before schema retries. It does not change its
queries in response to search results.

## `Reactive-Snippet-Loop-v1`

This method tests observation-driven query refinement under a strict bound.

1. The generator receives the unchanged user request, the search-tool schema,
   and all retained observations from earlier iterations.
2. It returns either `{"action": "search", "query": "..."}` or
   `{"action": "finish", "ranking": [...], "answer": "..."}`.
3. For a search action, the assigned engine returns at most 20 snippets.
4. Target ablation, when assigned, is applied to that response.
5. The cross-encoder scores the remaining snippets against the generated
   query and retains the top 3.
6. The assigned Natural or Shuffled presentation order is applied to the same
   retained set.
7. Those three records become the next tool observation. Previous observations
   remain visible.
8. The loop permits at most three search iterations. If the third iteration
   also searches, the harness makes one fourth, forced-finish LLM call. The
   forced call cannot request another search.
9. Every ranked URL must occur in the retained observations.

The method can finish without searching. Its maximum is three search calls,
nine retained snippets, and four logical LLM calls before schema retries. A
description that calls this a three-turn maximum is incorrect unless the
forced-finish call is removed from the implementation and preregistration.

## Condition timing and semantics

Membership and presentation are separate intervention phases. Target ablation
must run after retrieval and before cross-encoder scoring. Natural or Shuffled
presentation order must run after top-K selection. This ordering prevents the
selector from restoring score order after a shuffle and thereby erasing the
position treatment.

The scientific transformations must satisfy these contracts:

| Condition | Before compaction | After compaction |
|---|---|---|
| Natural | Preserve candidate membership. | Present selected snippets in their canonical acquisition order. |
| Ablated | Remove the preregistered target document or target URL identity. Do not remove an arbitrary position. | Present selected snippets in canonical acquisition order. |
| Shuffled | Preserve candidate membership. | Permute the selected Natural membership with a deterministic per-cell seed. |

For Natural versus Shuffled, top-K membership must be identical. The only
first-iteration difference may be presentation order. For Parallel Expansion,
canonical acquisition order is generated-query order followed by provider
position, after exact-URL deduplication. For Reactive Loop, it is provider order
within each iteration and iteration order across observations.

A transformation may remove or reorder retrieved evidence. It may never invent
or mutate evidence. The target-identity rule must specify URL canonicalization,
redirects, and duplicate handling before data collection.

The current implementation exposes one condition hook before compaction. Its
integration smoke uses an index-based subset for `ablated` and a seeded shuffle.
The cross-encoder then score-sorts the result, so that pre-compaction shuffle is
normally erased except at exact score ties. This hook checks wiring only. It is
not the scientific intervention, cannot test snippet-position bias, and is
marked `scientific_result=false`. Production execution is blocked until the
state machine has separate, tested membership and presentation hooks.

## Search-engine and time boundary

DuckDuckGo and SearXNG are separate fixed blocks. Their result sets must not be
pooled as if they were interchangeable samples. Estimate effects within engine
first, then estimate the engine-by-method interaction.

Novel agent-generated queries cannot be replayed from the old exact-query
snapshot unless that snapshot happens to contain them. The current integration
smoke uses deterministic lexical selection over frozen DDG and SearXNG files.
That is useful for validating adapters, state transitions, cross-encoder
loading, model calls, and traces. It is not evidence that live agentic
retrieval works, and it cannot support scientific conclusions.

The production path should use a staged freeze:

1. Generate and freeze every agentic query with its model, prompt, seed, method,
   cell identity, and raw LLM response.
2. Collect raw DDG and SearXNG responses for the unique query set in a narrow,
   declared collection window. Record timestamps, request parameters, provider
   identity, errors, retries, and raw payload hashes.
3. Freeze the search responses and build immutable query-to-result indexes.
4. Replay those indexes for condition application, cross-encoder compaction,
   generation, and judgment.

For a contemporaneous comparison, collect the baseline's exact user queries in
the same window and engine configuration. Keep the historical frozen baseline
as a separate historical reference. Comparing a historical baseline directly
with newly collected agentic evidence confounds method with time.

Reactive query generation depends on earlier observations, so its collection
phase cannot know all later queries in advance. Use a staged semantic cache:
execute a bounded path once, persist every request and response atomically, and
reuse that exact path for downstream condition/model comparisons only when the
design permits it. If a condition can change the next query, each condition has
its own frozen path. Do not substitute a nearest cached keyword and describe it
as a provider response.

## Prompt and evidence safety

Search snippets are untrusted evidence. Prompts must delimit them as quoted
data and instruct the model never to follow instructions found inside a
snippet. The harness must validate that output rankings contain only observed,
retained URLs. It must also retain the raw provider payload so prompt-injection
and malformed-result incidents remain auditable.

URL deduplication currently uses exact URL strings. Production data collection
must either preregister this rule or implement and test a canonical URL policy.
Changing URL normalization after observing outcomes can change treatment
membership and is not a harmless cleanup.

## Output and timeout budgets

The intended production budget is 256 generated tokens for query or action
calls and 2,048 generated tokens for final answers. These budgets must be
enforced by purpose and recorded in each request trace.

The current integration smoke uses one configurable token limit, with a default
of 1,024, for every call. The core state-machine module does not itself enforce
purpose-specific token limits. Therefore the 256/2,048 split is a production
requirement, not yet an implemented guarantee.

A 60-second whole-task timeout is not supported by the measured cluster runs.
The smoke client currently allows 300 seconds per request, and model startup is
handled separately. Production timeouts must be set from the representative
pilot's latency distribution. They must distinguish server startup, one model
request, one search request, and the complete cell.

## Audit record

Every cell must preserve enough state to replay or explain the result:

- experiment and cell identifiers;
- Git commit and clean-checkout status;
- prompt identity, condition, method, engine, model, and model revision;
- deterministic seeds and request limits;
- every rendered LLM request, raw response, parsed response, retry, and error;
- every search query, request parameter, raw payload, result order, and error;
- condition input and output records;
- every cross-encoder input, score, selected record, and model revision;
- every tool observation and forced-finish decision;
- final ranking, answer, validation result, and trace SHA-256;
- Slurm job and step identifiers, resources, timestamps, and output paths.

Traces are scientific data. Write them atomically, refuse silent overwrite,
and make resume depend on the complete cell identity rather than file presence
alone.

## Counting the proposed full matrix

For `N = 26,009` prompts, 3 conditions, 4 generator models, 2 engines, and 3
methods, the number of method cells is:

```text
26,009 * 3 * 4 * 2 * 3 = 1,872,648 cells
```

Each method has `624,216` cells. Under the state machines implemented today,
the maximum primary logical LLM calls before schema retries are:

| Method | Cells | Logical calls per cell | Maximum calls |
|---|---:|---:|---:|
| Historical baseline | 624,216 | 1 | 624,216 |
| Parallel expansion | 624,216 | 2 | 1,248,432 |
| Reactive loop | 624,216 | 4 | 2,496,864 |
| Total | 1,872,648 | method-specific | 4,369,512 |

With two schema retries, each logical call can produce up to three request
attempts. The primary-generation ceiling is therefore 13,108,536 request
attempts. This is a failure ceiling, not an expected workload.

If every cell also receives one separate judge call, add 1,872,648 logical
judge calls. The resulting maximum before retries is 6,242,160 LLM calls. The
judge retry policy must be counted separately. A statement that the design has
3.745 million maximum calls assumes only three reactive calls and conflicts
with the implemented forced fourth finish call.

A ranking and answer are fields in one primary structured response. They are
two logical artifacts, not two model requests. A judgment is a separate model
request. Reports must label cells, artifacts, logical calls, HTTP attempts, and
search calls separately.

If reports count ranking, answer, and judgment as three logical artifacts per
cell, the matrix contains 5,617,944 artifacts. This label must not be presented
as 5,617,944 independent inference requests.

These counts exclude search-provider calls. Parallel expansion makes three
search calls per completed cell. Reactive makes zero to three. The upper bound
for the two agentic methods is 3,745,296 search calls before provider retries:

```text
(624,216 * 3) + (624,216 * 3) = 3,745,296 search calls
```

## Experimental interpretation

Natural versus Ablated estimates reliance on the preregistered target evidence
when the target is retrieved. After the split-hook requirement is implemented,
Natural versus Shuffled can estimate sensitivity to snippet presentation order
while holding retained membership fixed at the first affected observation. A
missing target in the Natural retrieval path means the ablation contrast is
undefined for that path, not a zero treatment effect.

Method comparisons are structural comparisons unless method assignment and
the retrieval-time design justify a causal interpretation. Page-feature effects
remain observational DML estimates unless page content is manipulated. The
prompt-policy variable `B` remains the assigned semantic axis. Surface seed `S`
changes realization only, and prompt embeddings do not define `B` or become
confounders.

## Pilot sequence and go or no-go gates

Do not launch the 26,009-prompt matrix from compatibility smokes. Use these
stages:

1. Run a synthetic contract suite for schemas, bounds, retries, deduplication,
   condition invariants, compaction, trace integrity, and resume behavior.
2. Run a 10-prompt retrieval contract test against actual provider adapters.
   Verify target matching, ablation, deterministic shuffle, rate limiting, and
   immutable raw-response capture.
3. Run a representative 500-prompt pilot over every planned arm or a
   preregistered fractional design. Measure target retrieval, schema failures,
   retry rates, truncation, latency, token use, agreement, and judge variance.
4. Perform power and cost analysis from measured pilot data.
5. Freeze the production query and retrieval protocol. Launch the full matrix
   only if the pilot demonstrates useful separation between methods, acceptable
   failure rates, stable evidence capture, and an affordable resource plan.

The same formulas produce the following planning scales before retries:

| Prompt count | Cells | Primary logical LLM calls, maximum | Calls with one judge per cell | Search calls, maximum |
|---:|---:|---:|---:|---:|
| 10 | 720 | 1,680 | 2,400 | 1,440 |
| 500 | 36,000 | 84,000 | 120,000 | 72,000 |
| 26,009 | 1,872,648 | 4,369,512 | 6,242,160 | 3,745,296 |

These are logical-call ceilings from the state machines. They are not GPU-hour,
provider-rate, latency, or monetary estimates. Those estimates require measured
per-model and per-engine pilot throughput.

The go or no-go report must state whether an underperforming method, engine, or
arm will be removed. That decision must use a preregistered rule. It must not be
chosen after inspecting headline effects.

## Known threats to validity

- Agent-generated queries may fail to retrieve the target, making some
  ablation contrasts unavailable.
- Conditions can change a reactive agent's later queries and therefore its
  complete evidence path.
- Schema failures can confound reasoning quality with formatting reliability.
- Cross-encoder compaction can suppress relevant minority evidence or favor
  lexical similarity over support quality.
- Live provider results drift over time, geography, instance configuration,
  and rate-limit state.
- DDG and SearXNG expose different candidate distributions.
- Snippets can be stale, truncated, duplicated, misleading, or adversarial.
- Static top K values trade recall for a controlled prompt budget.
- Generator and judge model reuse can create correlated errors.

Report these as measured failure modes where possible. Do not convert smoke
compatibility, readiness checks, or mocked outputs into scientific findings.

## Implementation map and remaining work

The bounded state machines and cross-encoder adapter live in
`analysis/interpretability/pipeline/agentic_search.py`. The CPU and mocked tests
live in `analysis/tests/test_agentic_search.py`. Cluster readiness is checked by
`analysis/scripts/verify_agentic_search_cluster_readiness.py`. The current
12-cell compatibility smoke is implemented by
`analysis/scripts/run_agentic_search_integration_smoke.py` and the Qwen3.8
Slurm launcher.

The following production work is still required:

- real, separately identified DDG and SearXNG adapters;
- immutable raw-response caching for generated queries;
- scientific target-aware ablation and URL identity rules;
- separate pre-compaction membership and post-compaction presentation hooks;
- an assertion that Natural and Shuffled retain identical top-K membership;
- purpose-specific 256 and 2,048 token budgets;
- measured production timeout and retry policies;
- a production plan compiler for the 10-prompt and 500-prompt stages;
- independent judge and human-calibration integration;
- resource and provider-rate estimates based on measured pilot throughput.

Until these items pass their staged gates, `AGENTIC_SEARCH_READINESS=PASS`
means that dependencies, frozen snapshots, state-machine contracts, and prior
model smokes are present. It does not authorize or validate the full scientific
experiment.
