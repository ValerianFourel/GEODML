# Experiment V2: 500-prompt agentic-search pilot

## Purpose

Experiment V2 studies how search behavior changes across the
information-to-action-readiness axis. Unlike Experiment V1, the generator chooses
search queries and sees only the snippets returned by its search path.

The pilot asks:

> When the same prompt enters different search methods, engines, and evidence
> conditions, how do the search path, evidence ranking, and final answer change?

The pilot uses frozen search snapshots. It approximates agentic snippet search but
does not browse live pages.

## Prompt selection

The code deterministically selects 500 prompts from the audited prompt population.
It balances selection across the observed readiness-axis bins. The same prompt IDs
enter every planned method, engine, condition, and generator cell.

The readiness bins stratify the pilot sample. They do not replace the assigned
readiness variable.

## Complete task matrix

The original pilot crosses these factors:

| Factor | Levels |
| --- | --- |
| Prompt | 500 axis-balanced prompts |
| Generator | Qwen3.8 and Llama 4 Scout |
| Search method | Parallel Expansion and Reactive Snippet Loop |
| Search engine | DuckDuckGo and SearXNG |
| Evidence condition | Natural, Ablated, and Shuffled |

Every prompt enters both search methods, both engines, and all three conditions for
each generator model.

```text
500 prompts
* 2 methods
* 2 engines
* 3 conditions
= 6,000 generator cells per model

6,000 Qwen cells + 6,000 Llama cells
= 12,000 generator cells
```

This count describes the frozen plan. Scientific completion requires a valid saved
outcome for every planned cell.

## Search evidence

DuckDuckGo and SearXNG are separate frozen snapshot blocks. An agent-generated query
retrieves at most 20 matching snapshot rows from its assigned engine.

Each row contains:

- a URL;
- a title;
- a search-result snippet.

The agent does not open URLs, fetch full page text, parse a page, or follow links.
The pilot therefore studies snippet retrieval rather than full web browsing.

## Shared evidence compactor

Both methods use the pinned `BAAI/bge-reranker-v2-m3` cross-encoder. The compactor
scores each snippet against the active information need and keeps a bounded set.

The cross-encoder performs evidence selection. It does not verify factual accuracy,
produce the final answer, or replace the generator's final ranking.

The trace preserves every score and selected snippet.

## Search method 1: Parallel Expansion

Parallel Expansion chooses all search directions before it sees any evidence.

1. Qwen or Llama receives the unchanged generated prompt.
2. The model returns exactly three distinct search queries.
3. The harness submits all three queries concurrently to one assigned engine.
4. Each query returns at most 20 snippets.
5. The harness concatenates the responses in generated-query order.
6. It removes later exact-URL duplicates.
7. It applies the assigned evidence condition.
8. The cross-encoder scores the remaining snippets against the original prompt.
9. It retains the top seven snippets.
10. The generator returns one ranking and one answer.

The query-generation response is:

```json
{
  "queries": ["query one", "query two", "query three"]
}
```

The final response is:

```json
{
  "ranking": ["S2", "S1", "S5"],
  "answer": "A model-written answer based only on the retained snippets."
}
```

Parallel Expansion always makes three search calls when query generation succeeds.
It uses two logical LLM calls before schema retries.

## Search method 2: Reactive Snippet Loop

The Reactive Snippet Loop chooses later queries after it sees earlier evidence.

1. The first model call must request a search.
2. The assigned engine returns at most 20 snippets.
3. The harness applies the evidence condition.
4. The cross-encoder scores the snippets against the generated query.
5. It retains the top three snippets as observations.
6. The next model call sees the original request and all retained observations.
7. The model either searches again or finishes.
8. The loop permits at most three searches.
9. If the third iteration searches, a fourth LLM call forces completion.

A search action is:

```json
{
  "action": "search",
  "query": "a refined query based on earlier observations"
}
```

A finish action is:

```json
{
  "action": "finish",
  "ranking": ["S4", "S1", "S5"],
  "answer": "A model-written answer based on the accumulated observations."
}
```

The reactive method performs one to three searches. It retains at most three
snippets per iteration and at most nine snippets in total. It uses two to four
logical LLM calls before schema retries.

## Evidence conditions

Each prompt, method, engine, and generator combination runs under all three
conditions.

### Natural

Natural preserves the retrieved evidence.

### Ablated

The preparation stage freezes one target URL for each prompt and engine. The harness
removes that URL when the search path retrieves it.

If the Natural search path does not retrieve the target, the ablation contrast is
undefined for that path. The analysis must not report it as a zero effect.

### Shuffled

Shuffled deterministically permutes the retrieved snippets while preserving their
membership.

The original 500-prompt implementation applies this shuffle before cross-encoder
compaction. The cross-encoder then sorts snippets by score, which can erase the
presentation-order change. The original pilot therefore does not provide a clean
test of snippet-position bias. A later production protocol needs separate membership
and post-compaction presentation hooks.

## Generator output from each cell

Each successful Qwen or Llama cell saves:

- the final ranked evidence URLs;
- the final answer, limited to 1,200 characters;
- the final snippets shown to the generator;
- the number of search calls;
- the method, engine, condition, prompt, model, and revision;
- the immutable trace path and SHA-256 hash;
- validation, retry, recovery, and runtime diagnostics.

The trace contains every recorded LLM request and response, generated search query,
search response, condition transformation, cross-encoder score, selected snippet,
and reactive observation.

### Fictional generator example

The following record demonstrates the format. It is not an observed pilot result.

```json
{
  "method": "Parallel-Expansion-v1",
  "engine": "duckduckgo",
  "condition": "natural",
  "ranking": [
    "https://example.org/comparison",
    "https://vendor.example/docs",
    "https://example.net/implementation-guide"
  ],
  "answer": "The comparison supports option A for this request, while the implementation guide explains the deployment steps.",
  "final_snippet_count": 7,
  "search_count": 3,
  "trace_sha256": "EXAMPLE_ONLY"
}
```

## Nemotron judgment

The current original-500 plan uses
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` as the bulk judge. It creates one
judge task for every completed generator cell after both generator inventories pass
the completion barrier.

For a complete two-generator pilot, the judge queue contains 12,000 tasks.

### Recorded-conversation input

The recorded-conversation protocol gives Nemotron:

- the original user request;
- every recorded logical generator prompt and response;
- schema failures and controller repairs;
- the compacted snippets shown to the generator;
- reactive tool observations;
- the final generator ranking;
- the final answer.

Nemotron does not receive explicit generator, method, engine, or condition labels.
It can still infer parts of the workflow from the recorded conversation. This mode
is therefore not ranking-blinded.

The judge does not see hidden reasoning, discarded pre-compaction results,
pre-ablation evidence, full webpage text, or every low-level transport retry.

### Nemotron output

Nemotron returns:

- `request_fulfillment`, an integer from 1 to 5;
- `evidence_grounding`, an integer from 1 to 5;
- `ideal_relevance_ranking`, which orders every supplied evidence item;
- `realized_support_ranking`, which contains only evidence that materially supports
  the answer;
- `unsupported_claim_count`;
- `judge_confidence`, an integer from 1 to 5.

The ideal ranking ignores the generator's answer and asks how the available evidence
should rank for the request. The realized-support ranking asks which evidence
actually supports the generated answer.

### Fictional Nemotron example

This example matches the schema but is not an observed judgment.

```json
{
  "request_fulfillment": 4,
  "evidence_grounding": 3,
  "ideal_relevance_ranking": ["E2", "E1", "E4", "E3"],
  "realized_support_ranking": [
    {
      "evidence_id": "E1",
      "use_score": 5
    },
    {
      "evidence_id": "E2",
      "use_score": 4
    }
  ],
  "unsupported_claim_count": 1,
  "judge_confidence": 4
}
```

Judge evidence IDs use an independent order. A private mapping joins the judge IDs,
generator evidence IDs, and source URLs during analysis.

## Three ranking views

A judged cell can contain three distinct rankings.

| Ranking | Producer | Meaning |
| --- | --- | --- |
| Generator ranking | Qwen or Llama | Evidence that the generator chose to rank after its search process |
| Ideal relevance ranking | Nemotron | How all supplied evidence should rank for the user request |
| Realized support ranking | Nemotron | Evidence that materially supports the generated answer |

The analysis must keep ideal relevance and realized support separate.

## Nemotron context capacity

The first serving profile used `max_model_len=73728`. The exact-tokenizer preflight
measured all 12,000 recorded-conversation tasks and found:

| Quantity | Tokens |
| --- | ---: |
| Original serving limit | 73,728 |
| Largest required task | 81,871 |
| Reserved output allowance | 2,048 |
| Tasks exceeding the original limit | 3 |

The required-token count includes the rendered judge prompt, the chat template, the
generation prefix, and the output allowance. The checker uses `truncation=False`.

The pinned Nemotron configuration declares a native 262,144-token context. The model
therefore has enough native capacity. The original vLLM setting caused the failure.

The automatic preparation path rounds the largest requirement to the next
4,096-token boundary:

```text
ceil(81,871 / 4,096) * 4,096 = 81,920
```

A run with `max_model_len=81920` can admit every frozen task without truncation. It
has only 49 tokens of measured headroom, so any changed prompt or tokenizer requires
a new full-queue measurement. A larger serving window also requires a fresh runtime
estimate and allocation approval.

## Alternative long-context judges

Nemotron remains the preferred primary judge because it is already cached, pinned,
integrated, and tested with the structured schema.

Two alternatives fit the longest measured task:

| Model | Parameter structure | Native context | Main trade-off |
| --- | --- | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | 30.5B total, 3.3B active | 262,144 | Easy vLLM path, but shares a model family with the Qwen generator |
| Mistral Small 3.1 24B | 24B dense | 131,072 | More independent, but denser and previously problematic in this environment |

An alternative judge changes the model and protocol identity. It requires a new
serving pilot, context measurement, task plan, output directory, runtime estimate,
and allocation approval. Judge-model agreement should be checked on a frozen,
stratified validation subset rather than mixing outputs silently.

## Aggregate pilot outcomes

The final pilot report should stratify results by:

- assigned readiness;
- generator model;
- search method;
- search engine;
- evidence condition.

Useful outcomes include:

- generated-query diversity;
- search depth and final evidence count;
- URL and domain diversity;
- generator-ranking agreement with Nemotron;
- request-fulfillment and grounding scores;
- unsupported-claim rates;
- changes caused by target removal;
- failure, recovery, and truncation flags.

Artifact completion alone does not establish answer quality or a scientific effect.
The earlier 24-task and 696-task judge outputs are separate smoke or partial runs.
They do not equal complete 12,000-cell judging.

## Relationship to Experiment V1

| Experiment V1 | Experiment V2 |
| --- | --- |
| Uses fixed full-text document sets | Uses agent-selected search snippets |
| Disables search during inference | Lets the generator choose search queries |
| Runs separate reranking and answer requests | Returns ranking and answer together after search |
| Isolates prompt effects with fixed candidates | Measures prompt effects on the complete search path |
| Uses a blinded answer judge | Uses recorded-conversation judging for the original 500 plan |

The experiments answer different questions. Their task counts, rankings, and judge
outputs must not be pooled without an explicit experiment indicator.

## Frozen Nemotron queue status

The original 500-prompt generation stage is complete and read-only. It supplies
12,000 recorded-conversation Nemotron tasks from the fixed product, prompt,
generator, search-method, search-engine, and evidence-condition matrix. A
continuation may fill missing judge claims, but it may not rebuild generation,
change a conversation, alter the task order, or use a different claim namespace.

The allocation ending on 2026-09-22 produced this checkpoint:

| Field | Recorded value |
| --- | ---: |
| Expected Nemotron judgments | 12,000 |
| Validated completed claims | 1,298 |
| Remaining claims | 10,702 |
| Queue completion | 10.82% |
| Failures written | 0 |
| Worker runtime | 47m56.8s |
| Observed worker throughput | 1,624 judgments/hour |

The allocation ended normally at its deadline. The API served authenticated
requests successfully, the worker checkpointed every completed task, and the
server shut down after admission stopped. Shutdown warnings about leaked Python
semaphores and shared-memory objects occurred after result checkpointing and do
not represent failed judgments. The run manifest remains
`scientific_result=false` and `eligible_for_analysis=false` while any expected
claim is missing or terminally failed.

## Continuation lifecycle

The manager treats preparation, scheduler acceptance, execution, and queue
completion as different states:

```text
frozen source allocation
        |
        v
prepared continuation
        |
        v
submission intent written
        |
        +--> submission failed or uncertain --> manual scheduler audit
        |
        v
scheduler accepted
        |
        v
worker validates allocation and frozen identity
        |
        v
checkpointed at deadline or complete
        |
        +--> missing work remains --> fresh estimate and approval
        |
        v
all 12,000 claims validated
```

`status` reports the evidence available in local receipts. It does not infer a
live Slurm state. `preparation_status=prepared` means the frozen launch exists.
`submission_status=accepted` means `sbatch` returned a job identifier. The
`failed` and `uncertain` submission states stay terminal for automated submission
and require manual inspection. A successful Slurm allocation is not the same as
a complete queue.

## Identity and ownership rules

Every continuation reuses the original plan, context budget, task bytes, model
revision, output schema, and shared Nemotron claim root. It records a new launch
receipt and an immutable link from its predecessor. The first allocation remains
the ownership root for later continuations. Stable task IDs ensure that already
completed claims are validated and reused rather than regenerated.

The exact serving contract includes the model revision, bulk role, disabled
thinking, 2,048 output tokens, maximum three attempts, four concurrent requests,
and `request_timeout=120.0`. Numeric type is part of the serialized identity.
Commit `aa9aa7173896a66f796ec7cce7e447c357d89f1a` corrected the earlier integer and
floating-point timeout mismatch. A continuation with a changed scientific or
serving contract is rejected.

Preparation requires one exact terminal predecessor allocation from `sacct`, no
busy claims, no unresolved legacy failure journal, and at least one eligible task
in every requested partition. Submission writes an exclusive intent before
calling `sbatch`; both intent and result receipts are write-once. An orphaned or
uncertain receipt therefore stops automated resubmission instead of risking a
duplicate allocation.

## Measured continuation budget

The 2026-09-22 worker completed 1,298 tasks in 0.799 hours. At the observed rate,
10,702 remaining tasks require about 6h35m of model-serving time. The same
allocation spent about 10m10s on environment checks, server startup, model load,
and warmup before worker inference. A one-allocation point estimate is therefore
about 6h45m before a small drain allowance.

Task-length variation makes 6h48m to 8h a realistic completion range. The
proposed continuation is one node with four GH200 GPUs, 32 CPUs, 512 GiB memory,
and an eight-hour wall time. Its maximum charge is 32 GPU-hours. The additional
margin covers the observed cold start, the configured two-minute admission
margin, the 45-second cleanup margin, and slower remaining conversations. This
proposal is not approved merely by appearing in this document. The launch
receipt must record Valerian's explicit approval before preparation or
submission.

A four-hour resumable allocation limits exposure to 16 GPU-hours and should
complete roughly 6,100 tasks after startup at the measured rate. A two-hour
allocation limits exposure to 8 GPU-hours and should complete roughly 2,900.
Repeated one-hour allocations are supported, but repeated cold starts make them
less credit-efficient. Every later continuation still requires a fresh estimate
from the then-current remainder and a new wall-time approval.

## Completion reporting

Final reporting validates claim envelopes against the frozen task identity and
reports completed, missing, busy, and terminal-failed counts. It also checks that
the product, prompt, generator, search-method, search-engine, and evidence-condition
strata sum to the expected 12,000 tasks. `bulk_complete=true` requires 12,000
validated successful claims. HTTP 200 responses, a completed Slurm job, or an
empty failures file cannot substitute for that invariant.

Validation, adjudication, and scientific analysis remain later milestones. This
continuation only finishes bulk Nemotron judging. It must not infer scientific
effects from partial results, smoke tests, or synthetic fixtures.

## Implementation references

- [Agentic retrieval protocol](agentic_search_retrieval_protocol.md)
- [Inference-wave design](agentic_inference_waves.md)
- [Task matrix builder](../interpretability/pipeline/agentic_generation_tasks.py)
- [Agentic state machines](../interpretability/pipeline/agentic_search.py)
- [Judge contracts](../interpretability/pipeline/agentic_judging.py)
- [Original-500 judge manager](../scripts/manage_agentic_pilot_judging.py)
- [Original-500 continuation runbook](original_pilot_nemotron_continuation.md)
- [Context-budget checker](../scripts/check_agentic_judge_context.py)

## External model references

- [NVIDIA Nemotron 3 Nano model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Qwen3-30B-A3B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Mistral Small 3.1 configuration](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json)
