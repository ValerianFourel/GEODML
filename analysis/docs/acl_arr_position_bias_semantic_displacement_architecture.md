# ACL ARR experiment plan: position bias, semantic displacement, and document use

**Status:** design specification for the ACL ARR cycle

**Updated:** 2026-09-04

**Planned population:** 26,009 audited search prompts

**Compute:** one node with four NVIDIA GH200 GPUs, approximately 576 GB total HBM
**Serving:** vLLM with an OpenAI-compatible API

The implemented pipeline and file contracts are documented in
[`acl_arr_experiment_pipeline.md`](acl_arr_experiment_pipeline.md). The code
prepares tasks and runs CPU smoke checks. No production model inference has run.

## Executive summary

This experiment measures how document removal and document order affect an
LLM's stated ranking, generated answer, citations, and realized use of evidence.
Every prompt uses one frozen set of retrieved documents. The same prompt and
documents pass through four open-weight models under three paired input
conditions:

1. **Natural:** keep the search engine's frozen document order.
2. **Ablated:** remove one preregistered target document.
3. **Shuffled:** keep every document but apply one deterministic,
   counterbalanced permutation.

The study has two pipelines. Pipeline 1 asks each model to rank document IDs.
Pipeline 2 asks each model to answer with document-ID citations. A blinded judge
then estimates which documents the answer actually used and ranks their
contribution. We compare this realized-use ranking with Pipeline 1's stated
ranking.

The primary design therefore contains 312,000 model-condition observations per
pipeline:

```text
26,000 prompts x 4 models x 3 conditions = 312,000 observations
```

Across both pipelines, the panel produces 624,000 primary model outputs. If
every generated answer receives one judge evaluation, judging adds 312,000
requests, for 936,000 total inference requests before retries. The currently
audited population contains 26,009 prompts. If all 26,009 are frozen, the exact
counts become 312,108 per pipeline, 624,216 across both pipelines, and 936,324
including one judgment per generated answer.

## Research questions

### RQ1: Position bias

When document content is held fixed, how much does changing document order
change the model's ranking, citations, and answer?

### RQ2: Document ablation and displacement

When one target document is removed, which remaining documents replace it in
the ranking and in the generated answer? How much does the answer's content
move after that removal?

### RQ3: Stated ranking versus realized use

Does the ranking that a model states in Pipeline 1 agree with the evidence it
uses in Pipeline 2?

### RQ4: Model architecture and scale

Do the observed sensitivities differ across the two dense models and two
mixture-of-experts models in the panel?

The model comparison is descriptive. Four models form a balanced two-dense,
two-MoE panel, but model family, training data, total size, active size, and
architecture vary together. The study must not interpret a dense-versus-MoE
difference as an isolated causal architecture effect.

## Scientific variables and invariants

### Assigned prompt variable

The existing prompt population varies from information seeking to action
readiness. Its assigned readiness coordinate is the prompt-level experimental
variable. Embedding coordinates can verify that the realized prompt moves along
the intended semantic axis. They do not define the assigned coordinate and are
not confounders.

### Condition interventions

The ablation and shuffle assignments are randomized or deterministically
counterbalanced before model inference. Because every condition is paired
within prompt, comparisons between Natural and Ablated or Natural and Shuffled
hold the prompt and frozen candidate set fixed, except for the intended
intervention.

### Fixed components

For a given prompt, keep these components unchanged across models and
conditions:

- exact prompt text and prompt ID;
- assigned semantic coordinate and surface-realization seed;
- metadata-bound search keyword or query;
- frozen document snapshot and document text;
- document IDs, parsing, and truncation policy;
- output size and output schema;
- decoding policy within each pipeline;
- model revision within a model arm; and
- analysis code and exclusion rules.

Search collection and model inference remain separate. Do not enable a model's
native web-search tool during either pipeline. A live search agent could issue
different queries or retrieve different pages, which would change the candidate
set and invalidate the paired comparison.

The historical neutral and biased prompt pipelines remain unchanged as legacy
baselines. This ARR design adds a new experiment and does not overwrite those
outputs.

## Model panel

| Requested arm | Architecture | Total parameters | Active parameters | Planned role | Freeze status |
| --- | --- | ---: | ---: | --- | --- |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | MoE | 109B | 17B | Reranking and answer generation | Verify revision and vLLM build |
| `Qwen/Qwen2.5-72B-Instruct` | Dense | 72B | 72B | Reranking, answer generation, preferred judge | Approved substitute for unavailable Qwen3.8-72B |
| `mistralai/Mistral-Small-4-119B-2603` | MoE | 119B | about 6.5B | Reranking and answer generation | Verify revision and vLLM build |
| `Qwen/Qwen3.8-27B` | Dense | about 27B | about 27B | Reranking and answer generation | Verify revision and vLLM build |

The Llama and Mistral names above match official model cards. Mistral describes
its model as 119B total and about 6.5B active parameters, commonly shortened to
`A6B`. The official Qwen inventory exposes Qwen3.8-27B but not Qwen3.8-72B.
Valerian approved `Qwen/Qwen2.5-72B-Instruct` as the dense 72B substitute for
the pilot. Record this change because the two dense Qwen arms now come from
different model generations.

## Input data contract

Each frozen prompt record contains at least:

```json
{
  "prompt_id": "stable ID",
  "prompt_text": "natural-language search prompt",
  "keyword": "metadata-bound search query",
  "assigned_readiness": 0.0,
  "surface_seed": 0,
  "documents": [
    {
      "document_id": "D01",
      "natural_rank": 1,
      "url": "https://example.org/page",
      "title": "Page title",
      "text": "frozen cleaned page text",
      "content_hash": "sha256:..."
    }
  ]
}
```

The input builder must reject a prompt if its document set is incomplete, has
duplicate IDs, fails a content-hash check, or exceeds the frozen truncation
policy. It must never fetch a replacement page during inference.

## The three conditions

### 1. Natural

Natural preserves the order in the frozen search snapshot. It is the reference
condition for both pipelines.

```text
D01, D02, D03, D04, D05
```

### 2. Ablated

Ablated removes exactly one preregistered target document. It preserves the
relative order of all remaining documents.

```text
Target: D03
Input:  D01, D02, D04, D05
```

Select the target before observing any model output. Two valid target policies
are:

- select a substantively defined target, such as the first eligible first-party
  page; or
- assign the target rank with a stable hash and balance target ranks across
  prompts.

The run manifest must name one policy. It must also define what happens when a
prompt has no eligible target.

This primary design uses one target ablation per prompt so that it has exactly
three conditions. Removing every document in turn is a leave-one-document-out
extension, not a three-condition design. With `N` documents, that extension has
`N + 2` variants per prompt and must receive a separate compute estimate.

### 3. Shuffled

Shuffled keeps the complete document set and changes only its order.

```text
Natural:  D01, D02, D03, D04, D05
Shuffled: D04, D01, D05, D02, D03
```

Derive the permutation from a stable hash of the prompt ID and a master seed.
Use a derangement when the document count permits it. Counterbalance mappings
from natural position to shuffled position across the corpus. Reuse the same
permutation for all models and both pipelines.

One shuffle per prompt estimates the effect of the assigned shuffled condition.
It does not measure variance across all possible permutations. A preregistered
subset with repeated counterbalanced shuffles can estimate that variance and
serve as a robustness check.

## Pipeline 1: search simulation and reranking

### Purpose

Pipeline 1 measures the model's stated ordering of a fixed document set. It does
not perform live search.

### Step-by-step logic

1. Load one frozen prompt and its verified document snapshot.
2. Construct the Natural, Ablated, and Shuffled document lists.
3. Render one common reranking instruction around the prompt and documents.
4. Require the model to return document IDs only in a strict JSON schema.
5. Validate membership, uniqueness, list length, and schema.
6. Retry only under a preregistered repair rule that does not change content.
7. Store the full input order, raw output, parsed ranking, token counts, timing,
   model revision, and task hash.
8. Compare each intervention ranking with the Natural ranking for the same
   prompt and model.

### Output contract

```json
{
  "task_id": "stable hash",
  "prompt_id": "prompt-000001",
  "model_id": "immutable model ID",
  "condition": "natural",
  "input_document_ids": ["D01", "D02", "D03"],
  "ranked_document_ids": ["D03", "D01", "D02"]
}
```

The model must not receive natural-rank numbers as semantic evidence unless
those labels are part of every condition. Document IDs must be position-neutral.

### Primary ranking outcomes

- Kendall rank correlation between Natural and Shuffled;
- rank-biased overlap or weighted top-k overlap;
- top-k Jaccard overlap;
- mean absolute rank change per document;
- probability that a document follows its new input position;
- promotion of documents after the ablation target is removed; and
- change in rank as a function of original and assigned input position.

## Pipeline 2: generation, citations, and realized document use

### Purpose

Pipeline 2 measures which documents influence an answer, not only which
documents the model says it prefers.

### Step-by-step logic

1. Reuse the exact three document variants from Pipeline 1.
2. Ask the model to answer the user's prompt using only the provided evidence.
3. Require inline citations with stable document IDs, such as `[D03]`.
4. Parse citations deterministically and flag unknown or malformed IDs.
5. Store the answer, citation sequence, cited-ID set, token counts, and input
   order.
6. Send the prompt, documents, and answer to a blinded judge.
7. Ask the judge for claim support, source utilization, target-document value,
   unsupported claims, and a ranked list of documents actually used.
8. Compare the judge's realized-use ranking and parsed citations with Pipeline
   1's stated ranking.

### Generation output contract

```json
{
  "task_id": "stable hash",
  "prompt_id": "prompt-000001",
  "model_id": "immutable model ID",
  "condition": "shuffled",
  "answer": "... [D03] ... [D01]",
  "parsed_citation_ids": ["D03", "D01"]
}
```

### Judge output contract

```json
{
  "generation_task_id": "stable hash",
  "judge_model_id": "immutable judge ID",
  "answer_quality": 0,
  "evidence_coverage": 0,
  "citation_correctness": 0,
  "unsupported_claim_count": 0,
  "realized_document_ranking": [
    {"document_id": "D03", "use_score": 0, "supporting_claim_ids": ["C1"]}
  ],
  "target_document_assessment": {
    "document_id": "D03",
    "marginal_value": 0,
    "replacement_document_ids": ["D01"]
  }
}
```

### Judge controls

The judge is a measurement instrument, not ground truth. Apply these controls:

- pin the judge model, revision, prompt, schema, and decoding settings;
- blind the judge to generator identity and experimental condition;
- present documents to the judge in a canonical or independently
  counterbalanced order;
- parse citations without an LLM before applying the judge;
- calibrate the rubric on a human-labeled sample;
- report inter-rater agreement on a subset with a second judge or human;
- audit position sensitivity in the judge itself; and
- do not use Qwen self-judgment as the only primary evaluation of Qwen outputs.

If the 72B Qwen checkpoint remains unresolved, select the judge only after the
model freeze. A panel model may serve as a judge for other generators, but
self-evaluation must remain a sensitivity analysis.

### Primary generation and agreement outcomes

- cited-document precision and recall against claim support;
- citation order and frequency by input position;
- answer quality and evidence coverage;
- answer semantic displacement from Natural to Ablated;
- answer semantic displacement from Natural to Shuffled;
- realized-use rank change after ablation or shuffling;
- Kendall correlation between Pipeline 1 and realized-use rankings;
- top-k overlap between stated ranking, cited documents, and realized use; and
- rate at which a removed target is replaced by each remaining document.

Embedding-based answer displacement is an outcome or manipulation check. It is
not the assigned treatment and must not be described as a confounder.

## Experimental table and randomization

The atomic observation is:

```text
prompt x model x condition x pipeline
```

The three conditions are paired within each prompt and model. Interleave their
execution in randomized blocks so that condition is not confounded with server
warm-up, time, or transient load. Use the same condition assignment and
document permutation across all four models and both pipelines.

Recommended identifiers are:

```text
prompt_id
snapshot_id
document_set_hash
condition_id
ablation_target_id
permutation_id
pipeline_id
model_id
model_revision
judge_id
task_id
```

## Hardware and vLLM serving architecture

### Planning assumptions

- one node;
- four GH200 GPUs;
- approximately 144 GB HBM per GPU and 576 GB total HBM;
- local high-speed model and artifact storage; and
- one vLLM server process group for the active model arm.

Verify the actual HBM, CUDA version, driver, interconnect, and free storage at
run time. Record them in the manifest.

### Serving strategy

Do not load all four models at once. Run model arms sequentially:

```text
frozen tasks
    -> deterministic condition builder
    -> bounded asyncio request queue
    -> one active vLLM model server on the 4-GPU node
    -> validated, resumable output shards
    -> next model arm
    -> blinded judge server after generation completes
```

Use tensor parallelism across four GPUs as the initial configuration for models
that support it. For MoE models, benchmark vLLM's supported tensor-parallel and
expert-parallel configurations. Do not assume one topology works for every
checkpoint. For the 27B dense arm, compare one four-GPU replica with multiple
smaller replicas during the pilot. Freeze the fastest configuration that
preserves the exact numerical and decoding contract.

Keep the context limit identical across models at the smallest value that holds
the frozen evidence payload and output budget. Do not use a model's advertised
maximum context merely because it is available. Measure prompt-token lengths,
KV-cache use, prefill throughput, decode throughput, and out-of-memory margin in
a representative pilot.

vLLM performs continuous batching inside the server. The client should submit
bounded concurrent requests instead of concatenating unrelated prompts into one
chat. Automatic prefix caching may improve throughput, but its setting must be
recorded and verified not to change outputs.

## Execution phases

### Phase 0: freeze the design

1. Freeze the exact prompt ID set.
2. Freeze search snapshots and content hashes.
3. Freeze document truncation and output length.
4. Freeze ablation-target assignment and shuffle seed.
5. Resolve all four model IDs and immutable revisions.
6. Freeze the judge and rubric.
7. Preregister primary outcomes and exclusions.

### Phase 1: CPU validation

1. Build all task manifests without model inference.
2. Assert three variants per prompt.
3. Assert that Ablated removes only its target.
4. Assert that Shuffled preserves document membership.
5. Verify balancing across natural and shuffled positions.
6. Validate task IDs, hashes, resume behavior, and schemas.

### Phase 2: representative GPU pilot

Run a small stratified sample for every model and both pipelines. Measure model
load time, prompt tokens, output tokens, requests per second, invalid output
rate, peak HBM, retries, and judge latency. Use those measurements to estimate
the full allocation. The pilot is infrastructure evidence, not a scientific
result.

### Phase 3: Pipeline 1 production

Run one model arm at a time. Interleave the three conditions. Write atomic
shards, checkpoint completed task IDs, validate every output, and retry only
failed tasks under the frozen policy.

### Phase 4: Pipeline 2 production

Use the same model order or a preregistered counterbalanced order. Generate
answers and citations. Validate and checkpoint outputs independently of
Pipeline 1.

### Phase 5: blinded judging

Remove generator and condition labels from judge inputs. Run the frozen judge,
parse structured outputs, and route invalid judgments to a separate retry
queue. Complete the human calibration subset before treating judge scores as
outcomes.

### Phase 6: analysis and audit

Join only by stable task IDs. Confirm complete paired cells before estimating
effects. Publish missingness, retry counts, invalid-output rates, exclusions,
and per-model coverage with the main results.

## Statistical analysis

Estimate paired within-prompt contrasts for:

```text
shuffle effect  = outcome(shuffled) - outcome(natural)
ablation effect = outcome(ablated) - outcome(natural)
```

Model outcomes as functions of condition, assigned readiness, model arm, and
their preregistered interactions. Include prompt or query grouping in the
uncertainty calculation. Candidate-level ranking analyses should account for
documents nested within candidate sets and prompts.

Recommended primary models include:

- mixed-effects or cluster-robust models for scalar outcomes;
- paired randomization inference for Natural versus Shuffled;
- survival or ordinal rank models for document positions;
- bootstrap confidence intervals clustered by query or keyword; and
- false-discovery control for secondary model-by-condition interactions.

Page-feature effects remain observational unless page content or the page
feature itself is randomized. Randomized document order identifies position
effects. Randomized target removal identifies the effect of access to that
document under the stated assignment policy.

## Brief vLLM and asyncio pseudocode

The vLLM server runs outside this client and exposes an OpenAI-compatible API.
The production implementation should stream task manifests and write atomic
Parquet or JSONL shards. This sketch shows the control flow, not a launch
command.

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from openai import AsyncOpenAI


class Condition(StrEnum):
    NATURAL = "natural"
    ABLATED = "ablated"
    SHUFFLED = "shuffled"


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt_id: str
    prompt_text: str
    model_id: str
    pipeline: str
    condition: Condition
    documents: tuple[dict, ...]
    ablation_target_id: str | None


def stable_seed(master_seed: int, prompt_id: str) -> int:
    raw = f"{master_seed}:{prompt_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def build_conditions(record: dict, master_seed: int) -> dict[Condition, list[dict]]:
    natural = list(record["documents"])
    target = record["ablation_target_id"]
    ablated = [doc for doc in natural if doc["document_id"] != target]

    shuffled = list(natural)
    random.Random(stable_seed(master_seed, record["prompt_id"])).shuffle(shuffled)
    assert {d["document_id"] for d in shuffled} == {
        d["document_id"] for d in natural
    }
    assert len(ablated) == len(natural) - 1
    return {
        Condition.NATURAL: natural,
        Condition.ABLATED: ablated,
        Condition.SHUFFLED: shuffled,
    }


def render_messages(task: Task) -> list[dict]:
    evidence = "\n\n".join(
        f'<document id="{d["document_id"]}">\n{d["text"]}\n</document>'
        for d in task.documents
    )
    if task.pipeline == "rerank":
        instruction = (
            "Rank the documents for the user request. Return one JSON object "
            'with key "ranked_document_ids" and no other text.'
        )
    else:
        instruction = (
            "Answer only from the supplied documents. Cite supporting sources "
            "inline with stable IDs such as [D03]."
        )
    return [{
        "role": "user",
        "content": f"{instruction}\n\nREQUEST:\n{task.prompt_text}\n\n{evidence}",
    }]


async def infer_one(
    client: AsyncOpenAI,
    task: Task,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        response = await client.chat.completions.create(
            model=task.model_id,
            messages=render_messages(task),
            temperature=0,
            max_tokens=512 if task.pipeline == "answer" else 256,
            extra_body={"guided_json": schema_for(task.pipeline)},
        )
    raw = response.choices[0].message.content
    parsed = validate_output(task, raw)
    return {"task_id": task.task_id, "raw": raw, "parsed": parsed}


async def run_model_arm(model_id: str, manifest: Path, output: Path) -> None:
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
    completed = load_completed_task_ids(output)
    semaphore = asyncio.Semaphore(64)  # Set from the per-model pilot.

    pending = []
    async for task in stream_tasks(manifest, model_id=model_id):
        if task.task_id not in completed:
            pending.append(asyncio.create_task(infer_one(client, task, semaphore)))
        if len(pending) >= 512:
            for future in asyncio.as_completed(pending):
                append_validated_record_atomically(output, await future)
            pending.clear()

    for future in asyncio.as_completed(pending):
        append_validated_record_atomically(output, await future)


async def judge_answers(judge_id: str, generations: Path, output: Path) -> None:
    # Build blinded judge tasks. Present documents in an independently
    # counterbalanced order and require the frozen structured judge schema.
    await run_model_arm(judge_id, build_judge_manifest(generations), output)


async def main() -> None:
    # Start and stop one external vLLM server per model arm. Do not try to keep
    # all model weights resident at the same time.
    for model_id in FROZEN_GENERATOR_MODEL_IDS:
        await wait_for_matching_vllm_server(model_id)
        await run_model_arm(model_id, RERANK_MANIFEST, rerank_output(model_id))
        await run_model_arm(model_id, ANSWER_MANIFEST, answer_output(model_id))

    await wait_for_matching_vllm_server(FROZEN_JUDGE_MODEL_ID)
    await judge_answers(FROZEN_JUDGE_MODEL_ID, ALL_GENERATIONS, JUDGE_OUTPUT)


if __name__ == "__main__":
    asyncio.run(main())
```

The production version should use a bounded queue rather than allowing an
unbounded list of tasks. It should add exponential backoff, timeout classes,
server identity checks, output-schema tests, per-task attempt logs, and graceful
checkpointing on `SIGTERM`.

## Reproducibility manifest

Every run must record:

- Git commit SHA;
- prompt file hash and exact prompt count;
- search snapshot and page-content hashes;
- master seed, target-assignment policy, and permutation IDs;
- model IDs, revisions, licenses, and tokenizer revisions;
- vLLM, PyTorch, CUDA, driver, and container versions;
- tensor, pipeline, data, and expert parallel settings;
- precision, quantization, context limit, and KV-cache settings;
- decoding settings and structured-output schemas;
- Slurm job ID and allocated resources;
- start and end timestamps;
- raw, parsed, failed, and retried task counts; and
- hashes of final output shards.

Never silently overwrite an earlier run. Write to a new run directory and mark
completion only after count, schema, hash, and paired-cell audits pass.

## Pre-run decision gates

The experiment is ready for production only after these items are closed:

1. Pin the approved `Qwen/Qwen2.5-72B-Instruct` revision.
2. Confirm that every selected prompt has a frozen, answerable document set.
3. Choose and preregister the one-target ablation policy.
4. Generate and audit the counterbalanced shuffle assignments.
5. Freeze the number of documents and common context budget.
6. Freeze reranking and answer schemas.
7. Select and calibrate the judge without relying on self-judgment.
8. Run a representative pilot for all four models.
9. Estimate full runtime and request a separate Slurm wall-time approval from
   measured throughput.
10. Freeze the primary estimands, exclusions, and multiplicity correction.

## Relationship to the wider GEODML plan

This ARR experiment uses the audited semantic-readiness prompt population. It
does not replace the separate stochastic first-party source-preference design:

```text
B in [0, 1]
S = surface-realization seed
P = G(B, S)
```

If that policy axis appears in the same paper, keep `B` as its assigned
treatment and keep `S` restricted to surface realization. The readiness
coordinate, source-preference coordinate, document-order condition, and
ablation condition are distinct variables and require distinct estimands.

## Authoritative references for model and serving details

- [Llama 4 Scout model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [Official Qwen model inventory](https://huggingface.co/Qwen/models)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Mistral Small 4 model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
