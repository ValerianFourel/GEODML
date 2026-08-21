# Master plan v2 compared with the current implementation

Date: 2026-08-21  
Repository snapshot: `84e36dd5acc9210563a7c17c3a2a6aa0f825f0f6`  
Branch: `codex/semantic-readiness-phase2-jupiter`

## Purpose

This document compares the implementation in this repository with the workspace
specification:

`GEODML GEO Master Codex Prompt v2 — Continuous Semantic Readiness, Four-Model Behavioral Panel, and Hierarchical Search-RAG.md`

It is a status and design comparison, not a claim that the master plan has been
completed. In particular, successful unit tests, mocked generation, and small
prompt pilots are not scientific results.

Status terms used below:

- **Implemented**: the relevant behavioral contract exists in committed code
  and has focused tests.
- **Partial**: useful components exist, but a required freeze, validation gate,
  or end-to-end integration is still missing.
- **Not implemented**: the current repository does not yet satisfy the master
  plan's contract for that phase.
- **Historical only**: older code or outputs may be reusable, but they are not
  an implementation of the new specification.

## Executive comparison

The implementation has made substantial progress on semantic-corpus assembly,
multi-judge readiness annotation, supervised LLM2Vec geometry, cross-embedding
robustness, and prompt-population tooling. The most recent milestone is a
support-aware, iterative question-generation loop that plans approximately 30
questions per keyword, generates multiple proposals, independently validates
them, re-embeds them through two frozen views, globally matches them to targets,
and schedules only deficient targets for another round.

That loop is not the complete experiment described by the master prompt. The
canonical paired search panel, new source identities, hierarchical Search-RAG,
paired source-order randomization, four-model answer panel, citation and quote
parsing, counterfactual influence, and final statistical analysis remain later
milestones.

The master prompt and the current project instructions also define different
primary treatments. That difference must be resolved before the downstream
experiment is frozen.

## Scientific contract: master plan versus current project

| Concept | Master prompt v2 | Current governing project instructions | Consequence |
|---|---|---|---|
| Primary randomized variable | Assigned decision/action readiness, `A*` | Assigned first-party product-source preference intensity, `B` | `B`, not an embedding coordinate, is the current policy treatment. |
| Surface randomization | `S` changes wording around a fixed readiness assignment | `S` changes wording, syntax, order, and tone without adding ranking criteria | The implementations agree that surface variation must not introduce new semantics. |
| Generated object | A query-conditioned search prompt `P = G(q, A*, S)` | A reranking instruction `P = G(B, S)` with the query, candidates, evidence, output size, and format fixed | Readiness-question generation cannot be substituted directly for the final `B`-indexed instruction generator. |
| Embedding coordinate | `A_obs` is a manipulation check for assigned `A*` | Prompt embeddings may describe `P` but must not define `B` | The current code correctly records this guard in plans and manifests. |
| Behavioral task | Natural answer generation with selection, retrieval, citations, quotations, and reliance | Prompted LLM reranking remains the current project mission | Master-plan answer-generation phases are not automatically authorized by the current mission. |
| Page-feature estimand | Later source-use analyses | Page-feature effects remain observational DML estimates unless page content is manipulated | Existing DML results must not be relabeled as randomized prompt-policy effects. |

The practical interpretation is:

1. the semantic-readiness work is a valid auxiliary measurement and prompt-
   population program;
2. its coordinates describe generated text and generator controllability;
3. it does not define the randomized policy axis `B`; and
4. a future bridge must state exactly how the readiness-question population is
   used, if at all, in the final `B` experiment.

## Phase-by-phase status

| Master phase | Current implementation | Status | Main gap or deviation |
|---|---|---|---|
| Phase 0: repository and experiment audit | Repository, SERP, prompt, model, and cluster components have been inspected across several milestones. | Partial | The requested canonical `phase0_audit.md` and `phase0_status.json`, including a final paired-query reconciliation, are not present. |
| Phase 1: freeze semantic corpora | `semantic_readiness_phase1.py`, the transfer registry, deterministic acquisition, hashes, source policies, and split-leakage tests exist. The original 5,091-row corpus was reconstructed and audited. | Partial | The original eight-source gate was not completed as specified. The operational expansion uses six sources; MS MARCO and LMSYS are not part of that six-source panel. |
| Phase 2: readiness annotation | Versioned judge tasks, raw cache preservation, retries, abstentions, multi-GPU runners, incremental four-judge queues, and a 20k-panel workflow exist. | Partial | The currently fitted map explicitly uses three completed judge slots and excludes the incomplete fourth. This is reproducible, but it is not a frozen complete four-judge panel. |
| Phase 3: discover semantic direction | Ridge, proportional-odds diagnostics, PCA diagnostics, development/confirmation separation, Qwen and Mistral LLM2Vec views, held-out checks, and a robustness battery are implemented. | Substantially implemented | The implementation extends the master plan from one scalar direction to a supervised two-axis subspace. This is a documented design extension, not a silent equivalent of the original one-axis plan. |
| Phase 4: generate a 30-level continuum | Rectangular pilot planning and the newer support-aware population/refinement pipeline are implemented and tested. | In progress | The generated objects are search questions along readiness dimensions, not final reranking instructions varying only `B`. Large-run completion and realized-coverage validation are not yet established. |
| Phase 5: freeze search/page evidence | Historical SearXNG/DDG pools, cached HTML, extracted text, and page features exist. | Partial / historical | The new canonical paired panel, opaque `source_uid` freeze, complete page-extraction manifest, and new experiment version have not been finalized. |
| Phase 6: hierarchical Search-RAG | Legacy chunking, dense embeddings, and passage-enhanced reranking code exist. | Not implemented under master v2 | There is no conforming end-to-end `20 snippets -> LLM selects K pages -> BM25+dense retrieval with deterministic fusion -> answer context` pipeline. Legacy RAG must not be described as Mode HR. |
| Phase 7: randomization and manifests | Stable hashes and manifests exist in prompt and annotation pipelines. | Partial | Paired source-order permutations across models and evidence modes, baseline manifests, selector manifests, and tokenized contexts are not frozen. |
| Phase 8: four-model compute pilot | Model-loading and throughput experience exists from annotation, embedding, and prompt-generation jobs. | Not implemented for the master experiment | No balanced benchmark of the four answer models on finalized Mode S and Mode HR contexts has been completed. |
| Phase 9: baseline experiment | Historical reranking experiments exist and are preserved. | Not implemented for the master experiment | The proposed four-model, two-engine, two-mode answer panel has not been run. Historical neutral/biased reranking is a baseline, not this phase. |
| Phase 10: response-conditioned LOO | Historical ablation code exists. | Not implemented for the master experiment | It does not yet implement the specified full/content LOO scoring over frozen generative answers. |
| Phase 11: regenerative deletion | No conforming new-pipeline implementation is frozen. | Not implemented | Mode-HR deletion would have to rerun selection, retrieval, and generation under a separately approved manifest. |
| Phase 12: final statistical analysis | Extensive historical DML and reporting code exists. | Not implemented for the new experiment | No new experimental outputs exist from which to estimate the planned causal dose responses or mechanism chain. |
| Phase 13: latent space to permutation space | Prompt-to-permutation prototypes exist. | Partial research prototype | Final mapping requires frozen experimental prompts and source-use permutations; embeddings must not replace assigned treatment `B`. |

## What the readiness implementation currently provides

### Corpus and annotation layer

The committed workflows provide:

- deterministic corpus IDs and text hashes;
- development and locked-confirmation/source-held-out splits;
- raw per-judge preservation;
- explicit `not_applicable` and `dont_know` abstentions;
- resumable and failure-preserving judge queues;
- redistribution-aware restricted-local and Hugging Face-safe scopes; and
- immutable manifests for completed snapshots.

Relevant code and documentation:

- `analysis/interpretability/pipeline/semantic_readiness_phase1.py`
- `analysis/interpretability/pipeline/semantic_readiness_dataset.py`
- `analysis/interpretability/pipeline/readiness_hf_dataset.py`
- `analysis/docs/semantic_readiness_phase1_super_prompt_status.md`
- `analysis/docs/semantic_readiness_hf_dataset_jupiter_runbook.md`

### Geometry layer

The current map is more expressive than the master prompt's single readiness
coordinate. It contains:

- a scalar Ridge readiness prediction;
- a supervised two-dimensional rubric subspace;
- a proportional-odds robustness direction;
- PCA components used only as unsupervised diagnostics;
- development-fitted calibration and confirmation-only evaluation; and
- independent Qwen and Mistral LLM2Vec maps with development-fitted alignment.

The robustness battery compares the two frozen views on identical items and
does not refit either high-dimensional map using generated questions.

Relevant code and documentation:

- `analysis/interpretability/pipeline/readiness_embedding_map.py`
- `analysis/interpretability/pipeline/readiness_hf_subspace.py`
- `analysis/interpretability/pipeline/readiness_subspace_battery.py`
- `analysis/docs/readiness_subspace_robustness_battery.md`

## The new prompt-generation loop

The new loop is implemented in
`analysis/interpretability/pipeline/readiness_prompt_population.py` and exposed
by `analysis/scripts/build_readiness_prompt_population.py`.

It is a round-based generate-measure-select-refine loop:

```text
frozen development support and map
              |
              v
deterministic per-keyword target plan
              |
              v
Qwen/Gemma proposal generation with surface variation
              |
              v
independent search-question validation
              |
              v
frozen Qwen projection + frozen Mistral projection
              |
              v
development-fitted cross-view alignment
              |
              v
global one-to-one spatial assignment
              |
              v
coverage/spacing gate
       |                    |
       | pass               | deficient targets
       v                    v
freeze selected bank    next-round tasks with feedback
                             |
                             +--------> generate again
```

An additional proposal arm now implements a faster inner feedback loop in
`analysis/interpretability/pipeline/readiness_latent_feedback.py`, exposed by
`analysis/scripts/generate_llm2vec_gen_feedback_proposals.py`:

```text
random valid seed question
          |
          v
frozen LLM2Vec measurement ------> target error
          |                              |
          v                              v
LLM2Vec-Gen reconstruction state + development-calibrated latent directions
          |
          v
several bounded latent steps and decodes
          |
          v
exact-query anchoring + hard checks + independent LLM review
          |
          v
frozen LLM2Vec re-measurement
     |                         |
     | inside tolerance        | still deficient
     v                         v
emit proposal             repeat from closest valid decode
```

This inner loop does not decode an arbitrary two-dimensional LLM2Vec point.
The frozen readiness coordinates and LLM2Vec-Gen reconstruction states are
different spaces. A ridge bridge fitted only on development rows estimates
state changes associated with each readiness direction; every decoded final
text is then re-embedded, and only that measurement controls acceptance. The
default behavior fails closed outside tolerance and writes a complete trace of
latent hashes, decoded text, validation decisions, and measured coordinates.

### 1. Freeze a support-aware target plan

Large-population mode uses `support-aware-random`, rather than forcing every
keyword onto the corners of a rectangular grid.

The planner:

1. uses only usable development coordinates;
2. normalizes the two axes using frozen bounds;
3. partitions the space into a 20 by 20 support grid by default;
4. excludes cells with fewer than three development points;
5. balances pooled target allocations across eligible cells;
6. draws each target by interpolating between development points in the same
   cell; and
7. records the seed, support counts, allocation range, and input hashes.

For the currently reported plan, 1,011 keywords times 30 targets produces
30,330 target questions. The observed planning audit reported 306 eligible
support cells and a pooled allocation-count range of one. These are plan
properties, not realized prompt-coverage results.

### 2. Build deterministic generation tasks

Each keyword-target pair receives a stable task ID derived from the keyword,
target, round, generator, seed, and specification version. Generator assignment
rotates across keyword, target, and round so that Qwen and Gemma contribute
throughout the space without defining its coordinates.

The current large plan requests four candidate slots per target, allowing up to
121,320 proposals before validation and matching. Four proposals do not change
the final target count; they give the assignment stage alternatives.

Surface instructions rotate deterministically across candidate slots. They may
change expression, such as direct versus context-first wording, but they must
not add a new source policy or experimental ranking criterion.

### 3. Generate in resumable slices

The `generate` stage filters the shared task file by `generator_id` and supports
`--start-index`, `--limit`, and `--resume`. Each task is cached independently,
outputs have stable candidate IDs and text hashes, and the combined JSONL file
is written atomically.

The local generator retries malformed generations up to a configured maximum.
Every accepted proposal must:

- contain the exact keyword phrase;
- contain 8 to 60 words on one line;
- end in exactly one question mark;
- contain no axis, embedding, reranking, or source-policy language; and
- provide the requested number of candidate slots.

The current cluster pilot uses Qwen3-32B and Gemma4-31B as proposal generators.
That does not constitute the master prompt's four-model behavioral answer panel.

### 4. Validate with an independent model

`validate-candidates` requires complete candidate coverage and fails closed on
malformed judge output. A candidate passes only if it retains the exact keyword,
is one question, passes all five boolean checks, and receives at least four out
of five for relevance.

The five semantic checks are:

- topic relevance;
- genuine online-search intent;
- web answerability;
- standalone wording; and
- natural language.

This validation checks whether a candidate is a usable search question. It does
not prove that the intended semantic coordinate was reached.

### 5. Re-embed every candidate through two frozen views

The same accepted candidate texts are independently embedded and projected
through the frozen Qwen and Mistral maps. Alignment is learned from the original
development corpus, never from the generated candidate bank. Raw coordinates
from the two models are not compared as if they shared a scale.

The consensus coordinate is the mean of the reference coordinate and the
development-aligned candidate-view coordinate. Cross-view distance is retained
as a disagreement penalty.

### 6. Match globally rather than by requested cell

For every keyword, the selector constructs the full target-to-candidate cost
matrix:

```text
cost = distance(target, consensus coordinate)
       + disagreement_weight * cross_view_disagreement
```

It then uses the Hungarian algorithm to find a minimum-cost one-to-one
assignment. A proposal generated for one target may therefore fill another
target when its measured semantics fit better there. This avoids preserving a
bad requested-cell assignment merely because of its generation label.

Only candidates accepted by every supplied independent validator are eligible.

### 7. Audit realized coverage

For support-aware plans, the pooled gate currently requires:

- a selected question for every pooled target;
- mean target distance at most 0.25;
- at least 80% of selected questions within the configured tolerance;
- observed spans of both axes covering at least 80% of target spans;
- at least 80% of target histogram bins represented; and
- target-versus-observed 10 by 10 histogram total-variation distance at most
  0.25.

Per-keyword diagnostics additionally compare observed span, spacing, and
occupied bins with that keyword's planned targets. The thresholds are prompt-
construction diagnostics; they are not definitions of `B` and are not causal
findings.

### 8. Refine only deficient targets

If a target has no accepted assignment or its selected question lies outside
the distance tolerance, `spatial-select` writes a task for the next round. The
feedback records where the closest candidate landed and requests the smallest
semantic shift toward the frozen target while preserving the exact keyword.

The next round rotates generator assignment. All earlier and new proposals are
then validated, projected, and matched together into a fresh immutable output
directory. The loop stops when no refinement tasks remain or a separately
frozen maximum round is reached.

The implementation does not currently contain an autonomous job that runs all
rounds indefinitely. Cluster execution remains an explicitly staged sequence
with human-reviewed allocation and audit gates.

## Current large-plan operational status

The last reported cluster state for the support-aware plan was:

- commit `84e36dd5acc9210563a7c17c3a2a6aa0f825f0f6`;
- 30,330 planned targets and at most 121,320 proposals;
- initial slice jobs failed during runtime preflight because
  `PYTHONNOUSERSITE=1` hid `huggingface_hub`;
- the cluster-local wrapper was repaired by allowing the required user site;
- replacement Qwen and Gemma pilot slices passed preflight and began loading
  their pinned models; and
- no completed pilot artifacts or realized-coverage audit were available in
  the repository at the time of this comparison.

Therefore this document does not claim that the large generation completed or
that its coverage gate passed. The cluster-only slice-wrapper repair should be
reproduced in the Git repository before a later scientific run relies on it.

## Why the legacy RAG implementation is not yet Mode HR

The repository already contains useful components such as:

- cached SERP loading;
- cached HTML extraction;
- recursive text chunking;
- dense chunk embeddings;
- passage retrieval; and
- historical snippet, passage, and RAG reranking variants.

These components can inform a later implementation, but the master prompt's
Mode HR requires additional contracts:

1. the answer model first selects at most `K` opaque source IDs from the same
   randomized 20-snippet overview;
2. only selected pages are opened and chunked;
3. lexical BM25 and a pinned dense retriever operate within those pages;
4. ranks are combined by a deterministic frozen fusion rule such as RRF;
5. approximately `L` chunks per selected page are retained with exact chunk
   provenance;
6. final context order follows the paired randomized source order rather than
   selection rank; and
7. a natural answer with citations is generated and parsed.

The legacy implementation does not satisfy that complete contract. It also
uses historical domain-ranking outputs rather than the master plan's grounded
answer, citation, quotation, and influence outcomes. It should be labeled
historical or prototype code until a new experiment specification deliberately
reuses and versions its components.

## Requirements that remain before a full experiment

Before any production behavioral run, the project still needs a frozen decision
on the treatment contract. The minimum sequence is:

1. decide whether the next experiment follows the master prompt's readiness
   treatment `A*` or the current project's source-policy treatment `B`;
2. finish and audit the prompt population appropriate to that treatment;
3. freeze the canonical paired query and evidence panel;
4. assign opaque stable source and chunk identities;
5. decide whether the outcome remains reranking or changes to grounded answer
   generation;
6. if authorized, implement Mode S and Mode HR under the chosen outcome;
7. freeze paired order randomization and immutable manifests;
8. benchmark the exact model panel and contexts;
9. obtain allocation-specific compute approval; and
10. run a small balanced pilot before any production scale-up.

No downstream result should be used to weaken prompt-population thresholds,
change the semantic map, redefine `B`, or modify source pools after the
experiment version is frozen.

## Traceability

Primary implementation references:

- `analysis/interpretability/pipeline/semantic_readiness_phase1.py`
- `analysis/interpretability/pipeline/readiness_embedding_map.py`
- `analysis/interpretability/pipeline/readiness_hf_subspace.py`
- `analysis/interpretability/pipeline/readiness_subspace_battery.py`
- `analysis/interpretability/pipeline/readiness_prompt_population.py`
- `analysis/scripts/build_readiness_hf_dataset.py`
- `analysis/scripts/build_readiness_prompt_population.py`
- `analysis/scripts/slurm/jupiter/run_readiness_prompt_round1.sh`
- `analysis/tests/test_readiness_hf_subspace.py`
- `analysis/tests/test_readiness_subspace_battery.py`
- `analysis/tests/test_readiness_prompt_population.py`
- `analysis/tests/test_jupiter_semantic_readiness_jobs.py`
- `analysis/docs/readiness_question_population.md`
- `analysis/docs/readiness_subspace_robustness_battery.md`

Historical RAG references that are not, by themselves, Mode HR:

- `analysis/interpretability/pipeline/build_rag_index.py`
- `analysis/interpretability/pipeline/chunker.py`
- `analysis/interpretability/pipeline/rerank.py`
