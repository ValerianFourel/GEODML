# Iterative readiness-question population

This pipeline creates about 30 topic-faithful natural questions per keyword
while covering the two-dimensional semantic-readiness subspace found in the
frozen LLM2Vec analysis.

It is a prompt-population tool, not a new treatment definition. `B` remains the
experimental policy variable. The LLM2Vec coordinates only describe the text
that generator models produced.

## Design

For each keyword, the pipeline creates a 6 by 5 grid:

- axis 1 spans exploratory information seeking through immediate action;
- axis 2 spans compare/select through implement/execute;
- raw grid limits are the 5th and 95th percentiles of the **development**
  coordinates from the frozen readiness map;
- the confirmation split is never used to design the grid.

One generator model is assigned to each cell in round 0. Assignments rotate by
cell, keyword, and round, so Gemma, Qwen, Mistral, or other generators contribute
without defining the coordinates. Each question must retain the exact keyword,
be a single unanswered natural question, and contain no axis or source-policy
language.

After generation, the model is unloaded. Separate jobs then:

1. use an independent LLM to verify topic fidelity, online-search intent,
   web answerability, standalone wording, and natural language;
2. embed identical candidate text through both frozen LLM2Vec views;
3. align Mistral coordinates to Qwen using development data only;
4. globally match validated candidates to the entire 6 by 5 grid using the
   average aligned coordinate and a cross-view disagreement penalty;
5. report axis spans, nearest-neighbor spacing, occupied grid bins, and target
   error;
6. emit new tasks only for cells outside the configured distance tolerance.

Global matching is important. A candidate generated for one intended cell may
actually land closer to another cell. Restricting selection to its requested
cell needlessly preserves bad assignments. The Hungarian assignment used by
`spatial-select` finds the minimum-cost one-to-one matching over the whole
candidate pool for each keyword.

Round 1 is scored together with round 0. This repeats until coverage is adequate
or a preregistered maximum round count is reached.

## Why LLM2Vec-Gen is only a proposal source

The frozen readiness axes live in pooled LLM2Vec input-embedding space.
LLM2Vec-Gen decodes reconstruction hidden states, which are a different space.
Therefore an arbitrary point on the frozen readiness map cannot be passed
directly to the LLM2Vec-Gen decoder as though it were an inverse map.

LLM2Vec-Gen decoded text can still join the candidate bank. It must provide one
JSONL row per proposal with `task_id`, `question`, and optional
`candidate_slot`. The `import-proposals` stage validates that text, and the
normal `score-select` stage re-embeds it with frozen LLM2Vec. No decoded proposal
is accepted based on its decoder coordinate.

Before a large generation run, execute the two-embedding robustness battery in
`analysis/docs/readiness_subspace_robustness_battery.md`. For a small pilot,
project the same candidate JSONL independently through Qwen and Mistral with
`project-candidates`, then use `compare-projections`. The alignment is learned
from the original development corpus and remains frozen for generated questions.

## Inputs

Keywords can be a text file with one keyword per line or JSONL:

```json
{"keyword_id":"kw:001","keyword":"abandoned cart recovery"}
```

The frozen map directory supplies:

- `readiness_embedding_map.json`
- `readiness_supervised_subspace_coordinates.jsonl`

## Round 0

Run planning on a login or CPU node:

```bash
export SUBSPACE_ROOT="$(cat "$HOME/geodml-readiness-subspace-latest.txt")"
export MAP_ROOT="$SUBSPACE_ROOT/maps/qwen3-8b-mntp-unsup-simcse-three-judge-gpu-v2"
export POPULATION_ROOT="$SUBSPACE_ROOT/question-populations/readiness-grid-v1"

python analysis/scripts/build_readiness_prompt_population.py plan \
  --keywords "$KEYWORDS_JSONL" \
  --map "$MAP_ROOT/readiness_embedding_map.json" \
  --reference-coordinates "$MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
  --generator-ids gemma4-31b,qwen3-32b \
  --candidates-per-task 2 \
  --output-dir "$POPULATION_ROOT/plan"
```

For 1,000 keywords this freezes 30,000 cells. With two proposals per cell, round
0 produces at most 60,000 candidate questions and selects 30,000.

Run each generator in its own Slurm job or allocation. The same tasks file is
used, but each command executes only the matching `generator-id`:

```bash
python analysis/scripts/build_readiness_prompt_population.py generate \
  --tasks "$POPULATION_ROOT/plan/generation_tasks_round_00.jsonl" \
  --generator-id qwen3-32b \
  --backend local \
  --model "$QWEN_GENERATOR_MODEL" \
  --precision full \
  --cache-dir "$POPULATION_ROOT/cache/qwen3-32b/round-00" \
  --output "$POPULATION_ROOT/candidates/qwen3-32b-round-00.jsonl" \
  --resume
```

Repeat with the other generator IDs and model snapshots. Tasks and cache keys
are deterministic, so interrupted jobs can resume safely. `--start-index` and
`--limit` support Slurm arrays or short wall-time slices.

The current `Ministral-3-8B-Instruct-2512-BF16` snapshot has a multimodal
`Mistral3Config` and is not a compatible text-only generator for the repository
`AutoModelForCausalLM` ranker. It remains valid as a judge-panel model in its
specialized runner. Use Qwen, Gemma, Llama, or another verified text-only model
for this generation stage.

## Independent search-question validation

Use a model that did not generate the candidate bank, such as the frozen Llama
snapshot, and cache every decision:

```bash
python analysis/scripts/build_readiness_prompt_population.py validate-candidates \
  --candidates "$POPULATION_ROOT"/candidates/*-round-00.jsonl \
  --judge-id llama3.3-70b-search-validator \
  --model "$LLAMA_MODEL" \
  --backend local \
  --precision full \
  --cache-dir "$POPULATION_ROOT/cache/search-validator" \
  --output "$POPULATION_ROOT/validation/llama3.3-70b.jsonl" \
  --resume
```

Acceptance requires the exact keyword, exactly one question, all five semantic
checks, and a relevance score of at least four out of five. Multiple validation
files can be supplied to `spatial-select`; a candidate must pass every supplied
validator.

## Frozen LLM2Vec scoring and selection

Use a one-GPU job with the exact Qwen LLM2Vec base, MNTP adapter, and SimCSE
adapter used to fit the Qwen map:

```bash
python analysis/scripts/build_readiness_prompt_population.py score-select \
  --plan-dir "$POPULATION_ROOT/plan" \
  --map "$MAP_ROOT/readiness_embedding_map.json" \
  --candidates \
    "$POPULATION_ROOT/candidates/gemma4-31b-round-00.jsonl" \
    "$POPULATION_ROOT/candidates/qwen3-32b-round-00.jsonl" \
  --embedding-model "$QWEN3_8B_SNAPSHOT" \
  --mntp-model "$LLM2VEC_MNTP_SNAPSHOT" \
  --peft-model "$LLM2VEC_UNSUP_SNAPSHOT" \
  --embedding-batch-size 8 \
  --distance-tolerance 0.22 \
  --next-round-index 1 \
  --output-dir "$POPULATION_ROOT/selection-round-00"
```

The result contains:

- `selected_questions.jsonl`: one question per covered keyword/cell;
- `candidate_projections.jsonl`: frozen coordinates and target error;
- `candidate_embeddings.restricted-local.npz`: aligned embeddings;
- `selection_diagnostics.json`: coverage and generator summaries;
- `generation_tasks_round_01.jsonl`: only cells requiring refinement;
- `run_manifest.json`: code, map, inputs, and scientific safeguards.

The single-view `score-select` result is useful as an initial diagnostic. Final
pilot selection should use both projection directories and the frozen battery:

```bash
python analysis/scripts/build_readiness_prompt_population.py spatial-select \
  --plan-dir "$POPULATION_ROOT/plan" \
  --candidates "$POPULATION_ROOT"/candidates/*-round-00.jsonl \
  --reference-projections "$POPULATION_ROOT/projections/qwen" \
  --candidate-projections "$POPULATION_ROOT/projections/mistral" \
  --robustness-battery "$BATTERY_ROOT" \
  --validations "$POPULATION_ROOT/validation/llama3.3-70b.jsonl" \
  --generator-ids gemma4-31b,qwen3-32b \
  --distance-tolerance 0.22 \
  --disagreement-weight 0.10 \
  --next-round-index 1 \
  --output-dir "$POPULATION_ROOT/spatial-selection-round-00"
```

The spacing gate requires all 30 cells, mean target distance at most 0.25, at
least 80% of cells within tolerance, at least 0.70 observed span on each axis,
median nearest-neighbor distance at least 0.08, and at least 18 occupied bins.
These thresholds are fixed diagnostics for prompt-population construction, not
scientific findings and not definitions of `B`.

## Refinement

Generate the round-1 task file with the assigned models. Then score all prior
and new candidate files together into a new immutable output directory:

```bash
python analysis/scripts/build_readiness_prompt_population.py score-select \
  --plan-dir "$POPULATION_ROOT/plan" \
  --map "$MAP_ROOT/readiness_embedding_map.json" \
  --candidates "$POPULATION_ROOT"/candidates/*-round-00.jsonl \
               "$POPULATION_ROOT"/candidates/*-round-01.jsonl \
  --embedding-model "$QWEN3_8B_SNAPSHOT" \
  --mntp-model "$LLM2VEC_MNTP_SNAPSHOT" \
  --peft-model "$LLM2VEC_UNSUP_SNAPSHOT" \
  --next-round-index 2 \
  --output-dir "$POPULATION_ROOT/selection-round-01"
```

Stop when the next-round task file is empty, or at the preregistered maximum
round. Do not tune distance thresholds after looking at downstream reranking
outcomes.

For the frozen Qwen/Mistral round-1 pilot, the JUPITER wrapper
`analysis/scripts/slurm/jupiter/run_readiness_prompt_round1.sh` performs both
projections, comparison, spatial selection, and a final continuity/artifact
audit inside an existing allocation. Completed projection stages are verified
and skipped; partial immutable outputs fail closed. The wrapper exits with
status 3 when artifact checks pass but the preregistered spatial gate still
requires refinement, preventing an accidental scale-up to 30,000 questions.

## CPU plumbing test

`generate --backend fake` validates files and restart behavior without loading a
model. Fake output is never scientific evidence.
