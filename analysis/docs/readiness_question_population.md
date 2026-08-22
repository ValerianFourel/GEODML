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

## Support-aware design for 30,000 selected questions

The rectangular 6 by 5 grid remains available for the historical pilot. Large
population construction should use `--target-design support-aware-random`.
This mode:

- uses usable development coordinates only;
- divides normalized two-axis space into support cells;
- excludes empty and low-count cells rather than targeting unsupported corners;
- balances pooled target counts over eligible cells, rather than reproducing
  the density of the original prompt corpus;
- samples a deterministic within-cell interpolation for every keyword target;
- assigns a different seeded target set to each keyword; and
- records the seed and allocation diagnostics in `support_design.json`.

Support-aware generation passes the exact normalized destination to the model as
a graded blend between adjacent semantic anchors. It also deterministically
rotates surface realization—such as direct wording versus a context-first
question—across candidate slots. The surface instruction changes expression,
not the requested information need. The historical rectangular pilot keeps its
original categorical generator prompt and cache identity.

For 1,000 keywords and 30 targets per keyword, the frozen plan contains exactly
30,000 target questions. `--candidates-per-task 4` requests up to 120,000
proposals before validation and spatial matching. Generating more proposals
does not increase the final target count; it gives the assignment step more
choices and can improve realized coverage.

To retain more than 30 selected questions per keyword, increase
`--targets-per-keyword`. Once every eligible support cell has been used for a
keyword, the planner starts another balanced pass and samples a new within-cell
coordinate. This keeps the pooled allocation balanced even when the requested
target count exceeds the number of eligible support cells.

Uniformity means approximately equal pooled target allocation over the
**eligible empirical support area**. It does not mean uniform coverage of the
entire bounding rectangle. That distinction avoids impossible targets in sparse
or empty regions. The allocation is deterministic for a fixed keyword file,
coordinate artifact, configuration, and master seed.

The re-embedded selected bank is audited separately. Passing the large-scale
gate requires pooled target coverage, axis-span retention, at least 80% of
questions within the target tolerance, and a bounded 10 by 10 target-versus-
observed histogram total-variation distance. Thus a uniform target plan cannot
be mistaken for a uniform realized prompt bank.

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

### Latent-feedback proposal loop

`analysis/scripts/generate_llm2vec_gen_feedback_proposals.py` implements the
bounded bridge-and-feedback version of this proposal source. It does not assume
that the two spaces are identical.

1. Join development-only corpus text to the frozen supervised-subspace
   coordinates and deterministically retain at most the configured calibration
   count.
2. Encode those texts as reconstruction tasks with LLM2Vec-Gen and fit a
   ridge-regularized affine bridge from the two normalized readiness coordinates
   to changes in the full reconstruction state.
3. Start from an already valid, stochastically generated question. Re-embed the
   question with the frozen LLM2Vec readiness map to measure its actual
   coordinate.
4. Move its reconstruction state along the two fitted directions. Each round
   tries several bounded step scales while preserving the seed state's residual
   content and surface realization.
5. Decode each proposed state, deterministically attach the exact keyword if the
   decoder omitted it, and apply the ordinary 8--60 word, one-line, one-question
   contract.
6. Ask an independent model to check topic relevance, genuine search intent,
   web answerability, standalone wording, and natural language.
7. Re-embed every surviving final question with the frozen map. Continue from
   the closest valid result, stop inside the configured tolerance, or fail closed
   after the bounded number of rounds.

The bridge is a local proposal controller, not an inverse map. Exact equality to
a continuous target is neither assumed nor required. By default, the script
emits only independently valid questions inside the target tolerance;
`--emit-best-effort` is an explicit diagnostic opt-in.

The calibration can be supplied as one prepared JSONL or built directly from
the existing corpus and frozen coordinate artifacts. A small pilot invocation
has this form:

```bash
python analysis/scripts/generate_llm2vec_gen_feedback_proposals.py \
  --tasks "$PLAN_ROOT/generation_tasks_round_00.jsonl" \
  --initial-candidates "$ROUND_ROOT/qwen_candidates.jsonl" \
  --calibration-corpus "$READINESS_CORPUS" \
  --calibration-coordinates "$MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
  --map "$MAP_ROOT/readiness_embedding_map.json" \
  --bounds "$PLAN_ROOT/subspace_bounds.json" \
  --embedding-model "$LLM2VEC_MODEL" \
  --mntp-model "$LLM2VEC_MNTP_MODEL" \
  --peft-model "$LLM2VEC_PEFT_MODEL" \
  --llm2vec-gen-model "$LLM2VEC_GEN_MODEL" \
  --judge-model "$INDEPENDENT_JUDGE_MODEL" \
  --judge-backend api \
  --judge-cache-dir "$CACHE_ROOT/latent-feedback-judge" \
  --maximum-calibration-items 512 \
  --maximum-rounds 3 \
  --step-scales 0.5,1.0,1.5 \
  --coordinate-step-limit 0.35 \
  --distance-tolerance 0.12 \
  --limit 10 \
  --output-dir "$ROUND_ROOT/llm2vec-gen-feedback-pilot"
```

The output directory contains:

- `feedback_proposals.jsonl`, compatible with `import-proposals`;
- `feedback_results.jsonl`, one terminal result per seed question;
- `feedback_trace.jsonl`, every latent step, raw decode, validation decision,
  final measured coordinate, and distance;
- `bridge_diagnostics.json`, without the high-dimensional bridge weights;
- `bridge_state.restricted-local.npz`, the fitted coordinate mean, state mean,
  and two full reconstruction-state directions; and
- `run_manifest.json`, recording input hashes, model identities, Git SHA,
  controller settings, Slurm metadata when present, and the scientific guard.

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

For the support-aware large-population design, use:

```bash
python analysis/scripts/build_readiness_prompt_population.py plan \
  --keywords "$KEYWORDS_JSONL" \
  --map "$MAP_ROOT/readiness_embedding_map.json" \
  --reference-coordinates "$MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
  --generator-ids gemma4-31b,qwen3-32b \
  --target-design support-aware-random \
  --targets-per-keyword 30 \
  --support-grid-resolution 20 \
  --minimum-support-bin-count 3 \
  --candidates-per-task 4 \
  --master-seed 20260820 \
  --output-dir "$POPULATION_ROOT/plan-support-aware"
```

Before GPU generation, verify the frozen CPU plan:

```bash
wc -l \
  "$POPULATION_ROOT/plan-support-aware/keyword_target_grid.jsonl" \
  "$POPULATION_ROOT/plan-support-aware/generation_tasks_round_00.jsonl"

python -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d, indent=2))
assert d["pooled_target_count"] == 30000
assert d["target_bin_count_range"] <= 1
' "$POPULATION_ROOT/plan-support-aware/support_design.json"
```

The first count is the selected-bank target size, not the proposal count. Each
task carries `requested_candidate_count`, so four proposals per target produce
up to 120,000 generated candidates. Run a smaller frozen subset first and do
not launch the full generation until both embedding projections and the pooled
spatial audit pass for that subset.

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
`--limit` support Slurm arrays or short wall-time slices. `--shard-count` and
`--shard-index` deterministically partition one generator across data-parallel
GPU workers. `--maximum-runtime-seconds` stops cleanly between tasks. Each
accepted question is atomically checkpointed in the task cache immediately,
including partial multi-candidate tasks, before generation continues.

For a four-GPU throughput pilot, use
`analysis/scripts/slurm/jupiter/run_readiness_30k_four_gpu_pilot.sh` inside an
approved four-GPU allocation. It runs two workers per generator, stops
generation after 50 minutes by default, audits the combined candidate bank, and
writes `throughput_summary.json` with projected full-run GPU-hours.

Before validation or projection, audit a balanced generation slice for wording
collapse.  This check removes each exact keyword phrase before comparing the
questions, so copying one question frame across topics cannot masquerade as
diversity merely because the topic text changed:

```bash
python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
  --questions \
    "$POPULATION_ROOT/candidates/qwen3-32b-round-00-pilot.jsonl" \
    "$POPULATION_ROOT/candidates/gemma4-31b-round-00-pilot.jsonl" \
  --minimum-delexicalized-unique-fraction 0.90 \
  --maximum-template-fraction 0.01 \
  --minimum-median-keyword-unique-fraction 0.90 \
  --minimum-keyword-unique-fraction 0.70 \
  --maximum-opening-frame-fraction 0.05 \
  --output-dir "$POPULATION_ROOT/audits/diversity-round-00-pilot"
```

The command exits with status 2 when a diversity check fails and writes the
most frequent delexicalized templates and opening frames for review.  Run the
same audit again on the final `spatially_selected_questions.jsonl`; spatial
coverage and wording diversity are separate acceptance gates.

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
  --require-both-views-within-tolerance \
  --require-delexicalized-template-uniqueness \
  --disagreement-weight 0.10 \
  --next-round-index 1 \
  --output-dir "$POPULATION_ROOT/spatial-selection-round-00"
```

For confirmatory population construction,
`--require-both-views-within-tolerance` is the coordinate-acceptance contract.
A question is retained for a target only when both the frozen Qwen coordinate
and the development-aligned Mistral coordinate independently lie within the
configured Euclidean tolerance. This prevents opposing projection errors from
cancelling in their consensus. Unverified targets remain absent from the
selected bank and are emitted as refinement tasks. The output records both
per-view distances, the joint pass indicator, the tolerance, and immutable
projection identities; this is empirical verification, not a guarantee before
generation.

`--require-delexicalized-template-uniqueness` separately enforces the surface
contract. After replacing the exact keyword phrase with one sentinel and
normalizing case, punctuation, and numbers, the selector retains at most one
copy of any resulting question template. It keeps the jointly verified copy
with the smallest worst-view target distance and emits every removed target as
a refinement task. This makes keyword substitution insufficient for admission.

The broader spacing gate requires all 30 cells, mean target distance at most 0.25, at
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
