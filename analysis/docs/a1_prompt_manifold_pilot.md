# A1 decision-readiness prompt-manifold pilot

This experiment varies only the business evaluator's decision readiness:

```text
understand the B2B SaaS category
    -> develop practical evaluation criteria
    -> evaluate and shortlist solutions
    -> acquire or implement a solution
```

The exact query, business actor, candidate set, relevance task, neutral source
policy, output size, and output format are compiler-owned fields. The generator
writes only `search_objective_clause`. Assigned A1 is the treatment; pairwise
Bradley--Terry scores, LLM2Vec input embeddings, and LLM2Vec-Gen
anticipated-response embeddings are pre-outcome manipulation checks.

## Construction cycle

1. Generate 12 diverse natural-language candidates per A1 level and style.
2. Judge A1 pairwise in both presentation orders with two independent models.
3. Embed every query-bound candidate with primary LLM2Vec.
4. Embed the same candidates with LLM2Vec-Gen.
5. Select each complete style trajectory jointly with strict realized-A1
   monotonicity.
6. Balance coordinate error, equal local step sizes, low curvature, local
   linguistic continuity, and diversity between style trajectories.
7. Inspect the frozen population. If it fails, generate a new candidate round
   before collecting reranking outcomes; never repair it after seeing outcomes.

All embedding rows are normalized inside the selector, so the two models'
different dimensions and raw norms cannot dominate merely by scale. Smoothness
is measured separately in each space. No latent vector is decoded.

Start with 7 A1 levels, four styles, and 12 candidates per level. This produces
336 raw prompts and 28 selected prompts. Scale the number of styles only after
strict monotonicity, human semantic review, and acceptable tortuosity and local
step variation on the development run.

Generation caches each accepted candidate independently. If a generation job
ends before the final candidate and comparison manifests are written, rerun the
same pinned configuration with `generate --resume`. Resume refuses completed or
unrelated directories and revalidates every cached identity and objective
against the current structural contract before reuse. Endpoint validation
treats assessment as evaluation language while preserving negated forms such
as “before assessing” at the informational endpoint.

## Query-conditioned randomized study

The selected 28-prompt manifold is frozen once, then crossed with every real
search term. This is a randomized complete-block design:

- `assigned_a1` is the semantic treatment;
- `style_seed` is the surface-realization factor;
- `search_term` is the block;
- every search term receives all 7 A1 levels and all 4 styles;
- seeded randomization determines search-term order and prompt order within a
  search term, but never changes treatment membership.

The canonical top-20 pools are engine-specific: SearXNG contains 1,009 search
terms and DuckDuckGo contains 1,011. A SearXNG schedule therefore contains 28
prompts per term and 28,252 assignments. Using the complete calibrated 7 x 4
manifold is preferable to rounding up to 30 with duplicated or uncalibrated
prompts.

Build the prompt-only schedule from the canonical SearXNG pool and the frozen
pilot artifacts:

```bash
python3 analysis/scripts/build_a1_query_panel.py \
  --selected-manifold "$A1_OUTPUT/selected_a1_prompt_manifold.jsonl" \
  --source-run-manifest "$A1_OUTPUT/run_manifest.json" \
  --serp-parquet "$GEODML_DATA_ROOT/data/serp/phase0_top20_searxng.parquet" \
  --expected-keywords 1009 \
  --master-seed 20260817 \
  --output-dir "$A1_QUERY_PANEL_OUTPUT"
```

The builder binds the literal query into each prompt while leaving
`{CANDIDATES}` and `{TOP_N}` for the downstream reranking stage. It writes the
randomized JSONL schedule, diagnostics, a source-hashed run manifest, and a
short report atomically. It refuses an existing output directory. This stage
does not invoke a model, alter candidate sets, or observe outcomes.

## Dense 30-level semantic study

The 7-level pilot validates the construction method but is not dense enough for
the final semantic-axis study. Do not assign new continuous A1 coordinates to
those 28 existing texts. Instead, construct and calibrate a new manifold at 30
assigned levels.

`--randomized-a1-levels 30` creates a deterministic stratified-random grid:

- A1=0 and A1=1 are fixed semantic anchors;
- each of the 28 interior coordinates is jittered around one equal-width grid
  location using `--master-seed`;
- jitter is bounded so coordinates remain strictly increasing and cover the
  whole axis without clusters;
- the generated objective clause, pairwise calibration, and dual embeddings
  are recomputed at every new coordinate.

With four surface styles and 12 generation candidates per cell, the dense
construction has 1,440 raw candidates and selects 120 calibrated source prompts
(30 levels x 4 styles). For the final query panel, use
`--style-assignment balanced-one-per-a1`. Each query then receives all 30 A1
levels exactly once. A seeded balanced cycle chooses the style at each level,
so each of the four styles occurs either seven or eight times per query.

For the canonical 1,009-query SearXNG top-20 pool, the final schedule therefore
contains 30,270 query-bound prompts. A1 is the semantic treatment, query is the
block, and style is randomized surface variation. The final schedule does not
reuse the 7-level pilot as if it had 30 semantic coordinates.

Start the new dense manifold in a new run directory:

```bash
srun -n1 --gres=gpu:1 python3 \
  analysis/scripts/run_a1_prompt_manifold_pilot.py generate \
  --output-dir "$A1_DENSE_OUTPUT" \
  --search-term "abandoned cart recovery" \
  --generator-model "$GENERATOR_SNAPSHOT" \
  --precision full \
  --randomized-a1-levels 30 \
  --style-seeds 0,1,2,3 \
  --number-candidates 12 \
  --master-seed 20260817 \
  --temperature 0.9 \
  --max-new-tokens 500 \
  --maximum-attempts 8
```

After completing the same two-judge, dual-embedding, and selection stages used
by the pilot, build the dense query schedule:

```bash
python3 analysis/scripts/build_a1_query_panel.py \
  --selected-manifold "$A1_DENSE_OUTPUT/selected_a1_prompt_manifold.jsonl" \
  --source-run-manifest "$A1_DENSE_OUTPUT/run_manifest.json" \
  --serp-parquet "$GEODML_DATA_ROOT/data/serp/phase0_top20_searxng.parquet" \
  --expected-keywords 1009 \
  --master-seed 20260817 \
  --style-assignment balanced-one-per-a1 \
  --output-dir "$A1_DENSE_QUERY_OUTPUT"
```

## Primary embedding-coordinate correction

For the semantic-vector study, `assigned_a1` in the generation bank is a
proposal coordinate only. It helps generate broad candidate coverage but does
not define the final A1 value. Qwen pairwise judgments are auxiliary validation
only.

The primary A1 coordinate is identified in LLM2Vec input-prompt space. For each
of the 1,009 SearXNG queries and each of four surface frames, construct a matched
pair of complete listwise reranking prompts. Query, actor, ranking task, source
policy, candidate placeholder, output contract, and surface frame are identical
within the pair. Only search purpose differs:

```text
informational: understand mechanisms, uses, applications, and limitations
transactional: evaluate, compare, shortlist, select, acquire, or implement
```

All embedding rows are unit-normalized. The primary vector is the normalized
mean of the 4,036 paired transactional-minus-informational differences. Global
endpoint centroids orient and scale projection so their means are zero and one.
This makes the query corpus the topic prior while paired differences remove
query and surface offsets.

Fit this axis before positioning any generated candidates:

```bash
srun -n1 --gres=gpu:1 python3 analysis/scripts/fit_a1_embedding_axis.py \
  --output-dir "$A1_EMBEDDING_AXIS_OUTPUT" \
  --serp-parquet "$GEODML_DATA_ROOT/data/serp/phase0_top20_searxng.parquet" \
  --expected-keywords 1009 \
  --embedding-model "$QWEN25_SNAPSHOT" \
  --mntp-model "$LLM2VEC_MNTP_SNAPSHOT" \
  --peft-model "$LLM2VEC_SIMCSE_SNAPSHOT" \
  --style-seeds 0,1,2,3 \
  --encode-batch-size 1 \
  --encode-max-length 512 \
  --query-chunk-size 32
```

Endpoint embeddings are cached atomically by query chunk. Use `--resume` after
an interruption. Before candidate scoring, inspect the positive pair-gap rate,
positive query-mean-gap rate, minimum query-mean gap, and gap variation. The
next stage may proceed only if the corpus-level vector consistently orients
informational prompts below transactional prompts.

The final 30,270 prompts will be complete query-bound reranking instructions.
Each final prompt's A1 value will be its LLM2Vec projection onto the frozen
query-prior vector, computed before candidate rankings or outcomes. Generator
proposal coordinates and judge scores will remain in the artifacts for error
and agreement analysis, but will not replace the embedding coordinate.
