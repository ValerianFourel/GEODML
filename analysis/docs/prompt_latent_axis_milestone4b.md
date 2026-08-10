# Query-conditioned informational-to-transactional prompt latent axis

## Correction to the deterministic scaffold

The five-phrase `SearchPurposeTemplateGenerator` is only a smoke-test fallback.
It does not define the scientific prompt trajectory.

The scientific mechanism is:

```text
paired informational/transactional endpoint prompts
    -> embed endpoints
    -> learn and freeze a direction
    -> generate many query-conditioned prompt candidates
    -> embed and project candidates
    -> select candidates nearest assigned target coordinates
    -> render with a fixed candidate pool
    -> obtain strict candidate-ID rankings
```

An arbitrary point in an embedding space cannot be reliably decoded directly
into natural language. Candidate generation followed by projection and
selection makes this limitation explicit and auditable.

## Conditioning on query

Endpoint prompts are paired by underlying query. The direction is the normalized
mean of the paired transactional-minus-informational embedding differences.
This reduces query/topic offsets when learning the axis.

Candidate generation receives the real query, target coordinate, and style
seed, but never receives candidate pages, their order, or ranking outcomes.
Generated templates retain `{QUERY}`, `{CANDIDATES}`, and `{TOP_N}`. The real
query and frozen candidate evidence are inserted only after a prompt is selected.

For each query, all target coordinates use the same candidate pages in the same
presentation order. This permits within-query ranking-trajectory comparisons.
If retrieval itself is allowed to change, result sets differ and cannot be
treated as permutations of the same candidates.

## Assigned and observed coordinates

The assigned target coordinate is the experimental variable. The observed
embedding projection records where the selected natural-language prompt landed.
It is a diagnostic and does not redefine assignment.

The endpoint-centroid projection is calibrated so the informational centroid is
zero and the transactional centroid is one. Selected prompts may project beyond
that interval; those values are retained rather than silently clipped.

## Current scope

`prompt_latent_axis.py` provides:

- provider-independent prompt generation and embedding protocols;
- paired-endpoint axis construction;
- calibrated projection;
- generate/embed/project/select logic;
- a lazy sentence-transformer adapter;
- a lazy repository-local HF generation adapter;
- deterministic fake providers for CPU tests;
- rendering through stable candidate IDs and strict ranking validation.

Generated prompts remain `latent-selected-unvalidated`. Semantic validation,
endpoint-bank freezing, a cluster pilot manifest, real reranking, and ranking
analysis remain separate milestones. No scientific result is inferred from the
fake providers or unit tests.

## Pilot commands

CPU-only fake-provider smoke run:

```bash
python3 analysis/scripts/generate_latent_prompt_pilot.py \
  --provider fake \
  --data-root ./geodml_data \
  --engine searxng \
  --pool 20 \
  --top-n 10 \
  --max-keywords 2 \
  --target-grid 0,0.25,0.5,0.75,1 \
  --number-style-seeds 2 \
  --number-candidates 3 \
  --output-dir analysis/output/latent_prompt_fake_smoke
```

Cluster-side candidate-generation template, to be executed only inside an
allocated GPU job:

```bash
python3 analysis/scripts/generate_latent_prompt_pilot.py \
  --provider local \
  --generator-model "$PROMPT_GENERATOR_MODEL" \
  --embedding-model all-MiniLM-L6-v2 \
  --precision 4bit \
  --data-root "$GEODML_DATA_ROOT" \
  --engine searxng \
  --pool 20 \
  --top-n 10 \
  --max-keywords 8 \
  --target-grid 0,0.25,0.5,0.75,1 \
  --number-style-seeds 2 \
  --number-candidates 24 \
  --output-dir "$LATENT_PROMPT_PILOT_OUTPUT"
```

The command generates prompt candidates and rendered prompt manifests; it does
not invoke the ranking model.
