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
