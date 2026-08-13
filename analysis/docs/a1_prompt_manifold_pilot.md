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
