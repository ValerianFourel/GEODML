# Two-axis calibrated prompt-population pilot

This milestone replaces free latent interpolation and decoding with:

```text
constrained candidate clauses
    -> frozen prompt compiler with structural {QUERY}
    -> blind A1/A2 comparisons in both presentation orders
    -> separate Bradley--Terry realized coordinates R1/R2
    -> prompt embeddings for geometric diagnostics
    -> global constrained selection of complete style trajectories
    -> frozen prompts linked to strict reranking permutations
```

## Scientific objects

- `assigned_a1`: understand the B2B SaaS category → select/acquire a solution;
- `assigned_a2`: prefer seller-independent evidence → no ownership preference
  → prefer seller-controlled evidence, conditional on relevance;
- `realized_a1`, `realized_a2`: pairwise-comparison manipulation checks;
- `prompt_embedding`: representation of the final prompt text;
- ranking permutation: downstream response from a frozen candidate set.

Assigned coordinates remain the randomized causal treatments. Realized
coordinates do not silently replace them. The exact query is inserted only when
the selected template is rendered, so clause generation cannot corrupt it.

## Global selection

The selector uses binary mixed-integer optimization for each complete style
trajectory in deterministic seed order. It carries previously selected hashes
forward as exclusions, so no exact prompt is reused across trajectories. For
every style trajectory it requires:

- exactly one prompt per target cell;
- no selected prompt hash reused;
- row-wise nondecreasing realized A1;
- column-wise nondecreasing realized A2;
- an optional maximum neighboring embedding distance.

The objective minimizes squared realized-coordinate error plus a small length
imbalance penalty. The deterministic trajectory order is recorded by the style
seeds; this v1 selector is not claimed to find the single joint optimum across
all style families. An infeasible bank fails explicitly rather than being
silently repaired after reranking outcomes are observed.

## CPU smoke test

The command below executes a complete fake 7×7×2 contract run. Fake generation,
judging, and embeddings support no scientific claim.

```bash
python3 analysis/scripts/build_two_axis_prompt_population.py \
  --mode fake-complete \
  --search-term "abandoned cart recovery" \
  --style-seeds 0,1 \
  --number-candidates 6 \
  --embedding-backend fake \
  --output-dir analysis/output/two_axis_population_fake
```

The default scientific pilot shape is 7×7 cells, 24 style trajectories, and six
candidates per cell: 1,176 selected prompts from 7,056 raw candidates. The real
run must replace the fake generator and judges with pinned providers and retain
all candidate, comparison, judgment, calibration, embedding, and selection
artifacts.

`--mode mock-bank-only` writes a deterministic **mock** candidate bank and its
blind comparison queue without pretending that either is a scientific request
or observation. A later cluster milestone will add pinned instruction-LLM and
pairwise-judge adapters; no expensive generation or judging is launched by
this milestone.
