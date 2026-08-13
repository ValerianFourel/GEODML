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
or observation. Use the real staged pilot below for pinned instruction-LLM,
pairwise-judge, and embedding runs.

## Real staged semantic pilot

`run_real_two_axis_prompt_pilot.py` replaces all three fake components while
keeping their model lifetimes separate:

1. `generate` loads a pinned local instruction LLM and generates constrained
   objective/source clause pairs. The query is absent from this generation
   request and remains a structural `{QUERY}` field.
2. `judge` loads a pinned local instruction LLM as a blind pairwise semantic
   judge. The same exact query is inserted into both compared prompts. Each
   presentation order remains a separate judgment.
3. `diagnose-judgments` runs on CPU before embedding. It reports the exact
   endpoint slice gaps, per-judge direction, reversed-presentation consistency,
   and endpoint clauses without changing any judgment or treatment coordinate.
4. `embed-select` loads primary LLM2Vec on exactly one visible GPU, embeds the
   query-bound input prompt text, performs constrained selection, and writes
   latent field diagnostics.
5. `response-diagnostics` optionally loads LLM2Vec-Gen on exactly one visible
   GPU and measures anticipated-response geometry for the frozen selection. It
   never decodes a reconstruction state.

On HoreKa, install primary LLM2Vec through
`analysis/scripts/install_llm2vec_runtime.sh`. The script pins the official
Qwen2/Qwen3 support revision and installs it without dependencies. The PyPI
0.2.3 package imports a Transformers class removed from the cluster runtime;
downgrading Transformers would also disturb the validated LLM2Vec-Gen stack.
For adapter-only Qwen checkpoints, pass all three layers explicitly to
`embed-select`: the complete Qwen2.5 checkpoint as `--embedding-model`, the
MNTP adapter as `--mntp-model`, and the SimCSE adapter as `--peft-model`. The
loader merges MNTP into the bidirectional base before attaching SimCSE. Passing
the adapter-only MNTP directory as `--embedding-model` is invalid because it
does not contain the base model configuration or weights.

All local-model outputs are cached by stable request/model/configuration hashes.
The cache identity also includes the semantic-contract version, so a stricter
coordinate-direction screen cannot reuse candidates accepted under an older
screen. Candidate generation rejects and retries clauses whose A1 meaning or A2
ownership direction contradicts the assigned cell. Each candidate slot is
generated and validated independently; an accepted slot is retained while only
its failed slot is retried. This prevents a multi-candidate request from using
semantic-axis variation to make its outputs superficially distinct.
Comparative ownership clauses are parsed by the first ownership object governed
by the preference verb. Thus, "prefer vendor-controlled evidence over
seller-independent evidence" is correctly directional rather than treated as
simultaneous preference for both poles.
Generated clauses may not state a numeric or spelled-out candidate-set size;
candidate cardinality remains a structural property of `{CANDIDATES}`.
For Qwen3 structured requests, thinking is disabled through the chat template.
Transport wrappers such as a residual thinking block or Markdown JSON fence are
ignored only when they contain exactly one unambiguous schema-valid object. If
all retries fail, the raw responses and validation errors are retained in a
`*.failed.json` cache artifact named in the exception.
Assigned A1/A2 remain treatments; Bradley--Terry and LLM2Vec values remain
manipulation checks. Start with a 3×3×1 development run. A 7×7×24 run is a
later scale-up after examining every selected prompt and judge disagreement.

The real run produces:

- `two_axis_candidates.jsonl`;
- `pairwise_comparison_requests.jsonl`;
- `pairwise_judgments.jsonl`;
- `pairwise_judgment_diagnostics.json`;
- `pairwise_judgment_report.md`;
- `candidate_calibrations.jsonl`;
- `selected_prompt_population.jsonl`;
- `selection_diagnostics.json`;
- `llm2vec_latent_diagnostics.json`;
- `llm2vec_gen_response_diagnostics.json`;
- `real_two_axis_prompt_report.md`;
- stable raw generation and judgment caches.

`run_manifest.json` records the Git SHA, Slurm environment, models, seeds, grids,
precision, generation configuration, and completion status. A real pilot is
still labelled `scientific_result: false` until human semantic review and the
pre-specified gates have been applied.
