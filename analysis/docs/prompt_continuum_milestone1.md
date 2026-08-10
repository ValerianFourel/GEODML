# Prompt continuum: Milestone 1

The new prompt family represents a reranking instruction as `P = G(B, S)`:

- `B` (`assigned_bias`) is a value in `[0, 1]` assigned on the experimental
  policy axis. It changes only the strength of preference for first-party
  software-product sources. `B = 0` expresses no source-type preference;
  `B = 1` expresses a strong preference while retaining relevance.
- `S` (`style_seed`) is an integer used by a local random-number generator to
  select surface wording, syntax, clause order, tone, verbosity, and
  output-contract wording. `B` never enters the style-plan calculation.
- `G(B, S)` is currently `TemplatePromptGenerator`, which returns a typed
  `PromptRecord` containing the normalized template, structured style plan,
  stable SHA-256 identity, and generation metadata.

`TemplatePromptGenerator` is an engineering scaffold for unit tests and small
CPU smoke runs. It maps continuous `B` onto a finite monotonic set of hand-written
preference phrases; it is not the final scientific prompt generator. Templates
retain `{QUERY}`, `{CANDIDATES}`, and `{TOP_N}` for later rendering.

Later milestones intentionally own learned or LLM-based prompt generation,
prompt embeddings and judges, integration with reranking runs, run manifests,
new outcomes, dataset changes, and DML estimation. The historical neutral and
biased prompt pipeline remains unchanged and separate.
