# Prompt latent validation and reranking-permutation mapping

The experimental object is a natural-language prompt generated before it is
embedded:

```text
(assigned B, randomized S) -> P = G(B, S)
P -> embedding E(P) -> observed latent diagnostics
(P, fixed query, fixed candidates) -> reranker -> permutation Y
```

`B` remains the assigned intensity of preference for first-party
software-product sources. `S` changes surface realization only. The embedding
projection locates and validates `P`; it neither defines `B` nor decodes the
final prompt.

## Axis construction

For every surface seed, matched prompts at `B=0` and `B=1` are embedded. The
normalized mean of the matched endpoint differences defines the frozen policy
direction. This pairing removes the surface realization shared by both
endpoints. Leave-one-style-out signs and pair-direction cosines diagnose whether
the same semantic contrast is present across surface forms.

Every prompt receives:

- its assigned `B`;
- its observed calibrated axis coordinate;
- absolute assigned-versus-observed coordinate error;
- a matched-style off-axis residual;
- its full prompt embedding;
- a unique assignment ID derived from prompt ID, `B`, and `S`;
- stable prompt, embedding, axis, and instance hashes.

Acceptance thresholds are explicit inputs. Validation can reject a prompt or a
nonmonotonic within-`S` trajectory, but it never changes the assigned treatment.

## Mapping to the outcome

The prompt is rendered with the exact query and one frozen, ordered candidate
set. The reranker must return exactly `TOP_N` candidate identifiers. Invalid,
duplicate, unknown, explanatory, or wrong-length outputs fail validation; there
is no silent fallback.

The outcome record directly links:

```text
assigned B, S, prompt ID, observed latent coordinate,
candidate-set ID, ordered candidate IDs, source-position vector,
prompt-embedding hash, permutation hash, reranker run ID, and reranker model
```

This is the desired prompt-latent-to-permutation map. Causal analyses use
assigned `B` as treatment and permutation-derived outcomes as responses.
Observed embedding coordinates are diagnostics or descriptive measurements of
the realized prompt, not replacement treatments or confounders.

## Current scope

`prompt_policy_mapping.py` is provider-independent and performs no inference at
import time. It includes a deterministic fake embedder only for CPU tests. A
real embedding model and real reranker are separate cluster execution steps.
The exact legacy neutral and biased pipelines remain unchanged.

CPU-only contract smoke test:

```bash
python3 analysis/scripts/map_policy_prompts_to_permutations.py \
  --backend fake \
  --b-grid 0,0.25,0.5,0.75,1 \
  --number-style-seeds 4 \
  --max-coordinate-error 0.05 \
  --max-off-axis-residual 0.05 \
  --output-dir analysis/output/policy_prompt_mapping_smoke
```

For a real prompt-bank validation, replace `--backend fake` with
`--backend sentence-transformer --embedding-model <pinned-model>`. To create
the rendered prompt handoff, also supply `--query` and `--candidates-json`.
After the reranker produces one JSONL response per `prompt_assignment_id`, rerun
with `--responses-jsonl`; the script validates and writes strict permutation
records. This assignment-level key remains unique even when a finite phrase
schedule yields identical prompt text at adjacent `B` values.
