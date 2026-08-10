# Search-purpose continuum: deterministic foundation

> **Scientific-path correction:** this document describes a deterministic smoke
> scaffold only. The actual prompt trajectory is now learned and sampled in
> prompt-embedding space through generate/embed/project/select. See
> `prompt_latent_axis_milestone4b.md`. Do not use the five phrases below as the
> scientific informational-to-transactional continuum.

## Scientific question

This milestone prepares an experiment asking how LLM page rankings change as
the user's assigned search purpose moves from informational understanding to
immediate action.

The assigned variable is:

```text
I in [0, 1]
```

- `I = 0`: learn and understand the topic without selecting or carrying out a
  solution;
- `I = 1`: complete the concrete action implied by the query now.

Intermediate values move through exploration, evaluation, and preparation.
The axis is action proximity, not commerciality: actions can include selecting,
downloading, registering, configuring, or deploying without requiring a
purchase.

The former first-party continuum and the exact legacy neutral/biased prompts
remain available as separate historical and policy-axis experiments.

## Template, keyword, and candidates

The generator produces a template from assigned intent and surface style:

```text
T = G(I, S)
```

The experiment then renders the template with a real keyword and a frozen
candidate set:

```text
P = render(T, Q, C)
```

`Q` and `C` remain identical along an `I` trajectory. Each page receives a
local identifier such as `C001`; the manifest stores the exact mapping back to
URL, title, snippet, and original SERP position.

Model output must contain exactly `TOP_N` unique known identifiers and nothing
else. Invalid output raises an error. The experiment does not silently replace
failed output with the original SERP ranking.

## Current limitation

`SearchPurposeTemplateGenerator` uses five deterministic phrases. It is an
engineering scaffold for CPU tests, manifest inspection, and end-to-end smoke
runs. It is not a scientifically validated continuous semantic generator.

DataForSEO intent categories and probabilities can describe existing keywords,
but are observational classifier outputs. They do not define assigned `I`, and
classification probability must not be interpreted as action intensity.
Navigational intent is outside this one-dimensional axis.

No model inference, prompt embedding, semantic judging, reranking, DML, or
scientific inference is performed in this milestone.

## Local prompt-only pilot

After downloading the frozen SERP tables, run:

```bash
python3 analysis/scripts/generate_search_purpose_pilot.py \
  --data-root ./geodml_data \
  --engine searxng \
  --pool 20 \
  --top-n 10 \
  --max-keywords 8 \
  --intent-grid 0,0.25,0.5,0.75,1 \
  --number-style-seeds 2 \
  --output-dir analysis/output/search_purpose_pilot
```

This produces:

- `search_purpose_prompt_instances.jsonl`;
- `search_purpose_pilot_report.md`.

The manifest contains complete rendered prompts and candidate mappings, but no
rankings because no model is invoked.

## Superseded next step

The latent-axis candidate generator and selector are implemented in
`prompt_latent_axis.py`. Its selected prompts still require semantic validation
before a cluster reranking runner may use them.
