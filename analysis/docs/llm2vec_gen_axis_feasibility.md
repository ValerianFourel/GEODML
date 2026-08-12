# LLM2Vec-Gen informational-to-transactional axis feasibility

## Purpose

This milestone tests a narrower claim than the existing
generate/embed/project/select pipeline:

> Paired informational and transactional reranking instructions may define a
> topic-general direction in LLM2Vec-Gen reconstruction-state space, and points
> on that direction may remain decodable as natural-language instructions.

That is a hypothesis, not an established property of LLM2Vec-Gen. The official
API documents reconstruction of states returned by `encode`; it does not claim
that arithmetic interpolation between independently encoded states stays on a
valid text manifold.

This search-purpose axis is a separate experiment from the first-party product-
source policy axis. It does not change or reinterpret the latter.

## Representation and axis

`McGill-NLP/LLM2Vec-Gen-Qwen3-8B` returns two representations:

- a pooled embedding with shape `(batch, hidden_dim)`;
- reconstruction hidden states with shape
  `(batch, compression_tokens, hidden_dim)`.

Only the reconstruction states are passed to `generate`. Consequently, the
decodable axis is estimated in that full state space. The pooled-embedding axis
is retained as an independent geometry diagnostic and is never decoded.

For paired endpoint states `H_i^info` and `H_i^trans`, the fitted direction is:

```text
D = mean_i(H_i^trans - H_i^info)
```

The global centroid path at assigned coordinate `I` is:

```text
H(I) = mean_i(H_i^info) + I * D
```

The pilot also decodes topic-matched paths:

```text
H_i(I) = H_i^info + I * (H_i^trans - H_i^info)
```

The topic-matched path is the less aggressive test. Averaging hidden states
across topics may leave the learned manifold, so the global-centroid path is
explicitly labelled a stress test.

## What counts as evidence

The script reports paired direction cosines and leave-one-pair-out geometry. In
each leave-one-pair-out fold, the direction is estimated without one topic and
the held-out informational-to-transactional displacement is projected onto it.
This avoids treating the mechanically positive training-centroid separation as
validation.

The axis is worth a second milestone only if all of the following hold on the
real model output:

1. held-out pair gaps are consistently positive in reconstruction space;
2. decoded endpoints preserve the intended endpoint meanings;
3. intermediate decoded texts change monotonically in action proximity;
4. the query, candidates, output count, and identifier-only format remain fixed;
5. no new criteria such as authority, popularity, freshness, commerciality, or
   source ownership appear;
6. interpolation produces fluent, usable templates rather than mixtures or
   degenerate text.

The script measures item 1 and structural pieces of item 4. Items 2, 3, 5, and
6 require review of `decoded_latent_grid.jsonl`. It also decodes and re-encodes
each generated template, then reports whether its projected coordinate remains
monotonic. That same-model cycle check catches collapse and reversals, but it is
not independent semantic validation. A positive geometry report alone must not
be described as a working semantic continuum.

## Query strategy

The first feasibility run uses post-decode injection only. Endpoint templates
contain literal `{QUERY}`, `{CANDIDATES}`, and `{TOP_N}` placeholders. The axis is
estimated and decoded first; a probe query is substituted for `{QUERY}` only in
a separate output field.

This keeps the *probe query* out of the axis estimate. The versioned endpoint
bank is still topic-conditioned; paired differences and leave-one-topic-out
checks are used to reduce and diagnose topic offsets. Query-vector addition is
deferred because it would add another unvalidated latent-arithmetic assumption
before basic interpolation has passed.

An opt-in query-conditioned path tests a different, smaller claim. With
`--query-conditioned-axis`, the exact `--probe-query` text is placed inside both
direct endpoint requests before encoding:

```text
For the fixed search topic "abandoned cart recovery", explain how it works and
what approaches are available so the user can learn and understand it.

For the fixed search topic "abandoned cart recovery", help the user choose a
suitable approach and begin implementing it now.
```

Only the surrounding search purpose changes. The full coordinate grid is
interpolated between this single topic-matched pair and reported under
`query-conditioned-direct-request`. These decoded rows are not expected to
preserve ranking placeholders; ranking structure would be added by a
deterministic wrapper only if the smaller semantic path first proves monotonic.
Because this is a one-query path, it supplies no evidence of topic
generalization and does not test query-vector addition.

## Query-specific multi-prompt centroids

The next feasibility path uses multiple matched requests at both endpoints for
every query. The versioned bank contains six surface frames. Each frame is
filled twice with the same exact query: once with informational intent and once
with buy/adopt intent. Within a matched pair, only the intent clause changes.

For query `q`, the model encodes all endpoint requests and recomputes:

```text
C_info(q) = mean of informational reconstruction states containing q
C_buy(q)  = mean of buy-intent reconstruction states containing q
H(q, B)   = C_info(q) + B * (C_buy(q) - C_info(q))
```

Thus every assigned `H(q, B)` lies exactly on the query-specific centroid line.
The raw decoder is not responsible for preserving literal task inputs. The
validator therefore also places every raw latent realization below a fixed
`Fixed query: "..."` anchor and separately re-encodes that anchored treatment.
This keeps the query identical at every coordinate while leaving the decoded
search-purpose realization as the stochastic semantic component. Raw and
anchored diagnostics are both retained; anchoring is not counted as evidence
that the latent decoder itself preserved the query.

For a wider feasibility stress test, the validator calls the extended latent
coordinate `L` and defaults to 13 evenly spaced points from `L=-1` through
`L=2`. `L=-1` is one complete centroid displacement before the informational
centroid; `L=2` is one complete displacement beyond the buy centroid. Only
coordinates in `[0, 1]` are experimental `B`; extrapolated values are explicitly
marked as diagnostic probes. The first decoded realization is also separated
from leaked model control tokens before anchoring and re-encoding.

The extended run records two decode-cycle definitions. The original direct
cycle encodes the decoded sentence as a new model input. The instruction-matched
cycle instead asks the model to reproduce that exact realization, preserving
the output-reconstruction task implied by the endpoint requests. Reports include
the exact adjacent decreases and ties for the matched cycle, exact duplicate
realization groups, and a narrow diagnostic for drift into a first-person
shopper recovering their own cart.
The scientific diagnostic is what happens after decoding and re-encoding:
whether the query is retained, the recovered coordinate is monotonic, and the
recovered state remains near the line rather than acquiring a large orthogonal
residual. Leave-one-frame-out geometry measures robustness to surface wording
for that query only; it is not an unseen-query test.

The dedicated CLI does not run the older placeholder or global-centroid paths:

```bash
export QUERY_CENTROID_OUTPUT="$PWD/runs/query_centroid_axis/${SLURM_JOB_ID}-$(git rev-parse --short HEAD)"

srun --ntasks=1 --gres=gpu:1 python3 \
  analysis/scripts/validate_query_centroid_axis.py \
  --backend local \
  --model "$MODEL_SNAPSHOT" \
  --query "abandoned cart recovery" \
  --axis-min -1 \
  --axis-max 2 \
  --number-points 13 \
  --encode-batch-size 1 \
  --max-new-tokens 64 \
  --output-dir "$QUERY_CENTROID_OUTPUT"
```

The script writes `query_centroid_diagnostics.json`,
`decoded_query_centroid_grid.jsonl`, `query_centroid_state.npz`, and
`query_centroid_report.md`. The template bank and its hash, all filled endpoint
requests, both centroids, the direction, assigned states, and re-encoded states
are recorded. A deterministic reranking wrapper is deliberately deferred until
the decoded purpose sequence passes semantic review.

## Publisher-ownership field

The scientifically preferred B2B SaaS feasibility experiment replaces the
mixed search-purpose path with one symmetric intervention:

```text
lambda = -1: prefer seller-independent evidence
lambda =  0: no preference based on publisher ownership
lambda = +1: prefer seller-controlled evidence
```

For every exact query, six matched surface frames are encoded in all three
regions. The independent-to-controlled difference defines the direction, while
the separately encoded neutral centroid defines the origin. Its position on and
distance from the endpoint line are measured before decoding. Only the decoded
ownership policy varies inside a deterministic B2B software-evaluator reranking
wrapper; query, candidates, relevance, evidence quality, task, output contract,
time horizon, and company context remain fixed.

```bash
export SOURCE_OWNERSHIP_OUTPUT="$PWD/runs/source_ownership_axis/${SLURM_JOB_ID}-$(git rev-parse --short HEAD)"

srun --ntasks=1 --gres=gpu:1 python3 \
  analysis/scripts/validate_source_ownership_axis.py \
  --backend local \
  --model "$MODEL_SNAPSHOT" \
  --query "abandoned cart recovery" \
  --number-points 13 \
  --encode-batch-size 1 \
  --max-new-tokens 96 \
  --output-dir "$SOURCE_OWNERSHIP_OUTPUT"
```

This writes `source_ownership_diagnostics.json`,
`decoded_source_ownership_grid.jsonl`, `source_ownership_state.npz`, and
`source_ownership_report.md`. A real run is still feasibility-only until the
neutral point, geometry, decoded monotonicity, and semantic invariants pass.

## Two-factor ownership-by-intent plane

The two-factor feasibility design estimates matched factorial corners for each
exact query and surface frame, then constructs:

```text
Z(q,O,I,S) = C(q) + O D_O_orth(q) + I D_I_orth(q) + R_S_orth(q)
```

`O` varies seller-independent to seller-controlled evidence preference, `I`
varies informational to transactional search intent, and `S` selects a surface
residual with its projection on both semantic axes removed. The raw main-effect
cosine, factorial interaction curvature, and corner reconstruction error are
recorded before decoding. Monotonicity is evaluated within fixed-coordinate
slices, not by collapsing the two dimensions into one score.

The default feasibility grid uses `O,I in {-2,-1,0,1,2}` and style seeds `0,1`,
producing 50 prompts. Coordinates beyond `[-1,1]` continue the fitted direction
but are not experimental treatments.

```bash
export OWNERSHIP_INTENT_OUTPUT="$PWD/runs/ownership_intent_plane/${SLURM_JOB_ID}-$(git rev-parse --short HEAD)"

srun --ntasks=1 --gres=gpu:1 python3 \
  analysis/scripts/validate_ownership_intent_plane.py \
  --backend local \
  --model "$MODEL_SNAPSHOT" \
  --query "abandoned cart recovery" \
  --ownership-grid=-2,-1,0,1,2 \
  --intent-grid=-2,-1,0,1,2 \
  --style-seeds 0,1 \
  --encode-batch-size 1 \
  --max-new-tokens 128 \
  --output-dir "$OWNERSHIP_INTENT_OUTPUT"
```

Artifacts are `ownership_intent_plane_diagnostics.json`,
`decoded_ownership_intent_grid.jsonl`, `ownership_intent_plane_state.npz`, and
`ownership_intent_plane_report.md`.

## CPU plumbing smoke test

The fake backend checks artifacts and numerical contracts without loading a
model:

```bash
python3 analysis/scripts/validate_llm2vec_gen_axis.py \
  --backend fake \
  --target-grid 0,0.5,1 \
  --decode-pairs 1 \
  --probe-query "abandoned cart recovery" \
  --output-dir /tmp/geodml-llm2vec-gen-smoke
```

Mock output is labelled non-scientific in both JSON and Markdown.

## HoreKa one-GPU feasibility run

Install the inference-only runtime into the active project environment and
ensure the exact model is already present in the Hugging Face cache available
to compute nodes:

```bash
bash analysis/scripts/install_llm2vec_gen_runtime.sh
```

Do not replace this command with `pip install llm2vec-gen`. The upstream wheel
declares an older full training/evaluation stack, including `torch==2.6.0`,
`transformers==4.56.2`, and a source build of `flash-attn`. The helper installs
the small 0.1.3 package without dependencies and preserves the CUDA-compatible
Torch/Transformers versions already validated for this repository. The code
path used here does not import FlashAttention.

The package's current high-level loader moves the full model to one CUDA device
and its generation path names `cuda` directly. Expose exactly one allocated GPU;
do not reuse the old four-GPU prompt-generator job.

From the login node:

```bash
export HOREKA_ACCOUNT=hk-project-p0026831

salloc --account="$HOREKA_ACCOUNT" \
  --partition=accelerated \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:1 \
  --mem=120G \
  --time=01:00:00
```

From the shell holding that allocation, prepare the environment and launch one
job step. This works whether the allocation shell itself remains on the login
node or HoreKa places it on the compute node:

```bash

cd /path/to/geodml
source .venv311/bin/activate

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export AXIS_OUTPUT="$PWD/runs/llm2vec_gen_axis/${SLURM_JOB_ID}-$(git rev-parse --short HEAD)"

python3 -c 'import llm2vec_gen, torch; print(llm2vec_gen.__version__, torch.cuda.device_count())'

srun --ntasks=1 python3 \
  analysis/scripts/validate_llm2vec_gen_axis.py \
  --backend local \
  --model McGill-NLP/LLM2Vec-Gen-Qwen3-8B \
  --target-grid 0,0.25,0.5,0.75,1 \
  --decode-pairs 2 \
  --probe-query "abandoned cart recovery" \
  --query-conditioned-axis \
  --encode-batch-size 1 \
  --max-new-tokens 180 \
  --output-dir "$AXIS_OUTPUT"
```

The local adapter deliberately rejects zero or multiple visible GPUs before
loading the model.

## Artifacts

The run writes atomically and refuses to overwrite by default:

- `axis_diagnostics.json`: model/configuration identity, representation hashes,
  pooled and reconstruction geometry, and leave-one-pair-out rows;
- `decoded_latent_grid.jsonl`: decoded global and topic-matched grid points,
  placeholder checks, and optional post-decode query substitution;
- `axis_state.npz`: endpoint states, centroids, and unit direction for inspection;
- `axis_feasibility_report.md`: compact review summary.

No SERP data, candidate pages, reranker, DML estimator, or mocked scientific
result is used in this milestone.

## Upstream references

- LLM2Vec-Gen implementation: <https://github.com/McGill-NLP/llm2vec-gen>
- Qwen3-8B model card: <https://huggingface.co/McGill-NLP/LLM2Vec-Gen-Qwen3-8B>
- Paper: <https://arxiv.org/abs/2603.10913>

## Smallest next step after a successful run

Review and label every decoded row for endpoint fidelity, monotonic action
proximity, off-axis criteria, and format preservation. Only after those checks
pass should a later milestone add seeded orthogonal surface variation or compare
post-decode query insertion with a carefully specified query-vector operation.
