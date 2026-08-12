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
