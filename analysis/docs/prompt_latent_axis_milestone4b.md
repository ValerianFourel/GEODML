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

## HoreKa Slurm execution

HoreKa uses Slurm, but its account and resource choices are deliberately kept
out of the repository. The dedicated wrapper is separate from the existing
Jülich scripts and does not inherit their `booster`, `jutil`, module, or scratch
settings.

After pulling the exact Git commit on HoreKa, first submit a short validation
job. It checks the virtual environment, cached SERP table, Python dependencies,
CLI import, and the four allocated GPUs without loading model weights:

```bash
export HOREKA_ACCOUNT=YOUR_HOREKA_PROJECT_ACCOUNT
export GEODML_VENV="$PWD/.venv311"
export GEODML_DATA_ROOT="$PWD/geodml_data"

bash analysis/scripts/slurm/horeka/submit_latent_prompt_pilot.sh \
  --validate-only
```

Inspect the resulting `logs/geodml-latent-prompts-<job-id>.out` and `.err`.
Then submit the small generation pilot. The model must already be available in
`HF_HOME` (or supplied as a local path), because offline mode defaults to on:

```bash
export HOREKA_ACCOUNT=YOUR_HOREKA_PROJECT_ACCOUNT
export PROMPT_GENERATOR_MODEL=/path/to/cached/instruction-model
export HF_HOME=/path/to/huggingface-cache
export HOREKA_PARTITION=accelerated
export HOREKA_GPUS=4
export HOREKA_CPUS=16
export HOREKA_TIME=02:00:00
export GEODML_VENV="$PWD/.venv311"
export GEODML_DATA_ROOT="$PWD/geodml_data"
export LATENT_PROMPT_PILOT_OUTPUT="$PWD/runs/latent_prompt_pilot/pilot-v1"

bash analysis/scripts/slurm/horeka/submit_latent_prompt_pilot.sh
```

`HOREKA_MODULES` may contain a space-separated module list when the local venv
depends on cluster modules; it is empty by default. This HoreKa workflow always
requests and runtime-validates exactly four GPUs. The local Hugging Face loader
uses a balanced Accelerate device map, equal per-GPU memory budgets, SDPA
attention, inference mode, KV caching, and TF32 where applicable. A non-four-GPU
override is rejected rather than silently changing the compute design.

For interactive debugging, request the same four-GPU contract on the development
partition and then invoke the job body from the allocated node:

```bash
salloc --account="$HOREKA_ACCOUNT" \
  --partition=dev_accelerated \
  --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --gres=gpu:4 --time=01:00:00

bash analysis/scripts/slurm/horeka/run_latent_prompt_pilot.sbatch
```

The development partition is for smoke tests only. Production generation still
uses the submission helper and the regular `accelerated` partition.

The job records `run_manifest.json` in its output directory with the Git SHA,
configuration, model identifiers, seeds, Slurm allocation, timestamps, log
paths, and completion status. It also writes:

- `prompt_latent_axis.json`;
- `latent_prompt_candidates.jsonl`;
- `rendered_latent_prompts.jsonl`;
- `latent_prompt_pilot_report.md`.

These remain unvalidated prompt candidates. This job does not invoke a ranking
model and does not establish semantic validity or scientific results.
