# Semantic-readiness Hugging Face dataset export

This workflow snapshots the currently retained 20k-panel annotations, embeds
the exact prompt texts with three distinct LLM2Vec views, and produces a
multi-configuration Parquet dataset for Hugging Face.

It deliberately creates two scopes:

- `restricted-local`: all prompts, annotations, failures, and missing tasks;
- `huggingface-safe`: only sources whose frozen source policy allows
  redistribution.

WildChat and every artifact derived from its prompts are excluded from the Hub
scope. A private Hugging Face repository is still a third-party transfer, so
making the repository private does not permit uploading the restricted scope.
The publisher validates this boundary and accepts only the finalized safe
Parquet directory.

The annotation panel need not be complete. Every parser-valid cache is kept;
failed and never-started tasks are explicit audit tables. Re-run assembly into
a new export root after additional annotations arrive.

## 1. Reconnect and check out the exact export code

Replace `HF_EXPORT_COMMIT` with the committed SHA supplied with this runbook.

```bash
ssh -i ~/.ssh/id_ed25519 fourel1@login.jupiter.fz-juelich.de
```

```bash
source "$HOME/geodml_setup.sh"

export GEODML_EXPECTED_COMMIT="HF_EXPORT_COMMIT"
export ORIGINAL_REPOSITORY="$PROJECT/$USER/geodml/src/geodml-mono"
export GEODML_REPOSITORY="$PROJECT/$USER/geodml/src/geodml-mono-$GEODML_EXPECTED_COMMIT"

cd "$ORIGINAL_REPOSITORY"
test -z "$(git status --porcelain)"
git fetch origin codex/semantic-readiness-phase2-jupiter

if [[ ! -e "$GEODML_REPOSITORY" ]]; then
  git worktree add --detach "$GEODML_REPOSITORY" "$GEODML_EXPECTED_COMMIT"
fi

test "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)"
```

## 2. Freeze the current annotation snapshot without GPUs

Use a new timestamped export root. Never overwrite an earlier snapshot.

```bash
export RESULT_COMMIT="4c9cd203e7b0d5fc9cc0e6f0e271cda511238f12"
export READINESS_20K_ROOT="$GEODML_RUNS_ROOT/semantic-readiness-20k-abstention/$RESULT_COMMIT-combined-atleast20000-four-judge-v2"
export READINESS_20K_QUEUE_ROOT="$READINESS_20K_ROOT/judge-queue"

export INPUT_COMMIT="40527b0fbc1fddb9f3a377f70e1405effd4ebc19"
export INPUT_ROOT="$GEODML_RUNS_ROOT/semantic-readiness-incremental/$INPUT_COMMIT-six-source-atleast10000-four-judge"
export READINESS_CORPUS="$INPUT_ROOT/incremental-corpus/semantic_readiness_expanded_corpus.jsonl"
export READINESS_TASKS="$READINESS_20K_ROOT/task-bank-four-judge-v2/readiness_label_tasks_blinded.jsonl"
export READINESS_CODEBOOK="$READINESS_20K_ROOT/task-bank-four-judge-v2/readiness_label_codebook_private.jsonl"

export EXPORT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export READINESS_HF_EXPORT_ROOT="$GEODML_RUNS_ROOT/semantic-readiness-hf-export/$GEODML_EXPECTED_COMMIT-$EXPORT_ID"
export READINESS_HF_BUNDLE_ROOT="$READINESS_HF_EXPORT_ROOT/bundle"
export READINESS_HF_EMBEDDING_ROOT="$READINESS_HF_EXPORT_ROOT/embeddings"
export READINESS_HF_DATASET_ROOT="$READINESS_HF_EXPORT_ROOT/huggingface-dataset"

mkdir -p "$READINESS_HF_EXPORT_ROOT"

printf '%s\n' "$READINESS_HF_EXPORT_ROOT" \
  > "$HOME/geodml-readiness-hf-export-latest.txt"

{
  printf 'export GEODML_EXPECTED_COMMIT=%q\n' "$GEODML_EXPECTED_COMMIT"
  printf 'export GEODML_REPOSITORY=%q\n' "$GEODML_REPOSITORY"
  printf 'export READINESS_20K_ROOT=%q\n' "$READINESS_20K_ROOT"
  printf 'export READINESS_20K_QUEUE_ROOT=%q\n' "$READINESS_20K_QUEUE_ROOT"
  printf 'export READINESS_HF_EXPORT_ROOT=%q\n' "$READINESS_HF_EXPORT_ROOT"
  printf 'export READINESS_HF_BUNDLE_ROOT=%q\n' "$READINESS_HF_BUNDLE_ROOT"
  printf 'export READINESS_HF_EMBEDDING_ROOT=%q\n' "$READINESS_HF_EMBEDDING_ROOT"
  printf 'export READINESS_HF_DATASET_ROOT=%q\n' "$READINESS_HF_DATASET_ROOT"
} > "$READINESS_HF_EXPORT_ROOT/export-environment.sh"

cd "$GEODML_REPOSITORY"

python3 analysis/scripts/build_readiness_hf_dataset.py assemble \
  --corpus "$READINESS_CORPUS" \
  --tasks "$READINESS_TASKS" \
  --codebook "$READINESS_CODEBOOK" \
  --queue-root "$READINESS_20K_QUEUE_ROOT" \
  --output-dir "$READINESS_HF_BUNDLE_ROOT" \
  --git-commit-sha "$GEODML_EXPECTED_COMMIT"

cat "$READINESS_HF_BUNDLE_ROOT/assembly_manifest.json"
```

At this point no model has been loaded and nothing has been uploaded.

## 3. Freeze exact model snapshots without GPUs

The three views are:

1. Qwen3-8B + MNTP + unsupervised SimCSE;
2. Qwen3-8B + MNTP + supervised contrastive adapter;
3. LLM2Vec-Gen Qwen3-8B expected-response representation.

The Qwen checkpoint is already present from the judge panel. Resolve the
current adapter/model repository heads once, record the full SHAs, and download
those exact revisions. Do not use `main` during inference.

```bash
export QWEN3_8B_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export QWEN3_8B_SNAPSHOT="$GEODML_MODELS_ROOT/qwen/Qwen3-8B/$QWEN3_8B_REVISION"
test -s "$QWEN3_8B_SNAPSHOT/config.json"

resolve_hf_head() {
  git ls-remote "https://huggingface.co/$1" HEAD | awk '{print $1}'
}

export LLM2VEC_MNTP_REPO="McGill-NLP/LLM2Vec-Qwen3-8B-mntp"
export LLM2VEC_UNSUP_SIMCSE_REPO="McGill-NLP/LLM2Vec-Qwen3-8B-mntp-unsup-simcse"
export LLM2VEC_SUPERVISED_REPO="McGill-NLP/LLM2Vec-Qwen3-8B-mntp-supervised"
export LLM2VEC_GEN_REPO="McGill-NLP/LLM2Vec-Gen-Qwen3-8B"

export LLM2VEC_MNTP_REVISION="$(resolve_hf_head "$LLM2VEC_MNTP_REPO")"
export LLM2VEC_UNSUP_SIMCSE_REVISION="$(resolve_hf_head "$LLM2VEC_UNSUP_SIMCSE_REPO")"
export LLM2VEC_SUPERVISED_REVISION="$(resolve_hf_head "$LLM2VEC_SUPERVISED_REPO")"
export LLM2VEC_GEN_REVISION="$(resolve_hf_head "$LLM2VEC_GEN_REPO")"

for revision in \
  "$LLM2VEC_MNTP_REVISION" \
  "$LLM2VEC_UNSUP_SIMCSE_REVISION" \
  "$LLM2VEC_SUPERVISED_REVISION" \
  "$LLM2VEC_GEN_REVISION"; do
  test "${#revision}" -eq 40
done

export LLM2VEC_MNTP_SNAPSHOT="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/$LLM2VEC_MNTP_REVISION"
export LLM2VEC_UNSUP_SIMCSE_SNAPSHOT="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/$LLM2VEC_UNSUP_SIMCSE_REVISION"
export LLM2VEC_SUPERVISED_SNAPSHOT="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-supervised/$LLM2VEC_SUPERVISED_REVISION"
export LLM2VEC_GEN_SNAPSHOT="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Gen-Qwen3-8B/$LLM2VEC_GEN_REVISION"

hf download "$LLM2VEC_MNTP_REPO" \
  --revision "$LLM2VEC_MNTP_REVISION" \
  --local-dir "$LLM2VEC_MNTP_SNAPSHOT"

hf download "$LLM2VEC_UNSUP_SIMCSE_REPO" \
  --revision "$LLM2VEC_UNSUP_SIMCSE_REVISION" \
  --local-dir "$LLM2VEC_UNSUP_SIMCSE_SNAPSHOT"

hf download "$LLM2VEC_SUPERVISED_REPO" \
  --revision "$LLM2VEC_SUPERVISED_REVISION" \
  --local-dir "$LLM2VEC_SUPERVISED_SNAPSHOT"

hf download "$LLM2VEC_GEN_REPO" \
  --revision "$LLM2VEC_GEN_REVISION" \
  --local-dir "$LLM2VEC_GEN_SNAPSHOT"
```

Record these values beside the export so a reconnect never re-resolves a newer
head:

```bash
for variable in \
  QWEN3_8B_REVISION QWEN3_8B_SNAPSHOT \
  LLM2VEC_MNTP_REPO LLM2VEC_MNTP_REVISION LLM2VEC_MNTP_SNAPSHOT \
  LLM2VEC_UNSUP_SIMCSE_REPO LLM2VEC_UNSUP_SIMCSE_REVISION LLM2VEC_UNSUP_SIMCSE_SNAPSHOT \
  LLM2VEC_SUPERVISED_REPO LLM2VEC_SUPERVISED_REVISION LLM2VEC_SUPERVISED_SNAPSHOT \
  LLM2VEC_GEN_REPO LLM2VEC_GEN_REVISION LLM2VEC_GEN_SNAPSHOT; do
  printf 'export %s=%q\n' "$variable" "${!variable}"
done >> "$READINESS_HF_EXPORT_ROOT/export-environment.sh"
```

## 4. Prepare the isolated embedding environment

```bash
module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

export GEODML_LLM2VEC_EXPORT_VENV="$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec"

if [[ ! -x "$GEODML_LLM2VEC_EXPORT_VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$GEODML_LLM2VEC_EXPORT_VENV"
fi

source "$GEODML_LLM2VEC_EXPORT_VENV/bin/activate"

bash "$GEODML_REPOSITORY/analysis/scripts/install_llm2vec_runtime.sh"
bash "$GEODML_REPOSITORY/analysis/scripts/install_llm2vec_gen_runtime.sh"

export GEODML_MODEL_VENV="$GEODML_LLM2VEC_EXPORT_VENV"

{
  printf 'export GEODML_LLM2VEC_EXPORT_VENV=%q\n' "$GEODML_LLM2VEC_EXPORT_VENV"
  printf 'export GEODML_MODEL_VENV=%q\n' "$GEODML_MODEL_VENV"
} >> "$READINESS_HF_EXPORT_ROOT/export-environment.sh"
```

## 5. GPU embedding run requires a separately approved allocation

Do not allocate yet. Before any `salloc`, allocating `srun`, or `sbatch`, report
the requested walltime, GPU count, maximum GPU-hours, and a measured or bounded
runtime estimate, then obtain explicit approval.

Inside an approved compute shell with exactly one GPU visible, restore the
saved model environment and run:

```bash
export READINESS_HF_EXPORT_ROOT="$(cat "$HOME/geodml-readiness-hf-export-latest.txt")"
source "$READINESS_HF_EXPORT_ROOT/export-environment.sh"
export CUDA_VISIBLE_DEVICES=0

bash "$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_hf_embedding_views.sh"
```

Each completed 512-prompt shard is written atomically. Re-running the command
checks and skips completed shards without loading that model. To benchmark only
one view first, set for example:

```bash
export READINESS_HF_EMBEDDING_VIEWS="qwen3-8b-mntp-unsup-simcse"
```

## 6. Finalize the HF-safe Parquet repository without GPUs

Include only embedding directories whose manifests say `is_complete=true`.
This example requires all three views:

```bash
cd "$GEODML_REPOSITORY"
source "$GEODML_LLM2VEC_EXPORT_VENV/bin/activate"

python analysis/scripts/build_readiness_hf_dataset.py finalize \
  --bundle-root "$READINESS_HF_BUNDLE_ROOT" \
  --embedding-dir "$READINESS_HF_EMBEDDING_ROOT/qwen3-8b-mntp-unsup-simcse" \
  --embedding-dir "$READINESS_HF_EMBEDDING_ROOT/qwen3-8b-mntp-supervised" \
  --embedding-dir "$READINESS_HF_EMBEDDING_ROOT/qwen3-8b-llm2vec-gen" \
  --output-dir "$READINESS_HF_DATASET_ROOT" \
  --git-commit-sha "$GEODML_EXPECTED_COMMIT"

python analysis/scripts/build_readiness_hf_dataset.py verify \
  --dataset-dir "$READINESS_HF_DATASET_ROOT"

find "$READINESS_HF_DATASET_ROOT" -type f -maxdepth 4 -print
cat "$READINESS_HF_DATASET_ROOT/dataset_manifest.json"
```

## 7. Explicit private Hugging Face upload

Uploading is deliberately separate from assembly and finalization. Choose the
repository ID explicitly and repeat it as confirmation. `HF_TOKEN` must be in
the environment, but never print it.

```bash
export HF_REPO_ID="ValerianFourel/geodml-semantic-readiness-20k"

python analysis/scripts/build_readiness_hf_dataset.py publish \
  --dataset-dir "$READINESS_HF_DATASET_ROOT" \
  --repo-id "$HF_REPO_ID" \
  --confirm-repo-id "$HF_REPO_ID"
```

The default is a private dataset repository. Public publication additionally
requires both `--public` and `--confirm-public`, after source attribution and
dataset-card review.

Load individual configurations later with:

```python
from datasets import load_dataset

prompts = load_dataset(HF_REPO_ID, "prompts")
annotations = load_dataset(HF_REPO_ID, "annotations")
unsupervised = load_dataset(HF_REPO_ID, "embeddings-qwen3-8b-mntp-unsup-simcse")
```
