# Semantic readiness Phase 2 on JUPITER

This runbook executes only the frozen readiness-annotation phase. It does not
fit LLM2Vec geometry and does not inspect GEO outcomes.

## Scientific contract

- Use the frozen 5,091-item base corpus and its 15,273 blinded tasks.
- Use three pinned judge snapshots from distinct model families.
- Do not choose or remove judges after comparing them with LLM2Vec.
- Keep every raw response, rejected parse attempt, confidence, ambiguity, and
  ordinal field.
- Treat the earlier Qwen2.5-7B five-task run as plumbing evidence only.
- Do not upload private codebooks or later restricted transfer text.

Possible model families include Qwen, Mistral, and Gemma/Llama, but model
selection and access terms must be reviewed and frozen before downloading.
The launcher deliberately has no default production models.

The four downstream answer models (Qwen3-8B, Qwen3-32B, Ministral3-8B, and
Gemma4-31B) are not the annotation ensemble. Keep those snapshots for the
later behavioral panel. Strong hosted models such as an approved Qwen-Max
version may be used here, but only after the exact provider model identifiers,
versions, access terms, and three distinct model families are frozen. Do not
interpret a marketing name such as `Max` or `Muse` as an immutable revision.

## 1. Recommended route: export three hosted-provider batches

This route supports strong hosted judges without putting API keys in the
repository and without making an accidental paid request. The repository
produces OpenAI-compatible batch JSONL. Submission and result download happen
through the approved provider account separately.

Start from the exact production commit and the frozen task bank:

```bash
export GEODML_EXPECTED_COMMIT="<phase2-production-commit-sha>"
export GEODML_PROJECT_ROOT="$PROJECT/$USER/geodml"
export READINESS_RUN="$GEODML_PROJECT_ROOT/runs/semantic-readiness-phase2/$GEODML_EXPECTED_COMMIT"
export READINESS_JUDGE_TASKS="$GEODML_PROJECT_ROOT/runs/semantic-readiness-base-axis/f6d9e6df42c90b425e4035bb9f28cb551be63175/label-tasks/readiness_label_tasks_blinded.jsonl"
export READINESS_JUDGE_TASKS_SHA256="9c1e084332d4fc3129a1f1c5400b8118d7a3425a01f3c771edb133d66d496775"
export READINESS_EXPECTED_TASKS_PER_SLOT="5091"
export READINESS_CODEBOOK="$GEODML_PROJECT_ROOT/runs/semantic-readiness-base-axis/f6d9e6df42c90b425e4035bb9f28cb551be63175/label-tasks/readiness_label_codebook_private.jsonl"

umask 077
cd "$GEODML_PROJECT_ROOT/src/geodml-mono"
test "$(git rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum "$READINESS_JUDGE_TASKS" | awk '{print $1}')" = \
  "$READINESS_JUDGE_TASKS_SHA256"

mkdir -p "$READINESS_RUN/batches" "$READINESS_RUN/judges"
```

Freeze one provider, exact model identifier, independent family label, and
provider version for each judge. These are intentionally placeholders; model
selection must not be inferred from the downstream four-model panel:

```bash
export PRIMARY_JUDGE_PROVIDER="<approved-provider>"
export PRIMARY_JUDGE_MODEL="<exact-provider-model-id>"
export PRIMARY_JUDGE_FAMILY="<independent-family-1>"
export PRIMARY_JUDGE_REVISION="<immutable-provider-version>"
export PRIMARY_PROVIDER_RESPONSE_MODEL="<exact-model-string-returned-by-provider>"

export REPLICATE_A_JUDGE_PROVIDER="<approved-provider>"
export REPLICATE_A_JUDGE_MODEL="<exact-provider-model-id>"
export REPLICATE_A_JUDGE_FAMILY="<independent-family-2>"
export REPLICATE_A_JUDGE_REVISION="<immutable-provider-version>"
export REPLICATE_A_PROVIDER_RESPONSE_MODEL="<exact-model-string-returned-by-provider>"

export REPLICATE_B_JUDGE_PROVIDER="<approved-provider>"
export REPLICATE_B_JUDGE_MODEL="<exact-provider-model-id>"
export REPLICATE_B_JUDGE_FAMILY="<independent-family-3>"
export REPLICATE_B_JUDGE_REVISION="<immutable-provider-version>"
export REPLICATE_B_PROVIDER_RESPONSE_MODEL="<exact-model-string-returned-by-provider>"

test "$PRIMARY_JUDGE_FAMILY" != "$REPLICATE_A_JUDGE_FAMILY"
test "$PRIMARY_JUDGE_FAMILY" != "$REPLICATE_B_JUDGE_FAMILY"
test "$REPLICATE_A_JUDGE_FAMILY" != "$REPLICATE_B_JUDGE_FAMILY"
```

Export the initial 5,091 requests for each independent slot:

```bash
python analysis/scripts/run_semantic_readiness_judge_batch.py export \
  --tasks "$READINESS_JUDGE_TASKS" \
  --tasks-sha256 "$READINESS_JUDGE_TASKS_SHA256" \
  --expected-tasks "$READINESS_EXPECTED_TASKS_PER_SLOT" \
  --judge-slot primary-frontier \
  --provider "$PRIMARY_JUDGE_PROVIDER" \
  --model "$PRIMARY_JUDGE_MODEL" \
  --model-family "$PRIMARY_JUDGE_FAMILY" \
  --model-revision "$PRIMARY_JUDGE_REVISION" \
  --expected-provider-model "$PRIMARY_PROVIDER_RESPONSE_MODEL" \
  --output-dir "$READINESS_RUN/batches/primary-frontier-attempt-001" \
  --judge-output-dir "$READINESS_RUN/judges/primary-frontier"

python analysis/scripts/run_semantic_readiness_judge_batch.py export \
  --tasks "$READINESS_JUDGE_TASKS" \
  --tasks-sha256 "$READINESS_JUDGE_TASKS_SHA256" \
  --expected-tasks "$READINESS_EXPECTED_TASKS_PER_SLOT" \
  --judge-slot replicate-frontier-a \
  --provider "$REPLICATE_A_JUDGE_PROVIDER" \
  --model "$REPLICATE_A_JUDGE_MODEL" \
  --model-family "$REPLICATE_A_JUDGE_FAMILY" \
  --model-revision "$REPLICATE_A_JUDGE_REVISION" \
  --expected-provider-model "$REPLICATE_A_PROVIDER_RESPONSE_MODEL" \
  --output-dir "$READINESS_RUN/batches/replicate-frontier-a-attempt-001" \
  --judge-output-dir "$READINESS_RUN/judges/replicate-frontier-a"

python analysis/scripts/run_semantic_readiness_judge_batch.py export \
  --tasks "$READINESS_JUDGE_TASKS" \
  --tasks-sha256 "$READINESS_JUDGE_TASKS_SHA256" \
  --expected-tasks "$READINESS_EXPECTED_TASKS_PER_SLOT" \
  --judge-slot replicate-frontier-b \
  --provider "$REPLICATE_B_JUDGE_PROVIDER" \
  --model "$REPLICATE_B_JUDGE_MODEL" \
  --model-family "$REPLICATE_B_JUDGE_FAMILY" \
  --model-revision "$REPLICATE_B_JUDGE_REVISION" \
  --expected-provider-model "$REPLICATE_B_PROVIDER_RESPONSE_MODEL" \
  --output-dir "$READINESS_RUN/batches/replicate-frontier-b-attempt-001" \
  --judge-output-dir "$READINESS_RUN/judges/replicate-frontier-b"

wc -l "$READINESS_RUN"/batches/*-attempt-001/batch_requests.jsonl
sha256sum "$READINESS_RUN"/batches/*-attempt-001/* > \
  "$READINESS_RUN/batches/attempt-001-sha256.txt"
```

Each count must be exactly 5,091 before submission. Submit each
`batch_requests.jsonl` through its named provider's approved Batch API and
record the returned provider batch ID. Do not put an API key in the request
JSONL, shell history, request-options file, or run directory.

After downloading the provider result JSONL, import it with the corresponding
export manifest and provider batch ID. For example, the primary slot is:

```bash
export PRIMARY_BATCH_RESULT="<downloaded-provider-result.jsonl>"
export PRIMARY_PROVIDER_BATCH_ID="<provider-batch-id>"

python analysis/scripts/run_semantic_readiness_judge_batch.py import \
  --tasks "$READINESS_JUDGE_TASKS" \
  --export-manifest \
    "$READINESS_RUN/batches/primary-frontier-attempt-001/batch_manifest.json" \
  --batch-output "$PRIMARY_BATCH_RESULT" \
  --provider-batch-id "$PRIMARY_PROVIDER_BATCH_ID" \
  --output-dir "$READINESS_RUN/judges/primary-frontier"
```

Repeat the import for replicate A and B using their own manifest, result, batch
ID, and judge output directory. The importer preserves the complete provider
rows and usage, validates the strict continuous/Likert schema, and creates
task-level success or failure caches.

If a result is malformed or rejected, do not edit it. Export the next attempt
to a new directory while retaining the same judge output directory:

```bash
python analysis/scripts/run_semantic_readiness_judge_batch.py export \
  --tasks "$READINESS_JUDGE_TASKS" \
  --tasks-sha256 "$READINESS_JUDGE_TASKS_SHA256" \
  --expected-tasks "$READINESS_EXPECTED_TASKS_PER_SLOT" \
  --judge-slot primary-frontier \
  --provider "$PRIMARY_JUDGE_PROVIDER" \
  --model "$PRIMARY_JUDGE_MODEL" \
  --model-family "$PRIMARY_JUDGE_FAMILY" \
  --model-revision "$PRIMARY_JUDGE_REVISION" \
  --expected-provider-model "$PRIMARY_PROVIDER_RESPONSE_MODEL" \
  --output-dir "$READINESS_RUN/batches/primary-frontier-attempt-002" \
  --judge-output-dir "$READINESS_RUN/judges/primary-frontier"
```

Only uncached tasks appear in the retry file. Its prompt contains the frozen
rubric plus the prior invalid response and parse error. Import the retry result
in the same way with its new provider batch ID. The importer refuses an altered
task bank, submitted request file, judge identity, provider result model, or
provider batch provenance.

## 2. Alternative route: prepare three local immutable snapshots

Load the same module stack and AI virtual environment used by the smoke test.
For each approved repository, resolve and record the immutable revision before
downloading:

```bash
export MODEL_REPO="<organization/model>"
export MODEL_REVISION="$(python -c 'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["MODEL_REPO"]).sha)')"
export MODEL_DIR="$GEODML_PROJECT_ROOT/models/<family>/$MODEL_REVISION"

mkdir -p "$MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download \
  "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

test -s "$MODEL_DIR/config.json"
echo "$MODEL_REPO $MODEL_REVISION $MODEL_DIR"
```

Repeat only after changing all three variables. Gated repositories require the
account holder to accept their terms first.

## 3. Freeze the local panel environment

Copy and complete:

```bash
cp analysis/scripts/slurm/jupiter/semantic_readiness_panel.env.example \
  "$GEODML_PROJECT_ROOT/manifests/semantic-readiness-panel.env"

chmod 600 "$GEODML_PROJECT_ROOT/manifests/semantic-readiness-panel.env"
source "$GEODML_PROJECT_ROOT/manifests/semantic-readiness-panel.env"
```

The three family labels and model directories must be different, and each
revision must be immutable. The submitter rejects duplicate family labels;
use labels such as `qwen`, `mistral`, and `gemma`, not three aliases for one
architecture. Preserve the completed private environment file with the run.
The template also pins the observed canonical task-bank SHA-256 and requires
exactly 5,091 tasks in each slot.

## 4. Submit all three independent local judges

Submit from a clean repository checked out at `GEODML_EXPECTED_COMMIT`:

```bash
cd "$GEODML_PROJECT_ROOT/src/geodml-mono"
test "$(git rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

analysis/scripts/slurm/jupiter/submit_semantic_readiness_panel.sh
```

The submitter validates the task bank, Git state, three local model snapshots,
distinct family labels, distinct model paths, and the exact task-bank hash
before creating jobs. It refuses to replace an existing panel declaration with
a different panel. Each job is resumable and writes to:

```text
$READINESS_RUN/judges/<judge-slot>/
```

Monitor without changing the frozen panel:

```bash
squeue -u "$USER"
tail -f "$READINESS_RUN"/logs/*.out
```

Re-running the submitter resumes cached tasks. Do not delete a partial judge
directory merely because a job reaches its time limit.

## 5. Verify exact completion

After all jobs finish:

```bash
for slot in primary-frontier replicate-frontier-a replicate-frontier-b; do
  wc -l "$READINESS_RUN/judges/$slot/judge_responses.jsonl"
  python -m json.tool "$READINESS_RUN/judges/$slot/run_manifest.json" | sed -n '1,180p'
done
```

Expected response count per slot is 5,091 with zero undeclared missing tasks.

## 6. Compile consensus and reliability diagnostics

```bash
export COMPILED_LABELS="$READINESS_RUN/compiled-labels"
test ! -e "$COMPILED_LABELS"

python analysis/scripts/fit_semantic_readiness_map.py compile-labels \
  --tasks "$READINESS_JUDGE_TASKS" \
  --codebooks "$READINESS_CODEBOOK" \
  --responses \
    "$READINESS_RUN/judges/primary-frontier/judge_responses.jsonl" \
    "$READINESS_RUN/judges/replicate-frontier-a/judge_responses.jsonl" \
    "$READINESS_RUN/judges/replicate-frontier-b/judge_responses.jsonl" \
  --output-dir "$COMPILED_LABELS"

python -m json.tool "$COMPILED_LABELS/label_diagnostics.json"
python -m json.tool "$COMPILED_LABELS/judge_agreement.json"
sha256sum "$COMPILED_LABELS"/* > "$COMPILED_LABELS/artifact-sha256.txt"
```

Stop after reviewing agreement, missingness, ambiguity, confidence, and the
number of usable items. Judge disagreement is evidence; never discard the
judge that is least aligned with the desired axis. Phase 3 begins only after
the Phase-2 aggregation rule and artifacts are frozen.
