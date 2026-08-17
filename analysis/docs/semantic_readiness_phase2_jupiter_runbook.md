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

## 1. Prepare three immutable snapshots on a login node

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

## 2. Freeze the panel environment

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

## 3. Submit all three independent judges

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

## 4. Verify exact completion

After all jobs finish:

```bash
for slot in primary-frontier replicate-frontier-a replicate-frontier-b; do
  wc -l "$READINESS_RUN/judges/$slot/judge_responses.jsonl"
  python -m json.tool "$READINESS_RUN/judges/$slot/run_manifest.json" | sed -n '1,180p'
done
```

Expected response count per slot is 5,091 with zero undeclared missing tasks.

## 5. Compile consensus and reliability diagnostics

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
