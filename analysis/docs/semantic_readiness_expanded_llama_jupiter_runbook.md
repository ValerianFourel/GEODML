# Expanded semantic-readiness panel: LMSYS prompts and Llama 3.3 70B

This run extends the frozen 5,091-item base corpus with the already frozen
five-source open transfer panel plus a deterministic 1,000-prompt sample from
`lmsys/lmsys-chat-1m`. The original rubric and the first three judge slots are
unchanged. `meta-llama/Llama-3.3-70B-Instruct` is added only as
`replicate-frontier-c`. Qwen3-8B also labels every new prompt as a sensitivity
judge, but it retains its historical `primary-frontier` identity and therefore
must not be combined with Gemma as though it were a fifth independent slot.

LMSYS-Chat-1M is gated. Accept its license through Hugging Face before running
these commands. Its extracted text and all downstream artifacts must remain
local to the project storage; do not upload them to GitHub or Hugging Face.
Meta Llama is also gated and requires acceptance of the Llama 3.3 license.

Replace `EXPANSION_COMMIT` below with the exact commit containing this runbook.

## 1. Fresh login-node session

```bash
set -euo pipefail
umask 077

jutil env activate -p scifi

export GEODML_EXPECTED_COMMIT="EXPANSION_COMMIT"
export GEODML_PROJECT_ROOT="$PROJECT/$USER/geodml"
export GEODML_REPOSITORY="$GEODML_PROJECT_ROOT/src/geodml-mono"
export GEODML_MODEL_VENV="$FSCRATCH/$USER/geodml/python/.venv-model-panel-transformers5141"

cd "$GEODML_REPOSITORY"
git fetch origin
git checkout --detach "$GEODML_EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

## 2. Resolve and download immutable gated snapshots

```bash
source "$GEODML_MODEL_VENV/bin/activate"

export LMSYS_REPO="lmsys/lmsys-chat-1m"
export LLAMA_REPO="meta-llama/Llama-3.3-70B-Instruct"

export LMSYS_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("lmsys/lmsys-chat-1m").sha)')"
export LLAMA_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("meta-llama/Llama-3.3-70B-Instruct").sha)')"

test "${#LMSYS_REVISION}" -eq 40
test "${#LLAMA_REVISION}" -eq 40

export LMSYS_SNAPSHOT="$GEODML_PROJECT_ROOT/restricted-data/lmsys-chat-1m/$LMSYS_REVISION"
export LLAMA_MODEL="$GEODML_PROJECT_ROOT/models/meta-llama/Llama-3.3-70B-Instruct/$LLAMA_REVISION"

mkdir -p "$LMSYS_SNAPSHOT" "$LLAMA_MODEL"

HF_XET_HIGH_PERFORMANCE=1 hf download \
  "$LMSYS_REPO" \
  --repo-type dataset \
  --revision "$LMSYS_REVISION" \
  --local-dir "$LMSYS_SNAPSHOT"

HF_XET_HIGH_PERFORMANCE=1 hf download \
  "$LLAMA_REPO" \
  --revision "$LLAMA_REVISION" \
  --local-dir "$LLAMA_MODEL"

test -s "$LLAMA_MODEL/config.json"
```

## 3. Build the expanded corpus and two task banks

```bash
export PHASE1_BUNDLE="$GEODML_PROJECT_ROOT/staging/geodml-semantic-readiness-phase1-open-20260817.tar.gz"
printf '%s  %s\n' \
  'f0422481b094da8f8e4db5e0d4ed7668fed603e7857db0b53069deac0387d344' \
  "$PHASE1_BUNDLE" \
  | sha256sum --check -

export EXPANSION_ROOT="$GEODML_PROJECT_ROOT/runs/semantic-readiness-expansion/$GEODML_EXPECTED_COMMIT-lmsys1000-llama33-70b"
export EXPANSION_INPUT="$EXPANSION_ROOT/input"
export LMSYS_RECORDS="$EXPANSION_ROOT/lmsys-transfer-records"
export MERGED_CORPUS="$EXPANSION_ROOT/merged-corpus"
export TRANSFER_TASK_BANK="$EXPANSION_ROOT/transfer-task-bank-four-judge"
export EXPANDED_TASK_BANK="$EXPANSION_ROOT/expanded-task-bank-four-judge"

mkdir -p "$EXPANSION_INPUT"
tar -xzf "$PHASE1_BUNDLE" \
  -C "$EXPANSION_INPUT" \
  semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl \
  semantic_readiness_transfer_records_open_v1/semantic_readiness_transfer_records.jsonl

export BASE_CORPUS="$EXPANSION_INPUT/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl"
export OPEN_TRANSFER_RECORDS="$EXPANSION_INPUT/semantic_readiness_transfer_records_open_v1/semantic_readiness_transfer_records.jsonl"

printf '%s  %s\n' \
  'c851c31f99bdfecd31238f36d6fe24d1e379a4ebf48c8dadc98ebfb2af1b26a8' \
  "$BASE_CORPUS" \
  | sha256sum --check -
printf '%s  %s\n' \
  'f5f98202dbec61e23589b07ac9627ba1edba616f4f69cdd43f31b8da7ffd163a' \
  "$OPEN_TRANSFER_RECORDS" \
  | sha256sum --check -

python analysis/scripts/build_semantic_readiness_dataset.py collect-transfer \
  --output-dir "$LMSYS_RECORDS" \
  --maximum-per-source 1000 \
  --master-seed 20260817 \
  --source-input "lmsys-chat-1m=$LMSYS_SNAPSHOT/data" \
  --source-revision "lmsys-chat-1m=$LMSYS_REVISION"

python analysis/scripts/build_semantic_readiness_dataset.py merge-transfer \
  --base-corpus "$BASE_CORPUS" \
  --transfer-records "$OPEN_TRANSFER_RECORDS" \
  --additional-transfer-records "$LMSYS_RECORDS/semantic_readiness_transfer_records.jsonl" \
  --output-dir "$MERGED_CORPUS"

export JUDGE_SLOTS='primary-frontier,replicate-frontier-a,replicate-frontier-b,replicate-frontier-c'

python analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
  --corpus "$MERGED_CORPUS/semantic_readiness_transfer_corpus.jsonl" \
  --judge-slots "$JUDGE_SLOTS" \
  --output-dir "$TRANSFER_TASK_BANK"

python analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
  --corpus "$MERGED_CORPUS/semantic_readiness_expanded_corpus.jsonl" \
  --judge-slots "$JUDGE_SLOTS" \
  --output-dir "$EXPANDED_TASK_BANK"
```

Derive counts and hashes from the frozen outputs rather than assuming that no
exact-text duplicate was removed:

```bash
export READINESS_TRANSFER_TASKS="$TRANSFER_TASK_BANK/readiness_label_tasks_blinded.jsonl"
export READINESS_EXPANDED_TASKS="$EXPANDED_TASK_BANK/readiness_label_tasks_blinded.jsonl"

export READINESS_TRANSFER_TASKS_PER_SLOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_count"])' "$TRANSFER_TASK_BANK/run_manifest.json")"
export READINESS_EXPANDED_TASKS_PER_SLOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_count"])' "$EXPANDED_TASK_BANK/run_manifest.json")"

export READINESS_TRANSFER_TASKS_SHA256="$(sha256sum "$READINESS_TRANSFER_TASKS" | awk '{print $1}')"
export READINESS_EXPANDED_TASKS_SHA256="$(sha256sum "$READINESS_EXPANDED_TASKS" | awk '{print $1}')"

echo "transfer prompts: $READINESS_TRANSFER_TASKS_PER_SLOT"
echo "expanded prompts: $READINESS_EXPANDED_TASKS_PER_SLOT"
echo "new judgments requested: $((4 * READINESS_TRANSFER_TASKS_PER_SLOT + READINESS_EXPANDED_TASKS_PER_SLOT))"
```

Verify that the fourth slot and added rows did not alter any old task ID,
prompt, or anchor presentation:

```bash
export FROZEN_TASKS="$GEODML_PROJECT_ROOT/runs/semantic-readiness-base-axis/f6d9e6df42c90b425e4035bb9f28cb551be63175/label-tasks/readiness_label_tasks_blinded.jsonl"

python - "$FROZEN_TASKS" "$READINESS_EXPANDED_TASKS" <<'PY'
import json
import pathlib
import sys

def load(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]

old = load(sys.argv[1])
new = {(row["item_id"], row["judge_slot"]): row for row in load(sys.argv[2])}
for row in old:
    candidate = new[(row["item_id"], row["judge_slot"])]
    for key in ("task_id", "prompt", "presentation_variant", "rubric_version"):
        assert candidate[key] == row[key], (row["task_id"], key)
print(f"FROZEN TASK COMPATIBILITY: PASS ({len(old)} tasks)")
PY
```

## 4. Submit the unattended four-GPU queue

```bash
export READINESS_EXPANDED_QUEUE_ROOT="$EXPANSION_ROOT/judge-queue"
export LLAMA_BATCH_SIZE=16
export SLURM_LOG_DIR="$EXPANSION_ROOT/slurm"
mkdir -p "$SLURM_LOG_DIR"

export EXPANSION_JOB_ID="$(
  sbatch \
    --parsable \
    --output="$SLURM_LOG_DIR/expanded-panel-%j.out" \
    --error="$SLURM_LOG_DIR/expanded-panel-%j.err" \
    --export=ALL \
    analysis/scripts/slurm/jupiter/run_readiness_expanded_llama_queue.sbatch
)"

echo "EXPANSION_JOB_ID=$EXPANSION_JOB_ID"
printf '%s\n' "$EXPANSION_JOB_ID" > "$EXPANSION_ROOT/slurm-job-id.txt"
squeue -j "$EXPANSION_JOB_ID"
```

The queue is resumable. If the 12-hour job ends, check out the same commit,
restore the same environment variables, and submit the same script again with
the same `READINESS_EXPANDED_QUEUE_ROOT`.

## 5. Inspect results from a fresh login

```bash
sacct -j "$EXPANSION_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
cat "$READINESS_EXPANDED_QUEUE_ROOT/queue.log"

find "$READINESS_EXPANDED_QUEUE_ROOT/full" \
  -name judge_responses.jsonl \
  -exec wc -l {} \;

grep -R -E 'Traceback|ERROR|OutOfMemory|exhausted|CANCELLED|TIMEOUT' \
  "$READINESS_EXPANDED_QUEUE_ROOT/logs" \
  "$SLURM_LOG_DIR" \
  2>/dev/null
```

Expected complete full outputs are four transfer-only response files, each
with `READINESS_TRANSFER_TASKS_PER_SLOT` rows, and one Llama response file with
`READINESS_EXPANDED_TASKS_PER_SLOT` rows. These are raw annotation artifacts,
not scientific conclusions.
