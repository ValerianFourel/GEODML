# Incremental four-judge semantic-readiness expansion

This run adds at least 10,000 genuinely new prompts beyond the frozen
10,460-item expanded corpus. Every added prompt is annotated with the unchanged
`decision-readiness-ordinal-v1` Likert rubric by the four canonical slots:

- Gemma 4 31B as `primary-frontier`;
- Qwen 3 32B as `replicate-frontier-a`;
- Ministral 3 8B as `replicate-frontier-b`;
- Llama 3.3 70B as `replicate-frontier-c`.

Qwen 3 8B is not part of this queue. It remains a sensitivity judge and must
not replace or be combined with Gemma as a second `primary-frontier` label.

The candidate panel uses the same six source snapshots, bottom-hash seed,
source-heldout splits, text eligibility rules, and frozen prompt rubric as the
completed expansion: OpenAssistant, CCPE-M, Taskmaster-1, Schema-Guided
Dialogue, Amazon Shopping Queries, and WildChat. The per-source cap increases
from 1,000 to 3,200. Nested bottom-hash sampling retains every old selection;
deduplication against the current expanded corpus leaves only new texts. The
run refuses to submit unless at least 10,000 exact-new prompts remain.
Restricted WildChat text and all derived artifacts must remain in project
storage.

Replace `INCREMENTAL_COMMIT` with the exact commit containing this runbook and
queue script.

## 1. Enter the exact committed checkout

```bash
source "$HOME/geodml_setup.sh"

export GEODML_EXPECTED_COMMIT="INCREMENTAL_COMMIT"

git fetch origin codex/semantic-readiness-phase2-jupiter
git checkout --detach "$GEODML_EXPECTED_COMMIT"

test "$(git rev-parse HEAD)" = "$GEODML_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

## 2. Recover the frozen input identities

```bash
export OLD_EXPANSION_COMMIT="d5363028b509a3dc9686f7e1755a9f1c83e985b3"
export OLD_EXPANSION_ROOT="$GEODML_RUNS_ROOT/semantic-readiness-expansion/$OLD_EXPANSION_COMMIT-wildchat1000-llama33-70b"
export OLD_EXPANDED_CORPUS="$OLD_EXPANSION_ROOT/merged-corpus/semantic_readiness_expanded_corpus.jsonl"
export OLD_OPEN_RECORDS="$OLD_EXPANSION_ROOT/open-transfer-records/semantic_readiness_transfer_records.jsonl"
export OLD_WILDCHAT_RECORDS="$OLD_EXPANSION_ROOT/wildchat-transfer-records/semantic_readiness_transfer_records.jsonl"
export OLD_LLAMA_MANIFEST="$OLD_EXPANSION_ROOT/judge-queue/full/llama3.3-70b-replicate-c/run_manifest.json"
export TRANSFER_ROOT="$OLD_EXPANSION_ROOT/input/semantic_readiness_transfer_sources_v1"

test -s "$OLD_EXPANDED_CORPUS"
test -s "$OLD_OPEN_RECORDS"
test -s "$OLD_WILDCHAT_RECORDS"
test -s "$OLD_LLAMA_MANIFEST"
test "$(wc -l < "$OLD_OPEN_RECORDS")" -eq 4370
test "$(wc -l < "$OLD_WILDCHAT_RECORDS")" -eq 1000
test -s "$TRANSFER_ROOT/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz"
test -s "$TRANSFER_ROOT/_upstream_git/ccpe/data.json"
test -d "$TRANSFER_ROOT/_upstream_git/taskmaster/TM-1-2019"
test -d "$TRANSFER_ROOT/google-schema-guided-dialogue"
test -s "$TRANSFER_ROOT/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"

export WILDCHAT_REVISION="$(
  python3 - "$OLD_WILDCHAT_RECORDS" <<'PY'
import json
import pathlib
import sys

revisions = {
    row["source_revision"]
    for line in pathlib.Path(sys.argv[1]).read_text().splitlines()
    if line.strip()
    for row in (json.loads(line),)
    if row.get("source_id") == "allenai-wildchat-1m"
}
if len(revisions) != 1:
    raise SystemExit(f"expected one frozen WildChat revision, found {sorted(revisions)}")
print(next(iter(revisions)))
PY
)"

export WILDCHAT_SNAPSHOT="$GEODML_RESTRICTED_DATA_ROOT/wildchat-1m/$WILDCHAT_REVISION"
export LLAMA_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$OLD_LLAMA_MANIFEST")"
export LLAMA_MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$OLD_LLAMA_MANIFEST")"

test "${#WILDCHAT_REVISION}" -eq 40
test "${#LLAMA_REVISION}" -eq 40
test -d "$WILDCHAT_SNAPSHOT"
test -s "$LLAMA_MODEL/config.json"
```

The transfer corpus stores `source_name` and not private source metadata in the
blinded task bank. The extraction above reads only the private corpus artifact
and does not expose source identity to a judge.

## 3. Build an exact-new incremental task bank

```bash
module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

source "$GEODML_MODEL_VENV/bin/activate"

export CANDIDATE_MAXIMUM_PER_SOURCE=3200
export MINIMUM_NEW_PROMPTS=10000
export READINESS_MASTER_SEED=20260817
export INCREMENTAL_ROOT="$GEODML_RUNS_ROOT/semantic-readiness-incremental/$GEODML_EXPECTED_COMMIT-six-source-atleast10000-four-judge"
export CANDIDATE_RECORDS="$INCREMENTAL_ROOT/six-source-candidates"
export INCREMENTAL_CORPUS="$INCREMENTAL_ROOT/incremental-corpus"
export INCREMENTAL_TASK_BANK="$INCREMENTAL_ROOT/incremental-task-bank-four-judge"

test ! -e "$INCREMENTAL_ROOT"
mkdir -p "$INCREMENTAL_ROOT"

python analysis/scripts/build_semantic_readiness_dataset.py collect-transfer \
  --output-dir "$CANDIDATE_RECORDS" \
  --maximum-per-source "$CANDIDATE_MAXIMUM_PER_SOURCE" \
  --master-seed "$READINESS_MASTER_SEED" \
  --source-input "openassistant-oasst1=$TRANSFER_ROOT/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz" \
  --source-revision "openassistant-oasst1=fdf72ae0827c1cda404aff25b6603abec9e3399b" \
  --source-input "google-ccpe-m=$TRANSFER_ROOT/_upstream_git/ccpe/data.json" \
  --source-revision "google-ccpe-m=2c9cd30f33f3a154b5a27d015333679262ff36f5" \
  --source-input "google-taskmaster-1=$TRANSFER_ROOT/_upstream_git/taskmaster/TM-1-2019" \
  --source-revision "google-taskmaster-1=d92cb6af3005f1dc09c39e75e7daf4a04905e00b" \
  --source-input "google-schema-guided-dialogue=$TRANSFER_ROOT/google-schema-guided-dialogue" \
  --source-revision "google-schema-guided-dialogue=e852981ae34990f4358979625854259302feaa78" \
  --source-input "amazon-shopping-queries=$TRANSFER_ROOT/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet" \
  --source-revision "amazon-shopping-queries=7916cdf6ab75a462e77f20ab40428a10923998d5" \
  --source-input "allenai-wildchat-1m=$WILDCHAT_SNAPSHOT/data" \
  --source-revision "allenai-wildchat-1m=$WILDCHAT_REVISION"

python3 - \
  "$OLD_OPEN_RECORDS" \
  "$OLD_WILDCHAT_RECORDS" \
  "$CANDIDATE_RECORDS/semantic_readiness_transfer_records.jsonl" <<'PY'
import json
import pathlib
import sys

def hashes(path):
    return {
        json.loads(line)["text_sha256"]
        for line in pathlib.Path(path).read_text().splitlines()
        if line.strip()
    }

old = hashes(sys.argv[1]) | hashes(sys.argv[2])
candidate = hashes(sys.argv[3])
assert old <= candidate
print(f"NESTED SAMPLE: PASS ({len(old)} frozen text hashes retained)")
PY

python analysis/scripts/build_semantic_readiness_dataset.py merge-transfer \
  --base-corpus "$OLD_EXPANDED_CORPUS" \
  --transfer-records "$CANDIDATE_RECORDS/semantic_readiness_transfer_records.jsonl" \
  --output-dir "$INCREMENTAL_CORPUS"

export JUDGE_SLOTS='primary-frontier,replicate-frontier-a,replicate-frontier-b,replicate-frontier-c'

python analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
  --corpus "$INCREMENTAL_CORPUS/semantic_readiness_transfer_corpus.jsonl" \
  --judge-slots "$JUDGE_SLOTS" \
  --output-dir "$INCREMENTAL_TASK_BANK"

export READINESS_INCREMENTAL_TASKS="$INCREMENTAL_TASK_BANK/readiness_label_tasks_blinded.jsonl"
export READINESS_INCREMENTAL_TASKS_PER_SLOT="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_count"])' \
    "$INCREMENTAL_TASK_BANK/run_manifest.json"
)"
export READINESS_INCREMENTAL_TASKS_SHA256="$(sha256sum "$READINESS_INCREMENTAL_TASKS" | awk '{print $1}')"

test "$READINESS_INCREMENTAL_TASKS_PER_SLOT" -ge "$MINIMUM_NEW_PROMPTS"
echo "new unique prompts: $READINESS_INCREMENTAL_TASKS_PER_SLOT"
echo "canonical judgments requested: $((4 * READINESS_INCREMENTAL_TASKS_PER_SLOT))"
```

## 4. Verify novelty, nesting, and four-slot coverage

```bash
python3 - \
  "$OLD_EXPANDED_CORPUS" \
  "$INCREMENTAL_CORPUS/semantic_readiness_transfer_corpus.jsonl" \
  "$READINESS_INCREMENTAL_TASKS" \
  "$READINESS_INCREMENTAL_TASKS_PER_SLOT" <<'PY'
import collections
import json
import pathlib
import sys

def rows(path):
    return [
        json.loads(line)
        for line in pathlib.Path(path).read_text().splitlines()
        if line.strip()
    ]

old = rows(sys.argv[1])
new = rows(sys.argv[2])
tasks = rows(sys.argv[3])
expected = int(sys.argv[4])

old_hashes = {row["text_sha256"] for row in old}
new_hashes = {row["text_sha256"] for row in new}
assert len(new) == len(new_hashes) == expected
assert old_hashes.isdisjoint(new_hashes)

split_counts = collections.Counter(row["split"] for row in new)
source_counts = collections.Counter(row["source_name"] for row in new)
assert split_counts["development"] >= 3500, split_counts
assert split_counts["confirmation"] >= 5500, split_counts
assert len(source_counts) >= 5, source_counts

counts = collections.Counter(row["judge_slot"] for row in tasks)
required = {
    "primary-frontier",
    "replicate-frontier-a",
    "replicate-frontier-b",
    "replicate-frontier-c",
}
assert set(counts) == required, counts
assert set(counts.values()) == {expected}, counts
assert len({row["task_id"] for row in tasks}) == 4 * expected

print(f"NOVELTY: PASS ({expected} new prompts)")
print(f"FOUR-SLOT COVERAGE: PASS ({4 * expected} tasks)")
print(f"SPLIT COVERAGE: PASS ({dict(split_counts)})")
print(f"SOURCE COVERAGE: PASS ({dict(source_counts)})")
PY
```

Write the private launch environment without tokens:

```bash
export READINESS_INCREMENTAL_QUEUE_ROOT="$INCREMENTAL_ROOT/judge-queue"
export LLAMA_BATCH_SIZE=16
export SLURM_LOG_DIR="$INCREMENTAL_ROOT/slurm"
mkdir -p "$SLURM_LOG_DIR"

umask 077
{
  printf 'export GEODML_EXPECTED_COMMIT=%q\n' "$GEODML_EXPECTED_COMMIT"
  printf 'export GEODML_PROJECT_ROOT=%q\n' "$GEODML_PROJECT_ROOT"
  printf 'export GEODML_REPOSITORY=%q\n' "$GEODML_REPOSITORY"
  printf 'export GEODML_MODEL_VENV=%q\n' "$GEODML_MODEL_VENV"
  printf 'export READINESS_INCREMENTAL_TASKS=%q\n' "$READINESS_INCREMENTAL_TASKS"
  printf 'export READINESS_INCREMENTAL_TASKS_SHA256=%q\n' "$READINESS_INCREMENTAL_TASKS_SHA256"
  printf 'export READINESS_INCREMENTAL_TASKS_PER_SLOT=%q\n' "$READINESS_INCREMENTAL_TASKS_PER_SLOT"
  printf 'export READINESS_INCREMENTAL_QUEUE_ROOT=%q\n' "$READINESS_INCREMENTAL_QUEUE_ROOT"
  printf 'export LLAMA_MODEL=%q\n' "$LLAMA_MODEL"
  printf 'export LLAMA_REVISION=%q\n' "$LLAMA_REVISION"
  printf 'export LLAMA_BATCH_SIZE=%q\n' "$LLAMA_BATCH_SIZE"
} > "$INCREMENTAL_ROOT/launch-environment.txt"
```

## 5. Submit the resumable four-GPU queue

```bash
export INCREMENTAL_JOB_ID="$(
  sbatch \
    --parsable \
    --account="${JUPITER_ACCOUNT:-scifi}" \
    --output="$SLURM_LOG_DIR/incremental-four-judge-%j.out" \
    --error="$SLURM_LOG_DIR/incremental-four-judge-%j.err" \
    --export=ALL \
    analysis/scripts/slurm/jupiter/run_readiness_incremental_four_judge_queue.sbatch
)"

printf '%s\n' "$INCREMENTAL_JOB_ID" > "$INCREMENTAL_ROOT/slurm-job-id.txt"
echo "INCREMENTAL_JOB_ID=$INCREMENTAL_JOB_ID"
squeue -j "$INCREMENTAL_JOB_ID"
```

The job first runs eight-task smoke checks and then full annotation for each
canonical judge. If the 12-hour allocation ends, source
`launch-environment.txt` and submit the same queue script again; existing task
caches are resumed. Do not alter the task bank, model revisions, or queue root.

## 6. Completion criteria

```bash
source "$INCREMENTAL_ROOT/launch-environment.txt"
export INCREMENTAL_JOB_ID="$(cat "$INCREMENTAL_ROOT/slurm-job-id.txt")"

sacct -j "$INCREMENTAL_JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed
cat "$READINESS_INCREMENTAL_QUEUE_ROOT/queue.log"

find "$READINESS_INCREMENTAL_QUEUE_ROOT/full" \
  -name judge_responses.jsonl \
  -exec wc -l {} \;

grep -R -E 'Traceback|ERROR|OutOfMemory|exhausted|CANCELLED|TIMEOUT' \
  "$READINESS_INCREMENTAL_QUEUE_ROOT/logs" \
  "$SLURM_LOG_DIR" \
  2>/dev/null

sha256sum --check --quiet "$READINESS_INCREMENTAL_QUEUE_ROOT/artifact-sha256.txt"
```

Completion requires four full response files, each containing exactly
`READINESS_INCREMENTAL_TASKS_PER_SLOT` valid rows, with no failed or missing
tasks. These annotations are raw measurement artifacts, not scientific
conclusions.
