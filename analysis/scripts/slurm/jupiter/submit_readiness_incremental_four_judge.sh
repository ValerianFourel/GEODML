#!/usr/bin/env bash
# Build an exact-new multi-source prompt bank and submit the four-judge queue.

set -euo pipefail
umask 077

jutil env activate -p "${JUPITER_PROJECT:-scifi}"

: "${PROJECT:?PROJECT is required}"
: "${FSCRATCH:?FSCRATCH is required}"
: "${USER:?USER is required}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"

GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono}"
GEODML_MODEL_VENV="${GEODML_MODEL_VENV:-$FSCRATCH/$USER/geodml/python/.venv-model-panel-transformers5141}"
JUPITER_ACCOUNT="${JUPITER_ACCOUNT:-scifi}"

old_expansion_commit="${READINESS_OLD_EXPANSION_COMMIT:-d5363028b509a3dc9686f7e1755a9f1c83e985b3}"
old_expansion_root="${READINESS_OLD_EXPANSION_ROOT:-$GEODML_PROJECT_ROOT/runs/semantic-readiness-expansion/$old_expansion_commit-wildchat1000-llama33-70b}"
old_expanded_corpus="$old_expansion_root/merged-corpus/semantic_readiness_expanded_corpus.jsonl"
old_open_records="$old_expansion_root/open-transfer-records/semantic_readiness_transfer_records.jsonl"
old_wildchat_records="$old_expansion_root/wildchat-transfer-records/semantic_readiness_transfer_records.jsonl"
old_llama_manifest="$old_expansion_root/judge-queue/full/llama3.3-70b-replicate-c/run_manifest.json"
transfer_root="$old_expansion_root/input/semantic_readiness_transfer_sources_v1"

candidate_maximum_per_source="${READINESS_CANDIDATE_MAXIMUM_PER_SOURCE:-3200}"
minimum_new_prompts="${READINESS_MINIMUM_NEW_PROMPTS:-10000}"
master_seed="${READINESS_MASTER_SEED:-20260817}"
incremental_root="${READINESS_INCREMENTAL_ROOT:-$GEODML_PROJECT_ROOT/runs/semantic-readiness-incremental/$GEODML_EXPECTED_COMMIT-six-source-atleast10000-four-judge}"
candidate_records="$incremental_root/six-source-candidates"
incremental_corpus="$incremental_root/incremental-corpus"
incremental_task_bank="$incremental_root/incremental-task-bank-four-judge"

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "incremental scientific run requires a clean checkout" >&2
    exit 2
fi

for path in \
    "$old_expanded_corpus" \
    "$old_open_records" \
    "$old_wildchat_records" \
    "$old_llama_manifest" \
    "$transfer_root/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz" \
    "$transfer_root/_upstream_git/ccpe/data.json" \
    "$transfer_root/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
do
    if [[ ! -s "$path" ]]; then
        echo "missing frozen input: $path" >&2
        exit 2
    fi
done
for path in \
    "$transfer_root/_upstream_git/taskmaster/TM-1-2019" \
    "$transfer_root/google-schema-guided-dialogue"
do
    if [[ ! -d "$path" ]]; then
        echo "missing frozen input directory: $path" >&2
        exit 2
    fi
done
if [[ "$(wc -l < "$old_open_records")" -ne 4370 ]]; then
    echo "frozen open transfer record count changed" >&2
    exit 2
fi
if [[ "$(wc -l < "$old_wildchat_records")" -ne 1000 ]]; then
    echo "frozen WildChat transfer record count changed" >&2
    exit 2
fi
if [[ -e "$incremental_root" ]]; then
    echo "refusing to overwrite incremental root: $incremental_root" >&2
    exit 2
fi

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

GEODML_MODULE_PYTHONPATH="${PYTHONPATH-}"
source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages:$GEODML_MODULE_PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

wildchat_revision="$(
    python3 - "$old_wildchat_records" <<'PY'
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
wildchat_snapshot="$GEODML_PROJECT_ROOT/restricted-data/wildchat-1m/$wildchat_revision"
LLAMA_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$old_llama_manifest")"
LLAMA_MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$old_llama_manifest")"
export LLAMA_REVISION LLAMA_MODEL

if [[ "${#wildchat_revision}" -ne 40 || ! -d "$wildchat_snapshot/data" ]]; then
    echo "invalid frozen WildChat snapshot: $wildchat_snapshot" >&2
    exit 2
fi
if [[ "${#LLAMA_REVISION}" -ne 40 || ! -s "$LLAMA_MODEL/config.json" ]]; then
    echo "invalid frozen Llama snapshot: $LLAMA_MODEL" >&2
    exit 2
fi

mkdir -p "$incremental_root"

echo "===== BUILDING NESTED SIX-SOURCE CANDIDATES ====="
python analysis/scripts/build_semantic_readiness_dataset.py collect-transfer \
    --output-dir "$candidate_records" \
    --maximum-per-source "$candidate_maximum_per_source" \
    --master-seed "$master_seed" \
    --source-input "openassistant-oasst1=$transfer_root/openassistant-oasst1/2023-04-12_oasst_all.messages.jsonl.gz" \
    --source-revision "openassistant-oasst1=fdf72ae0827c1cda404aff25b6603abec9e3399b" \
    --source-input "google-ccpe-m=$transfer_root/_upstream_git/ccpe/data.json" \
    --source-revision "google-ccpe-m=2c9cd30f33f3a154b5a27d015333679262ff36f5" \
    --source-input "google-taskmaster-1=$transfer_root/_upstream_git/taskmaster/TM-1-2019" \
    --source-revision "google-taskmaster-1=d92cb6af3005f1dc09c39e75e7daf4a04905e00b" \
    --source-input "google-schema-guided-dialogue=$transfer_root/google-schema-guided-dialogue" \
    --source-revision "google-schema-guided-dialogue=e852981ae34990f4358979625854259302feaa78" \
    --source-input "amazon-shopping-queries=$transfer_root/_upstream_git/amazon-esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet" \
    --source-revision "amazon-shopping-queries=7916cdf6ab75a462e77f20ab40428a10923998d5" \
    --source-input "allenai-wildchat-1m=$wildchat_snapshot/data" \
    --source-revision "allenai-wildchat-1m=$wildchat_revision"

candidate_path="$candidate_records/semantic_readiness_transfer_records.jsonl"
python3 - "$old_open_records" "$old_wildchat_records" "$candidate_path" <<'PY'
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
if not old <= candidate:
    raise SystemExit(f"nested sample lost {len(old - candidate)} frozen text hashes")
print(f"NESTED SAMPLE: PASS ({len(old)} frozen text hashes retained)")
PY

echo "===== REMOVING ALL PREVIOUSLY ANNOTATED PROMPTS ====="
python analysis/scripts/build_semantic_readiness_dataset.py merge-transfer \
    --base-corpus "$old_expanded_corpus" \
    --transfer-records "$candidate_path" \
    --output-dir "$incremental_corpus"

echo "===== EXPORTING FOUR-CANONICAL-JUDGE TASK BANK ====="
judge_slots="primary-frontier,replicate-frontier-a,replicate-frontier-b,replicate-frontier-c"
python analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
    --corpus "$incremental_corpus/semantic_readiness_transfer_corpus.jsonl" \
    --judge-slots "$judge_slots" \
    --output-dir "$incremental_task_bank"

READINESS_INCREMENTAL_TASKS="$incremental_task_bank/readiness_label_tasks_blinded.jsonl"
READINESS_INCREMENTAL_TASKS_PER_SLOT="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_count"])' \
        "$incremental_task_bank/run_manifest.json"
)"
READINESS_INCREMENTAL_TASKS_SHA256="$(sha256sum "$READINESS_INCREMENTAL_TASKS" | awk '{print $1}')"
READINESS_INCREMENTAL_QUEUE_ROOT="$incremental_root/judge-queue"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-16}"
export READINESS_INCREMENTAL_TASKS
export READINESS_INCREMENTAL_TASKS_PER_SLOT
export READINESS_INCREMENTAL_TASKS_SHA256
export READINESS_INCREMENTAL_QUEUE_ROOT
export LLAMA_BATCH_SIZE

python3 - \
    "$old_expanded_corpus" \
    "$incremental_corpus/semantic_readiness_transfer_corpus.jsonl" \
    "$READINESS_INCREMENTAL_TASKS" \
    "$READINESS_INCREMENTAL_TASKS_PER_SLOT" \
    "$minimum_new_prompts" <<'PY'
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
minimum = int(sys.argv[5])

old_hashes = {row["text_sha256"] for row in old}
new_hashes = {row["text_sha256"] for row in new}
if len(new) != len(new_hashes) or len(new) != expected:
    raise SystemExit("incremental corpus count or uniqueness mismatch")
if expected < minimum:
    raise SystemExit(f"only {expected} new prompts survived; require at least {minimum}")
if not old_hashes.isdisjoint(new_hashes):
    raise SystemExit("incremental corpus overlaps an already annotated text")

split_counts = collections.Counter(row["split"] for row in new)
source_counts = collections.Counter(row["source_name"] for row in new)
if split_counts["development"] < 3500:
    raise SystemExit(f"insufficient development coverage: {dict(split_counts)}")
if split_counts["confirmation"] < 5500:
    raise SystemExit(f"insufficient confirmation coverage: {dict(split_counts)}")
if len(source_counts) < 5:
    raise SystemExit(f"insufficient source coverage: {dict(source_counts)}")

counts = collections.Counter(row["judge_slot"] for row in tasks)
required = {
    "primary-frontier",
    "replicate-frontier-a",
    "replicate-frontier-b",
    "replicate-frontier-c",
}
if set(counts) != required or set(counts.values()) != {expected}:
    raise SystemExit(f"invalid four-slot task coverage: {dict(counts)}")
if len({row["task_id"] for row in tasks}) != 4 * expected:
    raise SystemExit("incremental task IDs are not unique")

print(f"NOVELTY: PASS ({expected} new prompts)")
print(f"FOUR-SLOT COVERAGE: PASS ({4 * expected} tasks)")
print(f"SPLIT COVERAGE: PASS ({dict(split_counts)})")
print(f"SOURCE COVERAGE: PASS ({dict(source_counts)})")
PY

slurm_log_dir="$incremental_root/slurm"
mkdir -p "$slurm_log_dir"
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
} > "$incremental_root/launch-environment.txt"

sha256sum \
    "$candidate_path" \
    "$incremental_corpus/semantic_readiness_transfer_corpus.jsonl" \
    "$READINESS_INCREMENTAL_TASKS" \
    > "$incremental_root/prelaunch-artifact-sha256.txt"

echo "===== SUBMITTING FOUR-CANONICAL-JUDGE QUEUE ====="
incremental_job_id="$(
    sbatch \
        --parsable \
        --account="$JUPITER_ACCOUNT" \
        --output="$slurm_log_dir/incremental-four-judge-%j.out" \
        --error="$slurm_log_dir/incremental-four-judge-%j.err" \
        --export=ALL \
        analysis/scripts/slurm/jupiter/run_readiness_incremental_four_judge_queue.sbatch
)"
printf '%s\n' "$incremental_job_id" > "$incremental_root/slurm-job-id.txt"

echo "INCREMENTAL_JOB_ID=$incremental_job_id"
echo "INCREMENTAL_ROOT=$incremental_root"
echo "NEW_PROMPTS=$READINESS_INCREMENTAL_TASKS_PER_SLOT"
echo "CANONICAL_JUDGMENTS=$((4 * READINESS_INCREMENTAL_TASKS_PER_SLOT))"
squeue -j "$incremental_job_id"
echo "ALL INPUTS VALIDATED AND INCREMENTAL JOB SUBMITTED: PASS"
