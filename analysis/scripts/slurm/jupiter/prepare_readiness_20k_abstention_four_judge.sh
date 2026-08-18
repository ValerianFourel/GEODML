#!/bin/bash -l
# Prepare (but never submit) the combined 20k abstention-v2 task bank.

set -euo pipefail
umask 077

: "${PROJECT:?PROJECT is required}"
: "${USER:?USER is required}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"

JUPITER_PROJECT="${JUPITER_PROJECT:-scifi}"
jutil env activate -p "$JUPITER_PROJECT"

GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono}"
GEODML_MODEL_VENV="${GEODML_MODEL_VENV:?GEODML_MODEL_VENV is required}"
GEODML_RUNS_ROOT="${GEODML_RUNS_ROOT:-$GEODML_PROJECT_ROOT/runs}"

input_commit="40527b0fbc1fddb9f3a377f70e1405effd4ebc19"
input_root="$GEODML_RUNS_ROOT/semantic-readiness-incremental/$input_commit-six-source-atleast10000-four-judge"
combined_corpus="$input_root/incremental-corpus/semantic_readiness_expanded_corpus.jsonl"
old_expansion_commit="d5363028b509a3dc9686f7e1755a9f1c83e985b3"
old_expansion_root="$GEODML_RUNS_ROOT/semantic-readiness-expansion/$old_expansion_commit-wildchat1000-llama33-70b"
old_llama_manifest="$old_expansion_root/judge-queue/full/llama3.3-70b-replicate-c/run_manifest.json"

minimum_prompts="${READINESS_MINIMUM_COMBINED_PROMPTS:-20000}"
run_root="${READINESS_20K_ROOT:-$GEODML_RUNS_ROOT/semantic-readiness-20k-abstention/$GEODML_EXPECTED_COMMIT-combined-atleast20000-four-judge-v2}"
task_bank="$run_root/task-bank-four-judge-v2"

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "scientific preparation requires a clean checkout" >&2
    exit 2
fi
for path in "$combined_corpus" "$old_llama_manifest"; do
    if [[ ! -s "$path" ]]; then
        echo "missing frozen input: $path" >&2
        exit 2
    fi
done
if [[ -e "$run_root" ]]; then
    echo "refusing to overwrite prepared run root: $run_root" >&2
    exit 2
fi

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

module_pythonpath="${PYTHONPATH-}"
source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages:$module_pythonpath"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mkdir -p "$run_root"
judge_slots="primary-frontier,replicate-frontier-a,replicate-frontier-b,replicate-frontier-c"
rubric_version="decision-readiness-ordinal-abstention-v2"

python analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
    --corpus "$combined_corpus" \
    --judge-slots "$judge_slots" \
    --rubric-version "$rubric_version" \
    --output-dir "$task_bank"

READINESS_20K_TASKS="$task_bank/readiness_label_tasks_blinded.jsonl"
READINESS_20K_TASKS_PER_SLOT="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus_count"])' \
        "$task_bank/run_manifest.json"
)"
READINESS_20K_TASKS_SHA256="$(sha256sum "$READINESS_20K_TASKS" | awk '{print $1}')"
READINESS_20K_QUEUE_ROOT="$run_root/judge-queue"
LLAMA_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$old_llama_manifest")"
LLAMA_MODEL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$old_llama_manifest")"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-16}"

python3 - \
    "$combined_corpus" \
    "$READINESS_20K_TASKS" \
    "$READINESS_20K_TASKS_PER_SLOT" \
    "$minimum_prompts" \
    "$rubric_version" <<'PY'
import collections
import json
import pathlib
import sys

def rows(path):
    return [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

corpus = rows(sys.argv[1])
tasks = rows(sys.argv[2])
expected = int(sys.argv[3])
minimum = int(sys.argv[4])
rubric = sys.argv[5]

if len(corpus) != expected or expected < minimum:
    raise SystemExit(
        f"combined corpus count mismatch: corpus={len(corpus)} expected={expected} minimum={minimum}"
    )
if len({row["item_id"] for row in corpus}) != expected:
    raise SystemExit("combined corpus contains duplicate item IDs")
if len({row["text_sha256"] for row in corpus}) != expected:
    raise SystemExit("combined corpus contains duplicate normalized texts")

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
    raise SystemExit("v2 task IDs are not unique")
if {row["rubric_version"] for row in tasks} != {rubric}:
    raise SystemExit("task bank does not use only the abstention-v2 rubric")
if any('"dont_know"' not in row["prompt"] for row in tasks):
    raise SystemExit("a blinded task prompt lacks the dont_know answer option")

items_by_slot = collections.defaultdict(set)
for row in tasks:
    items_by_slot[row["judge_slot"]].add(row["item_id"])
if any(values != items_by_slot["primary-frontier"] for values in items_by_slot.values()):
    raise SystemExit("judge slots do not cover identical prompt items")

print(f"COMBINED CORPUS: PASS ({expected} unique prompts)")
print(f"FOUR-SLOT COVERAGE: PASS ({4 * expected} v2 tasks)")
print("DONT-KNOW CONTRACT: PASS")
print(f"SOURCE COUNTS: {dict(collections.Counter(row['source_name'] for row in corpus))}")
print(f"SPLIT COUNTS: {dict(collections.Counter(row['split'] for row in corpus))}")
PY

mkdir -p "$run_root/slurm"
{
    printf 'export GEODML_EXPECTED_COMMIT=%q\n' "$GEODML_EXPECTED_COMMIT"
    printf 'export GEODML_PROJECT_ROOT=%q\n' "$GEODML_PROJECT_ROOT"
    printf 'export GEODML_REPOSITORY=%q\n' "$GEODML_REPOSITORY"
    printf 'export GEODML_MODEL_VENV=%q\n' "$GEODML_MODEL_VENV"
    printf 'export READINESS_20K_ROOT=%q\n' "$run_root"
    printf 'export READINESS_20K_TASKS=%q\n' "$READINESS_20K_TASKS"
    printf 'export READINESS_20K_TASKS_SHA256=%q\n' "$READINESS_20K_TASKS_SHA256"
    printf 'export READINESS_20K_TASKS_PER_SLOT=%q\n' "$READINESS_20K_TASKS_PER_SLOT"
    printf 'export READINESS_20K_QUEUE_ROOT=%q\n' "$READINESS_20K_QUEUE_ROOT"
    printf 'export LLAMA_MODEL=%q\n' "$LLAMA_MODEL"
    printf 'export LLAMA_REVISION=%q\n' "$LLAMA_REVISION"
    printf 'export LLAMA_BATCH_SIZE=%q\n' "$LLAMA_BATCH_SIZE"
} > "$run_root/launch-environment.txt"

sha256sum \
    "$combined_corpus" \
    "$READINESS_20K_TASKS" \
    "$task_bank/readiness_label_codebook_private.jsonl" \
    > "$run_root/prelaunch-artifact-sha256.txt"

echo "PREPARATION: PASS"
echo "READINESS_20K_ROOT=$run_root"
echo "PROMPTS=$READINESS_20K_TASKS_PER_SLOT"
echo "CANONICAL_JUDGMENTS=$((4 * READINESS_20K_TASKS_PER_SLOT))"
echo "No Slurm allocation was submitted."
