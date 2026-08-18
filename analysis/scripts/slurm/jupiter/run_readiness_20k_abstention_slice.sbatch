#!/bin/bash -l
# Run one approved, bounded slice of the resumable 20k abstention-v2 panel.
# The submission command must provide --time after explicit user approval.

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=readiness-20k-v2

set -euo pipefail
umask 077

: "${PROJECT:?PROJECT is required}"
: "${FSCRATCH:?FSCRATCH is required}"
: "${USER:?USER is required}"
: "${SLURM_JOB_ID:?submit this script through Slurm}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"
: "${READINESS_20K_TASKS:?READINESS_20K_TASKS is required}"
: "${READINESS_20K_TASKS_SHA256:?READINESS_20K_TASKS_SHA256 is required}"
: "${READINESS_20K_TASKS_PER_SLOT:?READINESS_20K_TASKS_PER_SLOT is required}"
: "${READINESS_20K_QUEUE_ROOT:?READINESS_20K_QUEUE_ROOT is required}"
: "${READINESS_APPROVED_WALLTIME:?record the explicitly approved Slurm walltime}"
: "${READINESS_ALLOCATION_ESTIMATE:?record the estimate that justified this allocation}"
: "${READINESS_SLICE_BUDGET_SECONDS:?set a safe internal budget below the approved walltime}"
: "${LLAMA_MODEL:?LLAMA_MODEL must name a local Llama 3.3 70B snapshot}"
: "${LLAMA_REVISION:?LLAMA_REVISION must be an immutable snapshot revision}"

if (( READINESS_SLICE_BUDGET_SECONDS < 900 )); then
    echo "READINESS_SLICE_BUDGET_SECONDS must leave at least 15 minutes of useful work" >&2
    exit 2
fi

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono}"
GEODML_MODEL_VENV="${GEODML_MODEL_VENV:-$FSCRATCH/$USER/geodml/python/.venv-model-panel-transformers5141}"
module_pythonpath="${PYTHONPATH-}"

source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages:$module_pythonpath"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export GEODML_REQUIRED_GPU_COUNT=4
export GEODML_DEVICE_MAP=balanced
export GEODML_ATTENTION_IMPLEMENTATION=sdpa

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "scientific slice requires a clean checkout" >&2
    exit 2
fi
if [[ "$(sha256sum "$READINESS_20K_TASKS" | awk '{print $1}')" != "$READINESS_20K_TASKS_SHA256" ]]; then
    echo "task-bank hash mismatch" >&2
    exit 2
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "4" ]]; then
    echo "expected four visible GPUs" >&2
    exit 2
fi
while IFS= read -r gpu_name; do
    if [[ "$gpu_name" != *GH200* ]]; then
        echo "expected GH200 GPUs, found $gpu_name" >&2
        exit 2
    fi
done < <(nvidia-smi --query-gpu=name --format=csv,noheader)

python3 - "$READINESS_20K_TASKS" "$READINESS_20K_TASKS_PER_SLOT" <<'PY'
import collections
import json
import pathlib
import sys

tasks = [
    json.loads(line)
    for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
expected = int(sys.argv[2])
counts = collections.Counter(row["judge_slot"] for row in tasks)
required = {
    "primary-frontier",
    "replicate-frontier-a",
    "replicate-frontier-b",
    "replicate-frontier-c",
}
if set(counts) != required or set(counts.values()) != {expected}:
    raise SystemExit(f"invalid four-slot task coverage: {dict(counts)}")
if {row["rubric_version"] for row in tasks} != {
    "decision-readiness-ordinal-abstention-v2"
}:
    raise SystemExit("task bank is not frozen to abstention-v2")
print(f"20k v2 task bank OK: prompts={expected}; tasks={len(tasks)}")
PY

qwen32_revision="9216db5781bf21249d130ec9da846c4624c16137"
qwen32_model="$GEODML_PROJECT_ROOT/models/qwen/Qwen3-32B/$qwen32_revision"
ministral_revision="f6fae9795746f63c9be8344932f01275f3c63734"
ministral_model="$GEODML_PROJECT_ROOT/models/mistral/Ministral-3-8B-Instruct-2512-BF16/$ministral_revision"
gemma_revision="842da3794eaa0b77d5f08bae87a17459d91ff475"
gemma_model="$GEODML_PROJECT_ROOT/models/gemma/gemma-4-31B-it/$gemma_revision"
for model in "$qwen32_model" "$ministral_model" "$gemma_model" "$LLAMA_MODEL"; do
    if [[ ! -s "$model/config.json" ]]; then
        echo "missing model snapshot: $model" >&2
        exit 2
    fi
done

queue_root="$READINESS_20K_QUEUE_ROOT"
mkdir -p "$queue_root/logs" "$queue_root/smoke" "$queue_root/full" "$queue_root/slices"
if [[ -s "$queue_root/queue.complete" ]]; then
    echo "queue already complete: $queue_root"
    exit 0
fi

slice_started_epoch="$(date +%s)"
slice_dir="$queue_root/slices/$SLURM_JOB_ID"
mkdir -p "$slice_dir"
printf '%s\n' "$READINESS_APPROVED_WALLTIME" > "$slice_dir/approved-walltime.txt"
printf '%s\n' "$READINESS_ALLOCATION_ESTIMATE" > "$slice_dir/allocation-estimate.txt"
printf '%s\n' "$READINESS_SLICE_BUDGET_SECONDS" > "$slice_dir/internal-budget-seconds.txt"
scontrol show job "$SLURM_JOB_ID" > "$slice_dir/slurm-job.txt"
nvidia-smi > "$slice_dir/gpu-environment-start.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$slice_dir/started-at.txt"
git rev-parse HEAD > "$slice_dir/git-commit.txt"

cache_count() {
    local output="$1"
    if [[ ! -d "$output/task_cache" ]]; then
        echo 0
        return
    fi
    find "$output/task_cache" -maxdepth 1 -type f -name '*.json' ! -name '*.failed.json' | wc -l
}

remaining_budget() {
    local now elapsed
    now="$(date +%s)"
    elapsed=$((now - slice_started_epoch))
    echo $((READINESS_SLICE_BUDGET_SECONDS - elapsed))
}

checkpoint_and_exit() {
    local stage="$1"
    local output="$2"
    local completed="$3"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] slice checkpointed stage=$stage completed=$completed output=$output" \
        | tee -a "$queue_root/queue.log" "$slice_dir/status.txt"
    nvidia-smi > "$slice_dir/gpu-environment-end.txt" || true
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$slice_dir/completed-at.txt"
    exit 0
}

run_bounded() {
    local stage="$1"
    local output="$2"
    local expected="$3"
    shift 3
    local completed remaining status now
    completed="$(cache_count "$output")"
    if [[ "$completed" -eq "$expected" ]]; then
        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] skipping complete $stage ($completed/$expected)" \
            | tee -a "$queue_root/queue.log"
        return 0
    fi
    remaining="$(remaining_budget)"
    if (( remaining < 600 )); then
        checkpoint_and_exit "$stage" "$output" "$completed"
    fi
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] starting/resuming $stage cached=$completed expected=$expected budget=${remaining}s" \
        | tee -a "$queue_root/queue.log"
    set +e
    timeout --signal=TERM --kill-after=90s "${remaining}s" "$@"
    status=$?
    set -e
    completed="$(cache_count "$output")"
    if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
        now="$(remaining_budget)"
        if (( now > 180 )); then
            echo "bounded runner exited $status before its deadline" >&2
            return "$status"
        fi
        checkpoint_and_exit "$stage" "$output" "$completed"
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "stage failed: $stage exit=$status cached=$completed/$expected" >&2
        return "$status"
    fi
    if [[ "$completed" -ne "$expected" ]]; then
        echo "stage returned success but is incomplete: $stage cached=$completed/$expected" >&2
        return 2
    fi
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] finished $stage cached=$completed/$expected" \
        | tee -a "$queue_root/queue.log"
}

run_data_parallel() {
    local stage="$1" tag="$2" slot="$3" model="$4" family="$5" revision="$6" batch_size="$7" limit="$8"
    local output="$queue_root/$stage/$tag"
    local expected="$READINESS_20K_TASKS_PER_SLOT"
    local purpose=production
    local -a args=(
        python -m torch.distributed.run --standalone --nproc-per-node=4
        analysis/scripts/run_semantic_readiness_judge_4gpu.py
        --tasks "$READINESS_20K_TASKS"
        --tasks-sha256 "$READINESS_20K_TASKS_SHA256"
        --expected-tasks "$READINESS_20K_TASKS_PER_SLOT"
        --judge-slot "$slot"
        --model "$model"
        --model-family "$family"
        --model-revision "$revision"
        --output-dir "$output"
        --batch-size "$batch_size"
        --max-input-tokens 2048
        --max-new-tokens 300
        --maximum-attempts 5
        --expected-world-size 4
        --attention-implementation sdpa
    )
    if [[ -n "$limit" ]]; then
        args+=(--limit "$limit")
        expected="$limit"
        purpose=debug
    fi
    args+=(--run-purpose "$purpose")
    if [[ "$family" == qwen ]]; then
        args+=(--disable-thinking)
    fi
    if [[ -e "$output" ]]; then
        args+=(--resume)
    fi
    run_bounded "$stage/$tag" "$output" "$expected" "${args[@]}"
}

run_llama() {
    local stage="$1" limit="$2"
    local tag="llama3.3-70b-replicate-c"
    local output="$queue_root/$stage/$tag"
    local expected="$READINESS_20K_TASKS_PER_SLOT"
    local purpose=production
    local -a args=(
        python analysis/scripts/run_semantic_readiness_judge.py
        --tasks "$READINESS_20K_TASKS"
        --judge-slot replicate-frontier-c
        --backend local
        --model "$LLAMA_MODEL"
        --model-family llama
        --model-revision "$LLAMA_REVISION"
        --precision full
        --batch-size "${LLAMA_BATCH_SIZE:-16}"
        --max-new-tokens 300
        --maximum-attempts 5
        --output-dir "$output"
    )
    if [[ -n "$limit" ]]; then
        args+=(--limit "$limit")
        expected="$limit"
        purpose=debug
    fi
    args+=(--run-purpose "$purpose")
    if [[ -e "$output" ]]; then
        args+=(--resume)
    fi
    run_bounded "$stage/$tag" "$output" "$expected" "${args[@]}"
}

# Every smoke cache is retained, so subsequent slices skip these checks.
run_data_parallel smoke qwen3-32b-replicate-a replicate-frontier-a "$qwen32_model" qwen "$qwen32_revision" 16 8
run_data_parallel smoke ministral3-8b-replicate-b replicate-frontier-b "$ministral_model" mistral "$ministral_revision" 32 8
run_data_parallel smoke gemma4-31b-primary primary-frontier "$gemma_model" gemma "$gemma_revision" 8 8
run_llama smoke 8

# Finish quick models first, then use later slices for the slower models.
run_data_parallel full qwen3-32b-replicate-a replicate-frontier-a "$qwen32_model" qwen "$qwen32_revision" 16 ""
run_data_parallel full ministral3-8b-replicate-b replicate-frontier-b "$ministral_model" mistral "$ministral_revision" 32 ""
run_data_parallel full gemma4-31b-primary primary-frontier "$gemma_model" gemma "$gemma_revision" 8 ""
run_llama full ""

python3 - "$queue_root" "$READINESS_20K_TASKS" "$READINESS_20K_TASKS_PER_SLOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
tasks_path = pathlib.Path(sys.argv[2])
expected = int(sys.argv[3])
tasks = [
    json.loads(line)
    for line in tasks_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by_slot = {}
for row in tasks:
    by_slot.setdefault(row["judge_slot"], {})[row["task_id"]] = row

outputs = {
    "primary-frontier": "gemma4-31b-primary",
    "replicate-frontier-a": "qwen3-32b-replicate-a",
    "replicate-frontier-b": "ministral3-8b-replicate-b",
    "replicate-frontier-c": "llama3.3-70b-replicate-c",
}
for slot, tag in outputs.items():
    cache = root / "full" / tag / "task_cache"
    successful = [path for path in cache.glob("*.json") if not path.name.endswith(".failed.json")]
    failed = list(cache.glob("*.failed.json"))
    if len(successful) != expected or failed:
        raise SystemExit(
            f"incomplete {slot}: successful={len(successful)} failed={len(failed)} expected={expected}"
        )
    seen = set()
    for path in successful:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["task_id"] not in by_slot[slot] or row["task_id"] in seen:
            raise SystemExit(f"unexpected or duplicate cache task: {path}")
        seen.add(row["task_id"])
    if seen != set(by_slot[slot]):
        raise SystemExit(f"cache coverage mismatch for {slot}")
    print(f"{slot}: PASS ({len(seen)}/{expected})")
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$slice_dir/completed-at.txt"
nvidia-smi > "$slice_dir/gpu-environment-end.txt"
echo "queue complete: $queue_root" | tee -a "$queue_root/queue.log" "$queue_root/queue.complete"
find "$queue_root" -type f ! -name artifact-sha256.txt -print0 | sort -z | xargs -0 sha256sum > "$queue_root/artifact-sha256.txt"
