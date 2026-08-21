#!/bin/bash -l
# Measure four-GPU readiness-question generation throughput and wording diversity.
# This runner never requests resources; invoke it inside an approved 4-GPU job.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    cleaned_path=""
    IFS=: read -r -a path_entries <<< "$PATH"
    for path_entry in "${path_entries[@]}"; do
        [[ "$path_entry" == "$VIRTUAL_ENV/bin" ]] && continue
        cleaned_path="${cleaned_path:+$cleaned_path:}$path_entry"
    done
    export PATH="$cleaned_path"
fi
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CUDA_VISIBLE_DEVICES
hash -r

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1
jutil env activate -p "${JUPITER_PROJECT:-scifi}"
hash -r

export GEODML_MODULE_PYTHONPATH="${PYTHONPATH:-}"
export GEODML_PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$GEODML_PYTHON_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_RUNS_ROOT="${GEODML_RUNS_ROOT:-$GEODML_PROJECT_ROOT/runs}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"
export GEODML_MODEL_VENV="${GEODML_MODEL_VENV:-$GEODML_CACHE_ROOT/python/.venv-model-panel-transformers5141}"

source "$GEODML_MODEL_VENV/bin/activate"
export PYTHONPATH="$GEODML_MODEL_VENV/lib/python3.13/site-packages${GEODML_MODULE_PYTHONPATH:+:$GEODML_MODULE_PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

actual_commit="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
[[ "$actual_commit" == "$GEODML_EXPECTED_COMMIT" ]] || {
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
}
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]] || {
    echo "generation requires a clean exact-commit checkout" >&2
    exit 2
}

plan_pointer="${READINESS_30K_PLAN_POINTER:-$GEODML_PROJECT_ROOT/geodml-readiness-30k-v2-plan-latest.txt}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$(<"$plan_pointer")}"
export READINESS_30K_TASKS="$READINESS_30K_PLAN_ROOT/generation_tasks_round_00.jsonl"
test -s "$READINESS_30K_TASKS"

qwen_revision="9216db5781bf21249d130ec9da846c4624c16137"
gemma_revision="842da3794eaa0b77d5f08bae87a17459d91ff475"
export QWEN_GENERATOR_MODEL="${QWEN_GENERATOR_MODEL:-$GEODML_MODELS_ROOT/qwen/Qwen3-32B/$qwen_revision}"
export GEMMA_GENERATOR_MODEL="${GEMMA_GENERATOR_MODEL:-$GEODML_MODELS_ROOT/gemma/gemma-4-31B-it/$gemma_revision}"
test -s "$QWEN_GENERATOR_MODEL/config.json"
test -s "$GEMMA_GENERATOR_MODEL/config.json"

export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}"
export READINESS_30K_PILOT_ROOT="${READINESS_30K_PILOT_ROOT:-$GEODML_RUNS_ROOT/readiness-30k-four-gpu-pilot/$run_id}"
mkdir -p "$READINESS_30K_PILOT_ROOT"/{candidates,cache,logs,audits}

cd "$GEODML_REPOSITORY"
git rev-parse HEAD > "$READINESS_30K_PILOT_ROOT/git-commit.txt"
printf '%s\n' "01:00:00" > "$READINESS_30K_PILOT_ROOT/approved-walltime.txt"
printf '%s\n' "four one-GPU workers for 50 minutes plus audit margin" > "$READINESS_30K_PILOT_ROOT/runtime-estimate-basis.txt"
scontrol show job "$SLURM_JOB_ID" -o > "$READINESS_30K_PILOT_ROOT/slurm-job.txt"
module list > "$READINESS_30K_PILOT_ROOT/modules.txt" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "$READINESS_30K_PILOT_ROOT/started-at.txt"

run_worker() {
    local generator_id="$1" model="$2" shard_index="$3" worker_id="$4"
    local output="$READINESS_30K_PILOT_ROOT/candidates/$worker_id.jsonl"
    local cache="$READINESS_30K_PILOT_ROOT/cache/$worker_id"
    local log="$READINESS_30K_PILOT_ROOT/logs/$worker_id.log"
    mkdir -p "$cache"
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        --cpu-bind=none \
        python analysis/scripts/build_readiness_prompt_population.py generate \
        --tasks "$READINESS_30K_TASKS" \
        --generator-id "$generator_id" \
        --backend local \
        --model "$model" \
        --precision full \
        --cache-dir "$cache" \
        --output "$output" \
        --temperature 0.9 \
        --max-new-tokens 180 \
        --maximum-attempts 5 \
        --shard-count 2 \
        --shard-index "$shard_index" \
        --maximum-runtime-seconds "$READINESS_GENERATION_SECONDS" \
        --resume > "$log" 2>&1
}

run_worker qwen3-32b "$QWEN_GENERATOR_MODEL" 0 qwen-worker-0 &
worker_pids=("$!")
run_worker qwen3-32b "$QWEN_GENERATOR_MODEL" 1 qwen-worker-1 &
worker_pids+=("$!")
run_worker gemma4-31b "$GEMMA_GENERATOR_MODEL" 0 gemma-worker-0 &
worker_pids+=("$!")
run_worker gemma4-31b "$GEMMA_GENERATOR_MODEL" 1 gemma-worker-1 &
worker_pids+=("$!")

worker_failure=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        worker_failure=1
    fi
done

candidate_files=("$READINESS_30K_PILOT_ROOT"/candidates/*.jsonl)
if [[ "${candidate_files[0]}" == *'*'* ]]; then
    echo "no worker produced a candidate file" >&2
    exit 3
fi

set +e
python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
    --questions "${candidate_files[@]}" \
    --minimum-delexicalized-unique-fraction 0.90 \
    --maximum-template-fraction 0.01 \
    --minimum-median-keyword-unique-fraction 0.90 \
    --minimum-keyword-unique-fraction 0.70 \
    --maximum-opening-frame-fraction 0.05 \
    --output-dir "$READINESS_30K_PILOT_ROOT/audits/diversity"
diversity_exit="$?"
set -e

python - "$READINESS_30K_PILOT_ROOT" "$READINESS_30K_PLAN_ROOT/plan_manifest.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
workers = []
for path in sorted((root / "candidates").glob("*.jsonl.manifest.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    workers.append(
        {
            "worker": path.name.removesuffix(".jsonl.manifest.json"),
            "generator_id": manifest["generator_id"],
            "candidate_count": manifest["candidate_count"],
            "completed_task_count": manifest["completed_task_count"],
            "elapsed_seconds": manifest["elapsed_seconds"],
            "slice_complete": manifest["slice_complete"],
        }
    )
candidate_count = sum(row["candidate_count"] for row in workers)
gpu_seconds = sum(row["elapsed_seconds"] for row in workers)
rate = candidate_count / gpu_seconds if gpu_seconds else 0.0
planned = int(plan["maximum_planned_candidate_count"])
estimated_gpu_hours = planned / rate / 3600 if rate else None
summary = {
    "format_version": "readiness-30k-four-gpu-throughput-v1",
    "planned_target_count": int(plan["task_count"]),
    "planned_candidate_count": planned,
    "pilot_candidate_count": candidate_count,
    "pilot_gpu_hours": gpu_seconds / 3600,
    "candidates_per_gpu_hour": rate * 3600,
    "estimated_full_generation_gpu_hours": estimated_gpu_hours,
    "estimated_full_generation_hours_on_four_gpus": (
        estimated_gpu_hours / 4 if estimated_gpu_hours is not None else None
    ),
    "workers": workers,
}
(root / "throughput_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf '%s\n' "$worker_failure" > "$READINESS_30K_PILOT_ROOT/worker-failure.txt"
printf '%s\n' "$diversity_exit" > "$READINESS_30K_PILOT_ROOT/diversity-exit-code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$READINESS_30K_PILOT_ROOT/finished-at.txt"
printf '%s\n' "$READINESS_30K_PILOT_ROOT" > "$GEODML_PROJECT_ROOT/geodml-readiness-30k-four-gpu-pilot-latest.txt"

echo "worker_failure=$worker_failure"
echo "diversity_exit=$diversity_exit"
echo "RUN_ROOT=$READINESS_30K_PILOT_ROOT"
