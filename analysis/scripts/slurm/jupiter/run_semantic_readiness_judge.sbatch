#!/bin/bash -l
# Resume one semantic-readiness judge slot on one complete JUPITER GH200 node.

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=readiness-judge

set -euo pipefail
umask 077

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:?Submit from the GEODML repository root}"
cd "$REPOSITORY_ROOT"

: "${JUDGE_SLOT:?Set JUDGE_SLOT}"
: "${JUDGE_TAG:?Set JUDGE_TAG}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to a local model snapshot}"
: "${JUDGE_MODEL_FAMILY:?Set JUDGE_MODEL_FAMILY to the frozen family label}"
: "${JUDGE_MODEL_REVISION:?Set JUDGE_MODEL_REVISION to an immutable revision}"
: "${READINESS_RUN:?Set READINESS_RUN}"
: "${READINESS_JUDGE_TASKS:?Set READINESS_JUDGE_TASKS}"
: "${READINESS_JUDGE_TASKS_SHA256:?Set READINESS_JUDGE_TASKS_SHA256}"
: "${READINESS_EXPECTED_TASKS_PER_SLOT:?Set READINESS_EXPECTED_TASKS_PER_SLOT}"
: "${GEODML_EXPECTED_COMMIT:?Set GEODML_EXPECTED_COMMIT}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1

jutil env activate -p "${JUPITER_PROJECT:-scifi}"
: "${FSCRATCH:?FSCRATCH is unset after project activation}"

GEODML_AI_VENV="${GEODML_AI_VENV:-$FSCRATCH/$USER/geodml/python/.venv-ai-pytorch291}"
if [[ ! -f "$GEODML_AI_VENV/bin/activate" ]]; then
    echo "Missing virtual environment: $GEODML_AI_VENV" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$GEODML_AI_VENV/bin/activate"

export HF_HOME="${HF_HOME:-$FSCRATCH/$USER/geodml/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$FSCRATCH/$USER/geodml/pip}"
export TMPDIR="${TMPDIR:-$FSCRATCH/$USER/geodml/tmp}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export GEODML_REQUIRED_GPU_COUNT=4
export GEODML_DEVICE_MAP="${GEODML_DEVICE_MAP:-balanced}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

if [[ "$(cat /etc/FZJ/systemname)" != "jupiter" ]]; then
    echo "JUPITER system contract violated" >&2
    exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "booster" ]]; then
    echo "JUPITER partition contract violated: ${SLURM_JOB_PARTITION:-unset}" >&2
    exit 2
fi

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "Commit mismatch: expected $GEODML_EXPECTED_COMMIT, found $actual_commit" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Scientific judge runs require a clean Git worktree" >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$READINESS_JUDGE_TASKS" ]]; then
    echo "Missing judge tasks: $READINESS_JUDGE_TASKS" >&2
    exit 2
fi
actual_task_sha256="$(sha256sum "$READINESS_JUDGE_TASKS" | awk '{print $1}')"
if [[ "$actual_task_sha256" != "$READINESS_JUDGE_TASKS_SHA256" ]]; then
    echo "Task-bank hash mismatch: expected $READINESS_JUDGE_TASKS_SHA256, found $actual_task_sha256" >&2
    exit 2
fi
if [[ ! -f "$JUDGE_MODEL/config.json" ]]; then
    echo "Missing model snapshot: $JUDGE_MODEL" >&2
    exit 2
fi

visible_gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$visible_gpu_count" != "4" ]]; then
    echo "Four-GPU contract violated: PyTorch sees $visible_gpu_count GPUs" >&2
    exit 2
fi
mapfile -t visible_gpu_names < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
)
if [[ "${#visible_gpu_names[@]}" != "4" ]]; then
    echo "GH200 hardware contract violated: nvidia-smi found ${#visible_gpu_names[@]} GPUs" >&2
    exit 2
fi
for gpu_name in "${visible_gpu_names[@]}"; do
    if [[ "$gpu_name" != *GH200* ]]; then
        echo "GH200 hardware contract violated: found $gpu_name" >&2
        exit 2
    fi
done

output_directory="$READINESS_RUN/judges/$JUDGE_TAG"
mkdir -p "$output_directory"

module -t list 2> "$output_directory/modules.txt"
python --version > "$output_directory/python-version.txt"
python -m pip freeze > "$output_directory/python-packages.txt"
scontrol show job "$SLURM_JOB_ID" > "$output_directory/slurm-job.txt"
nvidia-smi > "$output_directory/gpu-environment.txt"
printf '%s\n' "$actual_commit" > "$output_directory/git-commit.txt"
printf '%s\n' "$JUDGE_MODEL_FAMILY" > "$output_directory/model-family.txt"
printf '%s\n' "$JUDGE_MODEL_REVISION" > "$output_directory/model-revision.txt"

echo "[readiness] job=$SLURM_JOB_ID"
echo "[readiness] commit=$actual_commit"
echo "[readiness] slot=$JUDGE_SLOT"
echo "[readiness] model=$JUDGE_MODEL"
echo "[readiness] model_family=$JUDGE_MODEL_FAMILY"
echo "[readiness] model_revision=$JUDGE_MODEL_REVISION"
echo "[readiness] output=$output_directory"
echo "[readiness] device_map=$GEODML_DEVICE_MAP"

skip_arguments=()
if [[ -n "${READINESS_SKIP_TASK_ID:-}" ]]; then
    skip_arguments+=(--skip-task-id "$READINESS_SKIP_TASK_ID")
fi

srun --ntasks=1 python -u analysis/scripts/run_semantic_readiness_judge.py \
    --tasks "$READINESS_JUDGE_TASKS" \
    --judge-slot "$JUDGE_SLOT" \
    --backend local \
    --model "$JUDGE_MODEL" \
    --model-family "$JUDGE_MODEL_FAMILY" \
    --model-revision "$JUDGE_MODEL_REVISION" \
    --precision full \
    --max-new-tokens 300 \
    --maximum-attempts 5 \
    --run-purpose production \
    --resume \
    "${skip_arguments[@]}" \
    --output-dir "$output_directory"

python - "$output_directory/run_manifest.json" "$READINESS_EXPECTED_TASKS_PER_SLOT" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if manifest["task_count_for_slot"] != expected:
    raise SystemExit(
        f"slot task-count mismatch: expected {expected}, "
        f"found {manifest['task_count_for_slot']}"
    )
if manifest["selected_task_count"] != expected:
    raise SystemExit(
        f"selected task-count mismatch: expected {expected}, "
        f"found {manifest['selected_task_count']}"
    )
if manifest["completed_count"] + manifest["skipped_count"] != expected:
    raise SystemExit(
        f"incomplete judge slot: completed={manifest['completed_count']}, "
        f"skipped={manifest['skipped_count']}, expected={expected}"
    )
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$output_directory/completed-at.txt"
sha256sum \
    "$output_directory/judge_responses.jsonl" \
    "$output_directory/run_manifest.json" \
    > "$output_directory/artifact-sha256.txt"
