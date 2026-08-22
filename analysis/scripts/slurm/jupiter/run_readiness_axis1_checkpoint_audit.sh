#!/bin/bash -l
# Project and audit an immutable axis-1 generation checkpoint on four GPUs.
#
# GPU 0: Qwen LLM2Vec projection
# GPU 1: Mistral LLM2Vec projection
# GPU 2: independent validator shard 0
# GPU 3: independent validator shard 1
#
# This runner never allocates resources. Run it inside an approved four-GPU
# Slurm allocation. The raw continuity audit is emitted as soon as both
# projections finish; validator cache records continue accumulating afterward.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing JUPITER Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved allocation wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting the allocation}"
: "${READINESS_30K_PLAN_ROOT:?Set the immutable axis-1 plan root}"
: "${READINESS_SOURCE_PIPELINE_ROOT:?Set the generation checkpoint root}"
: "${READINESS_AXIS1_AUDIT_ROOT:?Set a fresh or resumable audit root}"

clear_runtime() {
    local inherited="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin}" cleaned="" entry
    local entries=()
    if [[ -n "$inherited" ]]; then
        IFS=: read -r -a entries <<< "$PATH"
        for entry in "${entries[@]}"; do
            [[ "$entry" == "$inherited" ]] && continue
            cleaned="${cleaned:+$cleaned:}$entry"
        done
        export PATH="$cleaned"
    fi
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
    hash -r
}

load_stack() {
    module --force purge
    module load Stages/2026
    module load GCCcore/14.3.0
    module load SciPy-Stack/2025b
    module load git
    module load PyTorch/2.9.1
    jutil env activate -p "${JUPITER_PROJECT:-scifi}"
}

activate_control_runtime() {
    local module_pythonpath="${PYTHONPATH:-}" python_prefix
    python_prefix="$(python3 -c 'import sys; print(sys.base_prefix)')"
    export LD_LIBRARY_PATH="$python_prefix/lib:${LD_LIBRARY_PATH:-}"
    source "$GEODML_GENERATOR_VENV/bin/activate"
    export PYTHONPATH="$GEODML_GENERATOR_VENV/lib/python3.13/site-packages${module_pythonpath:+:$module_pythonpath}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
}

artifact_count_matches() {
    local manifest="$1" expected="$2"
    [[ -s "$manifest" ]] || return 1
    python - "$manifest" "$expected" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
raise SystemExit(0 if int(value["candidate_count"]) == int(sys.argv[2]) else 1)
PY
}

stop_active_steps() {
    local pid
    for pid in "${active_srun_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${active_srun_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    active_srun_pids=()
}

interrupt_audit() {
    trap - INT TERM
    echo "controller interrupted; stopping active Slurm steps; caches are preserved" >&2
    stop_active_steps
    exit 130
}

trap interrupt_audit INT TERM

clear_runtime
load_stack

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"
export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"

gpu_descriptor="${READINESS_ALLOCATED_GPU_COUNT:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-}}}}"
[[ -n "$gpu_descriptor" ]] || {
    echo "cannot determine allocated GPU count; set READINESS_ALLOCATED_GPU_COUNT=4" >&2
    exit 2
}
allocated_gpu_count="$(python3 - "$gpu_descriptor" <<'PY'
import re
import sys

matches = re.findall(r"[0-9]+", sys.argv[1])
if not matches:
    raise SystemExit("allocated GPU descriptor contains no count")
print(matches[0])
PY
)"
[[ "$allocated_gpu_count" -eq 4 ]] || {
    echo "axis-1 checkpoint audit requires exactly four allocated GPUs; found $allocated_gpu_count" >&2
    exit 2
}
export READINESS_ALLOCATED_GPU_COUNT="$allocated_gpu_count"

for runtime in "$GEODML_GENERATOR_VENV" "$QWEN_LLM2VEC_VENV" "$MISTRAL_LLM2VEC_VENV"; do
    [[ -x "$runtime/bin/python" ]] || {
        echo "missing required isolated runtime: $runtime" >&2
        exit 2
    }
done

cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
[[ "$actual_commit" == "$GEODML_EXPECTED_COMMIT" ]] || {
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
    echo "checkpoint audit requires a clean exact-commit checkout" >&2
    exit 2
}

test -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json"
test -s "$READINESS_30K_PLAN_ROOT/target_grid.jsonl"
test -s "$READINESS_30K_PLAN_ROOT/subspace_bounds.json"
python3 - "$READINESS_30K_PLAN_ROOT/plan_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("target_design") != "axis-1-linear":
    raise SystemExit("the checkpoint audit requires an axis-1-linear plan")
PY

candidate_directory="$READINESS_SOURCE_PIPELINE_ROOT/round-00/generation/candidates"
mapfile -t candidate_files < <(
    find "$candidate_directory" -maxdepth 1 -type f -name '*.jsonl' \
        ! -name '*.failures.jsonl' | sort
)
[[ "${#candidate_files[@]}" -gt 0 ]] || {
    echo "no checkpointed candidate JSONL files found in $candidate_directory" >&2
    exit 2
}
for candidate_file in "${candidate_files[@]}"; do
    test -s "$candidate_file"
    test -s "$candidate_file.manifest.json" || {
        echo "candidate worker has not checkpointed its manifest: $candidate_file" >&2
        exit 2
    }
done

mkdir -p "$READINESS_AXIS1_AUDIT_ROOT/logs" "$READINESS_AXIS1_AUDIT_ROOT/projections"
candidate_list="$READINESS_AXIS1_AUDIT_ROOT/candidate-files.txt"
candidate_list_temporary="$candidate_list.tmp"
printf '%s\n' "${candidate_files[@]}" > "$candidate_list_temporary"
if [[ -s "$candidate_list" ]] && ! cmp -s "$candidate_list" "$candidate_list_temporary"; then
    echo "audit root belongs to a different immutable candidate checkpoint" >&2
    exit 2
fi
mv "$candidate_list_temporary" "$candidate_list"
export READINESS_CANDIDATE_FILE_LIST="$candidate_list"
candidate_count="$(python3 - "$candidate_list" <<'PY'
import sys

print(sum(1 for path in open(sys.argv[1]) for line in open(path.strip()) if line.strip()))
PY
)"
echo "immutable_checkpoint_candidates=$candidate_count"

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
test -s "$subspace_pointer"
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$subspace_pointer")}"

qwen_base_revision="b968826d9c46dd6066d109eabc6255188de91218"
qwen_mntp_revision="c84774c1366ea79f033504994bd254155d956d57"
qwen_simcse_revision="86b17660b1b1a8efe0b822e90c995f1ac7294645"
mistral_base_revision="63a8b081895390a26e140280378bc85ec8bce07a"
mistral_mntp_revision="e76f9757923897a0c5204b3075f1062f484d033b"
mistral_simcse_revision="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8"
ministral_revision="f6fae9795746f63c9be8344932f01275f3c63734"

export QWEN_MAP_ROOT="${QWEN_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/qwen3-8b-mntp-unsup-simcse-three-judge-gpu-v2}"
export QWEN_LLM2VEC_BASE="${QWEN_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/qwen/Qwen3-8B/$qwen_base_revision}"
export QWEN_LLM2VEC_MNTP="${QWEN_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/$qwen_mntp_revision}"
export QWEN_LLM2VEC_SIMCSE="${QWEN_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/$qwen_simcse_revision}"
export MISTRAL_MAP_ROOT="${MISTRAL_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/mistral7b-mntp-unsup-simcse-three-judge-gpu-v3}"
export MISTRAL_LLM2VEC_BASE="${MISTRAL_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/mistralai/Mistral-7B-Instruct-v0.2/$mistral_base_revision}"
export MISTRAL_LLM2VEC_MNTP="${MISTRAL_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp/$mistral_mntp_revision}"
export MISTRAL_LLM2VEC_SIMCSE="${MISTRAL_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/$mistral_simcse_revision}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"
export READINESS_VALIDATOR_ID="${READINESS_VALIDATOR_ID:-ministral3-8b-independent-search-validator}"
export READINESS_VALIDATOR_MODEL="${READINESS_VALIDATOR_MODEL:-$GEODML_MODELS_ROOT/mistral/Ministral-3-8B-Instruct-2512-BF16/$ministral_revision}"
export READINESS_EMBEDDING_BATCH_SIZE="${READINESS_EMBEDDING_BATCH_SIZE:-8}"

required_files=(
    "$QWEN_MAP_ROOT/readiness_embedding_map.json"
    "$QWEN_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl"
    "$QWEN_LLM2VEC_BASE/config.json"
    "$QWEN_LLM2VEC_MNTP/adapter_config.json"
    "$QWEN_LLM2VEC_SIMCSE/adapter_config.json"
    "$MISTRAL_MAP_ROOT/readiness_embedding_map.json"
    "$MISTRAL_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl"
    "$MISTRAL_LLM2VEC_BASE/config.json"
    "$MISTRAL_LLM2VEC_MNTP/adapter_config.json"
    "$MISTRAL_LLM2VEC_SIMCSE/adapter_config.json"
    "$READINESS_BATTERY_ROOT/battery_manifest.json"
    "$READINESS_BATTERY_ROOT/readiness_robustness_battery.json"
    "$READINESS_VALIDATOR_MODEL/config.json"
)
for path in "${required_files[@]}"; do
    test -s "$path" || { echo "missing required model/map artifact: $path" >&2; exit 2; }
done

export READINESS_VALIDATION_CACHE="${READINESS_VALIDATION_CACHE:-$READINESS_SOURCE_PIPELINE_ROOT/cache/$READINESS_VALIDATOR_ID}"
mkdir -p "$READINESS_VALIDATION_CACHE"

activate_control_runtime

python - "$READINESS_AXIS1_AUDIT_ROOT" "$candidate_count" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
path = root / "run_manifest.json"
value = {
    "format_version": "readiness-axis-1-checkpoint-audit-run-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "allocated_gpu_count": int(os.environ["READINESS_ALLOCATED_GPU_COUNT"]),
    "plan_root": str(pathlib.Path(os.environ["READINESS_30K_PLAN_ROOT"]).resolve()),
    "source_pipeline_root": str(pathlib.Path(os.environ["READINESS_SOURCE_PIPELINE_ROOT"]).resolve()),
    "candidate_count": int(sys.argv[2]),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

worker="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_pipeline_stage.sh"
qwen_projection="$READINESS_AXIS1_AUDIT_ROOT/projections/qwen"
mistral_projection="$READINESS_AXIS1_AUDIT_ROOT/projections/mistral"
attempt="$SLURM_JOB_ID-${BASHPID:-$$}-$(date -u +%Y%m%dT%H%M%SZ)"
qwen_temporary="$READINESS_AXIS1_AUDIT_ROOT/projections/.qwen-attempt-$attempt"
mistral_temporary="$READINESS_AXIS1_AUDIT_ROOT/projections/.mistral-attempt-$attempt"
validation_shard_0="$READINESS_AXIS1_AUDIT_ROOT/validation-shard-0.jsonl"
validation_shard_1="$READINESS_AXIS1_AUDIT_ROOT/validation-shard-1.jsonl"
active_srun_pids=()

READINESS_VALIDATION_OUTPUT="$validation_shard_0" \
READINESS_VALIDATION_SHARD_COUNT=2 \
READINESS_VALIDATION_SHARD_INDEX=0 \
srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
    "$worker" validate > "$READINESS_AXIS1_AUDIT_ROOT/logs/validate-shard-0.log" 2>&1 &
validator_pid_0="$!"
active_srun_pids+=("$validator_pid_0")

READINESS_VALIDATION_OUTPUT="$validation_shard_1" \
READINESS_VALIDATION_SHARD_COUNT=2 \
READINESS_VALIDATION_SHARD_INDEX=1 \
srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
    "$worker" validate > "$READINESS_AXIS1_AUDIT_ROOT/logs/validate-shard-1.log" 2>&1 &
validator_pid_1="$!"
active_srun_pids+=("$validator_pid_1")

qwen_launched=0
if ! artifact_count_matches "$qwen_projection/projection_manifest.json" "$candidate_count"; then
    [[ ! -e "$qwen_projection" ]] || {
        echo "partial Qwen projection exists; use a fresh audit root" >&2
        stop_active_steps
        exit 2
    }
    qwen_launched=1
    QWEN_PROJECTION_ROOT="$qwen_temporary" \
    srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        "$worker" project-qwen > "$READINESS_AXIS1_AUDIT_ROOT/logs/project-qwen.log" 2>&1 &
    qwen_pid="$!"
    active_srun_pids+=("$qwen_pid")
fi

mistral_launched=0
if ! artifact_count_matches "$mistral_projection/projection_manifest.json" "$candidate_count"; then
    [[ ! -e "$mistral_projection" ]] || {
        echo "partial Mistral projection exists; use a fresh audit root" >&2
        stop_active_steps
        exit 2
    }
    mistral_launched=1
    MISTRAL_PROJECTION_ROOT="$mistral_temporary" \
    srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
        "$worker" project-mistral > "$READINESS_AXIS1_AUDIT_ROOT/logs/project-mistral.log" 2>&1 &
    mistral_pid="$!"
    active_srun_pids+=("$mistral_pid")
fi

projection_failure=0
if [[ "$qwen_launched" -eq 1 ]] && ! wait "$qwen_pid"; then
    echo "Qwen projection failed; inspect logs/project-qwen.log" >&2
    projection_failure=1
fi
if [[ "$mistral_launched" -eq 1 ]] && ! wait "$mistral_pid"; then
    echo "Mistral projection failed; inspect logs/project-mistral.log" >&2
    projection_failure=1
fi
[[ "$projection_failure" -eq 0 ]] || {
    stop_active_steps
    exit 2
}
if [[ "$qwen_launched" -eq 1 ]]; then
    mv "$qwen_temporary" "$qwen_projection"
fi
if [[ "$mistral_launched" -eq 1 ]]; then
    mv "$mistral_temporary" "$mistral_projection"
fi
active_srun_pids=("$validator_pid_0" "$validator_pid_1")

comparison="$READINESS_AXIS1_AUDIT_ROOT/comparison"
if ! artifact_count_matches "$comparison/comparison_manifest.json" "$candidate_count"; then
    [[ ! -e "$comparison" ]] || {
        echo "partial comparison directory exists: $comparison" >&2
        stop_active_steps
        exit 2
    }
    python analysis/scripts/build_readiness_prompt_population.py compare-projections \
        --reference-projections "$qwen_projection" \
        --candidate-projections "$mistral_projection" \
        --robustness-battery "$READINESS_BATTERY_ROOT" \
        --output-dir "$comparison"
fi

raw_audit="$READINESS_AXIS1_AUDIT_ROOT/raw-axis1-continuity"
if [[ ! -s "$raw_audit/axis_1_continuity_audit.json" ]]; then
    [[ ! -e "$raw_audit" ]] || {
        echo "partial raw audit directory exists: $raw_audit" >&2
        stop_active_steps
        exit 2
    }
    python analysis/scripts/audit_readiness_axis1_continuity.py \
        --plan-dir "$READINESS_30K_PLAN_ROOT" \
        --candidates "${candidate_files[@]}" \
        --aligned-projections "$comparison/aligned_question_projections.jsonl" \
        --primary-tolerance-steps 0.5 \
        --tolerance-steps 0.5 1 2 3 \
        --output-dir "$raw_audit"
fi
printf '%s\n' "raw-axis1-audit-ready" > "$READINESS_AXIS1_AUDIT_ROOT/status.txt"
echo "===== RAW HALF-STEP AXIS-1 AUDIT READY ====="
cat "$raw_audit/axis_1_continuity_report.md"
echo "validator cache continues on the remaining two GPUs"

validation_failure=0
if ! wait "$validator_pid_0"; then
    validation_failure=1
fi
if ! wait "$validator_pid_1"; then
    validation_failure=1
fi
active_srun_pids=()
if [[ "$validation_failure" -ne 0 ]]; then
    echo "validator slice ended before both shards completed; cached judgments are preserved" >&2
    exit 10
fi

validated_audit="$READINESS_AXIS1_AUDIT_ROOT/validated-axis1-continuity"
if [[ ! -s "$validated_audit/axis_1_continuity_audit.json" ]]; then
    python analysis/scripts/audit_readiness_axis1_continuity.py \
        --plan-dir "$READINESS_30K_PLAN_ROOT" \
        --candidates "${candidate_files[@]}" \
        --aligned-projections "$comparison/aligned_question_projections.jsonl" \
        --validations "$validation_shard_0" "$validation_shard_1" \
        --primary-tolerance-steps 0.5 \
        --tolerance-steps 0.5 1 2 3 \
        --output-dir "$validated_audit"
fi
printf '%s\n' "validated-axis1-audit-ready" > "$READINESS_AXIS1_AUDIT_ROOT/status.txt"
echo "===== VALIDATED HALF-STEP AXIS-1 AUDIT READY ====="
cat "$validated_audit/axis_1_continuity_report.md"
