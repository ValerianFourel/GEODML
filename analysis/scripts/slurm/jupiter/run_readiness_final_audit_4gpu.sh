#!/bin/bash -l
# Run the resumable relaxed-compliance and dual-view audit inside four GPUs.
# This runner never requests or changes a Slurm allocation.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing four-GPU Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved allocation wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the runtime estimate}"
: "${READINESS_CHECKPOINT_ROOT:?Set the merged checkpoint root}"
: "${READINESS_OUTPUT_ROOT:?Set a fresh or resumable audit root}"

inherited_venv_bin="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin}"
if [[ -n "$inherited_venv_bin" ]]; then
    cleaned_path=""
    IFS=: read -r -a path_entries <<< "$PATH"
    for path_entry in "${path_entries[@]}"; do
        [[ "$path_entry" == "$inherited_venv_bin" ]] && continue
        cleaned_path="${cleaned_path:+$cleaned_path:}$path_entry"
    done
    export PATH="$cleaned_path"
fi
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
hash -r

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load git
module load PyTorch/2.9.1
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"
export QWEN_LLM2VEC_VENV="${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
export MISTRAL_LLM2VEC_VENV="${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$HOME/geodml-readiness-subspace-latest.txt")}"

qwen_base_revision="b968826d9c46dd6066d109eabc6255188de91218"
qwen_mntp_revision="c84774c1366ea79f033504994bd254155d956d57"
qwen_simcse_revision="86b17660b1b1a8efe0b822e90c995f1ac7294645"
mistral_base_revision="63a8b081895390a26e140280378bc85ec8bce07a"
mistral_mntp_revision="e76f9757923897a0c5204b3075f1062f484d033b"
mistral_simcse_revision="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8"

export QWEN_MAP_ROOT="${QWEN_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/qwen3-8b-mntp-unsup-simcse-three-judge-gpu-v2}"
export MISTRAL_MAP_ROOT="${MISTRAL_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/mistral7b-mntp-unsup-simcse-three-judge-gpu-v3}"
export QWEN_LLM2VEC_BASE="${QWEN_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/qwen/Qwen3-8B/$qwen_base_revision}"
export QWEN_LLM2VEC_MNTP="${QWEN_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/$qwen_mntp_revision}"
export QWEN_LLM2VEC_SIMCSE="${QWEN_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/$qwen_simcse_revision}"
export MISTRAL_LLM2VEC_BASE="${MISTRAL_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/mistralai/Mistral-7B-Instruct-v0.2/$mistral_base_revision}"
export MISTRAL_LLM2VEC_MNTP="${MISTRAL_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp/$mistral_mntp_revision}"
export MISTRAL_LLM2VEC_SIMCSE="${MISTRAL_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/$mistral_simcse_revision}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

[[ -x "$QWEN_LLM2VEC_VENV/bin/python" ]]
[[ -x "$MISTRAL_LLM2VEC_VENV/bin/python" ]]
cd "$GEODML_REPOSITORY"
actual_commit="$(git rev-parse HEAD)"
[[ "$actual_commit" == "$GEODML_EXPECTED_COMMIT" ]] || {
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
    echo "final audit requires a clean exact-commit checkout" >&2
    exit 2
}

required_files=(
    "$READINESS_CHECKPOINT_ROOT/strict-selection/spatially_selected_questions.jsonl"
    "$READINESS_CHECKPOINT_ROOT/merged/candidates.jsonl"
    "$READINESS_CHECKPOINT_ROOT/merged/validation.jsonl"
    "$QWEN_MAP_ROOT/readiness_embedding_map.json"
    "$QWEN_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl"
    "$MISTRAL_MAP_ROOT/readiness_embedding_map.json"
    "$MISTRAL_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl"
    "$READINESS_BATTERY_ROOT/battery_manifest.json"
    "$READINESS_BATTERY_ROOT/readiness_robustness_battery.json"
    "$QWEN_LLM2VEC_BASE/config.json"
    "$QWEN_LLM2VEC_MNTP/adapter_config.json"
    "$QWEN_LLM2VEC_SIMCSE/adapter_config.json"
    "$MISTRAL_LLM2VEC_BASE/config.json"
    "$MISTRAL_LLM2VEC_MNTP/adapter_config.json"
    "$MISTRAL_LLM2VEC_SIMCSE/adapter_config.json"
)
for path in "${required_files[@]}"; do
    [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done

mkdir -p "$READINESS_OUTPUT_ROOT"
pointer="${READINESS_FINAL_AUDIT_POINTER:-$HOME/geodml-final-audit-latest.txt}"
printf '%s\n' "$READINESS_OUTPUT_ROOT" > "$pointer.tmp"
mv "$pointer.tmp" "$pointer"

exec "$QWEN_LLM2VEC_VENV/bin/python" \
    analysis/scripts/run_readiness_final_audit_4gpu.py \
    --checkpoint-root "$READINESS_CHECKPOINT_ROOT" \
    --output-root "$READINESS_OUTPUT_ROOT" \
    --repository "$GEODML_REPOSITORY" \
    --qwen-map-root "$QWEN_MAP_ROOT" \
    --qwen-embedding-model "$QWEN_LLM2VEC_BASE" \
    --qwen-mntp-model "$QWEN_LLM2VEC_MNTP" \
    --qwen-peft-model "$QWEN_LLM2VEC_SIMCSE" \
    --mistral-map-root "$MISTRAL_MAP_ROOT" \
    --mistral-embedding-model "$MISTRAL_LLM2VEC_BASE" \
    --mistral-mntp-model "$MISTRAL_LLM2VEC_MNTP" \
    --mistral-peft-model "$MISTRAL_LLM2VEC_SIMCSE" \
    --robustness-battery "$READINESS_BATTERY_ROOT" \
    --qwen-python "$QWEN_LLM2VEC_VENV/bin/python" \
    --mistral-python "$MISTRAL_LLM2VEC_VENV/bin/python" \
    --embedding-batch-size "${READINESS_EMBEDDING_BATCH_SIZE:-8}" \
    --shard-count "${READINESS_PROJECTION_SHARD_COUNT:-8}"
