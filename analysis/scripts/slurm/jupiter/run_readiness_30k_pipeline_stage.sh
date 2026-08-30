#!/bin/bash -l
# Internal isolated worker for run_readiness_30k_end_to_end.sh.

set -euo pipefail
umask 077

stage="${1:?stage is required}"
: "${GEODML_REPOSITORY:?GEODML_REPOSITORY is required}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"

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
    # CUDA_VISIBLE_DEVICES is Slurm's per-step GPU isolation contract. Removing
    # it would expose every allocated GPU to each nominally one-GPU worker.
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

activate_runtime() {
    local runtime="$1" module_pythonpath python_prefix
    [[ -x "$runtime/bin/python" ]] || {
        echo "missing stage runtime: $runtime" >&2
        exit 2
    }
    module_pythonpath="${PYTHONPATH:-}"
    python_prefix="$(python3 -c 'import sys; print(sys.base_prefix)')"
    export LD_LIBRARY_PATH="$python_prefix/lib:${LD_LIBRARY_PATH:-}"
    source "$runtime/bin/activate"
    export PYTHONPATH="$runtime/lib/python3.13/site-packages${module_pythonpath:+:$module_pythonpath}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
}

read_candidates() {
    : "${READINESS_CANDIDATE_FILE_LIST:?READINESS_CANDIDATE_FILE_LIST is required}"
    mapfile -t candidate_files < "$READINESS_CANDIDATE_FILE_LIST"
    [[ "${#candidate_files[@]}" -gt 0 ]] || {
        echo "candidate file list is empty" >&2
        exit 2
    }
}

clear_runtime
load_stack
cd "$GEODML_REPOSITORY"
[[ "$(git rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]

case "$stage" in
    generate)
        activate_runtime "${GEODML_GENERATOR_VENV:?GEODML_GENERATOR_VENV is required}"
        python analysis/scripts/build_readiness_prompt_population.py generate \
            --tasks "${READINESS_GENERATION_TASKS:?READINESS_GENERATION_TASKS is required}" \
            --generator-id "${READINESS_STAGE_GENERATOR_ID:?READINESS_STAGE_GENERATOR_ID is required}" \
            --backend local \
            --model "${READINESS_STAGE_GENERATOR_MODEL:?READINESS_STAGE_GENERATOR_MODEL is required}" \
            --precision full \
            --cache-dir "${READINESS_STAGE_CACHE:?READINESS_STAGE_CACHE is required}" \
            --output "${READINESS_STAGE_OUTPUT:?READINESS_STAGE_OUTPUT is required}" \
            --temperature "${READINESS_GENERATION_TEMPERATURE:-0.9}" \
            --max-new-tokens "${READINESS_GENERATION_MAX_NEW_TOKENS:-180}" \
            --maximum-attempts "${READINESS_GENERATION_MAXIMUM_ATTEMPTS:-5}" \
            --text-contract "${READINESS_TEXT_CONTRACT:-question-v1}" \
            --shard-count "${READINESS_GENERATION_SHARD_COUNT:-2}" \
            --shard-index "${READINESS_GENERATION_SHARD_INDEX:?READINESS_GENERATION_SHARD_INDEX is required}" \
            --maximum-runtime-seconds "${READINESS_GENERATION_SECONDS:?READINESS_GENERATION_SECONDS is required}" \
            --allow-failed-tasks \
            --resume
        ;;
    validate)
        read_candidates
        activate_runtime "${GEODML_GENERATOR_VENV:?GEODML_GENERATOR_VENV is required}"
        validation_reuse_args=()
        if [[ -n "${READINESS_BASE_VALIDATION_OUTPUT:-}" ]]; then
            validation_reuse_args+=(--base-validation "$READINESS_BASE_VALIDATION_OUTPUT")
        fi
        python analysis/scripts/build_readiness_prompt_population.py validate-candidates \
            --candidates "${candidate_files[@]}" \
            --judge-id "${READINESS_VALIDATOR_ID:?READINESS_VALIDATOR_ID is required}" \
            --model "${READINESS_VALIDATOR_MODEL:?READINESS_VALIDATOR_MODEL is required}" \
            --backend local \
            --precision full \
            --cache-dir "${READINESS_VALIDATION_CACHE:?READINESS_VALIDATION_CACHE is required}" \
            --output "${READINESS_VALIDATION_OUTPUT:?READINESS_VALIDATION_OUTPUT is required}" \
            --maximum-attempts "${READINESS_VALIDATION_MAXIMUM_ATTEMPTS:-3}" \
            --acceptance-contract "${READINESS_ACCEPTANCE_CONTRACT:-question-v1}" \
            --inference-batch-size "${READINESS_VALIDATION_BATCH_SIZE:-8}" \
            --shard-count "${READINESS_VALIDATION_SHARD_COUNT:-1}" \
            --shard-index "${READINESS_VALIDATION_SHARD_INDEX:-0}" \
            --shard-salt "${READINESS_VALIDATION_SHARD_SALT:-}" \
            "${validation_reuse_args[@]}" \
            --resume
        ;;
    project-qwen)
        read_candidates
        activate_runtime "${QWEN_LLM2VEC_VENV:?QWEN_LLM2VEC_VENV is required}"
        projection_reuse_args=()
        if [[ -n "${READINESS_BASE_PROJECTION_ROOT:-}" ]]; then
            projection_reuse_args+=(--base-projections "$READINESS_BASE_PROJECTION_ROOT")
        fi
        python analysis/scripts/build_readiness_prompt_population.py project-candidates \
            --candidates "${candidate_files[@]}" \
            --map "${QWEN_MAP_ROOT:?QWEN_MAP_ROOT is required}/readiness_embedding_map.json" \
            --reference-coordinates "$QWEN_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
            --embedding-model "${QWEN_LLM2VEC_BASE:?QWEN_LLM2VEC_BASE is required}" \
            --mntp-model "${QWEN_LLM2VEC_MNTP:?QWEN_LLM2VEC_MNTP is required}" \
            --peft-model "${QWEN_LLM2VEC_SIMCSE:?QWEN_LLM2VEC_SIMCSE is required}" \
            --embedding-batch-size "${READINESS_EMBEDDING_BATCH_SIZE:-8}" \
            --embedding-max-length "${READINESS_EMBEDDING_MAX_LENGTH:-512}" \
            --attention-implementation "${READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION:-eager}" \
            "${projection_reuse_args[@]}" \
            --output-dir "${QWEN_PROJECTION_ROOT:?QWEN_PROJECTION_ROOT is required}"
        ;;
    project-mistral)
        read_candidates
        activate_runtime "${MISTRAL_LLM2VEC_VENV:?MISTRAL_LLM2VEC_VENV is required}"
        projection_reuse_args=()
        if [[ -n "${READINESS_BASE_PROJECTION_ROOT:-}" ]]; then
            projection_reuse_args+=(--base-projections "$READINESS_BASE_PROJECTION_ROOT")
        fi
        python analysis/scripts/build_readiness_prompt_population.py project-candidates \
            --candidates "${candidate_files[@]}" \
            --map "${MISTRAL_MAP_ROOT:?MISTRAL_MAP_ROOT is required}/readiness_embedding_map.json" \
            --reference-coordinates "$MISTRAL_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
            --embedding-model "${MISTRAL_LLM2VEC_BASE:?MISTRAL_LLM2VEC_BASE is required}" \
            --mntp-model "${MISTRAL_LLM2VEC_MNTP:?MISTRAL_LLM2VEC_MNTP is required}" \
            --peft-model "${MISTRAL_LLM2VEC_SIMCSE:?MISTRAL_LLM2VEC_SIMCSE is required}" \
            --embedding-batch-size "${READINESS_EMBEDDING_BATCH_SIZE:-8}" \
            --embedding-max-length "${READINESS_EMBEDDING_MAX_LENGTH:-512}" \
            --attention-implementation "${READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION:-eager}" \
            "${projection_reuse_args[@]}" \
            --output-dir "${MISTRAL_PROJECTION_ROOT:?MISTRAL_PROJECTION_ROOT is required}"
        ;;
    *)
        echo "unknown pipeline stage: $stage" >&2
        exit 2
        ;;
esac
