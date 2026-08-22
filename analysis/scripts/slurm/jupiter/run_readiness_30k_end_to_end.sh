#!/bin/bash -l
# Generate, independently validate, dual-LLM2Vec project, and strictly select
# readiness questions. This runner uses an existing allocation and is resumable.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing JUPITER Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved allocation wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting the allocation}"
export GEODML_EXPECTED_COMMIT READINESS_APPROVED_WALLTIME READINESS_ALLOCATION_ESTIMATE SLURM_JOB_ID

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
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CUDA_VISIBLE_DEVICES
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

clear_runtime
load_stack

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_RUNS_ROOT="${GEODML_RUNS_ROOT:-$GEODML_PROJECT_ROOT/runs}"
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
[[ "$allocated_gpu_count" -ge 4 ]] || {
    echo "the end-to-end loop requires four allocated GPUs; found $allocated_gpu_count" >&2
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
    echo "end-to-end construction requires a clean exact-commit checkout" >&2
    exit 2
}

plan_pointer="${READINESS_30K_PLAN_POINTER:-$GEODML_PROJECT_ROOT/geodml-readiness-30k-v2-plan-latest.txt}"
test -s "$plan_pointer"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$(<"$plan_pointer")}"
test -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json"
test -s "$READINESS_30K_PLAN_ROOT/generation_tasks_round_00.jsonl"

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
test -s "$subspace_pointer"
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$subspace_pointer")}"

qwen32_revision="9216db5781bf21249d130ec9da846c4624c16137"
gemma4_revision="842da3794eaa0b77d5f08bae87a17459d91ff475"
ministral_revision="f6fae9795746f63c9be8344932f01275f3c63734"
export READINESS_GENERATOR_A_ID="${READINESS_GENERATOR_A_ID:-qwen3-32b}"
export READINESS_GENERATOR_A_MODEL="${READINESS_GENERATOR_A_MODEL:-$GEODML_MODELS_ROOT/qwen/Qwen3-32B/$qwen32_revision}"
export READINESS_GENERATOR_B_ID="${READINESS_GENERATOR_B_ID:-gemma4-31b}"
export READINESS_GENERATOR_B_MODEL="${READINESS_GENERATOR_B_MODEL:-$GEODML_MODELS_ROOT/gemma/gemma-4-31B-it/$gemma4_revision}"
export READINESS_VALIDATOR_ID="${READINESS_VALIDATOR_ID:-ministral3-8b-independent-search-validator}"
export READINESS_VALIDATOR_MODEL="${READINESS_VALIDATOR_MODEL:-$GEODML_MODELS_ROOT/mistral/Ministral-3-8B-Instruct-2512-BF16/$ministral_revision}"

python3 - "$READINESS_30K_PLAN_ROOT/plan_manifest.json" "$READINESS_GENERATOR_A_ID" "$READINESS_GENERATOR_B_ID" <<'PY'
import json
import sys

planned = set(json.load(open(sys.argv[1]))["generator_ids"])
configured = {sys.argv[2], sys.argv[3]}
if planned != configured:
    raise SystemExit(
        f"configured generator IDs differ from the frozen plan: "
        f"planned={sorted(planned)} configured={sorted(configured)}"
    )
PY

qwen_base_revision="b968826d9c46dd6066d109eabc6255188de91218"
qwen_mntp_revision="c84774c1366ea79f033504994bd254155d956d57"
qwen_simcse_revision="86b17660b1b1a8efe0b822e90c995f1ac7294645"
mistral_base_revision="63a8b081895390a26e140280378bc85ec8bce07a"
mistral_mntp_revision="e76f9757923897a0c5204b3075f1062f484d033b"
mistral_simcse_revision="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8"

export QWEN_MAP_ROOT="${QWEN_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/qwen3-8b-mntp-unsup-simcse-three-judge-gpu-v2}"
export QWEN_LLM2VEC_BASE="${QWEN_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/qwen/Qwen3-8B/$qwen_base_revision}"
export QWEN_LLM2VEC_MNTP="${QWEN_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/$qwen_mntp_revision}"
export QWEN_LLM2VEC_SIMCSE="${QWEN_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/$qwen_simcse_revision}"
export MISTRAL_MAP_ROOT="${MISTRAL_MAP_ROOT:-$READINESS_SUBSPACE_ROOT/maps/mistral7b-mntp-unsup-simcse-three-judge-gpu-v3}"
export MISTRAL_LLM2VEC_BASE="${MISTRAL_LLM2VEC_BASE:-$GEODML_MODELS_ROOT/mistralai/Mistral-7B-Instruct-v0.2/$mistral_base_revision}"
export MISTRAL_LLM2VEC_MNTP="${MISTRAL_LLM2VEC_MNTP:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp/$mistral_mntp_revision}"
export MISTRAL_LLM2VEC_SIMCSE="${MISTRAL_LLM2VEC_SIMCSE:-$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/$mistral_simcse_revision}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"

required_files=(
    "$READINESS_GENERATOR_A_MODEL/config.json"
    "$READINESS_GENERATOR_B_MODEL/config.json"
    "$READINESS_VALIDATOR_MODEL/config.json"
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
)
for path in "${required_files[@]}"; do
    test -s "$path" || { echo "missing required model/map artifact: $path" >&2; exit 2; }
done

if [[ "$READINESS_VALIDATOR_MODEL" == "$READINESS_GENERATOR_A_MODEL" || "$READINESS_VALIDATOR_MODEL" == "$READINESS_GENERATOR_B_MODEL" ]]; then
    echo "the independent validator must differ from both proposal generators" >&2
    exit 2
fi

pointer="${READINESS_30K_PIPELINE_POINTER:-$GEODML_PROJECT_ROOT/geodml-readiness-30k-end-to-end-latest.txt}"
source_pilot="${READINESS_SOURCE_PILOT_ROOT:-}"
export READINESS_SOURCE_PILOT_ROOT="$source_pilot"
if [[ -n "${READINESS_30K_PIPELINE_ROOT:-}" ]]; then
    pipeline_root="$READINESS_30K_PIPELINE_ROOT"
elif [[ -n "$source_pilot" ]]; then
    pipeline_root="$source_pilot/strict-dual-view-${GEODML_EXPECTED_COMMIT:0:8}"
elif [[ -s "$pointer" ]]; then
    pipeline_root="$(<"$pointer")"
else
    pipeline_root="$GEODML_RUNS_ROOT/readiness-30k-end-to-end/$(date -u +%Y%m%dT%H%M%SZ)-${GEODML_EXPECTED_COMMIT:0:8}"
fi
export READINESS_30K_PIPELINE_ROOT="$pipeline_root"
mkdir -p "$pipeline_root" "$pipeline_root/logs" "$pipeline_root/cache"
printf '%s\n' "$pipeline_root" > "$pointer"

activate_control_runtime

python - "$pipeline_root" "$READINESS_30K_PLAN_ROOT" "$source_pilot" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
plan = pathlib.Path(sys.argv[2])
source = sys.argv[3] or None
manifest_path = root / "pipeline_manifest.json"
identity = {
    "format_version": "readiness-30k-end-to-end-v1",
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "plan_manifest_sha256": hashlib.sha256((plan / "plan_manifest.json").read_bytes()).hexdigest(),
    "source_pilot_root": source,
    "generator_ids": [os.environ["READINESS_GENERATOR_A_ID"], os.environ["READINESS_GENERATOR_B_ID"]],
    "generator_models": [os.environ["READINESS_GENERATOR_A_MODEL"], os.environ["READINESS_GENERATOR_B_MODEL"]],
    "validator_id": os.environ["READINESS_VALIDATOR_ID"],
    "validator_model": os.environ["READINESS_VALIDATOR_MODEL"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "allocated_gpu_count": int(os.environ["READINESS_ALLOCATED_GPU_COUNT"]),
}
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    stable_keys = set(identity) - {
        "approved_walltime",
        "allocation_estimate",
        "slurm_job_id",
        "allocated_gpu_count",
    }
    if any(existing.get(key) != identity[key] for key in stable_keys):
        raise SystemExit("pipeline root identity differs from this invocation")
    existing.setdefault("allocation_slices", []).append({
        "approved_walltime": identity["approved_walltime"],
        "allocation_estimate": identity["allocation_estimate"],
        "slurm_job_id": identity["slurm_job_id"],
        "allocated_gpu_count": identity["allocated_gpu_count"],
    })
    value = existing
else:
    value = dict(identity)
    value["allocation_slices"] = [{
        "approved_walltime": identity["approved_walltime"],
        "allocation_estimate": identity["allocation_estimate"],
        "slurm_job_id": identity["slurm_job_id"],
        "allocated_gpu_count": identity["allocated_gpu_count"],
    }]
temporary = manifest_path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(manifest_path)
PY

worker="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_pipeline_stage.sh"
max_rounds="${READINESS_MAX_REFINEMENT_ROUNDS:-2}"
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
candidate_files=()
previous_selection=""
pipeline_status="refine"

generation_terminal() {
    local manifest
    for manifest in "$@"; do
        [[ -s "$manifest" ]] || return 1
        python -c 'import json,sys; assert json.load(open(sys.argv[1]))["slice_terminal"]' "$manifest" || return 1
    done
}

artifact_count_matches() {
    local manifest="$1" expected="$2"
    [[ -s "$manifest" ]] || return 1
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["candidate_count"] == int(sys.argv[2])' "$manifest" "$expected"
}

validation_shard_complete() {
    local manifest="$1" expected_total="$2" expected_count="$3" expected_index="$4"
    [[ -s "$manifest" ]] || return 1
    python -c '
import json
import sys
row = json.load(open(sys.argv[1]))
assert row["total_candidate_count"] == int(sys.argv[2])
assert row["shard_count"] == int(sys.argv[3])
assert row["shard_index"] == int(sys.argv[4])
assert row["reviewed_count"] == row["candidate_count"]
' "$manifest" "$expected_total" "$expected_count" "$expected_index"
}

run_generation_round() {
    local round_root="$1" tasks="$2"
    mkdir -p "$round_root/candidates" "$round_root/cache" "$round_root/logs"
    local manifests=() pids=() generator_id generator_model task_count shard_count shard output cache log
    for generator_id in "$READINESS_GENERATOR_A_ID" "$READINESS_GENERATOR_B_ID"; do
        if [[ "$generator_id" == "$READINESS_GENERATOR_A_ID" ]]; then
            generator_model="$READINESS_GENERATOR_A_MODEL"
        else
            generator_model="$READINESS_GENERATOR_B_MODEL"
        fi
        task_count="$(python -c 'import json,sys; print(sum(json.loads(x)["generator_id"] == sys.argv[2] for x in open(sys.argv[1]) if x.strip()))' "$tasks" "$generator_id")"
        [[ "$task_count" -gt 0 ]] || continue
        shard_count=2
        [[ "$task_count" -ge 2 ]] || shard_count=1
        for ((shard=0; shard<shard_count; shard++)); do
            output="$round_root/candidates/$generator_id-shard-$shard.jsonl"
            cache="$round_root/cache/$generator_id-shard-$shard"
            log="$round_root/logs/$generator_id-shard-$shard.log"
            mkdir -p "$cache"
            manifests+=("$output.manifest.json")
            if [[ -s "$output.manifest.json" ]] && python -c 'import json,sys; assert json.load(open(sys.argv[1]))["slice_terminal"]' "$output.manifest.json"; then
                continue
            fi
            READINESS_GENERATION_TASKS="$tasks" \
            READINESS_STAGE_GENERATOR_ID="$generator_id" \
            READINESS_STAGE_GENERATOR_MODEL="$generator_model" \
            READINESS_STAGE_CACHE="$cache" \
            READINESS_STAGE_OUTPUT="$output" \
            READINESS_GENERATION_SHARD_COUNT="$shard_count" \
            READINESS_GENERATION_SHARD_INDEX="$shard" \
            srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
                "$worker" generate > "$log" 2>&1 &
            pids+=("$!")
        done
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" -eq 0 ]] || { echo "generation worker failure; inspect $round_root/logs" >&2; return 2; }
    if ! generation_terminal "${manifests[@]}"; then
        echo "GENERATION CHECKPOINTED: rerun this same script in the next approved allocation"
        return 10
    fi
    mapfile -t round_candidates < <(find "$round_root/candidates" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.failures.jsonl' | sort)
    candidate_files+=("${round_candidates[@]}")
    return 0
}

for ((round_index=0; round_index<=max_rounds; round_index++)); do
    printf -v round_name 'round-%02d' "$round_index"
    round_root="$pipeline_root/$round_name"
    mkdir -p "$round_root"

    if [[ "$round_index" -eq 0 && -n "$source_pilot" ]]; then
        mapfile -t source_candidates < <(find "$source_pilot/candidates" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.failures.jsonl' | sort)
        [[ "${#source_candidates[@]}" -gt 0 ]] || { echo "source pilot has no candidates" >&2; exit 2; }
        candidate_files+=("${source_candidates[@]}")
    else
        if [[ "$round_index" -eq 0 ]]; then
            tasks="$READINESS_30K_PLAN_ROOT/generation_tasks_round_00.jsonl"
        else
            tasks="$previous_selection/generation_tasks_round_$(printf '%02d' "$round_index").jsonl"
        fi
        test -e "$tasks"
        if [[ ! -s "$tasks" ]]; then
            pipeline_status="pass"
            break
        fi
        set +e
        run_generation_round "$round_root/generation" "$tasks"
        generation_exit=$?
        set -e
        if [[ "$generation_exit" -eq 10 ]]; then
            exit 0
        fi
        [[ "$generation_exit" -eq 0 ]] || exit "$generation_exit"
    fi

    candidate_list="$round_root/candidate-files.txt"
    printf '%s\n' "${candidate_files[@]}" > "$candidate_list"
    export READINESS_CANDIDATE_FILE_LIST="$candidate_list"
    candidate_count="$(python -c 'import json,sys; print(sum(1 for p in open(sys.argv[1]) for x in open(p.strip()) if x.strip()))' "$candidate_list")"
    echo "===== $round_name CANDIDATES: $candidate_count ====="

    raw_diversity="$round_root/raw-diversity"
    if [[ ! -s "$raw_diversity/question_diversity_audit.json" ]]; then
        set +e
        python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
            --questions "${candidate_files[@]}" \
            --output-dir "$raw_diversity"
        raw_diversity_exit=$?
        set -e
        printf '%s\n' "$raw_diversity_exit" > "$round_root/raw-diversity-exit-code.txt"
    fi

    export READINESS_VALIDATION_OUTPUT="$round_root/validation.jsonl"
    export READINESS_VALIDATION_CACHE="$pipeline_root/cache/$READINESS_VALIDATOR_ID"
    export QWEN_PROJECTION_ROOT="$round_root/projections/qwen"
    export MISTRAL_PROJECTION_ROOT="$round_root/projections/mistral"
    mkdir -p "$READINESS_VALIDATION_CACHE" "$round_root/logs"

    stage_pids=() stage_names=()
    validation_shard_count=2
    validation_shard_files=()
    for ((validation_shard_index=0; validation_shard_index<validation_shard_count; validation_shard_index++)); do
        validation_shard_output="$round_root/validation-shard-$validation_shard_index.jsonl"
        validation_shard_files+=("$validation_shard_output")
        if artifact_count_matches "$READINESS_VALIDATION_OUTPUT.manifest.json" "$candidate_count" || \
            validation_shard_complete "$validation_shard_output.manifest.json" "$candidate_count" \
                "$validation_shard_count" "$validation_shard_index"; then
            continue
        fi
        READINESS_VALIDATION_OUTPUT="$validation_shard_output" \
        READINESS_VALIDATION_SHARD_COUNT="$validation_shard_count" \
        READINESS_VALIDATION_SHARD_INDEX="$validation_shard_index" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" validate > "$round_root/logs/validate-shard-$validation_shard_index.log" 2>&1 &
        stage_pids+=("$!"); stage_names+=("validate-shard-$validation_shard_index")
    done
    qwen_projection_temporary="$round_root/projections/.qwen-job-$SLURM_JOB_ID"
    mistral_projection_temporary="$round_root/projections/.mistral-job-$SLURM_JOB_ID"
    qwen_projection_launched=0
    mistral_projection_launched=0
    if ! artifact_count_matches "$QWEN_PROJECTION_ROOT/projection_manifest.json" "$candidate_count"; then
        [[ ! -e "$QWEN_PROJECTION_ROOT" ]] || { echo "partial Qwen projection; choose a fresh pipeline root" >&2; exit 2; }
        [[ ! -e "$qwen_projection_temporary" ]] || { echo "stale current-job Qwen projection: $qwen_projection_temporary" >&2; exit 2; }
        qwen_projection_launched=1
        QWEN_PROJECTION_ROOT="$qwen_projection_temporary" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" project-qwen > "$round_root/logs/project-qwen.log" 2>&1 &
        stage_pids+=("$!"); stage_names+=("project-qwen")
    fi
    if ! artifact_count_matches "$MISTRAL_PROJECTION_ROOT/projection_manifest.json" "$candidate_count"; then
        [[ ! -e "$MISTRAL_PROJECTION_ROOT" ]] || { echo "partial Mistral projection; choose a fresh pipeline root" >&2; exit 2; }
        [[ ! -e "$mistral_projection_temporary" ]] || { echo "stale current-job Mistral projection: $mistral_projection_temporary" >&2; exit 2; }
        mistral_projection_launched=1
        MISTRAL_PROJECTION_ROOT="$mistral_projection_temporary" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" project-mistral > "$round_root/logs/project-mistral.log" 2>&1 &
        stage_pids+=("$!"); stage_names+=("project-mistral")
    fi

    stage_failure=0
    for index in "${!stage_pids[@]}"; do
        if ! wait "${stage_pids[$index]}"; then
            echo "stage failed: ${stage_names[$index]}; inspect $round_root/logs" >&2
            stage_failure=1
        fi
    done
    [[ "$stage_failure" -eq 0 ]] || exit 2
    if [[ "$qwen_projection_launched" -eq 1 ]]; then
        mv "$qwen_projection_temporary" "$QWEN_PROJECTION_ROOT"
    fi
    if [[ "$mistral_projection_launched" -eq 1 ]]; then
        mv "$mistral_projection_temporary" "$MISTRAL_PROJECTION_ROOT"
    fi

    if ! artifact_count_matches "$READINESS_VALIDATION_OUTPUT.manifest.json" "$candidate_count"; then
        python - "$READINESS_VALIDATION_OUTPUT" "$candidate_list" "${validation_shard_files[@]}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
candidate_list = pathlib.Path(sys.argv[2])
shards = [pathlib.Path(value) for value in sys.argv[3:]]
candidate_paths = [
    pathlib.Path(value)
    for value in candidate_list.read_text().splitlines()
    if value.strip()
]
candidate_ids = {
    json.loads(line)["candidate_id"]
    for path in candidate_paths
    for line in path.read_text().splitlines()
    if line.strip()
}
rows = {}
for shard in shards:
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidate_id = row["candidate_id"]
        if candidate_id in rows:
            raise SystemExit(f"duplicate validation across shards: {candidate_id}")
        rows[candidate_id] = row
if set(rows) != candidate_ids:
    raise SystemExit("validation shards do not cover the exact candidate set")
manifests = [
    json.loads(shard.with_suffix(shard.suffix + ".manifest.json").read_text())
    for shard in shards
]
if sorted(manifest["shard_index"] for manifest in manifests) != list(range(len(shards))):
    raise SystemExit("validation shard indices are incomplete")
if any(
    manifest["shard_count"] != len(shards)
    or manifest["total_candidate_count"] != len(candidate_ids)
    for manifest in manifests
):
    raise SystemExit("validation shard geometry differs from the candidate set")
stable_fields = ("judge_id", "judge_model", "judge_backend", "judge_precision")
for field in stable_fields:
    if len({manifest[field] for manifest in manifests}) != 1:
        raise SystemExit(f"validation shard {field} differs")
ordered = [rows[candidate_id] for candidate_id in sorted(rows)]
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
)
temporary.replace(output)
manifest = {
    "format_version": manifests[0]["format_version"],
    "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": manifests[0]["git_commit_sha"],
    "slurm": manifests[0]["slurm"],
    "candidate_files": manifests[0]["candidate_files"],
    "candidate_count": len(ordered),
    "reviewed_count": len(ordered),
    "accepted_count": sum(bool(row["accepted"]) for row in ordered),
    "validation_shards": [
        {
            "path": str(shard.resolve()),
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "candidate_count": shard_manifest["candidate_count"],
            "shard_count": shard_manifest["shard_count"],
            "shard_index": shard_manifest["shard_index"],
        }
        for shard, shard_manifest in zip(shards, manifests)
    ],
    **{field: manifests[0][field] for field in stable_fields},
    "acceptance_contract": manifests[0]["acceptance_contract"],
}
manifest_path = output.with_suffix(output.suffix + ".manifest.json")
manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
manifest_temporary.replace(manifest_path)
print(
    f"merged_validation_shards={len(shards)} reviewed={len(ordered)} "
    f"accepted={manifest['accepted_count']}"
)
PY
    fi

    comparison="$round_root/comparison"
    if ! artifact_count_matches "$comparison/comparison_manifest.json" "$candidate_count"; then
        [[ ! -e "$comparison" ]] || { echo "partial comparison directory: $comparison" >&2; exit 2; }
        python analysis/scripts/build_readiness_prompt_population.py compare-projections \
            --reference-projections "$QWEN_PROJECTION_ROOT" \
            --candidate-projections "$MISTRAL_PROJECTION_ROOT" \
            --robustness-battery "$READINESS_BATTERY_ROOT" \
            --output-dir "$comparison"
    fi

    selection="$round_root/strict-selection"
    previous_selection="$selection"
    if ! artifact_count_matches "$selection/run_manifest.json" "$candidate_count"; then
        [[ ! -e "$selection" ]] || { echo "partial selection directory: $selection" >&2; exit 2; }
        python analysis/scripts/build_readiness_prompt_population.py spatial-select \
            --plan-dir "$READINESS_30K_PLAN_ROOT" \
            --candidates "${candidate_files[@]}" \
            --reference-projections "$QWEN_PROJECTION_ROOT" \
            --candidate-projections "$MISTRAL_PROJECTION_ROOT" \
            --robustness-battery "$READINESS_BATTERY_ROOT" \
            --validations "$READINESS_VALIDATION_OUTPUT" \
            --generator-ids "$READINESS_GENERATOR_A_ID,$READINESS_GENERATOR_B_ID" \
            --next-round-index "$((round_index + 1))" \
            --distance-tolerance "${READINESS_DISTANCE_TOLERANCE:-0.22}" \
            --require-both-views-within-tolerance \
            --require-delexicalized-template-uniqueness \
            --disagreement-weight "${READINESS_DISAGREEMENT_WEIGHT:-0.10}" \
            --candidates-per-task "${READINESS_REFINEMENT_CANDIDATES_PER_TASK:-4}" \
            --master-seed "${READINESS_MASTER_SEED:-20260820}" \
            --output-dir "$selection"
    fi

    selected="$selection/spatially_selected_questions.jsonl"
    final_diversity="$round_root/selected-diversity"
    selected_diversity_exit=2
    if [[ -s "$selected" ]]; then
        if [[ ! -s "$final_diversity/question_diversity_audit.json" ]]; then
            set +e
            python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
                --questions "$selected" \
                --output-dir "$final_diversity"
            selected_diversity_exit=$?
            set -e
            printf '%s\n' "$selected_diversity_exit" > "$round_root/selected-diversity-exit-code.txt"
        else
            selected_diversity_exit="$(python -c 'import json,sys; print(0 if json.load(open(sys.argv[1]))["all_checks_passed"] else 2)' "$final_diversity/question_diversity_audit.json")"
        fi
    fi

    python - "$round_root" "$candidate_count" "$selected_diversity_exit" "$source_pilot" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidate_count = int(sys.argv[2])
diversity_exit = int(sys.argv[3])
source_pilot_mode = bool(sys.argv[4])
validation = json.loads((root / "validation.jsonl.manifest.json").read_text())
selection = json.loads((root / "strict-selection/run_manifest.json").read_text())
diagnostics = json.loads((root / "strict-selection/spatial_coverage_diagnostics.json").read_text())
summary = {
    "format_version": "readiness-30k-verified-round-v1",
    "candidate_count": candidate_count,
    "independently_accepted_count": validation["accepted_count"],
    "selected_count": selection["selected_count"],
    "refinement_task_count": selection["next_round_task_count"],
    "strict_dual_view_contract_enabled": selection["coordinate_acceptance_contract"]["enabled"],
    "delexicalized_template_uniqueness_enabled": selection["surface_acceptance_contract"]["enabled"],
    "selected_diversity_gate_passed": diversity_exit == 0,
    "spacing_gate_passed": diagnostics["overall_spacing_gate_passed"],
    "source_pilot_mode": source_pilot_mode,
    "verified_population_passed": (
        not source_pilot_mode
        and selection["next_round_task_count"] == 0
        and diversity_exit == 0
        and diagnostics["overall_spacing_gate_passed"]
    ),
}
(root / "verified_round_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

    next_tasks="$selection/generation_tasks_round_$(printf '%02d' "$((round_index + 1))").jsonl"
    if [[ -n "$source_pilot" ]]; then
        if [[ -s "$next_tasks" ]]; then
            pipeline_status="pilot-refine"
            echo "SOURCE PILOT REFINEMENT: missed dual-view targets will enter the next four-GPU loop."
            continue
        else
            pipeline_status="pilot-verified-subset"
        fi
        echo "SOURCE PILOT VERIFIED: results apply only to its generated subset, never the complete 30,330-target plan."
        break
    fi
    if [[ ! -s "$next_tasks" ]]; then
        if python -c 'import json,sys; assert json.load(open(sys.argv[1]))["verified_population_passed"]' \
            "$round_root/verified_round_summary.json"; then
            pipeline_status="pass"
        else
            pipeline_status="quality-gate-failed"
        fi
        break
    fi
done

printf '%s\n' "$pipeline_status" > "$pipeline_root/status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$pipeline_root/updated-at.txt"
echo "PIPELINE_ROOT=$pipeline_root"
echo "PIPELINE_STATUS=$pipeline_status"
if [[ "$pipeline_status" != "pass" ]]; then
    echo "STRICT VERIFIED POPULATION: REFINE" >&2
    exit 3
fi
echo "STRICT VERIFIED POPULATION: PASS"
