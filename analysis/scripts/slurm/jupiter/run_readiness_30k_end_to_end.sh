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

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "READINESS_END_TO_END_CONTROL_RUNTIME=PASS"

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
    echo "cannot determine allocated GPU count; set READINESS_ALLOCATED_GPU_COUNT explicitly" >&2
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
    echo "the end-to-end loop requires at least four allocated GPUs; found $allocated_gpu_count" >&2
    exit 2
}
[[ "$((allocated_gpu_count % 2))" -eq 0 ]] || {
    echo "the two-generator loop requires an even allocated GPU count; found $allocated_gpu_count" >&2
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
test -s "$READINESS_30K_PLAN_ROOT/keyword_target_grid.jsonl"
test -s "$READINESS_30K_PLAN_ROOT/target_design.json"

export READINESS_DISTANCE_TOLERANCE="${READINESS_DISTANCE_TOLERANCE:-0.017}"
export READINESS_DISAGREEMENT_WEIGHT="${READINESS_DISAGREEMENT_WEIGHT:-0.10}"
export READINESS_TEXT_CONTRACT="${READINESS_TEXT_CONTRACT:-question-v1}"
export READINESS_ACCEPTANCE_CONTRACT="${READINESS_ACCEPTANCE_CONTRACT:-question-v1}"
export READINESS_GENERATION_PROFILE="${READINESS_GENERATION_PROFILE:-balanced-v1}"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="${READINESS_REFINEMENT_CANDIDATES_PER_TASK:-4}"
export READINESS_REFINEMENT_MIN_TARGET_AXIS_1="${READINESS_REFINEMENT_MIN_TARGET_AXIS_1:-}"
export READINESS_REFINEMENT_TASK_PRIORITY="${READINESS_REFINEMENT_TASK_PRIORITY:-stable-hash}"
export READINESS_MASTER_SEED="${READINESS_MASTER_SEED:-20260820}"
export READINESS_VALIDATION_SHARD_COUNT="${READINESS_VALIDATION_SHARD_COUNT:-4}"
export READINESS_COORDINATE_ONLY_PROJECTION_REUSE="${READINESS_COORDINATE_ONLY_PROJECTION_REUSE:-0}"
[[ "$READINESS_COORDINATE_ONLY_PROJECTION_REUSE" == "0" || "$READINESS_COORDINATE_ONLY_PROJECTION_REUSE" == "1" ]] || {
    echo "READINESS_COORDINATE_ONLY_PROJECTION_REUSE must equal 0 or 1" >&2
    exit 2
}

python3 - "$READINESS_30K_PLAN_ROOT" <<'PY'
import collections
import json
import math
import os
import pathlib
import sys

plan = pathlib.Path(sys.argv[1])
manifest = json.loads((plan / "plan_manifest.json").read_text())
design = json.loads((plan / "target_design.json").read_text())
rows = [
    json.loads(line)
    for line in (plan / "keyword_target_grid.jsonl").read_text().splitlines()
    if line.strip()
]
if manifest.get("target_design") != "axis-1-quantized-uniform":
    raise SystemExit("the strict axis-1 loop requires axis-1-quantized-uniform targets")
expected = {
    "keyword_count": 1011,
    "target_count_per_keyword": 30,
    "task_count": 30330,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"plan {key} must equal {value}; found {manifest.get(key)!r}")
if len(rows) != 30330:
    raise SystemExit(f"keyword target grid must contain 30330 rows; found {len(rows)}")
counts = collections.Counter(str(row["keyword_id"]) for row in rows)
if len(counts) != 1011 or set(counts.values()) != {30}:
    raise SystemExit("keyword target grid must contain exactly 30 targets for each of 1011 keywords")
increment = float(design.get("lattice_increment", math.nan))
if not math.isclose(increment, 0.001, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(f"axis-1 lattice increment must equal 0.001; found {increment!r}")
if design.get("lattice_point_count") != 1001:
    raise SystemExit("axis-1 target design must contain 1001 lattice points")
if design.get("pooled_target_count") != 30330:
    raise SystemExit("axis-1 target design must contain 30330 pooled targets")
if design.get("occupied_lattice_point_count") != 1001:
    raise SystemExit("axis-1 target design must occupy all 1001 lattice points")
tolerance = float(os.environ["READINESS_DISTANCE_TOLERANCE"])
text_contract = os.environ["READINESS_TEXT_CONTRACT"]
acceptance_contract = os.environ["READINESS_ACCEPTANCE_CONTRACT"]
contracts = {"question-v1", "search-trigger-v2"}
generation_profile = os.environ["READINESS_GENERATION_PROFILE"]
profiles = {"balanced-v1", "high-axis-action-v1"}
if text_contract not in contracts or acceptance_contract not in contracts:
    raise SystemExit("unsupported readiness text or acceptance contract")
if text_contract != acceptance_contract:
    raise SystemExit("readiness text and acceptance contracts must use one version")
if generation_profile not in profiles:
    raise SystemExit(f"unsupported readiness generation profile: {generation_profile}")
if generation_profile == "high-axis-action-v1" and text_contract != "search-trigger-v2":
    raise SystemExit("high-axis-action-v1 requires search-trigger-v2")
minimum_axis = os.environ["READINESS_REFINEMENT_MIN_TARGET_AXIS_1"]
if minimum_axis:
    minimum_axis = float(minimum_axis)
    if not 0.0 <= minimum_axis <= 1.0:
        raise SystemExit("refinement minimum target axis 1 must lie in [0, 1]")
if generation_profile == "high-axis-action-v1" and (
    minimum_axis == "" or minimum_axis < 0.70
):
    raise SystemExit(
        "high-axis-action-v1 requires refinement minimum target axis 1 >= 0.70"
    )
if os.environ["READINESS_REFINEMENT_TASK_PRIORITY"] not in {
    "stable-hash", "descending-axis-1"
}:
    raise SystemExit("unsupported readiness refinement task priority")
expected_tolerance = 0.017 if text_contract == "question-v1" else 0.035
if not math.isclose(tolerance, expected_tolerance, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        f"{text_contract} tolerance must equal {expected_tolerance:.3f}; "
        f"found {tolerance!r}"
    )
if int(os.environ["READINESS_REFINEMENT_CANDIDATES_PER_TASK"]) <= 0:
    raise SystemExit("refinement candidates per task must be positive")
print(
    "strict axis-1 plan preflight: PASS "
    f"keywords=1011 targets=30330 lattice=0.001 "
    f"tolerance={tolerance:.3f} contract={text_contract}"
    f" profile={generation_profile}"
)
PY

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
recovery_pipeline="${READINESS_RECOVERY_PIPELINE_ROOT:-}"
initial_candidate_root="${READINESS_INITIAL_CANDIDATE_ROOT:-}"
initial_candidate_file_list="${READINESS_INITIAL_CANDIDATE_FILE_LIST:-}"
initial_projection_root="${READINESS_INITIAL_PROJECTION_ROOT:-}"
initial_validation_output="${READINESS_INITIAL_VALIDATION_OUTPUT:-}"
export READINESS_SOURCE_PILOT_ROOT="$source_pilot"
export READINESS_RECOVERY_PIPELINE_ROOT="$recovery_pipeline"
export READINESS_INITIAL_CANDIDATE_ROOT="$initial_candidate_root"
export READINESS_INITIAL_CANDIDATE_FILE_LIST="$initial_candidate_file_list"
export READINESS_INITIAL_PROJECTION_ROOT="$initial_projection_root"
export READINESS_INITIAL_VALIDATION_OUTPUT="$initial_validation_output"
export READINESS_INITIAL_LOGICAL_ROUND_INDEX="${READINESS_INITIAL_LOGICAL_ROUND_INDEX:-0}"
export READINESS_MAX_REFINEMENT_ROUNDS="${READINESS_MAX_REFINEMENT_ROUNDS:-1000}"
export READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND="${READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND:-128}"
export READINESS_WORK_PARTITION_COUNT="${READINESS_WORK_PARTITION_COUNT:-1}"
export READINESS_WORK_PARTITION_INDEX="${READINESS_WORK_PARTITION_INDEX:-0}"
export READINESS_WORK_PARTITION_SALT="${READINESS_WORK_PARTITION_SALT:-readiness-target-partition-v1}"
[[ "$READINESS_WORK_PARTITION_COUNT" -ge 1 ]] || {
    echo "READINESS_WORK_PARTITION_COUNT must be positive" >&2
    exit 2
}
[[ "$READINESS_WORK_PARTITION_INDEX" -ge 0 && "$READINESS_WORK_PARTITION_INDEX" -lt "$READINESS_WORK_PARTITION_COUNT" ]] || {
    echo "work partition must satisfy 0 <= index < count" >&2
    exit 2
}
[[ "$READINESS_INITIAL_LOGICAL_ROUND_INDEX" =~ ^[0-9]+$ ]] || {
    echo "READINESS_INITIAL_LOGICAL_ROUND_INDEX must be a nonnegative integer" >&2
    exit 2
}
[[ -z "$source_pilot" || ( -z "$initial_candidate_root" && -z "$initial_candidate_file_list" ) ]] || {
    echo "configure either a source pilot or an initial checkpoint, not both" >&2
    exit 2
}
[[ -z "$initial_projection_root" || -n "$initial_candidate_root" || -n "$initial_candidate_file_list" ]] || {
    echo "initial projections require an initial candidate checkpoint" >&2
    exit 2
}
[[ -z "$initial_validation_output" || -n "$initial_candidate_root" || -n "$initial_candidate_file_list" ]] || {
    echo "initial validation requires an initial candidate checkpoint" >&2
    exit 2
}
if [[ -n "$initial_candidate_file_list" ]]; then
    test -s "$initial_candidate_file_list"
fi
if [[ -n "$initial_validation_output" ]]; then
    test -s "$initial_validation_output"
    test -s "$initial_validation_output.manifest.json"
fi
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
command -v flock >/dev/null || {
    echo "flock is required to prevent concurrent pipeline controllers" >&2
    exit 2
}
exec 9>"$pipeline_root/.controller.lock"
flock -n 9 || {
    echo "another controller already owns this pipeline root: $pipeline_root" >&2
    exit 2
}
printf 'job_id=%s pid=%s started_at=%s\n' \
    "$SLURM_JOB_ID" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$pipeline_root/controller-owner.txt"
validation_cache_root="${READINESS_VALIDATION_CACHE_ROOT:-$pipeline_root/cache}"
export READINESS_VALIDATION_CACHE_ROOT="$validation_cache_root"
mkdir -p "$validation_cache_root"
if [[ -n "$recovery_pipeline" ]]; then
    [[ -d "$recovery_pipeline" ]] || {
        echo "recovery pipeline root does not exist: $recovery_pipeline" >&2
        exit 2
    }
    [[ "$recovery_pipeline" != "$pipeline_root" ]] || {
        echo "recovery pipeline root must differ from the new pipeline root" >&2
        exit 2
    }
fi
printf '%s\n' "$pipeline_root" > "$pointer"

activate_control_runtime

# Cache reuse is an explicit, audited union.  Search roots let a recovery run
# discover every previous cache for the same immutable judge rather than
# accidentally copying only the most recent (possibly partial) allocation.
validation_cache_sources=()
if [[ -n "$recovery_pipeline" && -d "$recovery_pipeline/cache/$READINESS_VALIDATOR_ID" ]]; then
    validation_cache_sources+=("$recovery_pipeline/cache/$READINESS_VALIDATOR_ID")
fi
if [[ -n "${READINESS_VALIDATION_CACHE_SOURCES:-}" ]]; then
    IFS=: read -r -a explicit_validation_cache_sources \
        <<< "$READINESS_VALIDATION_CACHE_SOURCES"
    validation_cache_sources+=("${explicit_validation_cache_sources[@]}")
fi
if [[ -n "${READINESS_VALIDATION_CACHE_SEARCH_ROOTS:-}" ]]; then
    IFS=: read -r -a validation_cache_search_roots \
        <<< "$READINESS_VALIDATION_CACHE_SEARCH_ROOTS"
    for cache_search_root in "${validation_cache_search_roots[@]}"; do
        [[ -d "$cache_search_root" ]] || {
            echo "validation cache search root does not exist: $cache_search_root" >&2
            exit 2
        }
        while IFS= read -r -d '' discovered_cache; do
            validation_cache_sources+=("$discovered_cache")
        done < <(
            find "$cache_search_root" -type d -name "$READINESS_VALIDATOR_ID" \
                -print0
        )
    done
fi
if [[ "${#validation_cache_sources[@]}" -gt 0 ]]; then
    cache_merge_arguments=()
    for validation_cache_source in "${validation_cache_sources[@]}"; do
        cache_merge_arguments+=(--source "$validation_cache_source")
    done
    python analysis/scripts/merge_readiness_validation_caches.py \
        "${cache_merge_arguments[@]}" \
        --destination "$validation_cache_root/$READINESS_VALIDATOR_ID" \
        --judge-id "$READINESS_VALIDATOR_ID" \
        --judge-model "$READINESS_VALIDATOR_MODEL" \
        --report "$pipeline_root/logs/validator-cache-merge-job-$SLURM_JOB_ID.json"
fi

python - "$pipeline_root" "$READINESS_30K_PLAN_ROOT" "$source_pilot" "$recovery_pipeline" \
    "$initial_candidate_root" "$initial_candidate_file_list" "$initial_projection_root" \
    "$initial_validation_output" "$validation_cache_root" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
plan = pathlib.Path(sys.argv[2])
source = sys.argv[3] or None
recovery = sys.argv[4] or None
initial_candidates = sys.argv[5] or None
initial_candidate_file_list = sys.argv[6] or None
initial_projections = sys.argv[7] or None
initial_validation = sys.argv[8] or None
validation_cache_root = sys.argv[9]
keyword_section_plan = os.getenv("READINESS_KEYWORD_SECTION_PLAN") or None
high_axis_baseline = os.getenv("READINESS_HIGH_AXIS_BASELINE_SELECTED") or None
manifest_path = root / "pipeline_manifest.json"
identity = {
    "format_version": "readiness-30k-end-to-end-v1",
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "plan_manifest_sha256": hashlib.sha256((plan / "plan_manifest.json").read_bytes()).hexdigest(),
    "source_pilot_root": source,
    "recovery_pipeline_root": recovery,
    "initial_candidate_root": initial_candidates,
    "initial_candidate_file_list": initial_candidate_file_list,
    "initial_candidate_file_list_sha256": (
        hashlib.sha256(pathlib.Path(initial_candidate_file_list).read_bytes()).hexdigest()
        if initial_candidate_file_list
        else None
    ),
    "initial_logical_round_index": int(os.environ["READINESS_INITIAL_LOGICAL_ROUND_INDEX"]),
    "initial_projection_root": initial_projections,
    "initial_validation_output": initial_validation,
    "validation_cache_root": validation_cache_root,
    "generator_ids": [os.environ["READINESS_GENERATOR_A_ID"], os.environ["READINESS_GENERATOR_B_ID"]],
    "generator_models": [os.environ["READINESS_GENERATOR_A_MODEL"], os.environ["READINESS_GENERATOR_B_MODEL"]],
    "validator_id": os.environ["READINESS_VALIDATOR_ID"],
    "validator_model": os.environ["READINESS_VALIDATOR_MODEL"],
    "distance_tolerance": float(os.environ["READINESS_DISTANCE_TOLERANCE"]),
    "text_contract": os.environ["READINESS_TEXT_CONTRACT"],
    "acceptance_contract_version": os.environ["READINESS_ACCEPTANCE_CONTRACT"],
    "generation_profile": os.environ["READINESS_GENERATION_PROFILE"],
    "disagreement_weight": float(os.environ["READINESS_DISAGREEMENT_WEIGHT"]),
    "refinement_candidates_per_task": int(os.environ["READINESS_REFINEMENT_CANDIDATES_PER_TASK"]),
    "master_seed": int(os.environ["READINESS_MASTER_SEED"]),
    "maximum_refinement_rounds": int(os.environ["READINESS_MAX_REFINEMENT_ROUNDS"]),
    "refinement_task_limit_per_round": int(os.environ["READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND"]),
    "refinement_minimum_target_axis_1": (
        float(os.environ["READINESS_REFINEMENT_MIN_TARGET_AXIS_1"])
        if os.environ["READINESS_REFINEMENT_MIN_TARGET_AXIS_1"]
        else None
    ),
    "refinement_task_priority": os.environ["READINESS_REFINEMENT_TASK_PRIORITY"],
    "high_axis_baseline_selected": high_axis_baseline,
    "high_axis_baseline_selected_sha256": (
        hashlib.sha256(pathlib.Path(high_axis_baseline).read_bytes()).hexdigest()
        if high_axis_baseline
        else None
    ),
    "work_partition_count": int(os.environ["READINESS_WORK_PARTITION_COUNT"]),
    "work_partition_index": int(os.environ["READINESS_WORK_PARTITION_INDEX"]),
    "work_partition_salt": os.environ["READINESS_WORK_PARTITION_SALT"],
    "keyword_section_plan": keyword_section_plan,
    "keyword_section_plan_sha256": (
        hashlib.sha256(pathlib.Path(keyword_section_plan).read_bytes()).hexdigest()
        if keyword_section_plan
        else None
    ),
    "validation_shard_count": int(os.environ["READINESS_VALIDATION_SHARD_COUNT"]),
    "coordinate_only_projection_reuse": os.environ["READINESS_COORDINATE_ONLY_PROJECTION_REUSE"] == "1",
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "allocated_gpu_count": int(os.environ["READINESS_ALLOCATED_GPU_COUNT"]),
}
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text())
    existing.setdefault("generation_profile", "balanced-v1")
    existing.setdefault("refinement_minimum_target_axis_1", None)
    existing.setdefault("refinement_task_priority", "stable-hash")
    existing.setdefault("high_axis_baseline_selected", None)
    existing.setdefault("high_axis_baseline_selected_sha256", None)
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
        "stop_after_physical_round": os.getenv("READINESS_STOP_AFTER_PHYSICAL_ROUND"),
    })
    value = existing
else:
    value = dict(identity)
    value["allocation_slices"] = [{
        "approved_walltime": identity["approved_walltime"],
        "allocation_estimate": identity["allocation_estimate"],
        "slurm_job_id": identity["slurm_job_id"],
        "allocated_gpu_count": identity["allocated_gpu_count"],
        "stop_after_physical_round": os.getenv("READINESS_STOP_AFTER_PHYSICAL_ROUND"),
    }]
temporary = manifest_path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(manifest_path)
PY

worker="$GEODML_REPOSITORY/analysis/scripts/slurm/jupiter/run_readiness_30k_pipeline_stage.sh"
max_rounds="$READINESS_MAX_REFINEMENT_ROUNDS"
refinement_task_limit="$READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND"
stop_after_physical_round="${READINESS_STOP_AFTER_PHYSICAL_ROUND:-}"
[[ "$max_rounds" -ge 1 ]] || { echo "READINESS_MAX_REFINEMENT_ROUNDS must be positive" >&2; exit 2; }
[[ "$refinement_task_limit" -ge 1 ]] || { echo "READINESS_REFINEMENT_TASK_LIMIT_PER_ROUND must be positive" >&2; exit 2; }
if [[ -n "$stop_after_physical_round" ]]; then
    [[ "$stop_after_physical_round" =~ ^[0-9]+$ ]] || {
        echo "READINESS_STOP_AFTER_PHYSICAL_ROUND must be a nonnegative integer" >&2
        exit 2
    }
fi
export READINESS_GENERATION_SECONDS="${READINESS_GENERATION_SECONDS:-3000}"
candidate_files=()
previous_selection=""
pipeline_status="refine"
active_srun_pids=()
recovered_generation_candidates=()
logical_round_offset="$READINESS_INITIAL_LOGICAL_ROUND_INDEX"
previous_qwen_projection_root=""
previous_mistral_projection_root=""
previous_validation_output="$initial_validation_output"

if [[ -n "$recovery_pipeline" && -z "$initial_candidate_file_list" ]]; then
    while IFS= read -r recovered_candidate; do
        recovered_manifest="$recovered_candidate.manifest.json"
        [[ -s "$recovered_manifest" ]] || {
            echo "recovery candidate lacks a checkpoint manifest: $recovered_candidate" >&2
            exit 2
        }
        recovered_count="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_count"])' "$recovered_manifest")"
        [[ "$recovered_count" -gt 0 ]] || continue
        recovered_generation_candidates+=("$recovered_candidate")
        recovered_round_name="$(basename "$(dirname "$(dirname "$(dirname "$recovered_candidate")")")")"
        recovered_round_number="${recovered_round_name#round-}"
        recovered_round_number="$((10#$recovered_round_number))"
        if [[ "$recovered_round_number" -gt "$logical_round_offset" ]]; then
            logical_round_offset="$recovered_round_number"
        fi
    done < <(
        find "$recovery_pipeline" -path '*/round-*/generation/candidates/*.jsonl' \
            ! -name '*.failures.jsonl' -type f -print | sort
    )
fi
if [[ "${#recovered_generation_candidates[@]}" -gt 0 ]]; then
    echo "RECOVERED GENERATION CHECKPOINTS: files=${#recovered_generation_candidates[@]} logical_round_offset=$logical_round_offset"
fi

interrupt_pipeline() {
    local signal_name="$1" exit_code=130 pid
    [[ "$signal_name" == "TERM" ]] && exit_code=143
    trap - INT TERM
    echo "INTERRUPTED: terminating ${#active_srun_pids[@]} active Slurm steps; cached work is preserved." >&2
    for pid in "${active_srun_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${active_srun_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$exit_code"
}

trap 'interrupt_pipeline INT' INT
trap 'interrupt_pipeline TERM' TERM

generation_terminal() {
    local manifest
    for manifest in "$@"; do
        [[ -s "$manifest" ]] || return 1
        python -c 'import json,sys; assert json.load(open(sys.argv[1]))["slice_terminal"]' "$manifest" || return 1
    done
}

allocation_seconds_left() {
    local value
    local values=()
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        mapfile -t values < <(
            squeue -h -r -j "$SLURM_ARRAY_JOB_ID" -o '%A|%a|%L' |
                awk -F'|' \
                    -v array_id="$SLURM_ARRAY_JOB_ID" \
                    -v task_id="$SLURM_ARRAY_TASK_ID" \
                    '$1 == array_id && $2 == task_id {
                        gsub(/[[:space:]]/, "", $3)
                        print $3
                    }'
        )
    else
        mapfile -t values < <(
            squeue -h -j "$SLURM_JOB_ID" -o '%A|%L' |
                awk -F'|' -v job_id="$SLURM_JOB_ID" \
                    '$1 == job_id {
                        gsub(/[[:space:]]/, "", $2)
                        print $2
                    }'
        )
    fi
    [[ "${#values[@]}" -eq 1 ]] || {
        echo "expected exactly one Slurm time-left value for this allocation; found ${#values[@]}" >&2
        return 1
    }
    value="${values[0]}"
    [[ -n "$value" ]] || return 1
    python - "$value" <<'PY'
import sys

value = sys.argv[1]
days = 0
if "-" in value:
    day, value = value.split("-", 1)
    days = int(day)
parts = [int(item) for item in value.split(":")]
if len(parts) == 3:
    hours, minutes, seconds = parts
elif len(parts) == 2:
    hours = 0
    minutes, seconds = parts
else:
    raise SystemExit(f"unsupported Slurm time-left value: {sys.argv[1]}")
print(days * 86400 + hours * 3600 + minutes * 60 + seconds)
PY
}

artifact_count_matches() {
    local manifest="$1" expected="$2"
    [[ -s "$manifest" ]] || return 1
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["candidate_count"] == int(sys.argv[2])' "$manifest" "$expected"
}

projection_artifact_matches() {
    local root="$1" expected="$2" candidates="$3"
    local expected_attention="${READINESS_LLM2VEC_ATTENTION_IMPLEMENTATION:-eager}"
    [[ -s "$root/projection_manifest.json" ]] || return 1
    [[ -s "$root/question_projections.jsonl" ]] || return 1
    if [[ "$READINESS_COORDINATE_ONLY_PROJECTION_REUSE" != "1" ]]; then
        [[ -s "$root/question_embeddings.restricted-local.npz" ]] || return 1
    fi
    local coordinate_args=()
    if [[ "$READINESS_COORDINATE_ONLY_PROJECTION_REUSE" == "1" ]]; then
        python -c 'import json,sys; assert json.load(open(sys.argv[1]))["embedding_arrays_included"] is False' \
            "$root/projection_manifest.json" || return 1
        coordinate_args+=(--allow-coordinate-only)
    fi
    python \
        "$GEODML_REPOSITORY/analysis/scripts/verify_readiness_projection_checkpoint.py" \
        --projection-manifest "$root/projection_manifest.json" \
        --expected-count "$expected" \
        --candidate-file-list "$candidates" \
        --expected-attention "$expected_attention" \
        "${coordinate_args[@]}"
}

recover_projection_attempt() {
    local final_root="$1" attempt_pattern="$2" expected="$3" candidates="$4"
    shift 4
    [[ ! -e "$final_root" ]] || return 0
    local search_root attempt recovered_temporary
    local complete_attempts=()
    for search_root in "$@"; do
        [[ -d "$search_root" ]] || continue
        while IFS= read -r attempt; do
            if projection_artifact_matches "$attempt" "$expected" "$candidates"; then
                complete_attempts+=("$attempt")
            fi
        done < <(find "$search_root" -mindepth 1 -maxdepth 1 -type d -name "$attempt_pattern" -print | sort)
    done
    if [[ "${#complete_attempts[@]}" -gt 1 ]]; then
        echo "multiple complete projection attempts match the exact candidate set for $final_root" >&2
        printf '  %s\n' "${complete_attempts[@]}" >&2
        return 2
    fi
    [[ "${#complete_attempts[@]}" -eq 1 ]] || return 0
    recovered_temporary="$final_root.recovering-${SLURM_JOB_ID}-${BASHPID:-$$}"
    [[ ! -e "$recovered_temporary" ]] || {
        echo "projection recovery collision: $recovered_temporary" >&2
        return 2
    }
    cp -a "${complete_attempts[0]}" "$recovered_temporary"
    mv "$recovered_temporary" "$final_root"
    echo "recovered completed projection: ${complete_attempts[0]} -> $final_root"
}

recover_projection_source() {
    local final_root="$1" source_root="$2" expected="$3" candidates="$4"
    [[ ! -e "$final_root" ]] || return 0
    projection_artifact_matches "$source_root" "$expected" "$candidates" || {
        echo "initial projection does not match the immutable candidate checkpoint: $source_root" >&2
        return 2
    }
    local recovered_temporary="$final_root.recovering-${SLURM_JOB_ID}-${BASHPID:-$$}"
    [[ ! -e "$recovered_temporary" ]] || {
        echo "projection recovery collision: $recovered_temporary" >&2
        return 2
    }
    cp -a "$source_root" "$recovered_temporary"
    mv "$recovered_temporary" "$final_root"
    echo "recovered initial checkpoint projection: $source_root -> $final_root"
}

validation_shard_complete() {
    local manifest="$1" expected_total="$2" expected_count="$3" expected_index="$4" expected_salt="$5"
    [[ -s "$manifest" ]] || return 1
    python -c '
import json
import sys
row = json.load(open(sys.argv[1]))
assert row["total_candidate_count"] == int(sys.argv[2])
assert row["shard_count"] == int(sys.argv[3])
assert row["shard_index"] == int(sys.argv[4])
assert row.get("shard_salt", "") == sys.argv[5]
assert row["reviewed_count"] == row["candidate_count"]
' "$manifest" "$expected_total" "$expected_count" "$expected_index" "$expected_salt"
}

prepare_refinement_task_batch() {
    local source_tasks="$1" batch_tasks="$2" limit="$3"
    local targeting_args=()
    if [[ -n "$READINESS_REFINEMENT_MIN_TARGET_AXIS_1" ]]; then
        targeting_args+=(
            --minimum-target-axis-1 "$READINESS_REFINEMENT_MIN_TARGET_AXIS_1"
        )
    fi
    targeting_args+=(--task-priority "$READINESS_REFINEMENT_TASK_PRIORITY")
    python analysis/scripts/partition_readiness_refinement_tasks.py \
        --source-tasks "$source_tasks" \
        --output "$batch_tasks" \
        --limit "$limit" \
        --partition-count "$READINESS_WORK_PARTITION_COUNT" \
        --partition-index "$READINESS_WORK_PARTITION_INDEX" \
        --partition-salt "$READINESS_WORK_PARTITION_SALT" \
        "${targeting_args[@]}"
}

run_generation_round() {
    local round_root="$1" tasks="$2"
    mkdir -p "$round_root/candidates" "$round_root/cache" "$round_root/logs"
    local remaining_seconds reserve_seconds generation_slice_seconds
    local manifests pids generator_id generator_model task_count shard_count shard output cache log
    local generation_shards_per_generator="$((allocated_gpu_count / 2))"
    while true; do
        remaining_seconds="$(allocation_seconds_left)" || {
            echo "cannot determine remaining Slurm time; refusing to start a generation slice" >&2
            return 2
        }
        reserve_seconds="${READINESS_FINALIZATION_RESERVE_SECONDS:-900}"
        generation_slice_seconds="$((remaining_seconds - reserve_seconds))"
        if [[ "$generation_slice_seconds" -lt "${READINESS_MINIMUM_GENERATION_SECONDS:-600}" ]]; then
            echo "GENERATION CHECKPOINTED: insufficient allocation time remains to load $allocated_gpu_count generators safely"
            return 10
        fi
        if [[ "$generation_slice_seconds" -gt "$READINESS_GENERATION_SECONDS" ]]; then
            generation_slice_seconds="$READINESS_GENERATION_SECONDS"
        fi
        echo "generation_slice_seconds=$generation_slice_seconds remaining_allocation_seconds=$remaining_seconds"
        manifests=()
        pids=()
        active_srun_pids=()
        for generator_id in "$READINESS_GENERATOR_A_ID" "$READINESS_GENERATOR_B_ID"; do
            if [[ "$generator_id" == "$READINESS_GENERATOR_A_ID" ]]; then
                generator_model="$READINESS_GENERATOR_A_MODEL"
            else
                generator_model="$READINESS_GENERATOR_B_MODEL"
            fi
            task_count="$(python -c 'import json,sys; print(sum(json.loads(x)["generator_id"] == sys.argv[2] for x in open(sys.argv[1]) if x.strip()))' "$tasks" "$generator_id")"
            [[ "$task_count" -gt 0 ]] || continue
            shard_count="$generation_shards_per_generator"
            [[ "$task_count" -ge "$shard_count" ]] || shard_count="$task_count"
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
                READINESS_GENERATION_SECONDS="$generation_slice_seconds" \
                srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
                    "$worker" generate >> "$log" 2>&1 &
                pids+=("$!")
                active_srun_pids+=("$!")
            done
        done
        local failed=0 pid
        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
        active_srun_pids=()
        [[ "$failed" -eq 0 ]] || { echo "generation worker failure; inspect $round_root/logs" >&2; return 2; }
        if generation_terminal "${manifests[@]}"; then
            break
        fi
        echo "GENERATION CONTINUING: slice checkpointed; resuming unfinished shards in this allocation"
    done
    mapfile -t round_candidates < <(find "$round_root/candidates" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.failures.jsonl' | sort)
    candidate_files+=("${round_candidates[@]}")
    return 0
}

for ((round_index=0; round_index<=max_rounds; round_index++)); do
    round_started_seconds="$SECONDS"
    printf -v round_name 'round-%02d' "$round_index"
    logical_round_index="$((round_index + logical_round_offset))"
    round_root="$pipeline_root/$round_name"
    mkdir -p "$round_root"

    if [[ "$round_index" -eq 0 && -n "$initial_candidate_file_list" ]]; then
        mapfile -t initial_candidates < "$initial_candidate_file_list"
        [[ "${#initial_candidates[@]}" -gt 0 ]] || {
            echo "initial checkpoint candidate file list is empty: $initial_candidate_file_list" >&2
            exit 2
        }
        for initial_candidate in "${initial_candidates[@]}"; do
            test -s "$initial_candidate" || {
                echo "initial checkpoint candidate file is missing: $initial_candidate" >&2
                exit 2
            }
            test -s "$initial_candidate.manifest.json" || {
                echo "initial checkpoint candidate lacks its checkpoint manifest: $initial_candidate" >&2
                exit 2
            }
        done
        candidate_files+=("${initial_candidates[@]}")
    elif [[ "$round_index" -eq 0 && -n "$initial_candidate_root" ]]; then
        mapfile -t initial_candidates < <(find "$initial_candidate_root" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.failures.jsonl' | sort)
        [[ "${#initial_candidates[@]}" -gt 0 ]] || {
            echo "initial checkpoint has no candidate JSONL files: $initial_candidate_root" >&2
            exit 2
        }
        for initial_candidate in "${initial_candidates[@]}"; do
            test -s "$initial_candidate.manifest.json" || {
                echo "initial checkpoint candidate lacks its checkpoint manifest: $initial_candidate" >&2
                exit 2
            }
        done
        candidate_files+=("${initial_candidates[@]}")
        candidate_files+=("${recovered_generation_candidates[@]}")
    elif [[ "$round_index" -eq 0 && -n "$source_pilot" ]]; then
        mapfile -t source_candidates < <(find "$source_pilot/candidates" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.failures.jsonl' | sort)
        [[ "${#source_candidates[@]}" -gt 0 ]] || { echo "source pilot has no candidates" >&2; exit 2; }
        candidate_files+=("${source_candidates[@]}")
    else
        if [[ "$round_index" -eq 0 ]]; then
            source_tasks="$READINESS_30K_PLAN_ROOT/generation_tasks_round_00.jsonl"
            if [[ "$READINESS_WORK_PARTITION_COUNT" -gt 1 ]]; then
                tasks="$round_root/generation-task-batch.jsonl"
                prepare_refinement_task_batch "$source_tasks" "$tasks" "$refinement_task_limit"
            else
                tasks="$source_tasks"
            fi
        else
            source_tasks="$previous_selection/generation_tasks_round_$(printf '%02d' "$logical_round_index").jsonl"
            tasks="$round_root/refinement-task-batch.jsonl"
            prepare_refinement_task_batch "$source_tasks" "$tasks" "$refinement_task_limit"
        fi
        test -e "$tasks"
        if [[ ! -s "$tasks" ]]; then
            if [[ -n "$READINESS_REFINEMENT_MIN_TARGET_AXIS_1" ]]; then
                pipeline_status="targeted-scope-complete"
            elif [[ "$READINESS_WORK_PARTITION_COUNT" -gt 1 ]]; then
                pipeline_status="partition-complete"
            else
                pipeline_status="pass"
            fi
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
            --text-contract "$READINESS_TEXT_CONTRACT" \
            --output-dir "$raw_diversity"
        raw_diversity_exit=$?
        set -e
        printf '%s\n' "$raw_diversity_exit" > "$round_root/raw-diversity-exit-code.txt"
    fi

    export READINESS_VALIDATION_OUTPUT="$round_root/validation.jsonl"
    export READINESS_VALIDATION_CACHE="$validation_cache_root/$READINESS_VALIDATOR_ID"
    export QWEN_PROJECTION_ROOT="$round_root/projections/qwen"
    export MISTRAL_PROJECTION_ROOT="$round_root/projections/mistral"
    mkdir -p "$READINESS_VALIDATION_CACHE" "$round_root/logs" "$round_root/projections"

    projection_search_roots=("$round_root/projections")
    if [[ -n "$recovery_pipeline" ]]; then
        projection_search_roots+=("$recovery_pipeline/$round_name/projections")
    fi
    if [[ "$round_index" -eq 0 && -n "$initial_projection_root" ]]; then
        recover_projection_source \
            "$QWEN_PROJECTION_ROOT" "$initial_projection_root/qwen" \
            "$candidate_count" "$candidate_list" || exit 2
        recover_projection_source \
            "$MISTRAL_PROJECTION_ROOT" "$initial_projection_root/mistral" \
            "$candidate_count" "$candidate_list" || exit 2
    fi
    recover_projection_attempt \
        "$QWEN_PROJECTION_ROOT" '.qwen-attempt-*' "$candidate_count" "$candidate_list" \
        "${projection_search_roots[@]}" || exit 2
    recover_projection_attempt \
        "$MISTRAL_PROJECTION_ROOT" '.mistral-attempt-*' "$candidate_count" "$candidate_list" \
        "${projection_search_roots[@]}" || exit 2

    validation_pids=() validation_names=()
    active_srun_pids=()
    validation_shard_count="$READINESS_VALIDATION_SHARD_COUNT"
    validation_shard_salt="${READINESS_VALIDATION_SHARD_SALT:-}"
    [[ "$validation_shard_count" -ge 1 && "$validation_shard_count" -le "$allocated_gpu_count" ]] || {
        echo "validation shard count must be between 1 and $allocated_gpu_count" >&2
        exit 2
    }
    validation_shard_files=()
    pending_validation_indices=()
    for ((validation_shard_index=0; validation_shard_index<validation_shard_count; validation_shard_index++)); do
        validation_shard_output="$round_root/validation-shard-$validation_shard_index.jsonl"
        validation_shard_files+=("$validation_shard_output")
        if artifact_count_matches "$READINESS_VALIDATION_OUTPUT.manifest.json" "$candidate_count" || \
            validation_shard_complete "$validation_shard_output.manifest.json" "$candidate_count" \
                "$validation_shard_count" "$validation_shard_index" "$validation_shard_salt"; then
            continue
        fi
        pending_validation_indices+=("$validation_shard_index")
    done

    launch_validation_shard() {
        local shard_index="$1"
        local shard_output="$round_root/validation-shard-$shard_index.jsonl"
        READINESS_VALIDATION_OUTPUT="$shard_output" \
        READINESS_VALIDATION_SHARD_COUNT="$validation_shard_count" \
        READINESS_VALIDATION_SHARD_INDEX="$shard_index" \
        READINESS_VALIDATION_SHARD_SALT="$validation_shard_salt" \
        READINESS_BASE_VALIDATION_OUTPUT="$previous_validation_output" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" validate > "$round_root/logs/validate-shard-$shard_index.log" 2>&1 &
        validation_pids+=("$!")
        validation_names+=("validate-shard-$shard_index")
        active_srun_pids+=("$!")
    }

    projection_attempt="$SLURM_JOB_ID-${BASHPID:-$$}-$(date -u +%Y%m%dT%H%M%SZ)"
    qwen_projection_temporary="$round_root/projections/.qwen-attempt-$projection_attempt"
    mistral_projection_temporary="$round_root/projections/.mistral-attempt-$projection_attempt"
    qwen_projection_launched=0
    mistral_projection_launched=0
    projection_launch_count=0
    if ! projection_artifact_matches "$QWEN_PROJECTION_ROOT" "$candidate_count" "$candidate_list"; then
        [[ ! -e "$QWEN_PROJECTION_ROOT" ]] || { echo "partial Qwen projection; choose a fresh pipeline root" >&2; exit 2; }
        [[ ! -e "$qwen_projection_temporary" ]] || { echo "projection attempt collision: $qwen_projection_temporary" >&2; exit 2; }
        qwen_projection_launched=1
        projection_launch_count=$((projection_launch_count + 1))
    fi
    if ! projection_artifact_matches "$MISTRAL_PROJECTION_ROOT" "$candidate_count" "$candidate_list"; then
        [[ ! -e "$MISTRAL_PROJECTION_ROOT" ]] || { echo "partial Mistral projection; choose a fresh pipeline root" >&2; exit 2; }
        [[ ! -e "$mistral_projection_temporary" ]] || { echo "projection attempt collision: $mistral_projection_temporary" >&2; exit 2; }
        mistral_projection_launched=1
        projection_launch_count=$((projection_launch_count + 1))
    fi

    # Reserve one GPU for each missing LLM2Vec view.  The remaining GPUs begin
    # validation immediately.  As soon as both short projection jobs finish,
    # launch the remaining validation shards so all allocated GPUs stay productive.
    initial_validation_slots="$((allocated_gpu_count - projection_launch_count))"
    if [[ "$initial_validation_slots" -lt 0 ]]; then
        echo "projection stages exceed the allocated GPU count" >&2
        exit 2
    fi
    initial_validation_count="${#pending_validation_indices[@]}"
    if [[ "$initial_validation_count" -gt "$initial_validation_slots" ]]; then
        initial_validation_count="$initial_validation_slots"
    fi
    for ((pending_index=0; pending_index<initial_validation_count; pending_index++)); do
        launch_validation_shard "${pending_validation_indices[$pending_index]}"
    done

    projection_pids=() projection_names=()
    if [[ "$qwen_projection_launched" -eq 1 ]]; then
        READINESS_BASE_PROJECTION_ROOT="$previous_qwen_projection_root" \
        QWEN_PROJECTION_ROOT="$qwen_projection_temporary" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" project-qwen > "$round_root/logs/project-qwen.log" 2>&1 &
        projection_pids+=("$!"); projection_names+=("project-qwen")
        active_srun_pids+=("$!")
    fi
    if [[ "$mistral_projection_launched" -eq 1 ]]; then
        READINESS_BASE_PROJECTION_ROOT="$previous_mistral_projection_root" \
        MISTRAL_PROJECTION_ROOT="$mistral_projection_temporary" \
        srun --exact --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
            "$worker" project-mistral > "$round_root/logs/project-mistral.log" 2>&1 &
        projection_pids+=("$!"); projection_names+=("project-mistral")
        active_srun_pids+=("$!")
    fi

    projection_failure=0
    for index in "${!projection_pids[@]}"; do
        if ! wait "${projection_pids[$index]}"; then
            echo "stage failed: ${projection_names[$index]}; inspect $round_root/logs" >&2
            projection_failure=1
        fi
    done
    if [[ "$projection_failure" -ne 0 ]]; then
        for pid in "${validation_pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        for pid in "${validation_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        active_srun_pids=()
        exit 2
    fi
    if [[ "$qwen_projection_launched" -eq 1 ]]; then
        mv "$qwen_projection_temporary" "$QWEN_PROJECTION_ROOT"
    fi
    if [[ "$mistral_projection_launched" -eq 1 ]]; then
        mv "$mistral_projection_temporary" "$MISTRAL_PROJECTION_ROOT"
    fi
    previous_qwen_projection_root="$QWEN_PROJECTION_ROOT"
    previous_mistral_projection_root="$MISTRAL_PROJECTION_ROOT"

    for ((pending_index=initial_validation_count; pending_index<${#pending_validation_indices[@]}; pending_index++)); do
        launch_validation_shard "${pending_validation_indices[$pending_index]}"
    done

    validation_failure=0
    for index in "${!validation_pids[@]}"; do
        if ! wait "${validation_pids[$index]}"; then
            echo "stage failed: ${validation_names[$index]}; inspect $round_root/logs" >&2
            validation_failure=1
        fi
    done
    active_srun_pids=()
    [[ "$validation_failure" -eq 0 ]] || exit 2

    if ! artifact_count_matches "$READINESS_VALIDATION_OUTPUT.manifest.json" "$candidate_count"; then
        python - "$READINESS_VALIDATION_OUTPUT" "$candidate_list" "${validation_shard_files[@]}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
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
shard_salts = {manifest.get("shard_salt", "") for manifest in manifests}
if len(shard_salts) != 1:
    raise SystemExit("validation shard salts differ")
acceptance_contracts = {
    manifest.get("acceptance_contract_version", "question-v1")
    for manifest in manifests
}
if len(acceptance_contracts) != 1:
    raise SystemExit("validation shard acceptance contracts differ")
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
    "maximum_shard_elapsed_seconds": max(
        float(row.get("elapsed_seconds", 0.0)) for row in manifests
    ),
    "sum_shard_elapsed_seconds": sum(
        float(row.get("elapsed_seconds", 0.0)) for row in manifests
    ),
    "shard_salt": shard_salts.pop(),
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
    "acceptance_contract_version": acceptance_contracts.pop(),
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
    previous_validation_output="$READINESS_VALIDATION_OUTPUT"

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
        selection_temporary="$round_root/.strict-selection-attempt-$SLURM_JOB_ID-${BASHPID:-$$}-$(date -u +%Y%m%dT%H%M%SZ)"
        [[ ! -e "$selection_temporary" ]] || { echo "selection attempt collision: $selection_temporary" >&2; exit 2; }
        python analysis/scripts/build_readiness_prompt_population.py spatial-select \
            --plan-dir "$READINESS_30K_PLAN_ROOT" \
            --candidates "${candidate_files[@]}" \
            --reference-projections "$QWEN_PROJECTION_ROOT" \
            --candidate-projections "$MISTRAL_PROJECTION_ROOT" \
            --robustness-battery "$READINESS_BATTERY_ROOT" \
            --validations "$READINESS_VALIDATION_OUTPUT" \
            --generator-ids "$READINESS_GENERATOR_A_ID,$READINESS_GENERATOR_B_ID" \
            --next-round-index "$((logical_round_index + 1))" \
            --distance-tolerance "$READINESS_DISTANCE_TOLERANCE" \
            --text-contract "$READINESS_TEXT_CONTRACT" \
            --acceptance-contract "$READINESS_ACCEPTANCE_CONTRACT" \
            --require-both-views-within-tolerance \
            --require-delexicalized-template-uniqueness \
            --disagreement-weight "$READINESS_DISAGREEMENT_WEIGHT" \
            --candidates-per-task "$READINESS_REFINEMENT_CANDIDATES_PER_TASK" \
            --master-seed "$READINESS_MASTER_SEED" \
            --output-dir "$selection_temporary"
        mv "$selection_temporary" "$selection"
    fi

    selected="$selection/spatially_selected_questions.jsonl"
    final_diversity="$round_root/selected-diversity"
    selected_diversity_exit=2
    if [[ -s "$selected" ]]; then
        if [[ ! -s "$final_diversity/question_diversity_audit.json" ]]; then
            set +e
            python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
                --questions "$selected" \
                --text-contract "$READINESS_TEXT_CONTRACT" \
                --output-dir "$final_diversity"
            selected_diversity_exit=$?
            set -e
            printf '%s\n' "$selected_diversity_exit" > "$round_root/selected-diversity-exit-code.txt"
        else
            selected_diversity_exit="$(python -c 'import json,sys; print(0 if json.load(open(sys.argv[1]))["all_checks_passed"] else 2)' "$final_diversity/question_diversity_audit.json")"
        fi
    fi

    round_elapsed_seconds="$((SECONDS - round_started_seconds))"
    python - "$round_root" "$candidate_count" "$selected_diversity_exit" "$source_pilot" "$round_elapsed_seconds" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidate_count = int(sys.argv[2])
diversity_exit = int(sys.argv[3])
source_pilot_mode = bool(sys.argv[4])
round_elapsed_seconds = int(sys.argv[5])
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
    "text_contract": selection.get("text_contract", "question-v1"),
    "acceptance_contract_version": selection.get(
        "acceptance_contract_version", "question-v1"
    ),
    "generation_profile": os.environ.get(
        "READINESS_GENERATION_PROFILE", "balanced-v1"
    ),
    "selected_diversity_gate_passed": diversity_exit == 0,
    "spacing_gate_passed": diagnostics["overall_spacing_gate_passed"],
    "source_pilot_mode": source_pilot_mode,
    "round_elapsed_seconds": round_elapsed_seconds,
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
    echo "ROUND COMPLETE: round=$round_name elapsed_seconds=$round_elapsed_seconds candidates=$candidate_count"

    if [[ -n "${READINESS_HIGH_AXIS_BASELINE_SELECTED:-}" && "$round_index" -eq 0 ]]; then
        python - "$READINESS_HIGH_AXIS_BASELINE_SELECTED" "$selected" <<'PY'
import json
import pathlib
import sys

def cells(path):
    rows = [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        (str(row["keyword_id"]), str(row["target_id"])): str(row["candidate_id"])
        for row in rows
    }

baseline = cells(sys.argv[1])
observed = cells(sys.argv[2])
if observed != baseline:
    raise SystemExit(
        "round-00 v2 selection does not reproduce the registered high-axis baseline: "
        f"baseline={len(baseline)} observed={len(observed)}"
    )
print(f"HIGH_AXIS_BASELINE_REPRODUCTION=PASS selected={len(observed)}")
PY
    fi

    if [[ -n "${READINESS_HIGH_AXIS_BASELINE_SELECTED:-}" && -s "$round_root/refinement-task-batch.jsonl" ]]; then
        test -s "$READINESS_HIGH_AXIS_BASELINE_SELECTED" || {
            echo "missing high-axis baseline selection: $READINESS_HIGH_AXIS_BASELINE_SELECTED" >&2
            exit 2
        }
        high_axis_yield="$round_root/high-axis-yield"
        python analysis/scripts/audit_readiness_high_axis_generation_yield.py \
            --baseline-selected "$READINESS_HIGH_AXIS_BASELINE_SELECTED" \
            --round-root "$round_root" \
            --output-dir "$high_axis_yield" \
            --minimum-target-axis-1 "$READINESS_REFINEMENT_MIN_TARGET_AXIS_1"
        cat "$high_axis_yield/high_axis_yield.md"
    fi

    if [[ -n "$stop_after_physical_round" && "$round_index" -ge "$stop_after_physical_round" ]]; then
        pipeline_status="operational-checkpoint"
        echo "OPERATIONAL CHECKPOINT: completed $round_name; refusing to start a later round"
        break
    fi

    next_tasks="$selection/generation_tasks_round_$(printf '%02d' "$((logical_round_index + 1))").jsonl"
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
if [[ "$pipeline_status" == "targeted-scope-complete" ]]; then
    echo "TARGETED HIGH-AXIS SCOPE: COMPLETE"
    echo "STRICT VERIFIED POPULATION: INCOMPLETE OUTSIDE TARGETED SCOPE"
elif [[ "$pipeline_status" != "pass" ]]; then
    echo "STRICT VERIFIED POPULATION: REFINE" >&2
    exit 3
fi
echo "STRICT VERIFIED POPULATION: PASS"
