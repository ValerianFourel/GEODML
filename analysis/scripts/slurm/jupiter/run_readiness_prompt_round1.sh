#!/bin/bash -l
# Finish and audit the two-view round-1 readiness prompt pilot in an allocation.
# This script never requests Slurm resources and is safe to rerun after a
# completed stage. Partial immutable output directories fail closed.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside an existing JUPITER Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set GEODML_EXPECTED_COMMIT to the exact pushed SHA}"

clear_inherited_python_runtime() {
    local inherited_venv_bin="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin}"
    if [[ -n "$inherited_venv_bin" ]]; then
        local cleaned_path="" path_entry
        local path_entries=()
        IFS=: read -r -a path_entries <<< "$PATH"
        for path_entry in "${path_entries[@]}"; do
            [[ "$path_entry" == "$inherited_venv_bin" ]] && continue
            cleaned_path="${cleaned_path:+$cleaned_path:}$path_entry"
        done
        export PATH="$cleaned_path"
    fi
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV VENV_SITE MODULE_PYTHONPATH
    hash -r
}

load_jupiter_stack() {
    module --force purge
    module load Stages/2026
    module load GCCcore/14.3.0
    module load SciPy-Stack/2025b
    module load git
    module load PyTorch/2.9.1
    jutil env activate -p "${JUPITER_PROJECT:-scifi}"
}

clear_inherited_python_runtime
load_jupiter_stack

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"

if [[ ! -d "$GEODML_REPOSITORY/.git" && ! -f "$GEODML_REPOSITORY/.git" ]]; then
    echo "missing exact-commit worktree: $GEODML_REPOSITORY" >&2
    exit 2
fi
actual_commit="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
if [[ "$actual_commit" != "$GEODML_EXPECTED_COMMIT" ]]; then
    echo "commit mismatch: expected=$GEODML_EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
fi
if [[ -n "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]; then
    echo "round-1 audit requires a clean exact-commit worktree" >&2
    exit 2
fi

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
pilot_pointer="${READINESS_PILOT_POINTER:-$HOME/geodml-readiness-prompt-pilot-latest.txt}"
test -s "$subspace_pointer"
test -s "$pilot_pointer"
export SUBSPACE_ROOT="${SUBSPACE_ROOT:-$(<"$subspace_pointer")}"
export PILOT_ROOT="${PILOT_ROOT:-$(<"$pilot_pointer")}"
test -d "$SUBSPACE_ROOT"
test -d "$PILOT_ROOT"

export PLAN_ROOT="${READINESS_PLAN_ROOT:-$PILOT_ROOT/plan-round-00-qwen-gemma}"
export R0_QWEN="${READINESS_R0_QWEN:-$PILOT_ROOT/generated-v2/qwen3-32b.jsonl}"
export R0_GEMMA="${READINESS_R0_GEMMA:-$PILOT_ROOT/generated-v2/gemma4-31b.jsonl}"
export R1_QWEN="${READINESS_R1_QWEN:-$PILOT_ROOT/generated-round-01/qwen3-32b.jsonl}"
export R1_GEMMA="${READINESS_R1_GEMMA:-$PILOT_ROOT/generated-round-01/gemma4-31b.jsonl}"
export VALIDATION_OUTPUT="${READINESS_VALIDATION_OUTPUT:-$PILOT_ROOT/validation-v1/llama3.3-70b.jsonl}"
export BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"

projection_pointer="${READINESS_ROUND1_PROJECTION_POINTER:-$HOME/geodml-readiness-round1-projections-latest.txt}"
if [[ -n "${READINESS_PROJECTION_ROOT:-}" ]]; then
    export PROJECTION_ROOT="$READINESS_PROJECTION_ROOT"
elif [[ -s "$projection_pointer" ]]; then
    export PROJECTION_ROOT="$(<"$projection_pointer")"
else
    export PROJECTION_ROOT="$PILOT_ROOT/projections-round-01-${GEODML_EXPECTED_COMMIT:0:8}"
fi
export QWEN_R1_PROJECTIONS="$PROJECTION_ROOT/qwen"
export MISTRAL_R1_PROJECTIONS="$PROJECTION_ROOT/mistral"

run_tag="${READINESS_ROUND1_RUN_TAG:-${GEODML_EXPECTED_COMMIT:0:8}}"
export COMPARISON_R1="${READINESS_COMPARISON_ROOT:-$PILOT_ROOT/projection-comparison-round-01-$run_tag}"
export SPATIAL_R1="${READINESS_SPATIAL_ROOT:-$PILOT_ROOT/spatial-selection-round-01-$run_tag}"
export AUDIT_R1="${READINESS_AUDIT_ROOT:-$PILOT_ROOT/round-01-audit-$run_tag}"
export EXPECTED_CANDIDATE_COUNT="${READINESS_EXPECTED_CANDIDATE_COUNT:-102}"

candidate_files=("$R0_QWEN" "$R0_GEMMA" "$R1_QWEN" "$R1_GEMMA")
required_inputs=(
    "$PLAN_ROOT/plan_manifest.json"
    "$PLAN_ROOT/subspace_bounds.json"
    "$PLAN_ROOT/target_grid.jsonl"
    "${candidate_files[@]}"
    "$VALIDATION_OUTPUT"
    "$BATTERY_ROOT/battery_manifest.json"
    "$BATTERY_ROOT/readiness_robustness_battery.json"
)
for path in "${required_inputs[@]}"; do
    if [[ ! -s "$path" ]]; then
        echo "missing required round-1 artifact: $path" >&2
        exit 2
    fi
done

cd "$GEODML_REPOSITORY"

echo "===== ROUND-1 INPUT AUDIT ====="
python3 - "$EXPECTED_CANDIDATE_COUNT" "$VALIDATION_OUTPUT" "${candidate_files[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

expected = int(sys.argv[1])
validation_path = pathlib.Path(sys.argv[2])
candidate_paths = [pathlib.Path(value) for value in sys.argv[3:]]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

candidates = [row for path in candidate_paths for row in rows(path)]
ids = [str(row["candidate_id"]) for row in candidates]
if len(candidates) != expected or len(set(ids)) != expected:
    raise SystemExit(f"candidate identity/count failure: rows={len(candidates)} unique={len(set(ids))} expected={expected}")
for row in candidates:
    digest = hashlib.sha256(str(row["question"]).encode()).hexdigest()
    if digest != row["question_sha256"]:
        raise SystemExit(f"question hash mismatch: {row['candidate_id']}")
    if str(row["keyword"]) not in str(row["question"]):
        raise SystemExit(f"exact keyword missing: {row['candidate_id']}")
reviews = rows(validation_path)
review_ids = [str(row["candidate_id"]) for row in reviews]
if len(review_ids) != expected or set(review_ids) != set(ids):
    raise SystemExit("validator does not cover the exact candidate set")
accepted = sum(bool(row["accepted"]) for row in reviews)
print(f"candidates={len(candidates)} unique={len(set(ids))}")
print(f"validator_coverage={len(reviews)}/{expected} accepted={accepted}")
print("ROUND-1 INPUT AUDIT: PASS")
PY

projection_is_complete() {
    local output="$1" expected_map="$2"
    [[ -s "$output/question_projections.jsonl" ]] || return 1
    [[ -s "$output/question_embeddings.restricted-local.npz" ]] || return 1
    [[ -s "$output/projection_manifest.json" ]] || return 1
    python3 - "$EXPECTED_CANDIDATE_COUNT" "$expected_map" "$output" "${candidate_files[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

expected = int(sys.argv[1])
expected_map = sys.argv[2]
root = pathlib.Path(sys.argv[3])
candidate_paths = [pathlib.Path(value) for value in sys.argv[4:]]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

candidates = [row for path in candidate_paths for row in rows(path)]
expected_hashes = {str(row["candidate_id"]): str(row["question_sha256"]) for row in candidates}
projected = rows(root / "question_projections.jsonl")
actual = {}
for row in projected:
    candidate_id = str(row["candidate_id"])
    nested = row["projection"]
    if nested["item_id"] != candidate_id:
        raise SystemExit(f"nested projection identity mismatch: {candidate_id}")
    actual[candidate_id] = str(nested["text_sha256"])
if len(projected) != expected or actual != expected_hashes:
    raise SystemExit(f"projection does not exactly cover {expected} candidate texts")
manifest = json.loads((root / "projection_manifest.json").read_text(encoding="utf-8"))
if manifest["candidate_count"] != expected or manifest["map_id"] != expected_map:
    raise SystemExit("projection manifest count or map identity mismatch")
print(f"verified_projection={root} map_id={manifest['map_id']} candidates={expected}")
PY
}

require_absent_or_complete_projection() {
    local output="$1" expected_map="$2"
    if projection_is_complete "$output" "$expected_map"; then
        return 0
    fi
    if [[ -e "$output" ]]; then
        echo "partial or conflicting immutable projection directory: $output" >&2
        echo "choose a fresh READINESS_PROJECTION_ROOT and rerun" >&2
        exit 2
    fi
    return 1
}

activate_llm2vec_runtime() {
    local venv="$1"
    clear_inherited_python_runtime
    load_jupiter_stack
    local module_pythonpath="${PYTHONPATH-}"
    # shellcheck disable=SC1090
    source "$venv/bin/activate"
    local venv_site="$venv/lib/python3.13/site-packages"
    export PYTHONPATH="$venv_site${module_pythonpath:+:$module_pythonpath}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES="${READINESS_EMBEDDING_GPU_INDEX:-0}"
}

export QWEN_MAP_ROOT="${READINESS_QWEN_MAP_ROOT:-$SUBSPACE_ROOT/maps/qwen3-8b-mntp-unsup-simcse-three-judge-gpu-v2}"
export QWEN8_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export QWEN_MNTP_REVISION="c84774c1366ea79f033504994bd254155d956d57"
export QWEN_SIMCSE_REVISION="86b17660b1b1a8efe0b822e90c995f1ac7294645"
export QWEN8_MODEL="$GEODML_MODELS_ROOT/qwen/Qwen3-8B/$QWEN8_REVISION"
export QWEN_MNTP="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp/$QWEN_MNTP_REVISION"
export QWEN_SIMCSE="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse/$QWEN_SIMCSE_REVISION"

export MISTRAL_MAP_ROOT="${READINESS_MISTRAL_MAP_ROOT:-$SUBSPACE_ROOT/maps/mistral7b-mntp-unsup-simcse-three-judge-gpu-v3}"
export MISTRAL_BASE_REVISION="63a8b081895390a26e140280378bc85ec8bce07a"
export MISTRAL_MNTP_REVISION="e76f9757923897a0c5204b3075f1062f484d033b"
export MISTRAL_SIMCSE_REVISION="2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8"
export MISTRAL_BASE="$GEODML_MODELS_ROOT/mistralai/Mistral-7B-Instruct-v0.2/$MISTRAL_BASE_REVISION"
export MISTRAL_MNTP="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp/$MISTRAL_MNTP_REVISION"
export MISTRAL_SIMCSE="$GEODML_MODELS_ROOT/mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/$MISTRAL_SIMCSE_REVISION"

for path in \
    "$QWEN_MAP_ROOT/readiness_embedding_map.json" \
    "$QWEN_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
    "$MISTRAL_MAP_ROOT/readiness_embedding_map.json" \
    "$MISTRAL_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
    "$QWEN8_MODEL/config.json" \
    "$QWEN_MNTP/adapter_config.json" \
    "$QWEN_SIMCSE/adapter_config.json" \
    "$MISTRAL_BASE/config.json" \
    "$MISTRAL_MNTP/adapter_config.json" \
    "$MISTRAL_SIMCSE/adapter_config.json"; do
    if [[ ! -s "$path" ]]; then
        echo "missing frozen map/model artifact: $path" >&2
        exit 2
    fi
done

qwen_map_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["map_id"])' "$QWEN_MAP_ROOT/readiness_embedding_map.json")"
mistral_map_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["map_id"])' "$MISTRAL_MAP_ROOT/readiness_embedding_map.json")"

echo "===== QWEN LLM2VEC PROJECTION ====="
if require_absent_or_complete_projection "$QWEN_R1_PROJECTIONS" "$qwen_map_id"; then
    echo "QWEN PROJECTION: COMPLETE; SKIPPING"
else
    (
        activate_llm2vec_runtime "${QWEN_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-torch291}"
        python - <<'PY'
import torch
import transformers
import peft
from llm2vec import LLM2Vec
assert torch.cuda.device_count() == 1
print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
print("QWEN LLM2VEC RUNTIME: PASS")
del LLM2Vec
PY
        python analysis/scripts/build_readiness_prompt_population.py project-candidates \
            --candidates "${candidate_files[@]}" \
            --map "$QWEN_MAP_ROOT/readiness_embedding_map.json" \
            --reference-coordinates "$QWEN_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
            --embedding-model "$QWEN8_MODEL" \
            --mntp-model "$QWEN_MNTP" \
            --peft-model "$QWEN_SIMCSE" \
            --embedding-batch-size "${READINESS_EMBEDDING_BATCH_SIZE:-8}" \
            --embedding-max-length "${READINESS_EMBEDDING_MAX_LENGTH:-512}" \
            --output-dir "$QWEN_R1_PROJECTIONS"
    )
    projection_is_complete "$QWEN_R1_PROJECTIONS" "$qwen_map_id"
fi

echo "===== MISTRAL LLM2VEC PROJECTION ====="
if require_absent_or_complete_projection "$MISTRAL_R1_PROJECTIONS" "$mistral_map_id"; then
    echo "MISTRAL PROJECTION: COMPLETE; SKIPPING"
else
    (
        activate_llm2vec_runtime "${MISTRAL_LLM2VEC_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-hf-llm2vec-mistral-torch291}"
        python - <<'PY'
import torch
import transformers
import peft
from llm2vec import LLM2Vec
assert torch.cuda.device_count() == 1
print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
print("MISTRAL LLM2VEC RUNTIME: PASS")
del LLM2Vec
PY
        python analysis/scripts/build_readiness_prompt_population.py project-candidates \
            --candidates "${candidate_files[@]}" \
            --map "$MISTRAL_MAP_ROOT/readiness_embedding_map.json" \
            --reference-coordinates "$MISTRAL_MAP_ROOT/readiness_supervised_subspace_coordinates.jsonl" \
            --embedding-model "$MISTRAL_BASE" \
            --mntp-model "$MISTRAL_MNTP" \
            --peft-model "$MISTRAL_SIMCSE" \
            --embedding-batch-size "${READINESS_EMBEDDING_BATCH_SIZE:-8}" \
            --embedding-max-length "${READINESS_EMBEDDING_MAX_LENGTH:-512}" \
            --output-dir "$MISTRAL_R1_PROJECTIONS"
    )
    projection_is_complete "$MISTRAL_R1_PROJECTIONS" "$mistral_map_id"
fi

printf '%s\n' "$PROJECTION_ROOT" > "$projection_pointer"

load_jupiter_stack
cd "$GEODML_REPOSITORY"

complete_comparison() {
    [[ -s "$COMPARISON_R1/aligned_question_projections.jsonl" ]] || return 1
    [[ -s "$COMPARISON_R1/projection_comparison.json" ]] || return 1
    [[ -s "$COMPARISON_R1/projection_comparison_report.md" ]] || return 1
    [[ -s "$COMPARISON_R1/comparison_manifest.json" ]] || return 1
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["candidate_count"] == int(sys.argv[2])' \
        "$COMPARISON_R1/projection_comparison.json" "$EXPECTED_CANDIDATE_COUNT"
}

echo "===== CROSS-EMBEDDING COMPARISON ====="
if complete_comparison; then
    echo "COMPARISON: COMPLETE; SKIPPING"
elif [[ -e "$COMPARISON_R1" ]]; then
    echo "partial or conflicting immutable comparison directory: $COMPARISON_R1" >&2
    exit 2
else
    python3 analysis/scripts/build_readiness_prompt_population.py compare-projections \
        --reference-projections "$QWEN_R1_PROJECTIONS" \
        --candidate-projections "$MISTRAL_R1_PROJECTIONS" \
        --robustness-battery "$BATTERY_ROOT" \
        --output-dir "$COMPARISON_R1"
    complete_comparison
fi

complete_spatial_selection() {
    [[ -s "$SPATIAL_R1/spatially_selected_questions.jsonl" ]] || return 1
    [[ -e "$SPATIAL_R1/generation_tasks_round_02.jsonl" ]] || return 1
    [[ -s "$SPATIAL_R1/spatial_coverage_diagnostics.json" ]] || return 1
    [[ -s "$SPATIAL_R1/spatial_coverage_report.md" ]] || return 1
    [[ -s "$SPATIAL_R1/run_manifest.json" ]] || return 1
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["candidate_count"] == int(sys.argv[2])' \
        "$SPATIAL_R1/run_manifest.json" "$EXPECTED_CANDIDATE_COUNT"
}

echo "===== ROUND-1 SPATIAL SELECTION ====="
if complete_spatial_selection; then
    echo "SPATIAL SELECTION: COMPLETE; SKIPPING"
elif [[ -e "$SPATIAL_R1" ]]; then
    echo "partial or conflicting immutable spatial-selection directory: $SPATIAL_R1" >&2
    exit 2
else
    python3 analysis/scripts/build_readiness_prompt_population.py spatial-select \
        --plan-dir "$PLAN_ROOT" \
        --candidates "${candidate_files[@]}" \
        --reference-projections "$QWEN_R1_PROJECTIONS" \
        --candidate-projections "$MISTRAL_R1_PROJECTIONS" \
        --robustness-battery "$BATTERY_ROOT" \
        --validations "$VALIDATION_OUTPUT" \
        --generator-ids qwen3-32b,gemma4-31b \
        --next-round-index 2 \
        --distance-tolerance "${READINESS_DISTANCE_TOLERANCE:-0.22}" \
        --disagreement-weight "${READINESS_DISAGREEMENT_WEIGHT:-0.10}" \
        --candidates-per-task "${READINESS_CANDIDATES_PER_TASK:-3}" \
        --master-seed "${READINESS_MASTER_SEED:-20260820}" \
        --output-dir "$SPATIAL_R1"
    complete_spatial_selection
fi

echo "===== CONTINUITY AND ARTIFACT AUDIT ====="
if [[ -e "$AUDIT_R1" ]]; then
    if [[ ! -s "$AUDIT_R1/round1_audit.json" || ! -s "$AUDIT_R1/round1_audit_report.md" ]]; then
        echo "partial immutable audit directory: $AUDIT_R1" >&2
        exit 2
    fi
    echo "AUDIT: COMPLETE; SKIPPING"
else
    mkdir -p "$AUDIT_R1"
    python3 - \
        "$EXPECTED_CANDIDATE_COUNT" \
        "$VALIDATION_OUTPUT" \
        "$QWEN_R1_PROJECTIONS/question_projections.jsonl" \
        "$MISTRAL_R1_PROJECTIONS/question_projections.jsonl" \
        "$COMPARISON_R1/projection_comparison.json" \
        "$SPATIAL_R1/spatial_coverage_diagnostics.json" \
        "$SPATIAL_R1/spatially_selected_questions.jsonl" \
        "$SPATIAL_R1/generation_tasks_round_02.jsonl" \
        "$AUDIT_R1" \
        "${candidate_files[@]}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr

expected = int(sys.argv[1])
validation_path = pathlib.Path(sys.argv[2])
qwen_path = pathlib.Path(sys.argv[3])
mistral_path = pathlib.Path(sys.argv[4])
comparison_path = pathlib.Path(sys.argv[5])
diagnostics_path = pathlib.Path(sys.argv[6])
selected_path = pathlib.Path(sys.argv[7])
next_tasks_path = pathlib.Path(sys.argv[8])
output = pathlib.Path(sys.argv[9])
candidate_paths = [pathlib.Path(value) for value in sys.argv[10:]]

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

candidates = [row for path in candidate_paths for row in rows(path)]
candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
reviews = rows(validation_path)
accepted = sum(bool(row["accepted"]) for row in reviews)

def projection_index(path):
    values = {}
    for row in rows(path):
        candidate_id = str(row["candidate_id"])
        if candidate_id in values:
            raise SystemExit(f"duplicate projection: {candidate_id}")
        if str(row["projection"]["text_sha256"]) != candidate_by_id[candidate_id]["question_sha256"]:
            raise SystemExit(f"projection text mismatch: {candidate_id}")
        values[candidate_id] = row["projection"]
    if set(values) != set(candidate_by_id):
        raise SystemExit("projection candidate set differs from generated candidate set")
    return values

qwen = projection_index(qwen_path)
mistral = projection_index(mistral_path)
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
selected = rows(selected_path)
next_tasks = rows(next_tasks_path)

if not (
    len(candidates) == expected
    and len(reviews) == expected
    and len(qwen) == expected
    and len(mistral) == expected
    and comparison["candidate_count"] == expected
    and diagnostics["candidate_count"] == expected
):
    raise SystemExit("round-1 artifact counts are inconsistent")

def correlation(rows_for_metric, axis):
    target = np.asarray([row[f"target_normalized_axis_{axis}"] for row in rows_for_metric])
    observed = np.asarray([qwen[str(row["candidate_id"])][f"normalized_axis_{axis}"] for row in rows_for_metric])
    result = spearmanr(target, observed)
    return {
        "item_count": len(rows_for_metric),
        "spearman": float(result.statistic),
        "p_value": float(result.pvalue),
    }

rounds = sorted({int(row["round_index"]) for row in candidates})
continuity = {
    "all_candidates": {
        "axis_1": correlation(candidates, 1),
        "axis_2": correlation(candidates, 2),
    },
    "by_round": {
        str(round_index): {
            "axis_1": correlation([row for row in candidates if int(row["round_index"]) == round_index], 1),
            "axis_2": correlation([row for row in candidates if int(row["round_index"]) == round_index], 2),
        }
        for round_index in rounds
    },
}

scale_gate = bool(diagnostics["all_keywords_pass_spacing_gate"])
audit = {
    "format_version": "readiness-prompt-round1-audit-v1",
    "status": "pass" if scale_gate else "refine",
    "artifact_integrity_passed": True,
    "scale_to_30000_gate_passed": scale_gate,
    "candidate_count": len(candidates),
    "accepted_candidate_count": accepted,
    "selected_count": len(selected),
    "next_round_task_count": len(next_tasks),
    "generator_target_continuity_qwen_view": continuity,
    "cross_embedding_agreement": {
        "axis_1_spearman": comparison["axis_1"]["spearman"],
        "axis_2_spearman": comparison["axis_2"]["spearman"],
        "scalar_spearman": comparison["scalar_readiness"]["spearman"],
    },
    "spatial_coverage": diagnostics,
    "scientific_guard": (
        "These coordinates describe generated question semantics. They do not define "
        "the randomized policy variable B and are not causal evidence."
    ),
}
atomic_json(output / "round1_audit.json", audit)

all_continuity = continuity["all_candidates"]
report = f"""# Readiness prompt round-1 audit

- Status: **{'PASS' if scale_gate else 'REFINE'}**
- Artifact integrity: PASS
- Candidate questions: {len(candidates)}
- Independently accepted: {accepted}
- Selected questions: {len(selected)}
- Round-2 refinement cells: {len(next_tasks)}
- Intended target vs Qwen axis 1 Spearman: {all_continuity['axis_1']['spearman']:.4f}
- Intended target vs Qwen axis 2 Spearman: {all_continuity['axis_2']['spearman']:.4f}
- Qwen/Mistral aligned axis 1 Spearman: {comparison['axis_1']['spearman']:.4f}
- Qwen/Mistral aligned axis 2 Spearman: {comparison['axis_2']['spearman']:.4f}
- Qwen/Mistral scalar Spearman: {comparison['scalar_readiness']['spearman']:.4f}
- Scale-to-30,000 spacing gate: {'PASS' if scale_gate else 'BLOCKED PENDING REFINEMENT'}

The continuity correlations measure generator controllability relative to intended
cells. The spatial gate measures the selected bank's actual coverage and spacing.
Coordinates describe prompt semantics; they do not define the randomized policy B.
"""
(output / "round1_audit_report.md").write_text(report, encoding="utf-8")
print(report)
PY
fi

cat "$COMPARISON_R1/projection_comparison_report.md"
cat "$SPATIAL_R1/spatial_coverage_report.md"
cat "$AUDIT_R1/round1_audit_report.md"

audit_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$AUDIT_R1/round1_audit.json")"
echo "PROJECTION_ROOT=$PROJECTION_ROOT"
echo "COMPARISON_ROOT=$COMPARISON_R1"
echo "SPATIAL_ROOT=$SPATIAL_R1"
echo "AUDIT_ROOT=$AUDIT_R1"
if [[ "$audit_status" != "pass" ]]; then
    echo "ROUND-1 AUDIT: REFINE — do not scale to 30,000 yet" >&2
    exit 3
fi
echo "ROUND-1 AUDIT: PASS — spatial scale gate satisfied"
