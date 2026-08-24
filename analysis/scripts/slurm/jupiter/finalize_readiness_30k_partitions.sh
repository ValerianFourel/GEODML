#!/bin/bash -l
# Run by the last successful producer inside either one-node GPU allocation.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside one of the approved partition allocations}"
: "${GEODML_REPOSITORY:?GEODML_REPOSITORY is required}"
: "${GEODML_EXPECTED_COMMIT:?GEODML_EXPECTED_COMMIT is required}"
: "${GEODML_GENERATOR_VENV:?GEODML_GENERATOR_VENV is required}"
: "${READINESS_PARTITION_COORDINATION_ROOT:?coordination root is required}"
: "${READINESS_PARTITION_ROOT_0:?partition root 0 is required}"
: "${READINESS_PARTITION_ROOT_1:?partition root 1 is required}"
: "${READINESS_30K_PLAN_ROOT:?plan root is required}"
: "${READINESS_BATTERY_ROOT:?robustness battery is required}"
: "${READINESS_GENERATOR_A_ID:?generator A id is required}"
: "${READINESS_GENERATOR_B_ID:?generator B id is required}"
: "${READINESS_DISTANCE_TOLERANCE:?distance tolerance is required}"
: "${READINESS_DISAGREEMENT_WEIGHT:?disagreement weight is required}"
: "${READINESS_REFINEMENT_CANDIDATES_PER_TASK:?candidate count is required}"
: "${READINESS_MASTER_SEED:?master seed is required}"

coordination_root="$READINESS_PARTITION_COORDINATION_ROOT"
marker_0="$coordination_root/producer-0.ready.json"
marker_1="$coordination_root/producer-1.ready.json"
mkdir -p "$coordination_root"

if [[ ! -s "$marker_0" || ! -s "$marker_1" ]]; then
    echo "PARTITION FINALIZER DEFERRED: both producer markers are not ready"
    exit 0
fi

final_run_id="$(python3 - "$marker_0" "$marker_1" "$GEODML_EXPECTED_COMMIT" <<'PY'
import json
import pathlib
import sys

rows = [json.loads(pathlib.Path(value).read_text()) for value in sys.argv[1:3]]
expected_commit = sys.argv[3]
if [int(row["partition_index"]) for row in rows] != [0, 1]:
    raise SystemExit("producer markers are not ordered partition 0 and 1")
if any(row["partition_count"] != 2 for row in rows):
    raise SystemExit("producer marker partition counts differ")
if len({row["partition_salt"] for row in rows}) != 1:
    raise SystemExit("producer marker partition salts differ")
if any(row["git_commit_sha"] != expected_commit for row in rows):
    raise SystemExit("producer marker commit differs from finalizer commit")
print("-".join(str(row["slurm_job_id"]) for row in rows))
PY
)"
final_root="$coordination_root/final-jobs-$final_run_id"

command -v flock >/dev/null
exec 9>"$coordination_root/.finalizer.lock"
if ! flock -n 9; then
    echo "PARTITION FINALIZER: peer job owns the finalizer lock"
    exit 0
fi
if [[ -s "$final_root/verified_round_summary.json" ]]; then
    echo "PARTITION FINALIZER: final union already exists at $final_root"
    exit 0
fi

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

clear_runtime
load_stack
activate_control_runtime
cd "$GEODML_REPOSITORY"
[[ "$(git rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]

attempt="$coordination_root/.final-attempt-$SLURM_JOB_ID-${BASHPID:-$$}-$(date -u +%Y%m%dT%H%M%SZ)"
[[ ! -e "$attempt" ]]
mkdir -p "$attempt"

python analysis/scripts/merge_readiness_partition_checkpoints.py \
    --partition-root "$READINESS_PARTITION_ROOT_0" \
    --partition-root "$READINESS_PARTITION_ROOT_1" \
    --output-dir "$attempt/merged"

set +e
python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
    --questions "$attempt/merged/candidates.jsonl" \
    --output-dir "$attempt/raw-diversity"
raw_diversity_exit=$?
set -e
printf '%s\n' "$raw_diversity_exit" > "$attempt/raw-diversity-exit-code.txt"

python analysis/scripts/build_readiness_prompt_population.py compare-projections \
    --reference-projections "$attempt/merged/projections/qwen" \
    --candidate-projections "$attempt/merged/projections/mistral" \
    --robustness-battery "$READINESS_BATTERY_ROOT" \
    --output-dir "$attempt/comparison"

next_round_index="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["maximum_candidate_round_index"] + 1)' "$attempt/merged/merge_manifest.json")"
python analysis/scripts/build_readiness_prompt_population.py spatial-select \
    --plan-dir "$READINESS_30K_PLAN_ROOT" \
    --candidates "$attempt/merged/candidates.jsonl" \
    --reference-projections "$attempt/merged/projections/qwen" \
    --candidate-projections "$attempt/merged/projections/mistral" \
    --robustness-battery "$READINESS_BATTERY_ROOT" \
    --validations "$attempt/merged/validation.jsonl" \
    --generator-ids "$READINESS_GENERATOR_A_ID,$READINESS_GENERATOR_B_ID" \
    --next-round-index "$next_round_index" \
    --distance-tolerance "$READINESS_DISTANCE_TOLERANCE" \
    --require-both-views-within-tolerance \
    --require-delexicalized-template-uniqueness \
    --disagreement-weight "$READINESS_DISAGREEMENT_WEIGHT" \
    --candidates-per-task "$READINESS_REFINEMENT_CANDIDATES_PER_TASK" \
    --master-seed "$READINESS_MASTER_SEED" \
    --output-dir "$attempt/strict-selection"

selected="$attempt/strict-selection/spatially_selected_questions.jsonl"
selected_diversity_exit=2
if [[ -s "$selected" ]]; then
    set +e
    python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
        --questions "$selected" \
        --output-dir "$attempt/selected-diversity"
    selected_diversity_exit=$?
    set -e
fi
printf '%s\n' "$selected_diversity_exit" > "$attempt/selected-diversity-exit-code.txt"

python - "$attempt" "$selected_diversity_exit" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
diversity_exit = int(sys.argv[2])
merged = json.loads((root / "merged/merge_manifest.json").read_text())
validation = json.loads((root / "merged/validation.jsonl.manifest.json").read_text())
selection = json.loads((root / "strict-selection/run_manifest.json").read_text())
diagnostics = json.loads(
    (root / "strict-selection/spatial_coverage_diagnostics.json").read_text()
)
summary = {
    "format_version": "readiness-30k-two-partition-final-v1",
    "candidate_count": merged["candidate_count"],
    "independently_accepted_count": validation["accepted_count"],
    "selected_count": selection["selected_count"],
    "refinement_task_count": selection["next_round_task_count"],
    "strict_dual_view_contract_enabled": selection[
        "coordinate_acceptance_contract"
    ]["enabled"],
    "delexicalized_template_uniqueness_enabled": selection[
        "surface_acceptance_contract"
    ]["enabled"],
    "selected_diversity_gate_passed": diversity_exit == 0,
    "spacing_gate_passed": diagnostics["overall_spacing_gate_passed"],
    "partition_count": 2,
    "verified_population_passed": (
        selection["next_round_task_count"] == 0
        and diversity_exit == 0
        and diagnostics["overall_spacing_gate_passed"]
    ),
}
(root / "verified_round_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
(root / "status.txt").write_text(
    ("pass" if summary["verified_population_passed"] else "refine") + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

mv "$attempt" "$final_root"
printf '%s\n' "$final_root" > "$coordination_root/final-latest.txt"
echo "PARTITION FINALIZER COMPLETE: $final_root"
