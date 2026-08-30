#!/bin/bash -l
# Merge completed keyword sections, rerun strict global selection, and freeze
# a new deterministic ten-section continuation plan.

#SBATCH --account=scifi
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=none
# Wall time is supplied by sbatch only after allocation-specific approval.
#SBATCH --job-name=readiness-30k-repartition

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run through sbatch on JUPITER}"
: "${GEODML_EXPECTED_COMMIT:?Export the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the approved merge wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the merge estimate}"
: "${READINESS_SOURCE_TEN_SECTION_RUN_ROOT:?Set the completed ten-section run root}"
: "${READINESS_REPARTITION_ROOT:?Set a fresh repartition output root}"
: "${READINESS_NEW_PARTITION_SALT:?Set the new deterministic partition salt}"

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

export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono-$GEODML_EXPECTED_COMMIT}"
export GEODML_GENERATOR_VENV="${GEODML_GENERATOR_VENV:-$GEODML_CACHE_ROOT/python/.venv-readiness-generators-transformers562}"
export READINESS_30K_PLAN_ROOT="${READINESS_30K_PLAN_ROOT:-$GEODML_PROJECT_ROOT/runs/readiness-30k-axis1/plans/axis1-uniform-001-8b6446f}"
export READINESS_GENERATOR_A_ID="${READINESS_GENERATOR_A_ID:-qwen3-32b}"
export READINESS_GENERATOR_B_ID="${READINESS_GENERATOR_B_ID:-gemma4-31b}"
export READINESS_DISTANCE_TOLERANCE="${READINESS_DISTANCE_TOLERANCE:-0.017}"
export READINESS_DISAGREEMENT_WEIGHT="${READINESS_DISAGREEMENT_WEIGHT:-0.10}"
export READINESS_TEXT_CONTRACT="${READINESS_TEXT_CONTRACT:-question-v1}"
export READINESS_ACCEPTANCE_CONTRACT="${READINESS_ACCEPTANCE_CONTRACT:-question-v1}"
export READINESS_REFINEMENT_CANDIDATES_PER_TASK="${READINESS_REFINEMENT_CANDIDATES_PER_TASK:-4}"
export READINESS_MASTER_SEED="${READINESS_MASTER_SEED:-20260820}"

subspace_pointer="${READINESS_SUBSPACE_POINTER:-$HOME/geodml-readiness-subspace-latest.txt}"
[[ -s "$subspace_pointer" ]]
export READINESS_SUBSPACE_ROOT="${READINESS_SUBSPACE_ROOT:-$(<"$subspace_pointer")}"
export READINESS_BATTERY_ROOT="${READINESS_BATTERY_ROOT:-$READINESS_SUBSPACE_ROOT/robustness/qwen3-vs-mistral7b-976bae5110ec4b985b7c6e7c972bce021b8efdba}"

[[ "${SLURM_JOB_NUM_NODES:-0}" -eq 1 ]]
[[ -d "$READINESS_SOURCE_TEN_SECTION_RUN_ROOT" ]]
[[ -d "$READINESS_REPARTITION_ROOT" ]]
[[ -s "$READINESS_30K_PLAN_ROOT/plan_manifest.json" ]]
[[ -s "$READINESS_BATTERY_ROOT/battery_manifest.json" ]]
[[ -x "$GEODML_GENERATOR_VENV/bin/python" ]]
[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]

activate_control_runtime
cd "$GEODML_REPOSITORY"

source_root="$(realpath "$READINESS_SOURCE_TEN_SECTION_RUN_ROOT")"
repartition_root="$(realpath "$READINESS_REPARTITION_ROOT")"
final_root="$repartition_root/checkpoint"
section_plan="$repartition_root/keyword-section-plan.json"
audit_report="$repartition_root/fully-compliant-audit.json"

exec 9>"$repartition_root/.repartition.lock"
flock -n 9 || {
    echo "another merge/repartition job owns $repartition_root" >&2
    exit 2
}

mapfile -t partition_roots < <(
    find "$source_root" -mindepth 1 -maxdepth 1 \
        -type d -name 'section-*-of-10' -print | sort -V
)
[[ "${#partition_roots[@]}" -eq 10 ]] || {
    echo "expected exactly ten source section roots; found ${#partition_roots[@]}" >&2
    exit 2
}
for index in "${!partition_roots[@]}"; do
    [[ "$(basename "${partition_roots[$index]}")" == "section-$index-of-10" ]]
    [[ -s "${partition_roots[$index]}/pipeline_manifest.json" ]]
    latest_summary="$({
        find "${partition_roots[$index]}" -mindepth 2 -maxdepth 2 \
            -path '*/round-*/verified_round_summary.json' -type f -print
    } | sort -V | tail -n 1)"
    [[ -s "$latest_summary" ]]
done

if [[ ! -s "$final_root/verified_round_summary.json" ]]; then
    [[ ! -e "$final_root" ]] || {
        echo "partial final checkpoint already exists: $final_root" >&2
        exit 2
    }
    attempt="$repartition_root/.checkpoint-attempt-$SLURM_JOB_ID-${BASHPID:-$$}"
    [[ ! -e "$attempt" ]]
    mkdir -p "$attempt"

    merge_arguments=()
    for partition_root in "${partition_roots[@]}"; do
        merge_arguments+=(--partition-root "$partition_root")
    done
    python analysis/scripts/merge_readiness_partition_checkpoints.py \
        "${merge_arguments[@]}" \
        --output-dir "$attempt/merged"

    python - "$attempt/merged/merge_manifest.json" <<'PY'
import json
import os
import sys

manifest = json.load(open(sys.argv[1]))
expected = (
    os.environ["READINESS_TEXT_CONTRACT"],
    os.environ["READINESS_ACCEPTANCE_CONTRACT"],
)
observed = (
    manifest.get("text_contract", "question-v1"),
    manifest.get("acceptance_contract_version", "question-v1"),
)
if observed != expected:
    raise SystemExit(
        f"merged contracts {observed!r} differ from requested {expected!r}"
    )
PY

    set +e
    python analysis/scripts/build_readiness_prompt_population.py audit-diversity \
        --questions "$attempt/merged/candidates.jsonl" \
        --text-contract "$READINESS_TEXT_CONTRACT" \
        --output-dir "$attempt/raw-diversity"
    raw_diversity_exit=$?
    set -e
    printf '%s\n' "$raw_diversity_exit" > "$attempt/raw-diversity-exit-code.txt"

    python analysis/scripts/build_readiness_prompt_population.py compare-projections \
        --reference-projections "$attempt/merged/projections/qwen" \
        --candidate-projections "$attempt/merged/projections/mistral" \
        --robustness-battery "$READINESS_BATTERY_ROOT" \
        --output-dir "$attempt/comparison"

    next_round_index="$(python -c '
import json, sys
print(json.load(open(sys.argv[1]))["maximum_candidate_round_index"] + 1)
' "$attempt/merged/merge_manifest.json")"

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
        --text-contract "$READINESS_TEXT_CONTRACT" \
        --acceptance-contract "$READINESS_ACCEPTANCE_CONTRACT" \
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
            --text-contract "$READINESS_TEXT_CONTRACT" \
            --output-dir "$attempt/selected-diversity"
        selected_diversity_exit=$?
        set -e
    fi
    printf '%s\n' "$selected_diversity_exit" \
        > "$attempt/selected-diversity-exit-code.txt"

    python - "$attempt" "$selected_diversity_exit" "$source_root" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
diversity_exit = int(sys.argv[2])
source_root = str(pathlib.Path(sys.argv[3]).resolve())
merged = json.loads((root / "merged/merge_manifest.json").read_text())
validation = json.loads((root / "merged/validation.jsonl.manifest.json").read_text())
selection = json.loads((root / "strict-selection/run_manifest.json").read_text())
diagnostics = json.loads(
    (root / "strict-selection/spatial_coverage_diagnostics.json").read_text()
)
summary = {
    "format_version": "readiness-30k-keyword-repartition-v1",
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "source_git_commit_sha": merged["git_commit_sha"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "source_ten_section_run_root": source_root,
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
    "text_contract": selection.get("text_contract", "question-v1"),
    "acceptance_contract_version": selection.get(
        "acceptance_contract_version", "question-v1"
    ),
    "selected_diversity_gate_passed": diversity_exit == 0,
    "spacing_gate_passed": diagnostics["overall_spacing_gate_passed"],
    "partition_count": merged["partition_count"],
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

    ln "$attempt/merged/candidates.jsonl" "$attempt/candidates.jsonl"
    ln "$attempt/merged/candidates.jsonl.manifest.json" \
        "$attempt/candidates.jsonl.manifest.json"
    ln "$attempt/merged/validation.jsonl" "$attempt/validation.jsonl"
    ln "$attempt/merged/validation.jsonl.manifest.json" \
        "$attempt/validation.jsonl.manifest.json"
    ln -s merged/projections "$attempt/projections"
    printf '%s\n' "$final_root/candidates.jsonl" > "$attempt/candidate-files.txt"
    mv "$attempt" "$final_root"
fi

if [[ ! -s "$audit_report" ]]; then
    python analysis/scripts/audit_fully_compliant_readiness_prompts.py \
        --final-root "$final_root" \
        --report-file "$audit_report"
fi
python - "$audit_report" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
if not report["audit_passed"]:
    raise SystemExit(
        "merged checkpoint failed the independent fully compliant prompt audit"
    )
print(f"merged_fully_compliant_prompts={report['fully_compliant_prompt_count']}")
PY

if [[ ! -s "$section_plan" ]]; then
    python analysis/scripts/prepare_readiness_keyword_sections.py \
        --checkpoint-root "$final_root" \
        --output "$section_plan" \
        --section-count 10 \
        --partition-salt "$READINESS_NEW_PARTITION_SALT"
else
    python analysis/scripts/prepare_readiness_keyword_sections.py \
        --verify-plan "$section_plan"
fi

if [[ ! -s "$repartition_root/run_manifest.json" ]]; then
    python - "$repartition_root/run_manifest.json" "$final_root" \
        "$section_plan" "$source_root" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2]).resolve()
plan = pathlib.Path(sys.argv[3]).resolve()
source = pathlib.Path(sys.argv[4]).resolve()
payload = {
    "format_version": "readiness-30k-ten-section-repartition-run-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "source_ten_section_run_root": str(source),
    "checkpoint_root": str(checkpoint),
    "keyword_section_plan": str(plan),
    "keyword_section_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    "partition_salt": os.environ["READINESS_NEW_PARTITION_SALT"],
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
fi

printf '%s\n' "$final_root" > "$repartition_root/checkpoint-latest.txt"
printf '%s\n' "$section_plan" > "$repartition_root/keyword-section-plan-latest.txt"
echo "REPARTITION COMPLETE"
echo "checkpoint=$final_root"
echo "section_plan=$section_plan"
