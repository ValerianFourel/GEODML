#!/bin/bash -l
# Target unresolved action-ready readiness cells inside an approved GPU allocation.
# This harness never requests or creates a Slurm allocation.
# Prompt embeddings diagnose generated text; they do not define randomized policy B.

set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?Run inside a specifically approved Slurm allocation}"
: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${READINESS_APPROVED_WALLTIME:?Record the specifically approved wall time}"
: "${READINESS_ALLOCATION_ESTIMATE:?Record the estimate supporting the allocation}"
: "${READINESS_SEARCH_TRIGGER_COUNTERFACTUAL_ROOT:?Set the completed existing-candidate counterfactual root}"
: "${READINESS_30K_PIPELINE_ROOT:?Set a persistent high-axis pipeline root}"

clear_inherited_python_runtime() {
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

load_control_stack() {
    if ! type module >/dev/null 2>&1; then
        set +u
        source /etc/profile
        set -u
    fi
    module --force purge
    module load Stages/2026
    module load GCCcore/14.3.0
    module load SciPy-Stack/2025b
    module load git
    module load PyTorch/2.9.1
    jutil env activate -p "${JUPITER_PROJECT:-scifi}"
    hash -r
    python3 -c 'import json, pathlib; print("HIGH_AXIS_CONTROL_RUNTIME=PASS")'
}

# `bash "$HARNESS"` ignores the login-shell shebang. Compute shells are also
# commonly opened with --noprofile --norc, so bootstrap Python before preflight.
clear_inherited_python_runtime
load_control_stack

export READINESS_GENERATION_PROFILE="high-axis-action-v1"
export READINESS_REFINEMENT_MIN_TARGET_AXIS_1="${READINESS_REFINEMENT_MIN_TARGET_AXIS_1:-0.700}"
export READINESS_REFINEMENT_TASK_PRIORITY="descending-axis-1"

counterfactual_root="$(realpath "$READINESS_SEARCH_TRIGGER_COUNTERFACTUAL_ROOT")"
counterfactual_summary="$counterfactual_root/counterfactual_summary.json"
baseline_selected="$counterfactual_root/scenarios/search_trigger_v2_relaxed_tolerance/selected.jsonl"
test -s "$counterfactual_summary"
test -s "$baseline_selected"
export READINESS_HIGH_AXIS_BASELINE_SELECTED="$baseline_selected"

python3 - \
    "$counterfactual_summary" \
    "$baseline_selected" \
    "$READINESS_REFINEMENT_MIN_TARGET_AXIS_1" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = sum(1 for line in pathlib.Path(sys.argv[2]).open(encoding="utf-8") if line.strip())
minimum = float(sys.argv[3])
scenario = summary["scenarios"]["search_trigger_v2_relaxed_tolerance"]
if not 0.70 <= minimum <= 1.0:
    raise SystemExit("high-axis minimum must lie in [0.70, 1]")
if float(summary["relaxed_distance_tolerance"]) != 0.035:
    raise SystemExit("high-axis harness requires the audited 0.035 counterfactual")
if selected != int(scenario["selected_count"]):
    raise SystemExit("counterfactual selected file disagrees with its summary")
print(
    "HIGH_AXIS_COUNTERFACTUAL=PASS "
    f"baseline_selected={selected} missing={scenario['missing_count']} "
    f"minimum_axis_1={minimum:.3f}"
)
PY

mkdir -p "$READINESS_30K_PIPELINE_ROOT"
harness_manifest="$READINESS_30K_PIPELINE_ROOT/high-axis-action-harness.json"
python3 - "$harness_manifest" "$counterfactual_summary" "$baseline_selected" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
summary = Path(sys.argv[2]).resolve()
selected = Path(sys.argv[3]).resolve()
value = {
    "format_version": "readiness-high-axis-action-harness-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ["READINESS_APPROVED_WALLTIME"],
    "allocation_estimate": os.environ["READINESS_ALLOCATION_ESTIMATE"],
    "text_contract": "search-trigger-v2",
    "acceptance_contract_version": "search-trigger-v2",
    "generation_profile": "high-axis-action-v1",
    "distance_tolerance": 0.035,
    "minimum_target_axis_1": float(
        os.environ["READINESS_REFINEMENT_MIN_TARGET_AXIS_1"]
    ),
    "task_priority": "descending-axis-1",
    "counterfactual_summary": {
        "path": str(summary),
        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    },
    "baseline_selected": {
        "path": str(selected),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    },
    "action_search_boundary": (
        "Triggers request web-findable instructions needed for imminent action; "
        "they do not claim that the search system already executed the action."
    ),
    "scientific_guard": (
        "The generation profile changes only readiness-stage realization. Prompt "
        "embeddings diagnose generated text and do not define randomized policy B."
    ),
}
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    stable = set(value) - {"created_at", "slurm_job_id", "approved_walltime", "allocation_estimate"}
    if any(previous.get(key) != value[key] for key in stable):
        raise SystemExit("existing high-axis harness manifest has a different identity")
else:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY

echo "SEARCH-TRIGGER-V2 HIGH-AXIS ACTION HARNESS"
echo "profile=$READINESS_GENERATION_PROFILE"
echo "minimum_axis_1=$READINESS_REFINEMENT_MIN_TARGET_AXIS_1"
echo "priority=$READINESS_REFINEMENT_TASK_PRIORITY"
echo "baseline_selected=$READINESS_HIGH_AXIS_BASELINE_SELECTED"

driver="${GEODML_REPOSITORY:?Set the exact repository checkout}/analysis/scripts/slurm/jupiter/run_readiness_30k_search_trigger_v2.sh"
test -s "$driver"
# Preserve the module stack bootstrapped above. Executing the file directly
# would honor its login-shell shebang and could replace LD_LIBRARY_PATH before
# the v2 driver's first Python preflight.
exec bash "$driver"
