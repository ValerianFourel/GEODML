#!/usr/bin/env bash
# Allocation is supplied explicitly by the user-facing submission command.
set -eo pipefail
repo=${1:?repository}
project=${2:?project root}
control=${3:?control directory}
expected_commit=${4:?pinned commit}
environment_file=${5:?environment file}
source "$environment_file"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"
set -u
umask 077
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
pilot="$project/runs/acl-arr-search-experience/pilot-3-627ad348c2a0"
selection="$project/runs/acl-arr-search-experience/pilot-500-axis-balanced-a559b7058277/selection-manifest.json"
mkdir -p "$control/provenance"
trap 'rc=$?; printf "JOB_EXIT=%s\n" "$rc" > "$control/job-status.txt"' EXIT
printf '%s\n' "$expected_commit" > "$control/provenance/source-commit.txt"
date -Is > "$control/provenance/started-at.txt"
hostname > "$control/provenance/host.txt"
scontrol show job "${SLURM_JOB_ID:?}" > "$control/provenance/allocation.txt"
printf 'approved_walltime=01:00:00\nnodes=1\ngpus=4\nmaximum_gpu_hours=4\nestimate=20-60 minutes or longer with filesystem stalls; based on prior 437078 files/24G pilot and 89G total project runs\nno_inference=true\nautomatic_resubmission=false\n' > "$control/provenance/budget.txt"
git -C "$repo" archive --format=tar HEAD > "$control/provenance/source.tar"
python3 -m pip freeze > "$control/provenance/python-packages.txt"
(df -h "$project" "$control"; df -i "$project" "$control") > "$control/provenance/storage.txt"
# This established pilot validator remains a separate exact audit.
timeout --signal=TERM --kill-after=15s 300s python3 "$repo/analysis/scripts/report_agentic_500_pilot_results.py" \
  --judge-run "$pilot/agentic-judge-original500-nemotron-seven-hour-b35ab032b3d6-v4" --json \
  > "$control/provenance/original500.json" 2> "$control/pilot-audit.log" || \
  printf 'Original-500 validator did not finish successfully; see pilot-audit.log\n' > "$control/provenance/original500-audit-failed.txt"
# Leave five minutes before Slurm expiry. Do not extend or resubmit on timeout.
end_time=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')
seconds_left=$(( $(date -d "$end_time" +%s) - $(date +%s) - 300 ))
test "$seconds_left" -gt 0
population_root=$(python3 - "$selection" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print((p.parent / json.loads(p.read_text())['sources']['prompts']['path']).resolve().parent)
PY
)
timeout --signal=TERM --kill-after=30s "${seconds_left}s" \
  python3 "$repo/analysis/scripts/build_agentic_recovery_dataset.py" build \
  --root "$project/runs" --root "$project/exports" --root "$project/manifests" \
  --root "$population_root" --root "$control/provenance" \
  --selection-manifest "$selection" --output "$control/dataset"
printf 'DATASET_READY=%s\n' "$control/dataset/hub"
