#!/bin/bash
# Submit one approved merge/repartition job followed by ten approved,
# dependency-gated four-GPU keyword-section jobs.

set -euo pipefail
umask 077

source_root="${1:?Usage: $0 SOURCE_TEN_SECTION_RUN_ROOT}"
: "${READINESS_MERGE_APPROVED_WALLTIME:?Set the approved merge wall time}"
: "${READINESS_MERGE_ALLOCATION_ESTIMATE:?Record the merge estimate}"
: "${READINESS_SECTION_APPROVED_WALLTIME:?Set the approved section wall time}"
: "${READINESS_SECTION_ALLOCATION_ESTIMATE:?Record the section estimate}"

command -v sbatch >/dev/null
command -v jutil >/dev/null
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

repository="$(git rev-parse --show-toplevel)"
expected_commit="$(git -C "$repository" rev-parse HEAD)"
[[ -z "$(git -C "$repository" status --porcelain)" ]] || {
    echo "submission requires a clean exact-commit checkout" >&2
    exit 2
}

source_root="$(realpath "$source_root")"
[[ -d "$source_root" ]]
[[ "$(find "$source_root" -mindepth 1 -maxdepth 1 -type d \
    -name 'section-*-of-10' | wc -l)" -eq 10 ]]

run_id="${READINESS_REPARTITION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
axis_root="$(dirname "$source_root")"
repartition_root="${READINESS_REPARTITION_ROOT:-$axis_root/repartition-${expected_commit:0:8}-$run_id}"
[[ ! -e "$repartition_root" ]] || {
    echo "refusing to reuse repartition root: $repartition_root" >&2
    exit 2
}
mkdir -p "$repartition_root/logs" "$repartition_root/continuation"

section_plan="$repartition_root/keyword-section-plan.json"
continuation_root="$repartition_root/continuation"
partition_salt="${READINESS_NEW_PARTITION_SALT:-axis1-30330-ten-sections-${expected_commit:0:8}-$run_id}"
request_manifest="$repartition_root/submission_request.json"
submitted_jobs="$repartition_root/submitted-job-ids.tsv"

export GEODML_EXPECTED_COMMIT="$expected_commit"
export GEODML_REPOSITORY="$repository"
export READINESS_MERGE_APPROVED_WALLTIME
export READINESS_MERGE_ALLOCATION_ESTIMATE
export READINESS_SECTION_APPROVED_WALLTIME
export READINESS_SECTION_ALLOCATION_ESTIMATE
export READINESS_SOURCE_TEN_SECTION_RUN_ROOT="$source_root"
export READINESS_REPARTITION_ROOT="$repartition_root"
export READINESS_NEW_PARTITION_SALT="$partition_salt"

python3 - "$request_manifest" "$source_root" "$repartition_root" \
    "$section_plan" "$continuation_root" "$partition_salt" <<'PY'
from datetime import datetime, timezone
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "format_version": "readiness-30k-repartition-submission-request-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "git_commit_sha": os.environ["GEODML_EXPECTED_COMMIT"],
    "source_ten_section_run_root": str(pathlib.Path(sys.argv[2]).resolve()),
    "repartition_root": str(pathlib.Path(sys.argv[3]).resolve()),
    "keyword_section_plan": str(pathlib.Path(sys.argv[4]).resolve()),
    "continuation_root": str(pathlib.Path(sys.argv[5]).resolve()),
    "partition_salt": sys.argv[6],
    "merge_allocation": {
        "approved_walltime": os.environ["READINESS_MERGE_APPROVED_WALLTIME"],
        "estimate": os.environ["READINESS_MERGE_ALLOCATION_ESTIMATE"],
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 32,
        "memory": "128G",
        "gres": "none",
    },
    "section_allocations": {
        "approved_walltime_each": os.environ[
            "READINESS_SECTION_APPROVED_WALLTIME"
        ],
        "estimate": os.environ["READINESS_SECTION_ALLOCATION_ESTIMATE"],
        "job_count": 10,
        "nodes_per_job": 1,
        "gpus_per_job": 4,
        "cpus_per_job": 32,
        "maximum_gpu_hours": 320,
    },
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY

merge_job_raw="$(
    READINESS_APPROVED_WALLTIME="$READINESS_MERGE_APPROVED_WALLTIME" \
    READINESS_ALLOCATION_ESTIMATE="$READINESS_MERGE_ALLOCATION_ESTIMATE" \
    sbatch --parsable \
        --export=ALL \
        --time="$READINESS_MERGE_APPROVED_WALLTIME" \
        --output="$repartition_root/logs/merge-%j.out" \
        --error="$repartition_root/logs/merge-%j.err" \
        "$repository/analysis/scripts/slurm/jupiter/run_readiness_30k_repartition_keyword_sections.sbatch"
)"
merge_job_id="${merge_job_raw%%;*}"
[[ "$merge_job_id" =~ ^[0-9]+$ ]]
printf 'merge\t%s\n' "$merge_job_id" > "$submitted_jobs"

export READINESS_KEYWORD_SECTION_PLAN="$section_plan"
export READINESS_TEN_SECTION_RUN_ROOT="$continuation_root"
section_job_ids=()
for index in {0..9}; do
    section_job_raw="$(
        READINESS_WORK_PARTITION_INDEX="$index" \
        READINESS_APPROVED_WALLTIME="$READINESS_SECTION_APPROVED_WALLTIME" \
        READINESS_ALLOCATION_ESTIMATE="$READINESS_SECTION_ALLOCATION_ESTIMATE" \
        sbatch --parsable \
            --export=ALL \
            --dependency="afterok:$merge_job_id" \
            --time="$READINESS_SECTION_APPROVED_WALLTIME" \
            --output="$repartition_root/logs/section-$index-%j.out" \
            --error="$repartition_root/logs/section-$index-%j.err" \
            "$repository/analysis/scripts/slurm/jupiter/run_readiness_30k_axis1_keyword_section.sbatch"
    )"
    section_job_id="${section_job_raw%%;*}"
    [[ "$section_job_id" =~ ^[0-9]+$ ]]
    section_job_ids+=("$section_job_id")
    printf 'section-%s\t%s\n' "$index" "$section_job_id" >> "$submitted_jobs"
done

python3 - "$request_manifest" "$repartition_root/submission_manifest.json" \
    "$merge_job_id" "${section_job_ids[@]}" <<'PY'
from datetime import datetime, timezone
import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[1]).read_text())
request.update(
    {
        "format_version": "readiness-30k-repartition-submission-v1",
        "submitted_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "merge_job_id": sys.argv[3],
        "section_job_ids": sys.argv[4:],
        "dependency": f"afterok:{sys.argv[3]}",
    }
)
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(request, indent=2, sort_keys=True) + "\n"
)
PY

echo "SUBMISSION COMPLETE"
echo "merge_job=$merge_job_id"
echo "section_jobs=${section_job_ids[*]}"
echo "repartition_root=$repartition_root"
echo "submission_manifest=$repartition_root/submission_manifest.json"
