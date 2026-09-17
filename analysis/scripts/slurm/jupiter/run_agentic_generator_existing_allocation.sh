#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_agentic_generator_existing_allocation.sh prepare|launch JOB_ID MODEL_SLUG RESUME_RUN_ROOT RUN_ROOT APPROVED_WALLTIME MAXIMUM_GPU_HOURS
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" != "prepare" && "${1:-}" != "launch" ]]; then
  printf '%s\n' 'First argument must be prepare or launch.' >&2
  exit 64
fi
if [[ "$#" -ne 7 ]]; then
  usage >&2
  exit 64
fi

action="$1"
job_id="$2"
model_slug="$3"
resume_root="$4"
run_root="$5"
approved_walltime="$6"
maximum_gpu_hours="$7"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
manifest="$run_root/run_manifest.json"

source "${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"

json_value() {
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); [value:=value[key] for key in sys.argv[2:]]; print(value)' "$@"
}

allocation_estimate="Existing allocation ${job_id}: one exclusive Booster node, four GH200 GPUs, 32 requested CPUs, 512G RAM, approved wall-time ${approved_walltime}, maximum ${maximum_gpu_hours} GPU-hours. The actual Slurm end time limits work. No retry, requeue, extension, or replacement."
export GEODML_EXECUTION_REPOSITORY="$repository"
export GEODML_EXECUTION_COMMIT="$(git -C "$repository" rev-parse HEAD)"
export GEODML_APPROVED_WALLTIME="$approved_walltime"
export GEODML_MAXIMUM_TOTAL_GPU_HOURS="$maximum_gpu_hours"
export GEODML_ALLOCATION_ESTIMATE="$allocation_estimate"
export GEODML_START_MARGIN_SECONDS="120"
export GEODML_CLEANUP_MARGIN_SECONDS="45"

if [[ "$action" == "prepare" ]]; then
  [[ "$(hostname -s)" == jpbl-* ]] || {
    printf 'Preparation requires a jpbl login node, got %s\n' "$(hostname -s)" >&2
    exit 64
  }
  test "$(squeue --noheader --jobs="$job_id" --states=RUNNING --format='%A' | tr -d ' ')" = "$job_id"
  test -s "$resume_root/run_manifest.json"

  source_manifest="$resume_root/run_manifest.json"
  cohort_root="$(json_value "$source_manifest" request cohort_root)"
  profile="$(json_value "$source_manifest" request models "$model_slug" profile)"
  account="$(json_value "$source_manifest" request account)"
  partition="$(json_value "$source_manifest" request partition)"
  export SEARCH_AGENTIC_DDG_SNAPSHOT="$(find "$resume_root/search" -maxdepth 1 -type f -name 'duckduckgo.*' -print -quit)"
  export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$(find "$resume_root/search" -maxdepth 1 -type f -name 'searxng.*' -print -quit)"
  test -s "$profile"
  test -s "$SEARCH_AGENTIC_DDG_SNAPSHOT"
  test -s "$SEARCH_AGENTIC_SEARXNG_SNAPSHOT"

  cd "$repository"
  python3 analysis/scripts/submit_agentic_generator_backlog.py \
    --cohort-root "$cohort_root" \
    --run-root "$run_root" \
    --profile-root "$(dirname "$profile")" \
    --source-git-commit "$GEODML_EXECUTION_COMMIT" \
    --account "$account" \
    --partition "$partition" \
    --approved-walltime "$approved_walltime" \
    --workers-per-model 1 \
    --maximum-total-gpu-hours "$maximum_gpu_hours" \
    --resume-from-run-root "$resume_root" \
    --model "$model_slug" \
    --prepare-only
  printf 'EXISTING_ALLOCATION_PREPARE=PASS job=%s model=%s run=%s\n' "$job_id" "$model_slug" "$run_root"
  exit 0
fi

[[ "$(hostname -s)" == jpbo-* ]] || {
  printf 'Launch requires the existing jpbo srun shell, got %s\n' "$(hostname -s)" >&2
  exit 64
}
test "${SLURM_JOB_ID:?}" = "$job_id"
test "${SLURM_JOB_NUM_NODES:?}" = "1"
: "${SLURM_JOB_CPUS_PER_NODE:?}"
: "${SLURM_JOB_END_TIME:?}"
test -s "$manifest"
test "$(json_value "$manifest" status)" = "prepared"

cross_snapshot="$(json_value "$manifest" request runtime SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT)"
cross_revision="$(json_value "$manifest" request runtime SEARCH_AGENTIC_CROSS_ENCODER_REVISION)"
claim_root="$(json_value "$manifest" claim_root)"
prompt_count="$(json_value "$manifest" prompt_count)"
prompt_seed="$(json_value "$manifest" prompt_selection_seed)"
ddg_snapshot="$(find "$run_root/search" -maxdepth 1 -type f -name 'duckduckgo.*' -print -quit)"
searxng_snapshot="$(find "$run_root/search" -maxdepth 1 -type f -name 'searxng.*' -print -quit)"
test -s "$cross_snapshot"
test -s "$ddg_snapshot"
test -s "$searxng_snapshot"

logs="$run_root/models/$model_slug/logs"
stdout="$logs/srun-${job_id}.out"
stderr="$logs/srun-${job_id}.err"
export GEODML_MODEL_SLUG="$model_slug"
export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY="1"
export GEODML_WAVE_ROOT="$run_root/models/$model_slug/wave"
export GEODML_WAVE_OUTPUT_ROOT="$run_root/models/$model_slug/outputs"
export GEODML_WAVE_LOG_ROOT="$logs"
export GEODML_INFERENCE_CLAIM_ROOT="$claim_root"
export GEODML_WORKER_LAUNCHER="$repository/analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"
export GEODML_WORKER_STDOUT="$stdout"
export GEODML_WORKER_STDERR="$stderr"
export SEARCH_AGENTIC_PROFILE="$run_root/profiles/$model_slug.json"
export SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT="$cross_snapshot"
export SEARCH_AGENTIC_CROSS_ENCODER_REVISION="$cross_revision"
export SEARCH_AGENTIC_DDG_SNAPSHOT="$ddg_snapshot"
export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$searxng_snapshot"
export SEARCH_AGENTIC_PROMPTS_JSONL="$run_root/cohort/pilot-prompts.jsonl"
export SEARCH_AGENTIC_SELECTION_RECORDS_JSONL="$run_root/cohort/selection-records.jsonl"
export SEARCH_AGENTIC_PROMPT_COUNT="$prompt_count"
export SEARCH_AGENTIC_PROMPT_SELECTION_SEED="$prompt_seed"
export SEARCH_AGENTIC_PROMPT_SHARD_INDEX="0"
export SEARCH_AGENTIC_PROMPT_SHARD_COUNT="1"
export SEARCH_AGENTIC_PRODUCTION_CONDITIONS="1"
export SEARCH_AGENTIC_REQUEST_CONCURRENCY="4"
export SEARCH_AGENTIC_CELL_CONCURRENCY="12"

printf 'RUN_ROOT=%s\nSTDOUT=%s\nSTDERR=%s\n' "$run_root" "$stdout" "$stderr"
"$repository/analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch" >"$stdout" 2>"$stderr"
