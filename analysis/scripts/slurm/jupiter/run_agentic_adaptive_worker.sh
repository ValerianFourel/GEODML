#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: %s PLAN_JSON WORKER_INDEX\n' "$0" >&2
  exit 64
fi
plan="$(realpath "$1")"
worker_index="$2"
[[ "$worker_index" =~ ^[0-4]$ ]] || { printf '%s\n' 'Worker index must be 0..4.' >&2; exit 64; }
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"

source "${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"
cd "$repository"

test "${SLURM_JOB_ID:?}" = "${GEODML_EXPECTED_JOB_ID:?}"
test "${SLURM_JOB_NUM_NODES:?}" = 5
test "${SLURM_STEP_NUM_NODES:?}" = 1
verified_gpu_count="$(python3 analysis/scripts/agentic_adaptive_gpu_resources.py)"
export SLURM_GPUS_ON_NODE="$verified_gpu_count"
printf 'ADAPTIVE_GPU_RESOURCE_GATE=PASS gpus=%s\n' "$SLURM_GPUS_ON_NODE"
if [[ "${SLURM_CPUS_PER_TASK:-}" != 32 || "${SLURM_MEM_PER_NODE:-}" != 524288 ]]; then
  printf 'ADAPTIVE_RESOURCE_GATE=FAIL expected_cpus=32 expected_memory_mib=524288 actual_cpus=%s actual_memory_mib=%s\n' \
    "${SLURM_CPUS_PER_TASK:-unset}" "${SLURM_MEM_PER_NODE:-unset}" >&2
  exit 1
fi
test -s "$plan"
test -z "$(git status --porcelain --untracked-files=all)"
export GEODML_EXECUTION_REPOSITORY="$repository"
export GEODML_EXECUTION_COMMIT="$(git rev-parse HEAD)"
export GEODML_APPROVED_WALLTIME=07:00:00
export GEODML_START_MARGIN_SECONDS=120
export GEODML_CLEANUP_MARGIN_SECONDS=45
export GEODML_ADAPTIVE_PLAN="$plan"
export GEODML_ADAPTIVE_WORKER_INDEX="$worker_index"
export PYTHONUNBUFFERED=1

json_value() {
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v:=v[k] for k in sys.argv[2:]]; print("" if v is None else v)' "$@"
}

test "$(json_value "$plan" source_git_commit)" = "$GEODML_EXECUTION_COMMIT"
test "$(json_value "$plan" approved_allocation walltime)" = 07:00:00
test "$(json_value "$plan" approved_allocation allocation_count)" = 1
test "$(json_value "$plan" approved_allocation node_count)" = 5
test "$(json_value "$plan" approved_allocation maximum_gpu_hours)" = 140

run_root="$(dirname "$plan")"
study_root="$(json_value "$plan" original_study root)"
llama_profile="$(json_value "$plan" original_study llama serving_profile)"
llama_claim_root="$(json_value "$plan" original_study llama claim_root)"
prompts="$(json_value "$plan" original_study prompt_sources prompts_jsonl path)"
selection="$(json_value "$plan" original_study prompt_sources selection_records_jsonl path)"
ddg="$(json_value "$plan" original_study search_snapshots duckduckgo path)"
searxng="$(json_value "$plan" original_study search_snapshots searxng path)"
cross_snapshot="$(json_value "$plan" original_study cross_encoder_snapshot)"
cross_revision="$(json_value "$plan" original_study cross_encoder_revision)"
tasks="$(json_value "$plan" original_study generation_tasks path)"
judge_root="$(json_value "$plan" judge plan_root)"
nemotron_profile="$(json_value "$plan" judge serving_profile)"
validation_args=(--bulk-only)
if [[ -n "$(json_value "$plan" judge validation_model)" ]]; then
  validation_args=(
    --validation-model-id "$(json_value "$plan" judge validation_model model_id)"
    --validation-model-revision "$(json_value "$plan" judge validation_model model_revision)"
    --validation-fraction 0.02
  )
fi
test -s "$llama_profile"
test -s "$nemotron_profile"
test -s "$tasks"

mkdir -p "$run_root/workers/worker-$(printf '%05d' "$worker_index")"
worker_root="$run_root/workers/worker-$(printf '%05d' "$worker_index")"
export GEODML_ADAPTIVE_ROLE_ROOT="$worker_root"

materialize_and_plan_judge() {
  exec 8>>"$run_root/.generation-barrier.lock"
  flock 8
  if [[ ! -s "$judge_root/run_manifest.json" ]]; then
    if ! python3 analysis/scripts/materialize_agentic_generator_claims.py --plan "$plan"; then
      flock -u 8
      exec 8>&-
      return 1
    fi
    python3 analysis/scripts/prepare_agentic_judge_tasks.py \
      --generator-root "Qwen/Qwen3.8-27B=$study_root/models/qwen38" \
      --generator-root "meta-llama/Llama-4-Scout-17B-16E-Instruct=$study_root/models/llama4" \
      --prompts-jsonl "$prompts" \
      --selection-records-jsonl "$selection" \
      --generation-tasks "$tasks" \
      --bulk-model-id "$(json_value "$plan" judge bulk_model model_id)" \
      --bulk-model-revision "$(json_value "$plan" judge bulk_model model_revision)" \
      "${validation_args[@]}" \
      --master-seed 20260915 \
      --output-dir "$judge_root" \
      --recorded-conversation
  fi
  flock -u 8
  exec 8>&-
}

llama_started="$(date +%s)"
attempt=0
while [[ ! -s "$judge_root/run_manifest.json" ]]; do
  if materialize_and_plan_judge 2>"$worker_root/barrier-$attempt.err"; then
    break
  fi
  now="$(date +%s)"
  if (( now >= SLURM_JOB_END_TIME - 1020 )); then
    printf 'ADAPTIVE_WORKER=CHECKPOINTED reason=allocation_deadline role=llama4\n'
    exit 0
  fi
  if (( attempt > 0 || now - llama_started < 300 )); then
    sleep 60
    if materialize_and_plan_judge 2>"$worker_root/barrier-wait-$attempt.err"; then
      break
    fi
  fi
  role="$worker_root/llama4-attempt-$(printf '%02d' "$attempt")"
  export SEARCH_AGENTIC_OUTPUT="$role/output"
  export SEARCH_AGENTIC_PROFILE="$llama_profile"
  export SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT="$cross_snapshot"
  export SEARCH_AGENTIC_CROSS_ENCODER_REVISION="$cross_revision"
  export SEARCH_AGENTIC_DDG_SNAPSHOT="$ddg"
  export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$searxng"
  export SEARCH_AGENTIC_SERVER_LOG="$role/server.log"
  export SEARCH_AGENTIC_GPU_TELEMETRY="$role/gpu.csv"
  export SEARCH_AGENTIC_SHARED_CLAIM_ROOT="$llama_claim_root"
  export SEARCH_AGENTIC_WORKER_INDEX="$worker_index"
  export SEARCH_AGENTIC_WORKER_COUNT=5
  export SEARCH_AGENTIC_CELL_IDS_JSONL="$tasks"
  export SEARCH_AGENTIC_EXPECTED_CELL_COUNT=6000
  export SEARCH_AGENTIC_PROMPTS_JSONL="$prompts"
  export SEARCH_AGENTIC_SELECTION_RECORDS_JSONL="$selection"
  export SEARCH_AGENTIC_PROMPT_COUNT=500
  export SEARCH_AGENTIC_PROMPT_SELECTION_SEED=20260912
  export SEARCH_AGENTIC_PROMPT_SHARD_INDEX=0
  export SEARCH_AGENTIC_PROMPT_SHARD_COUNT=1
  export SEARCH_AGENTIC_PRODUCTION_CONDITIONS=1
  export SEARCH_AGENTIC_REQUEST_CONCURRENCY=4
  export SEARCH_AGENTIC_CELL_CONCURRENCY=12
  mkdir -p "$role"
  bash analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh
  attempt=$((attempt + 1))
done

nemotron_snapshot="${HF_HUB_CACHE:?}/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/$(json_value "$plan" judge bulk_model model_revision)"
export GEODML_NEMOTRON_PROFILE="$nemotron_profile"
export GEODML_NEMOTRON_SNAPSHOT="$nemotron_snapshot"
bash analysis/scripts/slurm/jupiter/run_agentic_adaptive_nemotron.sh
bash analysis/scripts/slurm/jupiter/run_agentic_adaptive_overflow.sh
