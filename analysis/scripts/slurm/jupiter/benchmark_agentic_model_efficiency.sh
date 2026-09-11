#!/usr/bin/env bash
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module load Stages/2026 GCC Python CUDA
module load git
source "${ACL_ARR_VENV:?}/bin/activate"

geodml_job="${GEODML_EXPECTED_JOB_ID:?}"
geodml_commit="${GEODML_EXECUTION_COMMIT:?}"
geodml_repo="${GEODML_EXECUTION_REPOSITORY:?}"
geodml_benchmark_root="${SEARCH_AGENTIC_BENCHMARK_ROOT:?}"
geodml_root="${GEODML_PROJECT_ROOT:?}/runs/acl-arr-search-experience/pilot-3-627ad348c2a0"
geodml_ce_revision=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
geodml_ce_snapshot="${HF_HUB_CACHE:?}/models--BAAI--bge-reranker-v2-m3/snapshots/$geodml_ce_revision"
geodml_ddg="${GEODML_DATA_ROOT:?}/data/serp/phase0_top20_ddg.parquet"
geodml_searxng="${ACL_ARR_SEARCH_SNAPSHOT:?}"

test "${SLURM_JOB_ID:?Run in the allocation-owning shell}" = "$geodml_job"
test "$(squeue --me --jobs="$geodml_job" --noheader --format='%T')" = RUNNING
test "$(git -C "$geodml_repo" rev-parse HEAD)" = "$geodml_commit"
test -z "$(git -C "$geodml_repo" status --porcelain --untracked-files=all)"
test -d "$geodml_ce_snapshot"
test -s "$geodml_ddg"
test -s "$geodml_searxng"
test ! -e "$geodml_benchmark_root"
mkdir -p "$geodml_benchmark_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GEODML_EXPECTED_JOB_ID="$geodml_job"
export GEODML_EXECUTION_COMMIT="$geodml_commit"
export GEODML_EXECUTION_REPOSITORY="$geodml_repo"
export SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT="$geodml_ce_snapshot"
export SEARCH_AGENTIC_CROSS_ENCODER_REVISION="$geodml_ce_revision"
export SEARCH_AGENTIC_DDG_SNAPSHOT="$geodml_ddg"
export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$geodml_searxng"

run_trial() {
  local model_key="$1"
  local model_id="$2"
  local concurrency="$3"
  local step_time="$4"
  local launcher="$5"
  local profile="$6"
  local trial_root="$geodml_benchmark_root/$model_key/concurrency-$concurrency"
  local started ended step_status

  mkdir -p "$trial_root"
  printf '%s\n' "$model_id" > "$trial_root/model_id.txt"
  printf '%s\n' "$concurrency" > "$trial_root/concurrency.txt"
  printf '4\n' > "$trial_root/allocated_gpus.txt"
  started="$(date +%s)"

  export SEARCH_AGENTIC_REQUEST_CONCURRENCY="$concurrency"
  export SEARCH_AGENTIC_OUTPUT="$trial_root/output"
  export SEARCH_AGENTIC_PROFILE="$profile"
  export SEARCH_AGENTIC_SERVER_LOG="$trial_root/server.log"
  export SEARCH_AGENTIC_GPU_TELEMETRY="$trial_root/gpu.csv"

  set +e
  srun --jobid="$geodml_job" --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --mem=512G --gres=gpu:4 --time="$step_time" --chdir="$geodml_repo" \
    --export=ALL bash "$geodml_repo/$launcher" \
    > >(tee "$trial_root/step.log") 2>&1
  step_status=$?
  set -e

  ended="$(date +%s)"
  printf '%s\n' "$((ended - started))" > "$trial_root/elapsed_seconds.txt"
  printf '%s\n' "$step_status" > "$trial_root/step_status.txt"
  printf 'EFFICIENCY_STEP model=%s concurrency=%s status=%s elapsed_seconds=%s\n' \
    "$model_key" "$concurrency" "$step_status" "$((ended - started))"
}

qwen38_profile="$geodml_root/primary-answer2048-smoke-39aaf746b484/model-config-a58eb7c40e545350abc1.serving-profile.json"
qwen25_profile="$geodml_root/primary-answer2048-hf-overrides-a5648c6cd160/model-config-fbe629168c7384f96048.serving-profile.json"
llama4_profile="$geodml_root/primary-answer2048-smoke-39aaf746b484/model-config-2403370677dae49f8cd1.serving-profile.json"
nemotron_snapshot="$HF_HUB_CACHE/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/2dc98e2afe4face0e4ce40972a915c45368bd34a"
nemotron_manifest="$geodml_root/nemotron3-super-2dc98e2afe4f.download-manifest.json"

test -s "$qwen38_profile"
test -s "$qwen25_profile"
test -s "$llama4_profile"
test -d "$nemotron_snapshot"
test -s "$nemotron_manifest"

run_trial qwen38 Qwen/Qwen3.8-27B 1 00:20:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh \
  "$qwen38_profile"
run_trial qwen38 Qwen/Qwen3.8-27B 4 00:20:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh \
  "$qwen38_profile"

run_trial qwen25 Qwen/Qwen2.5-72B-Instruct 4 00:25:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_qwen25_smoke.sh \
  "$qwen25_profile"
run_trial qwen25 Qwen/Qwen2.5-72B-Instruct 1 00:25:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_qwen25_smoke.sh \
  "$qwen25_profile"

run_trial llama4 meta-llama/Llama-4-Scout-17B-16E-Instruct 1 00:25:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh \
  "$llama4_profile"
run_trial llama4 meta-llama/Llama-4-Scout-17B-16E-Instruct 4 00:25:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh \
  "$llama4_profile"

export SEARCH_AGENTIC_MODEL_SNAPSHOT="$nemotron_snapshot"
export SEARCH_AGENTIC_DOWNLOAD_MANIFEST="$nemotron_manifest"
run_trial nemotron nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 4 00:30:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_nemotron_smoke.sh \
  "$geodml_benchmark_root/nemotron/concurrency-4/serving-profile.json"
run_trial nemotron nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 1 00:30:00 \
  analysis/scripts/slurm/jupiter/run_agentic_search_nemotron_smoke.sh \
  "$geodml_benchmark_root/nemotron/concurrency-1/serving-profile.json"

python3 "$geodml_repo/analysis/scripts/summarize_agentic_efficiency_benchmark.py" \
  --benchmark-root "$geodml_benchmark_root" \
  --output "$geodml_benchmark_root/efficiency-summary.json"

printf '%s\n' "$geodml_benchmark_root" > "$geodml_root/agentic-efficiency-latest.txt"
printf 'EFFICIENCY_SUMMARY=%s\n' "$geodml_benchmark_root/efficiency-summary.json"
