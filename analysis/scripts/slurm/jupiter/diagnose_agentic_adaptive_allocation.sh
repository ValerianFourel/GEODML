#!/usr/bin/env bash
# Run only the serving safety check in one worker-sized step of an existing job.
set -euo pipefail
if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s EXISTING_JOB_ID\n' "$0" >&2
  exit 64
fi
job_id="$1"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
record="$(scontrol show job --oneliner "$job_id")"
state=""; owner=""; node_count=""; node_list=""; actual_id=""
for token in $record; do
  case "$token" in
    JobId=*) actual_id="${token#*=}" ;;
    UserId=*) owner="${token#*=}" ;;
    JobState=*) state="${token#*=}" ;;
    NumNodes=*) node_count="${token#*=}" ;;
    NodeList=*) node_list="${token#*=}" ;;
  esac
done
if [[ "$actual_id" != "$job_id" || "$state" != RUNNING || "$node_count" != 5 || "$owner" != *"($(id -u))" ]]; then
  printf 'STOP: diagnostic requires your existing running five-node allocation. No allocation was requested.\n' >&2
  exit 1
fi
nodes=()
while IFS= read -r node; do nodes+=("$node"); done < <(scontrol show hostnames "$node_list")
test "${#nodes[@]}" = 5
export SLURM_JOB_ID="$job_id" SLURM_JOB_NUM_NODES=5
# Match the adaptive launcher's environment cleanup and explicit resource request.
for variable in "${!SLURM_@}" "${!SRUN_@}"; do
  case "$variable" in
    SLURM_JOB_ID|SLURM_JOB_NUM_NODES|SLURM_JOB_START_TIME|SLURM_JOB_END_TIME) ;;
    SLURM_CONF|SLURM_CONF_SERVER|SLURM_JWT|SLURM_CLUSTER_NAME|SLURM_CLUSTERS) ;;
    *) unset "$variable" ;;
  esac
done
unset CUDA_VISIBLE_DEVICES
exec srun --jobid="$job_id" --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
  --exclusive --gpus-per-node=4 --cpus-per-task=32 --mem=512G --immediate=15 \
  bash -c '
    set -euo pipefail
    source "${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
    if ! type module >/dev/null 2>&1; then source /etc/profile; fi
    module load Stages/2026 GCC Python CUDA git
    source "${ACL_ARR_VENV:?}/bin/activate"
    cd "$1"
    exec python3 analysis/scripts/inference_network_namespace.py --diagnose-slurm
  ' diagnostic "$repository"
