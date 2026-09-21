#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'Usage: bash %s PLAN_JSON\n' "$0" >&2
  exit 64
fi
plan="$(realpath "$1")"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
job_id="${SLURM_JOB_ID:?Run this child script inside the approved allocation shell}"
test -s "$plan"
test "${SLURM_JOB_NUM_NODES:?}" = 5

job_record="$(scontrol show job --oneliner "$job_id")"
record_start=""
record_end=""
record_nodes=""
for token in $job_record; do
  case "$token" in
    StartTime=*) record_start="${token#*=}" ;;
    EndTime=*) record_end="${token#*=}" ;;
    NodeList=*) record_nodes="${token#*=}" ;;
  esac
done
test -n "$record_start"
test -n "$record_end"
test -n "$record_nodes"
export SLURM_JOB_START_TIME="$(date -d "$record_start" +%s)"
export SLURM_JOB_END_TIME="$(date -d "$record_end" +%s)"
export GEODML_EXPECTED_JOB_ID="$job_id"
export GEODML_APPROVED_WALLTIME=07:00:00
export GEODML_ALLOCATION_ESTIMATE="Approved five-node interactive Booster allocation: 20 GH200 GPUs, 160 requested CPUs, 2560G requested aggregate memory, seven hours, maximum 140 GPU-hours. Actual Slurm start/end times include startup and cap every role. No requeue, extension, replacement, or auto-submission."

readarray -t nodes < <(scontrol show hostnames "$record_nodes")
test "${#nodes[@]}" = 5
run_root="$(dirname "$plan")"
mkdir -p "$run_root/launcher-logs"

pids=()
cleanup() {
  trap - INT TERM
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait "${pids[@]}" 2>/dev/null || true
  printf '%s\n' 'Adaptive steps stopped; the allocation-owning shell remains open.' >&2
}
trap cleanup INT TERM

for worker_index in 0 1 2 3 4; do
  stdout="$run_root/launcher-logs/worker-$(printf '%05d' "$worker_index").out"
  stderr="$run_root/launcher-logs/worker-$(printf '%05d' "$worker_index").err"
  (
    # A launcher may run inside a one-CPU inspection step. New steps must use
    # the allocation and explicit requests below, not that step's environment.
    for variable in "${!SLURM_@}" "${!SRUN_@}"; do
      case "$variable" in
        SLURM_JOB_ID|SLURM_JOB_NUM_NODES|SLURM_JOB_START_TIME|SLURM_JOB_END_TIME) ;;
        SLURM_CONF|SLURM_CONF_SERVER|SLURM_JWT|SLURM_CLUSTER_NAME|SLURM_CLUSTERS) ;;
        *) unset "$variable" ;;
      esac
    done
    unset CUDA_VISIBLE_DEVICES
    exec srun --jobid="$job_id" --nodes=1 --ntasks=1 --nodelist="${nodes[$worker_index]}" \
      --exclusive --gpus-per-node=4 --cpus-per-task=32 --mem=512G \
      bash "$repository/analysis/scripts/slurm/jupiter/run_agentic_adaptive_worker.sh" \
        "$plan" "$worker_index"
  ) >"$stdout" 2>"$stderr" &
  pids+=("$!")
done

statuses=()
failed=0
set +e
for pid in "${pids[@]}"; do
  wait "$pid"
  status=$?
  statuses+=("$status")
  if (( status != 0 )); then failed=1; fi
done
set -e

python3 - "$run_root/launcher-summary.json" "$job_id" "$failed" "${statuses[@]}" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
value = {
    "format_version": "agentic-adaptive-launcher-summary-v1",
    "job_id": sys.argv[2], "failed": bool(int(sys.argv[3])),
    "worker_exit_codes": [int(value) for value in sys.argv[4:]],
    "slurm_start_epoch": int(os.environ["SLURM_JOB_START_TIME"]),
    "slurm_end_epoch": int(os.environ["SLURM_JOB_END_TIME"]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY
exit "$failed"
