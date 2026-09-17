#!/usr/bin/env bash
# Submit with separately approved walltime and resources. No allocation defaults.
set -euo pipefail
: "${GEODML_EXECUTION_REPOSITORY:?Use the absolute pinned repository path}"
: "${GEODML_JUDGE_CLAIM_ROOT:?All related judge jobs must share this durable claim directory}"
[[ "$GEODML_JUDGE_CLAIM_ROOT" = /* ]] || { printf '%s\n' 'Claim root must be an absolute shared path.' >&2; exit 2; }
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${GEODML_JUDGE_WORKER_INDEX:-}" ]]; then
  if [[ ! "${SLURM_ARRAY_TASK_COUNT:-}" =~ ^[1-9][0-9]*$ ||
        ! "${SLURM_ARRAY_TASK_MAX:-}" =~ ^[0-9]+$ ||
        "${SLURM_ARRAY_TASK_MIN:-}" != 0 || "${SLURM_ARRAY_TASK_STEP:-}" != 1 ]]; then
    printf '%s\n' 'Automatic judge slots require a contiguous zero-based Slurm array.' >&2
    exit 2
  fi
  test "$SLURM_ARRAY_TASK_MAX" -eq "$((SLURM_ARRAY_TASK_COUNT - 1))"
fi
export GEODML_JUDGE_WORKER_INDEX="${GEODML_JUDGE_WORKER_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
export GEODML_JUDGE_WORKER_COUNT="${GEODML_JUDGE_WORKER_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
export GEODML_JUDGE_DISPATCH_MODE="${GEODML_JUDGE_DISPATCH_MODE:-backlog}"
export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1
if [[ -n "${GEODML_JUDGE_QUEUE_ROOT:-}" ]]; then
  : "${SLURM_JOB_ID:?Run through Slurm}"
  export GEODML_JUDGE_PLAN_ROOT="$GEODML_JUDGE_QUEUE_ROOT/plan"
  export GEODML_JUDGE_PILOT_ROOT="$GEODML_JUDGE_QUEUE_ROOT/attempts/job${SLURM_JOB_ID}-worker${GEODML_JUDGE_WORKER_INDEX}"
  export GEODML_JUDGE_STDOUT="${GEODML_JUDGE_STDOUT:-$GEODML_JUDGE_QUEUE_ROOT/logs/slurm-${SLURM_JOB_ID}.out}"
  export GEODML_JUDGE_STDERR="${GEODML_JUDGE_STDERR:-$GEODML_JUDGE_QUEUE_ROOT/logs/slurm-${SLURM_JOB_ID}.err}"
fi
export GEODML_JUDGE_EXECUTION_MODE=throughput
exec bash "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/slurm/jupiter/run_nemotron_judge_pilot.sbatch"
