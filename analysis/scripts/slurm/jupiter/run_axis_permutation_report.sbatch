#!/usr/bin/env bash
# Submission supplies the specifically approved wall-time and resource request.
# This job reads saved artifacts. It does not start inference or a model server.
set +u
set -eo pipefail
export DEBUGINFOD_URLS="${DEBUGINFOD_URLS:-}"
if ! type module >/dev/null 2>&1; then
  source /etc/profile
fi
module load Stages/2026 GCC Python
module load git
source "${ACL_ARR_VENV:?}/bin/activate"
set -u

: "${SLURM_JOB_ID:?Run through Slurm}"
: "${GEODML_EXECUTION_REPOSITORY:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1
cd "$GEODML_EXECUTION_REPOSITORY"
exec python3 analysis/scripts/run_axis_permutation_report_batch.py
