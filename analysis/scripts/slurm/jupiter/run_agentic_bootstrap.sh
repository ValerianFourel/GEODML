#!/usr/bin/env bash
# Run a bootstrap preparation command inside an already approved allocation.
# For an existing backlog, use run_jupiter_prepared_backlog.py as the command.
# No registration, recovery acceptance, or plan construction happens here.
set -eo pipefail
: "${SLURM_JOB_ID:?Run on the allocated compute node}"
: "${ACL_ARR_ENVIRONMENT_FILE:?Set the site environment file}"
: "${1:?Pass the bootstrap or prepared-backlog command}"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
source "$ACL_ARR_ENVIRONMENT_FILE"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"
export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY="1"
unset GEODML_PRIVATE_NETWORK_NAMESPACE
set -u
exec python3 "$repository/analysis/scripts/verify_inference_allocation.py" \
  --cluster jupiter --exec "$@"
