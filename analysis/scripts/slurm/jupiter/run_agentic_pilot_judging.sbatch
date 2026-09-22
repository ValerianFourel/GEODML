#!/usr/bin/env bash
# Resource and time approval is supplied by manage_agentic_pilot_judging.py.
# One independently scheduled one-node worker, not a multi-node serving process.
set +u
set -eo pipefail
test "$#" = 1
source "${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
export DEBUGINFOD_URLS="${DEBUGINFOD_URLS:-}"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"
set -u
: "${GEODML_EXECUTION_REPOSITORY:?Use the pinned checkout, not the Slurm spool directory}"
: "${GEODML_CACHE_ROOT:?}"
exec python3 "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/manage_agentic_pilot_judging.py" worker --run-root "$1"
