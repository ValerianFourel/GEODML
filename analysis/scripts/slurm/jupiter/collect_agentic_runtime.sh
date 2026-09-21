#!/usr/bin/env bash
# Observational steps inside an existing allocation, never a new allocation.
set -euo pipefail
if [[ "$#" != 4 ]]; then
  printf 'Usage: bash %s collect|profile|summary JOB_ID RUN_ROOT REPORT_DIRECTORY\n' "$0" >&2
  exit 64
fi
case "$1" in collect|profile|summary) ;; *) exit 64 ;; esac
source "${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module load Stages/2026 GCC Python CUDA git
source "${ACL_ARR_VENV:?}/bin/activate"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
exec python3 "$repository/analysis/scripts/collect_agentic_runtime.py" "$1" \
  --job-id "$2" --run-root "$3" --output "$4"
