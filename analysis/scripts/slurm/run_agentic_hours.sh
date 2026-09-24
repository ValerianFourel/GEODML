#!/usr/bin/env bash
# No allocation directives: use an explicit, approved admission ticket.
# Also callable with srun --jobid=<existing ID> from an interactive shell.
set -euo pipefail
: "${1:?Pass the staged attempt.json}"
: "${SLURM_JOB_ID:?Run inside an approved allocation}"
attempt="$1"
environment_file="$(python3 - "$attempt" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["cluster_profile"]["environment_file"])
PY
)"
test -f "$environment_file"
# Site-owned setup activates the correct architecture's environment and modules.
# shellcheck disable=SC1090
source "$environment_file"
repository="$(python3 - "$attempt" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["repository"])
PY
)"
export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY="1"
unset GEODML_PRIVATE_NETWORK_NAMESPACE
exec python3 "$repository/analysis/scripts/manage_agentic_hours.py" execute \
  --attempt "$attempt" --job-id "$SLURM_JOB_ID"
