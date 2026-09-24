#!/usr/bin/env bash
# Both cluster login shells; no allocation or automatic resubmission.
set -euo pipefail
: "${GEODML_SITE_CONFIG:?Set GEODML_SITE_CONFIG to the external site JSON}"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${GEODML_PYTHON:-python3}" "$repository/analysis/scripts/manage_agentic_hours.py" "$@" --site "$GEODML_SITE_CONFIG"
