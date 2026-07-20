#!/bin/bash
# Convenience: audit then dispatch in one go.
# Run from /e/project1/scifi/fourel1/GEODML_Analysis on JUPITER.
#
# Examples:
#   ./scripts/repair_run.sh                    # audit + submit up to 32 jobs
#   ./scripts/repair_run.sh status             # just print status, no submit
#   ./scripts/repair_run.sh --stage probing    # audit + only submit probing
#   ./scripts/repair_run.sh --dry-run          # show what would be submitted
#
# Re-run as often as you like; it's idempotent.

set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "[repair_run] no .venv at $(pwd) — bootstrap with python -m venv .venv first"
  exit 2
fi

PY=.venv/bin/python

if [ "${1:-}" = "status" ]; then
  shift
  "$PY" scripts/repair_audit.py --print-only "$@" || exit $?
  if [ -f manifests/repair_manifest.parquet ]; then
    "$PY" scripts/repair_dispatch.py --status "$@" || exit $?
  fi
  exit 0
fi

echo "[repair_run] === audit ==="
"$PY" scripts/repair_audit.py || exit $?

echo
echo "[repair_run] === dispatch ==="
"$PY" scripts/repair_dispatch.py "$@" || exit $?

echo
echo "[repair_run] Done. Re-run me later to keep submitting as jobs finish."
