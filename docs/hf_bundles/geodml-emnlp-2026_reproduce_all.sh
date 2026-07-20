#!/usr/bin/env bash
# reproduce_all.sh — re-runs the DML fit + every figure in the paper.
#
# Usage:
#   bash reproduce_all.sh            # full re-run (~25 min for the DML refit)
#   bash reproduce_all.sh --skip-dml # reuse the precomputed parquet, just re-render figures
#
# Every script reads input from $GEODML_DATA_ROOT/data/... and writes figures
# to $GEODML_DATA_ROOT/figures/. The default root is this script's directory.

set -e

if [ -z "${GEODML_DATA_ROOT:-}" ]; then
  export GEODML_DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
  echo "[reproduce] GEODML_DATA_ROOT not set; defaulting to $GEODML_DATA_ROOT"
fi

cd "$(dirname "$0")"
mkdir -p figures

echo
echo "==========  Step 1/4  DML refit  =========="
echo "(Skip with --skip-dml if you want to use the precomputed parquet.)"
if [ "${1:-}" != "--skip-dml" ]; then
  python scripts/dml_canonical.py
fi

echo
echo "==========  Step 2/4  Canonical figures (fig01-fig14)  =========="
python scripts/make_canonical_figures.py

echo
echo "==========  Step 3/4  DML framework diagram  =========="
python scripts/make_dml_framework.py

echo
echo "==========  Step 4/4  Stage-F figures (probing, admission, saliency)  =========="
python scripts/make_fig_probing.py
python scripts/make_fig_admission_probe.py
python scripts/make_fig_saliency.py

echo
echo "Done. All figures landed in figures/."
