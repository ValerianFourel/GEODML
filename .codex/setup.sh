#!/usr/bin/env bash
# Prepare a CPU-only Codex cloud development environment for GEODML.

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

test -s analysis/requirements.txt
test -s .codex/requirements.txt

if [[ "${1:-}" == "--check-only" ]]; then
    python3 --version
    printf 'CODEX_CLOUD_SETUP_CHECK=PASS\n'
    exit 0
fi

venv_root="${GEODML_CODEX_VENV:-$repository_root/.venv}"
if [[ ! -x "$venv_root/bin/python" ]]; then
    python3 -m venv "$venv_root"
fi

"$venv_root/bin/python" -m pip install --upgrade pip wheel
"$venv_root/bin/python" -m pip install --requirement .codex/requirements.txt

site_packages="$($venv_root/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' "$repository_root" "$repository_root/analysis" \
    > "$site_packages/geodml_repository.pth"

"$venv_root/bin/python" - <<'PY'
import aiohttp
import datasets
import pandas
import pyarrow
import sklearn

from analysis.interpretability.pipeline import acl_arr_document_experiment

print("CODEX_CLOUD_SETUP=PASS")
PY
