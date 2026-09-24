#!/usr/bin/env bash
# Run in a HoreKa login shell. Creates/reuses a workspace; allocates no compute.
set -euo pipefail
umask 077
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
: "${GEODML_HOREKA_ACCOUNT:?Set your HoreKa project account}"
name="${GEODML_WORKSPACE_NAME:-geodml-qwen}"
[[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
test -z "$(git -C "$repository" status --porcelain --untracked-files=all)"
pin="$(git -C "$repository" rev-parse HEAD)"
workspace="$(ws_find "$name" 2>/dev/null || true)"
if [[ -z "$workspace" ]]; then
  ws_allocate "$name" 60
  workspace="$(ws_find "$name")"
fi
test -d "$workspace"
workspace="$(cd "$workspace" && pwd -P)"
python_bin="${GEODML_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,10))'; then
      python_bin="$(command -v "$candidate")"
      break
    fi
  done
fi
: "${python_bin:?No Python >=3.10 found; set GEODML_PYTHON to a compatible installed interpreter}"
prep="$workspace/preparation/$pin"
transfer="$workspace/environment/qwen-transfer"
mkdir -p "$prep" "$workspace/environment" "$workspace/models" "$workspace/datasets" "$workspace/runs"
ws_list > "$prep/workspaces.txt"
export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$repository/analysis/scripts/prepare_horeka_qwen.py" quota \
  --workspace "$workspace" --project "$GEODML_HOREKA_ACCOUNT" --output "$prep/quota.json" > "$prep/quota.log"
"$python_bin" - "$transfer" "$prep/quota.json" <<'PY'
import json, sys
from pathlib import Path
from analysis.scripts.prepare_horeka_qwen import check_storage, immutable_json
root, quota = map(Path, sys.argv[1:])
check_storage(root, json.loads(quota.read_bytes()), 1024**3, 10000)
marker = root.parent / 'qwen-transfer-owner.json'
if root.exists() and not marker.exists():
    raise SystemExit('Existing transfer environment is not owned by this preparation.')
immutable_json(marker, {'python': str(Path(sys.executable).resolve()), 'path': str(root.resolve())})
PY
if [[ ! -f "$transfer/pyvenv.cfg" ]]; then
  "$python_bin" -m venv "$transfer"
fi
export PIP_CACHE_DIR="$workspace/environment/pip-cache"
"$transfer/bin/python" -m pip install 'huggingface_hub>=0.24,<2'
"$transfer/bin/python" -m pip freeze --all > "$prep/transfer-environment-freeze.txt"
export HF_HUB_CACHE="$workspace/models"
export HF_XET_CACHE="$workspace/environment/xet-cache"
tool="$repository/analysis/scripts/prepare_horeka_qwen.py"
"$transfer/bin/python" "$tool" inventory --manifest "$prep/models.json" > "$prep/model-inventory.log"
"$transfer/bin/python" "$tool" quota --workspace "$workspace" --project "$GEODML_HOREKA_ACCOUNT" \
  --output "$prep/quota.json" > "$prep/quota.log"
"$transfer/bin/python" "$tool" download --manifest "$prep/models.json" --cache "$HF_HUB_CACHE" \
  --quota-evidence "$prep/quota.json" --report "$prep/models-verified.json" > "$prep/model-download.log"
"$transfer/bin/python" - "$prep" "$repository" "$workspace" "$transfer" "$pin" "$GEODML_HOREKA_ACCOUNT" <<'PY'
import json, shlex, sys
from pathlib import Path
from analysis.scripts.prepare_horeka_qwen import immutable_json
prep, repo, workspace, transfer = map(Path, sys.argv[1:5])
pin, account = sys.argv[5:]
env = {'GEODML_REPOSITORY': str(repo), 'GEODML_HOREKA_WORKSPACE': str(workspace),
       'GEODML_PREPARATION': str(prep), 'GEODML_TRANSFER_PYTHON': str(transfer / 'bin/python'),
       'GEODML_HOREKA_ACCOUNT': account, 'HF_HUB_CACHE': str(workspace / 'models'),
       'HF_XET_CACHE': str(workspace / 'environment/xet-cache')}
text = ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in env.items())
(prep / 'environment.sh').write_text(text)
immutable_json(prep / 'preparation.json', {'git_commit': pin, 'environment': env,
    'models': json.loads((prep / 'models-verified.json').read_bytes()),
    'inputs': 'awaiting_verified_bundle', 'gpu_validation': 'not_performed', 'allocation_submitted': False})
print('MODEL_PREPARATION=PASS')
print('PREPARATION=' + str(prep))
print('ENVIRONMENT=' + str(prep / 'environment.sh'))
print('NEXT=verify JUPITER reconciliation and private input bundle before downloading data')
PY
