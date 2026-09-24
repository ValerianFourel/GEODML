#!/usr/bin/env bash
# Login/transfer-host setup only. No allocation, model download or model load.
set -euo pipefail
umask 077
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
: "${GEODML_HOREKA_WORKSPACE:?Set the active workspace returned by ws_find}"
: "${GEODML_QWEN_DATASET:?Set the downloaded, verified dataset directory}"
: "${GEODML_QUOTA_EVIDENCE:?Set a fresh HoreKa quota JSON}"
python_bin="${GEODML_PYTHON:-python3}"
runtime="$GEODML_HOREKA_WORKSPACE/environment/qwen-runtime"
requirements="$GEODML_HOREKA_WORKSPACE/environment/qwen-runtime-requirements.txt"
export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - "$runtime" "$requirements" "$GEODML_QWEN_DATASET" "$repository" "$GEODML_QUOTA_EVIDENCE" <<'PY'
import json, os, platform, re, sys
from pathlib import Path
from analysis.scripts.prepare_agentic_qwen_inputs import MANIFEST, verify_inputs
from analysis.scripts.prepare_horeka_qwen import check_storage, immutable_json
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
runtime, requirements, dataset, repository, quota = map(Path, sys.argv[1:])
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info < (3, 10):
    raise SystemExit('Use Linux x86_64 Python >=3.10; set GEODML_PYTHON to an installed compatible interpreter.')
if os.environ.get('GEODML_SHARED_INPUT_MANIFEST'):
    from analysis.scripts.prepare_shared_hour_inputs import verify_inputs as verify_shared
    value = verify_shared(dataset, Path(os.environ['GEODML_SHARED_INPUT_MANIFEST']), model='qwen38')
else:
    verify_inputs(dataset)
    value = json.loads((dataset / MANIFEST).read_bytes())
version = value['reference_vllm_version'].strip()
if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9.+_-]*', version):
    raise SystemExit('Reference vLLM version needs review; refusing to install an unpinned fallback.')
check_storage(runtime, json.loads(quota.read_bytes()), 30 * 1024**3, 100000)
if runtime.resolve().is_relative_to(repository.resolve()):
    raise SystemExit('Runtime environment must be outside the source checkout.')
marker = runtime.parent / 'qwen-runtime-owner.json'
identity = {'reference_profile_sha256': value['reference_profile_sha256'], 'runtime': str(runtime.resolve()),
            'python': str(Path(sys.executable).resolve()), 'vllm': version, 'sentence_transformers': '6.0.1'}
if runtime.exists() and not marker.exists():
    raise SystemExit('Existing runtime is not owned by this preparation; choose a new workspace.')
immutable_json(marker, identity)
text = f'-r {repository / "analysis/requirements.txt"}\nvllm=={version}\nsentence-transformers==6.0.1\n'
if requirements.exists() and requirements.read_text() != text:
    raise SystemExit('Existing runtime requirements differ; preserve them and choose a new workspace.')
atomic(requirements, text.encode())
PY
if [[ ! -f "$runtime/pyvenv.cfg" ]]; then
  "$python_bin" -m venv "$runtime"
fi
export PIP_CACHE_DIR="$GEODML_HOREKA_WORKSPACE/environment/pip-cache"
"$runtime/bin/python" -m pip install --only-binary=:all: -r "$requirements"
"$runtime/bin/python" -m pip check
"$runtime/bin/python" -m pip freeze --all > "$GEODML_HOREKA_WORKSPACE/environment/qwen-runtime-freeze.txt"
"$runtime/bin/python" - <<'PY'
from importlib.metadata import version
assert version('sentence-transformers') == '6.0.1'
print('ENVIRONMENT_INSTALLED_GPU_VALIDATION_PENDING')
print('vllm=' + version('vllm'))
print('torch=' + version('torch'))
PY
