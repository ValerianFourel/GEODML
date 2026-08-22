#!/bin/bash -l
# Install the pinned Qwen3/Gemma4 proposal-generation runtime on JUPITER.
# This performs no model download and requires no Slurm allocation.

set -euo pipefail
umask 077

: "${FSCRATCH:?FSCRATCH is required}"
: "${USER:?USER is required}"

module --force purge
module load Stages/2026
module load GCCcore/14.3.0
module load SciPy-Stack/2025b
module load PyTorch/2.9.1
jutil env activate -p "${JUPITER_PROJECT:-scifi}"

runtime="${GEODML_GENERATOR_VENV:-$FSCRATCH/$USER/geodml/python/.venv-readiness-generators-transformers562}"
if [[ -e "$runtime" ]]; then
    echo "refusing to replace existing runtime: $runtime" >&2
    exit 2
fi

runtime_parent="$(dirname "$runtime")"
mkdir -p "$runtime_parent"
python3 -m venv --system-site-packages "$runtime"
source "$runtime/bin/activate"
export PYTHONNOUSERSITE=1
runtime_site_packages="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$runtime_site_packages${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pip install --upgrade "pip==25.2"
python3 -m pip install \
    "transformers==5.6.2" \
    "accelerate==1.14.0" \
    "protobuf==6.32.0" \
    "requests==2.32.5" \
    "python-dotenv==1.1.1" \
    "huggingface-hub==1.16.1" \
    "PyYAML==6.0.3" \
    "tqdm==4.67.3" \
    "typer==0.25.1" \
    "annotated-doc==0.0.4" \
    "hf-xet==1.5.0" \
    "httpx==0.28.1"

python3 - <<'PY'
import torch
import transformers
import dotenv
import huggingface_hub
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

assert transformers.__version__ == "5.6.2", transformers.__version__
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Generator runtime imports: OK")
del AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
PY

deactivate
printf '%s\n' "$runtime"
