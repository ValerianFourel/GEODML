#!/usr/bin/env bash
# Install only the LLM2Vec-Gen inference runtime into the active environment.
#
# The upstream 0.1.3 wheel declares training/evaluation dependencies that pin
# torch==2.6.0, transformers==4.56.2, and flash-attn==2.7.4.post1. Installing it
# normally could replace the CUDA-compatible HoreKa stack. The inference code
# used here needs only torch, transformers, peft, PyYAML, and huggingface-hub.

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_REQUIREMENTS="$REPOSITORY_ROOT/analysis/requirements-horeka-llm2vec-gen.txt"
LLM2VEC_GEN_VERSION="0.1.3"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the intended Python virtual environment first." >&2
    exit 2
fi

python3 -m pip install -r "$RUNTIME_REQUIREMENTS"
python3 -m pip install --no-deps "llm2vec-gen==$LLM2VEC_GEN_VERSION"

python3 - <<'PY'
from importlib.metadata import version

import peft
import torch
import transformers
import yaml
from llm2vec_gen import LLM2VecGenModel

del LLM2VecGenModel, peft, yaml
print("LLM2Vec-Gen inference imports: OK")
for distribution in ("llm2vec-gen", "torch", "transformers", "peft", "PyYAML"):
    print(f"{distribution}={version(distribution)}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_gpu_count={torch.cuda.device_count()}")
PY
