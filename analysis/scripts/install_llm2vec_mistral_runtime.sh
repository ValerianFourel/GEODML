#!/usr/bin/env bash
set -euo pipefail

# The frozen Mistral LLM2Vec checkpoints require the Mistral attention classes
# retained through Transformers 4.47.1. Keep this runtime separate from the
# Qwen2/Qwen3 environment and do not replace HoreKa's CUDA PyTorch.

LLM2VEC_GIT_REVISION="0fbcf3304139099bda75c3d6b5d8e835d4894563"
LLM2VEC_GIT_URL="https://github.com/McGill-NLP/llm2vec.git"
REGEX_VERSION="2025.11.3"
TRANSFORMERS_VERSION="4.47.1"
PEFT_VERSION="0.14.0"
HUGGINGFACE_HUB_VERSION="0.36.2"
TOKENIZERS_VERSION="0.21.4"
SAFETENSORS_VERSION="0.8.0"
ACCELERATE_VERSION="1.2.1"
TQDM_VERSION="4.67.1"
PYYAML_VERSION="6.0.2"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the intended Python virtual environment first." >&2
    exit 2
fi

VENV_SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="${VENV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m pip install \
    --no-deps \
    "regex==${REGEX_VERSION}" \
    "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" \
    "tokenizers==${TOKENIZERS_VERSION}" \
    "safetensors==${SAFETENSORS_VERSION}" \
    "accelerate==${ACCELERATE_VERSION}" \
    "tqdm==${TQDM_VERSION}" \
    "PyYAML==${PYYAML_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "peft==${PEFT_VERSION}"

python3 -m pip install \
    --no-deps \
    --force-reinstall \
    "llm2vec @ git+${LLM2VEC_GIT_URL}@${LLM2VEC_GIT_REVISION}"

python3 - <<'PY'
from llm2vec import LLM2Vec
from llm2vec.models.bidirectional_mistral import MistralBiModel
import llm2vec
import peft
import torch
import transformers

print("LLM2Vec Mistral imports: OK")
print("LLM2Vec version:", getattr(llm2vec, "__version__", "unknown"))
print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("PEFT version:", peft.__version__)
del LLM2Vec, MistralBiModel
PY
