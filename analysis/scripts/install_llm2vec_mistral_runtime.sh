#!/usr/bin/env bash
set -euo pipefail

# The frozen Mistral LLM2Vec checkpoints use the pre-attention-refactor
# Mistral encoder classes. Keep this runtime separate from the newer
# Qwen2/Qwen3 environment and install without replacing HoreKa's CUDA PyTorch.

LLM2VEC_GIT_REVISION="68dc1d3244cc710942a5bbbf11d9677de9f8f68a"
LLM2VEC_GIT_URL="https://github.com/McGill-NLP/llm2vec.git"
REGEX_VERSION="2025.11.3"
TRANSFORMERS_VERSION="4.40.2"
PEFT_VERSION="0.10.0"
HUGGINGFACE_HUB_VERSION="0.23.2"
TOKENIZERS_VERSION="0.19.1"
SAFETENSORS_VERSION="0.4.3"
ACCELERATE_VERSION="0.30.1"
TQDM_VERSION="4.66.4"
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
