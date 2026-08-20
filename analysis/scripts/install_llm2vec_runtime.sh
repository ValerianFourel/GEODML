#!/usr/bin/env bash
set -euo pipefail

# Install the official Qwen2/Qwen3-capable LLM2Vec revision without allowing
# its dependency metadata to replace the CUDA/Transformers stack already
# validated by this repository on HoreKa.  PyPI 0.2.3 eagerly imports a removed
# MistralFlashAttention2 class and cannot import with the HoreKa runtime.

LLM2VEC_GIT_REVISION="0fbcf3304139099bda75c3d6b5d8e835d4894563"
LLM2VEC_GIT_URL="https://github.com/McGill-NLP/llm2vec.git"
REGEX_VERSION="2025.11.3"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activate the intended Python virtual environment first." >&2
    exit 2
fi

python3 -m pip install \
    "regex==${REGEX_VERSION}"

python3 -m pip install \
    --no-deps \
    --force-reinstall \
    "llm2vec @ git+${LLM2VEC_GIT_URL}@${LLM2VEC_GIT_REVISION}"

python3 - <<'PY'
from llm2vec import LLM2Vec
from llm2vec.models.bidirectional_qwen2 import Qwen2BiModel
from llm2vec.models.bidirectional_qwen3 import Qwen3BiModel
import llm2vec

print("LLM2Vec Qwen2/Qwen3 imports: OK")
print("LLM2Vec version:", getattr(llm2vec, "__version__", "unknown"))
del LLM2Vec, Qwen2BiModel, Qwen3BiModel
PY
