#!/usr/bin/env bash
set -euo pipefail

# Install the small official LLM2Vec wrapper without allowing its historical
# Transformers upper bound to replace the CUDA/Transformers stack already
# validated by this repository on HoreKa.

python3 -m pip install --no-deps 'llm2vec==0.2.3'

python3 - <<'PY'
from llm2vec import LLM2Vec
import llm2vec

print("LLM2Vec import: OK")
print("LLM2Vec version:", getattr(llm2vec, "__version__", "unknown"))
del LLM2Vec
PY
