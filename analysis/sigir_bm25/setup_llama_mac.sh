#!/bin/bash
# One-time Mac setup for run_llama_api.sh: a Python environment with the cross-encoder stack and the bge-reranker-v2-m3
# snapshot at the original run's revision (about 2.3 GB). Rerunning skips what exists.
#   bash analysis/sigir_bm25/setup_llama_mac.sh
set -euo pipefail
DATA=${SIGIR_BM25_DATA:-$HOME/Hamburg/GEODML_Unified/sigir_bm25_data}
BASE=${BASE_PYTHON:-$HOME/miniconda3/bin/python3}
mkdir -p "$DATA/snapshots"
[ -x "$DATA/venv/bin/python" ] || "$BASE" -m venv "$DATA/venv"
"$DATA/venv/bin/pip" install -q --upgrade pip
"$DATA/venv/bin/pip" install -q torch sentence-transformers transformers pandas pyarrow numpy scipy aiohttp huggingface_hub
"$DATA/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("BAAI/bge-reranker-v2-m3", revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
print("BGE snapshot:", path)
PY
"$DATA/venv/bin/python" -c "import torch, sentence_transformers, transformers; print('torch', torch.__version__, 'sentence-transformers', sentence_transformers.__version__, 'transformers', transformers.__version__)"
echo "Next: copy the two search snapshots from HoreKa into $DATA/snapshots/ (see analysis/sigir_bm25/README.md)."
