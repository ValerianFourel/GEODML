#!/usr/bin/env python3
"""Download the six LLM2Vec snapshots behind the frozen readiness maps.

Exact revisions come from the archived final-audit projection manifests
(geoaxis-prompts-generation-26k). Layout mirrors JUPITER:
ROOT/<org>/<name>/<revision>, so the frozen map's @revision check passes.
Existing complete snapshots are reused; nothing else is downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

MODELS = (
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "qwen/Qwen3-8B"),
    ("McGill-NLP/LLM2Vec-Qwen3-8B-mntp", "c84774c1366ea79f033504994bd254155d956d57",
     "mcgill-nlp/LLM2Vec-Qwen3-8B-mntp"),
    ("McGill-NLP/LLM2Vec-Qwen3-8B-mntp-unsup-simcse", "86b17660b1b1a8efe0b822e90c995f1ac7294645",
     "mcgill-nlp/LLM2Vec-Qwen3-8B-mntp-unsup-simcse"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a",
     "mistralai/Mistral-7B-Instruct-v0.2"),
    ("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", "e76f9757923897a0c5204b3075f1062f484d033b",
     "mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp"),
    ("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse", "2c055a5d77126c0d3dc6cd8ffa30e2908f4f45f8",
     "mcgill-nlp/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"),
)
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "tokenizer*"]


def main(argv=None) -> int:
    from huggingface_hub import snapshot_download

    if len(argv if argv is not None else sys.argv[1:]) != 1:
        print("usage: download_llm2vec_readiness_models.py MODELS_ROOT", file=sys.stderr)
        return 2
    root = Path((argv if argv is not None else sys.argv[1:])[0]).resolve()
    for repo, revision, folder in MODELS:
        target = root / folder / revision
        snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=PATTERNS)
        print("OK", repo, revision, target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
