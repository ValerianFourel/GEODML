#!/usr/bin/env python3
"""Download and verify the pinned Nemotron 3 Super BF16 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
MODEL_REVISION = "2dc98e2afe4face0e4ce40972a915c45368bd34a"
EXPECTED_ARCHITECTURE = "NemotronHForCausalLM"
MINIMUM_WEIGHT_BYTES = 240_000_000_000
MINIMUM_FREE_BYTES = 280 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    if snapshot.name != MODEL_REVISION:
        raise ValueError("snapshot directory does not match the pinned revision")
    config_path = snapshot / "config.json"
    tokenizer_path = snapshot / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise ValueError("snapshot is missing config.json or tokenizer.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != [EXPECTED_ARCHITECTURE]:
        raise ValueError(f"unexpected architectures: {config.get('architectures')}")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight index has no weight_map")
        weight_files = sorted({snapshot / str(value) for value in weight_map.values()})
    else:
        weight_files = [snapshot / "model.safetensors"]
    missing = [str(path) for path in weight_files if not path.is_file()]
    if missing:
        raise ValueError(f"snapshot is missing {len(missing)} weight files")
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    if weight_bytes < MINIMUM_WEIGHT_BYTES:
        raise ValueError(f"checkpoint weights are unexpectedly small: {weight_bytes}")
    return {
        "format_version": "nemotron3-super-snapshot-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": EXPECTED_ARCHITECTURE,
        "snapshot": str(snapshot.resolve()),
        "config_sha256": _sha256(config_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "weight_file_count": len(weight_files),
        "weight_bytes": weight_bytes,
        "status": "PASS",
    }


def snapshot_path(cache_dir: Path) -> Path:
    repository = "models--" + MODEL_ID.replace("/", "--")
    return cache_dir / repository / "snapshots" / MODEL_REVISION


def download(cache_dir: Path) -> Path:
    expected = snapshot_path(cache_dir)
    try:
        verify_snapshot(expected)
        print(f"DOWNLOAD_REUSED snapshot={expected}", flush=True)
        return expected
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "insufficient free cache space for the pinned BF16 checkpoint: "
            f"free_bytes={free_bytes} required_bytes={MINIMUM_FREE_BYTES}"
        )
    from huggingface_hub import snapshot_download

    resolved = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
            ignore_patterns=["*.gguf", "*.pth", "*.pt", "original/*"],
        )
    )
    if resolved.resolve() != expected.resolve():
        raise ValueError(f"download resolved to unexpected snapshot: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION) is None:
        raise ValueError("model revision is not immutable")
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_path(arguments.cache_dir)
    if not arguments.verify_only:
        snapshot = download(arguments.cache_dir)
    manifest = verify_snapshot(snapshot)
    if arguments.output.exists():
        previous = json.loads(arguments.output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("existing download manifest differs")
    else:
        _write_json_atomic(arguments.output, manifest)
    print("NEMOTRON_SNAPSHOT=" + str(snapshot.resolve()))
    print("NEMOTRON_DOWNLOAD_VERIFICATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
