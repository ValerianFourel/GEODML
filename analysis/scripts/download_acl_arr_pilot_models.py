#!/usr/bin/env python3
"""Pin and download the four models for the ACL ARR pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any


MODEL_IDS = (
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-Small-4-119B-2603",
    "Qwen/Qwen3.8-27B",
)


def _write_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_locks(locks: list[dict[str, Any]]) -> None:
    if tuple(str(row.get("model_id")) for row in locks) != MODEL_IDS:
        raise ValueError("model lock contains a different model panel")
    for row in locks:
        revision = str(row.get("revision", ""))
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(
                f"invalid immutable revision for {row.get('model_id')}: {revision}"
            )


def verify_downloads(run_root: Path) -> None:
    models_path = run_root / "models.json"
    lock_path = run_root / "model-snapshots.json"
    models = _read_json(models_path).get("models")
    locks = _read_json(lock_path).get("models")
    if not isinstance(models, list) or not isinstance(locks, list):
        raise ValueError("models.json and model-snapshots.json must contain models lists")
    if len(models) != 4 or len(locks) != 4:
        raise ValueError("expected four models and four snapshot locks")
    _validate_locks(locks)
    for model, lock in zip(models, locks, strict=True):
        if model.get("model_id") != lock.get("model_id"):
            raise ValueError("model and lock order differs")
        if model.get("model_revision") != lock.get("revision"):
            raise ValueError(f"revision mismatch for {model.get('model_id')}")
        snapshot = Path(str(lock.get("snapshot", "")))
        if not snapshot.is_dir():
            raise ValueError(f"missing snapshot for {model.get('model_id')}: {snapshot}")


def download_models(run_root: Path, template_path: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    run_root.mkdir(parents=True, exist_ok=True)
    models_path = run_root / "models.json"
    lock_path = run_root / "model-snapshots.json"

    if lock_path.exists():
        locks_value = _read_json(lock_path).get("models")
        if not isinstance(locks_value, list):
            raise ValueError("model lock must contain a models list")
        locks = locks_value
        _validate_locks(locks)
    else:
        api = HfApi()
        locks = []
        for model_id in MODEL_IDS:
            revision = str(api.model_info(model_id).sha)
            locks.append(
                {"model_id": model_id, "revision": revision, "snapshot": None}
            )
        _validate_locks(locks)
        _write_atomic(lock_path, {"models": locks})

    revisions = {str(row["model_id"]): str(row["revision"]) for row in locks}
    config = _read_json(template_path)
    configured_models = config.get("models")
    if not isinstance(configured_models, list):
        raise ValueError("model template must contain a models list")
    configured_ids = tuple(str(row.get("model_id")) for row in configured_models)
    if configured_ids != MODEL_IDS:
        raise ValueError(f"model template mismatch: {configured_ids}")
    for row in configured_models:
        row["model_revision"] = revisions[str(row["model_id"])]
    _write_atomic(models_path, config)

    cache_dir = os.environ["HF_HUB_CACHE"]
    for row in locks:
        model_id = str(row["model_id"])
        revision = str(row["revision"])
        print(f"DOWNLOAD_START model={model_id} revision={revision}", flush=True)
        row["snapshot"] = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            ignore_patterns=[
                "*.gguf",
                "*.pth",
                "*.pt",
                "original/*",
                "consolidated-*",
            ],
        )
        _write_atomic(lock_path, {"models": locks})
        print(f"DOWNLOAD_DONE model={model_id} snapshot={row['snapshot']}", flush=True)
    verify_downloads(run_root)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-template", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.verify_only and arguments.model_template is None:
        parser.error("--model-template is required unless --verify-only is set")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    if arguments.verify_only:
        verify_downloads(arguments.run_root)
    else:
        download_models(arguments.run_root, arguments.model_template)
    print("MODEL_PANEL_VERIFICATION=PASS models=4")


if __name__ == "__main__":
    main()
