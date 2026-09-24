#!/usr/bin/env python3
"""Import one verified recovery hub into the incremental dataset once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.agentic_dataset import (
    JsonlShardWriter,
    iter_sealed_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path, *, copy_mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    selected = copy_mode
    try:
        if copy_mode in {"auto", "hardlink"}:
            try:
                os.link(source, temporary)
                selected = "hardlink"
            except OSError:
                if copy_mode == "hardlink":
                    raise
                selected = "copy"
        if selected == "copy":
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return selected


def _recovery_manifest(hub: Path) -> tuple[dict[str, str], str]:
    path = hub / "publication-manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != "geodml-recovery-publication-v1":
        raise ValueError("unsupported recovery publication manifest")
    files = value.get("files")
    if not isinstance(files, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in files.items()
    ):
        raise ValueError("recovery publication manifest has an invalid file map")
    actual = {str(item.relative_to(hub)) for item in hub.rglob("*") if item.is_file()}
    if actual != set(files) | {"publication-manifest.json"}:
        raise ValueError("recovery hub has unexpected or missing files")
    for name, digest in files.items():
        source = hub / name
        if source.is_symlink() or not source.resolve().is_relative_to(hub.resolve()):
            raise ValueError(f"recovery file escapes its hub: {name}")
        if _sha256(source) != digest:
            raise ValueError(f"recovery file checksum mismatch: {name}")
    return files, _sha256(path)


def import_recovery(
    *,
    dataset_root: Path,
    recovery_dataset: Path,
    copy_mode: str = "auto",
) -> dict[str, Any]:
    """Copy or link verified normalized recovery bytes without accepting them."""

    if copy_mode not in {"auto", "copy", "hardlink"}:
        raise ValueError("copy mode must be auto, copy, or hardlink")
    if not (dataset_root / "contract.json").is_file():
        raise ValueError("dataset root is not initialized")
    hub = recovery_dataset / "hub"
    files, manifest_sha256 = _recovery_manifest(hub)
    import_id = "recovery-" + manifest_sha256[:24]
    receipt_path = dataset_root / "control" / "imports" / f"{import_id}.json"
    lock_path = dataset_root / "control" / "imports" / ".recovery-import.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        destination = dataset_root / "artifacts" / "recovery" / import_id / "hub"
        modes: dict[str, int] = {}
        identities = {**files, "publication-manifest.json": manifest_sha256}
        for name, digest in sorted(identities.items()):
            source = hub / name
            target = destination / name
            if target.exists():
                if target.is_symlink() or _sha256(target) != digest:
                    raise ValueError(f"existing recovery import conflicts: {target}")
                modes["existing"] = modes.get("existing", 0) + 1
                continue
            mode = _atomic_copy(source, target, copy_mode=copy_mode)
            if _sha256(target) != digest:
                raise ValueError(f"copied recovery file checksum mismatch: {name}")
            modes[mode] = modes.get(mode, 0) + 1
        summary_path = hub / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record = {
            "import_id": import_id,
            "source_format_version": summary.get("format_version"),
            "source_status": summary.get("status"),
            "publication_manifest_sha256": manifest_sha256,
            "artifact_root": str(destination.relative_to(dataset_root)),
            "file_count": len(identities),
            "stages": summary.get("stages", []),
            "findings": summary.get("findings", {}),
            "scientific_completion": "unverified",
            "ledger_completions_created": 0,
            "local_forensics": {
                "included": False,
                "reference_kind": "local-content-addressed-snapshot",
                "snapshot_sha256": summary.get("capture", {}).get(
                    "snapshot_sha256"
                ),
                "reason": (
                    "raw forensic packs remain local; normalized records and their "
                    "artifact index are included in the imported recovery hub"
                ),
            },
            "limitations": summary.get("limitations", []),
        }
        existing = {
            row.get("import_id"): row
            for row in iter_sealed_rows(dataset_root, "legacy_imports")
        }
        prior = existing.get(import_id)
        if prior is not None and prior != record:
            raise ValueError("existing legacy import record conflicts")
        if prior is None:
            writer_id = import_id
            directory = dataset_root / "data" / "legacy_imports"
            active = directory / f"part-{writer_id}-000000.jsonl.inprogress"
            writer = JsonlShardWriter(
                dataset_root,
                table="legacy_imports",
                writer_id=writer_id,
                shard_sequence=0,
            )
            if active.exists() and writer.line_number:
                rows = [json.loads(line) for line in active.read_text().splitlines()]
                if len(rows) != 1 or rows[0].get("row") != record:
                    writer.close()
                    raise ValueError("interrupted recovery import record conflicts")
            else:
                writer.append(
                    record,
                    transaction_id=import_id,
                    record_id=f"legacy-import-{manifest_sha256}",
                )
            writer.seal()
        receipt = {
            "format_version": "geodml-agentic-recovery-import-receipt-v1",
            "status": "complete",
            "import_id": import_id,
            "source_recovery_dataset": str(recovery_dataset.resolve()),
            "destination": str(destination.resolve()),
            "publication_manifest_sha256": manifest_sha256,
            "file_count": len(identities),
            "copy_modes": modes,
            "scientific_completion_imported": False,
        }
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            comparable = {**receipt, "copy_modes": saved.get("copy_modes")}
            if saved != comparable:
                raise ValueError("existing recovery import receipt conflicts")
            return saved
        _atomic_json(receipt_path, receipt)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recovery-dataset", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=("auto", "copy", "hardlink"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = import_recovery(
        dataset_root=args.dataset_root,
        recovery_dataset=args.recovery_dataset,
        copy_mode=args.copy_mode,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
