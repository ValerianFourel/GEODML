#!/usr/bin/env python3
"""Quarantine one known corrupt readiness-validation checkpoint fail-closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _jsonl_diagnostics(path: Path) -> tuple[int, int, str | None]:
    newline_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            newline_count += chunk.count(b"\n")
    row_count = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at line {line_number}")
                row_count += 1
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return newline_count, row_count, f"{type(exc).__name__}: {exc}"
    return newline_count, row_count, None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary manifest already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def quarantine_corrupt_validation(
    source: str | Path,
    quarantine: str | Path,
    manifest: str | Path,
    *,
    source_job_id: str,
    recovery_job_id: str,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    quarantine_path = Path(quarantine).resolve()
    manifest_path = Path(manifest).resolve()
    completion_manifest = source_path.with_suffix(source_path.suffix + ".manifest.json")
    if source_path == quarantine_path:
        raise ValueError("source and quarantine paths must differ")

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("source", {}).get("path") != str(source_path)
            or payload.get("quarantine", {}).get("path") != str(quarantine_path)
            or payload.get("source_job_id") != source_job_id
        ):
            raise ValueError("existing quarantine manifest has a different identity")
        if not quarantine_path.is_file():
            raise ValueError("quarantined checkpoint recorded by manifest is missing")
        expected = payload["quarantine"]
        observed = _identity(quarantine_path)
        if any(observed[key] != expected.get(key) for key in ("sha256", "size_bytes")):
            raise ValueError("quarantined checkpoint identity changed")
        if source_path.exists():
            _, _, parse_error = _jsonl_diagnostics(source_path)
            if parse_error is not None:
                raise ValueError("rebuilt validation checkpoint is also corrupt")
            return {**payload, "current_status": "quarantined-source-rebuilt"}
        return {**payload, "current_status": "already-quarantined"}

    if quarantine_path.exists():
        raise ValueError("quarantine exists without its audit manifest")
    if completion_manifest.exists():
        raise ValueError("refusing to quarantine a completed validation shard")
    if not source_path.is_file():
        raise ValueError(f"corrupt validation checkpoint is missing: {source_path}")

    newline_count, row_count, parse_error = _jsonl_diagnostics(source_path)
    if source_path.stat().st_size <= 0:
        raise ValueError("corrupt checkpoint signature requires a nonempty file")
    if newline_count != 0 or row_count != 0 or parse_error is None:
        raise ValueError(
            "checkpoint does not match the audited zero-record corruption signature"
        )

    source_identity = _identity(source_path)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.replace(quarantine_path)
    quarantine_identity = _identity(quarantine_path)
    if source_identity["sha256"] != quarantine_identity["sha256"]:
        raise ValueError("quarantine rename did not preserve checkpoint bytes")
    payload = {
        "format_version": "readiness-validation-quarantine-v1",
        "created_at": _now(),
        "source_job_id": source_job_id,
        "recovery_job_id": recovery_job_id,
        "reason": "nonempty validation JSONL had zero records and failed JSON parsing",
        "parse_error": parse_error,
        "newline_count": newline_count,
        "parsed_row_count": row_count,
        "completion_manifest_present": False,
        "source": source_identity,
        "quarantine": quarantine_identity,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, payload)
    return {**payload, "current_status": "quarantined"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--recovery-job-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = quarantine_corrupt_validation(
        args.source,
        args.quarantine,
        args.manifest,
        source_job_id=args.source_job_id,
        recovery_job_id=args.recovery_job_id,
    )
    print(f"quarantine_status={result['current_status']}")
    print(f"quarantine={result['quarantine']['path']}")
    print(f"sha256={result['quarantine']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
