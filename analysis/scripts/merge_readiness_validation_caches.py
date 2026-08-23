#!/usr/bin/env python3
"""Safely union immutable readiness-validator cache records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"validator cache record must be an object: {path}")
    identity = payload.get("identity")
    review = payload.get("review")
    if not isinstance(identity, dict) or not isinstance(review, dict):
        raise ValueError(f"validator cache record lacks identity/review: {path}")
    expected_name = f"{_stable_hash(identity)}.json"
    if path.name != expected_name:
        raise ValueError(
            f"validator cache filename does not match its identity: {path}"
        )
    return payload


def _validate_judge(
    payload: Mapping[str, Any],
    *,
    judge_id: str | None,
    judge_model: str | None,
    path: Path,
) -> None:
    identity = payload["identity"]
    if judge_id is not None and identity.get("judge_id") != judge_id:
        raise ValueError(f"unexpected judge id in {path}")
    if judge_model is not None and identity.get("judge_model") != judge_model:
        raise ValueError(f"unexpected judge model in {path}")


def _choose_record(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    if existing["identity"] != incoming["identity"]:
        raise ValueError(f"validator cache identity collision: {path}")
    if existing == incoming:
        return existing, "identical"

    existing_terminal = bool(existing.get("terminal_parse_failure"))
    incoming_terminal = bool(incoming.get("terminal_parse_failure"))
    if existing_terminal != incoming_terminal:
        if incoming_terminal:
            return existing, "kept_nonterminal"
        return incoming, "replaced_terminal"

    if existing["review"] != incoming["review"]:
        raise ValueError(
            "conflicting validator reviews for the same immutable identity: "
            f"{path}"
        )

    # Failure traces are diagnostic metadata.  Equal identities and reviews
    # are scientifically equivalent; retain the already published record.
    return existing, "equivalent"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def merge_validation_caches(
    sources: Sequence[str | Path],
    destination: str | Path,
    *,
    judge_id: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(dict.fromkeys(Path(value).resolve() for value in sources))
    counters = {
        "source_file_count": 0,
        "copied_count": 0,
        "identical_count": 0,
        "equivalent_count": 0,
        "kept_nonterminal_count": 0,
        "replaced_terminal_count": 0,
    }

    for source in source_paths:
        if not source.is_dir():
            raise ValueError(f"validator cache source is not a directory: {source}")
        if source == destination_path:
            continue
        for source_file in sorted(source.glob("*.json")):
            counters["source_file_count"] += 1
            incoming = _read_record(source_file)
            _validate_judge(
                incoming,
                judge_id=judge_id,
                judge_model=judge_model,
                path=source_file,
            )
            destination_file = destination_path / source_file.name
            if not destination_file.exists():
                # copy2 into a private temporary file, then publish atomically.
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{source_file.name}.",
                    suffix=".tmp",
                    dir=destination_path,
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    shutil.copy2(source_file, temporary)
                    temporary.replace(destination_file)
                finally:
                    temporary.unlink(missing_ok=True)
                counters["copied_count"] += 1
                continue

            existing = _read_record(destination_file)
            _validate_judge(
                existing,
                judge_id=judge_id,
                judge_model=judge_model,
                path=destination_file,
            )
            chosen, resolution = _choose_record(
                existing, incoming, path=destination_file
            )
            counters[f"{resolution}_count"] += 1
            if chosen is incoming:
                _atomic_write(destination_file, incoming)

    report = {
        "format_version": "readiness-validator-cache-merge-v1",
        "destination": str(destination_path),
        "sources": [str(path) for path in source_paths],
        "judge_id": judge_id,
        "judge_model": judge_model,
        **counters,
        "destination_file_count": sum(
            1 for _ in destination_path.glob("*.json")
        ),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--destination", required=True)
    parser.add_argument("--judge-id")
    parser.add_argument("--judge-model")
    parser.add_argument("--report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = merge_validation_caches(
        args.source,
        args.destination,
        judge_id=args.judge_id,
        judge_model=args.judge_model,
    )
    if args.report:
        _atomic_write(Path(args.report).resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
