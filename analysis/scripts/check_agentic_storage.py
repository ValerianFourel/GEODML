#!/usr/bin/env python3
"""Capture storage admission health or explicitly acknowledge fixed incidents."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from analysis.interpretability.pipeline.agentic_storage import (
    acknowledge_incidents,
    storage_health,
)


def _write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite storage snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--minimum-free-inodes", type=int, default=100_000)
    parser.add_argument("--quota-evidence", type=Path)
    parser.add_argument("--acknowledge-incident", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_free_gib < 0:
        raise ValueError("minimum free GiB must be non-negative")
    acknowledged = acknowledge_incidents(
        args.dataset_root, args.acknowledge_incident
    ) if args.acknowledge_incident else []
    quota = (
        json.loads(args.quota_evidence.read_text(encoding="utf-8"))
        if args.quota_evidence else None
    )
    result = storage_health(
        args.dataset_root,
        minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
        minimum_free_inodes=args.minimum_free_inodes,
        quota_evidence=quota,
    )
    result["acknowledged_this_invocation"] = acknowledged
    if args.output:
        _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
