#!/usr/bin/env python3
"""Join extracted page text to a frozen SERP and write immutable document sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_freeze import (  # noqa: E402
    build_frozen_document_sets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serp", required=True, help="Frozen SERP Parquet or JSONL.")
    parser.add_argument(
        "--page-text", required=True, help="URL-keyed extracted page-text Parquet or JSONL."
    )
    parser.add_argument("--minimum-documents", type=int, default=11)
    parser.add_argument("--maximum-documents", type=int, default=20)
    parser.add_argument("--max-document-characters", type=int, default=12000)
    parser.add_argument(
        "--allow-snippet-fallback",
        action="store_true",
        help="Explicitly permit snippets where extracted page text is unavailable.",
    )
    parser.add_argument("--source-git-commit", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    serp_path = Path(args.serp).resolve()
    pages_path = Path(args.page_text).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen document output: {output}")
    output.mkdir(parents=True)
    sets_path = output / "frozen_document_sets.jsonl"
    manifest_path = output / "document_freeze_manifest.json"
    try:
        frozen, summary = build_frozen_document_sets(
            _rows(serp_path),
            _rows(pages_path),
            minimum_documents=args.minimum_documents,
            maximum_documents=args.maximum_documents,
            max_document_characters=args.max_document_characters,
            allow_snippet_fallback=args.allow_snippet_fallback,
            search_snapshot_sha256=_sha256(serp_path),
        )
        _atomic_text(
            sets_path,
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in frozen
            ),
        )
        manifest = {
            **summary,
            "status": "PASS",
            "scientific_result": False,
            "source_git_commit": args.source_git_commit or _git_commit(),
            "sources": {
                "serp": {"path": str(serp_path), "sha256": _sha256(serp_path)},
                "page_text": {
                    "path": str(pages_path),
                    "sha256": _sha256(pages_path),
                },
            },
            "artifact": {
                "path": str(sets_path),
                "sha256": _sha256(sets_path),
                "size_bytes": sets_path.stat().st_size,
            },
        }
        _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    print("DOCUMENT_FREEZE=PASS")
    print(f"KEYWORDS={summary['complete_keyword_count']}")
    print(f"DOCUMENTS={summary['document_count']}")
    print(f"SNIPPET_FALLBACKS={summary['snippet_fallback_count']}")
    print(f"DOCUMENT_SETS={sets_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
