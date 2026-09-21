#!/usr/bin/env python3
"""Freeze the complete agentic generator-cell queue for restartable waves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_generation_tasks import (  # noqa: E402
    build_cells,
    load_calibration_prompts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--selection-records-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--prompt-selection-seed", type=int, default=20260912)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite task queue: {output}")
    try:
        prompts_path = arguments.prompts_jsonl.resolve()
        selection_path = arguments.selection_records_jsonl.resolve()
        prompts = load_calibration_prompts(
            prompts_path,
            selection_path,
            prompt_count=arguments.prompt_count,
            seed=arguments.prompt_selection_seed,
        )
        cells = build_cells(prompts)
        output.mkdir(parents=True)
        tasks_path = output / "tasks.jsonl"
        with tasks_path.open("x", encoding="utf-8", buffering=1) as stream:
            for cell in cells:
                stream.write(
                    json.dumps(
                        {"cell_id": cell.cell_id, **cell.core},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path = output / "run_manifest.json"
        _atomic_json(
            manifest_path,
            {
                "format_version": "agentic-generation-task-queue-v1",
                "status": "planned",
                "scientific_result": False,
                "git_commit": os.environ.get("GEODML_EXECUTION_COMMIT"),
                "prompt_count": len(prompts),
                "cell_count": len(cells),
                "task_id_field": "cell_id",
                "prompt_selection_seed": arguments.prompt_selection_seed,
                "sources": {
                    "prompts_jsonl": {
                        "path": str(prompts_path),
                        "sha256": _sha256(prompts_path),
                    },
                    "selection_records_jsonl": {
                        "path": str(selection_path),
                        "sha256": _sha256(selection_path),
                    },
                },
                "tasks": {
                    "path": str(tasks_path),
                    "sha256": _sha256(tasks_path),
                },
            },
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(f"PROMPTS={len(prompts)}")
    print(f"CELLS={len(cells)}")
    print(f"TASKS={tasks_path}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
