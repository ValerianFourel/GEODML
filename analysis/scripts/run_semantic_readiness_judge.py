#!/usr/bin/env python3
"""Run one pinned high-quality LLM judge slot with resumable task caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    ReadinessLabelTask,
    parse_readiness_judgment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-slot", required=True)
    parser.add_argument("--backend", choices=("local", "api", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=("full", "4bit"), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists() and not args.resume:
        raise SystemExit(f"output directory exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "task_cache"
    cache.mkdir(exist_ok=True)
    all_tasks = tuple(
        ReadinessLabelTask(**row) for row in _read_jsonl(Path(args.tasks).resolve())
    )
    tasks = [item for item in all_tasks if item.judge_slot == args.judge_slot]
    if not tasks:
        raise SystemExit(f"no tasks for judge slot {args.judge_slot!r}")
    stop = None if args.limit is None else args.start_index + args.limit
    tasks = tasks[args.start_index:stop]
    from interpretability.utils import make_ranker

    ranker = make_ranker(args.backend, args.model, precision=args.precision)
    responses = []
    for index, task in enumerate(tasks, 1):
        cache_path = cache / f"{task.task_id.replace(':', '_')}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("task_id") != task.task_id or cached.get("model") != args.model:
                raise SystemExit(f"cache identity mismatch: {cache_path}")
            parse_readiness_judgment(task, str(cached["raw_response"]))
            responses.append(cached)
            continue
        attempts = []
        for attempt in range(args.maximum_attempts):
            prompt = task.prompt
            if attempt:
                prompt += (
                    "\nYour previous response violated the exact JSON contract. "
                    "Return one valid object only; do not add markdown or commentary."
                )
            raw = str(
                ranker.rank(
                    prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            )
            try:
                parse_readiness_judgment(task, raw)
            except ValueError as exc:
                attempts.append({"attempt": attempt + 1, "error": str(exc), "raw": raw})
                continue
            row = {
                "task_id": task.task_id,
                "item_id": task.item_id,
                "judge_slot": task.judge_slot,
                "model": args.model,
                "backend": args.backend,
                "precision": args.precision if args.backend == "local" else None,
                "raw_response": raw,
                "rejected_attempts": attempts,
            }
            _atomic_json(cache_path, row)
            responses.append(row)
            break
        else:
            failure = cache_path.with_suffix(".failed.json")
            _atomic_json(
                failure,
                {
                    "task_id": task.task_id,
                    "model": args.model,
                    "backend": args.backend,
                    "attempts": attempts,
                },
            )
            raise SystemExit(f"judge exhausted attempts: {failure}")
        if index % 50 == 0 or index == len(tasks):
            print(f"[{index}/{len(tasks)}] {args.judge_slot}", flush=True)
    _atomic_jsonl(output / "judge_responses.jsonl", responses)
    _atomic_json(
        output / "run_manifest.json",
        {
            "judge_slot": args.judge_slot,
            "model": args.model,
            "backend": args.backend,
            "precision": args.precision if args.backend == "local" else None,
            "task_file": str(Path(args.tasks).resolve()),
            "task_file_sha256": _sha256_file(Path(args.tasks).resolve()),
            "start_index": args.start_index,
            "limit": args.limit,
            "completed_count": len(responses),
            "scientific_result": False,
        },
    )
    print(f"output: {output}")
    return 0


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
