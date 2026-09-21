#!/usr/bin/env python3
"""Freeze and select the approved Nemotron concurrency sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.adaptive_inference import (
    SWEEP_CONCURRENCIES,
    select_sweep_concurrency,
    stratified_sweep_sample,
)
from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeTask,
    render_agentic_judge_prompt,
)
from analysis.interpretability.pipeline.inference_wave import (
    _atomic_json,
    _atomic_jsonl,
)

FORMAT_VERSION = "agentic-nemotron-concurrency-sweep-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _token_count(tokenizer: Any, task: AgenticJudgeTask) -> int:
    messages = [{"role": "user", "content": render_agentic_judge_prompt(task)}]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(tokens, "shape"):
        return int(tokens.shape[-1])
    return len(tokens)


def prepare(plan_root: Path, tokenizer_snapshot: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sweep: {output}")
    manifest_path = plan_root / "run_manifest.json"
    tasks_path = plan_root / "bulk_tasks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bulk = manifest.get("bulk_model", {})
    if (bulk.get("model_id"), bulk.get("model_revision")) != (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
    ):
        raise ValueError("judge plan is not pinned to the approved Nemotron revision")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot,
        local_files_only=True,
        trust_remote_code=True,
    )
    rows = _read_jsonl(tasks_path)
    tasks = [AgenticJudgeTask.from_dict(row) for row in rows]
    counts = {task.judge_task_id: _token_count(tokenizer, task) for task in tasks}
    sample = stratified_sweep_sample(counts)
    indexed = {row["judge_task_id"]: row for row in rows}
    output.mkdir(parents=True)
    measured_path, warmup_path = (
        output / "measured_tasks.jsonl",
        output / "warmup_tasks.jsonl",
    )
    _atomic_jsonl(measured_path, [indexed[key] for key in sample["measured"]])
    _atomic_jsonl(warmup_path, [indexed[key] for key in sample["warmup"]])
    value = {
        "format_version": FORMAT_VERSION,
        "status": "prepared",
        "scientific_result": False,
        "judge_plan": str(manifest_path.resolve()),
        "judge_plan_sha256": _sha256(manifest_path),
        "source_tasks_sha256": _sha256(tasks_path),
        "tokenizer_snapshot": str(tokenizer_snapshot.resolve()),
        "tokenizer_files": {
            name: _sha256(tokenizer_snapshot / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "max_model_len": 73728,
        "max_output_tokens": 2048,
        "disable_thinking": True,
        "hard_cap_seconds": 1800,
        "concurrencies": list(SWEEP_CONCURRENCIES),
        "worker_assignment": {
            str(index): value for index, value in enumerate(SWEEP_CONCURRENCIES)
        },
        "production_worker": 4,
        "measured_tasks": {
            "path": str(measured_path.resolve()),
            "sha256": _sha256(measured_path),
            "rows": 64,
        },
        "warmup_tasks": {
            "path": str(warmup_path.resolve()),
            "sha256": _sha256(warmup_path),
            "rows": 4,
        },
        "input_token_counts": {
            key: counts[key] for key in sample["measured"] + sample["warmup"]
        },
    }
    value["sample_sha256"] = hashlib.sha256(
        json.dumps(
            {"measured": sample["measured"], "warmup": sample["warmup"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _atomic_json(output / "run_manifest.json", value)
    return value


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def select(sweep_root: Path, profile_sha256: str) -> dict[str, Any]:
    manifest = json.loads((sweep_root / "run_manifest.json").read_text())
    results = []
    for concurrency in SWEEP_CONCURRENCIES:
        path = sweep_root / f"concurrency-{concurrency}/result.json"
        if path.is_file():
            results.append(json.loads(path.read_text()))
    decision = select_sweep_concurrency(
        results,
        sample_sha256=manifest["sample_sha256"],
        profile_sha256=profile_sha256,
    )
    decision.update(
        format_version="agentic-nemotron-concurrency-selection-v1",
        candidates_received=len(results),
        expected_candidates=4,
        complete_candidate_set=len(results) == 4,
    )
    destination = sweep_root / "selection.json"
    if destination.is_file():
        if json.loads(destination.read_text()) != decision:
            raise ValueError("existing immutable sweep decision differs")
    else:
        _exclusive_json(destination, decision)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--judge-plan-root", type=Path, required=True)
    create.add_argument("--tokenizer-snapshot", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--sweep-root", type=Path, required=True)
    choose.add_argument("--profile-sha256", required=True)
    args = parser.parse_args(argv)
    result = (
        prepare(
            args.judge_plan_root.resolve(),
            args.tokenizer_snapshot.resolve(),
            args.output.resolve(),
        )
        if args.command == "prepare"
        else select(args.sweep_root.resolve(), args.profile_sha256)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
