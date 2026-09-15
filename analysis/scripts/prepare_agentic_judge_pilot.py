#!/usr/bin/env python3
"""Freeze at most 24 blinded Nemotron judge tasks from one completed Qwen shard.

This plumbing pilot performs no inference and makes no judge-quality claim.
Production queue preparation remains in prepare_agentic_judge_tasks.py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_judging import (
    FORMAT_VERSION,
    AgenticJudgeModel,
    _atomic_json,
    _atomic_jsonl,
    _digest,
    _file_identity,
    _load_prompts,
    _task_and_mapping,
)
from analysis.scripts.prepare_agentic_judge_tasks import _merged_prompts

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
GENERATOR_MODEL_ID = "Qwen/Qwen3.8-27B"
FACTORS = frozenset(
    itertools.product(
        ("Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"),
        ("duckduckgo", "searxng"),
        ("natural", "ablated", "shuffled"),
    )
)


def prepare_pilot(
    *,
    generator_root: Path,
    prompts_jsonl: Path,
    selection_records_jsonl: Path,
    output_dir: Path,
    prompt_count: int = 2,
    master_seed: int = 20260915,
) -> Path:
    """Validate inputs, select extreme available axis bins, and freeze the queue."""
    if type(prompt_count) is not int or prompt_count not in (1, 2):
        raise ValueError("pilot prompt-count must be 1 or 2, at most 24 cases")
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite judge pilot: {output}")
    root = generator_root.resolve()
    manifest_path = root / "run_manifest.json"
    source_manifest_identity = _file_identity(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("model_id") != GENERATOR_MODEL_ID
        or manifest.get("remaining_count") != 0
        or manifest.get("failed_cell_ids", [])
        or type(manifest.get("prompt_shard_index")) is not int
        or manifest["prompt_shard_index"] not in (0, 1, 2)
    ):
        raise ValueError(
            "pilot requires one complete Qwen source shard from shards 0-2"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit", ""))):
        raise ValueError("source shard lacks an immutable Git commit")
    result_paths = sorted((root / "results").glob("*.json"))
    if (
        not result_paths
        or type(manifest.get("completed_count")) is not int
        or manifest["completed_count"] != len(result_paths)
        or manifest.get("cell_count") != len(result_paths)
    ):
        raise ValueError("source manifest and result counts disagree")

    sources = {
        "generator_manifest": source_manifest_identity,
        "prompts": _file_identity(prompts_jsonl),
        "selection_records": _file_identity(selection_records_jsonl),
    }
    rows = _merged_prompts(prompts_jsonl, selection_records_jsonl)
    axis_bins = {}
    for row in rows:
        axis_bin = row.get("axis_bin")
        if type(axis_bin) is not int or axis_bin < 0:
            raise ValueError("pilot selection requires non-negative integer axis bins")
        axis_bins[row["candidate_id"]] = axis_bin
        row["axis_bin"] = str(axis_bin)
    prompts = _load_prompts(rows)
    by_prompt = defaultdict(dict)
    seen_cells = set()
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        cell_id = result.get("cell_id")
        prompt_id = result.get("prompt_id")
        if (
            not isinstance(cell_id, str)
            or cell_id != path.stem
            or cell_id in seen_cells
        ):
            raise ValueError("source result has invalid or duplicate cell identity")
        seen_cells.add(cell_id)
        if prompt_id not in prompts:
            raise ValueError(f"source result references unknown prompt: {prompt_id}")
        factor = tuple(result.get(name) for name in ("method", "engine", "condition"))
        if factor not in FACTORS or factor in by_prompt[prompt_id]:
            raise ValueError(
                f"source result has invalid or duplicate factorial cell: {cell_id}"
            )
        by_prompt[prompt_id][factor] = path
    if any(set(cells) != FACTORS for cells in by_prompt.values()):
        raise ValueError("source shard has incomplete per-prompt factorial coverage")
    if len(by_prompt) < prompt_count:
        raise ValueError("source shard has too few completed prompts")

    def selection_key(prompt_id):
        return hashlib.sha256(
            f"{master_seed}:judge-pilot:{prompt_id}".encode()
        ).hexdigest()

    ordered = sorted(by_prompt, key=lambda pid: (axis_bins[pid], selection_key(pid)))
    selected = [ordered[0]]
    if prompt_count == 2:
        selected.append(ordered[-1])
    selected.sort(key=lambda pid: (axis_bins[pid], pid))
    tasks, mappings = [], []
    for prompt_id in selected:
        for path in sorted(by_prompt[prompt_id].values()):
            task, mapping = _task_and_mapping(
                path,
                prompt=prompts[prompt_id],
                generator_model_id=GENERATOR_MODEL_ID,
                master_seed=master_seed,
            )
            tasks.append(task)
            mappings.append(mapping)
    tasks.sort(key=lambda task: task.judge_task_id)
    mappings.sort(key=lambda mapping: mapping.judge_task_id)
    if len({task.judge_task_id for task in tasks}) != prompt_count * 12:
        raise ValueError("pilot task identities are not unique")
    # Do not freeze a plan against source metadata changed during preparation.
    for identity in sources.values():
        if _file_identity(Path(identity["path"])) != identity:
            raise ValueError("pilot source changed during preparation")
    git_commit = subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    model = asdict(
        AgenticJudgeModel(
            role="bulk",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )
    )
    pilot = {
        "selected_prompt_ids": selected,
        "prompt_count": prompt_count,
        "cells_per_prompt": 12,
        "selected_axis_bins": [axis_bins[p] for p in selected],
        "selection_method": "extreme-available-axis-bins-seeded-ties-v1",
        "source_generator_root": str(root),
        "source_git_commit": manifest["git_commit"],
        "source_completed_count": len(result_paths),
        "scope": "pipeline-plumbing-only-not-judge-quality-validation",
    }
    plan_id = (
        "agentic-judge-pilot-"
        + _digest(
            {
                "format_version": FORMAT_VERSION,
                "master_seed": master_seed,
                "model": model,
                "task_ids": [task.judge_task_id for task in tasks],
            }
        )[:24]
    )
    output.mkdir(parents=True)
    bulk_path = output / "bulk_tasks.jsonl"
    mapping_path = output / "private_mapping.jsonl"
    _atomic_jsonl(bulk_path, [task.to_dict() for task in tasks])
    _atomic_jsonl(mapping_path, [asdict(mapping) for mapping in mappings])
    destination = output / "run_manifest.json"
    _atomic_json(
        destination,
        {
            "format_version": FORMAT_VERSION,
            "judge_plan_id": plan_id,
            "status": "planned",
            "scientific_result": False,
            "pilot_only": True,
            "git_commit": git_commit,
            "master_seed": master_seed,
            "bulk_model": model,
            "blinding": "generator-treatment-and-ranking-hidden-v1",
            "pilot": pilot,
            "sources": sources,
            "summary": {"bulk_task_count": len(tasks), "validation_task_count": 0},
            "artifacts": {
                "bulk_tasks": _file_identity(bulk_path),
                "private_mapping": _file_identity(mapping_path),
            },
        },
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "generator-root",
        "prompts-jsonl",
        "selection-records-jsonl",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260915)
    args = parser.parse_args()
    try:
        manifest_path = prepare_pilot(**vars(args))
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    print(f"NEMOTRON_JUDGE_PILOT_TASKS={args.prompt_count * 12}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
