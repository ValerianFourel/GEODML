#!/usr/bin/env python3
"""Freeze a fixed or allocation-filling Nemotron pilot from a complete Qwen shard.

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
from analysis.scripts.run_acl_arr_vllm import (
    _agentic_judge_context,
    _prepare_agentic_judge,
    _validate_resume,
)

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


def _excluded_judgments(paths, tasks, *, master_seed):
    """Verify previous task, request, model, seed, raw output, and parsed result."""
    current = {task.judge_task_id: task for task in tasks}
    excluded = set()
    sources = []
    for path in paths:
        path = Path(path).resolve()
        if path.name != "outcomes.jsonl":
            raise ValueError("excluded judgments must be a runner outcomes.jsonl journal")
        runtime_path = path.parent / "run_manifest.json"
        runtime_identity = _file_identity(runtime_path)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        identity = runtime.get("resume_identity", {})
        expected = {
            "pipeline": "agentic-judge-bulk",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "judge_role": "bulk",
            "fake_backend": False,
            "max_output_tokens": 2048,
            "disable_thinking": True,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("excluded judgment model or request configuration mismatch")
        plan_path = Path(runtime["source_manifest"])
        prior_tasks_path = Path(runtime["tasks"]["path"])
        source = {
            "outcomes": _file_identity(path),
            "runtime_manifest": runtime_identity,
            "source_manifest": _file_identity(plan_path),
            "tasks": _file_identity(prior_tasks_path),
            "failures": _file_identity(path.parent / "failures.jsonl"),
            "attempts": _file_identity(path.parent / "attempts.jsonl"),
        }
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("master_seed") != master_seed:
            raise ValueError("excluded judgment master seed mismatch")
        if (
            source["source_manifest"]["sha256"] != identity.get("source_manifest_sha256")
            or source["tasks"]["sha256"] != identity.get("tasks_sha256")
        ):
            raise ValueError("excluded judgment source artifact hash mismatch")
        prior_tasks, _, model_id, revision = _agentic_judge_context(
            plan_path, prior_tasks_path, judge_role="bulk"
        )
        if (model_id, revision) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("excluded judgment plan model mismatch")
        prior = {task.judge_task_id: task for task in prior_tasks}
        if len(prior) != len(prior_tasks):
            raise ValueError("excluded judgment source contains duplicate tasks")
        completed = _validate_resume(
            path.parent,
            identity,
            prior_tasks,
            lambda task: _prepare_agentic_judge(task, max_tokens=2048),
            "judge_task_id",
        )
        for task_id in completed:
            if current.get(task_id) != prior[task_id]:
                raise ValueError("excluded judgment does not match a current task")
            if task_id in excluded:
                raise ValueError("duplicate or conflicting excluded judgment")
            excluded.add(task_id)
        sources.append(source)
    return excluded, sources


def prepare_pilot(
    *,
    generator_root: Path,
    prompts_jsonl: Path,
    selection_records_jsonl: Path,
    output_dir: Path,
    prompt_count: int = 2,
    master_seed: int = 20260915,
    all_available: bool = False,
    exclude_outcomes: tuple[Path, ...] = (),
) -> Path:
    """Freeze extreme-bin prompts or all prompts, excluding verified prior work."""
    if type(prompt_count) is not int or prompt_count not in (1, 2):
        raise ValueError("pilot prompt-count must be 1 or 2, at most 24 cases")
    if exclude_outcomes and not all_available:
        raise ValueError("exclude-outcomes requires all-available throughput mode")
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
    if len(by_prompt) < (1 if all_available else prompt_count):
        raise ValueError("source shard has too few completed prompts")

    def selection_key(prompt_id):
        return hashlib.sha256(
            f"{master_seed}:judge-pilot:{prompt_id}".encode()
        ).hexdigest()

    ordered = sorted(by_prompt, key=lambda pid: (axis_bins[pid], selection_key(pid)))
    selected = ordered if all_available else [ordered[0]]
    if not all_available and prompt_count == 2:
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
    if len({task.judge_task_id for task in tasks}) != len(selected) * 12:
        raise ValueError("pilot task identities are not unique")
    available_task_count = len(tasks)
    excluded, exclusion_sources = _excluded_judgments(
        exclude_outcomes, tasks, master_seed=master_seed
    )
    excluded_mappings = [mapping for mapping in mappings if mapping.judge_task_id in excluded]
    tasks = [task for task in tasks if task.judge_task_id not in excluded]
    mappings = [mapping for mapping in mappings if mapping.judge_task_id not in excluded]
    # Do not freeze a plan against source metadata changed during preparation.
    for identity in sources.values():
        if _file_identity(Path(identity["path"])) != identity:
            raise ValueError("pilot source changed during preparation")
    for source in exclusion_sources:
        for identity in source.values():
            if _file_identity(Path(identity["path"])) != identity:
                raise ValueError("excluded judgment changed during preparation")
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
        "prompt_count": len(selected),
        "cells_per_prompt": 12,
        "selected_axis_bins": [axis_bins[p] for p in selected],
        "selection_method": (
            "all-complete-prompt-groups-v1" if all_available
            else "extreme-available-axis-bins-seeded-ties-v1"
        ),
        "source_generator_root": str(root),
        "source_git_commit": manifest["git_commit"],
        "source_completed_count": len(result_paths),
        "scope": (
            "allocation-filling-throughput-not-scientific-validation" if all_available
            else "pipeline-plumbing-only-not-judge-quality-validation"
        ),
        "execution_mode": "throughput" if all_available else "fixed",
    }
    plan_id = (
        "agentic-judge-pilot-"
        + _digest(
            {
                "format_version": FORMAT_VERSION,
                "master_seed": master_seed,
                "model": model,
                "task_ids": [task.judge_task_id for task in tasks],
                **({"excluded_task_ids": sorted(excluded)} if excluded else {}),
            }
        )[:24]
    )
    output.mkdir(parents=True)
    bulk_path = output / "bulk_tasks.jsonl"
    mapping_path = output / "private_mapping.jsonl"
    _atomic_jsonl(bulk_path, [task.to_dict() for task in tasks])
    _atomic_jsonl(mapping_path, [asdict(mapping) for mapping in mappings])
    exclusion_path = output / "excluded_coverage.json"
    _atomic_json(exclusion_path, {
        "excluded_task_ids": sorted(excluded),
        "excluded_mappings": [asdict(mapping) for mapping in excluded_mappings],
        "sources": exclusion_sources,
    })
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
            "summary": {
                "bulk_task_count": len(tasks),
                "validation_task_count": 0,
                "available_task_count": available_task_count,
                "excluded_task_count": len(excluded),
                "pending_task_count": len(tasks),
            },
            "artifacts": {
                "bulk_tasks": _file_identity(bulk_path),
                "private_mapping": _file_identity(mapping_path),
                "excluded_coverage": _file_identity(exclusion_path),
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20260915)
    selection.add_argument("--all-available", action="store_true",
                        help="Freeze all complete prompt groups as a throughput pilot")
    parser.add_argument("--exclude-outcomes", type=Path, action="append", default=[],
                        help="Verify and exclude a prior Nemotron outcomes journal; repeatable")
    args = parser.parse_args()
    try:
        manifest_path = prepare_pilot(**vars(args))
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"NEMOTRON_JUDGE_PILOT_TASKS={manifest['summary']['bulk_task_count']}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
