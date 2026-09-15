#!/usr/bin/env python3
"""Freeze blinded Nemotron bulk and GLM validation queues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_judging import (  # noqa: E402
    AgenticJudgeModel,
    build_agentic_judge_plan,
    write_agentic_judge_plan,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _generator_binding(value: str) -> tuple[str, Path]:
    model_id, separator, raw_path = value.partition("=")
    if not separator or not model_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("generator roots use MODEL_ID=PATH")
    return model_id, Path(raw_path).resolve()


def _completed_results(model_root: Path, *, expected_model_id: str) -> list[Path]:
    shard_roots = sorted(model_root.glob("shard-*"))
    if (model_root / "results").is_dir():
        shard_roots = [model_root]
    elif not shard_roots:
        shard_roots = sorted(
            path.parent for path in model_root.rglob("results") if path.is_dir()
        )
    if not shard_roots:
        raise ValueError(f"generator root has no result shards: {model_root}")
    results: list[Path] = []
    for shard_root in shard_roots:
        manifest_path = shard_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"generator shard lacks a manifest: {shard_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {
            "complete",
            "complete_with_failures",
            "checkpointed",
            "interrupted_or_error",
        }:
            raise ValueError(f"generator shard status is invalid: {shard_root}")
        if manifest.get("model_id") != expected_model_id:
            raise ValueError(f"generator shard model identity disagrees: {shard_root}")
        shard_results = sorted((shard_root / "results").glob("*.json"))
        completed = manifest.get("completed_count")
        if not isinstance(completed, int) or completed > len(shard_results):
            raise ValueError(f"generator shard artifact counts disagree: {shard_root}")
        results.extend(shard_results)
    return results


def _merged_prompts(prompt_path: Path, selection_path: Path) -> list[dict[str, Any]]:
    prompt_rows = _read_jsonl(prompt_path)
    selection_rows = _read_jsonl(selection_path)
    selection_by_id: dict[str, dict[str, Any]] = {}
    for row in selection_rows:
        prompt_id = str(row.get("candidate_id", ""))
        if not prompt_id or prompt_id in selection_by_id:
            raise ValueError(
                "selection records have missing or duplicate candidate IDs"
            )
        selection_by_id[prompt_id] = row
    merged: list[dict[str, Any]] = []
    for row in prompt_rows:
        prompt_id = str(row.get("candidate_id", ""))
        selection = selection_by_id.get(prompt_id)
        if selection is None:
            raise ValueError(f"prompt has no selection record: {prompt_id}")
        merged.append({**row, "axis_bin": selection.get("axis_bin")})
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-root",
        action="append",
        type=_generator_binding,
        required=True,
        help="Completed generator output as MODEL_ID=PATH. Repeat per model.",
    )
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--selection-records-jsonl", type=Path, required=True)
    parser.add_argument(
        "--generation-tasks",
        type=Path,
        required=True,
        help="Canonical generator queue used to prove complete model coverage.",
    )
    parser.add_argument("--bulk-model-id", required=True)
    parser.add_argument("--bulk-model-revision", required=True)
    parser.add_argument("--validation-model-id", required=True)
    parser.add_argument("--validation-model-revision", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--master-seed", type=int, default=20260915)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        results: list[Path] = []
        roots: dict[str, str] = {}
        generation_rows = _read_jsonl(arguments.generation_tasks.resolve())
        expected_cell_ids = {str(row["cell_id"]) for row in generation_rows}
        if len(expected_cell_ids) != len(generation_rows) or "" in expected_cell_ids:
            raise ValueError(
                "canonical generator queue has invalid or duplicate cell IDs"
            )
        for model_id, root in arguments.generator_root:
            if str(root) in roots:
                raise ValueError(f"duplicate generator root: {root}")
            roots[str(root)] = model_id
            model_results = _completed_results(root, expected_model_id=model_id)
            model_cell_ids = {
                str(json.loads(path.read_text(encoding="utf-8"))["cell_id"])
                for path in model_results
            }
            if model_cell_ids != expected_cell_ids or len(model_results) != len(
                expected_cell_ids
            ):
                raise ValueError(
                    f"generator model does not exactly cover the canonical queue: {model_id}"
                )
            results.extend(model_results)
        plan = build_agentic_judge_plan(
            results,
            prompt_rows=_merged_prompts(
                arguments.prompts_jsonl.resolve(),
                arguments.selection_records_jsonl.resolve(),
            ),
            generator_model_by_root=roots,
            bulk_model=AgenticJudgeModel(
                role="bulk",
                model_id=arguments.bulk_model_id,
                model_revision=arguments.bulk_model_revision,
            ),
            validation_model=AgenticJudgeModel(
                role="validation",
                model_id=arguments.validation_model_id,
                model_revision=arguments.validation_model_revision,
            ),
            validation_fraction=arguments.validation_fraction,
            master_seed=arguments.master_seed,
        )
        artifacts = write_agentic_judge_plan(
            arguments.output_dir,
            plan=plan,
            generation_tasks_path=arguments.generation_tasks.resolve(),
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(f"JUDGE_PLAN_ID={plan.judge_plan_id}")
    print(f"BULK_TASKS={plan.summary['bulk_task_count']}")
    print(f"VALIDATION_TASKS={plan.summary['validation_task_count']}")
    print(f"MANIFEST={artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
