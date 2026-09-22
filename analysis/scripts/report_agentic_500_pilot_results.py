#!/usr/bin/env python3
"""Audit the exact 500-prompt agentic pilot and Nemotron coverage.

This command is read-only. It validates the frozen 2 x 2 x 3 search factorial,
counts exact generator cell identities, and validates every durable Nemotron
claim against the frozen judge queue. Raw file counts are never treated as
scientific completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_generation_tasks import (
    CONDITIONS,
    ENGINES,
    METHODS,
)
from analysis.interpretability.pipeline.inference_claims import (
    InferenceClaimStore,
)
from analysis.scripts import manage_agentic_pilot_judging as judge_manager
from analysis.scripts.run_acl_arr_vllm import (
    _agentic_judge_context,
    _prepare_agentic_judge,
    _shared_claim_identity,
    _validate_shared_failure,
    _validate_shared_outcome,
)

GENERATOR_MODELS = dict(judge_manager.GENERATORS)
JUDGE_FORMAT = "agentic-original-pilot-judging-v1"
IDENTITY_FIELDS = (
    "cell_id",
    "prompt_id",
    "prompt_sha256",
    "method",
    "engine",
    "condition",
)
TERMINAL_STATES = {"completed", "failed", "busy", "missing"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_file(identity: object, *, label: str) -> Path:
    if not isinstance(identity, Mapping):
        raise TypeError(f"{label} does not have a frozen file identity")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if identity.get("sha256") != _sha256(path):
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _count_statuses(statuses: Sequence[str]) -> dict[str, int]:
    counts = Counter(statuses)
    expected = len(statuses)
    completed = counts["completed"]
    failed = counts["failed"]
    busy = counts["busy"]
    missing = counts["missing"]
    return {
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "busy": busy,
        "missing": missing,
        "unresolved": expected - completed,
        "percent_completed": round(100 * completed / expected, 4) if expected else 0.0,
    }


def _count_generation(statuses: Sequence[str]) -> dict[str, int | float]:
    counts = Counter(statuses)
    expected = len(statuses)
    completed = counts["completed"]
    return {
        "expected": expected,
        "completed": completed,
        "failed": counts["failed"],
        "missing": counts["missing"],
        "percent_completed": round(100 * completed / expected, 4) if expected else 0.0,
    }


def _prompt_coverage(
    rows: Sequence[Mapping[str, Any]], statuses: Sequence[str]
) -> dict[str, int]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row, status in zip(rows, statuses, strict=True):
        grouped[str(row["prompt_id"])].append(status)
    complete = sum(all(status == "completed" for status in values) for values in grouped.values())
    untouched = sum(not any(status == "completed" for status in values) for values in grouped.values())
    return {
        "complete": complete,
        "partial": len(grouped) - complete - untouched,
        "untouched": untouched,
    }


def _breakdown(
    rows: Sequence[Mapping[str, Any]], statuses: Sequence[str], field: str, *, judge: bool
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row, status in zip(rows, statuses, strict=True):
        grouped[str(row[field])].append(status)
    if judge:
        return {key: _count_statuses(values) for key, values in sorted(grouped.items())}
    return {key: _count_generation(values) for key, values in sorted(grouped.items())}


def _validate_factorial(
    tasks: Sequence[Mapping[str, Any]], expected_prompt_count: int
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    if expected_prompt_count <= 0:
        raise ValueError("expected prompt count must be positive")
    methods = {method.method_id for method in METHODS}
    engines = set(ENGINES)
    conditions = {condition.value for condition in CONDITIONS}
    expected_arms = {
        (method, engine, condition)
        for method in methods
        for engine in engines
        for condition in conditions
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    by_prompt: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for index, row in enumerate(tasks, 1):
        if any(not isinstance(row.get(field), str) or not row[field] for field in IDENTITY_FIELDS):
            raise ValueError(f"generation task {index} lacks its complete identity")
        cell_id = str(row["cell_id"])
        if cell_id in by_id:
            raise ValueError(f"duplicate generation cell: {cell_id}")
        by_id[cell_id] = row
        arm = (str(row["method"]), str(row["engine"]), str(row["condition"]))
        if arm not in expected_arms or arm in by_prompt[str(row["prompt_id"])]:
            raise ValueError(f"invalid or duplicate factorial arm for {row['prompt_id']}: {arm}")
        by_prompt[str(row["prompt_id"])].add(arm)
    if len(by_prompt) != expected_prompt_count:
        raise ValueError(
            f"expected {expected_prompt_count} prompts, found {len(by_prompt)}"
        )
    incomplete = [prompt_id for prompt_id, arms in by_prompt.items() if arms != expected_arms]
    if incomplete:
        raise ValueError(
            f"generation plan is not the complete 2 x 2 x 3 factorial; "
            f"incomplete prompts={len(incomplete)} first={incomplete[0]}"
        )
    return sorted(by_prompt), by_id


def build_report(
    *,
    generation_tasks: Sequence[Mapping[str, Any]],
    generator_completed: Mapping[str, set[str]],
    generator_failed: Mapping[str, set[str]],
    judge_mappings: Sequence[Mapping[str, Any]],
    judge_states: Mapping[str, str],
    generator_models: Mapping[str, str] = GENERATOR_MODELS,
    expected_prompt_count: int = 500,
) -> dict[str, Any]:
    """Build a strict task-identity report from already loaded artifacts."""

    prompts, task_by_id = _validate_factorial(generation_tasks, expected_prompt_count)
    task_ids = set(task_by_id)
    aliases = list(generator_models)
    if set(generator_completed) != set(aliases) or set(generator_failed) != set(aliases):
        raise ValueError("generator status maps must exactly match the configured models")

    generation_models: dict[str, Any] = {}
    combined_rows: list[Mapping[str, Any]] = []
    combined_statuses: list[str] = []
    completed_by_alias: dict[str, set[str]] = {}
    for alias, model_id in generator_models.items():
        completed = set(generator_completed[alias])
        failed = set(generator_failed[alias])
        unknown = (completed | failed) - task_ids
        if unknown:
            raise ValueError(f"{alias} references unknown cells: {sorted(unknown)[:3]}")
        overlap = completed & failed
        if overlap:
            raise ValueError(f"{alias} cells are both completed and failed: {sorted(overlap)[:3]}")
        completed_by_alias[alias] = completed
        rows = [task_by_id[cell_id] for cell_id in sorted(task_ids)]
        statuses = [
            "completed" if row["cell_id"] in completed
            else "failed" if row["cell_id"] in failed
            else "missing"
            for row in rows
        ]
        combined_rows.extend(rows)
        combined_statuses.extend(statuses)
        generation_models[alias] = {
            "model_id": model_id,
            "overall": _count_generation(statuses),
            "prompts": _prompt_coverage(rows, statuses),
            "by_method": _breakdown(rows, statuses, "method", judge=False),
            "by_engine": _breakdown(rows, statuses, "engine", judge=False),
            "by_condition": _breakdown(rows, statuses, "condition", judge=False),
        }

    model_by_id = {model_id: alias for alias, model_id in generator_models.items()}
    mapping_by_id: dict[str, Mapping[str, Any]] = {}
    source_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(judge_mappings, 1):
        judge_task_id = row.get("judge_task_id")
        model_id = row.get("generator_model_id")
        cell_id = row.get("source_cell_id")
        if not isinstance(judge_task_id, str) or not judge_task_id:
            raise ValueError(f"judge mapping {index} lacks judge_task_id")
        if judge_task_id in mapping_by_id:
            raise ValueError(f"duplicate judge task mapping: {judge_task_id}")
        if model_id not in model_by_id or cell_id not in task_by_id:
            raise ValueError(f"judge mapping has an unknown generator or cell: {judge_task_id}")
        source = task_by_id[str(cell_id)]
        if any(row.get(field) != source[field] for field in ("prompt_id", "method", "engine", "condition")):
            raise ValueError(f"judge mapping differs from frozen generation cell: {judge_task_id}")
        pair = (str(model_id), str(cell_id))
        if pair in source_pairs:
            raise ValueError(f"duplicate judge source mapping: {pair}")
        source_pairs.add(pair)
        mapping_by_id[judge_task_id] = row
    expected_pairs = {
        (model_id, cell_id) for model_id in generator_models.values() for cell_id in task_ids
    }
    if source_pairs != expected_pairs:
        raise ValueError(
            "judge mapping does not cover every generator-model/cell pair: "
            f"expected={len(expected_pairs)} observed={len(source_pairs)}"
        )
    unknown_judgments = set(judge_states) - set(mapping_by_id)
    if unknown_judgments:
        raise ValueError(f"judge states contain unknown tasks: {sorted(unknown_judgments)[:3]}")
    if any(state not in TERMINAL_STATES for state in judge_states.values()):
        raise ValueError("judge states contain an unsupported state")

    judge_rows = [mapping_by_id[task_id] for task_id in sorted(mapping_by_id)]
    judge_status_list = [judge_states.get(str(row["judge_task_id"]), "missing") for row in judge_rows]
    judge_by_model: dict[str, Any] = {}
    for model_id, alias in sorted(model_by_id.items(), key=lambda item: item[1]):
        selected = [
            (row, status)
            for row, status in zip(judge_rows, judge_status_list, strict=True)
            if row["generator_model_id"] == model_id
        ]
        rows = [row for row, _ in selected]
        statuses = [status for _, status in selected]
        judge_by_model[alias] = {
            "model_id": model_id,
            "overall": _count_statuses(statuses),
            "prompts": _prompt_coverage(rows, statuses),
        }

    paired_cells = set.intersection(*(completed_by_alias[alias] for alias in aliases))
    completed_per_prompt = Counter(task_by_id[cell_id]["prompt_id"] for cell_id in paired_cells)
    cells_per_prompt = len(task_ids) // len(prompts)
    fully_generated_prompts = sum(
        completed_per_prompt[prompt_id] == cells_per_prompt for prompt_id in prompts
    )
    generation_overall = _count_generation(combined_statuses)
    nemotron_overall = _count_statuses(judge_status_list)
    return {
        "format_version": "agentic-500-pilot-results-v1",
        "plan": {
            "prompts": len(prompts),
            "cells_per_model": len(task_ids),
            "generator_tasks": len(task_ids) * len(generator_models),
            "judge_tasks": len(mapping_by_id),
            "cells_per_prompt_per_model": cells_per_prompt,
        },
        "generation": {
            "overall": generation_overall,
            "models": generation_models,
            "paired": {
                "cells_completed_by_all_models": len(paired_cells),
                "expected_cells": len(task_ids),
                "prompts_complete_for_all_models": fully_generated_prompts,
                "expected_prompts": len(prompts),
            },
        },
        "nemotron": {
            "overall": nemotron_overall,
            "prompts": _prompt_coverage(judge_rows, judge_status_list),
            "by_generator": judge_by_model,
            "by_method": _breakdown(judge_rows, judge_status_list, "method", judge=True),
            "by_engine": _breakdown(judge_rows, judge_status_list, "engine", judge=True),
            "by_condition": _breakdown(judge_rows, judge_status_list, "condition", judge=True),
        },
        "experiment_complete": (
            generation_overall["completed"] == generation_overall["expected"]
            and generation_overall["failed"] == 0
            and nemotron_overall["completed"] == nemotron_overall["expected"]
            and nemotron_overall["failed"] == 0
        ),
    }


def _generator_artifacts(
    study_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    generator_models: Mapping[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    expected = {str(row["cell_id"]): row for row in tasks}
    completed_by_model: dict[str, set[str]] = {}
    failed_by_model: dict[str, set[str]] = {}
    for alias, model_id in generator_models.items():
        model_root = study_root / "models" / alias
        shard_roots = sorted(model_root.glob("shard-*"))
        if (model_root / "results").is_dir():
            shard_roots = [model_root]
        if not shard_roots:
            raise ValueError(f"generator root has no result shards: {model_root}")
        completed: set[str] = set()
        failed: set[str] = set()
        for shard_root in shard_roots:
            manifest_path = shard_root / "run_manifest.json"
            manifest = _read_json(manifest_path)
            if manifest.get("model_id") != model_id:
                raise ValueError(f"generator model identity differs: {manifest_path}")
            failed_ids = manifest.get("failed_cell_ids", [])
            if not isinstance(failed_ids, list) or any(not isinstance(item, str) for item in failed_ids):
                raise ValueError(f"generator failures are invalid: {manifest_path}")
            failed.update(failed_ids)
            for result_path in sorted((shard_root / "results").glob("*.json")):
                result = _read_json(result_path)
                cell_id = result.get("cell_id")
                if not isinstance(cell_id, str) or cell_id not in expected or result_path.stem != cell_id:
                    raise ValueError(f"out-of-plan generator result: {result_path}")
                if cell_id in completed:
                    raise ValueError(f"duplicate generator result for {alias}: {cell_id}")
                if any(result.get(field) != expected[cell_id].get(field) for field in IDENTITY_FIELDS):
                    raise ValueError(f"generator result identity differs: {result_path}")
                completed.add(cell_id)
        unknown_failed = failed - set(expected)
        if unknown_failed:
            raise ValueError(f"{alias} failure list contains unknown cells")
        completed_by_model[alias] = completed
        failed_by_model[alias] = failed - completed
    return completed_by_model, failed_by_model


def _judge_claim_arguments():
    return judge_manager.judge_claim_arguments()


def _judge_artifacts(judge_run: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    receipt = judge_manager.verify(judge_run)
    tasks, _, model_id, revision = _agentic_judge_context(
        judge_run / "plan/run_manifest.json",
        judge_run / "plan/bulk_tasks.jsonl",
        judge_role="bulk",
    )
    mappings = _read_jsonl(judge_run / "plan/private_mapping.jsonl")
    store = InferenceClaimStore(receipt["claim_root"])
    args = _judge_claim_arguments()
    states: dict[str, str] = {}
    for number, task in enumerate(tasks, 1):
        item = _prepare_agentic_judge(task, max_tokens=2048)
        identity = _shared_claim_identity(
            item, args=args, model_id=model_id, model_revision=revision
        )
        state, _ = store.inspect(
            identity,
            validate=lambda outcome, item=item: _validate_shared_outcome(
                outcome, item=item, fake=False, pilot_only=False
            ),
            validate_failure=lambda outcome, item=item: _validate_shared_failure(
                outcome, item=item, fake=False, pilot_only=False
            ),
        )
        states[task.judge_task_id] = state
        if number % 500 == 0:
            print(f"Validated Nemotron claims: {number}/{len(tasks)}", file=sys.stderr)
    return mappings, states, receipt


def _latest_judge_run(pilot_root: Path) -> Path:
    candidates = []
    for path in pilot_root.rglob("launch.json"):
        try:
            value = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if value.get("format_version") == JUDGE_FORMAT:
            candidates.append(path)
    if not candidates:
        raise ValueError(f"no original-500 Nemotron judge run found under {pilot_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).parent.resolve()


def collect_report(
    *,
    adaptive_plan: Path,
    judge_run: Path,
    expected_prompt_count: int = 500,
) -> dict[str, Any]:
    adaptive_path = adaptive_plan.resolve()
    judge_root = judge_run.resolve()
    launch = _read_json(judge_root / "launch.json")
    frozen_adaptive = _checked_file(launch.get("adaptive_source"), label="adaptive source")
    if adaptive_path != frozen_adaptive:
        raise ValueError("explicit adaptive source differs from the frozen judge launch")
    adaptive = _read_json(adaptive_path)
    study = adaptive.get("original_study")
    if not isinstance(study, Mapping):
        raise TypeError("adaptive plan lacks original_study")
    generation_path = _checked_file(study.get("generation_tasks"), label="generation task plan")
    generation_tasks = _read_jsonl(generation_path)
    study_root = Path(str(study.get("root", ""))).resolve()
    if not study_root.is_dir():
        raise ValueError(f"original study root is not a directory: {study_root}")
    completed, failed = _generator_artifacts(study_root, generation_tasks, GENERATOR_MODELS)
    mappings, states, receipt = _judge_artifacts(judge_root)
    report = build_report(
        generation_tasks=generation_tasks,
        generator_completed=completed,
        generator_failed=failed,
        judge_mappings=mappings,
        judge_states=states,
        expected_prompt_count=expected_prompt_count,
    )
    report["sources"] = {
        "adaptive_plan": str(adaptive_path),
        "study_root": str(study_root),
        "generation_tasks": str(generation_path),
        "judge_run": str(judge_root),
        "claim_root": str(receipt["claim_root"]),
        "judge_status": receipt.get("status"),
        "max_model_len": receipt.get("max_model_len"),
        "execution_commit": receipt.get("execution_commit"),
    }
    return report


def _percent(value: Mapping[str, Any]) -> str:
    return f"{float(value['percent_completed']):.2f}%"


def _compact(value: Mapping[str, Any], *, judge: bool) -> str:
    text = (
        f"{value['completed']}/{value['expected']} "
        f"({_percent(value)}) failed={value['failed']} missing={value['missing']}"
    )
    if judge:
        text += f" busy={value['busy']}"
    return text


def render_text(report: Mapping[str, Any]) -> str:
    plan = report["plan"]
    generation = report["generation"]
    nemotron = report["nemotron"]
    lines = [
        "AGENTIC_500_PILOT_RESULTS_V1",
        (
            f"PLAN prompts={plan['prompts']} cells_per_prompt_per_model="
            f"{plan['cells_per_prompt_per_model']} cells_per_model={plan['cells_per_model']} "
            f"generator_tasks={plan['generator_tasks']} judge_tasks={plan['judge_tasks']}"
        ),
        (
            f"GENERATION {generation['overall']['completed']}/"
            f"{generation['overall']['expected']} ({_percent(generation['overall'])}) "
            f"failed={generation['overall']['failed']} missing={generation['overall']['missing']}"
        ),
    ]
    for alias, model in generation["models"].items():
        prompts = model["prompts"]
        lines.append(
            f"  MODEL {alias} {_compact(model['overall'], judge=False)} "
            f"prompts_complete={prompts['complete']}/{plan['prompts']} "
            f"partial={prompts['partial']} untouched={prompts['untouched']}"
        )
        for label, field in (
            ("METHOD", "by_method"),
            ("ENGINE", "by_engine"),
            ("CONDITION", "by_condition"),
        ):
            for name, counts in model[field].items():
                lines.append(f"    {label} {name} {_compact(counts, judge=False)}")
    paired = generation["paired"]
    lines.extend(
        (
            (
                f"PAIRED_GENERATION cells={paired['cells_completed_by_all_models']}/"
                f"{paired['expected_cells']} prompts={paired['prompts_complete_for_all_models']}/"
                f"{paired['expected_prompts']}"
            ),
            (
                f"NEMOTRON {nemotron['overall']['completed']}/"
                f"{nemotron['overall']['expected']} ({_percent(nemotron['overall'])}) "
                f"failed={nemotron['overall']['failed']} busy={nemotron['overall']['busy']} "
                f"missing={nemotron['overall']['missing']}"
            ),
            (
                f"  PROMPTS fully_judged={nemotron['prompts']['complete']}/"
                f"{plan['prompts']} partial={nemotron['prompts']['partial']} "
                f"untouched={nemotron['prompts']['untouched']}"
            ),
        )
    )
    for alias, model in nemotron["by_generator"].items():
        prompts = model["prompts"]
        lines.append(
            f"  GENERATOR {alias} {_compact(model['overall'], judge=True)} "
            f"prompts_complete={prompts['complete']}/{plan['prompts']}"
        )
    for label, field in (
        ("METHOD", "by_method"),
        ("ENGINE", "by_engine"),
        ("CONDITION", "by_condition"),
    ):
        for name, counts in nemotron[field].items():
            lines.append(f"  {label} {name} {_compact(counts, judge=True)}")
    lines.append(f"EXPERIMENT_COMPLETE={'YES' if report['experiment_complete'] else 'NO'}")
    sources = report.get("sources")
    if isinstance(sources, Mapping):
        lines.extend(
            (
                f"ADAPTIVE_PLAN={sources['adaptive_plan']}",
                f"STUDY_ROOT={sources['study_root']}",
                f"JUDGE_RUN={sources['judge_run']}",
                f"CLAIM_ROOT={sources['claim_root']}",
                f"JUDGE_STATUS={sources.get('judge_status') or '-'}",
                f"MAX_MODEL_LEN={sources.get('max_model_len') or '-'}",
                f"EXECUTION_COMMIT={sources.get('execution_commit') or '-'}",
            )
        )
    lines.append(
        "COMPLETION_RULE=exact validated task identities; retries and raw file counts are not completions"
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        help="Pilot directory; the newest exact original-500 judge launch is selected.",
    )
    parser.add_argument("--judge-run", type=Path, help="Exact original-500 Nemotron run root.")
    parser.add_argument(
        "--adaptive-plan",
        type=Path,
        help="Exact adaptive plan run_manifest.json; defaults to the judge launch source.",
    )
    parser.add_argument("--expected-prompts", type=int, default=500)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.judge_run is None:
            if args.pilot_root is None:
                raise ValueError("provide --pilot-root or --judge-run")
            judge_run = _latest_judge_run(args.pilot_root.resolve())
        else:
            judge_run = args.judge_run.resolve()
        launch = _read_json(judge_run / "launch.json")
        if launch.get("format_version") != JUDGE_FORMAT:
            raise ValueError(f"not an original-500 Nemotron run: {judge_run}")
        if args.adaptive_plan is None:
            adaptive = launch.get("adaptive_source")
            if not isinstance(adaptive, Mapping):
                raise ValueError("judge launch does not identify its adaptive source")
            adaptive_plan = _checked_file(adaptive, label="adaptive source")
        else:
            adaptive_plan = args.adaptive_plan.resolve()
        report = collect_report(
            adaptive_plan=adaptive_plan,
            judge_run=judge_run,
            expected_prompt_count=args.expected_prompts,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_text(report), end="")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
