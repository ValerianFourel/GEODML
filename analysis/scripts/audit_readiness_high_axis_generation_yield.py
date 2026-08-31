#!/usr/bin/env python3
"""Audit per-round recovery from targeted high-readiness generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


FORMAT_VERSION = "readiness-high-axis-generation-yield-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-selected", required=True)
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-target-axis-1", type=float, default=0.70)
    return parser


def _rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_value(row: Mapping[str, object]) -> float:
    target = row.get("target")
    if isinstance(target, Mapping):
        return float(target["normalized_axis_1"])
    return float(row["target_normalized_axis_1"])


def _cell(row: Mapping[str, object]) -> tuple[str, str]:
    target = row.get("target")
    target_id = (
        str(target["target_id"])
        if isinstance(target, Mapping)
        else str(row["target_id"])
    )
    return str(row["keyword_id"]), target_id


def _band(value: float) -> str:
    if value < 0.80:
        return "0.70-0.80"
    if value < 0.90:
        return "0.80-0.90"
    return "0.90-1.00"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _summarize(
    tasks: Iterable[Mapping[str, object]],
    generated: Iterable[Mapping[str, object]],
    accepted_ids: set[str],
    current_by_cell: Mapping[tuple[str, str], Mapping[str, object]],
    baseline_cells: set[tuple[str, str]],
) -> dict[str, object]:
    task_rows = list(tasks)
    generated_rows = list(generated)
    task_cells = {_cell(row) for row in task_rows}
    generated_ids = {str(row["candidate_id"]) for row in generated_rows}
    recovered_cells = (set(current_by_cell) - baseline_cells) & task_cells
    generated_selected_cells = {
        cell
        for cell, row in current_by_cell.items()
        if str(row["candidate_id"]) in generated_ids and cell in task_cells
    }
    accepted_generated = generated_ids & accepted_ids
    return {
        "task_count": len(task_rows),
        "generated_candidate_count": len(generated_rows),
        "accepted_generated_candidate_count": len(accepted_generated),
        "accepted_generated_fraction": _ratio(
            len(accepted_generated), len(generated_rows)
        ),
        "recovered_target_cell_count": len(recovered_cells),
        "task_recovery_fraction": _ratio(len(recovered_cells), len(task_cells)),
        "generated_candidate_selected_cell_count": len(generated_selected_cells),
    }


def audit(
    baseline_selected: Path,
    round_root: Path,
    output: Path,
    *,
    minimum_target_axis_1: float,
) -> dict[str, object]:
    if not 0.70 <= minimum_target_axis_1 <= 1.0:
        raise ValueError("minimum target axis 1 must lie in [0.70, 1]")
    task_paths = [
        path
        for path in (
            round_root / "refinement-task-batch.jsonl",
            round_root / "generation-task-batch.jsonl",
        )
        if path.is_file()
    ]
    if len(task_paths) != 1:
        raise ValueError("round must contain exactly one targeted generation task batch")
    task_path = task_paths[0]
    candidate_paths = sorted(
        path
        for path in (round_root / "generation" / "candidates").glob("*.jsonl")
        if not path.name.endswith(".failures.jsonl")
    )
    validation_path = round_root / "validation.jsonl"
    selected_path = (
        round_root / "strict-selection" / "spatially_selected_questions.jsonl"
    )
    required = (baseline_selected, task_path, validation_path, selected_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not candidate_paths:
        raise ValueError(f"high-axis yield inputs are missing: {missing}")

    tasks = _rows(task_path)
    if any(_target_value(row) < minimum_target_axis_1 for row in tasks):
        raise ValueError("task batch contains a target below the high-axis threshold")
    generated = [row for path in candidate_paths for row in _rows(path)]
    if len({str(row["candidate_id"]) for row in generated}) != len(generated):
        raise ValueError("generated candidate ids are not unique")
    validation = _rows(validation_path)
    accepted_ids = {
        str(row["candidate_id"]) for row in validation if bool(row["accepted"])
    }
    baseline = _rows(baseline_selected)
    current = _rows(selected_path)
    baseline_cells = {_cell(row) for row in baseline}
    current_by_cell = {_cell(row): row for row in current}

    overall = _summarize(
        tasks, generated, accepted_ids, current_by_cell, baseline_cells
    )
    bands = {}
    for name in ("0.70-0.80", "0.80-0.90", "0.90-1.00"):
        band_tasks = [row for row in tasks if _band(_target_value(row)) == name]
        band_cells = {_cell(row) for row in band_tasks}
        band_generated = [row for row in generated if _cell(row) in band_cells]
        bands[name] = _summarize(
            band_tasks,
            band_generated,
            accepted_ids,
            current_by_cell,
            baseline_cells,
        )

    selected_generated = [
        row
        for row in current
        if str(row["candidate_id"])
        in {str(candidate["candidate_id"]) for candidate in generated}
    ]
    view_errors = []
    for row in selected_generated:
        target = float(row["target_normalized_axis_1"])
        view_errors.append(
            max(
                abs(target - float(row["reference_normalized_axis_1"])),
                abs(
                    target
                    - float(row["candidate_aligned_normalized_axis_1"])
                ),
            )
        )
    result = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "minimum_target_axis_1": minimum_target_axis_1,
        "inputs": {
            "baseline_selected": _identity(baseline_selected),
            "tasks": _identity(task_path),
            "generated_candidates": [_identity(path) for path in candidate_paths],
            "validation": _identity(validation_path),
            "current_selected": _identity(selected_path),
        },
        "baseline_selected_count": len(baseline),
        "current_selected_count": len(current),
        "net_selected_change": len(current) - len(baseline),
        "overall": overall,
        "target_bands": bands,
        "selected_generated_dual_view_error": {
            "count": len(view_errors),
            "maximum": max(view_errors) if view_errors else None,
            "mean": sum(view_errors) / len(view_errors) if view_errors else None,
        },
        "scientific_guard": (
            "This is generation-yield and prompt-space coverage diagnostics. "
            "Prompt embeddings do not define randomized policy B."
        ),
    }
    if output.exists():
        manifest_path = output / "high_axis_yield.json"
        if not manifest_path.is_file() or json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["inputs"] != result["inputs"]:
            raise ValueError(f"refusing to overwrite a different yield audit: {output}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    (output / "high_axis_yield.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# High-axis generation yield",
        "",
        f"- Baseline selected: {len(baseline)}",
        f"- Current selected: {len(current)}",
        f"- Net selected change: {result['net_selected_change']:+d}",
        f"- Targeted tasks: {overall['task_count']}",
        f"- Generated candidates: {overall['generated_candidate_count']}",
        f"- Recovered targeted cells: {overall['recovered_target_cell_count']}",
        f"- Targeted task recovery: {overall['task_recovery_fraction'] or 0:.2%}",
        "",
        "| Target band | Tasks | Candidates | Accepted | Recovered | Recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in bands.items():
        report.append(
            f"| {name} | {row['task_count']} | {row['generated_candidate_count']} | "
            f"{row['accepted_generated_candidate_count']} | "
            f"{row['recovered_target_cell_count']} | "
            f"{row['task_recovery_fraction'] or 0:.2%} |"
        )
    report.extend(("", result["scientific_guard"]))
    (output / "high_axis_yield.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    result = audit(
        Path(args.baseline_selected).resolve(),
        Path(args.round_root).resolve(),
        Path(args.output_dir).resolve(),
        minimum_target_axis_1=args.minimum_target_axis_1,
    )
    print(f"targeted_tasks={result['overall']['task_count']}")
    print(f"generated_candidates={result['overall']['generated_candidate_count']}")
    print(f"recovered_cells={result['overall']['recovered_target_cell_count']}")
    print(f"output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
