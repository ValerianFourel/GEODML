#!/usr/bin/env python3
"""Verify, summarize, and prioritize exact readiness refinement gaps."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Mapping


FORMAT_VERSION = "readiness-gap-inventory-v1"
TASK_FILE = re.compile(r"generation_tasks_round_(\d+)\.jsonl$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _task_file(selection: Path) -> tuple[Path, int]:
    matches = sorted(selection.glob("generation_tasks_round_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            "strict selection must contain exactly one refinement task file"
        )
    match = TASK_FILE.fullmatch(matches[0].name)
    if match is None:
        raise ValueError(f"invalid refinement task filename: {matches[0]}")
    return matches[0], int(match.group(1))


def _target_total(diagnostics: Mapping[str, object]) -> int:
    keyword_count = int(diagnostics["keyword_count"])
    target_counts = diagnostics["target_count_per_keyword"]
    if isinstance(target_counts, int):
        return keyword_count * target_counts
    if isinstance(target_counts, Mapping):
        if len(target_counts) != keyword_count:
            raise ValueError("keyword-specific target counts are incomplete")
        return sum(int(value) for value in target_counts.values())
    raise ValueError("invalid target_count_per_keyword diagnostics")


def _selected_cell(row: Mapping[str, object]) -> tuple[str, str]:
    keyword_id = str(row.get("keyword_id", ""))
    target_id = str(row.get("target_id", ""))
    if not keyword_id or not target_id:
        raise ValueError("selected row lacks keyword_id or target_id")
    return keyword_id, target_id


def _task_fields(
    row: Mapping[str, object],
    *,
    expected_round: int,
) -> dict[str, object]:
    task_id = str(row.get("task_id", ""))
    keyword_id = str(row.get("keyword_id", ""))
    keyword = str(row.get("keyword", ""))
    generator_id = str(row.get("generator_id", ""))
    feedback = str(row.get("feedback", ""))
    target = row.get("target")
    if not all((task_id, keyword_id, keyword, generator_id, feedback)):
        raise ValueError(f"refinement task has incomplete identity: {task_id!r}")
    if not isinstance(target, Mapping):
        raise ValueError(f"refinement task lacks target object: {task_id}")
    target_id = str(target.get("target_id", ""))
    if not target_id:
        raise ValueError(f"refinement task lacks target_id: {task_id}")
    round_index = int(row.get("round_index", -1))
    if round_index != expected_round:
        raise ValueError(
            f"task {task_id} has round {round_index}, expected {expected_round}"
        )
    return {
        "task_id": task_id,
        "keyword_id": keyword_id,
        "keyword": keyword,
        "target_id": target_id,
        "target_index": int(target["target_index"]),
        "axis_1_index": int(target["axis_1_index"]),
        "target_normalized_axis_1": float(target["normalized_axis_1"]),
        "generator_id": generator_id,
        "round_index": round_index,
        "requested_candidate_count": int(row["requested_candidate_count"]),
        "feedback": feedback,
    }


def _reason(feedback: str) -> str:
    lowered = feedback.lower()
    if "no independently validated candidate" in lowered:
        return "no_independently_validated_candidate"
    if "no valid candidate" in lowered:
        return "no_valid_candidate"
    if "closest" in lowered:
        return "closest_candidate_outside_contract"
    return "refinement_required"


def _report(manifest: Mapping[str, object], top: int) -> str:
    lines = [
        "# Readiness refinement-gap inventory",
        "",
        f"- Selected cells: {manifest['selected_count']}",
        f"- Missing cells: {manifest['gap_count']}",
        f"- Frozen target cells: {manifest['target_cell_count']}",
        f"- Completion: {manifest['completion_fraction']:.4%}",
        f"- Next generation round: {manifest['next_round_index']}",
        "",
        "## Axis-1 levels with the most missing keyword cells",
        "",
        "| Axis index | Normalized axis 1 | Missing cells |",
        "|---:|---:|---:|",
    ]
    for row in manifest["axis_1_gaps"][:top]:
        lines.append(
            f"| {row['axis_1_index']} | "
            f"{row['target_normalized_axis_1']:.6f} | "
            f"{row['gap_count']} |"
        )
    lines.extend(
        [
            "",
            "## Keywords with the most missing axis cells",
            "",
            "| Keyword ID | Keyword | Missing cells |",
            "|---|---|---:|",
        ]
    )
    for row in manifest["keyword_gaps"][:top]:
        lines.append(
            f"| {row['keyword_id']} | {row['keyword']} | "
            f"{row['gap_count']} |"
        )
    lines.extend(
        [
            "",
            "`exact_gap_tasks.jsonl` is an unchanged copy of the pipeline's "
            "frozen refinement tasks. `prioritized_gap_tasks.jsonl` contains "
            "the same task objects reordered by shared axis deficit, then "
            "keyword deficit, with stable identity tie-breaking.",
            "",
            "This inventory describes readiness-question coverage. Prompt "
            "embeddings do not define randomized policy B.",
            "",
        ]
    )
    return "\n".join(lines)


def build_gap_inventory(
    checkpoint_root: str | Path,
    output_dir: str | Path,
    *,
    report_top: int = 30,
) -> dict[str, object]:
    if report_top <= 0:
        raise ValueError("report_top must be positive")
    checkpoint = Path(checkpoint_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")

    summary_path = checkpoint / "verified_round_summary.json"
    selection = checkpoint / "strict-selection"
    selected_path = selection / "spatially_selected_questions.jsonl"
    diagnostics_path = selection / "spatial_coverage_diagnostics.json"
    selection_manifest_path = selection / "run_manifest.json"
    for path in (
        summary_path,
        selected_path,
        diagnostics_path,
        selection_manifest_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required checkpoint artifact is missing: {path}")
    task_path, next_round = _task_file(selection)

    summary = _read_json(summary_path)
    diagnostics = _read_json(diagnostics_path)
    selection_manifest = _read_json(selection_manifest_path)
    selected = _read_jsonl(selected_path)
    tasks = _read_jsonl(task_path)
    selected_cells = [_selected_cell(row) for row in selected]
    if len(set(selected_cells)) != len(selected_cells):
        raise ValueError("selected population contains duplicate target cells")
    if any(not bool(row.get("both_views_within_tolerance")) for row in selected):
        raise ValueError("selected population contains a noncompliant dual-view row")

    parsed = [
        _task_fields(row, expected_round=next_round)
        for row in tasks
    ]
    task_ids = [str(row["task_id"]) for row in parsed]
    task_cells = [
        (str(row["keyword_id"]), str(row["target_id"])) for row in parsed
    ]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("refinement tasks have missing or duplicate task IDs")
    if len(set(task_cells)) != len(task_cells):
        raise ValueError("refinement tasks contain duplicate target cells")
    if set(selected_cells) & set(task_cells):
        raise ValueError("selected and refinement cells overlap")

    selected_count = len(selected)
    gap_count = len(tasks)
    target_count = _target_total(diagnostics)
    expected_counts = {
        "summary selected_count": int(summary["selected_count"]),
        "selection selected_count": int(selection_manifest["selected_count"]),
    }
    if any(value != selected_count for value in expected_counts.values()):
        raise ValueError(f"selected-count mismatch: {expected_counts}")
    expected_gaps = {
        "summary refinement_task_count": int(summary["refinement_task_count"]),
        "selection next_round_task_count": int(
            selection_manifest["next_round_task_count"]
        ),
    }
    if any(value != gap_count for value in expected_gaps.values()):
        raise ValueError(f"gap-count mismatch: {expected_gaps}")
    if selected_count + gap_count != target_count:
        raise ValueError(
            "selected and missing cells do not exhaust the frozen target lattice"
        )

    axis_counts = Counter(
        (
            int(row["axis_1_index"]),
            float(row["target_normalized_axis_1"]),
        )
        for row in parsed
    )
    keyword_counts = Counter(str(row["keyword_id"]) for row in parsed)
    keyword_text = {
        str(row["keyword_id"]): str(row["keyword"]) for row in parsed
    }
    generator_counts = Counter(str(row["generator_id"]) for row in parsed)
    reason_counts = Counter(_reason(str(row["feedback"])) for row in parsed)

    def priority_key(index: int) -> tuple[object, ...]:
        row = parsed[index]
        axis_key = (
            int(row["axis_1_index"]),
            float(row["target_normalized_axis_1"]),
        )
        return (
            -axis_counts[axis_key],
            -keyword_counts[str(row["keyword_id"])],
            int(row["axis_1_index"]),
            str(row["keyword_id"]),
            str(row["target_id"]),
            str(row["task_id"]),
        )

    priority_indices = sorted(range(gap_count), key=priority_key)
    ranked_rows = []
    for rank, index in enumerate(priority_indices, start=1):
        row = parsed[index]
        axis_key = (
            int(row["axis_1_index"]),
            float(row["target_normalized_axis_1"]),
        )
        ranked_rows.append(
            {
                "priority_rank": rank,
                **row,
                "gap_reason": _reason(str(row["feedback"])),
                "axis_level_gap_count": axis_counts[axis_key],
                "keyword_gap_count": keyword_counts[str(row["keyword_id"])],
            }
        )

    axis_gaps = [
        {
            "axis_1_index": key[0],
            "target_normalized_axis_1": key[1],
            "gap_count": count,
        }
        for key, count in sorted(
            axis_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]
    keyword_gaps = [
        {
            "keyword_id": keyword_id,
            "keyword": keyword_text[keyword_id],
            "gap_count": count,
        }
        for keyword_id, count in sorted(
            keyword_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": _now(),
        "checkpoint_root": str(checkpoint),
        "checkpoint_summary_sha256": _sha256(summary_path),
        "selected_path": str(selected_path),
        "selected_sha256": _sha256(selected_path),
        "source_task_path": str(task_path),
        "source_task_sha256": _sha256(task_path),
        "next_round_index": next_round,
        "target_cell_count": target_count,
        "selected_count": selected_count,
        "gap_count": gap_count,
        "completion_fraction": selected_count / target_count,
        "exhaustive_target_partition": True,
        "disjoint_selected_and_gap_cells": True,
        "priority_method": (
            "descending shared axis-level gap count, descending keyword gap "
            "count, then stable axis and identity ordering"
        ),
        "axis_1_gaps": axis_gaps,
        "keyword_gaps": keyword_gaps,
        "generator_gap_counts": dict(sorted(generator_counts.items())),
        "gap_reason_counts": dict(sorted(reason_counts.items())),
        "scientific_guard": (
            "This inventory describes readiness-question coverage; prompt "
            "embeddings do not define randomized policy B."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.attempt-", dir=output.parent)
    )
    try:
        shutil.copyfile(task_path, attempt / "exact_gap_tasks.jsonl")
        _write_jsonl(
            attempt / "prioritized_gap_tasks.jsonl",
            (tasks[index] for index in priority_indices),
        )
        _write_jsonl(attempt / "prioritized_gap_inventory.jsonl", ranked_rows)
        _write_json(attempt / "gap_manifest.json", manifest)
        (attempt / "gap_report.md").write_text(
            _report(manifest, report_top), encoding="utf-8"
        )
        attempt.replace(output)
    except BaseException:
        shutil.rmtree(attempt, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-top", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_gap_inventory(
        args.checkpoint_root,
        args.output_dir,
        report_top=args.report_top,
    )
    print(f"selected={manifest['selected_count']}")
    print(f"gaps={manifest['gap_count']}")
    print(f"target={manifest['target_cell_count']}")
    print(f"completion={manifest['completion_fraction']:.6f}")
    print(f"output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
