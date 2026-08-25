#!/usr/bin/env python3
"""Freeze one verified checkpoint into deterministic keyword-owned sections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.partition_readiness_refinement_tasks import target_partition


FORMAT_VERSION = "readiness-keyword-section-plan-v1"
TASK_NAME = re.compile(r"generation_tasks_round_(\d+)\.jsonl$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _required_checkpoint_files(checkpoint: Path) -> tuple[Path, ...]:
    return (
        checkpoint / "verified_round_summary.json",
        checkpoint / "candidate-files.txt",
        checkpoint / "validation.jsonl",
        checkpoint / "validation.jsonl.manifest.json",
        checkpoint / "projections/qwen/projection_manifest.json",
        checkpoint / "projections/mistral/projection_manifest.json",
        checkpoint / "strict-selection/run_manifest.json",
    )


def _next_task_file(checkpoint: Path) -> tuple[Path, int]:
    matches = sorted(
        (checkpoint / "strict-selection").glob("generation_tasks_round_*.jsonl")
    )
    if len(matches) != 1:
        raise ValueError(
            "verified checkpoint must contain exactly one next-round task file"
        )
    match = TASK_NAME.fullmatch(matches[0].name)
    if match is None:
        raise ValueError(f"invalid next-round task filename: {matches[0]}")
    next_round_index = int(match.group(1))
    if next_round_index <= 0:
        raise ValueError("next-round task index must be positive")
    return matches[0], next_round_index


def build_keyword_section_plan(
    checkpoint_root: str | Path,
    *,
    section_count: int,
    partition_salt: str,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_root).resolve()
    if section_count <= 1:
        raise ValueError("section_count must be greater than one")
    if not partition_salt.strip():
        raise ValueError("partition_salt must be nonempty")
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint root is missing: {checkpoint}")
    for required in _required_checkpoint_files(checkpoint):
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"checkpoint artifact is missing: {required}")

    task_file, next_round_index = _next_task_file(checkpoint)
    tasks = _read_jsonl(task_file)
    if not tasks:
        raise ValueError("checkpoint has no remaining refinement tasks")
    task_ids = [str(row.get("task_id", "")) for row in tasks]
    if not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("remaining refinement tasks have missing or duplicate ids")

    section_keywords: list[set[str]] = [set() for _ in range(section_count)]
    section_task_counts = [0 for _ in range(section_count)]
    keyword_owners: dict[str, int] = {}
    for row in tasks:
        keyword_id = str(row.get("keyword_id", ""))
        if not keyword_id:
            raise ValueError("remaining refinement task lacks keyword_id")
        owner = target_partition(
            row,
            partition_count=section_count,
            partition_salt=partition_salt,
        )
        previous = keyword_owners.setdefault(keyword_id, owner)
        if previous != owner:
            raise AssertionError("one keyword was assigned to multiple sections")
        section_keywords[owner].add(keyword_id)
        section_task_counts[owner] += 1

    owned_keywords = set().union(*section_keywords)
    if owned_keywords != set(keyword_owners):
        raise AssertionError("section plan does not exhaust the keyword set")
    if sum(section_task_counts) != len(tasks):
        raise AssertionError("section plan does not exhaust the task set")
    if any(
        section_keywords[left] & section_keywords[right]
        for left in range(section_count)
        for right in range(left + 1, section_count)
    ):
        raise AssertionError("section keyword ownership overlaps")

    summary = json.loads(
        (checkpoint / "verified_round_summary.json").read_text(encoding="utf-8")
    )
    sections = [
        {
            "section_index": index,
            "keyword_count": len(section_keywords[index]),
            "remaining_task_count": section_task_counts[index],
            "keyword_ids": sorted(section_keywords[index]),
        }
        for index in range(section_count)
    ]
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint_root": str(checkpoint),
        "checkpoint_summary_sha256": _sha256(
            checkpoint / "verified_round_summary.json"
        ),
        "candidate_file_list_sha256": _sha256(checkpoint / "candidate-files.txt"),
        "source_task_file": str(task_file.resolve()),
        "source_task_sha256": _sha256(task_file),
        "next_round_index": next_round_index,
        "initial_logical_round_index": next_round_index - 1,
        "section_count": section_count,
        "partition_salt": partition_salt,
        "selection_method": "stable-keyword-sha256-modulo-v1",
        "checkpoint_selected_count": int(summary["selected_count"]),
        "remaining_task_count": len(tasks),
        "remaining_keyword_count": len(keyword_owners),
        "sections": sections,
        "disjoint_keywords": True,
        "exhaustive_tasks": True,
    }


def write_keyword_section_plan(
    checkpoint_root: str | Path,
    output: str | Path,
    *,
    section_count: int,
    partition_salt: str,
) -> dict[str, object]:
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(f"refusing to overwrite section plan: {output_path}")
    plan = build_keyword_section_plan(
        checkpoint_root,
        section_count=section_count,
        partition_salt=partition_salt,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return plan


def verify_keyword_section_plan(path: str | Path) -> dict[str, object]:
    plan_path = Path(path).resolve()
    stored = json.loads(plan_path.read_text(encoding="utf-8"))
    if stored.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported keyword section plan format")
    rebuilt = build_keyword_section_plan(
        str(stored["checkpoint_root"]),
        section_count=int(stored["section_count"]),
        partition_salt=str(stored["partition_salt"]),
    )
    stored_stable = {key: value for key, value in stored.items() if key != "created_at"}
    rebuilt_stable = {
        key: value for key, value in rebuilt.items() if key != "created_at"
    }
    if stored_stable != rebuilt_stable:
        raise ValueError("keyword section plan differs from its immutable checkpoint")
    return stored


def _print_summary(plan: Mapping[str, object]) -> None:
    print("KEYWORD_SECTION_PLAN=PASS")
    print(f"checkpoint={plan['checkpoint_root']}")
    print(f"section_count={plan['section_count']}")
    print(f"remaining_tasks={plan['remaining_task_count']}")
    for section in plan["sections"]:  # type: ignore[index]
        print(
            f"section_{section['section_index']}_keywords={section['keyword_count']} "
            f"tasks={section['remaining_task_count']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint-root")
    mode.add_argument("--verify-plan")
    parser.add_argument("--output")
    parser.add_argument("--section-count", type=int, default=10)
    parser.add_argument("--partition-salt", default="axis1-30330-ten-sections-v1")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.verify_plan:
        if args.output:
            raise ValueError("--output cannot be used with --verify-plan")
        plan = verify_keyword_section_plan(args.verify_plan)
    else:
        if not args.output:
            raise ValueError("--output is required with --checkpoint-root")
        plan = write_keyword_section_plan(
            args.checkpoint_root,
            args.output,
            section_count=args.section_count,
            partition_salt=args.partition_salt,
        )
    _print_summary(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
