#!/usr/bin/env python3
"""Prepare blinded realized-use judge tasks from completed answer outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_experiment import (  # noqa: E402
    build_blinded_judge_tasks,
    load_plan_from_artifacts,
    write_blinded_judge_plan,
)


def _read_jsonl(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                rows.append(value)
    if not rows:
        raise ValueError("answer outcome inputs are empty")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-outcomes", action="append", required=True)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--judge-model-id", required=True)
    parser.add_argument("--judge-model-revision", required=True)
    parser.add_argument("--master-seed", type=int, default=20260905)
    parser.add_argument(
        "--allow-fake",
        action="store_true",
        help="Testing only. Fake inputs keep the judge plan ineligible for analysis.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = load_plan_from_artifacts(args.plan_manifest)
        answers = _read_jsonl(
            [Path(value).resolve() for value in args.answer_outcomes]
        )
        expected = int(plan.summary["planned_judge_task_count"])
        if not args.allow_fake and len(answers) != expected:
            raise ValueError(
                f"expected {expected} complete answer outcomes, received {len(answers)}"
            )
        judge_plan = build_blinded_judge_tasks(
            answers,
            plan=plan,
            judge_model_id=args.judge_model_id,
            judge_model_revision=args.judge_model_revision,
            master_seed=args.master_seed,
            allow_fake=args.allow_fake,
        )
        artifacts = write_blinded_judge_plan(
            args.output_dir, judge_plan=judge_plan
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"JUDGE_PLAN_ID={judge_plan.judge_plan_id}")
    print(f"JUDGE_TASKS={judge_plan.summary['judge_task_count']}")
    print(f"TASKS={artifacts.tasks_path}")
    print(f"PRIVATE_MAPPING={artifacts.private_mapping_path}")
    print(f"MANIFEST={artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
