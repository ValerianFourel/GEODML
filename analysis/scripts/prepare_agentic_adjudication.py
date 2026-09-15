#!/usr/bin/env python3
"""Route blind judge disagreements into a GLM adjudication queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_judging import (  # noqa: E402
    AgenticJudgeTask,
    build_adjudication_plan,
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
    return rows


def _outcomes(roots: list[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for root in roots:
        paths = (
            [root]
            if root.name == "outcomes.jsonl"
            else sorted(root.rglob("outcomes.jsonl"))
        )
        for path in paths:
            for row in _read_jsonl(path):
                case_id = row.get("blind_case_id")
                parsed = row.get("parsed_output")
                if not isinstance(case_id, str) or not isinstance(parsed, dict):
                    raise ValueError(f"invalid agentic judge outcome: {path}")
                if case_id in indexed:
                    raise ValueError(f"duplicate judge outcome for {case_id}")
                indexed[case_id] = parsed
    return indexed


def _atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument(
        "--bulk-results-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--validation-results-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite adjudication plan: {output}")
    try:
        judge_manifest_path = arguments.judge_manifest.resolve()
        manifest = json.loads(judge_manifest_path.read_text(encoding="utf-8"))
        bulk_path = Path(manifest["artifacts"]["bulk_tasks"]["path"])
        if _sha256(bulk_path) != manifest["artifacts"]["bulk_tasks"]["sha256"]:
            raise ValueError("canonical bulk judge queue hash mismatch")
        tasks = [AgenticJudgeTask.from_dict(row) for row in _read_jsonl(bulk_path)]
        task_by_case = {task.blind_case_id: task for task in tasks}
        bulk = _outcomes([path.resolve() for path in arguments.bulk_results_root])
        validation = _outcomes(
            [path.resolve() for path in arguments.validation_results_root]
        )
        unknown = (set(bulk) | set(validation)) - set(task_by_case)
        if unknown:
            raise ValueError("judge outcomes contain unknown blind case IDs")
        decisions = build_adjudication_plan(
            bulk_outcomes=bulk,
            validation_outcomes=validation,
            all_case_ids=sorted(task_by_case),
        )
        queued = [
            task_by_case[case_id]
            for case_id, decision in decisions.items()
            if decision["resolution"] == "queue_validation_judgment"
        ]
        resolved = []
        for case_id, decision in decisions.items():
            if decision["resolution"] == "accept_bulk_judgment":
                role, judgment = "bulk", bulk[case_id]
            elif decision["resolution"] == "use_existing_validation_judgment":
                role, judgment = "validation", validation[case_id]
            else:
                continue
            resolved.append(
                {
                    "blind_case_id": case_id,
                    "selected_judge_role": role,
                    "routing_reasons": decision["reasons"],
                    "judgment": judgment,
                }
            )
        output.mkdir(parents=True)
        tasks_path = output / "validation_adjudication_tasks.jsonl"
        with tasks_path.open("x", encoding="utf-8") as stream:
            for task in queued:
                stream.write(json.dumps(task.to_dict(), separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        decisions_path = output / "adjudication_decisions.json"
        _atomic_json(decisions_path, decisions)
        resolved_path = output / "resolved_judgments.jsonl"
        with resolved_path.open("x", encoding="utf-8") as stream:
            for row in resolved:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path = output / "run_manifest.json"
        _atomic_json(
            manifest_path,
            {
                "format_version": "agentic-adjudication-plan-v1",
                "status": "planned",
                "scientific_result": False,
                "source_judge_manifest": {
                    "path": str(judge_manifest_path),
                    "sha256": _sha256(judge_manifest_path),
                },
                "bulk_outcome_count": len(bulk),
                "existing_validation_outcome_count": len(validation),
                "adjudication_task_count": len(queued),
                "resolved_judgment_count": len(resolved),
                "tasks": {"path": str(tasks_path), "sha256": _sha256(tasks_path)},
                "decisions": {
                    "path": str(decisions_path),
                    "sha256": _sha256(decisions_path),
                },
                "resolved_judgments": {
                    "path": str(resolved_path),
                    "sha256": _sha256(resolved_path),
                },
            },
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"BULK_OUTCOMES={len(bulk)}")
    print(f"EXISTING_VALIDATION_OUTCOMES={len(validation)}")
    print(f"ADJUDICATION_TASKS={len(queued)}")
    print(f"RESOLVED_JUDGMENTS={len(resolved)}")
    print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
