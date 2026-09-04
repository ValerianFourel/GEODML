#!/usr/bin/env python3
"""Prepare immutable ACL ARR tasks from audited prompts and frozen documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_experiment import (  # noqa: E402
    ModelConfiguration,
    build_acl_arr_experiment_plan,
    write_acl_arr_experiment_plan,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def _models(path: Path) -> tuple[ModelConfiguration, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("model configuration must contain a non-empty models list")
    return tuple(ModelConfiguration(**row) for row in rows)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--axis-map-jsonl", required=True)
    parser.add_argument("--document-sets-jsonl", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=20260904)
    parser.add_argument(
        "--expected-prompt-count",
        type=int,
        default=26009,
        help="Fail unless the audited prompt count matches; use 0 only for a pilot.",
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=4,
        help="Fail unless this many pinned models are configured.",
    )
    parser.add_argument("--source-git-commit", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.top_n <= 0 or args.expected_prompt_count < 0:
        raise SystemExit("top-n must be positive and expected-prompt-count non-negative")
    if args.expected_model_count <= 0:
        raise SystemExit("expected-model-count must be positive")
    prompts_path = Path(args.prompts_jsonl).resolve()
    axis_path = Path(args.axis_map_jsonl).resolve()
    documents_path = Path(args.document_sets_jsonl).resolve()
    models_path = Path(args.models_json).resolve()
    try:
        prompts = _read_jsonl(prompts_path)
        axis = _read_jsonl(axis_path)
        document_sets = _read_jsonl(documents_path)
        models = _models(models_path)
        if args.expected_prompt_count and len(prompts) != args.expected_prompt_count:
            raise ValueError(
                f"expected {args.expected_prompt_count} prompts, received {len(prompts)}"
            )
        if len(models) != args.expected_model_count:
            raise ValueError(
                f"expected {args.expected_model_count} models, received {len(models)}"
            )
        plan = build_acl_arr_experiment_plan(
            prompts,
            axis,
            document_sets,
            models=models,
            top_n=args.top_n,
            master_seed=args.master_seed,
            prompt_source_sha256=_sha256(prompts_path),
            axis_source_sha256=_sha256(axis_path),
            document_source_sha256=_sha256(documents_path),
            source_git_commit=args.source_git_commit or _git_commit(),
        )
        artifacts = write_acl_arr_experiment_plan(args.output_dir, plan=plan)
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

    print(f"PLAN_ID={plan.plan_id}")
    print(f"PROMPTS={plan.summary['prompt_count']}")
    print(f"MODELS={plan.summary['model_count']}")
    print(f"TASKS_PER_PIPELINE={plan.summary['tasks_per_pipeline']}")
    print(f"PRIMARY_TASKS={plan.summary['primary_task_count']}")
    print(f"PLANNED_JUDGE_TASKS={plan.summary['planned_judge_task_count']}")
    print("MODEL_NATIVE_WEB_SEARCH=DISABLED")
    print(f"MANIFEST={artifacts.manifest_path}")
    for (configuration_id, pipeline), path in artifacts.task_files.items():
        print(f"TASK_FILE={configuration_id}:{pipeline}:{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
