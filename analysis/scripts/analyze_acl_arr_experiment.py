#!/usr/bin/env python3
"""Audit complete ACL ARR cells and write paired intervention outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_analysis import (  # noqa: E402
    analyze_acl_arr_outcomes,
    write_acl_arr_analysis,
)
from analysis.interpretability.pipeline.acl_arr_document_experiment import (  # noqa: E402
    load_plan_from_artifacts,
)


def _rows(values: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for value in values:
        path = Path(value).resolve()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                output.append(row)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--rerank-outcomes", action="append", required=True)
    parser.add_argument("--answer-outcomes", action="append", required=True)
    parser.add_argument("--judge-outcomes", action="append", required=True)
    parser.add_argument("--private-judge-mapping", required=True)
    parser.add_argument("--allow-fake", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = load_plan_from_artifacts(args.plan_manifest)
        analysis = analyze_acl_arr_outcomes(
            _rows(args.rerank_outcomes),
            _rows(args.answer_outcomes),
            _rows(args.judge_outcomes),
            _rows([args.private_judge_mapping]),
            plan=plan,
            allow_fake=args.allow_fake,
        )
        artifacts = write_acl_arr_analysis(args.output_dir, analysis=analysis)
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

    print(f"ANALYSIS={analysis.summary['result']}")
    print(f"SCIENTIFIC_RESULT={analysis.summary['scientific_result']}")
    print(f"PAIRED_ROWS={analysis.summary['paired_prompt_model_count']}")
    print(f"SUMMARY={artifacts.summary_path}")
    print(f"ROWS={artifacts.paired_rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
