#!/usr/bin/env python3
"""Generate the local, prompt-only Milestone 2 calibration corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.prompt_calibration import (  # noqa: E402
    DEFAULT_B_GRID,
    DEFAULT_FIRST_STYLE_SEED,
    DEFAULT_NUMBER_STYLE_SEEDS,
    DEFAULT_PROMPT_SPACE_VERSION,
    DEFAULT_TOP_N,
    write_calibration_corpus,
)

DEFAULT_OUTPUT_DIRECTORY = ANALYSIS_ROOT / "output" / "prompt_calibration"


def _parse_b_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("B grid must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("B grid must contain at least one number")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIRECTORY}).",
    )
    parser.add_argument(
        "--number-style-seeds",
        type=int,
        default=DEFAULT_NUMBER_STYLE_SEEDS,
        help="Number of consecutive style seeds (default: 20).",
    )
    parser.add_argument(
        "--first-style-seed",
        type=int,
        default=DEFAULT_FIRST_STYLE_SEED,
        help="First style seed in the consecutive range (default: 0).",
    )
    parser.add_argument(
        "--b-grid",
        type=_parse_b_grid,
        default=DEFAULT_B_GRID,
        help="Increasing comma-separated B values (default: 0.0,0.1,...,1.0).",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--prompt-space-version",
        default=DEFAULT_PROMPT_SPACE_VERSION,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing manifest and report.",
    )
    args = parser.parse_args()

    try:
        artifacts = write_calibration_corpus(
            args.output_dir,
            b_grid=args.b_grid,
            number_style_seeds=args.number_style_seeds,
            first_style_seed=args.first_style_seed,
            top_n=args.top_n,
            prompt_space_version=args.prompt_space_version,
            overwrite=args.overwrite,
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Generated {len(artifacts.records)} prompt records.")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Report:   {artifacts.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
