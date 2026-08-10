#!/usr/bin/env python3
"""Generate keyword-conditioned search-purpose prompts without model inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.search_purpose_continuum import (  # noqa: E402
    DEFAULT_INTENT_GRID,
    DEFAULT_PROMPT_SPACE_VERSION,
    SearchCandidate,
    write_search_purpose_pilot,
)


DEFAULT_OUTPUT_DIRECTORY = ANALYSIS_ROOT / "output" / "search_purpose_pilot"


def _parse_intent_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "intent grid must be comma-separated numbers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("intent grid must not be empty")
    return values


def _keyword_candidates(frame, *, pool: int, max_keywords: int | None):
    output: dict[str, tuple[SearchCandidate, ...]] = {}
    for keyword, rows in frame.groupby("keyword", sort=False):
        candidates: list[SearchCandidate] = []
        seen_urls: set[str] = set()
        for row in rows.sort_values("position").head(pool).itertuples(index=False):
            url = str(getattr(row, "url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            candidates.append(
                SearchCandidate(
                    source_position=int(row.position),
                    title=str(getattr(row, "title", "") or ""),
                    url=url,
                    snippet=str(getattr(row, "snippet", "") or ""),
                )
            )
        output[str(keyword)] = tuple(candidates)
        if max_keywords is not None and len(output) >= max_keywords:
            break
    return output


def _load_serp(*, data_root: str | None, engine: str, pool: int):
    root = Path(data_root or os.getenv("GEODML_DATA_ROOT", "./geodml_data")).resolve()
    path = root / "data" / "serp" / f"phase0_top{pool}_{engine}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"cached SERP table not found: {path}; download component 'serp' first"
        )
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--engine", choices=("searxng", "ddg"), default="searxng")
    parser.add_argument("--pool", type=int, choices=(20, 50), default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=8,
        help="Prompt-only smoke cap; use 0 for all cached keywords (default: 8).",
    )
    parser.add_argument(
        "--intent-grid",
        type=_parse_intent_grid,
        default=DEFAULT_INTENT_GRID,
        help="Assigned I values (default: 0,0.25,0.5,0.75,1).",
    )
    parser.add_argument("--number-style-seeds", type=int, default=2)
    parser.add_argument("--first-style-seed", type=int, default=0)
    parser.add_argument("--prompt-space-version", default=DEFAULT_PROMPT_SPACE_VERSION)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_keywords < 0:
        parser.error("max-keywords must be nonnegative")
    if args.number_style_seeds <= 0:
        parser.error("number-style-seeds must be greater than zero")
    try:
        frame = _load_serp(
            data_root=args.data_root, engine=args.engine, pool=args.pool
        )
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    keyword_candidates = _keyword_candidates(
        frame,
        pool=args.pool,
        max_keywords=None if args.max_keywords == 0 else args.max_keywords,
    )
    undersized = {
        keyword: len(candidates)
        for keyword, candidates in keyword_candidates.items()
        if len(candidates) < args.top_n
    }
    if undersized:
        parser.error(
            "candidate pools smaller than top_n: "
            + ", ".join(f"{keyword!r}={count}" for keyword, count in undersized.items())
        )
    style_seeds = tuple(
        range(args.first_style_seed, args.first_style_seed + args.number_style_seeds)
    )
    try:
        artifacts = write_search_purpose_pilot(
            args.output_dir,
            keyword_candidates=keyword_candidates,
            intent_grid=args.intent_grid,
            style_seeds=style_seeds,
            top_n=args.top_n,
            prompt_space_version=args.prompt_space_version,
            overwrite=args.overwrite,
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {artifacts.prompt_count} prompt instances; no model invoked.")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Report:   {artifacts.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
