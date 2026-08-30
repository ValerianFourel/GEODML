#!/usr/bin/env python3
"""Audit selected-prompt coverage by simulating retrieval across readiness axis 1."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import kstest, spearmanr


FORMAT_VERSION = "selected-readiness-axis-smoothness-v1"
REQUIRED_FIELDS = frozenset(
    {
        "keyword_id",
        "target_normalized_axis_1",
        "consensus_normalized_axis_1",
        "reference_normalized_axis_1",
        "candidate_aligned_normalized_axis_1",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        action="append",
        required=True,
        metavar="LABEL=SELECTED_JSONL",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-points", type=int, default=1001)
    parser.add_argument("--histogram-bins", type=int, default=100)
    parser.add_argument(
        "--coverage-tolerances",
        type=float,
        nargs="+",
        default=(0.017, 0.025, 0.050),
    )
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_sha() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"row at {path}:{line_number} lacks {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"selected population is empty: {path}")
    return rows


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty metric")
    return {
        "minimum": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def audit_selected_population(
    path: str | Path,
    *,
    grid_points: int = 1001,
    histogram_bins: int = 100,
    coverage_tolerances: Iterable[float] = (0.017, 0.025, 0.050),
) -> dict[str, object]:
    if grid_points < 2 or histogram_bins < 2:
        raise ValueError("grid points and histogram bins must be at least two")
    tolerances = tuple(sorted(set(float(value) for value in coverage_tolerances)))
    if not tolerances or any(value <= 0.0 for value in tolerances):
        raise ValueError("coverage tolerances must be positive")

    selected_path = Path(path).resolve()
    rows = _read_jsonl(selected_path)
    by_keyword: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_keyword[str(row["keyword_id"])].append(row)

    coordinates = np.asarray(
        [float(row["consensus_normalized_axis_1"]) for row in rows],
        dtype=np.float64,
    )
    histogram, _ = np.histogram(
        np.clip(coordinates, 0.0, 1.0),
        bins=histogram_bins,
        range=(0.0, 1.0),
    )
    probability = histogram / histogram.sum()
    uniform = np.full(histogram_bins, 1.0 / histogram_bins)
    total_variation = float(0.5 * np.abs(probability - uniform).sum())
    grid = np.linspace(0.0, 1.0, grid_points)

    counts = []
    spans = []
    target_errors = []
    disagreements = []
    correlations = []
    monotonicity = []
    consensus_search_errors = []
    robust_search_errors = []
    keyword_consensus_radii = []
    keyword_robust_radii = []

    for keyword_rows in by_keyword.values():
        counts.append(len(keyword_rows))
        consensus = np.asarray(
            [float(row["consensus_normalized_axis_1"]) for row in keyword_rows]
        )
        reference = np.asarray(
            [float(row["reference_normalized_axis_1"]) for row in keyword_rows]
        )
        candidate = np.asarray(
            [
                float(row["candidate_aligned_normalized_axis_1"])
                for row in keyword_rows
            ]
        )
        targets = np.asarray(
            [float(row["target_normalized_axis_1"]) for row in keyword_rows]
        )
        spans.append(float(np.ptp(consensus)))
        target_errors.extend(np.abs(consensus - targets))
        disagreements.extend(np.abs(reference - candidate))

        order = np.argsort(targets)
        ordered = consensus[order]
        if len(ordered) > 1:
            correlation = float(spearmanr(targets, consensus).statistic)
            if np.isfinite(correlation):
                correlations.append(correlation)
            monotonicity.append(float(np.mean(np.diff(ordered) >= 0.0)))

        nearest_consensus = np.min(
            np.abs(grid[:, None] - consensus[None, :]), axis=1
        )
        nearest_robust = np.min(
            np.maximum(
                np.abs(grid[:, None] - reference[None, :]),
                np.abs(grid[:, None] - candidate[None, :]),
            ),
            axis=1,
        )
        consensus_search_errors.extend(nearest_consensus)
        robust_search_errors.extend(nearest_robust)
        keyword_consensus_radii.append(float(np.max(nearest_consensus)))
        keyword_robust_radii.append(float(np.max(nearest_robust)))

    consensus_errors = np.asarray(consensus_search_errors)
    robust_errors = np.asarray(robust_search_errors)
    coverage = {
        f"{tolerance:.6f}": {
            "consensus_fraction": float(np.mean(consensus_errors <= tolerance)),
            "dual_view_robust_fraction": float(
                np.mean(robust_errors <= tolerance)
            ),
        }
        for tolerance in tolerances
    }
    return {
        "input": {
            "path": str(selected_path),
            "sha256": _sha256(selected_path),
        },
        "prompt_count": len(rows),
        "keyword_count": len(by_keyword),
        "grid_point_count": grid_points,
        "histogram_bin_count": histogram_bins,
        "prompts_per_keyword": _quantiles(counts),
        "outside_unit_interval_count": int(
            np.sum((coordinates < 0.0) | (coordinates > 1.0))
        ),
        "global_axis_span": float(np.ptp(coordinates)),
        "occupied_histogram_bin_count": int(np.count_nonzero(histogram)),
        "histogram_total_variation_from_uniform": total_variation,
        "uniform_ks_distance": float(kstest(coordinates, "uniform").statistic),
        "keyword_axis_span": _quantiles(spans),
        "target_consensus_error": _quantiles(target_errors),
        "cross_view_disagreement": _quantiles(disagreements),
        "target_observed_spearman": _quantiles(correlations),
        "monotonic_adjacent_fraction": _quantiles(monotonicity),
        "consensus_search_distance": _quantiles(consensus_errors),
        "dual_view_robust_search_distance": _quantiles(robust_errors),
        "keyword_worst_consensus_radius": _quantiles(keyword_consensus_radii),
        "keyword_worst_dual_view_radius": _quantiles(keyword_robust_radii),
        "coverage": coverage,
    }


def _parse_populations(values: Iterable[str]) -> list[tuple[str, Path]]:
    populations = []
    labels = set()
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError("--population must use LABEL=SELECTED_JSONL")
        if label in labels:
            raise ValueError(f"duplicate population label: {label}")
        labels.add(label)
        populations.append((label, Path(path)))
    return populations


def _report(payload: Mapping[str, object]) -> str:
    lines = ["# Selected readiness axis-smoothness audit", ""]
    for label, result in payload["populations"].items():
        lines.extend(
            [
                f"## {label}",
                "",
                f"- Prompts: {result['prompt_count']}",
                f"- Keywords: {result['keyword_count']}",
                f"- Global axis span: {result['global_axis_span']:.6f}",
                f"- Occupied histogram bins: "
                f"{result['occupied_histogram_bin_count']} / "
                f"{result['histogram_bin_count']}",
                f"- Histogram total variation from uniform: "
                f"{result['histogram_total_variation_from_uniform']:.6f}",
                f"- Uniform KS distance: {result['uniform_ks_distance']:.6f}",
                f"- Median target-observed Spearman: "
                f"{result['target_observed_spearman']['p50']:.6f}",
                "",
                "| Tolerance | Consensus coverage | Dual-view coverage |",
                "|---:|---:|---:|",
            ]
        )
        for tolerance, coverage in result["coverage"].items():
            lines.append(
                f"| {float(tolerance):.3f} | "
                f"{coverage['consensus_fraction']:.4f} | "
                f"{coverage['dual_view_robust_fraction']:.4f} |"
            )
        lines.append("")
    lines.append(
        "These are prompt-space representation diagnostics, not search-ranking "
        "effects. Embedding coordinates do not define randomized policy B."
    )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    populations = _parse_populations(args.population)
    results = {
        label: audit_selected_population(
            path,
            grid_points=args.grid_points,
            histogram_bins=args.histogram_bins,
            coverage_tolerances=args.coverage_tolerances,
        )
        for label, path in populations
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at": _now(),
        "git_commit_sha": _git_sha(),
        "populations": results,
        "interpretation_guard": (
            "Prompt-space representation diagnostic only; embeddings do not define "
            "randomized policy B and these metrics are not search-ranking effects."
        ),
    }
    output.mkdir(parents=True)
    _atomic_write(
        output / "axis_smoothness_audit.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output / "axis_smoothness_report.md", _report(payload))
    for label, result in results.items():
        print(
            f"population={label} prompts={result['prompt_count']} "
            f"keywords={result['keyword_count']} "
            f"tv={result['histogram_total_variation_from_uniform']:.6f} "
            f"ks={result['uniform_ks_distance']:.6f}"
        )
        for tolerance, coverage in result["coverage"].items():
            print(
                f"population={label} tolerance={float(tolerance):.3f} "
                f"consensus={coverage['consensus_fraction']:.4f} "
                f"dual_view={coverage['dual_view_robust_fraction']:.4f}"
            )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
