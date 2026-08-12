#!/usr/bin/env python3
"""Build a two-axis prompt candidate/calibration/selection pilot artifact set."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.prompt_latent_axis import (  # noqa: E402
    SentenceTransformerPromptEmbedder,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    DEFAULT_AXIS_GRID,
    FakePairwiseJudge,
    FakeTwoAxisCandidateGenerator,
    FakeTwoAxisPromptEmbedder,
    build_pairwise_comparison_requests,
    calibrate_candidates,
    generate_candidate_bank,
    judge_comparison_requests,
    select_prompt_population,
)


DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "two_axis_prompt_population"


def _parse_grid(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("axis grid must contain numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("axis grid must not be empty")
    return result


def _parse_style_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("style seeds must be integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("style seeds must not be empty")
    return result


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("fake-complete", "mock-bank-only"), default="fake-complete"
    )
    parser.add_argument("--search-term", required=True)
    parser.add_argument("--a1-grid", type=_parse_grid, default=DEFAULT_AXIS_GRID)
    parser.add_argument("--a2-grid", type=_parse_grid, default=DEFAULT_AXIS_GRID)
    parser.add_argument("--style-seeds", type=_parse_style_seeds, default=tuple(range(24)))
    parser.add_argument("--number-candidates", type=int, default=6)
    parser.add_argument("--master-seed", type=int, default=20260812)
    parser.add_argument("--embedding-backend", choices=("fake", "sentence-transformer"), default="fake")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-neighbor-distance", type=float)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    paths = {
        "manifest": output / "run_manifest.json",
        "candidates": output / "two_axis_candidates.jsonl",
        "comparisons": output / "pairwise_comparison_requests.jsonl",
        "judgments": output / "pairwise_judgments.jsonl",
        "calibrations": output / "candidate_calibrations.jsonl",
        "selected": output / "selected_prompt_population.jsonl",
        "diagnostics": output / "selection_diagnostics.json",
        "report": output / "two_axis_population_report.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(map(str, existing)))
    if args.number_candidates < 2:
        parser.error("number-candidates must be at least 2")

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    generator = FakeTwoAxisCandidateGenerator()
    try:
        candidates = generate_candidate_bank(
            search_term=args.search_term,
            a1_grid=args.a1_grid,
            a2_grid=args.a2_grid,
            style_seeds=args.style_seeds,
            number_candidates=args.number_candidates,
            master_seed=args.master_seed,
            generator=generator,
        )
        comparisons = build_pairwise_comparison_requests(candidates)
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_version": "two-axis-prompt-population-pilot-v1",
        "mode": args.mode,
        "scientific_result": False,
        "fake_outputs_support_scientific_claims": False,
        "git_commit_sha": _git_sha(),
        "generated_at": generated_at,
        "search_term": " ".join(args.search_term.split()),
        "a1_grid": list(args.a1_grid),
        "a2_grid": list(args.a2_grid),
        "style_seeds": list(args.style_seeds),
        "number_candidates_per_cell": args.number_candidates,
        "master_seed": args.master_seed,
        "candidate_generator_backend": generator.backend_name,
        "candidate_generator_model": generator.model_name,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "selection_constraints": {
            "monotonic_tolerance": args.monotonic_tolerance,
            "maximum_neighbor_distance": args.maximum_neighbor_distance,
            "no_prompt_reuse": True,
            "one_prompt_per_target_and_style": True,
        },
    }
    _atomic_json(paths["manifest"], manifest)
    _atomic_jsonl(paths["candidates"], (asdict(candidate) for candidate in candidates))
    _atomic_jsonl(paths["comparisons"], (asdict(request) for request in comparisons))
    print(f"wrote {paths['manifest']}")
    print(f"wrote {paths['candidates']}")
    print(f"wrote {paths['comparisons']}")

    if args.mode == "mock-bank-only":
        print("mock-bank-only: external generator/judge stages remain to be completed")
        return 0

    try:
        judgments = judge_comparison_requests(
            comparisons,
            candidates,
            (FakePairwiseJudge("fake-judge-1"), FakePairwiseJudge("fake-judge-2")),
        )
        calibrations = calibrate_candidates(candidates, comparisons, judgments)
        embedder = (
            FakeTwoAxisPromptEmbedder()
            if args.embedding_backend == "fake"
            else SentenceTransformerPromptEmbedder(
                args.embedding_model, device=args.embedding_device
            )
        )
        selected, diagnostics = select_prompt_population(
            candidates,
            calibrations,
            embedder=embedder,
            monotonic_tolerance=args.monotonic_tolerance,
            maximum_neighbor_embedding_distance=args.maximum_neighbor_distance,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    _atomic_jsonl(paths["judgments"], (asdict(judgment) for judgment in judgments))
    _atomic_jsonl(paths["calibrations"], (asdict(item) for item in calibrations))
    _atomic_jsonl(paths["selected"], (asdict(item) for item in selected))
    _atomic_json(paths["diagnostics"], asdict(diagnostics))
    report = f"""# Two-axis prompt-population pilot

> **Mock pipeline output only.** Fake generation, judging, and embedding support
> no scientific claim. This run validates artifact and optimization contracts.

- Search term inserted structurally: `{manifest['search_term']}`
- A1 grid: `{manifest['a1_grid']}`
- A2 grid: `{manifest['a2_grid']}`
- Style trajectories: `{len(args.style_seeds)}`
- Raw candidates: `{len(candidates)}`
- Structurally valid candidates: `{sum(candidate.structural_valid for candidate in candidates)}`
- Blind comparison requests, including reversed presentation: `{len(comparisons)}`
- Pairwise judgments: `{len(judgments)}`
- Globally selected prompts: `{diagnostics.selected_count}`
- Duplicate selected hashes: `{diagnostics.duplicate_hash_count}`
- A1 adjacent reversal rate: `{diagnostics.a1_adjacent_reversal_rate}`
- A2 adjacent reversal rate: `{diagnostics.a2_adjacent_reversal_rate}`
- Fully monotone style rate: `{diagnostics.fully_monotone_style_rate}`
- Mean realized-coordinate L1 error: `{diagnostics.mean_calibration_l1_error}`

Assigned A1/A2 remain the experimental treatments. Bradley--Terry coordinates
are manipulation checks. Prompt embeddings describe local geometry and do not
define assignment. No latent vector is decoded into prompt text.
"""
    _atomic_text(paths["report"], report)
    for key in ("judgments", "calibrations", "selected", "diagnostics", "report"):
        print(f"wrote {paths[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
