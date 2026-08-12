#!/usr/bin/env python3
"""Locate deterministic policy prompts and optionally attach reranker outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.prompt_calibration import (  # noqa: E402
    DEFAULT_B_GRID,
    generate_calibration_records,
)
from interpretability.pipeline.prompt_latent_axis import (  # noqa: E402
    SentenceTransformerPromptEmbedder,
)
from interpretability.pipeline.prompt_policy_mapping import (  # noqa: E402
    FakePolicyPromptEmbedder,
    build_policy_latent_axis,
    locate_policy_prompts,
    map_policy_prompt_to_permutation,
    render_policy_prompt,
    validate_policy_prompt_locations,
    write_policy_mapping_artifacts,
)
from interpretability.pipeline.search_purpose_continuum import (  # noqa: E402
    SearchCandidate,
)


DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "policy_prompt_mapping"


def _parse_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("B grid must contain numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("B grid must not be empty")
    return values


def _load_candidates(path: Path) -> tuple[SearchCandidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidate JSON must be a non-empty array")
    candidates = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"candidate {index} must be an object")
        candidates.append(
            SearchCandidate(
                source_position=row["source_position"],
                title=row.get("title", ""),
                url=row["url"],
                snippet=row.get("snippet", ""),
            )
        )
    return tuple(candidates)


def _load_response_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"response line {line_number} is blank")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"response line {line_number} must be an object")
        required = (
            "prompt_assignment_id",
            "raw_model_output",
            "reranker_run_id",
            "reranker_model",
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                f"response line {line_number} lacks: {', '.join(missing)}"
            )
        rows.append(row)
    if not rows:
        raise ValueError("response JSONL must not be empty")
    return rows


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fake", "sentence-transformer"), default="fake")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--b-grid", type=_parse_grid, default=DEFAULT_B_GRID)
    parser.add_argument("--number-style-seeds", type=int, default=20)
    parser.add_argument("--first-style-seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--prompt-space-version", default="policy-latent-map-v1")
    parser.add_argument("--max-coordinate-error", type=float, required=True)
    parser.add_argument("--max-off-axis-residual", type=float, required=True)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.0)
    parser.add_argument("--query")
    parser.add_argument("--candidates-json")
    parser.add_argument("--responses-jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.candidates_json is None) != (args.query is None):
        parser.error("--query and --candidates-json must be supplied together")
    if args.responses_jsonl and not args.candidates_json:
        parser.error("--responses-jsonl requires --query and --candidates-json")

    try:
        records = generate_calibration_records(
            b_grid=args.b_grid,
            number_style_seeds=args.number_style_seeds,
            first_style_seed=args.first_style_seed,
            top_n=args.top_n,
            prompt_space_version=args.prompt_space_version,
        )
        endpoints = tuple(
            record
            for record in records
            if record.assigned_bias in (0.0, 1.0)
        )
        expected_endpoint_count = args.number_style_seeds * 2
        if len(endpoints) != expected_endpoint_count:
            raise ValueError("B grid must contain both 0 and 1 for every surface seed")
        embedder = (
            FakePolicyPromptEmbedder()
            if args.backend == "fake"
            else SentenceTransformerPromptEmbedder(
                args.embedding_model, device=args.embedding_device
            )
        )
        axis = build_policy_latent_axis(embedder, endpoints)
        locations = locate_policy_prompts(axis, embedder, records)
        validation = validate_policy_prompt_locations(
            locations,
            max_absolute_coordinate_error=args.max_coordinate_error,
            max_matched_off_axis_residual=args.max_off_axis_residual,
            monotonic_tolerance=args.monotonic_tolerance,
        )

        rendered_by_assignment_id = {}
        if args.candidates_json:
            candidates = _load_candidates(Path(args.candidates_json))
            for record in records:
                rendered = render_policy_prompt(
                    record,
                    keyword=" ".join(args.query.split()),
                    candidates=candidates,
                    top_n=args.top_n,
                )
                if rendered.prompt_assignment_id in rendered_by_assignment_id:
                    raise RuntimeError("duplicate prompt assignment ID")
                rendered_by_assignment_id[rendered.prompt_assignment_id] = rendered

        outcomes = []
        if args.responses_jsonl:
            response_rows = _load_response_rows(Path(args.responses_jsonl))
            if len({str(row["prompt_assignment_id"]) for row in response_rows}) != len(response_rows):
                raise ValueError("response JSONL contains duplicate prompt assignment IDs")
            expected = set(rendered_by_assignment_id)
            observed = {str(row["prompt_assignment_id"]) for row in response_rows}
            if observed != expected:
                missing = sorted(expected - observed)
                unknown = sorted(observed - expected)
                raise ValueError(
                    "response prompt assignment IDs do not match prompt bank; "
                    f"missing={missing}, unknown={unknown}"
                )
            locations_by_id = {
                location.prompt_assignment_id: location for location in locations
            }
            for row in response_rows:
                assignment_id = str(row["prompt_assignment_id"])
                outcomes.append(
                    map_policy_prompt_to_permutation(
                        rendered_by_assignment_id[assignment_id],
                        locations_by_id[assignment_id],
                        str(row["raw_model_output"]),
                        reranker_run_id=str(row["reranker_run_id"]),
                        reranker_model=str(row["reranker_model"]),
                    )
                )

        artifacts = write_policy_mapping_artifacts(
            args.output_dir,
            axis=axis,
            locations=locations,
            validation=validation,
            outcomes=outcomes,
            overwrite=args.overwrite,
        )
        if rendered_by_assignment_id:
            rendered_path = Path(args.output_dir) / "rendered_policy_prompts.jsonl"
            if rendered_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite: {rendered_path}")
            _atomic_text(
                rendered_path,
                "".join(
                    json.dumps(asdict(rendered), ensure_ascii=False, separators=(",", ":")) + "\n"
                    for rendered in rendered_by_assignment_id.values()
                ),
            )
            print(f"wrote {rendered_path}")
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    for path in asdict(artifacts).values():
        print(f"wrote {path}")
    if not validation.passed:
        print("latent validation: FAIL (inspect policy_latent_validation.json)")
        return 2
    print("latent validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
