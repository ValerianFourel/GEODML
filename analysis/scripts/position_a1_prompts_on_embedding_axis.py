#!/usr/bin/env python3
"""Select query-bound prompts by position on the frozen primary LLM2Vec A1 axis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.a1_embedding_axis import (  # noqa: E402
    A1EndpointProjection,
    QueryPriorA1Axis,
)
from interpretability.pipeline.a1_embedding_panel import (  # noqa: E402
    A1CandidateCoordinate,
    A1_EMBEDDING_PANEL_VERSION,
    balanced_query_style_assignment,
    build_positioned_rows,
    measure_candidate_coordinates,
    randomize_positioned_schedule,
    render_candidate_for_measurement,
    select_embedding_trajectory,
)
from interpretability.pipeline.a1_prompt_manifold import (  # noqa: E402
    A1Candidate,
    stratified_random_a1_grid,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--axis-json", required=True)
    parser.add_argument("--endpoint-projections", required=True)
    parser.add_argument("--serp-parquet", required=True)
    parser.add_argument("--keyword-column", default="keyword")
    parser.add_argument("--expected-keywords", type=int)
    parser.add_argument("--expected-candidates", type=int, default=1440)
    parser.add_argument("--target-level-count", type=int, default=30)
    parser.add_argument("--master-seed", type=int, default=20260817)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--mntp-model")
    parser.add_argument("--peft-model")
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        output = Path(args.output_dir).resolve()
        _prepare_output(output, resume=args.resume)
        started_at = _now()
        candidate_path = Path(args.candidate_bank).resolve()
        axis_path = Path(args.axis_json).resolve()
        endpoint_path = Path(args.endpoint_projections).resolve()
        serp_path = Path(args.serp_parquet).resolve()
        candidates = tuple(A1Candidate(**row) for row in _read_jsonl(candidate_path))
        if len(candidates) != args.expected_candidates:
            raise ValueError(
                f"expected {args.expected_candidates} candidates, loaded {len(candidates)}"
            )
        if not all(candidate.structural_valid for candidate in candidates):
            raise ValueError("candidate bank contains structurally invalid prompts")
        if len({candidate.candidate_hash for candidate in candidates}) != len(candidates):
            raise ValueError("candidate bank contains duplicate prompt hashes")
        axis = QueryPriorA1Axis(**json.loads(axis_path.read_text(encoding="utf-8")))
        endpoints = tuple(
            A1EndpointProjection(**row) for row in _read_jsonl(endpoint_path)
        )
        endpoint_map = {(row.search_term, row.style_seed): row for row in endpoints}
        queries = _load_queries(serp_path, args.keyword_column)
        if args.expected_keywords is not None and len(queries) != args.expected_keywords:
            raise ValueError(
                f"expected {args.expected_keywords} search terms, loaded {len(queries)}"
            )
        styles = axis.style_seeds
        by_style = {
            style: tuple(
                sorted(
                    (candidate for candidate in candidates if candidate.style_seed == style),
                    key=lambda candidate: candidate.candidate_id,
                )
            )
            for style in styles
        }
        if any(not group for group in by_style.values()):
            raise ValueError("candidate bank lacks an axis surface style")
        style_pool_sizes = {style: len(group) for style, group in by_style.items()}
        if len(set(style_pool_sizes.values())) != 1:
            raise ValueError(f"candidate pools are not balanced by style: {style_pool_sizes}")
        if sum(style_pool_sizes.values()) != len(candidates):
            raise ValueError("candidate bank contains styles outside the frozen axis")
        if any((query, style) not in endpoint_map for query in queries for style in styles):
            raise ValueError("endpoint projections do not cover every query/style pair")
        targets = stratified_random_a1_grid(
            args.target_level_count,
            master_seed=args.master_seed,
        )
        assignments = balanced_query_style_assignment(
            queries,
            styles,
            master_seed=args.master_seed,
        )
        embedder = LLM2VecPromptEmbedder(
            args.embedding_model,
            mntp_model_name_or_path=args.mntp_model,
            peft_model_name_or_path=args.peft_model,
            batch_size=args.encode_batch_size,
            max_length=args.encode_max_length,
        )
        if embedder.model_name != axis.embedding_model:
            raise ValueError(
                "embedding model does not match the frozen axis: "
                f"{embedder.model_name!r} != {axis.embedding_model!r}"
            )

        cache = output / "cache" / "query_candidate_projections"
        candidate_sha = _sha256_file(candidate_path)
        axis_sha = _sha256_file(axis_path)
        candidates_by_style_and_id = {
            style: {candidate.candidate_id: candidate for candidate in group}
            for style, group in by_style.items()
        }
        all_rows = []
        for query, style, keyword_order in assignments:
            style_candidates = by_style[style]
            identity = {
                "version": A1_EMBEDDING_PANEL_VERSION,
                "axis_id": axis.axis_id,
                "axis_sha256": axis_sha,
                "candidate_bank_sha256": candidate_sha,
                "embedding_model": embedder.model_name,
                "encode_max_length": args.encode_max_length,
                "query": query,
                "style_seed": style,
            }
            cache_path = cache / f"{keyword_order:04d}-{_hash_json(identity)[:16]}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("identity") != identity:
                    raise ValueError(f"query projection cache identity mismatch: {cache_path}")
                coordinates = tuple(
                    A1CandidateCoordinate(**row) for row in payload["coordinates"]
                )
            else:
                texts = tuple(
                    render_candidate_for_measurement(candidate, query)
                    for candidate in style_candidates
                )
                embeddings = embedder.embed(texts)
                coordinates = measure_candidate_coordinates(
                    axis=axis,
                    endpoint=endpoint_map[(query, style)],
                    candidates=style_candidates,
                    embeddings=embeddings,
                )
                _atomic_json(
                    cache_path,
                    {
                        "identity": identity,
                        "coordinates": [asdict(row) for row in coordinates],
                    },
                )
            selected = select_embedding_trajectory(coordinates, targets)
            rows = build_positioned_rows(
                search_term=query,
                style_seed=style,
                keyword_order=keyword_order,
                targets=targets,
                selected_coordinates=selected,
                candidates_by_id=candidates_by_style_and_id[style],
                axis_id=axis.axis_id,
            )
            all_rows.extend(rows)
            print(f"positioned queries: {keyword_order}/{len(queries)}", flush=True)

        scheduled = randomize_positioned_schedule(
            all_rows,
            master_seed=args.master_seed,
        )
        diagnostics = _diagnostics(scheduled, styles, len(queries), targets)
        manifest = {
            "artifact_version": A1_EMBEDDING_PANEL_VERSION,
            "status": "embedding-positioned-unrun",
            "scientific_result": False,
            "primary_coordinate": "observed_a1",
            "coordinate_definition": (
                "matched-query-style projection on frozen query-prior LLM2Vec vector"
            ),
            "generator_assigned_a1_role": "proposal-metadata-only",
            "llm_judgment_role": "auxiliary-validation-only",
            "git_commit_sha": _git_sha(),
            "started_at": started_at,
            "completed_at": _now(),
            "master_seed": args.master_seed,
            "axis_id": axis.axis_id,
            "axis_json": str(axis_path),
            "axis_json_sha256": axis_sha,
            "endpoint_projections": str(endpoint_path),
            "endpoint_projections_sha256": _sha256_file(endpoint_path),
            "candidate_bank": str(candidate_path),
            "candidate_bank_sha256": candidate_sha,
            "serp_parquet": str(serp_path),
            "serp_parquet_sha256": _sha256_file(serp_path),
            "embedding_model": embedder.model_name,
            "query_count": len(queries),
            "target_level_count": len(targets),
            "assignment_count": len(scheduled),
            "style_assignment": "one-query-level-style-balanced",
            "candidate_pool_per_query": len(next(iter(by_style.values()))),
            "candidate_sets_bound": False,
            "outcomes_observed": False,
            "environment": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
        _atomic_jsonl(
            output / "embedding_positioned_a1_prompt_panel.jsonl",
            (asdict(row) for row in scheduled),
        )
        _atomic_json(output / "a1_embedding_panel_diagnostics.json", diagnostics)
        _atomic_text(output / "a1_embedding_panel_report.md", _report(manifest, diagnostics))
        _atomic_json(output / "run_manifest.json", manifest)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"positioned {len(scheduled)} prompts: {len(targets)} per query x "
        f"{len(queries)} queries"
    )
    print(f"output: {output}")
    return 0


def _prepare_output(output: Path, *, resume: bool) -> None:
    if not output.exists():
        output.mkdir(parents=True)
        return
    if not resume:
        raise FileExistsError(f"output directory already exists: {output}")
    if (output / "run_manifest.json").exists():
        raise ValueError("embedding-positioned panel is already complete")
    allowed = {
        "cache",
        "logs",
        "embedding_positioned_a1_prompt_panel.jsonl",
        "a1_embedding_panel_diagnostics.json",
        "a1_embedding_panel_report.md",
    }
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
    if unexpected:
        raise ValueError("unexpected files make resume unsafe: " + ", ".join(unexpected))


def _diagnostics(rows, styles, query_count, targets) -> dict[str, object]:
    errors = [row.absolute_target_error for row in rows]
    by_query: dict[str, list] = {}
    for row in rows:
        by_query.setdefault(row.search_term, []).append(row)
    if len(by_query) != query_count:
        raise ValueError("positioned rows do not cover every query")
    if any(len(group) != len(targets) for group in by_query.values()):
        raise ValueError("positioned query blocks do not contain every target")
    if any(len({row.style_seed for row in group}) != 1 for group in by_query.values()):
        raise ValueError("a positioned query block contains multiple surface styles")
    style_query_counts = {
        str(style): sum(group[0].style_seed == style for group in by_query.values())
        for style in styles
    }
    strict = sum(
        _strictly_increasing(group)
        for group in by_query.values()
    )
    return {
        "query_count": query_count,
        "targets_per_query": len(targets),
        "assignment_count": len(rows),
        "primary_coordinate": "observed_a1",
        "mean_absolute_target_error": statistics.fmean(errors),
        "median_absolute_target_error": statistics.median(errors),
        "maximum_absolute_target_error": max(errors),
        "fully_strict_embedding_trajectory_rate": strict / query_count,
        "below_zero_coordinate_rate": sum(row.observed_a1 < 0 for row in rows) / len(rows),
        "above_one_coordinate_rate": sum(row.observed_a1 > 1 for row in rows) / len(rows),
        "minimum_observed_a1": min(row.observed_a1 for row in rows),
        "maximum_observed_a1": max(row.observed_a1 for row in rows),
        "style_query_counts": style_query_counts,
        "maximum_query_style_count_imbalance": (
            max(style_query_counts.values()) - min(style_query_counts.values())
        ),
        "duplicate_panel_assignment_count": (
            len(rows) - len({row.panel_assignment_id for row in rows})
        ),
        "duplicate_query_candidate_pair_count": (
            len(rows)
            - len({(row.search_term, row.source_candidate_id) for row in rows})
        ),
    }


def _strictly_increasing(group) -> bool:
    ordered = sorted(group, key=lambda row: row.axis_order)
    return all(
        right.observed_a1 > left.observed_a1
        for left, right in zip(ordered, ordered[1:])
    )


def _report(manifest, diagnostics) -> str:
    return "\n".join(
        (
            "# Embedding-positioned A1 prompt panel",
            "",
            f"- Axis: `{manifest['axis_id']}`",
            f"- Queries: `{diagnostics['query_count']}`",
            f"- Targets per query: `{diagnostics['targets_per_query']}`",
            f"- Assignments: `{diagnostics['assignment_count']}`",
            f"- Mean absolute target error: `{diagnostics['mean_absolute_target_error']}`",
            f"- Maximum absolute target error: `{diagnostics['maximum_absolute_target_error']}`",
            "- Strict embedding trajectory rate: "
            f"`{diagnostics['fully_strict_embedding_trajectory_rate']}`",
            f"- Style query counts: `{diagnostics['style_query_counts']}`",
            "",
            "`observed_a1` is the primary semantic coordinate. It is the candidate's",
            "LLM2Vec projection on the frozen query-prior informational-to-transactional",
            "vector, scaled by the matched query/style endpoint pair. Generator proposal",
            "labels and Qwen judgments do not define the coordinate.",
            "",
            "Candidate sets and ranking outcomes are not yet bound or observed.",
            "",
        )
    )


def _load_queries(path: Path, column: str) -> tuple[str, ...]:
    frame = pd.read_parquet(path, columns=[column])
    normalized = tuple(
        " ".join(str(value).split())
        for value in frame[column].tolist()
        if value is not None and str(value).strip()
    )
    queries = tuple(dict.fromkeys(normalized))
    if len({query.casefold() for query in queries}) != len(queries):
        raise ValueError("search terms are duplicated after case-insensitive normalization")
    return queries


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _hash_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
