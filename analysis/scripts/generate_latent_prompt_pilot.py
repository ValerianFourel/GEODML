#!/usr/bin/env python3
"""Generate query-conditioned prompts by projection onto a learned latent axis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.prompt_latent_axis import (  # noqa: E402
    FakeLatentPromptProvider,
    FakePromptEmbedder,
    LatentPromptGenerationRequest,
    PromptProviderValidationError,
    RepositoryLocalLatentPromptProvider,
    SentenceTransformerPromptEmbedder,
    build_prompt_latent_axis,
    build_latent_prompt_request,
    generate_prompt_at_coordinate,
    render_selected_latent_prompt,
)
from interpretability.pipeline.search_purpose_continuum import SearchCandidate  # noqa: E402


DEFAULT_ENDPOINTS = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "search_purpose_endpoint_pairs_v1.json"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "latent_prompt_pilot"


def _parse_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target grid must contain numbers") from exc
    if not grid:
        raise argparse.ArgumentTypeError("target grid must not be empty")
    return grid


def _load_endpoints(path: Path) -> tuple[list[str], list[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("endpoint bank contains no pairs")
    informational = [str(pair["informational_prompt"]) for pair in pairs]
    transactional = [str(pair["transactional_prompt"]) for pair in pairs]
    return informational, transactional, str(payload["endpoint_bank_version"])


def _load_keyword_candidates(
    *, data_root: Path, engine: str, pool: int, max_keywords: int
) -> dict[str, tuple[SearchCandidate, ...]]:
    path = data_root / "data" / "serp" / f"phase0_top{pool}_{engine}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cached SERP table not found: {path}")
    frame = pd.read_parquet(path)
    output: dict[str, tuple[SearchCandidate, ...]] = {}
    for keyword, rows in frame.groupby("keyword", sort=False):
        seen: set[str] = set()
        candidates: list[SearchCandidate] = []
        for row in rows.sort_values("position").itertuples(index=False):
            url = str(getattr(row, "url", "") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            candidates.append(
                SearchCandidate(
                    source_position=int(row.position),
                    title=str(getattr(row, "title", "") or ""),
                    url=url,
                    snippet=str(getattr(row, "snippet", "") or ""),
                )
            )
            if len(candidates) >= pool:
                break
        output[str(keyword)] = tuple(candidates)
        if max_keywords and len(output) >= max_keywords:
            break
    return output


def _seed(master_seed: int, query: str, target: float, style_seed: int) -> int:
    payload = f"{master_seed}:{query}:{target:.6f}:{style_seed}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


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
    parser.add_argument("--provider", choices=("fake", "local"), default="fake")
    parser.add_argument("--generator-model", default=os.getenv("PROMPT_GENERATOR_MODEL", "fake"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--precision", choices=("full", "4bit"), default="full")
    parser.add_argument("--data-root", default=os.getenv("GEODML_DATA_ROOT", "./geodml_data"))
    parser.add_argument("--engine", choices=("searxng", "ddg"), default="searxng")
    parser.add_argument("--pool", choices=(20, 50), type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-keywords", type=int, default=8)
    parser.add_argument("--target-grid", type=_parse_grid, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--number-style-seeds", type=int, default=2)
    parser.add_argument("--first-style-seed", type=int, default=0)
    parser.add_argument("--number-candidates", type=int, default=3)
    parser.add_argument("--master-seed", type=int, default=20260810)
    parser.add_argument("--endpoint-bank", default=str(DEFAULT_ENDPOINTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_n <= 0 or args.number_style_seeds <= 0 or args.number_candidates <= 0:
        parser.error("top-n, number-style-seeds, and number-candidates must be positive")
    if args.max_keywords < 0:
        parser.error("max-keywords must be nonnegative")
    output_dir = Path(args.output_dir)
    failure_target = output_dir / "latent_prompt_failure.json"
    targets = (
        output_dir / "prompt_latent_axis.json",
        output_dir / "latent_prompt_candidates.jsonl",
        output_dir / "rendered_latent_prompts.jsonl",
        output_dir / "latent_prompt_pilot_report.md",
    )
    existing = [path for path in (*targets, failure_target) if path.exists()]
    if existing and not args.overwrite:
        parser.error("refusing to overwrite: " + ", ".join(str(path) for path in existing))

    try:
        informational, transactional, endpoint_version = _load_endpoints(Path(args.endpoint_bank))
        if args.provider == "fake":
            embedder = FakePromptEmbedder()
            provider = FakeLatentPromptProvider()
        else:
            embedder = SentenceTransformerPromptEmbedder(
                args.embedding_model, device=args.embedding_device
            )
            provider = RepositoryLocalLatentPromptProvider.from_model(
                args.generator_model, precision=args.precision
            )
        axis = build_prompt_latent_axis(
            embedder,
            informational_endpoint_prompts=informational,
            transactional_endpoint_prompts=transactional,
        )
        keyword_candidates = _load_keyword_candidates(
            data_root=Path(args.data_root).resolve(),
            engine=args.engine,
            pool=args.pool,
            max_keywords=args.max_keywords,
        )
        if any(len(values) < args.top_n for values in keyword_candidates.values()):
            raise ValueError("at least one keyword has fewer candidates than top_n")
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    selected_rows: list[dict[str, object]] = []
    rendered_rows: list[dict[str, object]] = []
    for query, candidates in keyword_candidates.items():
        for style_seed in range(
            args.first_style_seed, args.first_style_seed + args.number_style_seeds
        ):
            for target in args.target_grid:
                request = LatentPromptGenerationRequest(
                    query=query,
                    target_coordinate=target,
                    style_seed=style_seed,
                    generation_seed=_seed(args.master_seed, query, target, style_seed),
                    number_candidates=args.number_candidates,
                    generator_model=args.generator_model,
                )
                try:
                    record = generate_prompt_at_coordinate(
                        request,
                        axis=axis,
                        provider=provider,
                        embedder=embedder,
                        generation_parameters={
                            "max_new_tokens": 900,
                            "temperature": 0.9,
                            "top_p": 1.0,
                        },
                    )
                    rendered = render_selected_latent_prompt(
                        record, candidates=candidates, top_n=args.top_n
                    )
                except PromptProviderValidationError as exc:
                    generated_at = datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    )
                    serp_path = (
                        Path(args.data_root).resolve()
                        / "data"
                        / "serp"
                        / f"phase0_top{args.pool}_{args.engine}.parquet"
                    )
                    _atomic_json(
                        failure_target,
                        {
                            "diagnostic_version": "latent-prompt-failure-v1",
                            "generated_at": generated_at,
                            "query": query,
                            "target_coordinate": target,
                            "style_seed": style_seed,
                            "base_generation_seed": request.generation_seed,
                            "number_candidates_requested": args.number_candidates,
                            "candidate_page_count": len(candidates),
                            "serp_input": str(serp_path),
                            "data_root": str(Path(args.data_root).resolve()),
                            "endpoint_bank": str(Path(args.endpoint_bank).resolve()),
                            "axis_id": axis.axis_id,
                            "provider_backend": provider.backend_name,
                            "generator_model": args.generator_model,
                            "generator_precision": args.precision,
                            "embedding_model": args.embedding_model,
                            "embedding_device": args.embedding_device,
                            "meta_prompt_request": build_latent_prompt_request(request),
                            "generation_configuration": {
                                "max_new_tokens": 900,
                                "temperature": 0.9,
                                "top_p": 1.0,
                                "maximum_validation_attempts": len(exc.attempts),
                            },
                            "attempts": [
                                asdict(attempt) for attempt in exc.attempts
                            ],
                        },
                    )
                    parser.error(
                        f"query={query!r} target={target}: {exc}; "
                        f"failure diagnostic: {failure_target}"
                    )
                except (TypeError, ValueError) as exc:
                    parser.error(f"query={query!r} target={target}: {exc}")
                selected_rows.append(asdict(record))
                rendered_rows.append(asdict(rendered))

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _atomic_json(
        targets[0],
        {
            **asdict(axis),
            "endpoint_bank_version": endpoint_version,
            "generated_at": generated_at,
        },
    )
    _atomic_jsonl(targets[1], selected_rows)
    _atomic_jsonl(targets[2], rendered_rows)
    errors = [float(row["absolute_target_error"]) for row in selected_rows]
    report = "\n".join(
        [
            "# Latent prompt pilot report",
            "",
            "> Selected prompts are unvalidated candidates. No reranking or scientific",
            "> inference may use them before semantic validation.",
            "",
            f"- Axis ID: `{axis.axis_id}`",
            f"- Embedding model: `{axis.embedding_model}`",
            f"- Endpoint pairs: {axis.endpoint_pair_count}",
            f"- Queries: {len(keyword_candidates)}",
            f"- Selected prompts: {len(selected_rows)}",
            f"- Mean absolute target error: {sum(errors) / len(errors):.6f}",
            f"- Maximum absolute target error: {max(errors):.6f}",
            f"- Provider: `{provider.backend_name}`",
            "",
            "Assigned target coordinates and observed embedding coordinates are stored",
            "separately. The candidate pool is fixed within each query trajectory.",
            "",
        ]
    )
    _atomic_text(targets[3], report)
    print(f"Axis: {targets[0]}")
    print(f"Selected prompts: {targets[1]} ({len(selected_rows)})")
    print(f"Rendered prompts: {targets[2]}")
    print(f"Report: {targets[3]}")
    print("No reranking model was invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
