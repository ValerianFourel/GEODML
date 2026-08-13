#!/usr/bin/env python3
"""Fit the primary query-prior LLM2Vec informational-to-transactional A1 axis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.a1_embedding_axis import (  # noqa: E402
    A1_EMBEDDING_AXIS_VERSION,
    build_query_prior_endpoint_prompts,
    fit_query_prior_a1_axis,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--serp-parquet", required=True)
    parser.add_argument("--keyword-column", default="keyword")
    parser.add_argument("--expected-keywords", type=int)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--mntp-model")
    parser.add_argument("--peft-model")
    parser.add_argument("--style-seeds", default="0,1,2,3")
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--encode-max-length", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        styles = _integers(args.style_seeds)
        if args.query_chunk_size <= 0:
            raise ValueError("query-chunk-size must be positive")
        output = Path(args.output_dir).resolve()
        _prepare_output(output, resume=args.resume)
        serp_path = Path(args.serp_parquet).resolve()
        queries = _load_queries(serp_path, args.keyword_column)
        if args.expected_keywords is not None and len(queries) != args.expected_keywords:
            raise ValueError(
                f"expected {args.expected_keywords} search terms, loaded {len(queries)}"
            )

        embedder = LLM2VecPromptEmbedder(
            args.embedding_model,
            mntp_model_name_or_path=args.mntp_model,
            peft_model_name_or_path=args.peft_model,
            batch_size=args.encode_batch_size,
            max_length=args.encode_max_length,
        )
        cache = output / "cache" / "endpoint_embeddings"
        info_chunks: list[np.ndarray] = []
        trans_chunks: list[np.ndarray] = []
        pair_keys: list[tuple[str, int]] = []
        for start in range(0, len(queries), args.query_chunk_size):
            chunk_queries = queries[start : start + args.query_chunk_size]
            identity = {
                "version": A1_EMBEDDING_AXIS_VERSION,
                "embedding_model": embedder.model_name,
                "queries": chunk_queries,
                "style_seeds": styles,
                "encode_max_length": args.encode_max_length,
            }
            key = _hash_json(identity)
            path = cache / f"{start:05d}-{key[:16]}.npz"
            if path.exists():
                informational, transactional, keys = _read_cache(path, identity)
            else:
                info_texts, trans_texts, keys = build_query_prior_endpoint_prompts(
                    chunk_queries,
                    style_seeds=styles,
                )
                informational = np.asarray(embedder.embed(info_texts), dtype=np.float64)
                transactional = np.asarray(embedder.embed(trans_texts), dtype=np.float64)
                _atomic_npz(
                    path,
                    identity=np.asarray(json.dumps(identity, sort_keys=True)),
                    informational=informational.astype(np.float32),
                    transactional=transactional.astype(np.float32),
                    pair_keys=np.asarray(keys, dtype=str),
                )
            info_chunks.append(informational)
            trans_chunks.append(transactional)
            pair_keys.extend((str(query), int(style)) for query, style in keys)
            print(
                f"endpoint chunks: {min(start + len(chunk_queries), len(queries))}/"
                f"{len(queries)} queries",
                flush=True,
            )

        axis, endpoints, diagnostics = fit_query_prior_a1_axis(
            np.concatenate(info_chunks, axis=0),
            np.concatenate(trans_chunks, axis=0),
            pair_keys=pair_keys,
            embedding_model=embedder.model_name,
        )
        manifest = {
            "artifact_version": A1_EMBEDDING_AXIS_VERSION,
            "status": "axis-fitted-unreviewed",
            "scientific_result": False,
            "coordinate_definition": "primary-llm2vec-query-prior-projection",
            "git_commit_sha": _git_sha(),
            "generated_at": _now(),
            "serp_parquet": str(serp_path),
            "serp_parquet_sha256": _sha256_file(serp_path),
            "keyword_column": args.keyword_column,
            "query_count": len(queries),
            "style_seeds": styles,
            "embedding_model": args.embedding_model,
            "mntp_model": args.mntp_model,
            "peft_model": args.peft_model,
            "resolved_embedding_model": embedder.model_name,
            "encode_batch_size": args.encode_batch_size,
            "encode_max_length": args.encode_max_length,
            "query_chunk_size": args.query_chunk_size,
            "axis_id": axis.axis_id,
            "axis_dimension": axis.dimension,
            "endpoint_pair_count": axis.endpoint_pair_count,
        }
        _atomic_json(output / "a1_embedding_axis.json", asdict(axis))
        _atomic_npz(
            output / "a1_embedding_axis_state.npz",
            direction=np.asarray(axis.direction, dtype=np.float32),
            informational_anchor=np.asarray(axis.informational_anchor),
            transactional_anchor=np.asarray(axis.transactional_anchor),
        )
        _atomic_jsonl(
            output / "a1_endpoint_projections.jsonl",
            (asdict(row) for row in endpoints),
        )
        _atomic_json(output / "a1_embedding_axis_diagnostics.json", asdict(diagnostics))
        _atomic_json(output / "run_manifest.json", manifest)
        _atomic_text(output / "a1_embedding_axis_report.md", _report(axis, diagnostics))
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
    print(f"fitted {axis.axis_id} from {axis.endpoint_pair_count} query/style pairs")
    print(f"output: {output}")
    return 0


def _prepare_output(output: Path, *, resume: bool) -> None:
    if output.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {output}")
        if not output.is_dir():
            raise ValueError("output path is not a directory")
        if (output / "run_manifest.json").exists():
            raise ValueError("embedding axis is already complete")
        allowed = {
            "cache",
            "logs",
            "a1_embedding_axis.json",
            "a1_embedding_axis_state.npz",
            "a1_endpoint_projections.jsonl",
            "a1_embedding_axis_diagnostics.json",
            "a1_embedding_axis_report.md",
        }
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
        if unexpected:
            raise ValueError(
                "unexpected files make axis resume unsafe: " + ", ".join(unexpected)
            )
    else:
        output.mkdir(parents=True)


def _read_cache(path: Path, expected_identity: dict[str, object]):
    with np.load(path, allow_pickle=False) as payload:
        identity = json.loads(str(payload["identity"].item()))
        if identity != expected_identity:
            raise ValueError(f"endpoint embedding cache identity mismatch: {path}")
        informational = np.asarray(payload["informational"], dtype=np.float64)
        transactional = np.asarray(payload["transactional"], dtype=np.float64)
        keys = tuple((str(row[0]), int(row[1])) for row in payload["pair_keys"])
    if informational.shape != transactional.shape or informational.shape[0] != len(keys):
        raise ValueError(f"invalid endpoint embedding cache shapes: {path}")
    return informational, transactional, keys


def _load_queries(path: Path, keyword_column: str) -> tuple[str, ...]:
    frame = pd.read_parquet(path, columns=[keyword_column])
    values = tuple(
        dict.fromkeys(
            " ".join(str(value).split())
            for value in frame[keyword_column].tolist()
            if value is not None and str(value).strip()
        )
    )
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError("SERP search terms collide after case normalization")
    return values


def _integers(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("style-seeds must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise ValueError("style-seeds must be non-empty and unique")
    return values


def _report(axis, diagnostics) -> str:
    return "\n".join(
        (
            "# Primary query-prior LLM2Vec A1 axis",
            "",
            f"- Axis ID: `{axis.axis_id}`",
            f"- Search-term prior: `{diagnostics.query_count}` queries",
            f"- Matched surface styles: `{diagnostics.style_count}`",
            f"- Endpoint pairs: `{diagnostics.endpoint_pair_count}`",
            f"- Embedding dimension: `{diagnostics.embedding_dimension}`",
            f"- Global endpoint gap: `{diagnostics.global_centroid_gap}`",
            f"- Positive pair-gap rate: `{diagnostics.positive_pair_gap_rate}`",
            f"- Positive query-mean-gap rate: `{diagnostics.positive_query_mean_gap_rate}`",
            f"- Minimum query-mean gap: `{diagnostics.minimum_query_mean_gap}`",
            "",
            "The primary A1 coordinate is projection onto this frozen semantic vector,",
            "oriented from informational to transactional search purpose. The vector was",
            "identified before candidate selection and ranking outcomes. Generator target",
            "labels and LLM pairwise judgments do not define this coordinate.",
            "",
        )
    )


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
