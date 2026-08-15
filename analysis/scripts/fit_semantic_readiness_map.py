#!/usr/bin/env python3
"""Compile LLM labels, embed natural texts, and fit the LLM2Vec readiness map."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_embedding_map import (  # noqa: E402
    evaluate_readiness_embedding_map,
    fit_readiness_embedding_map,
)
from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    ReadinessConsensus,
    ReadinessLabelTask,
    SemanticReadinessItem,
    aggregate_readiness_consensus,
    parse_readiness_judgment,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    labels = stages.add_parser("compile-labels")
    labels.add_argument("--output-dir", required=True)
    labels.add_argument(
        "--tasks",
        required=True,
        nargs="+",
        help="One or more disjoint frozen/transfer task JSONL files.",
    )
    labels.add_argument("--responses", required=True, nargs="+")
    labels.add_argument(
        "--allow-missing-task-id",
        action="append",
        default=[],
        help="Exact response task ID allowed to remain missing; may be repeated.",
    )

    embed = stages.add_parser("embed")
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--corpus", required=True)
    embed.add_argument("--embedding-model", required=True)
    embed.add_argument("--mntp-model")
    embed.add_argument("--peft-model")
    embed.add_argument("--batch-size", type=int, default=8)
    embed.add_argument("--max-length", type=int, default=512)

    fit = stages.add_parser("fit")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--corpus", required=True)
    fit.add_argument("--consensus", required=True)
    fit.add_argument("--embeddings", required=True)
    fit.add_argument("--ridge-penalty", type=float, default=1.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    if args.stage == "compile-labels":
        _compile_labels(args, output)
    elif args.stage == "embed":
        _embed(args, output)
    else:
        _fit(args, output)
    print(f"output: {output}")
    return 0


def _compile_labels(args, output: Path) -> None:
    task_paths = _path_arguments(args.tasks)
    tasks = {}
    for task_path in task_paths:
        for row in _read_jsonl(task_path):
            task = ReadinessLabelTask(**row)
            if task.task_id in tasks:
                raise SystemExit(f"duplicate task_id across task files: {task.task_id}")
            tasks[task.task_id] = task
    responses = [
        row
        for response_path in args.responses
        for row in _read_jsonl(Path(response_path).resolve())
    ]
    response_by_task = {}
    for row in responses:
        task_id = str(row.get("task_id", ""))
        if task_id in response_by_task:
            raise SystemExit(f"duplicate response task_id: {task_id}")
        response_by_task[task_id] = str(row.get("raw_response", ""))
    task_ids = set(tasks)
    response_task_ids = set(response_by_task)
    missing_task_ids = task_ids - response_task_ids
    unknown_task_ids = response_task_ids - task_ids
    allowed_missing_task_ids = {
        str(value).strip()
        for value in args.allow_missing_task_id
        if str(value).strip()
    }
    if unknown_task_ids or missing_task_ids != allowed_missing_task_ids:
        raise SystemExit(
            "response/task mismatch: "
            f"missing={len(missing_task_ids)}, "
            f"unknown={len(unknown_task_ids)}, "
            f"allowed_missing={len(allowed_missing_task_ids)}"
        )
    judgments = tuple(
        parse_readiness_judgment(tasks[task_id], response_by_task[task_id])
        for task_id in sorted(response_task_ids)
    )
    consensus = aggregate_readiness_consensus(judgments)
    _atomic_jsonl(output / "readiness_judgments.jsonl", map(asdict, judgments))
    _atomic_jsonl(output / "readiness_consensus.jsonl", map(asdict, consensus))
    _atomic_json(
        output / "label_diagnostics.json",
        {
            "task_files": [str(path) for path in task_paths],
            "task_file_sha256s": {
                str(path): _sha256_file(path) for path in task_paths
            },
            "task_count": len(tasks),
            "judgment_count": len(judgments),
            "missing_response_count": len(missing_task_ids),
            "missing_response_task_ids": sorted(missing_task_ids),
            "item_count": len(consensus),
            "consensus_judge_count_counts": {
                str(judge_count): sum(
                    item.judge_count == judge_count for item in consensus
                )
                for judge_count in sorted({item.judge_count for item in consensus})
            },
            "usable_item_count": sum(item.usable_for_axis for item in consensus),
            "not_applicable_majority_count": sum(
                item.not_applicable_vote_fraction >= 0.5 for item in consensus
            ),
            "mean_confidence": float(
                np.mean([item.confidence_mean for item in consensus])
            ),
            "mean_global_mad": float(
                np.mean(
                    [item.overall_median_absolute_deviation for item in consensus]
                )
            ),
        },
    )


def _embed(args, output: Path) -> None:
    corpus_path = Path(args.corpus).resolve()
    corpus = tuple(
        SemanticReadinessItem(**row) for row in _read_jsonl(corpus_path)
    )
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    matrix = np.asarray(embedder.embed([item.text for item in corpus]), dtype=np.float32)
    _atomic_npz(
        output / "semantic_readiness_llm2vec_embeddings.npz",
        item_ids=np.asarray([item.item_id for item in corpus], dtype=str),
        embedding_model=np.asarray(embedder.model_name),
        embeddings=matrix,
    )
    _atomic_json(
        output / "embedding_manifest.json",
        {
            "corpus": str(corpus_path),
            "corpus_sha256": _sha256_file(corpus_path),
            "item_count": len(corpus),
            "embedding_model": embedder.model_name,
            "embedding_dimension": int(matrix.shape[1]),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    )


def _fit(args, output: Path) -> None:
    corpus = tuple(
        SemanticReadinessItem(**row)
        for row in _read_jsonl(Path(args.corpus).resolve())
    )
    consensus = tuple(
        ReadinessConsensus(**row)
        for row in _read_jsonl(Path(args.consensus).resolve())
    )
    with np.load(Path(args.embeddings).resolve(), allow_pickle=False) as payload:
        ids = tuple(str(value) for value in payload["item_ids"])
        if ids != tuple(item.item_id for item in corpus):
            raise SystemExit("embedding item IDs do not align with corpus")
        model = str(payload["embedding_model"].item())
        matrix = np.asarray(payload["embeddings"], dtype=np.float64)
    development_indices = [
        index for index, item in enumerate(corpus) if item.split == "development"
    ]
    confirmation_indices = [
        index for index, item in enumerate(corpus) if item.split == "confirmation"
    ]
    development = tuple(corpus[index] for index in development_indices)
    confirmation = tuple(corpus[index] for index in confirmation_indices)
    fitted = fit_readiness_embedding_map(
        development,
        consensus,
        matrix[development_indices],
        embedding_model=model,
        ridge_penalty=args.ridge_penalty,
    )
    dev_coordinates, dev_diagnostics = evaluate_readiness_embedding_map(
        fitted, development, consensus, matrix[development_indices]
    )
    confirm_coordinates, confirm_diagnostics = evaluate_readiness_embedding_map(
        fitted, confirmation, consensus, matrix[confirmation_indices]
    )
    _atomic_json(output / "readiness_embedding_map.json", asdict(fitted))
    _atomic_json(
        output / "readiness_embedding_map_diagnostics.json",
        {
            "development": asdict(dev_diagnostics),
            "confirmation": asdict(confirm_diagnostics),
            "development_by_source": _evaluate_by_source(
                fitted,
                development,
                consensus,
                matrix[development_indices],
            ),
            "confirmation_by_source": _evaluate_by_source(
                fitted,
                confirmation,
                consensus,
                matrix[confirmation_indices],
            ),
        },
    )
    _atomic_jsonl(
        output / "readiness_embedding_coordinates.jsonl",
        (
            {"evaluation_split": "development", **asdict(item)}
            for item in dev_coordinates
        ),
        append_rows=(
            {"evaluation_split": "confirmation", **asdict(item)}
            for item in confirm_coordinates
        ),
    )


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _path_arguments(values) -> tuple[Path, ...]:
    if isinstance(values, (str, Path)):
        values = (values,)
    return tuple(Path(value).resolve() for value in values)


def _evaluate_by_source(fitted, items, consensus, embeddings):
    usable_ids = {item.item_id for item in consensus if item.usable_for_axis}
    source_indices = {}
    for index, item in enumerate(items):
        source_indices.setdefault(item.source_name, []).append(index)
    rows = {}
    for source_name, indices in sorted(source_indices.items()):
        source_items = tuple(items[index] for index in indices)
        usable_count = sum(item.item_id in usable_ids for item in source_items)
        if usable_count == 0:
            rows[source_name] = {
                "status": "no-usable-consensus-labels",
                "item_count": len(source_items),
                "usable_item_count": 0,
            }
            continue
        try:
            _, diagnostics = evaluate_readiness_embedding_map(
                fitted,
                source_items,
                consensus,
                embeddings[indices],
            )
        except ValueError as exc:
            rows[source_name] = {
                "status": "evaluation-error",
                "item_count": len(source_items),
                "usable_item_count": usable_count,
                "error": str(exc),
            }
            continue
        rows[source_name] = {
            "status": "ok",
            "item_count": len(source_items),
            "usable_item_count": usable_count,
            **asdict(diagnostics),
        }
    return rows


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows, *, append_rows=()) -> None:
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    serialized += "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in append_rows
    )
    _atomic_text(path, serialized)


def _atomic_npz(path: Path, **arrays) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
