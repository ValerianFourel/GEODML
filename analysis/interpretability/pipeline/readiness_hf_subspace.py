"""Fit a supervised readiness subspace from an assembled annotation bundle."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, fields
import hashlib
import itertools
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from .readiness_embedding_map import (
    evaluate_readiness_embedding_map,
    fit_readiness_embedding_map,
)
from .readiness_hf_dataset import (
    READINESS_HF_FORMAT_VERSION,
    atomic_json,
    atomic_jsonl,
    load_complete_embedding_view,
    read_jsonl,
    sha256_file,
)
from .semantic_readiness_dataset import (
    ReadinessConsensus,
    SemanticReadinessItem,
    normalize_semantic_readiness_text,
)


READINESS_SUBSPACE_FORMAT_VERSION = "semantic-readiness-supervised-subspace-v1"
_SCORE_RANGES = {
    "overall_readiness_0_100": (0, 100),
    "information_seeking_1_7": (1, 7),
    "evaluation_1_7": (1, 7),
    "selection_commitment_1_7": (1, 7),
    "action_implementation_1_7": (1, 7),
}


def fit_readiness_hf_subspace(
    *,
    prompts_path: str | Path,
    annotations_path: str | Path,
    embedding_dir: str | Path,
    output_dir: str | Path,
    judge_slots: Sequence[str],
    git_commit_sha: str,
    ridge_penalty: float = 1.0,
    minimum_rating_judges: int = 2,
    minimum_mean_confidence: float = 0.60,
    maximum_global_mad: float = 15.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Join bundle artifacts by immutable IDs and fit on development only."""

    prompts_path = Path(prompts_path).resolve()
    annotations_path = Path(annotations_path).resolve()
    embedding_dir = Path(embedding_dir).resolve()
    output = Path(output_dir).resolve()
    slots = tuple(str(value).strip() for value in judge_slots if str(value).strip())
    if len(slots) < 2 or len(set(slots)) != len(slots):
        raise ValueError("judge_slots must contain at least two unique values")
    if minimum_rating_judges < 2 or minimum_rating_judges > len(slots):
        raise ValueError("minimum_rating_judges must be between two and panel size")
    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive")
    if not git_commit_sha.strip():
        raise ValueError("git_commit_sha must be nonempty")
    for path in (prompts_path, annotations_path):
        if not path.is_file():
            raise ValueError(f"missing required input: {path}")
    if output.exists():
        raise ValueError(f"refusing to overwrite subspace directory: {output}")

    prompt_rows = read_jsonl(prompts_path)
    items = _load_items(prompt_rows)
    _progress(progress, f"validated prompts: {len(items)}")
    annotations = read_jsonl(annotations_path)
    consensus, label_diagnostics = _aggregate_annotations(
        items,
        annotations,
        judge_slots=slots,
        minimum_rating_judges=minimum_rating_judges,
        minimum_mean_confidence=minimum_mean_confidence,
        maximum_global_mad=maximum_global_mad,
    )
    _progress(
        progress,
        "aggregated selected judges: "
        f"annotations={label_diagnostics['annotation_count']} "
        f"usable={label_diagnostics['usable_item_count']}",
    )
    embedding_manifest, embedding_rows = load_complete_embedding_view(embedding_dir)
    item_ids = tuple(item.item_id for item in items)
    if set(embedding_rows) != set(item_ids):
        raise ValueError("embedding and prompt item IDs differ")
    for item in items:
        embedded_hash, _ = embedding_rows[item.item_id]
        if embedded_hash != item.text_sha256:
            raise ValueError(f"embedding text hash mismatch: {item.item_id}")
    matrix = np.stack(
        [embedding_rows[item.item_id][1] for item in items], axis=0
    ).astype(np.float32, copy=False)
    del embedding_rows
    _progress(
        progress,
        f"validated embeddings: items={len(matrix)} dimension={matrix.shape[1]}",
    )

    development_indices = tuple(
        index for index, item in enumerate(items) if item.split == "development"
    )
    confirmation_indices = tuple(
        index for index, item in enumerate(items) if item.split == "confirmation"
    )
    if not development_indices or not confirmation_indices:
        raise ValueError("prompts must contain development and confirmation splits")
    development = tuple(items[index] for index in development_indices)
    confirmation = tuple(items[index] for index in confirmation_indices)
    embedding_model = _embedding_model_reference(embedding_manifest)
    _progress(
        progress,
        "fitting development subspace: "
        f"prompts={len(development)} ridge_penalty={ridge_penalty}",
    )
    fitted = fit_readiness_embedding_map(
        development,
        consensus,
        matrix[np.asarray(development_indices)],
        embedding_model=embedding_model,
        ridge_penalty=ridge_penalty,
    )
    _progress(progress, f"fitted map: {fitted.map_id}")
    dev_coordinates, dev_diagnostics = evaluate_readiness_embedding_map(
        fitted,
        development,
        consensus,
        matrix[np.asarray(development_indices)],
    )
    confirm_coordinates, confirm_diagnostics = evaluate_readiness_embedding_map(
        fitted,
        confirmation,
        consensus,
        matrix[np.asarray(confirmation_indices)],
    )
    _progress(
        progress,
        "evaluated frozen map: "
        f"development={len(dev_coordinates)} confirmation={len(confirm_coordinates)}",
    )
    diagnostics = {
        "label_panel": label_diagnostics,
        "geometry_robustness": {
            "ridge_ordinal_cosine_similarity": (
                fitted.ridge_ordinal_cosine_similarity
            ),
            "pca_method": fitted.pca_method,
            "pca_random_seed": fitted.pca_random_seed,
            "pca_explained_variance_ratio": fitted.pca_explained_variance_ratio,
            "ridge_pca_absolute_cosine_similarity": (
                fitted.ridge_pca_absolute_cosine_similarity
            ),
            "ordinal_pca_absolute_cosine_similarity": (
                fitted.ordinal_pca_absolute_cosine_similarity
            ),
        },
        "development": asdict(dev_diagnostics),
        "confirmation": asdict(confirm_diagnostics),
        "development_by_source": _evaluate_by_source(
            fitted,
            development,
            consensus,
            matrix[np.asarray(development_indices)],
        ),
        "confirmation_by_source": _evaluate_by_source(
            fitted,
            confirmation,
            consensus,
            matrix[np.asarray(confirmation_indices)],
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        atomic_jsonl(
            temporary / "readiness_consensus.jsonl",
            (asdict(item) for item in consensus),
        )
        atomic_json(temporary / "readiness_embedding_map.json", asdict(fitted))
        atomic_json(
            temporary / "readiness_embedding_map_diagnostics.json", diagnostics
        )
        atomic_jsonl(
            temporary / "readiness_embedding_coordinates.jsonl",
            itertools.chain(
                (
                    {"evaluation_split": "development", **asdict(item)}
                    for item in dev_coordinates
                ),
                (
                    {"evaluation_split": "confirmation", **asdict(item)}
                    for item in confirm_coordinates
                ),
            ),
        )
        artifacts = {
            name: {
                "sha256": sha256_file(temporary / name),
                "size_bytes": (temporary / name).stat().st_size,
            }
            for name in (
                "readiness_consensus.jsonl",
                "readiness_embedding_map.json",
                "readiness_embedding_map_diagnostics.json",
                "readiness_embedding_coordinates.jsonl",
            )
        }
        usable_ids = {item.item_id for item in consensus if item.usable_for_axis}
        manifest = {
            "format_version": READINESS_SUBSPACE_FORMAT_VERSION,
            "bundle_format_version": READINESS_HF_FORMAT_VERSION,
            "git_commit_sha": git_commit_sha,
            "inputs": {
                "prompts": _file_identity(prompts_path),
                "annotations": _file_identity(annotations_path),
                "embedding_dir": str(embedding_dir),
                "embedding_manifest_sha256": sha256_file(
                    embedding_dir / "embedding_manifest.json"
                ),
                "embedding_view_config_sha256": embedding_manifest.get(
                    "view_config_sha256"
                ),
            },
            "judge_slots": list(slots),
            "excluded_judge_slots": sorted(
                {
                    str(row.get("judge_slot", ""))
                    for row in annotations
                    if str(row.get("judge_slot", "")) not in slots
                }
                - {""}
            ),
            "label_policy": {
                "numeric_aggregation": "median overall; mean Likert fields",
                "abstentions": (
                    "not_applicable and dont_know do not contribute numeric scores"
                ),
                "minimum_rating_judges": minimum_rating_judges,
                "minimum_mean_confidence": minimum_mean_confidence,
                "maximum_global_mad": maximum_global_mad,
            },
            "split_policy": (
                "fit supervised and PCA geometry on development only; "
                "evaluate confirmation without refitting"
            ),
            "prompt_count": len(items),
            "development_prompt_count": len(development),
            "confirmation_prompt_count": len(confirmation),
            "usable_development_count": sum(
                item.item_id in usable_ids for item in development
            ),
            "usable_confirmation_count": sum(
                item.item_id in usable_ids for item in confirmation
            ),
            "embedding_dimension": int(matrix.shape[1]),
            "embedding_model": embedding_model,
            "map_id": fitted.map_id,
            "training_item_count": fitted.training_item_count,
            "ridge_penalty": ridge_penalty,
            "artifacts": artifacts,
        }
        atomic_json(temporary / "subspace_manifest.json", manifest)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _progress(progress, f"wrote immutable subspace: {output}")
    return manifest


def _load_items(
    rows: Sequence[Mapping[str, object]],
) -> tuple[SemanticReadinessItem, ...]:
    names = {field.name for field in fields(SemanticReadinessItem)}
    items = tuple(
        SemanticReadinessItem(**{name: row[name] for name in names}) for row in rows
    )
    if not items or len({item.item_id for item in items}) != len(items):
        raise ValueError("prompts must be nonempty and uniquely identified")
    if any(item.split not in {"development", "confirmation"} for item in items):
        raise ValueError("every prompt needs a frozen development/confirmation split")
    for item in items:
        actual = hashlib.sha256(
            normalize_semantic_readiness_text(item.text).encode("utf-8")
        ).hexdigest()
        if actual != item.text_sha256:
            raise ValueError(f"prompt text hash mismatch: {item.item_id}")
    return items


def _aggregate_annotations(
    items: Sequence[SemanticReadinessItem],
    rows: Sequence[Mapping[str, object]],
    *,
    judge_slots: Sequence[str],
    minimum_rating_judges: int,
    minimum_mean_confidence: float,
    maximum_global_mad: float,
) -> tuple[tuple[ReadinessConsensus, ...], dict[str, object]]:
    item_ids = {item.item_id for item in items}
    slots = set(judge_slots)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    seen = set()
    answer_types = Counter()
    slot_counts = Counter()
    for row in rows:
        slot = str(row.get("judge_slot", ""))
        if slot not in slots:
            continue
        item_id = str(row.get("item_id", ""))
        if item_id not in item_ids:
            raise ValueError(f"annotation references unknown prompt: {item_id}")
        key = (item_id, slot)
        if key in seen:
            raise ValueError(f"duplicate item/judge annotation: {key}")
        seen.add(key)
        answer_type = str(row.get("answer_type", ""))
        if answer_type not in {"rating", "not_applicable", "dont_know"}:
            raise ValueError(f"unknown annotation answer_type: {answer_type}")
        _validate_annotation(row, answer_type=answer_type)
        grouped.setdefault(item_id, []).append(row)
        answer_types[answer_type] += 1
        slot_counts[slot] += 1
    missing_slots = slots - set(slot_counts)
    if missing_slots:
        raise ValueError(
            f"selected judge slots have no annotations: {sorted(missing_slots)}"
        )

    consensus = []
    numeric_counts = Counter()
    panel_counts = Counter()
    no_rating_count = 0
    for item in items:
        group = grouped.get(item.item_id, [])
        panel_counts[len(group)] += 1
        ratings = [row for row in group if row["answer_type"] == "rating"]
        numeric_counts[len(ratings)] += 1
        if not ratings:
            no_rating_count += 1
            continue
        overall = np.asarray(
            [row["overall_readiness_0_100"] for row in ratings], dtype=np.float64
        )
        median = float(np.median(overall))
        mad = float(np.median(np.abs(overall - median)))
        confidence = float(np.mean([row["confidence_0_1"] for row in ratings]))
        abstention_fraction = 1.0 - len(ratings) / max(len(group), 1)
        not_applicable_fraction = (
            sum(row["answer_type"] == "not_applicable" for row in group)
            / max(len(group), 1)
        )
        consensus.append(
            ReadinessConsensus(
                item_id=item.item_id,
                judge_count=len(ratings),
                overall_readiness_0_100=median,
                information_seeking_1_7=float(
                    np.mean([row["information_seeking_1_7"] for row in ratings])
                ),
                evaluation_1_7=float(
                    np.mean([row["evaluation_1_7"] for row in ratings])
                ),
                selection_commitment_1_7=float(
                    np.mean([row["selection_commitment_1_7"] for row in ratings])
                ),
                action_implementation_1_7=float(
                    np.mean([row["action_implementation_1_7"] for row in ratings])
                ),
                not_applicable_vote_fraction=not_applicable_fraction,
                ambiguity_mean=float(
                    np.mean([row["ambiguity_1_7"] for row in ratings])
                ),
                confidence_mean=confidence,
                overall_median_absolute_deviation=mad,
                usable_for_axis=(
                    len(ratings) >= minimum_rating_judges
                    and abstention_fraction < 0.5
                    and confidence >= minimum_mean_confidence
                    and mad <= maximum_global_mad
                ),
            )
        )
    by_id = {item.item_id: item for item in consensus}
    usable_ids = {item.item_id for item in consensus if item.usable_for_axis}
    diagnostics = {
        "selected_judge_slots": list(judge_slots),
        "annotation_count": sum(slot_counts.values()),
        "annotation_counts_by_slot": dict(sorted(slot_counts.items())),
        "answer_type_counts": dict(sorted(answer_types.items())),
        "panel_annotation_count_distribution": {
            str(key): value for key, value in sorted(panel_counts.items())
        },
        "numeric_rating_count_distribution": {
            str(key): value for key, value in sorted(numeric_counts.items())
        },
        "consensus_item_count": len(consensus),
        "no_numeric_rating_item_count": no_rating_count,
        "usable_item_count": len(usable_ids),
        "unusable_consensus_item_count": sum(
            not item.usable_for_axis for item in consensus
        ),
        "usable_counts_by_split": {
            split: sum(
                item.item_id in usable_ids for item in items if item.split == split
            )
            for split in ("development", "confirmation")
        },
        "consensus_counts_by_split": {
            split: sum(
                item.item_id in by_id for item in items if item.split == split
            )
            for split in ("development", "confirmation")
        },
    }
    return tuple(consensus), diagnostics


def _validate_annotation(row: Mapping[str, object], *, answer_type: str) -> None:
    for key, (lower, upper) in _SCORE_RANGES.items():
        value = row.get(key)
        if answer_type == "rating":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer for rating annotations")
            if not lower <= value <= upper:
                raise ValueError(f"{key} is outside [{lower}, {upper}]")
        elif value is not None:
            raise ValueError(f"{key} must be null for {answer_type} annotations")
    for key, lower, upper in (
        ("ambiguity_1_7", 1, 7),
        ("confidence_0_1", 0.0, 1.0),
    ):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not lower <= float(value) <= upper:
            raise ValueError(f"{key} is outside [{lower}, {upper}]")


def _embedding_model_reference(manifest: Mapping[str, object]) -> str:
    view = manifest.get("view")
    if not isinstance(view, Mapping):
        raise ValueError("embedding manifest omits frozen view metadata")
    view_name = str(view.get("view_name", "")).strip()
    model_id = str(view.get("embedding_model_id", "")).strip()
    revision = str(view.get("embedding_model_revision", "")).strip()
    if not view_name or not model_id or not revision:
        raise ValueError("embedding view identity is incomplete")
    return f"{view_name}:{model_id}@{revision}"


def _evaluate_by_source(fitted, items, consensus, embeddings):
    usable_ids = {item.item_id for item in consensus if item.usable_for_axis}
    source_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        source_indices.setdefault(item.source_name, []).append(index)
    rows = {}
    for source_name, indices in sorted(source_indices.items()):
        source_items = tuple(items[index] for index in indices)
        usable_count = sum(item.item_id in usable_ids for item in source_items)
        if usable_count < 2:
            rows[source_name] = {
                "status": "insufficient-usable-consensus-labels",
                "item_count": len(source_items),
                "usable_item_count": usable_count,
            }
            continue
        try:
            _, diagnostics = evaluate_readiness_embedding_map(
                fitted,
                source_items,
                consensus,
                embeddings[np.asarray(indices)],
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


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)
