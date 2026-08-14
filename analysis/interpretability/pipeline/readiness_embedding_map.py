"""Supervised LLM2Vec map for natural-text decision readiness labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

import numpy as np

from .semantic_readiness_dataset import ReadinessConsensus, SemanticReadinessItem


READINESS_MAP_VERSION = "llm2vec-readiness-map-v1"


@dataclass(frozen=True, slots=True)
class ReadinessEmbeddingMap:
    map_id: str
    map_version: str
    embedding_model: str
    dimension: int
    ridge_penalty: float
    training_item_count: int
    embedding_mean: tuple[float, ...]
    label_mean: float
    scalar_direction: tuple[float, ...]
    scalar_unit_direction: tuple[float, ...]
    ordinal_boundaries_0_1: tuple[float, ...]
    ordinal_plane_offsets: tuple[float, ...]
    rubric_names: tuple[str, ...]
    rubric_coefficient_matrix: tuple[tuple[float, ...], ...]
    rubric_singular_values: tuple[float, ...]
    rubric_first_component_share: float
    supervised_subspace_axes: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class ReadinessMapCoordinate:
    item_id: str
    split: str
    source_name: str
    observed_readiness_0_1: float
    consensus_readiness_0_1: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class ReadinessMapDiagnostics:
    item_count: int
    source_count: int
    spearman: float
    mean_absolute_error: float
    pairwise_order_accuracy: float
    usable_label_range: tuple[float, float]
    rubric_first_component_share: float
    source_spearman: tuple[tuple[str, float | None], ...]


def fit_readiness_embedding_map(
    items: Sequence[SemanticReadinessItem],
    consensus: Sequence[ReadinessConsensus],
    embeddings: np.ndarray,
    *,
    embedding_model: str,
    ridge_penalty: float = 1.0,
) -> ReadinessEmbeddingMap:
    """Fit scalar level-set planes and a multirubric supervised subspace."""

    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive")
    rows, labels, matrix = _aligned_usable(items, consensus, embeddings)
    if len(rows) < 10:
        raise ValueError("at least ten usable labeled items are required")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    y = np.asarray([item.overall_readiness_0_100 / 100.0 for item in labels])
    label_mean = float(np.mean(y))
    scalar = _ridge_coefficients(centered, (y - label_mean)[:, None], ridge_penalty)[:, 0]
    norm = float(np.linalg.norm(scalar))
    if norm <= 1e-12:
        raise ValueError("labels do not identify a nonzero LLM2Vec direction")
    rubric_names = (
        "information_seeking_reverse",
        "evaluation",
        "selection_commitment",
        "action_implementation",
    )
    rubric = np.asarray(
        [
            (
                (7.0 - item.information_seeking_1_7) / 6.0,
                (item.evaluation_1_7 - 1.0) / 6.0,
                (item.selection_commitment_1_7 - 1.0) / 6.0,
                (item.action_implementation_1_7 - 1.0) / 6.0,
            )
            for item in labels
        ],
        dtype=np.float64,
    )
    rubric_centered = rubric - np.mean(rubric, axis=0)
    coefficients = _ridge_coefficients(centered, rubric_centered, ridge_penalty)
    left, singular, _ = np.linalg.svd(coefficients, full_matrices=False)
    variance = singular**2
    first_share = float(variance[0] / max(np.sum(variance), 1e-12))
    axes = left[:, : min(2, left.shape[1])].T
    boundaries = (0.125, 0.375, 0.625, 0.875)
    offsets = tuple(float(boundary - label_mean + scalar @ mean) for boundary in boundaries)
    identity = json.dumps(
        {
            "version": READINESS_MAP_VERSION,
            "embedding_model": embedding_model,
            "item_ids": [item.item_id for item in rows],
            "direction_hash": hashlib.sha256(scalar.astype("<f8").tobytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ReadinessEmbeddingMap(
        map_id=f"readiness-map:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        map_version=READINESS_MAP_VERSION,
        embedding_model=embedding_model,
        dimension=matrix.shape[1],
        ridge_penalty=ridge_penalty,
        training_item_count=len(rows),
        embedding_mean=tuple(float(value) for value in mean),
        label_mean=label_mean,
        scalar_direction=tuple(float(value) for value in scalar),
        scalar_unit_direction=tuple(float(value) for value in scalar / norm),
        ordinal_boundaries_0_1=boundaries,
        ordinal_plane_offsets=offsets,
        rubric_names=rubric_names,
        rubric_coefficient_matrix=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
        rubric_singular_values=tuple(float(value) for value in singular),
        rubric_first_component_share=first_share,
        supervised_subspace_axes=tuple(
            tuple(float(value) for value in row) for row in axes
        ),
    )


def evaluate_readiness_embedding_map(
    fitted: ReadinessEmbeddingMap,
    items: Sequence[SemanticReadinessItem],
    consensus: Sequence[ReadinessConsensus],
    embeddings: np.ndarray,
) -> tuple[tuple[ReadinessMapCoordinate, ...], ReadinessMapDiagnostics]:
    rows, labels, matrix = _aligned_usable(items, consensus, embeddings)
    if matrix.shape[1] != fitted.dimension:
        raise ValueError("embedding dimension does not match readiness map")
    direction = np.asarray(fitted.scalar_direction, dtype=np.float64)
    center = np.asarray(fitted.embedding_mean, dtype=np.float64)
    predicted = fitted.label_mean + (matrix - center) @ direction
    observed = np.asarray(
        [item.overall_readiness_0_100 / 100.0 for item in labels], dtype=np.float64
    )
    coordinates = tuple(
        ReadinessMapCoordinate(
            item_id=item.item_id,
            split=item.split,
            source_name=item.source_name,
            observed_readiness_0_1=float(score),
            consensus_readiness_0_1=float(target),
            absolute_error=abs(float(score - target)),
        )
        for item, score, target in zip(rows, predicted, observed)
    )
    source_metrics = []
    for source in sorted({item.source_name for item in rows}):
        indices = [index for index, item in enumerate(rows) if item.source_name == source]
        correlation = (
            _spearman(observed[indices], predicted[indices]) if len(indices) >= 3 else None
        )
        source_metrics.append((source, correlation))
    diagnostics = ReadinessMapDiagnostics(
        item_count=len(rows),
        source_count=len(source_metrics),
        spearman=_spearman(observed, predicted),
        mean_absolute_error=float(np.mean(np.abs(predicted - observed))),
        pairwise_order_accuracy=_pairwise_order_accuracy(observed, predicted),
        usable_label_range=(float(np.min(observed)), float(np.max(observed))),
        rubric_first_component_share=fitted.rubric_first_component_share,
        source_spearman=tuple(source_metrics),
    )
    return coordinates, diagnostics


def _aligned_usable(items, consensus, embeddings):
    rows = tuple(items)
    matrix = _unit_rows(np.asarray(embeddings, dtype=np.float64))
    if len(matrix) != len(rows):
        raise ValueError("embeddings must align with every corpus item")
    label_by_id = {item.item_id: item for item in consensus if item.usable_for_axis}
    indices = [index for index, item in enumerate(rows) if item.item_id in label_by_id]
    if not indices:
        raise ValueError("no usable consensus labels align with corpus items")
    return (
        tuple(rows[index] for index in indices),
        tuple(label_by_id[rows[index].item_id] for index in indices),
        matrix[indices],
    )


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    sample_count, dimension = x.shape
    if dimension <= sample_count:
        return np.linalg.solve(x.T @ x + penalty * np.eye(dimension), x.T @ y)
    dual = np.linalg.solve(x @ x.T + penalty * np.eye(sample_count), y)
    return x.T @ dual


def _unit_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite non-empty matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings contain a zero-norm row")
    return values / norms


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    correlation = np.corrcoef(left_rank, right_rank)[0, 1]
    return float(correlation)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _pairwise_order_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    order = np.argsort(observed, kind="mergesort")
    if len(order) < 2:
        raise ValueError("pairwise accuracy needs at least two items")
    values = [
        predicted[right] > predicted[left]
        for left, right in zip(order[:-1], order[1:])
        if observed[right] > observed[left]
    ]
    if not values:
        raise ValueError("pairwise accuracy has no distinct-label pairs")
    return float(np.mean(values))
