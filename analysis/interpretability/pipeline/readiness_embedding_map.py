"""Supervised LLM2Vec map for natural-text decision readiness labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Sequence

import numpy as np

from .semantic_readiness_dataset import ReadinessConsensus, SemanticReadinessItem


READINESS_MAP_VERSION = "llm2vec-readiness-map-v3"
PCA_RANDOM_SEED = 20260817
PCA_COMPONENT_COUNT = 10


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
    ordinal_rubric_names: tuple[str, ...]
    ordinal_direction: tuple[float, ...]
    ordinal_unit_direction: tuple[float, ...]
    ordinal_thresholds_by_rubric: tuple[tuple[float, ...], ...]
    ridge_ordinal_cosine_similarity: float
    pca_method: str
    pca_random_seed: int
    pca_axes: tuple[tuple[float, ...], ...]
    pca_explained_variance_ratio: tuple[float, ...]
    ridge_pca_absolute_cosine_similarity: tuple[float, ...]
    ordinal_pca_absolute_cosine_similarity: tuple[float, ...]
    compute_backend: str = "numpy"


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
    ordinal_spearman: float
    ordinal_pairwise_order_accuracy: float
    ridge_ordinal_cosine_similarity: float
    source_spearman: tuple[tuple[str, float | None], ...]


def fit_readiness_embedding_map(
    items: Sequence[SemanticReadinessItem],
    consensus: Sequence[ReadinessConsensus],
    embeddings: np.ndarray,
    *,
    embedding_model: str,
    ridge_penalty: float = 1.0,
    compute_backend: str = "numpy",
    progress: Callable[[str], None] | None = None,
) -> ReadinessEmbeddingMap:
    """Fit scalar level-set planes and a multirubric supervised subspace."""

    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive")
    if compute_backend not in {"numpy", "torch-cuda"}:
        raise ValueError("compute_backend must be numpy or torch-cuda")
    if compute_backend == "torch-cuda":
        _require_torch_cuda()
    all_matrix = _unit_rows(np.asarray(embeddings, dtype=np.float64))
    if len(all_matrix) != len(items):
        raise ValueError("embeddings must align with every corpus item")
    rows, labels, matrix = _aligned_usable(items, consensus, all_matrix)
    if len(rows) < 10:
        raise ValueError("at least ten usable labeled items are required")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    y = np.asarray([item.overall_readiness_0_100 / 100.0 for item in labels])
    label_mean = float(np.mean(y))
    _emit(progress, f"scalar ridge solve on {compute_backend}")
    scalar = _ridge_coefficients(
        centered,
        (y - label_mean)[:, None],
        ridge_penalty,
        compute_backend=compute_backend,
    )[:, 0]
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
    _emit(progress, f"multirubric ridge solve on {compute_backend}")
    coefficients = _ridge_coefficients(
        centered,
        rubric_centered,
        ridge_penalty,
        compute_backend=compute_backend,
    )
    left, singular, _ = np.linalg.svd(coefficients, full_matrices=False)
    variance = singular**2
    first_share = float(variance[0] / max(np.sum(variance), 1e-12))
    axes = left[:, : min(2, left.shape[1])].T
    _emit(progress, "ordinal proportional-odds fit on scipy-cpu")
    ordinal_coefficients, ordinal_thresholds = _fit_ordinal_coefficients(
        centered,
        labels,
        ridge_penalty,
    )
    ordinal_left, _, _ = np.linalg.svd(ordinal_coefficients, full_matrices=False)
    ordinal_direction = ordinal_left[:, 0]
    if float(ordinal_direction @ scalar) < 0:
        ordinal_direction = -ordinal_direction
    ordinal_norm = float(np.linalg.norm(ordinal_direction))
    if ordinal_norm <= 1e-12:
        raise ValueError("ordinal labels do not identify a nonzero direction")
    ordinal_unit = ordinal_direction / ordinal_norm
    scalar_unit = scalar / norm
    ridge_ordinal_cosine = float(scalar_unit @ ordinal_unit)
    _emit(progress, f"randomized PCA on {compute_backend}")
    pca_axes, pca_variance = _randomized_pca(
        all_matrix - np.mean(all_matrix, axis=0),
        component_count=PCA_COMPONENT_COUNT,
        random_seed=PCA_RANDOM_SEED,
        compute_backend=compute_backend,
    )
    ridge_pca_cosine = tuple(float(abs(axis @ scalar_unit)) for axis in pca_axes)
    ordinal_pca_cosine = tuple(float(abs(axis @ ordinal_unit)) for axis in pca_axes)
    boundaries = (0.125, 0.375, 0.625, 0.875)
    offsets = tuple(float(boundary - label_mean + scalar @ mean) for boundary in boundaries)
    identity = json.dumps(
        {
            "version": READINESS_MAP_VERSION,
            "embedding_model": embedding_model,
            "item_ids": [item.item_id for item in rows],
            "direction_hash": hashlib.sha256(scalar.astype("<f8").tobytes()).hexdigest(),
            "ordinal_direction_hash": hashlib.sha256(
                ordinal_unit.astype("<f8").tobytes()
            ).hexdigest(),
            "pca_seed": PCA_RANDOM_SEED,
            "compute_backend": compute_backend,
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
        scalar_unit_direction=tuple(float(value) for value in scalar_unit),
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
        ordinal_rubric_names=rubric_names,
        ordinal_direction=tuple(float(value) for value in ordinal_direction),
        ordinal_unit_direction=tuple(float(value) for value in ordinal_unit),
        ordinal_thresholds_by_rubric=ordinal_thresholds,
        ridge_ordinal_cosine_similarity=ridge_ordinal_cosine,
        pca_method=(
            "deterministic-randomized-svd-torch-cuda-v1"
            if compute_backend == "torch-cuda"
            else "deterministic-randomized-svd-v1"
        ),
        pca_random_seed=PCA_RANDOM_SEED,
        pca_axes=tuple(tuple(float(value) for value in row) for row in pca_axes),
        pca_explained_variance_ratio=tuple(float(value) for value in pca_variance),
        ridge_pca_absolute_cosine_similarity=ridge_pca_cosine,
        ordinal_pca_absolute_cosine_similarity=ordinal_pca_cosine,
        compute_backend=compute_backend,
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
    ordinal_direction = np.asarray(fitted.ordinal_unit_direction, dtype=np.float64)
    center = np.asarray(fitted.embedding_mean, dtype=np.float64)
    predicted = fitted.label_mean + (matrix - center) @ direction
    observed = np.asarray(
        [item.overall_readiness_0_100 / 100.0 for item in labels], dtype=np.float64
    )
    ordinal_predicted = (matrix - center) @ ordinal_direction
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
        ordinal_spearman=_spearman(observed, ordinal_predicted),
        ordinal_pairwise_order_accuracy=_pairwise_order_accuracy(
            observed, ordinal_predicted
        ),
        ridge_ordinal_cosine_similarity=fitted.ridge_ordinal_cosine_similarity,
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


def _ridge_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    *,
    compute_backend: str = "numpy",
) -> np.ndarray:
    if compute_backend == "torch-cuda":
        return _torch_ridge_coefficients(x, y, penalty, device_name="cuda")
    sample_count, dimension = x.shape
    if dimension <= sample_count:
        return np.linalg.solve(x.T @ x + penalty * np.eye(dimension), x.T @ y)
    dual = np.linalg.solve(x @ x.T + penalty * np.eye(sample_count), y)
    return x.T @ dual


def _fit_ordinal_coefficients(
    x: np.ndarray,
    labels: Sequence[ReadinessConsensus],
    penalty: float,
) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    """Fit one proportional-odds direction per frozen 1--7 Likert field."""

    ordinal_targets = np.asarray(
        [
            (
                8.0 - item.information_seeking_1_7,
                item.evaluation_1_7,
                item.selection_commitment_1_7,
                item.action_implementation_1_7,
            )
            for item in labels
        ],
        dtype=np.float64,
    )
    coefficients = []
    thresholds = []
    for column in ordinal_targets.T:
        # Consensus Likert values can be thirds. Rounding converts them back to the
        # declared seven ordered response levels without using the continuous label.
        target = np.clip(np.floor(column + 0.5), 1, 7).astype(np.int64)
        coefficient, cuts = _fit_proportional_odds(x, target, penalty)
        coefficients.append(coefficient)
        thresholds.append(tuple(float(value) for value in cuts))
    return np.column_stack(coefficients), tuple(thresholds)


def _fit_proportional_odds(
    x: np.ndarray,
    target: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import minimize

    levels, encoded = np.unique(target, return_inverse=True)
    if len(levels) < 2:
        raise ValueError("ordinal supervision needs at least two observed levels")
    cut_count = len(levels) - 1
    cumulative = np.asarray(
        [np.mean(encoded <= index) for index in range(cut_count)], dtype=np.float64
    )
    cumulative = np.clip(cumulative, 1e-5, 1.0 - 1e-5)
    initial_cuts = np.log(cumulative / (1.0 - cumulative))
    raw_cuts = np.empty(cut_count, dtype=np.float64)
    raw_cuts[0] = initial_cuts[0]
    if cut_count > 1:
        differences = np.maximum(np.diff(initial_cuts), 1e-4)
        raw_cuts[1:] = np.log(np.expm1(differences))
    initial = np.concatenate([np.zeros(x.shape[1], dtype=np.float64), raw_cuts])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient = parameters[: x.shape[1]]
        raw = parameters[x.shape[1] :]
        cuts = _ordered_cuts(raw)
        eta = x @ coefficient
        upper = np.ones(len(x), dtype=np.float64)
        lower = np.zeros(len(x), dtype=np.float64)
        upper_derivative = np.zeros(len(x), dtype=np.float64)
        lower_derivative = np.zeros(len(x), dtype=np.float64)
        for category in range(len(levels)):
            mask = encoded == category
            if category < cut_count:
                values = _sigmoid(cuts[category] - eta[mask])
                upper[mask] = values
                upper_derivative[mask] = values * (1.0 - values)
            if category > 0:
                values = _sigmoid(cuts[category - 1] - eta[mask])
                lower[mask] = values
                lower_derivative[mask] = values * (1.0 - values)
        probability = np.maximum(upper - lower, 1e-12)
        value = -float(np.sum(np.log(probability)))
        value += 0.5 * penalty * float(coefficient @ coefficient)
        probability_eta_derivative = -upper_derivative + lower_derivative
        eta_gradient = -probability_eta_derivative / probability
        coefficient_gradient = x.T @ eta_gradient + penalty * coefficient
        cut_gradient = np.zeros(cut_count, dtype=np.float64)
        for category in range(len(levels)):
            mask = encoded == category
            if category < cut_count:
                cut_gradient[category] -= float(
                    np.sum(upper_derivative[mask] / probability[mask])
                )
            if category > 0:
                cut_gradient[category - 1] += float(
                    np.sum(lower_derivative[mask] / probability[mask])
                )
        raw_gradient = np.empty_like(raw)
        raw_gradient[0] = float(np.sum(cut_gradient))
        if cut_count > 1:
            raw_gradient[1:] = _sigmoid(raw[1:]) * np.cumsum(cut_gradient[::-1])[::-1][1:]
        return value, np.concatenate([coefficient_gradient, raw_gradient])

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"ordinal optimizer failed: {result.message}")
    return result.x[: x.shape[1]], _ordered_cuts(result.x[x.shape[1] :])


def _ordered_cuts(raw: np.ndarray) -> np.ndarray:
    cuts = np.empty_like(raw)
    cuts[0] = raw[0]
    if len(raw) > 1:
        cuts[1:] = raw[0] + np.cumsum(np.logaddexp(0.0, raw[1:]) + 1e-8)
    return cuts


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _randomized_pca(
    centered: np.ndarray,
    *,
    component_count: int,
    random_seed: int,
    compute_backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic approximate PCs without treating them as readiness."""

    if compute_backend == "torch-cuda":
        return _torch_randomized_pca(
            centered,
            component_count=component_count,
            random_seed=random_seed,
            device_name="cuda",
        )

    maximum = min(centered.shape)
    count = min(component_count, maximum)
    projection_count = min(maximum, count + 8)
    rng = np.random.default_rng(random_seed)
    projection = rng.standard_normal((centered.shape[1], projection_count))
    basis, _ = np.linalg.qr(centered @ projection, mode="reduced")
    for _ in range(2):
        basis, _ = np.linalg.qr(centered @ (centered.T @ basis), mode="reduced")
    _, singular_values, right = np.linalg.svd(basis.T @ centered, full_matrices=False)
    axes = right[:count].copy()
    for axis in axes:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1
    total_variance = float(np.sum(centered**2))
    ratios = singular_values[:count] ** 2 / max(total_variance, 1e-12)
    return axes, ratios


def _require_torch_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise ValueError("torch-cuda backend requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise ValueError("torch-cuda backend requires a visible CUDA GPU")


def _torch_ridge_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    *,
    device_name: str,
) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device)
    sample_count, dimension = x_tensor.shape
    if dimension <= sample_count:
        gram = x_tensor.T @ x_tensor
        gram.diagonal().add_(penalty)
        coefficients = torch.linalg.solve(gram, x_tensor.T @ y_tensor)
    else:
        gram = x_tensor @ x_tensor.T
        gram.diagonal().add_(penalty)
        coefficients = x_tensor.T @ torch.linalg.solve(gram, y_tensor)
    return coefficients.cpu().numpy()


def _torch_randomized_pca(
    centered: np.ndarray,
    *,
    component_count: int,
    random_seed: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device(device_name)
    values = torch.as_tensor(centered, dtype=torch.float64, device=device)
    maximum = min(values.shape)
    count = min(component_count, maximum)
    projection_count = min(maximum, count + 8)
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    projection = torch.randn(
        (values.shape[1], projection_count),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    basis, _ = torch.linalg.qr(values @ projection, mode="reduced")
    for _ in range(2):
        basis, _ = torch.linalg.qr(values @ (values.T @ basis), mode="reduced")
    _, singular_values, right = torch.linalg.svd(
        basis.T @ values,
        full_matrices=False,
    )
    axes = right[:count].cpu().numpy().copy()
    for axis in axes:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1
    total_variance = float(torch.sum(values**2).item())
    ratios = singular_values[:count].cpu().numpy() ** 2 / max(
        total_variance, 1e-12
    )
    return axes, ratios


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


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
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
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
