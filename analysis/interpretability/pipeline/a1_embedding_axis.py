"""Primary LLM2Vec informational-to-transactional prompt axis.

The axis is identified from matched endpoint reranking prompts over a frozen
query prior.  Query and surface wording are identical within every pair; only
the informational-versus-transactional search purpose changes.  Candidate
prompts are measured by projection onto this frozen direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .a1_prompt_manifold import (
    BUSINESS_ACTOR,
    NEUTRAL_SOURCE_CLAUSE,
    OUTPUT_CONTRACT,
)
from .prompt_continuum import TemplatePromptGenerator, _normalize_template


A1_EMBEDDING_AXIS_VERSION = "query-prior-llm2vec-a1-v1"
INFORMATIONAL_OBJECTIVE = (
    "Understand the category's mechanisms, use cases, practical applications, "
    "and limitations without evaluating or selecting a product."
)
TRANSACTIONAL_OBJECTIVE = (
    "Evaluate, compare, and shortlist suitable B2B SaaS solutions in order to "
    "select, acquire, or implement one now."
)


@dataclass(frozen=True, slots=True)
class QueryPriorA1Axis:
    axis_id: str
    axis_version: str
    embedding_model: str
    dimension: int
    direction: tuple[float, ...]
    informational_anchor: float
    transactional_anchor: float
    endpoint_pair_count: int
    query_count: int
    style_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class A1EndpointProjection:
    search_term: str
    style_seed: int
    informational_projection: float
    transactional_projection: float
    projection_gap: float
    informational_global_coordinate: float
    transactional_global_coordinate: float


@dataclass(frozen=True, slots=True)
class A1EmbeddingAxisDiagnostics:
    query_count: int
    style_count: int
    endpoint_pair_count: int
    embedding_dimension: int
    global_centroid_gap: float
    positive_pair_gap_rate: float
    minimum_pair_gap: float
    mean_pair_gap: float
    pair_gap_cv: float
    positive_query_mean_gap_rate: float
    minimum_query_mean_gap: float
    mean_informational_coordinate: float
    mean_transactional_coordinate: float


def build_query_prior_endpoint_prompts(
    search_terms: Sequence[str],
    *,
    style_seeds: Sequence[int] = (0, 1, 2, 3),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, int], ...]]:
    """Construct matched full-reranking endpoint prompts for the query prior."""

    queries = _normalize_queries(search_terms)
    styles = tuple(int(value) for value in style_seeds)
    if not styles or len(set(styles)) != len(styles):
        raise ValueError("style_seeds must be non-empty and unique")
    informational: list[str] = []
    transactional: list[str] = []
    pair_keys: list[tuple[str, int]] = []
    for query in queries:
        for style_seed in styles:
            informational.append(
                render_axis_prompt(
                    query=query,
                    style_seed=style_seed,
                    objective=INFORMATIONAL_OBJECTIVE,
                )
            )
            transactional.append(
                render_axis_prompt(
                    query=query,
                    style_seed=style_seed,
                    objective=TRANSACTIONAL_OBJECTIVE,
                )
            )
            pair_keys.append((query, style_seed))
    return tuple(informational), tuple(transactional), tuple(pair_keys)


def render_axis_prompt(*, query: str, style_seed: int, objective: str) -> str:
    """Render the complete prompt text measured in LLM2Vec space."""

    normalized_query = " ".join(str(query).split())
    if not normalized_query:
        raise ValueError("query must be non-empty")
    style = TemplatePromptGenerator._build_style_plan(style_seed)
    verb = style.ranking_verb.lower()
    task = (
        f"{style.ranking_verb} the supplied candidates for the exact search term by relevance."
        if style.syntax == "imperative"
        else f"Please {verb} the supplied candidates for the exact search term by relevance."
        if style.syntax == "request"
        else f"Your task is to {verb} the supplied candidates for the exact search term by relevance."
    )
    fixed = (
        f"Act as {BUSINESS_ACTOR}. Keep that actor, the search term, candidate set, "
        "ranking task, source policy, and output contract fixed."
    )
    semantic = f"Search objective: {objective}\nSource policy: {NEUTRAL_SOURCE_CLAUSE}"
    inputs = (
        f'Search term: "{normalized_query}"\n\n'
        "Candidates:\n[FROZEN CANDIDATE SET]"
    )
    blocks = (fixed, task, semantic, OUTPUT_CONTRACT.replace("{TOP_N}", "10"), inputs)
    if style.clause_order == "inputs_first":
        blocks = (inputs, fixed, task, semantic, OUTPUT_CONTRACT.replace("{TOP_N}", "10"))
    rendered = _normalize_template("\n\n".join(blocks))
    if rendered.count(normalized_query) != 1:
        raise ValueError("rendered endpoint did not retain the exact query once")
    return rendered


def fit_query_prior_a1_axis(
    informational_embeddings: np.ndarray,
    transactional_embeddings: np.ndarray,
    *,
    pair_keys: Sequence[tuple[str, int]],
    embedding_model: str,
) -> tuple[
    QueryPriorA1Axis,
    tuple[A1EndpointProjection, ...],
    A1EmbeddingAxisDiagnostics,
]:
    """Fit and orient the primary axis from paired normalized embeddings."""

    informational = _unit_rows(informational_embeddings)
    transactional = _unit_rows(transactional_embeddings)
    keys = tuple(pair_keys)
    if informational.shape != transactional.shape:
        raise ValueError("endpoint embedding shapes do not match")
    if informational.shape[0] != len(keys) or not keys:
        raise ValueError("pair_keys must match the non-empty endpoint matrices")
    if len(set(keys)) != len(keys):
        raise ValueError("endpoint pair keys must be unique")

    direction = np.mean(transactional - informational, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("endpoint prior does not identify a nonzero semantic axis")
    direction /= norm
    info_projection = informational @ direction
    trans_projection = transactional @ direction
    lower = float(np.mean(info_projection))
    upper = float(np.mean(trans_projection))
    scale = upper - lower
    if scale <= 1e-12:
        raise ValueError("transactional centroid does not follow informational centroid")

    identity = {
        "version": A1_EMBEDDING_AXIS_VERSION,
        "embedding_model": embedding_model,
        "pair_keys": keys,
        "direction_hash": hashlib.sha256(direction.astype("<f8").tobytes()).hexdigest(),
    }
    axis_hash = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    queries = tuple(dict.fromkeys(query for query, _ in keys))
    styles = tuple(sorted({style for _, style in keys}))
    axis = QueryPriorA1Axis(
        axis_id=f"a1-embedding-axis:{axis_hash[:24]}",
        axis_version=A1_EMBEDDING_AXIS_VERSION,
        embedding_model=embedding_model,
        dimension=int(direction.shape[0]),
        direction=tuple(float(value) for value in direction),
        informational_anchor=lower,
        transactional_anchor=upper,
        endpoint_pair_count=len(keys),
        query_count=len(queries),
        style_seeds=styles,
    )
    rows = tuple(
        A1EndpointProjection(
            search_term=query,
            style_seed=style,
            informational_projection=float(left),
            transactional_projection=float(right),
            projection_gap=float(right - left),
            informational_global_coordinate=float((left - lower) / scale),
            transactional_global_coordinate=float((right - lower) / scale),
        )
        for (query, style), left, right in zip(keys, info_projection, trans_projection)
    )
    gaps = trans_projection - info_projection
    query_gaps = np.asarray(
        [
            np.mean([row.projection_gap for row in rows if row.search_term == query])
            for query in queries
        ],
        dtype=np.float64,
    )
    diagnostics = A1EmbeddingAxisDiagnostics(
        query_count=len(queries),
        style_count=len(styles),
        endpoint_pair_count=len(keys),
        embedding_dimension=axis.dimension,
        global_centroid_gap=scale,
        positive_pair_gap_rate=float(np.mean(gaps > 0)),
        minimum_pair_gap=float(np.min(gaps)),
        mean_pair_gap=float(np.mean(gaps)),
        pair_gap_cv=float(np.std(gaps) / max(abs(np.mean(gaps)), 1e-12)),
        positive_query_mean_gap_rate=float(np.mean(query_gaps > 0)),
        minimum_query_mean_gap=float(np.min(query_gaps)),
        mean_informational_coordinate=float(np.mean((info_projection - lower) / scale)),
        mean_transactional_coordinate=float(np.mean((trans_projection - lower) / scale)),
    )
    return axis, rows, diagnostics


def project_onto_query_prior_a1(
    axis: QueryPriorA1Axis,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Return the primary global semantic A1 coordinate for prompt embeddings."""

    values = _unit_rows(embeddings)
    if values.shape[1] != axis.dimension:
        raise ValueError("embedding dimension does not match the A1 axis")
    direction = np.asarray(axis.direction, dtype=np.float64)
    scale = axis.transactional_anchor - axis.informational_anchor
    return (values @ direction - axis.informational_anchor) / scale


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings contain a zero-norm row")
    return array / norms


def _normalize_queries(values: Sequence[str]) -> tuple[str, ...]:
    queries = tuple(" ".join(str(value).split()) for value in values)
    if not queries or any(not value for value in queries):
        raise ValueError("search terms must be non-empty")
    if len({value.casefold() for value in queries}) != len(queries):
        raise ValueError("search terms must be unique after normalization")
    return queries
