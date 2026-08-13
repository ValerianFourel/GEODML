"""Position query-bound reranking prompts on a frozen LLM2Vec A1 vector."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .a1_embedding_axis import A1EndpointProjection, QueryPriorA1Axis
from .a1_prompt_manifold import A1Candidate


A1_EMBEDDING_PANEL_VERSION = "llm2vec-positioned-a1-panel-v2"


@dataclass(frozen=True, slots=True)
class A1CandidateCoordinate:
    candidate_id: str
    candidate_hash: str
    generator_assigned_a1: float
    candidate_index: int
    global_a1: float
    observed_a1: float


@dataclass(frozen=True, slots=True)
class PositionedA1Prompt:
    schedule_order: int
    keyword_order: int
    within_keyword_order: int
    axis_order: int
    query_id: str
    search_term: str
    style_seed: int
    target_a1: float
    observed_a1: float
    global_a1: float
    absolute_target_error: float
    source_candidate_id: str
    source_candidate_hash: str
    source_generator_assigned_a1: float
    source_candidate_index: int
    search_objective_clause: str
    query_bound_prompt_template: str
    measurement_prompt_hash: str
    axis_id: str
    panel_assignment_id: str


def balanced_query_style_assignment(
    search_terms: Sequence[str],
    style_seeds: Sequence[int],
    *,
    master_seed: int,
) -> tuple[tuple[str, int, int], ...]:
    """Assign one style per query with seeded random order and global balance."""

    queries = _queries(search_terms)
    styles = tuple(int(value) for value in style_seeds)
    if not styles or len(set(styles)) != len(styles):
        raise ValueError("style_seeds must be non-empty and unique")
    query_order = sorted(
        queries,
        key=lambda query: _random_key(master_seed, "query-order", query),
    )
    style_cycle = sorted(
        styles,
        key=lambda style: _random_key(master_seed, "style-cycle", style),
    )
    return tuple(
        (query, style_cycle[index % len(style_cycle)], index + 1)
        for index, query in enumerate(query_order)
    )


def render_candidate_for_measurement(candidate: A1Candidate, query: str) -> str:
    """Bind the query and deterministic placeholders exactly as axis endpoints."""

    normalized = " ".join(str(query).split())
    if not normalized:
        raise ValueError("query must be non-empty")
    template = candidate.prompt_template
    if template.count("{QUERY}") != 1:
        raise ValueError("candidate template must contain one query placeholder")
    rendered = (
        template.replace("{QUERY}", normalized)
        .replace("{CANDIDATES}", "[FROZEN CANDIDATE SET]")
        .replace("{TOP_N}", "10")
    )
    if "{QUERY}" in rendered or normalized not in rendered:
        raise ValueError("measurement prompt did not bind the exact query")
    if "{CANDIDATES}" in rendered or "{TOP_N}" in rendered:
        raise ValueError("measurement prompt retained an unbound placeholder")
    return rendered


def deduplicate_candidates_by_hash(
    candidates: Sequence[A1Candidate],
) -> tuple[A1Candidate, ...]:
    """Return one stable representative for each identical prompt template."""

    items = tuple(candidates)
    if not items:
        raise ValueError("candidate pool must be non-empty")
    styles = {candidate.style_seed for candidate in items}
    if len(styles) != 1:
        raise ValueError("candidate deduplication requires one surface style")
    representatives: dict[str, A1Candidate] = {}
    for candidate in sorted(items, key=lambda item: item.candidate_id):
        representatives.setdefault(candidate.candidate_hash, candidate)
    return tuple(
        sorted(representatives.values(), key=lambda candidate: candidate.candidate_id)
    )


def measure_candidate_coordinates(
    *,
    axis: QueryPriorA1Axis,
    endpoint: A1EndpointProjection,
    candidates: Sequence[A1Candidate],
    embeddings: np.ndarray,
) -> tuple[A1CandidateCoordinate, ...]:
    """Project candidates globally and against their matched query/style anchors."""

    items = tuple(candidates)
    values = _unit_rows(embeddings)
    if len(items) != len(values):
        raise ValueError("candidate and embedding counts do not match")
    if values.shape[1] != axis.dimension:
        raise ValueError("candidate embedding dimension does not match the axis")
    if any(candidate.style_seed != endpoint.style_seed for candidate in items):
        raise ValueError("candidate style does not match the endpoint anchors")
    pair_scale = endpoint.transactional_projection - endpoint.informational_projection
    if pair_scale <= 1e-12:
        raise ValueError("matched endpoint anchors have non-positive separation")
    direction = np.asarray(axis.direction, dtype=np.float64)
    raw = values @ direction
    global_scale = axis.transactional_anchor - axis.informational_anchor
    if global_scale <= 1e-12:
        raise ValueError("global endpoint anchors have non-positive separation")
    global_coordinates = (raw - axis.informational_anchor) / global_scale
    matched_coordinates = (raw - endpoint.informational_projection) / pair_scale
    return tuple(
        A1CandidateCoordinate(
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.candidate_hash,
            generator_assigned_a1=candidate.assigned_a1,
            candidate_index=candidate.candidate_index,
            global_a1=float(global_coordinate),
            observed_a1=float(matched_coordinate),
        )
        for candidate, global_coordinate, matched_coordinate in zip(
            items,
            global_coordinates,
            matched_coordinates,
        )
    )


def select_embedding_trajectory(
    coordinates: Sequence[A1CandidateCoordinate],
    targets: Sequence[float],
) -> tuple[A1CandidateCoordinate, ...]:
    """Select a unique strictly ordered candidate nearest each target.

    This is an exact minimum-cost increasing-subsequence dynamic program. It
    uses only embedding-measured coordinates; generator labels are ignored.
    """

    desired = tuple(float(value) for value in targets)
    if not desired or tuple(sorted(set(desired))) != desired:
        raise ValueError("targets must be non-empty, unique, and increasing")
    if len({item.candidate_hash for item in coordinates}) != len(coordinates):
        raise ValueError("coordinates contain duplicate candidate hashes")
    ordered = sorted(
        coordinates,
        key=lambda item: (item.observed_a1, item.candidate_hash),
    )
    unique: list[A1CandidateCoordinate] = []
    for item in ordered:
        if unique and item.observed_a1 <= unique[-1].observed_a1:
            continue
        unique.append(item)
    if len(unique) < len(desired):
        raise ValueError("too few distinct measured coordinates for the target grid")

    count = len(unique)
    levels = len(desired)
    infinity = float("inf")
    cost = np.full((levels + 1, count + 1), infinity, dtype=np.float64)
    take = np.zeros((levels + 1, count + 1), dtype=np.bool_)
    cost[0, :] = 0.0
    for level in range(1, levels + 1):
        for position in range(1, count + 1):
            skip_cost = cost[level, position - 1]
            take_cost = cost[level - 1, position - 1] + (
                unique[position - 1].observed_a1 - desired[level - 1]
            ) ** 2
            if take_cost < skip_cost:
                cost[level, position] = take_cost
                take[level, position] = True
            else:
                cost[level, position] = skip_cost
    if not np.isfinite(cost[levels, count]):
        raise ValueError("no embedding trajectory covers the target grid")

    selected: list[A1CandidateCoordinate] = []
    level, position = levels, count
    while level:
        if take[level, position]:
            selected.append(unique[position - 1])
            level -= 1
            position -= 1
        else:
            position -= 1
            if position < level:
                raise RuntimeError("invalid embedding trajectory backtrack")
    selected.reverse()
    return tuple(selected)


def build_positioned_rows(
    *,
    search_term: str,
    style_seed: int,
    keyword_order: int,
    targets: Sequence[float],
    selected_coordinates: Sequence[A1CandidateCoordinate],
    candidates_by_id: Mapping[str, A1Candidate],
    axis_id: str,
) -> tuple[PositionedA1Prompt, ...]:
    """Create final query-bound panel rows from an embedding-selected path."""

    targets_tuple = tuple(float(value) for value in targets)
    selected = tuple(selected_coordinates)
    if len(targets_tuple) != len(selected):
        raise ValueError("selected trajectory must match the target count")
    query_id = f"query:{_hash(search_term)[:24]}"
    rows = []
    for within_order, (target, coordinate) in enumerate(
        zip(targets_tuple, selected),
        start=1,
    ):
        candidate = candidates_by_id[coordinate.candidate_id]
        query_template = candidate.prompt_template.replace("{QUERY}", search_term)
        measurement = render_candidate_for_measurement(candidate, search_term)
        identity = json.dumps(
            {
                "version": A1_EMBEDDING_PANEL_VERSION,
                "axis_id": axis_id,
                "query": search_term,
                "target": f"{target:.17g}",
                "candidate_id": candidate.candidate_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(
            PositionedA1Prompt(
                schedule_order=0,
                keyword_order=keyword_order,
                within_keyword_order=0,
                axis_order=within_order,
                query_id=query_id,
                search_term=search_term,
                style_seed=style_seed,
                target_a1=target,
                observed_a1=coordinate.observed_a1,
                global_a1=coordinate.global_a1,
                absolute_target_error=abs(coordinate.observed_a1 - target),
                source_candidate_id=candidate.candidate_id,
                source_candidate_hash=candidate.candidate_hash,
                source_generator_assigned_a1=candidate.assigned_a1,
                source_candidate_index=candidate.candidate_index,
                search_objective_clause=candidate.search_objective_clause,
                query_bound_prompt_template=query_template,
                measurement_prompt_hash=_hash(measurement),
                axis_id=axis_id,
                panel_assignment_id=f"a1-positioned:{_hash(identity)[:24]}",
            )
        )
    return tuple(rows)


def randomize_positioned_schedule(
    rows: Sequence[PositionedA1Prompt],
    *,
    master_seed: int,
) -> tuple[PositionedA1Prompt, ...]:
    """Randomize treatment order within each already-randomized query block."""

    grouped: dict[int, list[PositionedA1Prompt]] = {}
    for row in rows:
        grouped.setdefault(row.keyword_order, []).append(row)
    scheduled: list[PositionedA1Prompt] = []
    for keyword_order in sorted(grouped):
        group = grouped[keyword_order]
        if len({row.search_term for row in group}) != 1:
            raise ValueError("keyword_order contains multiple search terms")
        randomized = sorted(
            group,
            key=lambda row: _random_key(
                master_seed,
                "within-query-order",
                row.search_term,
                row.panel_assignment_id,
            ),
        )
        for within_order, row in enumerate(randomized, start=1):
            scheduled.append(
                replace(
                    row,
                    schedule_order=len(scheduled) + 1,
                    within_keyword_order=within_order,
                )
            )
    return tuple(scheduled)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings contain zero-norm rows")
    return array / norms


def _queries(values: Sequence[str]) -> tuple[str, ...]:
    queries = tuple(" ".join(str(value).split()) for value in values)
    if not queries or any(not query for query in queries):
        raise ValueError("search terms must be non-empty")
    if len({query.casefold() for query in queries}) != len(queries):
        raise ValueError("search terms must be unique after normalization")
    return queries


def _random_key(seed: int, *parts: object) -> str:
    return _hash(
        json.dumps(
            {"version": A1_EMBEDDING_PANEL_VERSION, "seed": seed, "parts": parts},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
