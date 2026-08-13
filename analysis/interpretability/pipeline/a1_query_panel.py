"""Query-conditioned randomized panel for a frozen semantic A1 manifold.

Every search term is a complete block.  Each block receives every selected A1
prompt, so the assigned semantic coordinate remains comparable across queries.
The query is inserted structurally before reranking, while ``{CANDIDATES}`` and
``{TOP_N}`` remain unresolved for the downstream ranking stage.

Randomization changes execution order only.  It never changes panel membership,
the A1 coordinate, the surface-realization seed, or the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Sequence

from .a1_prompt_manifold import SelectedA1Prompt


A1_QUERY_PANEL_VERSION = "a1-query-panel-v2"
_RESERVED_PLACEHOLDERS = ("{QUERY}", "{CANDIDATES}", "{TOP_N}")


@dataclass(frozen=True, slots=True)
class QueryConditionedA1Prompt:
    """One scheduled A1 prompt assignment for one query block."""

    schedule_order: int
    keyword_order: int
    within_keyword_order: int
    query_id: str
    search_term: str
    panel_assignment_id: str
    source_prompt_assignment_id: str
    source_candidate_id: str
    source_candidate_hash: str
    assigned_a1: float
    realized_a1: float
    style_seed: int
    candidate_index: int
    search_objective_clause: str
    query_bound_prompt_template: str
    query_bound_prompt_hash: str
    source_input_embedding_hash: str
    source_response_embedding_hash: str
    randomization_key: str


@dataclass(frozen=True, slots=True)
class A1QueryPanelDiagnostics:
    design: str
    style_assignment: str
    query_count: int
    prompts_per_query: int
    assignment_count: int
    a1_levels: tuple[float, ...]
    style_seeds: tuple[int, ...]
    exact_query_binding_rate: float
    a1_level_coverage_rate: float
    complete_block_rate: float
    maximum_within_query_style_imbalance: int
    duplicate_assignment_count: int
    duplicate_query_bound_prompt_count: int


def build_query_conditioned_a1_panel(
    *,
    search_terms: Sequence[str],
    selected_prompts: Sequence[SelectedA1Prompt],
    master_seed: int = 20260817,
    style_assignment: str = "full-cross",
) -> tuple[tuple[QueryConditionedA1Prompt, ...], A1QueryPanelDiagnostics]:
    """Bind every query to a complete A1 block in randomized execution order.

    ``full-cross`` includes every A1/style cell. ``balanced-one-per-a1`` keeps
    every A1 level exactly once per query and chooses styles with a seeded,
    balanced random cycle.
    """

    if style_assignment not in {"full-cross", "balanced-one-per-a1"}:
        raise ValueError(
            "style_assignment must be 'full-cross' or 'balanced-one-per-a1'"
        )
    prompts, a1_levels, style_seeds = _validate_selected_prompts(selected_prompts)
    queries = _normalize_search_terms(search_terms)

    query_order = sorted(
        queries,
        key=lambda query: _randomization_key(master_seed, "query", query),
    )
    rows: list[QueryConditionedA1Prompt] = []
    for keyword_order, query in enumerate(query_order, start=1):
        query_id = f"query:{_hash_text(query)[:24]}"
        block_prompts = _query_block_prompts(
            prompts,
            a1_levels=a1_levels,
            style_seeds=style_seeds,
            query=query,
            master_seed=master_seed,
            style_assignment=style_assignment,
        )
        prompt_order = sorted(
            block_prompts,
            key=lambda prompt: _randomization_key(
                master_seed,
                "within-query",
                query,
                prompt.prompt_assignment_id,
            ),
        )
        for within_keyword_order, prompt in enumerate(prompt_order, start=1):
            rendered = prompt.prompt_template.replace("{QUERY}", query)
            if "{QUERY}" in rendered or query not in rendered:
                raise ValueError(f"failed to bind search term structurally: {query!r}")
            for placeholder in ("{CANDIDATES}", "{TOP_N}"):
                if rendered.count(placeholder) != 1:
                    raise ValueError(
                        f"query binding changed placeholder contract for {query!r}: "
                        f"{placeholder}"
                    )
            identity = _canonical_json(
                {
                    "version": A1_QUERY_PANEL_VERSION,
                    "query": query,
                    "source_prompt_assignment_id": prompt.prompt_assignment_id,
                }
            )
            randomization_key = _randomization_key(
                master_seed,
                "within-query",
                query,
                prompt.prompt_assignment_id,
            )
            rows.append(
                QueryConditionedA1Prompt(
                    schedule_order=len(rows) + 1,
                    keyword_order=keyword_order,
                    within_keyword_order=within_keyword_order,
                    query_id=query_id,
                    search_term=query,
                    panel_assignment_id=f"a1-query-assignment:{_hash_text(identity)[:24]}",
                    source_prompt_assignment_id=prompt.prompt_assignment_id,
                    source_candidate_id=prompt.candidate_id,
                    source_candidate_hash=prompt.candidate_hash,
                    assigned_a1=prompt.assigned_a1,
                    realized_a1=prompt.realized_a1,
                    style_seed=prompt.style_seed,
                    candidate_index=prompt.candidate_index,
                    search_objective_clause=prompt.search_objective_clause,
                    query_bound_prompt_template=rendered,
                    query_bound_prompt_hash=_hash_text(rendered),
                    source_input_embedding_hash=prompt.input_embedding_hash,
                    source_response_embedding_hash=prompt.response_embedding_hash,
                    randomization_key=randomization_key,
                )
            )

    complete_blocks = 0
    covered_levels = 0
    maximum_style_imbalance = 0
    rows_by_query: dict[str, list[QueryConditionedA1Prompt]] = {}
    for row in rows:
        rows_by_query.setdefault(row.search_term, []).append(row)
    for query in queries:
        block = rows_by_query[query]
        observed_levels = {row.assigned_a1 for row in block}
        covered_levels += len(observed_levels)
        style_counts = [
            sum(row.style_seed == style_seed for row in block)
            for style_seed in style_seeds
        ]
        style_imbalance = max(style_counts) - min(style_counts)
        maximum_style_imbalance = max(maximum_style_imbalance, style_imbalance)
        observed = {
            (row.source_prompt_assignment_id, row.assigned_a1, row.style_seed)
            for row in block
        }
        if style_assignment == "full-cross":
            expected = {
                (prompt.prompt_assignment_id, prompt.assigned_a1, prompt.style_seed)
                for prompt in prompts
            }
            complete_blocks += int(observed == expected)
        else:
            complete_blocks += int(
                len(block) == len(a1_levels)
                and observed_levels == set(a1_levels)
                and style_imbalance <= 1
            )

    assignment_ids = [row.panel_assignment_id for row in rows]
    prompt_hashes = [row.query_bound_prompt_hash for row in rows]
    diagnostics = A1QueryPanelDiagnostics(
        design=(
            "randomized-complete-block"
            if style_assignment == "full-cross"
            else "randomized-complete-a1-block"
        ),
        style_assignment=style_assignment,
        query_count=len(queries),
        prompts_per_query=(
            len(prompts) if style_assignment == "full-cross" else len(a1_levels)
        ),
        assignment_count=len(rows),
        a1_levels=a1_levels,
        style_seeds=style_seeds,
        exact_query_binding_rate=(
            sum(row.search_term in row.query_bound_prompt_template for row in rows)
            / len(rows)
        ),
        a1_level_coverage_rate=covered_levels / (len(queries) * len(a1_levels)),
        complete_block_rate=complete_blocks / len(queries),
        maximum_within_query_style_imbalance=maximum_style_imbalance,
        duplicate_assignment_count=len(rows) - len(set(assignment_ids)),
        duplicate_query_bound_prompt_count=len(rows) - len(set(prompt_hashes)),
    )
    return tuple(rows), diagnostics


def _query_block_prompts(
    prompts: Sequence[SelectedA1Prompt],
    *,
    a1_levels: Sequence[float],
    style_seeds: Sequence[int],
    query: str,
    master_seed: int,
    style_assignment: str,
) -> tuple[SelectedA1Prompt, ...]:
    if style_assignment == "full-cross":
        return tuple(prompts)

    by_cell = {
        (prompt.assigned_a1, prompt.style_seed): prompt for prompt in prompts
    }
    randomized_levels = sorted(
        a1_levels,
        key=lambda level: _randomization_key(
            master_seed,
            "style-level-order",
            query,
            f"{level:.17g}",
        ),
    )
    randomized_styles = sorted(
        style_seeds,
        key=lambda style_seed: _randomization_key(
            master_seed,
            "style-cycle",
            query,
            style_seed,
        ),
    )
    selected = [
        by_cell[(level, randomized_styles[index % len(randomized_styles)])]
        for index, level in enumerate(randomized_levels)
    ]
    return tuple(selected)


def _validate_selected_prompts(
    selected_prompts: Sequence[SelectedA1Prompt],
) -> tuple[tuple[SelectedA1Prompt, ...], tuple[float, ...], tuple[int, ...]]:
    prompts = tuple(selected_prompts)
    if not prompts:
        raise ValueError("selected A1 manifold must not be empty")
    if len({item.prompt_assignment_id for item in prompts}) != len(prompts):
        raise ValueError("selected A1 prompt_assignment_id values must be unique")
    if len({item.candidate_hash for item in prompts}) != len(prompts):
        raise ValueError("selected A1 candidate hashes must be unique")

    styles = tuple(sorted({item.style_seed for item in prompts}))
    reference_grid: tuple[float, ...] | None = None
    for style_seed in styles:
        group = sorted(
            (item for item in prompts if item.style_seed == style_seed),
            key=lambda item: item.assigned_a1,
        )
        grid = tuple(item.assigned_a1 for item in group)
        if len(set(grid)) != len(grid):
            raise ValueError(f"duplicate A1 cell for style_seed={style_seed}")
        if grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("every style trajectory must include A1 endpoints 0 and 1")
        if any(right.realized_a1 <= left.realized_a1 for left, right in zip(group, group[1:])):
            raise ValueError(
                f"selected realized A1 trajectory is not strictly increasing for "
                f"style_seed={style_seed}"
            )
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError("every style trajectory must use the same assigned A1 grid")

    for prompt in prompts:
        for placeholder in _RESERVED_PLACEHOLDERS:
            if prompt.prompt_template.count(placeholder) != 1:
                raise ValueError(
                    f"source prompt {prompt.prompt_assignment_id} must contain exactly "
                    f"one {placeholder} placeholder"
                )
    assert reference_grid is not None
    return prompts, reference_grid, styles


def _normalize_search_terms(search_terms: Sequence[str]) -> tuple[str, ...]:
    queries: list[str] = []
    seen: set[str] = set()
    for raw in search_terms:
        query = re.sub(r"\s+", " ", str(raw)).strip()
        if not query:
            raise ValueError("search terms must be non-empty")
        if any(placeholder in query for placeholder in _RESERVED_PLACEHOLDERS):
            raise ValueError(f"search term contains a reserved placeholder: {query!r}")
        folded = query.casefold()
        if folded in seen:
            raise ValueError(f"duplicate search term after normalization: {query!r}")
        seen.add(folded)
        queries.append(query)
    if not queries:
        raise ValueError("at least one search term is required")
    return tuple(queries)


def _randomization_key(master_seed: int, *parts: object) -> str:
    return _hash_text(
        _canonical_json(
            {
                "version": A1_QUERY_PANEL_VERSION,
                "master_seed": int(master_seed),
                "parts": parts,
            }
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
