"""Calibrated two-axis population of natural-language reranking prompts.

The scientific treatment is the assigned pair ``(A1, A2)``.  Candidate prompt
texts are generated first, measured independently through blind pairwise
comparisons, embedded for geometric diagnostics, and selected jointly under
monotonicity and no-reuse constraints.  Embeddings and realized coordinates do
not redefine assignment, and no latent vector is decoded into text.

This module contains provider protocols and deterministic fake implementations
for CPU contract tests.  Fake outputs support no scientific claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import csr_matrix

from .prompt_continuum import StylePlan, TemplatePromptGenerator, _normalize_template
from .search_purpose_continuum import (
    RankingPermutation,
    RenderedSearchPurposePrompt,
    SearchCandidate,
    SearchPurposePromptRecord,
    parse_ranking_permutation,
    render_search_purpose_prompt,
)

POPULATION_VERSION = "two-axis-prompt-population-v1"
DEFAULT_AXIS_GRID = tuple(step / 6.0 for step in range(7))
_SPECIFICATION_PATH = (
    Path(__file__).with_name("specs") / "two_axis_prompt_population_v1.json"
)
_FORBIDDEN_PATTERNS = {
    "freshness": r"\bfresh(?:ness)?\b|\brecen(?:t|cy)\b",
    "authority": r"\bauthorit(?:y|ative)\b|\bcredib(?:le|ility)\b",
    "popularity": r"\bpopular(?:ity)?\b|\bbrand prestige\b|\bbrand fame\b",
    "cost": r"\bprice\b|\bpricing\b|\bcost\b|\bbudget\b",
    "company-size": r"\bcompany size\b|\bsmall business\b|\benterprise size\b",
    "geography": r"\bgeograph(?:y|ic)\b|\bregion\b|\bcountry\b",
    "page-length": r"\bpage length\b",
    "statistics-density": r"\bstatistics? density\b|\bnumerical density\b",
    "writing-quality": r"\bwriting quality\b",
    "review-score": r"\breview scores?\b|\bratings?\b",
    "hard-exclusion": r"\bexclude\b|\bnever rank\b|\bonly rank\b",
}

__all__ = [
    "DEFAULT_AXIS_GRID",
    "FakePairwiseJudge",
    "FakeTwoAxisCandidateGenerator",
    "FakeTwoAxisPromptEmbedder",
    "LLM2VecGenPromptEmbedder",
    "LLM2VecPromptEmbedder",
    "LocalLLMPairwiseJudge",
    "LocalLLMTwoAxisCandidateGenerator",
    "PairwiseComparisonRequest",
    "PairwiseJudgment",
    "PairwiseJudge",
    "PromptCandidateGenerator",
    "SelectedTwoAxisPrompt",
    "TwoAxisCandidate",
    "TwoAxisCandidateRequest",
    "TwoAxisCalibration",
    "TwoAxisPermutationOutcome",
    "TwoAxisLatentDiagnostics",
    "TwoAxisPromptEmbedder",
    "TwoAxisSelectionDiagnostics",
    "build_pairwise_comparison_requests",
    "calibrate_candidates",
    "generate_candidate_bank",
    "judge_comparison_requests",
    "load_population_specification",
    "map_two_axis_prompt_to_permutation",
    "measure_selected_latent_population",
    "render_selected_two_axis_prompt",
    "select_prompt_population",
    "semantic_contract_checks",
]


class PromptCandidateGenerator(Protocol):
    backend_name: str
    model_name: str

    def generate(self, request: "TwoAxisCandidateRequest") -> tuple[tuple[str, str], ...]:
        """Return objective/source clause pairs for one target and style."""


class PairwiseJudge(Protocol):
    judge_id: str

    def compare(
        self,
        request: "PairwiseComparisonRequest",
        candidates: Mapping[str, "TwoAxisCandidate"],
    ) -> str:
        """Return the winning candidate ID or ``tie``."""


class TwoAxisPromptEmbedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one finite embedding row per prompt template."""


@dataclass(frozen=True, slots=True)
class TwoAxisCandidateRequest:
    assigned_a1: float
    assigned_a2: float
    style_seed: int
    generation_seed: int
    number_candidates: int
    generator_model: str
    population_version: str = POPULATION_VERSION

    def __post_init__(self) -> None:
        _coordinate("assigned_a1", self.assigned_a1)
        _coordinate("assigned_a2", self.assigned_a2)
        for name, value in (
            ("style_seed", self.style_seed),
            ("generation_seed", self.generation_seed),
            ("number_candidates", self.number_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.number_candidates <= 0:
            raise ValueError("number_candidates must be positive")
        if not self.generator_model.strip():
            raise ValueError("generator_model must be non-empty")
        if self.population_version != POPULATION_VERSION:
            raise ValueError("unsupported population version")


@dataclass(frozen=True, slots=True)
class TwoAxisCandidate:
    candidate_id: str
    candidate_hash: str
    assigned_a1: float
    assigned_a2: float
    style_seed: int
    candidate_index: int
    generation_seed: int
    search_term: str
    business_actor: str
    search_objective_clause: str
    source_preference_clause: str
    output_contract: str
    style_plan: StylePlan
    prompt_template: str
    generator_backend: str
    generator_model: str
    population_version: str
    structural_valid: bool
    contract_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairwiseComparisonRequest:
    comparison_id: str
    axis: str
    style_seed: int
    fixed_coordinate: float
    left_candidate_id: str
    right_candidate_id: str
    presentation_order: str
    comparison_kind: str
    question: str


@dataclass(frozen=True, slots=True)
class PairwiseJudgment:
    comparison_id: str
    judge_id: str
    winner_candidate_id: str | None
    is_tie: bool


@dataclass(frozen=True, slots=True)
class TwoAxisCalibration:
    candidate_id: str
    realized_a1: float
    realized_a2: float
    a1_comparison_count: int
    a2_comparison_count: int


@dataclass(frozen=True, slots=True)
class SelectedTwoAxisPrompt:
    prompt_assignment_id: str
    candidate_id: str
    candidate_hash: str
    assigned_a1: float
    assigned_a2: float
    realized_a1: float
    realized_a2: float
    style_seed: int
    candidate_index: int
    search_term: str
    business_actor: str
    search_objective_clause: str
    source_preference_clause: str
    output_contract: str
    prompt_template: str
    embedding_model: str
    prompt_embedding: tuple[float, ...]
    embedding_hash: str
    calibration_cost: float


@dataclass(frozen=True, slots=True)
class TwoAxisSelectionDiagnostics:
    selected_count: int
    style_count: int
    cells_per_style: int
    duplicate_hash_count: int
    a1_adjacent_reversal_rate: float
    a2_adjacent_reversal_rate: float
    a1_adjacent_tie_rate: float
    a2_adjacent_tie_rate: float
    fully_monotone_style_rate: float
    maximum_neighbor_embedding_distance: float
    mean_calibration_l1_error: float


@dataclass(frozen=True, slots=True)
class TwoAxisPermutationOutcome:
    outcome_id: str
    prompt_assignment_id: str
    assigned_a1: float
    assigned_a2: float
    realized_a1: float
    realized_a2: float
    style_seed: int
    prompt_instance_id: str
    candidate_set_id: str
    prompt_embedding_hash: str
    ranking: RankingPermutation
    raw_model_output: str
    reranker_run_id: str
    reranker_model: str


@dataclass(frozen=True, slots=True)
class TwoAxisLatentDiagnostics:
    embedding_model: str
    selected_count: int
    embedding_dimension: int
    exact_query_structural_retention_rate: float
    a1_endpoint_distance: float
    a2_endpoint_distance: float
    a1_a2_direction_cosine: float
    a1_slice_spearman_mean: float
    a2_slice_spearman_mean: float
    a1_cross_axis_slope_ratio: float
    a2_cross_axis_slope_ratio: float
    mean_adjacent_embedding_distance: float
    mean_distant_embedding_distance: float
    adjacent_over_distant_distance_ratio: float


def load_population_specification() -> dict[str, object]:
    payload = json.loads(_SPECIFICATION_PATH.read_text(encoding="utf-8"))
    if payload.get("specification_version") != POPULATION_VERSION:
        raise ValueError("population specification file/version mismatch")
    return payload


def generate_candidate_bank(
    *,
    search_term: str,
    a1_grid: Sequence[float] = DEFAULT_AXIS_GRID,
    a2_grid: Sequence[float] = DEFAULT_AXIS_GRID,
    style_seeds: Sequence[int] = tuple(range(24)),
    number_candidates: int = 6,
    master_seed: int = 20260812,
    generator: PromptCandidateGenerator,
) -> tuple[TwoAxisCandidate, ...]:
    """Generate complete style trajectories with the exact query kept structural."""

    load_population_specification()
    query = _single_line(search_term)
    if not query:
        raise ValueError("search_term must be non-empty")
    grid1 = _grid("a1_grid", a1_grid)
    grid2 = _grid("a2_grid", a2_grid)
    if not style_seeds or len(set(style_seeds)) != len(style_seeds):
        raise ValueError("style_seeds must be non-empty and unique")
    business_actor = (
        "a business software evaluator assessing a B2B SaaS category for an organization"
    )
    output_contract = (
        "Return exactly {TOP_N} candidate identifiers only, with no explanation."
    )
    rows: list[TwoAxisCandidate] = []
    for style_seed in style_seeds:
        style = TemplatePromptGenerator._build_style_plan(style_seed)
        for a1 in grid1:
            for a2 in grid2:
                seed = _generation_seed(master_seed, style_seed, a1, a2)
                request = TwoAxisCandidateRequest(
                    assigned_a1=a1,
                    assigned_a2=a2,
                    style_seed=style_seed,
                    generation_seed=seed,
                    number_candidates=number_candidates,
                    generator_model=generator.model_name,
                )
                clauses = generator.generate(request)
                if len(clauses) != number_candidates:
                    raise ValueError(
                        f"generator returned {len(clauses)} candidates; expected {number_candidates}"
                    )
                for candidate_index, (objective, source) in enumerate(clauses):
                    template = _compile_prompt(
                        style=style,
                        business_actor=business_actor,
                        objective_clause=_single_line(objective),
                        source_clause=_single_line(source),
                        output_contract=output_contract,
                    )
                    failures = semantic_contract_checks(
                        template,
                        search_term=query,
                        business_actor=business_actor,
                        objective_clause=objective,
                        source_preference_clause=source,
                    )
                    candidate_hash = _hash(template)
                    identity = {
                        "version": POPULATION_VERSION,
                        "a1": f"{a1:.12g}",
                        "a2": f"{a2:.12g}",
                        "style_seed": style_seed,
                        "candidate_index": candidate_index,
                        "generation_seed": seed,
                        "candidate_hash": candidate_hash,
                    }
                    identity_hash = _hash(
                        json.dumps(identity, sort_keys=True, separators=(",", ":"))
                    )
                    rows.append(
                        TwoAxisCandidate(
                            candidate_id=f"two-axis-candidate:{identity_hash[:24]}",
                            candidate_hash=candidate_hash,
                            assigned_a1=a1,
                            assigned_a2=a2,
                            style_seed=style_seed,
                            candidate_index=candidate_index,
                            generation_seed=seed,
                            search_term=query,
                            business_actor=business_actor,
                            search_objective_clause=_single_line(objective),
                            source_preference_clause=_single_line(source),
                            output_contract=output_contract,
                            style_plan=style,
                            prompt_template=template,
                            generator_backend=generator.backend_name,
                            generator_model=generator.model_name,
                            population_version=POPULATION_VERSION,
                            structural_valid=not failures,
                            contract_failures=failures,
                        )
                    )
    return tuple(rows)


def semantic_contract_checks(
    prompt_template: str,
    *,
    search_term: str,
    business_actor: str,
    objective_clause: str,
    source_preference_clause: str,
) -> tuple[str, ...]:
    """Apply hard structural/lexical screens before judging or selection."""

    lowered = prompt_template.casefold()
    failures: list[str] = []
    for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
        if placeholder not in prompt_template:
            failures.append(f"missing-placeholder:{placeholder}")
    if search_term.casefold() in lowered:
        failures.append("query-leaked-into-generated-template")
    if business_actor.casefold() not in lowered:
        failures.append("business-actor-missing")
    if not objective_clause.strip() or not source_preference_clause.strip():
        failures.append("empty-semantic-clause")
    if "candidate identifiers only" not in lowered or "no explanation" not in lowered:
        failures.append("invalid-output-contract")
    if re.search(r"(?<![A-Za-z])A[12](?![A-Za-z])", prompt_template):
        failures.append("coordinate-symbol-exposed")
    if re.search(r"\b(?:0(?:\.\d+)?|1\.0+)\b", objective_clause + " " + source_preference_clause):
        failures.append("numeric-coordinate-exposed")
    for label, pattern in _FORBIDDEN_PATTERNS.items():
        if re.search(pattern, lowered):
            failures.append(f"off-axis:{label}")
    return tuple(failures)


def build_pairwise_comparison_requests(
    candidates: Sequence[TwoAxisCandidate],
    *,
    include_distant_pairs: bool = True,
) -> tuple[PairwiseComparisonRequest, ...]:
    """Build blind adjacent/distant comparisons in both presentation orders."""

    valid = [candidate for candidate in candidates if candidate.structural_valid]
    by_style_cell: dict[tuple[int, float, float], list[TwoAxisCandidate]] = {}
    for candidate in valid:
        by_style_cell.setdefault(
            (candidate.style_seed, candidate.assigned_a1, candidate.assigned_a2), []
        ).append(candidate)
    requests: list[PairwiseComparisonRequest] = []
    style_seeds = sorted({candidate.style_seed for candidate in valid})
    a1_grid = sorted({candidate.assigned_a1 for candidate in valid})
    a2_grid = sorted({candidate.assigned_a2 for candidate in valid})
    for axis in ("A1", "A2"):
        axis_grid = a1_grid if axis == "A1" else a2_grid
        fixed_grid = a2_grid if axis == "A1" else a1_grid
        for style_seed in style_seeds:
            for fixed in fixed_grid:
                cells = [
                    by_style_cell[(style_seed, level, fixed)]
                    if axis == "A1"
                    else by_style_cell[(style_seed, fixed, level)]
                    for level in axis_grid
                ]
                for cell in cells:
                    reference = sorted(cell, key=lambda item: item.candidate_index)[0]
                    for other in sorted(cell, key=lambda item: item.candidate_index)[1:]:
                        requests.extend(
                            _comparison_pair(axis, style_seed, fixed, reference, other, "within-cell")
                        )
                pairs = list(zip(cells, cells[1:]))
                if include_distant_pairs and len(cells) > 2:
                    pairs.append((cells[0], cells[-1]))
                for left_cell, right_cell in pairs:
                    left_by_index = {item.candidate_index: item for item in left_cell}
                    right_by_index = {item.candidate_index: item for item in right_cell}
                    for index in sorted(set(left_by_index) & set(right_by_index)):
                        kind = "distant" if left_cell is cells[0] and right_cell is cells[-1] else "adjacent"
                        requests.extend(
                            _comparison_pair(
                                axis,
                                style_seed,
                                fixed,
                                left_by_index[index],
                                right_by_index[index],
                                kind,
                            )
                        )
    return tuple(requests)


def judge_comparison_requests(
    requests: Sequence[PairwiseComparisonRequest],
    candidates: Sequence[TwoAxisCandidate],
    judges: Sequence[PairwiseJudge],
) -> tuple[PairwiseJudgment, ...]:
    if not judges:
        raise ValueError("at least one pairwise judge is required")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    judgments: list[PairwiseJudgment] = []
    for request in requests:
        for judge in judges:
            winner = judge.compare(request, by_id)
            allowed = {request.left_candidate_id, request.right_candidate_id, "tie"}
            if winner not in allowed:
                raise ValueError(f"judge {judge.judge_id} returned invalid winner {winner!r}")
            judgments.append(
                PairwiseJudgment(
                    comparison_id=request.comparison_id,
                    judge_id=judge.judge_id,
                    winner_candidate_id=None if winner == "tie" else winner,
                    is_tie=winner == "tie",
                )
            )
    return tuple(judgments)


def calibrate_candidates(
    candidates: Sequence[TwoAxisCandidate],
    requests: Sequence[PairwiseComparisonRequest],
    judgments: Sequence[PairwiseJudgment],
    *,
    regularization: float = 0.5,
) -> tuple[TwoAxisCalibration, ...]:
    """Fit slice-specific Bradley--Terry scales and freeze endpoint anchors."""

    if regularization <= 0:
        raise ValueError("regularization must be positive")
    request_by_id = {request.comparison_id: request for request in requests}
    outcomes: dict[str, list[PairwiseJudgment]] = {}
    for judgment in judgments:
        if judgment.comparison_id not in request_by_id:
            raise ValueError("judgment references an unknown comparison")
        outcomes.setdefault(judgment.comparison_id, []).append(judgment)
    valid_ids = {candidate.candidate_id for candidate in candidates if candidate.structural_valid}
    scores: dict[str, dict[str, float]] = {candidate_id: {} for candidate_id in valid_ids}
    counts: dict[str, dict[str, int]] = {
        candidate_id: {"A1": 0, "A2": 0} for candidate_id in valid_ids
    }
    for axis in ("A1", "A2"):
        slices = sorted(
            {
                (request.style_seed, request.fixed_coordinate)
                for request in requests
                if request.axis == axis
            }
        )
        for style_seed, fixed in slices:
            slice_requests = [
                request
                for request in requests
                if request.axis == axis
                and request.style_seed == style_seed
                and math.isclose(request.fixed_coordinate, fixed)
            ]
            ids = sorted(
                {
                    candidate_id
                    for request in slice_requests
                    for candidate_id in (
                        request.left_candidate_id,
                        request.right_candidate_id,
                    )
                }
            )
            fitted = _fit_bradley_terry(ids, slice_requests, outcomes, regularization)
            slice_candidates = [candidate for candidate in candidates if candidate.candidate_id in fitted]
            coordinate_name = "assigned_a1" if axis == "A1" else "assigned_a2"
            lower_ids = [
                candidate.candidate_id
                for candidate in slice_candidates
                if math.isclose(getattr(candidate, coordinate_name), 0.0)
            ]
            upper_ids = [
                candidate.candidate_id
                for candidate in slice_candidates
                if math.isclose(getattr(candidate, coordinate_name), 1.0)
            ]
            if not lower_ids or not upper_ids:
                raise ValueError(f"{axis} calibration slice lacks endpoint anchors")
            lower = float(np.mean([fitted[candidate_id] for candidate_id in lower_ids]))
            upper = float(np.mean([fitted[candidate_id] for candidate_id in upper_ids]))
            if upper - lower <= 1e-9:
                raise ValueError(f"{axis} pairwise judgments do not order endpoint anchors")
            for candidate_id, score in fitted.items():
                scores[candidate_id][axis] = (score - lower) / (upper - lower)
            for request in slice_requests:
                comparison_count = len(outcomes.get(request.comparison_id, ()))
                counts[request.left_candidate_id][axis] += comparison_count
                counts[request.right_candidate_id][axis] += comparison_count
    return tuple(
        TwoAxisCalibration(
            candidate_id=candidate.candidate_id,
            realized_a1=scores[candidate.candidate_id]["A1"],
            realized_a2=scores[candidate.candidate_id]["A2"],
            a1_comparison_count=counts[candidate.candidate_id]["A1"],
            a2_comparison_count=counts[candidate.candidate_id]["A2"],
        )
        for candidate in candidates
        if candidate.structural_valid
    )


def select_prompt_population(
    candidates: Sequence[TwoAxisCandidate],
    calibrations: Sequence[TwoAxisCalibration],
    *,
    embedder: TwoAxisPromptEmbedder,
    monotonic_tolerance: float = 0.0,
    maximum_neighbor_embedding_distance: float | None = None,
    length_penalty: float = 0.02,
) -> tuple[tuple[SelectedTwoAxisPrompt, ...], TwoAxisSelectionDiagnostics]:
    """Select each 2D trajectory globally using binary min-cost optimization."""

    if monotonic_tolerance < 0 or length_penalty < 0:
        raise ValueError("selection tolerances and penalties must be non-negative")
    if maximum_neighbor_embedding_distance is not None and maximum_neighbor_embedding_distance <= 0:
        raise ValueError("maximum_neighbor_embedding_distance must be positive")
    calibration_by_id = {item.candidate_id: item for item in calibrations}
    usable = [
        candidate
        for candidate in candidates
        if candidate.structural_valid and candidate.candidate_id in calibration_by_id
    ]
    if not usable:
        raise ValueError("no structurally valid calibrated candidates")
    embedding_texts = [
        _query_bound_template(candidate.prompt_template, candidate.search_term)
        for candidate in usable
    ]
    embeddings = _validated_embeddings(embedder, embedding_texts)
    embedding_by_id = {
        candidate.candidate_id: embedding for candidate, embedding in zip(usable, embeddings)
    }
    selected: list[SelectedTwoAxisPrompt] = []
    used_hashes: set[str] = set()
    for style_seed in sorted({candidate.style_seed for candidate in usable}):
        style_candidates = [candidate for candidate in usable if candidate.style_seed == style_seed]
        chosen = _select_one_style(
            style_candidates,
            calibration_by_id,
            embedding_by_id,
            monotonic_tolerance=monotonic_tolerance,
            maximum_neighbor_embedding_distance=maximum_neighbor_embedding_distance,
            length_penalty=length_penalty,
            forbidden_hashes=used_hashes,
        )
        used_hashes.update(candidate.candidate_hash for candidate in chosen)
        for candidate in chosen:
            calibration = calibration_by_id[candidate.candidate_id]
            embedding = np.asarray(embedding_by_id[candidate.candidate_id], dtype=np.float64)
            cost = _candidate_cost(candidate, calibration, length_penalty)
            identity = {
                "candidate_id": candidate.candidate_id,
                "a1": f"{candidate.assigned_a1:.12g}",
                "a2": f"{candidate.assigned_a2:.12g}",
                "style_seed": style_seed,
            }
            identity_hash = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
            selected.append(
                SelectedTwoAxisPrompt(
                    prompt_assignment_id=f"two-axis-assignment:{identity_hash[:24]}",
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash,
                    assigned_a1=candidate.assigned_a1,
                    assigned_a2=candidate.assigned_a2,
                    realized_a1=calibration.realized_a1,
                    realized_a2=calibration.realized_a2,
                    style_seed=style_seed,
                    candidate_index=candidate.candidate_index,
                    search_term=candidate.search_term,
                    business_actor=candidate.business_actor,
                    search_objective_clause=candidate.search_objective_clause,
                    source_preference_clause=candidate.source_preference_clause,
                    output_contract=candidate.output_contract,
                    prompt_template=candidate.prompt_template,
                    embedding_model=embedder.model_name,
                    prompt_embedding=tuple(float(value) for value in embedding),
                    embedding_hash=hashlib.sha256(embedding.astype("<f8").tobytes()).hexdigest(),
                    calibration_cost=cost,
                )
            )
    selected_tuple = tuple(
        sorted(selected, key=lambda item: (item.style_seed, item.assigned_a1, item.assigned_a2))
    )
    diagnostics = _selection_diagnostics(selected_tuple)
    return selected_tuple, diagnostics


def render_selected_two_axis_prompt(
    prompt: SelectedTwoAxisPrompt,
    *,
    candidates: Sequence[SearchCandidate],
    top_n: int,
) -> RenderedSearchPurposePrompt:
    """Insert the exact query and frozen candidate evidence after selection."""

    bridge = SearchPurposePromptRecord(
        prompt_id=prompt.prompt_assignment_id,
        prompt_hash=prompt.candidate_hash,
        assigned_action_intensity=prompt.assigned_a1,
        style_seed=prompt.style_seed,
        top_n=top_n,
        style_plan=TemplatePromptGenerator._build_style_plan(prompt.style_seed),
        purpose_level="calibrated-two-axis",
        purpose_clause=prompt.search_objective_clause,
        prompt_template=prompt.prompt_template,
        prompt_space_version=POPULATION_VERSION,
        axis_specification_version=POPULATION_VERSION,
        generator_backend="selected-candidate-bank",
    )
    rendered = render_search_purpose_prompt(
        bridge,
        keyword=prompt.search_term,
        candidates=candidates,
        top_n=top_n,
    )
    if prompt.search_term not in rendered.rendered_prompt:
        raise RuntimeError("rendered prompt lost the exact search term")
    return rendered


def map_two_axis_prompt_to_permutation(
    prompt: SelectedTwoAxisPrompt,
    rendered: RenderedSearchPurposePrompt,
    raw_model_output: str,
    *,
    reranker_run_id: str,
    reranker_model: str,
) -> TwoAxisPermutationOutcome:
    if rendered.prompt_id != prompt.prompt_assignment_id:
        raise ValueError("rendered prompt does not match selected assignment")
    ranking = parse_ranking_permutation(raw_model_output, rendered)
    identity = {
        "prompt_assignment_id": prompt.prompt_assignment_id,
        "candidate_set_id": rendered.candidate_set_id,
        "permutation_hash": ranking.permutation_hash,
        "reranker_run_id": reranker_run_id,
        "reranker_model": reranker_model,
    }
    digest = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return TwoAxisPermutationOutcome(
        outcome_id=f"two-axis-permutation:{digest[:24]}",
        prompt_assignment_id=prompt.prompt_assignment_id,
        assigned_a1=prompt.assigned_a1,
        assigned_a2=prompt.assigned_a2,
        realized_a1=prompt.realized_a1,
        realized_a2=prompt.realized_a2,
        style_seed=prompt.style_seed,
        prompt_instance_id=rendered.prompt_instance_id,
        candidate_set_id=rendered.candidate_set_id,
        prompt_embedding_hash=prompt.embedding_hash,
        ranking=ranking,
        raw_model_output=raw_model_output,
        reranker_run_id=reranker_run_id,
        reranker_model=reranker_model,
    )


def measure_selected_latent_population(
    selected: Sequence[SelectedTwoAxisPrompt],
) -> TwoAxisLatentDiagnostics:
    """Measure whether query-bound selected prompts form coherent latent fields."""

    if not selected:
        raise ValueError("selected prompt population must be non-empty")
    embeddings = np.asarray([item.prompt_embedding for item in selected], dtype=np.float64)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("selected prompt embeddings must be one finite matrix")
    models = {item.embedding_model for item in selected}
    if len(models) != 1:
        raise ValueError("selected prompts use multiple embedding models")
    a1 = np.asarray([item.assigned_a1 for item in selected], dtype=np.float64)
    a2 = np.asarray([item.assigned_a2 for item in selected], dtype=np.float64)
    a1_low = embeddings[np.isclose(a1, 0.0)].mean(axis=0)
    a1_high = embeddings[np.isclose(a1, 1.0)].mean(axis=0)
    a2_low = embeddings[np.isclose(a2, 0.0)].mean(axis=0)
    a2_high = embeddings[np.isclose(a2, 1.0)].mean(axis=0)
    direction1 = a1_high - a1_low
    direction2 = a2_high - a2_low
    norm1 = float(np.linalg.norm(direction1))
    norm2 = float(np.linalg.norm(direction2))
    if norm1 <= 1e-12 or norm2 <= 1e-12:
        raise ValueError("latent endpoint centroids do not define two nonzero directions")
    unit1 = direction1 / norm1
    unit2 = direction2 / norm2
    projection1 = embeddings @ unit1
    projection2 = embeddings @ unit2
    a1_spearman = _latent_slice_spearman(selected, projection1, moving="A1")
    a2_spearman = _latent_slice_spearman(selected, projection2, moving="A2")
    a1_intended = abs(_least_squares_slope(a1, projection1))
    a1_cross = abs(_least_squares_slope(a2, projection1))
    a2_intended = abs(_least_squares_slope(a2, projection2))
    a2_cross = abs(_least_squares_slope(a1, projection2))
    adjacent, distant = _latent_neighbor_distances(selected)
    retained = sum(
        item.search_term
        in _query_bound_template(item.prompt_template, item.search_term)
        for item in selected
    )
    return TwoAxisLatentDiagnostics(
        embedding_model=next(iter(models)),
        selected_count=len(selected),
        embedding_dimension=int(embeddings.shape[1]),
        exact_query_structural_retention_rate=retained / len(selected),
        a1_endpoint_distance=norm1,
        a2_endpoint_distance=norm2,
        a1_a2_direction_cosine=float(unit1 @ unit2),
        a1_slice_spearman_mean=float(np.mean(a1_spearman)),
        a2_slice_spearman_mean=float(np.mean(a2_spearman)),
        a1_cross_axis_slope_ratio=a1_cross / max(a1_intended, 1e-12),
        a2_cross_axis_slope_ratio=a2_cross / max(a2_intended, 1e-12),
        mean_adjacent_embedding_distance=float(np.mean(adjacent)),
        mean_distant_embedding_distance=float(np.mean(distant)),
        adjacent_over_distant_distance_ratio=float(np.mean(adjacent) / np.mean(distant)),
    )


class FakeTwoAxisCandidateGenerator:
    """Deterministic clause generator for tests only; not scientific output."""

    backend_name = "fake-two-axis-candidate-generator"
    model_name = "fake-two-axis-generator-v1"
    _A1 = (
        "Understand the category, mechanisms, use cases, limitations, and relevant concepts without selecting a product.",
        "Build a category understanding and identify the main solution approaches without choosing a product.",
        "Understand the category and examine practical solution approaches before product evaluation.",
        "Combine category understanding with practical evaluation criteria and possible solution approaches.",
        "Evaluate relevant solution approaches and prepare a B2B software shortlist.",
        "Compare and shortlist B2B software products in preparation for acquisition or implementation.",
        "Actively evaluate, select, acquire, or implement a suitable B2B SaaS solution.",
    )
    _A2 = (
        "Conditional on equal topical relevance, prefer seller-independent research, neutral analysis, comparisons, and customer evidence.",
        "Conditional on equal topical relevance, give a clear preference to seller-independent evidence.",
        "Conditional on equal topical relevance, give a slight preference to seller-independent evidence.",
        "Apply no publisher-ownership preference; use the strongest evidence regardless of publisher.",
        "Conditional on equal topical relevance, give a slight preference to seller-controlled evidence.",
        "Conditional on equal topical relevance, give a clear preference to seller-controlled evidence.",
        "Conditional on equal topical relevance, prefer vendor product pages, documentation, implementation material, and other seller-controlled evidence.",
    )

    def generate(self, request: TwoAxisCandidateRequest) -> tuple[tuple[str, str], ...]:
        a1_index = int(round(request.assigned_a1 * 6))
        a2_index = int(round(request.assigned_a2 * 6))
        rows = []
        for index in range(request.number_candidates):
            suffix = (
                " Keep this as one search-agent task."
                if index % 2 == 0
                else " Preserve the business evaluator perspective."
            )
            objective = self._A1[a1_index] + suffix
            source = self._A2[a2_index] + (
                " Continue to rank all candidates by relevance."
                if index % 3 == 0
                else " Do not treat ownership as a substitute for relevance."
            )
            rows.append((objective, source))
        return tuple(rows)


class FakePairwiseJudge:
    """Blind deterministic semantic judge for plumbing tests only."""

    def __init__(self, judge_id: str, *, tie_epsilon: float = 0.015) -> None:
        self.judge_id = judge_id
        self.tie_epsilon = tie_epsilon

    def compare(
        self,
        request: PairwiseComparisonRequest,
        candidates: Mapping[str, TwoAxisCandidate],
    ) -> str:
        left = candidates[request.left_candidate_id]
        right = candidates[request.right_candidate_id]
        attribute = "assigned_a1" if request.axis == "A1" else "assigned_a2"
        left_score = float(getattr(left, attribute)) + _fake_judge_jitter(left, request.axis, self.judge_id)
        right_score = float(getattr(right, attribute)) + _fake_judge_jitter(right, request.axis, self.judge_id)
        if abs(left_score - right_score) <= self.tie_epsilon:
            return "tie"
        return left.candidate_id if left_score > right_score else right.candidate_id


class FakeTwoAxisPromptEmbedder:
    """Deterministic semantic/style representation for tests only."""

    model_name = "fake-two-axis-prompt-embedder-v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.casefold()
            a1 = max(
                (index / 6 for index, phrase in enumerate(FakeTwoAxisCandidateGenerator._A1) if phrase.split(".")[0].casefold() in lowered),
                default=0.5,
            )
            a2 = max(
                (index / 6 for index, phrase in enumerate(FakeTwoAxisCandidateGenerator._A2) if phrase.split(";")[0].split(".")[0].casefold() in lowered),
                default=0.5,
            )
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(
                [
                    a1,
                    a2,
                    float(text.startswith("Please")),
                    float("Your task" in text),
                    len(text) / 2000.0,
                    digest[0] / 255.0,
                ]
            )
        return np.asarray(rows, dtype=np.float64)


class LocalLLMTwoAxisCandidateGenerator:
    """Constrained JSON clause generator backed by a repository local ranker.

    The exact search term is intentionally absent from the generation request.
    It is inserted structurally only after candidate selection.
    """

    backend_name = "repository-local-constrained-json"

    def __init__(
        self,
        ranker,
        *,
        model_name: str,
        cache_directory: str | Path,
        max_new_tokens: int = 1200,
        temperature: float = 0.8,
        maximum_attempts: int = 3,
    ) -> None:
        if max_new_tokens <= 0 or temperature < 0 or maximum_attempts <= 0:
            raise ValueError("invalid local candidate-generator configuration")
        self._ranker = ranker
        self.model_name = model_name
        self.cache_directory = Path(cache_directory)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.maximum_attempts = maximum_attempts

    @classmethod
    def from_model(
        cls,
        model_name: str,
        *,
        cache_directory: str | Path,
        precision: str = "full",
        max_new_tokens: int = 1200,
        temperature: float = 0.8,
        maximum_attempts: int = 3,
    ) -> "LocalLLMTwoAxisCandidateGenerator":
        from ..utils import make_ranker

        return cls(
            make_ranker("local", model_name, precision=precision),
            model_name=model_name,
            cache_directory=cache_directory,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            maximum_attempts=maximum_attempts,
        )

    def generate(self, request: TwoAxisCandidateRequest) -> tuple[tuple[str, str], ...]:
        request_text = _candidate_generation_request(request)
        cache_identity = {
            "kind": "two-axis-candidate-generation",
            "version": POPULATION_VERSION,
            "model": self.model_name,
            "request": request_text,
            "seed": request.generation_seed,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
        cache_key = _hash(json.dumps(cache_identity, sort_keys=True, separators=(",", ":")))
        cache_path = self.cache_directory / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return tuple(
                (item["search_objective_clause"], item["source_preference_clause"])
                for item in payload["candidates"]
            )
        failures: list[dict[str, object]] = []
        for attempt in range(self.maximum_attempts):
            seed = request.generation_seed + attempt
            raw = _seeded_local_generation(
                self._ranker,
                request_text + ("" if not attempt else _candidate_retry_instruction()),
                seed=seed,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            try:
                rows = _parse_generated_candidate_clauses(raw, request.number_candidates)
                semantic_failures = _generated_clause_semantic_failures(
                    rows, style_seed=request.style_seed
                )
                if semantic_failures:
                    raise ValueError(
                        "generated clauses failed hard semantic screens: "
                        + json.dumps(semantic_failures, sort_keys=True)
                    )
            except ValueError as exc:
                failures.append({"attempt": attempt + 1, "seed": seed, "error": str(exc), "raw": raw})
                continue
            _atomic_json(
                cache_path,
                {
                    "cache_identity": cache_identity,
                    "accepted_attempt": attempt + 1,
                    "accepted_seed": seed,
                    "raw_model_output": raw,
                    "rejected_attempts": failures,
                    "candidates": [
                        {
                            "search_objective_clause": objective,
                            "source_preference_clause": source,
                        }
                        for objective, source in rows
                    ],
                },
            )
            return rows
        raise ValueError(
            "local candidate generator failed structural JSON validation after "
            f"{self.maximum_attempts} attempts: "
            + "; ".join(str(item["error"]) for item in failures)
        )


class LocalLLMPairwiseJudge:
    """Blind cached pairwise judge backed by a repository local ranker."""

    def __init__(
        self,
        ranker,
        *,
        judge_id: str,
        model_name: str,
        cache_directory: str | Path,
        max_new_tokens: int = 80,
        maximum_attempts: int = 3,
    ) -> None:
        if (
            not judge_id.strip()
            or not model_name.strip()
            or max_new_tokens <= 0
            or maximum_attempts <= 0
        ):
            raise ValueError("invalid local pairwise-judge configuration")
        self._ranker = ranker
        self.judge_id = judge_id
        self.model_name = model_name
        self.cache_directory = Path(cache_directory)
        self.max_new_tokens = max_new_tokens
        self.maximum_attempts = maximum_attempts

    @classmethod
    def from_model(
        cls,
        model_name: str,
        *,
        judge_id: str,
        cache_directory: str | Path,
        precision: str = "full",
        max_new_tokens: int = 80,
        maximum_attempts: int = 3,
    ) -> "LocalLLMPairwiseJudge":
        from ..utils import make_ranker

        return cls(
            make_ranker("local", model_name, precision=precision),
            judge_id=judge_id,
            model_name=model_name,
            cache_directory=cache_directory,
            max_new_tokens=max_new_tokens,
            maximum_attempts=maximum_attempts,
        )

    def compare(
        self,
        request: PairwiseComparisonRequest,
        candidates: Mapping[str, TwoAxisCandidate],
    ) -> str:
        left = candidates[request.left_candidate_id]
        right = candidates[request.right_candidate_id]
        request_text = _pairwise_judge_request(request, left, right)
        cache_identity = {
            "kind": "two-axis-pairwise-judgment",
            "version": POPULATION_VERSION,
            "judge_id": self.judge_id,
            "model": self.model_name,
            "comparison_id": request.comparison_id,
            "request": request_text,
        }
        cache_key = _hash(json.dumps(cache_identity, sort_keys=True, separators=(",", ":")))
        cache_path = self.cache_directory / f"{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            label = payload["winner"]
        else:
            rejected: list[dict[str, object]] = []
            label = None
            raw = ""
            base_seed = int(cache_key[:8], 16)
            for attempt in range(self.maximum_attempts):
                raw = _seeded_local_generation(
                    self._ranker,
                    request_text
                    + ("" if not attempt else "\nReturn the required JSON object only."),
                    seed=base_seed + attempt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
                try:
                    label = _parse_pairwise_winner(raw)
                    break
                except ValueError as exc:
                    rejected.append(
                        {"attempt": attempt + 1, "error": str(exc), "raw": raw}
                    )
            if label is None:
                raise ValueError(
                    f"judge {self.judge_id} failed JSON validation after "
                    f"{self.maximum_attempts} attempts"
                )
            _atomic_json(
                cache_path,
                {
                    "cache_identity": cache_identity,
                    "raw_model_output": raw,
                    "rejected_attempts": rejected,
                    "winner": label,
                },
            )
        if label == "tie":
            return "tie"
        return left.candidate_id if label == "left" else right.candidate_id


class LLM2VecGenPromptEmbedder:
    """Pooled LLM2Vec-Gen expected-response representation of prompt text.

    This is a diagnostic embedding only. No reconstruction state is decoded.
    The upstream high-level loader currently requires exactly one visible GPU.
    """

    def __init__(self, model_name: str, *, batch_size: int = 1, max_length: int = 512) -> None:
        try:
            import torch
            from llm2vec_gen import LLM2VecGenModel
        except ImportError as exc:
            raise ImportError("LLM2Vec-Gen prompt embedding requires llm2vec-gen") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("LLM2Vec-Gen prompt embedding requires exactly one visible GPU")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = LLM2VecGenModel.from_pretrained(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            pooled, _ = self._model.encode(
                list(texts[start : start + self.batch_size]),
                max_length=self.max_length,
                get_recon_hidden_states=True,
            )
            batches.append(pooled.detach().float().cpu().numpy())
        if not batches:
            raise ValueError("cannot embed an empty prompt collection")
        return np.concatenate(batches, axis=0)


class LLM2VecPromptEmbedder:
    """Primary frozen input-text representation using official LLM2Vec."""

    def __init__(
        self,
        model_name: str,
        *,
        peft_model_name_or_path: str | None = None,
        batch_size: int = 1,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from llm2vec import LLM2Vec
        except ImportError as exc:
            raise ImportError("LLM2Vec prompt embedding requires the llm2vec package") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("LLM2Vec prompt embedding requires exactly one visible GPU")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name = (
            model_name
            if peft_model_name_or_path is None
            else f"{model_name}+peft:{peft_model_name_or_path}"
        )
        self.batch_size = batch_size
        self._model = LLM2Vec.from_pretrained(
            model_name,
            peft_model_name_or_path=peft_model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            max_length=max_length,
            attn_implementation="eager",
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self._model.encode(list(texts[start : start + self.batch_size]))
            if hasattr(encoded, "detach"):
                encoded = encoded.detach().float().cpu().numpy()
            batches.append(np.asarray(encoded, dtype=np.float64))
        if not batches:
            raise ValueError("cannot embed an empty prompt collection")
        return np.concatenate(batches, axis=0)


def _compile_prompt(
    *,
    style: StylePlan,
    business_actor: str,
    objective_clause: str,
    source_clause: str,
    output_contract: str,
) -> str:
    verb = style.ranking_verb.lower()
    if style.syntax == "imperative":
        task = f"{style.ranking_verb} the supplied candidates for the exact search term by relevance."
    elif style.syntax == "request":
        task = f"Please {verb} the supplied candidates for the exact search term by relevance."
    else:
        task = f"Your task is to {verb} the supplied candidates for the exact search term by relevance."
    fixed = (
        f"Act as {business_actor}. Keep that actor, the search term, candidate set, "
        "ranking task, and output contract fixed."
    )
    semantic = (
        f"Search objective: {objective_clause}\n"
        f"Source-ownership preference: {source_clause}"
    )
    inputs = 'Search term: "{QUERY}"\n\nCandidates:\n{CANDIDATES}'
    blocks = (fixed, task, semantic, output_contract, inputs)
    if style.clause_order == "inputs_first":
        blocks = (inputs, fixed, task, semantic, output_contract)
    return _normalize_template("\n\n".join(blocks))


def _comparison_pair(
    axis: str,
    style_seed: int,
    fixed: float,
    first: TwoAxisCandidate,
    second: TwoAxisCandidate,
    kind: str,
) -> list[PairwiseComparisonRequest]:
    if axis == "A1":
        question = (
            "Which prompt more strongly indicates that a business software evaluator is ready "
            "to evaluate, select, acquire, or implement a B2B SaaS solution rather than merely understand the category?"
        )
    else:
        question = (
            "Which prompt more strongly prefers seller-controlled evidence rather than "
            "seller-independent evidence, assuming equal topical relevance?"
        )
    rows = []
    for order, left, right in (
        ("forward", first, second),
        ("reverse", second, first),
    ):
        identity = {
            "axis": axis,
            "style_seed": style_seed,
            "fixed": f"{fixed:.12g}",
            "left": left.candidate_id,
            "right": right.candidate_id,
            "order": order,
            "kind": kind,
        }
        digest = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        rows.append(
            PairwiseComparisonRequest(
                comparison_id=f"two-axis-comparison:{digest[:24]}",
                axis=axis,
                style_seed=style_seed,
                fixed_coordinate=fixed,
                left_candidate_id=left.candidate_id,
                right_candidate_id=right.candidate_id,
                presentation_order=order,
                comparison_kind=kind,
                question=question,
            )
        )
    return rows


def _fit_bradley_terry(
    candidate_ids: Sequence[str],
    requests: Sequence[PairwiseComparisonRequest],
    outcomes: Mapping[str, Sequence[PairwiseJudgment]],
    regularization: float,
) -> dict[str, float]:
    index = {candidate_id: position for position, candidate_id in enumerate(candidate_ids)}
    pairs: list[tuple[int, int, float]] = []
    for request in requests:
        for judgment in outcomes.get(request.comparison_id, ()):
            left = index[request.left_candidate_id]
            right = index[request.right_candidate_id]
            if judgment.is_tie:
                pairs.append((left, right, 0.5))
            elif judgment.winner_candidate_id == request.left_candidate_id:
                pairs.append((left, right, 1.0))
            else:
                pairs.append((left, right, 0.0))
    if not pairs:
        raise ValueError("calibration slice has no pairwise judgments")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.5 * regularization * float(theta @ theta)
        gradient = regularization * theta
        for left, right, outcome in pairs:
            delta = float(theta[left] - theta[right])
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, delta))))
            value -= outcome * math.log(max(probability, 1e-12))
            value -= (1.0 - outcome) * math.log(max(1.0 - probability, 1e-12))
            error = probability - outcome
            gradient[left] += error
            gradient[right] -= error
        return value, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.zeros(len(candidate_ids), dtype=np.float64),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")
    centered = result.x - float(np.mean(result.x))
    return {candidate_id: float(centered[position]) for candidate_id, position in index.items()}


def _select_one_style(
    candidates: Sequence[TwoAxisCandidate],
    calibrations: Mapping[str, TwoAxisCalibration],
    embeddings: Mapping[str, np.ndarray],
    *,
    monotonic_tolerance: float,
    maximum_neighbor_embedding_distance: float | None,
    length_penalty: float,
    forbidden_hashes: set[str],
) -> tuple[TwoAxisCandidate, ...]:
    ordered = sorted(candidates, key=lambda item: (item.assigned_a1, item.assigned_a2, item.candidate_index))
    cells: dict[tuple[float, float], list[int]] = {}
    for index, candidate in enumerate(ordered):
        cells.setdefault((candidate.assigned_a1, candidate.assigned_a2), []).append(index)
    a1_grid = sorted({cell[0] for cell in cells})
    a2_grid = sorted({cell[1] for cell in cells})
    expected = {(a1, a2) for a1 in a1_grid for a2 in a2_grid}
    if set(cells) != expected:
        raise ValueError("candidate bank does not cover a complete grid")

    lower: list[float] = []
    upper: list[float] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    constraint_row = 0
    for indices in cells.values():
        for index in indices:
            row_indices.append(constraint_row)
            column_indices.append(index)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        constraint_row += 1
    by_hash: dict[str, list[int]] = {}
    for index, candidate in enumerate(ordered):
        by_hash.setdefault(candidate.candidate_hash, []).append(index)
    for indices in by_hash.values():
        if len(indices) <= 1:
            continue
        for index in indices:
            row_indices.append(constraint_row)
            column_indices.append(index)
            values.append(1.0)
        lower.append(-np.inf)
        upper.append(1.0)
        constraint_row += 1

    for axis in ("A1", "A2"):
        outer = a2_grid if axis == "A1" else a1_grid
        moving = a1_grid if axis == "A1" else a2_grid
        score_name = "realized_a1" if axis == "A1" else "realized_a2"
        for fixed in outer:
            for left_level, right_level in zip(moving, moving[1:]):
                left_cell = (left_level, fixed) if axis == "A1" else (fixed, left_level)
                right_cell = (right_level, fixed) if axis == "A1" else (fixed, right_level)
                for left_index in cells[left_cell]:
                    for right_index in cells[right_cell]:
                        left_candidate = ordered[left_index]
                        right_candidate = ordered[right_index]
                        left_score = getattr(calibrations[left_candidate.candidate_id], score_name)
                        right_score = getattr(calibrations[right_candidate.candidate_id], score_name)
                        distance = float(
                            np.linalg.norm(
                                embeddings[left_candidate.candidate_id]
                                - embeddings[right_candidate.candidate_id]
                            )
                        )
                        incompatible = left_score > right_score + monotonic_tolerance
                        if maximum_neighbor_embedding_distance is not None:
                            incompatible = incompatible or distance > maximum_neighbor_embedding_distance
                        if incompatible:
                            for index in (left_index, right_index):
                                row_indices.append(constraint_row)
                                column_indices.append(index)
                                values.append(1.0)
                            lower.append(-np.inf)
                            upper.append(1.0)
                            constraint_row += 1
    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(constraint_row, len(ordered)),
        dtype=np.float64,
    )
    costs = np.asarray(
        [_candidate_cost(candidate, calibrations[candidate.candidate_id], length_penalty) for candidate in ordered],
        dtype=np.float64,
    )
    variable_upper = np.asarray(
        [0.0 if candidate.candidate_hash in forbidden_hashes else 1.0 for candidate in ordered],
        dtype=np.float64,
    )
    result = milp(
        costs,
        integrality=np.ones(len(ordered), dtype=np.int8),
        bounds=Bounds(np.zeros(len(ordered)), variable_upper),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "no feasible globally selected trajectory under monotonicity, uniqueness, "
            "and neighbor-distance constraints"
        )
    chosen = tuple(candidate for candidate, value in zip(ordered, result.x) if value > 0.5)
    if len(chosen) != len(cells):
        raise RuntimeError("global selector did not choose exactly one prompt per cell")
    return chosen


def _candidate_cost(
    candidate: TwoAxisCandidate,
    calibration: TwoAxisCalibration,
    length_penalty: float,
) -> float:
    calibration_error = (
        calibration.realized_a1 - candidate.assigned_a1
    ) ** 2 + (
        calibration.realized_a2 - candidate.assigned_a2
    ) ** 2
    normalized_length = len(candidate.prompt_template) / 1000.0
    return float(calibration_error + length_penalty * normalized_length**2)


def _selection_diagnostics(
    selected: Sequence[SelectedTwoAxisPrompt],
) -> TwoAxisSelectionDiagnostics:
    grouped: dict[int, dict[tuple[float, float], SelectedTwoAxisPrompt]] = {}
    for prompt in selected:
        grouped.setdefault(prompt.style_seed, {})[
            (prompt.assigned_a1, prompt.assigned_a2)
        ] = prompt
    a1_reversals = a1_ties = a1_pairs = 0
    a2_reversals = a2_ties = a2_pairs = 0
    maximum_distance = 0.0
    monotone_styles = 0
    for cells in grouped.values():
        a1_grid = sorted({cell[0] for cell in cells})
        a2_grid = sorted({cell[1] for cell in cells})
        style_monotone = True
        for a2 in a2_grid:
            row = [cells[(a1, a2)] for a1 in a1_grid]
            for left, right in zip(row, row[1:]):
                a1_pairs += 1
                if right.realized_a1 < left.realized_a1:
                    a1_reversals += 1
                    style_monotone = False
                elif math.isclose(right.realized_a1, left.realized_a1, abs_tol=1e-9):
                    a1_ties += 1
                maximum_distance = max(
                    maximum_distance,
                    float(np.linalg.norm(np.asarray(right.prompt_embedding) - np.asarray(left.prompt_embedding))),
                )
        for a1 in a1_grid:
            column = [cells[(a1, a2)] for a2 in a2_grid]
            for left, right in zip(column, column[1:]):
                a2_pairs += 1
                if right.realized_a2 < left.realized_a2:
                    a2_reversals += 1
                    style_monotone = False
                elif math.isclose(right.realized_a2, left.realized_a2, abs_tol=1e-9):
                    a2_ties += 1
                maximum_distance = max(
                    maximum_distance,
                    float(np.linalg.norm(np.asarray(right.prompt_embedding) - np.asarray(left.prompt_embedding))),
                )
        monotone_styles += int(style_monotone)
    errors = [
        abs(prompt.realized_a1 - prompt.assigned_a1)
        + abs(prompt.realized_a2 - prompt.assigned_a2)
        for prompt in selected
    ]
    hashes = [prompt.candidate_hash for prompt in selected]
    return TwoAxisSelectionDiagnostics(
        selected_count=len(selected),
        style_count=len(grouped),
        cells_per_style=len(selected) // len(grouped),
        duplicate_hash_count=len(hashes) - len(set(hashes)),
        a1_adjacent_reversal_rate=a1_reversals / a1_pairs,
        a2_adjacent_reversal_rate=a2_reversals / a2_pairs,
        a1_adjacent_tie_rate=a1_ties / a1_pairs,
        a2_adjacent_tie_rate=a2_ties / a2_pairs,
        fully_monotone_style_rate=monotone_styles / len(grouped),
        maximum_neighbor_embedding_distance=maximum_distance,
        mean_calibration_l1_error=float(np.mean(errors)),
    )


def _validated_embeddings(
    embedder: TwoAxisPromptEmbedder, texts: Sequence[str]
) -> np.ndarray:
    embeddings = np.asarray(embedder.embed(texts), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] <= 0:
        raise ValueError(f"embedder returned invalid shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    return embeddings


def _coordinate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _grid(name: str, values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(_coordinate(name, value) for value in values)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise ValueError(f"{name} must be non-empty, strictly increasing, and unique")
    if not math.isclose(grid[0], 0.0) or not math.isclose(grid[-1], 1.0):
        raise ValueError(f"{name} must include endpoint anchors 0 and 1")
    return grid


def _generation_seed(master_seed: int, style_seed: int, a1: float, a2: float) -> int:
    payload = f"{master_seed}:{style_seed}:{a1:.12g}:{a2:.12g}"
    return int(_hash(payload)[:8], 16)


def _fake_judge_jitter(
    candidate: TwoAxisCandidate, axis: str, judge_id: str
) -> float:
    digest = hashlib.sha256(
        f"{candidate.candidate_id}:{axis}:{judge_id}".encode("utf-8")
    ).digest()
    return ((digest[0] / 255.0) - 0.5) * 0.02


def _single_line(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text fields must be strings")
    return " ".join(value.split())


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_generation_request(request: TwoAxisCandidateRequest) -> str:
    style = TemplatePromptGenerator._build_style_plan(request.style_seed)
    return f"""Generate alternative semantic clause pairs for a B2B SaaS search-agent reranking instruction.

The exact search term is intentionally not provided. Do not invent or mention a query, product category, company, geography, or candidate document.

Assigned A1 decision-readiness coordinate: {request.assigned_a1:.6f}
A1=0 means understand the category without selecting a product.
A1=0.5 means understand it and develop practical evaluation criteria and solution approaches.
A1=1 means actively evaluate, compare, shortlist, trial, acquire, or implement a B2B SaaS solution.

Assigned A2 source-ownership coordinate: {request.assigned_a2:.6f}
A2=0 means prefer seller-independent evidence, conditional on equal topical relevance.
A2=0.5 means no publisher-ownership preference.
A2=1 means prefer seller-controlled evidence, conditional on equal topical relevance.
At low A1, seller-controlled evidence means vendor educational content or technical documentation, not automatically pricing or demo pages.

Fixed actor: a business software evaluator assessing a B2B SaaS category for an organization.
Fixed task: rerank one supplied candidate set for one exact search term by relevance.
Fixed output contract: candidate identifiers only, with no explanation.
Surface style plan: {json.dumps(asdict(style), sort_keys=True, separators=(",", ":"))}

Hard constraints:
- Change only decision-readiness in the objective clause and publisher ownership in the source clause.
- State that ownership is conditional on relevance or is not a substitute for relevance.
- Do not introduce freshness, popularity, brand prestige, authority, credibility, price, cost, company size, geography, review score, writing quality, or hard exclusions.
- Do not answer the unknown search term.
- Do not include numeric A1/A2 coordinates in either clause.
- Each clause must be one line and must not contain placeholders.
- Produce {request.number_candidates} genuinely distinct pairs.

Return JSON only in this exact shape:
{{"candidates":[{{"search_objective_clause":"...","source_preference_clause":"..."}}]}}
"""


def _candidate_retry_instruction() -> str:
    return (
        "\nYour previous response failed structural validation. Regenerate the complete "
        "JSON object with the exact requested number of distinct clause pairs. Return "
        "JSON only, without Markdown fences or commentary."
    )


def _parse_generated_candidate_clauses(
    raw_model_output: str, expected_count: int
) -> tuple[tuple[str, str], ...]:
    try:
        payload = json.loads(raw_model_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ValueError("JSON must contain only the candidates field")
    values = payload["candidates"]
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"expected exactly {expected_count} candidates")
    rows: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "search_objective_clause",
            "source_preference_clause",
        }:
            raise ValueError("candidate has an invalid JSON shape")
        objective = _single_line(value["search_objective_clause"])
        source = _single_line(value["source_preference_clause"])
        if not objective or not source:
            raise ValueError("candidate clauses must be non-empty strings")
        if any(token in objective + source for token in ("{QUERY}", "{CANDIDATES}", "{TOP_N}")):
            raise ValueError("generated clauses must not contain structural placeholders")
        rows.append((objective, source))
    if len(set(rows)) != len(rows):
        raise ValueError("generated candidate clause pairs must be distinct")
    return tuple(rows)


def _generated_clause_semantic_failures(
    rows: Sequence[tuple[str, str]], *, style_seed: int
) -> dict[str, int]:
    actor = "a business software evaluator assessing a B2B SaaS category for an organization"
    style = TemplatePromptGenerator._build_style_plan(style_seed)
    failures: dict[str, int] = {}
    for objective, source in rows:
        template = _compile_prompt(
            style=style,
            business_actor=actor,
            objective_clause=objective,
            source_clause=source,
            output_contract="Return exactly {TOP_N} candidate identifiers only, with no explanation.",
        )
        reasons = semantic_contract_checks(
            template,
            search_term="__structural_query_sentinel__",
            business_actor=actor,
            objective_clause=objective,
            source_preference_clause=source,
        )
        for reason in reasons:
            failures[reason] = failures.get(reason, 0) + 1
    return failures


def _pairwise_judge_request(
    request: PairwiseComparisonRequest,
    left: TwoAxisCandidate,
    right: TwoAxisCandidate,
) -> str:
    left_text = _query_bound_template(left.prompt_template, left.search_term)
    right_text = _query_bound_template(right.prompt_template, right.search_term)
    return f"""You are a blind semantic calibration judge. Compare two B2B SaaS search-agent instructions along exactly one axis.

Axis question: {request.question}

Ignore wording quality, length, syntax, clause order, and every ranking criterion outside the stated axis. Both prompts contain the same exact search term. Do not infer numeric target coordinates.

LEFT PROMPT:
{left_text}

RIGHT PROMPT:
{right_text}

Return JSON only: {{"winner":"left"}}, {{"winner":"right"}}, or {{"winner":"tie"}}.
"""


def _parse_pairwise_winner(raw_model_output: str) -> str:
    try:
        payload = json.loads(raw_model_output.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("pairwise judge returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"winner"}:
        raise ValueError("pairwise judge returned an invalid JSON shape")
    winner = payload["winner"]
    if winner not in {"left", "right", "tie"}:
        raise ValueError("pairwise judge winner must be left, right, or tie")
    return str(winner)


def _seeded_local_generation(
    ranker,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return str(
            ranker.rank(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        ).strip()


def _query_bound_template(prompt_template: str, search_term: str) -> str:
    if prompt_template.count("{QUERY}") != 1:
        raise ValueError("prompt template must contain exactly one {QUERY} placeholder")
    query = _single_line(search_term)
    if not query:
        raise ValueError("search term must be non-empty")
    bound = prompt_template.replace("{QUERY}", query)
    if query not in bound:
        raise RuntimeError("query-bound embedding text lost the exact search term")
    return bound


def _latent_slice_spearman(
    selected: Sequence[SelectedTwoAxisPrompt],
    projections: np.ndarray,
    *,
    moving: str,
) -> tuple[float, ...]:
    from scipy.stats import spearmanr

    groups: dict[tuple[int, float], list[tuple[float, float]]] = {}
    for item, projection in zip(selected, projections):
        if moving == "A1":
            key = (item.style_seed, item.assigned_a2)
            assigned = item.assigned_a1
        else:
            key = (item.style_seed, item.assigned_a1)
            assigned = item.assigned_a2
        groups.setdefault(key, []).append((assigned, float(projection)))
    values: list[float] = []
    for rows in groups.values():
        rows.sort()
        result = spearmanr([row[0] for row in rows], [row[1] for row in rows])
        statistic = float(getattr(result, "statistic", result[0]))
        if math.isfinite(statistic):
            values.append(statistic)
    if not values:
        raise ValueError(f"no finite latent {moving} slice correlations")
    return tuple(values)


def _least_squares_slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - float(np.mean(x))
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        raise ValueError("cannot estimate slope from a constant coordinate")
    return float(centered @ (y - float(np.mean(y))) / denominator)


def _latent_neighbor_distances(
    selected: Sequence[SelectedTwoAxisPrompt],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    grouped: dict[int, dict[tuple[float, float], np.ndarray]] = {}
    for item in selected:
        grouped.setdefault(item.style_seed, {})[(item.assigned_a1, item.assigned_a2)] = np.asarray(
            item.prompt_embedding, dtype=np.float64
        )
    adjacent: list[float] = []
    distant: list[float] = []
    for cells in grouped.values():
        grid1 = sorted({cell[0] for cell in cells})
        grid2 = sorted({cell[1] for cell in cells})
        for fixed in grid2:
            adjacent.extend(
                float(np.linalg.norm(cells[(right, fixed)] - cells[(left, fixed)]))
                for left, right in zip(grid1, grid1[1:])
            )
            distant.append(float(np.linalg.norm(cells[(grid1[-1], fixed)] - cells[(grid1[0], fixed)])))
        for fixed in grid1:
            adjacent.extend(
                float(np.linalg.norm(cells[(fixed, right)] - cells[(fixed, left)]))
                for left, right in zip(grid2, grid2[1:])
            )
            distant.append(float(np.linalg.norm(cells[(fixed, grid2[-1])] - cells[(fixed, grid2[0])])))
    if not adjacent or not distant or float(np.mean(distant)) <= 1e-12:
        raise ValueError("latent population lacks adjacent or distant grid pairs")
    return tuple(adjacent), tuple(distant)


def _atomic_json(path: Path, payload: object) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
