"""Calibrated one-axis population of decision-readiness prompts.

The assigned treatment is ``A1`` in ``[0, 1]``.  Candidate objective clauses
are generated as text, calibrated with blind pairwise comparisons, represented
independently by LLM2Vec and LLM2Vec-Gen, and selected as complete monotone
style trajectories.  Embeddings influence smoothness and diversity during
pre-outcome corpus construction but never redefine the randomized treatment.

Fake providers in this module exist only for CPU contract tests.
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
from scipy.optimize import minimize

from .prompt_continuum import StylePlan, TemplatePromptGenerator, _normalize_template

A1_MANIFOLD_VERSION = "a1-prompt-manifold-v1"
NEUTRAL_SOURCE_CLAUSE = (
    "Apply no publisher-ownership preference; use the strongest evidence regardless "
    "of publisher, with topical relevance as the primary criterion."
)
BUSINESS_ACTOR = (
    "a business software evaluator assessing a B2B SaaS category for an organization"
)
OUTPUT_CONTRACT = "Return exactly {TOP_N} candidate identifiers only, with no explanation."
DEFAULT_A1_GRID = tuple(step / 6.0 for step in range(7))

_FORBIDDEN = re.compile(
    r"\b(?:fresh(?:ness)?|recen(?:t|cy)|popular(?:ity)?|brand prestige|authority|"
    r"credib(?:le|ility)|price|pricing|cost|budget|company size|geograph(?:y|ic)|"
    r"region|country|review scores?|ratings?|writing quality|exclude|only rank|"
    r"publisher|source ownership|seller-controlled|seller-independent|vendor-controlled|"
    r"vendor-independent|first-party|third-party)\b",
    re.IGNORECASE,
)
_INFORMATIONAL = re.compile(
    r"\b(?:understand|learn|explor|explain|mechanism|use cases?|limitations?|concepts?|"
    r"category knowledge|foundational)\w*\b",
    re.IGNORECASE,
)
_TRANSACTIONAL = re.compile(
    r"\b(?:assess|evaluat|compar|shortlist|trial|acquir|purchas|implement|select)\w*\b",
    re.IGNORECASE,
)
_NON_SELECTION = re.compile(
    r"\b(?:without|before|not)\s+(?:actively\s+)?(?:assess|evaluat|select|choos|shortlist|"
    r"trial|acquir|purchas|implement)\w*",
    re.IGNORECASE,
)
_CANDIDATE_CARDINALITY = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:supplied\s+)?candidates?\b",
    re.IGNORECASE,
)


class A1CandidateGenerator(Protocol):
    backend_name: str
    model_name: str

    def generate(self, request: "A1CandidateRequest") -> tuple[str, ...]: ...


class A1PairwiseJudge(Protocol):
    judge_id: str

    def compare(
        self,
        request: "A1ComparisonRequest",
        candidates: Mapping[str, "A1Candidate"],
    ) -> str: ...


class PromptEmbedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class A1CandidateRequest:
    assigned_a1: float
    style_seed: int
    generation_seed: int
    number_candidates: int
    generator_model: str

    def __post_init__(self) -> None:
        _coordinate(self.assigned_a1)
        if self.number_candidates <= 0:
            raise ValueError("number_candidates must be positive")


@dataclass(frozen=True, slots=True)
class A1Candidate:
    candidate_id: str
    candidate_hash: str
    assigned_a1: float
    style_seed: int
    candidate_index: int
    generation_seed: int
    search_term: str
    search_objective_clause: str
    prompt_template: str
    generator_backend: str
    generator_model: str
    structural_valid: bool
    contract_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class A1ComparisonRequest:
    comparison_id: str
    style_seed: int
    left_candidate_id: str
    right_candidate_id: str
    presentation_order: str
    comparison_kind: str
    question: str


@dataclass(frozen=True, slots=True)
class A1Judgment:
    comparison_id: str
    judge_id: str
    winner_candidate_id: str | None
    is_tie: bool


@dataclass(frozen=True, slots=True)
class A1Calibration:
    candidate_id: str
    realized_a1: float
    comparison_count: int


@dataclass(frozen=True, slots=True)
class A1Embedding:
    candidate_id: str
    representation: str
    model_name: str
    values: tuple[float, ...]
    embedding_hash: str


@dataclass(frozen=True, slots=True)
class SelectedA1Prompt:
    prompt_assignment_id: str
    candidate_id: str
    candidate_hash: str
    assigned_a1: float
    realized_a1: float
    style_seed: int
    candidate_index: int
    search_term: str
    search_objective_clause: str
    prompt_template: str
    input_embedding_hash: str
    response_embedding_hash: str


@dataclass(frozen=True, slots=True)
class A1ManifoldDiagnostics:
    selected_count: int
    style_count: int
    levels_per_style: int
    exact_query_structural_retention_rate: float
    duplicate_hash_count: int
    fully_strict_monotone_style_rate: float
    adjacent_reversal_rate: float
    mean_realized_a1_absolute_error: float
    mean_style_spearman: float
    input_mean_adjacent_distance: float
    input_adjacent_distance_cv: float
    input_mean_tortuosity: float
    response_mean_adjacent_distance: float
    response_adjacent_distance_cv: float
    response_mean_tortuosity: float
    mean_pairwise_lexical_similarity: float


def generate_a1_candidate_bank(
    *,
    search_term: str,
    a1_grid: Sequence[float] = DEFAULT_A1_GRID,
    style_seeds: Sequence[int] = tuple(range(24)),
    number_candidates: int = 12,
    master_seed: int = 20260817,
    generator: A1CandidateGenerator,
) -> tuple[A1Candidate, ...]:
    query = _one_line(search_term)
    if not query:
        raise ValueError("search_term must be non-empty")
    grid = tuple(float(value) for value in a1_grid)
    if not grid or sorted(set(grid)) != list(grid) or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("a1_grid must be unique, increasing, and include 0 and 1")
    if not style_seeds or len(set(style_seeds)) != len(style_seeds):
        raise ValueError("style_seeds must be non-empty and unique")
    rows: list[A1Candidate] = []
    for style_seed in style_seeds:
        style = TemplatePromptGenerator._build_style_plan(style_seed)
        for level_index, assigned_a1 in enumerate(grid):
            request = A1CandidateRequest(
                assigned_a1=assigned_a1,
                style_seed=style_seed,
                generation_seed=_seed(master_seed, style_seed, level_index),
                number_candidates=number_candidates,
                generator_model=generator.model_name,
            )
            objectives = generator.generate(request)
            if len(objectives) != number_candidates:
                raise ValueError("generator returned the wrong number of A1 candidates")
            for candidate_index, objective in enumerate(objectives):
                objective = _one_line(objective)
                template = _compile_prompt(style, objective)
                failures = a1_contract_checks(
                    template,
                    objective,
                    assigned_a1=assigned_a1,
                    search_term=query,
                )
                identity = {
                    "version": A1_MANIFOLD_VERSION,
                    "a1": f"{assigned_a1:.12g}",
                    "style": style_seed,
                    "candidate": candidate_index,
                    "objective": objective,
                    "template": template,
                }
                digest = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
                rows.append(
                    A1Candidate(
                        candidate_id=f"a1-candidate:{digest[:24]}",
                        candidate_hash=_hash(template),
                        assigned_a1=assigned_a1,
                        style_seed=style_seed,
                        candidate_index=candidate_index,
                        generation_seed=request.generation_seed + candidate_index,
                        search_term=query,
                        search_objective_clause=objective,
                        prompt_template=template,
                        generator_backend=generator.backend_name,
                        generator_model=generator.model_name,
                        structural_valid=not failures,
                        contract_failures=failures,
                    )
                )
    return tuple(rows)


def a1_contract_checks(
    prompt_template: str,
    objective_clause: str,
    *,
    assigned_a1: float,
    search_term: str,
) -> tuple[str, ...]:
    _coordinate(assigned_a1)
    failures: list[str] = []
    if prompt_template.count("{QUERY}") != 1:
        failures.append("query-placeholder-count")
    if search_term.casefold() in prompt_template.casefold():
        failures.append("literal-query-generated")
    for placeholder in ("{CANDIDATES}", "{TOP_N}"):
        if prompt_template.count(placeholder) != 1:
            failures.append(f"placeholder-count:{placeholder}")
    if NEUTRAL_SOURCE_CLAUSE not in prompt_template:
        failures.append("neutral-source-clause-changed")
    if _FORBIDDEN.search(objective_clause):
        failures.append("off-axis-criterion")
    if _CANDIDATE_CARDINALITY.search(objective_clause):
        failures.append("candidate-cardinality-leak")
    if re.search(r"\bA1\b|\b[01](?:\.\d+)?\b", objective_clause, re.IGNORECASE):
        failures.append("numeric-coordinate-leak")
    informational = bool(_INFORMATIONAL.search(objective_clause))
    transactional = bool(_TRANSACTIONAL.search(objective_clause))
    non_selection = bool(_NON_SELECTION.search(objective_clause))
    if assigned_a1 == 0.0 and (not informational or (transactional and not non_selection)):
        failures.append("A1-low-mismatch")
    if assigned_a1 == 1.0 and (not transactional or non_selection):
        failures.append("A1-high-mismatch")
    return tuple(failures)


def build_a1_comparison_requests(
    candidates: Sequence[A1Candidate],
) -> tuple[A1ComparisonRequest, ...]:
    valid = [item for item in candidates if item.structural_valid]
    grouped: dict[tuple[int, float], list[A1Candidate]] = {}
    for item in valid:
        grouped.setdefault((item.style_seed, item.assigned_a1), []).append(item)
    rows: list[A1ComparisonRequest] = []
    for style_seed in sorted({item.style_seed for item in valid}):
        levels = sorted({item.assigned_a1 for item in valid if item.style_seed == style_seed})
        cells = [sorted(grouped[(style_seed, level)], key=lambda item: item.candidate_index) for level in levels]
        for cell in cells:
            for other in cell[1:]:
                rows.extend(_comparison_pair(style_seed, cell[0], other, "within-cell"))
        level_pairs = list(zip(cells, cells[1:]))
        if len(cells) > 2:
            level_pairs.append((cells[0], cells[-1]))
        for left_cell, right_cell in level_pairs:
            kind = "endpoint" if left_cell is cells[0] and right_cell is cells[-1] else "adjacent"
            for left, right in zip(left_cell, right_cell):
                rows.extend(_comparison_pair(style_seed, left, right, kind))
    return tuple(rows)


def judge_a1_comparisons(
    requests: Sequence[A1ComparisonRequest],
    candidates: Sequence[A1Candidate],
    judges: Sequence[A1PairwiseJudge],
) -> tuple[A1Judgment, ...]:
    if not judges:
        raise ValueError("at least one judge is required")
    by_id = {item.candidate_id: item for item in candidates}
    rows: list[A1Judgment] = []
    for request in requests:
        for judge in judges:
            winner = judge.compare(request, by_id)
            if winner not in {request.left_candidate_id, request.right_candidate_id, "tie"}:
                raise ValueError("judge returned an invalid winner")
            rows.append(
                A1Judgment(
                    comparison_id=request.comparison_id,
                    judge_id=judge.judge_id,
                    winner_candidate_id=None if winner == "tie" else winner,
                    is_tie=winner == "tie",
                )
            )
    return tuple(rows)


def calibrate_a1_candidates(
    candidates: Sequence[A1Candidate],
    requests: Sequence[A1ComparisonRequest],
    judgments: Sequence[A1Judgment],
    *,
    regularization: float = 0.5,
) -> tuple[A1Calibration, ...]:
    request_by_id = {item.comparison_id: item for item in requests}
    outcomes: dict[str, list[A1Judgment]] = {}
    for item in judgments:
        if item.comparison_id not in request_by_id:
            raise ValueError("judgment references an unknown comparison")
        outcomes.setdefault(item.comparison_id, []).append(item)
    result: list[A1Calibration] = []
    for style_seed in sorted({item.style_seed for item in candidates}):
        style_candidates = [
            item for item in candidates if item.style_seed == style_seed and item.structural_valid
        ]
        style_requests = [item for item in requests if item.style_seed == style_seed]
        fitted = _fit_bradley_terry(
            [item.candidate_id for item in style_candidates],
            style_requests,
            outcomes,
            regularization,
        )
        low = np.mean([fitted[item.candidate_id] for item in style_candidates if item.assigned_a1 == 0.0])
        high = np.mean([fitted[item.candidate_id] for item in style_candidates if item.assigned_a1 == 1.0])
        if high - low <= 1e-9:
            raise ValueError(f"A1 endpoint judgments are reversed for style_seed={style_seed}")
        counts = {item.candidate_id: 0 for item in style_candidates}
        for request in style_requests:
            count = len(outcomes.get(request.comparison_id, ()))
            counts[request.left_candidate_id] += count
            counts[request.right_candidate_id] += count
        result.extend(
            A1Calibration(
                candidate_id=item.candidate_id,
                realized_a1=float((fitted[item.candidate_id] - low) / (high - low)),
                comparison_count=counts[item.candidate_id],
            )
            for item in style_candidates
        )
    return tuple(result)


def embed_a1_candidates(
    candidates: Sequence[A1Candidate],
    *,
    embedder: PromptEmbedder,
    representation: str,
) -> tuple[A1Embedding, ...]:
    if representation not in {"input", "anticipated-response"}:
        raise ValueError("unsupported representation")
    valid = [item for item in candidates if item.structural_valid]
    texts = [_query_bound(item.prompt_template, item.search_term) for item in valid]
    matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(valid) or not np.isfinite(matrix).all():
        raise ValueError("embedder returned an invalid matrix")
    return tuple(
        A1Embedding(
            candidate_id=item.candidate_id,
            representation=representation,
            model_name=embedder.model_name,
            values=tuple(float(value) for value in row),
            embedding_hash=_hash(row.astype("<f8", copy=False).tobytes()),
        )
        for item, row in zip(valid, matrix)
    )


def select_a1_manifold(
    candidates: Sequence[A1Candidate],
    calibrations: Sequence[A1Calibration],
    input_embeddings: Sequence[A1Embedding],
    response_embeddings: Sequence[A1Embedding],
    *,
    calibration_weight: float = 4.0,
    smoothness_weight: float = 1.0,
    curvature_weight: float = 0.5,
    diversity_weight: float = 0.25,
    minimum_realized_step: float = 0.0,
) -> tuple[tuple[SelectedA1Prompt, ...], A1ManifoldDiagnostics]:
    if min(calibration_weight, smoothness_weight, curvature_weight, diversity_weight) < 0:
        raise ValueError("selection weights must be nonnegative")
    calibration = {item.candidate_id: item for item in calibrations}
    input_map = _embedding_map(input_embeddings, "input")
    response_map = _embedding_map(response_embeddings, "anticipated-response")
    valid = [
        item
        for item in candidates
        if item.structural_valid
        and item.candidate_id in calibration
        and item.candidate_id in input_map
        and item.candidate_id in response_map
    ]
    selected: list[SelectedA1Prompt] = []
    forbidden_hashes: set[str] = set()
    prior_texts: list[str] = []
    for style_seed in sorted({item.style_seed for item in valid}):
        style_candidates = [item for item in valid if item.style_seed == style_seed]
        chosen, _ = _select_style_path(
            style_candidates,
            calibration,
            input_map,
            response_map,
            forbidden_hashes=forbidden_hashes,
            prior_texts=prior_texts,
            calibration_weight=calibration_weight,
            smoothness_weight=smoothness_weight,
            curvature_weight=curvature_weight,
            diversity_weight=diversity_weight,
            minimum_realized_step=minimum_realized_step,
        )
        for item in chosen:
            identity = f"{item.candidate_id}:{item.assigned_a1:.12g}:{item.style_seed}"
            selected.append(
                SelectedA1Prompt(
                    prompt_assignment_id=f"a1-assignment:{_hash(identity)[:24]}",
                    candidate_id=item.candidate_id,
                    candidate_hash=item.candidate_hash,
                    assigned_a1=item.assigned_a1,
                    realized_a1=calibration[item.candidate_id].realized_a1,
                    style_seed=item.style_seed,
                    candidate_index=item.candidate_index,
                    search_term=item.search_term,
                    search_objective_clause=item.search_objective_clause,
                    prompt_template=item.prompt_template,
                    input_embedding_hash=input_embeddings[
                        next(i for i, value in enumerate(input_embeddings) if value.candidate_id == item.candidate_id)
                    ].embedding_hash,
                    response_embedding_hash=response_embeddings[
                        next(i for i, value in enumerate(response_embeddings) if value.candidate_id == item.candidate_id)
                    ].embedding_hash,
                )
            )
            forbidden_hashes.add(item.candidate_hash)
            prior_texts.append(item.search_objective_clause)
    diagnostics = measure_a1_manifold(selected, input_map, response_map)
    return tuple(selected), diagnostics


def measure_a1_manifold(
    selected: Sequence[SelectedA1Prompt],
    input_embeddings: Mapping[str, np.ndarray],
    response_embeddings: Mapping[str, np.ndarray],
) -> A1ManifoldDiagnostics:
    if not selected:
        raise ValueError("selected prompt population must be non-empty")
    by_style: dict[int, list[SelectedA1Prompt]] = {}
    for item in selected:
        by_style.setdefault(item.style_seed, []).append(item)
    reversals = pairs = monotone = 0
    errors: list[float] = []
    correlations: list[float] = []
    input_steps: list[float] = []
    response_steps: list[float] = []
    input_tortuosity: list[float] = []
    response_tortuosity: list[float] = []
    for group in by_style.values():
        group.sort(key=lambda item: item.assigned_a1)
        assigned = np.asarray([item.assigned_a1 for item in group])
        realized = np.asarray([item.realized_a1 for item in group])
        changes = np.diff(realized)
        reversals += int(np.sum(changes <= 0))
        pairs += len(changes)
        monotone += int(np.all(changes > 0))
        errors.extend(abs(realized - assigned))
        correlations.append(_spearman(assigned, realized))
        for embedding_map, step_sink, tortuosity_sink in (
            (input_embeddings, input_steps, input_tortuosity),
            (response_embeddings, response_steps, response_tortuosity),
        ):
            matrix = np.asarray([embedding_map[item.candidate_id] for item in group])
            matrix = _unit_rows(matrix)
            steps = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
            step_sink.extend(float(value) for value in steps)
            direct = float(np.linalg.norm(matrix[-1] - matrix[0]))
            tortuosity_sink.append(float(np.sum(steps) / max(direct, 1e-12)))
    lexical = [
        _jaccard(left.search_objective_clause, right.search_objective_clause)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ]
    return A1ManifoldDiagnostics(
        selected_count=len(selected),
        style_count=len(by_style),
        levels_per_style=len(selected) // len(by_style),
        exact_query_structural_retention_rate=sum(
            item.search_term in _query_bound(item.prompt_template, item.search_term)
            for item in selected
        ) / len(selected),
        duplicate_hash_count=len(selected) - len({item.candidate_hash for item in selected}),
        fully_strict_monotone_style_rate=monotone / len(by_style),
        adjacent_reversal_rate=reversals / pairs,
        mean_realized_a1_absolute_error=float(np.mean(errors)),
        mean_style_spearman=float(np.mean(correlations)),
        input_mean_adjacent_distance=float(np.mean(input_steps)),
        input_adjacent_distance_cv=float(np.std(input_steps) / max(np.mean(input_steps), 1e-12)),
        input_mean_tortuosity=float(np.mean(input_tortuosity)),
        response_mean_adjacent_distance=float(np.mean(response_steps)),
        response_adjacent_distance_cv=float(np.std(response_steps) / max(np.mean(response_steps), 1e-12)),
        response_mean_tortuosity=float(np.mean(response_tortuosity)),
        mean_pairwise_lexical_similarity=float(np.mean(lexical)) if lexical else 0.0,
    )


class FakeA1CandidateGenerator:
    backend_name = "fake-a1-generator"
    model_name = "fake-a1-generator-v1"
    clauses = (
        "Understand the category mechanisms, use cases, limitations, and concepts without evaluating or selecting a product.",
        "Build category understanding and identify possible approaches before evaluating any product.",
        "Understand the category and organize practical criteria for a later evaluation.",
        "Combine category understanding with practical evaluation criteria and possible solution approaches.",
        "Evaluate solution approaches and develop a preliminary B2B software shortlist.",
        "Compare and shortlist B2B software products in preparation for acquisition or implementation.",
        "Actively evaluate, compare, shortlist, trial, acquire, or implement a suitable B2B SaaS solution.",
    )
    suffixes = (
        "",
        " Preserve the organizational evaluator perspective.",
        " Keep the request focused on the evaluator's decision stage.",
        " Express this objective in direct business language.",
        " Maintain one coherent search purpose.",
        " Keep the objective concise and operational.",
        " Frame the request for an organizational software decision.",
        " Retain a neutral professional tone.",
        " State the purpose as a single search task.",
        " Keep the evaluator's current readiness explicit.",
        " Use a practical B2B evaluation perspective.",
        " Preserve the intended decision stage without adding criteria.",
    )

    def generate(self, request: A1CandidateRequest) -> tuple[str, ...]:
        index = int(round(request.assigned_a1 * 6))
        return tuple(
            self.clauses[index] + self.suffixes[slot % len(self.suffixes)]
            for slot in range(request.number_candidates)
        )


class FakeA1PairwiseJudge:
    def __init__(self, judge_id: str) -> None:
        self.judge_id = judge_id

    def compare(self, request, candidates) -> str:
        left = candidates[request.left_candidate_id]
        right = candidates[request.right_candidate_id]
        if math.isclose(left.assigned_a1, right.assigned_a1):
            return "tie"
        return left.candidate_id if left.assigned_a1 > right.assigned_a1 else right.candidate_id


class FakeA1Embedder:
    def __init__(self, model_name: str, *, response: bool = False) -> None:
        self.model_name = model_name
        self.response = response

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            level = next(
                (index / 6 for index, clause in enumerate(FakeA1CandidateGenerator.clauses) if clause in text),
                0.5,
            )
            digest = hashlib.sha256(text.encode()).digest()
            rows.append([level, level**2 if self.response else level * 0.5, digest[0] / 255])
        return np.asarray(rows, dtype=np.float64)


class LocalLLMA1CandidateGenerator:
    backend_name = "local-llm-a1-objectives"

    def __init__(
        self,
        ranker,
        *,
        model_name: str,
        cache_directory: str | Path,
        temperature: float = 0.9,
        max_new_tokens: int = 500,
        maximum_attempts: int = 8,
    ) -> None:
        self._ranker = ranker
        self.model_name = model_name
        self.cache_directory = Path(cache_directory)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.maximum_attempts = maximum_attempts

    @classmethod
    def from_model(cls, model_name: str, **kwargs):
        from ..utils import make_ranker

        return cls(make_ranker("local", model_name, precision=kwargs.pop("precision", "full")), model_name=model_name, **kwargs)

    def generate(self, request: A1CandidateRequest) -> tuple[str, ...]:
        rows: list[str] = []
        for slot in range(request.number_candidates):
            identity = {"version": A1_MANIFOLD_VERSION, "request": asdict(request), "slot": slot}
            key = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
            path = self.cache_directory / f"{key}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("identity") != identity:
                    raise ValueError(f"cached A1 candidate identity mismatch: {path}")
                objective = _one_line(payload.get("objective", ""))
                failures = a1_contract_checks(
                    _compile_prompt(
                        TemplatePromptGenerator._build_style_plan(request.style_seed),
                        objective,
                    ),
                    objective,
                    assigned_a1=request.assigned_a1,
                    search_term="__query_sentinel__",
                )
                if failures:
                    raise ValueError(
                        f"cached A1 candidate failed current contract: {path}: "
                        + ", ".join(failures)
                    )
                if objective in rows:
                    raise ValueError(f"cached A1 candidate duplicates its cell: {path}")
                rows.append(objective)
                continue
            errors = []
            prompt = _generation_prompt(request, slot)
            for attempt in range(self.maximum_attempts):
                raw = _seeded_generate(
                    self._ranker,
                    prompt,
                    seed=request.generation_seed + slot * 1009 + attempt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                try:
                    objective = _parse_objective(raw)
                    if objective in rows:
                        raise ValueError("duplicate objective within the style trajectory cell")
                    failures = a1_contract_checks(
                        _compile_prompt(TemplatePromptGenerator._build_style_plan(request.style_seed), objective),
                        objective,
                        assigned_a1=request.assigned_a1,
                        search_term="__query_sentinel__",
                    )
                    if failures:
                        raise ValueError(", ".join(failures))
                except ValueError as exc:
                    errors.append({"attempt": attempt + 1, "error": str(exc), "raw": raw})
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"identity": identity, "objective": objective, "rejected": errors}, indent=2) + "\n")
                rows.append(objective)
                break
            else:
                failure = path.with_suffix(".failed.json")
                failure.parent.mkdir(parents=True, exist_ok=True)
                failure.write_text(json.dumps({"identity": identity, "rejected": errors}, indent=2) + "\n")
                raise ValueError(f"A1 generator exhausted retries for slot {slot}; {failure}")
        return tuple(rows)


class LocalLLMA1PairwiseJudge:
    def __init__(self, ranker, *, judge_id: str, model_name: str, cache_directory: str | Path, max_new_tokens: int = 80) -> None:
        self._ranker = ranker
        self.judge_id = judge_id
        self.model_name = model_name
        self.cache_directory = Path(cache_directory)
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_model(cls, model_name: str, **kwargs):
        from ..utils import make_ranker

        return cls(make_ranker("local", model_name, precision=kwargs.pop("precision", "full")), model_name=model_name, **kwargs)

    def compare(self, request, candidates) -> str:
        identity = {"version": A1_MANIFOLD_VERSION, "request": asdict(request), "judge": self.judge_id, "model": self.model_name}
        key = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        path = self.cache_directory / f"{key}.json"
        if path.exists():
            label = json.loads(path.read_text())["winner"]
        else:
            raw = self._ranker.rank(
                _judge_prompt(request, candidates[request.left_candidate_id], candidates[request.right_candidate_id]),
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
            label = _parse_winner(str(raw))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"identity": identity, "winner": label, "raw": raw}, indent=2) + "\n")
        if label == "tie":
            return "tie"
        return request.left_candidate_id if label == "left" else request.right_candidate_id


def _select_style_path(
    candidates,
    calibrations,
    input_map,
    response_map,
    *,
    forbidden_hashes,
    prior_texts,
    calibration_weight,
    smoothness_weight,
    curvature_weight,
    diversity_weight,
    minimum_realized_step,
):
    levels = sorted({item.assigned_a1 for item in candidates})
    cells = [
        [item for item in candidates if item.assigned_a1 == level and item.candidate_hash not in forbidden_hashes]
        for level in levels
    ]
    if any(not cell for cell in cells):
        raise ValueError("A1 trajectory has an empty candidate cell")
    normalized_input = {key: _unit(value) for key, value in input_map.items()}
    normalized_response = {key: _unit(value) for key, value in response_map.items()}
    input_target = _endpoint_target(cells, normalized_input)
    response_target = _endpoint_target(cells, normalized_response)

    def node(item):
        value = calibration_weight * (calibrations[item.candidate_id].realized_a1 - item.assigned_a1) ** 2
        if prior_texts:
            value += diversity_weight * max(_jaccard(item.search_objective_clause, text) for text in prior_texts) ** 2
        return value

    def edge(left, right):
        if calibrations[right.candidate_id].realized_a1 <= calibrations[left.candidate_id].realized_a1 + minimum_realized_step:
            return math.inf
        distance1 = np.linalg.norm(normalized_input[right.candidate_id] - normalized_input[left.candidate_id])
        distance2 = np.linalg.norm(normalized_response[right.candidate_id] - normalized_response[left.candidate_id])
        lexical_similarity = _jaccard(left.search_objective_clause, right.search_objective_clause)
        lexical_discontinuity = (1.0 - lexical_similarity) ** 2
        return smoothness_weight * (
            (distance1 - input_target) ** 2
            + (distance2 - response_target) ** 2
            + 0.1 * lexical_discontinuity
        )

    if len(cells) == 1:
        chosen = min(cells[0], key=node)
        return (chosen,), node(chosen)
    state: dict[tuple[str, str], tuple[float, tuple[A1Candidate, ...]]] = {}
    for left in cells[0]:
        for right in cells[1]:
            transition = edge(left, right)
            if math.isfinite(transition):
                state[(left.candidate_id, right.candidate_id)] = (node(left) + node(right) + transition, (left, right))
    for cell in cells[2:]:
        new_state = {}
        for (_, previous_id), (score, path) in state.items():
            before, previous = path[-2], path[-1]
            for current in cell:
                transition = edge(previous, current)
                if not math.isfinite(transition):
                    continue
                curve = _curvature(before, previous, current, normalized_input) + _curvature(before, previous, current, normalized_response)
                candidate_score = score + node(current) + transition + curvature_weight * curve
                key = (previous_id, current.candidate_id)
                if key not in new_state or candidate_score < new_state[key][0]:
                    new_state[key] = (candidate_score, path + (current,))
        state = new_state
    if not state:
        raise ValueError("no strictly monotone A1 trajectory exists in the candidate bank")
    best_score, best_path = min(state.values(), key=lambda item: item[0])
    return best_path, float(best_score)


def _compile_prompt(style: StylePlan, objective: str) -> str:
    verb = style.ranking_verb.lower()
    task = (
        f"{style.ranking_verb} the supplied candidates for the exact search term by relevance."
        if style.syntax == "imperative"
        else f"Please {verb} the supplied candidates for the exact search term by relevance."
        if style.syntax == "request"
        else f"Your task is to {verb} the supplied candidates for the exact search term by relevance."
    )
    fixed = f"Act as {BUSINESS_ACTOR}. Keep that actor, the search term, candidate set, ranking task, source policy, and output contract fixed."
    semantic = f"Search objective: {objective}\nSource policy: {NEUTRAL_SOURCE_CLAUSE}"
    inputs = 'Search term: "{QUERY}"\n\nCandidates:\n{CANDIDATES}'
    blocks = (fixed, task, semantic, OUTPUT_CONTRACT, inputs)
    if style.clause_order == "inputs_first":
        blocks = (inputs, fixed, task, semantic, OUTPUT_CONTRACT)
    return _normalize_template("\n\n".join(blocks))


def _generation_prompt(request: A1CandidateRequest, slot: int) -> str:
    anchors = (
        "Understand mechanisms, use cases, limitations, and concepts; explicitly avoid product evaluation or selection."
        if request.assigned_a1 == 0
        else "Actively evaluate, compare, shortlist, trial, acquire, or implement a B2B SaaS solution."
        if request.assigned_a1 == 1
        else f"Express decision readiness near {request.assigned_a1:.3f}: progressively move from category understanding toward practical evaluation and acquisition, without jumping to an endpoint."
    )
    return f"""Write one search-objective clause for a business software evaluator.

Assigned A1 is {request.assigned_a1:.6f}, where 0 means understand a B2B SaaS category and 1 means select/acquire/implement a solution.
Required meaning: {anchors}
Candidate slot {slot + 1} of {request.number_candidates}; use a distinct natural phrasing.

Do not mention A1 or any number. Do not mention a query, specific product, source ownership, publisher, candidate count, output format, freshness, popularity, authority, price, company size, or geography. Do not answer a query. Return JSON only:
{{"search_objective_clause":"..."}}
"""


def _judge_prompt(request, left, right) -> str:
    return f"""Compare two search-agent instructions only for decision readiness.

Which prompt more strongly indicates that a business software evaluator is ready to evaluate, select, acquire, or implement a B2B SaaS solution rather than merely understand the category?
Ignore wording quality, length, syntax, and every other criterion.

LEFT:\n{_query_bound(left.prompt_template, left.search_term)}

RIGHT:\n{_query_bound(right.prompt_template, right.search_term)}

Return JSON only: {{"winner":"left"}}, {{"winner":"right"}}, or {{"winner":"tie"}}.
"""


def _comparison_pair(style_seed, first, second, kind):
    rows = []
    for order, left, right in (("forward", first, second), ("reverse", second, first)):
        identity = f"{style_seed}:{left.candidate_id}:{right.candidate_id}:{order}:{kind}"
        rows.append(
            A1ComparisonRequest(
                comparison_id=f"a1-comparison:{_hash(identity)[:24]}",
                style_seed=style_seed,
                left_candidate_id=left.candidate_id,
                right_candidate_id=right.candidate_id,
                presentation_order=order,
                comparison_kind=kind,
                question="decision-readiness",
            )
        )
    return rows


def _fit_bradley_terry(ids, requests, outcomes, regularization):
    index = {value: position for position, value in enumerate(ids)}
    pairs = []
    for request in requests:
        for judgment in outcomes.get(request.comparison_id, ()):
            left, right = index[request.left_candidate_id], index[request.right_candidate_id]
            outcome = 0.5 if judgment.is_tie else float(judgment.winner_candidate_id == request.left_candidate_id)
            pairs.append((left, right, outcome))
    if not pairs:
        raise ValueError("A1 calibration has no pairwise judgments")

    def objective(theta):
        value = 0.5 * regularization * float(theta @ theta)
        gradient = regularization * theta
        for left, right, outcome in pairs:
            delta = float(np.clip(theta[left] - theta[right], -30, 30))
            probability = 1 / (1 + math.exp(-delta))
            value -= outcome * math.log(max(probability, 1e-12)) + (1 - outcome) * math.log(max(1 - probability, 1e-12))
            gradient[left] += probability - outcome
            gradient[right] -= probability - outcome
        return value, gradient

    result = minimize(lambda x: objective(x)[0], np.zeros(len(ids)), jac=lambda x: objective(x)[1], method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"A1 Bradley-Terry fit failed: {result.message}")
    centered = result.x - np.mean(result.x)
    return {value: float(centered[position]) for value, position in index.items()}


def _embedding_map(rows, representation):
    selected = {item.candidate_id: np.asarray(item.values, dtype=np.float64) for item in rows if item.representation == representation}
    if len(selected) != len(rows):
        raise ValueError(f"embedding artifact contains mixed or duplicate {representation} rows")
    return selected


def _endpoint_target(cells, embeddings):
    low = np.mean([embeddings[item.candidate_id] for item in cells[0]], axis=0)
    high = np.mean([embeddings[item.candidate_id] for item in cells[-1]], axis=0)
    return float(np.linalg.norm(high - low) / max(len(cells) - 1, 1))


def _curvature(before, middle, after, embeddings):
    first = embeddings[middle.candidate_id] - embeddings[before.candidate_id]
    second = embeddings[after.candidate_id] - embeddings[middle.candidate_id]
    lengths = abs(np.linalg.norm(first) - np.linalg.norm(second)) ** 2
    cosine = float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12))
    return float(lengths + (1 - cosine))


def _parse_objective(raw):
    values = _json_values(raw)
    payloads = [value for value in values if isinstance(value, dict) and set(value) == {"search_objective_clause"}]
    if len(payloads) != 1 or not isinstance(payloads[0]["search_objective_clause"], str):
        raise ValueError("generator must return one objective JSON object")
    objective = _one_line(payloads[0]["search_objective_clause"])
    if not objective:
        raise ValueError("generated objective is empty")
    return objective


def _parse_winner(raw):
    payloads = [value for value in _json_values(raw) if isinstance(value, dict) and set(value) == {"winner"}]
    if len(payloads) != 1 or payloads[0]["winner"] not in {"left", "right", "tie"}:
        raise ValueError("judge must return one valid winner JSON object")
    return payloads[0]["winner"]


def _json_values(raw):
    decoder = json.JSONDecoder()
    values = []
    end = -1
    for index, character in enumerate(raw):
        if index < end or character not in "[{":
            continue
        try:
            value, relative = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
        end = index + relative
    return values


def _seeded_generate(ranker, prompt, *, seed, max_new_tokens, temperature):
    import torch

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return str(ranker.rank(prompt, max_tokens=max_new_tokens, temperature=temperature, chat_template_kwargs={"enable_thinking": False}))


def _query_bound(template, query):
    return template.replace("{QUERY}", query).replace("{CANDIDATES}", "[FROZEN CANDIDATE SET]").replace("{TOP_N}", "10")


def _jaccard(left, right):
    a = set(re.findall(r"[a-z0-9]+", left.casefold()))
    b = set(re.findall(r"[a-z0-9]+", right.casefold()))
    return len(a & b) / max(len(a | b), 1)


def _spearman(left, right):
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def _unit(value):
    array = np.asarray(value, dtype=np.float64)
    return array / max(np.linalg.norm(array), 1e-12)


def _unit_rows(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def _one_line(value):
    if not isinstance(value, str):
        raise TypeError("text fields must be strings")
    return " ".join(value.split())


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A1 coordinate must be finite and in [0, 1]")


def _seed(master, style, level):
    return int.from_bytes(hashlib.sha256(f"{master}:{style}:{level}".encode()).digest()[:4], "big")


def _hash(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
