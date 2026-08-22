"""Iterative natural-question coverage of a frozen readiness subspace.

Generator models propose text.  A frozen LLM2Vec model and supervised map then
measure where that text landed.  The measured coordinates are diagnostics; they
do not define the experimental policy variable B.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

import numpy as np

from .readiness_embedding_map import ReadinessEmbeddingMap


READINESS_PROMPT_POPULATION_VERSION = "readiness-question-population-v2"


class QuestionGenerationExhaustedError(RuntimeError):
    """All deterministic validation attempts for one generation task failed."""


@dataclass(frozen=True, slots=True)
class ReadinessSubspaceBounds:
    axis_1_low: float
    axis_1_high: float
    axis_2_low: float
    axis_2_high: float
    lower_quantile: float
    upper_quantile: float
    reference_split: str
    reference_item_count: int


@dataclass(frozen=True, slots=True)
class ReadinessPromptTarget:
    target_id: str
    target_index: int
    axis_1_index: int
    axis_2_index: int
    normalized_axis_1: float
    normalized_axis_2: float
    raw_axis_1: float
    raw_axis_2: float


@dataclass(frozen=True, slots=True)
class ReadinessGenerationTask:
    task_id: str
    keyword_id: str
    keyword: str
    target: ReadinessPromptTarget
    round_index: int
    generator_id: str
    generation_seed: int
    requested_candidate_count: int
    feedback: str


@dataclass(frozen=True, slots=True)
class ReadinessQuestionCandidate:
    candidate_id: str
    task_id: str
    keyword_id: str
    keyword: str
    target_id: str
    target_index: int
    target_normalized_axis_1: float
    target_normalized_axis_2: float
    target_raw_axis_1: float
    target_raw_axis_2: float
    round_index: int
    generator_id: str
    generator_model: str
    candidate_slot: int
    generation_seed: int
    question: str
    question_sha256: str
    proposal_kind: str


@dataclass(frozen=True, slots=True)
class ProjectedReadinessQuestion:
    candidate_id: str
    raw_axis_1: float
    raw_axis_2: float
    normalized_axis_1: float
    normalized_axis_2: float
    predicted_scalar_readiness_0_1: float
    target_distance: float


@dataclass(frozen=True, slots=True)
class ReadinessTextProjection:
    item_id: str
    text_sha256: str
    raw_axis_1: float
    raw_axis_2: float
    normalized_axis_1: float
    normalized_axis_2: float
    predicted_scalar_readiness_0_1: float


@dataclass(frozen=True, slots=True)
class SelectedReadinessQuestion:
    selection_id: str
    keyword_id: str
    keyword: str
    target_id: str
    target_index: int
    candidate_id: str
    question: str
    generator_id: str
    generator_model: str
    round_index: int
    target_normalized_axis_1: float
    target_normalized_axis_2: float
    observed_normalized_axis_1: float
    observed_normalized_axis_2: float
    observed_raw_axis_1: float
    observed_raw_axis_2: float
    predicted_scalar_readiness_0_1: float
    target_distance: float
    maximum_similarity_to_previously_selected: float
    selection_objective: float


@dataclass(frozen=True, slots=True)
class SearchQuestionReview:
    candidate_id: str
    judge_id: str
    judge_model: str
    exact_keyword_present: bool
    single_question: bool
    topic_relevant: bool
    search_intent: bool
    web_answerable: bool
    standalone: bool
    natural_language: bool
    relevance_score_1_5: int
    accepted: bool
    concise_reason: str


@dataclass(frozen=True, slots=True)
class SpatiallySelectedReadinessQuestion:
    keyword_id: str
    keyword: str
    target_id: str
    target_index: int
    candidate_id: str
    question: str
    generator_id: str
    generator_model: str
    target_normalized_axis_1: float
    target_normalized_axis_2: float
    consensus_normalized_axis_1: float
    consensus_normalized_axis_2: float
    reference_normalized_axis_1: float
    reference_normalized_axis_2: float
    candidate_aligned_normalized_axis_1: float
    candidate_aligned_normalized_axis_2: float
    target_distance: float
    reference_target_distance: float
    candidate_aligned_target_distance: float
    both_views_within_tolerance: bool
    cross_embedding_disagreement: float
    assignment_cost: float


class ReadinessQuestionGenerator(Protocol):
    generator_id: str
    model_name: str
    proposal_kind: str

    def generate(
        self, task: ReadinessGenerationTask
    ) -> tuple[str, ...]: ...


class SearchQuestionValidator(Protocol):
    judge_id: str
    model_name: str

    def review(
        self, candidate: ReadinessQuestionCandidate
    ) -> SearchQuestionReview: ...


class FakeReadinessQuestionGenerator:
    """Deterministic plumbing generator.  Its output is not scientific data."""

    proposal_kind = "fake"

    def __init__(self, generator_id: str = "fake-generator") -> None:
        self.generator_id = generator_id
        self.model_name = "fake-readiness-question-generator-v1"

    def generate(self, task: ReadinessGenerationTask) -> tuple[str, ...]:
        a1 = task.target.normalized_axis_1
        a2 = task.target.normalized_axis_2
        stage = (
            "understand the main concepts and evidence"
            if a1 < 0.34
            else "compare the practical options and trade-offs"
            if a1 < 0.67
            else "carry out the next concrete step"
        )
        mode = "choose an approach" if a2 < 0.5 else "implement an approach"
        return tuple(
            f"How can a team {stage} for {task.keyword} and {mode} in scenario {slot + 1}?"
            for slot in range(task.requested_candidate_count)
        )


class LocalReadinessQuestionGenerator:
    """Strict JSON question generator backed by a repository ranker."""

    proposal_kind = "causal-lm"

    def __init__(
        self,
        ranker,
        *,
        generator_id: str,
        model_name: str,
        cache_directory: str | Path,
        max_new_tokens: int = 180,
        temperature: float = 0.9,
        maximum_attempts: int = 5,
    ) -> None:
        if not generator_id.strip() or not model_name.strip():
            raise ValueError("generator_id and model_name must be nonempty")
        if max_new_tokens <= 0 or temperature < 0 or maximum_attempts <= 0:
            raise ValueError("invalid generator configuration")
        self._ranker = ranker
        self.generator_id = generator_id
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
        generator_id: str,
        cache_directory: str | Path,
        backend: str = "local",
        precision: str = "full",
        max_new_tokens: int = 180,
        temperature: float = 0.9,
        maximum_attempts: int = 5,
    ) -> "LocalReadinessQuestionGenerator":
        from ..utils import make_ranker

        return cls(
            make_ranker(backend, model_name, precision=precision),
            generator_id=generator_id,
            model_name=model_name,
            cache_directory=cache_directory,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            maximum_attempts=maximum_attempts,
        )

    def generate(self, task: ReadinessGenerationTask) -> tuple[str, ...]:
        identity = {
            "version": READINESS_PROMPT_POPULATION_VERSION,
            "model": self.model_name,
            "generator_id": self.generator_id,
            "task": asdict(task),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
        cache_key = _stable_hash(identity)
        cache_path = self.cache_directory / f"{cache_key}.json"
        accepted: list[str] = []
        failures: list[dict[str, object]] = []
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("identity") != identity:
                raise ValueError(f"generation cache identity mismatch: {cache_path}")
            accepted = [str(value) for value in payload.get("questions", [])]
            failures = list(payload.get("failures", []))
            if len(accepted) > task.requested_candidate_count:
                raise ValueError(f"generation cache has too many questions: {cache_path}")
            for question in accepted:
                validate_generated_question(question, task.keyword)
            if len(set(accepted)) != len(accepted):
                raise ValueError(f"generation cache contains duplicate questions: {cache_path}")
            if len(accepted) == task.requested_candidate_count:
                return tuple(accepted)

        for slot in range(len(accepted), task.requested_candidate_count):
            for attempt in range(self.maximum_attempts):
                seed = task.generation_seed + slot * 1009 + attempt
                request = render_generation_request(task, candidate_slot=slot)
                if attempt:
                    request += (
                        "\nYour previous output was invalid. Return only the required "
                        "one-object JSON, with a new question."
                    )
                raw = _generate_with_seed(
                    self._ranker,
                    request,
                    seed=seed,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                try:
                    question = parse_generated_question(raw)
                    validate_generated_question(question, task.keyword)
                    if question in accepted:
                        raise ValueError("duplicate question within task")
                except ValueError as exc:
                    failures.append(
                        {"slot": slot, "attempt": attempt, "error": str(exc), "raw": raw}
                    )
                    _atomic_json(
                        cache_path,
                        {
                            "identity": identity,
                            "questions": accepted,
                            "failures": failures,
                            "complete": False,
                            "terminal_failure": False,
                        },
                    )
                    continue
                accepted.append(question)
                _atomic_json(
                    cache_path,
                    {
                        "identity": identity,
                        "questions": accepted,
                        "failures": failures,
                        "complete": False,
                    },
                )
                break
            else:
                terminal_error = (
                    str(failures[-1]["error"]) if failures else "unknown error"
                )
                _atomic_json(
                    cache_path,
                    {
                        "identity": identity,
                        "questions": accepted,
                        "failures": failures,
                        "complete": False,
                        "terminal_failure": True,
                        "terminal_slot": slot,
                        "terminal_error": terminal_error,
                    },
                )
                raise QuestionGenerationExhaustedError(
                    f"question generation failed for {task.task_id} slot {slot}: "
                    f"{terminal_error}"
                )
        _atomic_json(
            cache_path,
            {
                "identity": identity,
                "questions": accepted,
                "failures": failures,
                "complete": True,
                "terminal_failure": False,
            },
        )
        return tuple(accepted)


class LocalSearchQuestionValidator:
    """Independent cached review of topic fidelity and online-search utility."""

    def __init__(
        self,
        ranker,
        *,
        judge_id: str,
        model_name: str,
        cache_directory: str | Path,
        maximum_attempts: int = 3,
    ) -> None:
        if not judge_id.strip() or not model_name.strip() or maximum_attempts <= 0:
            raise ValueError("invalid search-question validator configuration")
        self._ranker = ranker
        self.judge_id = judge_id
        self.model_name = model_name
        self.cache_directory = Path(cache_directory)
        self.maximum_attempts = maximum_attempts

    @classmethod
    def from_model(
        cls,
        model_name: str,
        *,
        judge_id: str,
        cache_directory: str | Path,
        backend: str = "local",
        precision: str = "full",
        maximum_attempts: int = 3,
    ) -> "LocalSearchQuestionValidator":
        from ..utils import make_ranker

        return cls(
            make_ranker(backend, model_name, precision=precision),
            judge_id=judge_id,
            model_name=model_name,
            cache_directory=cache_directory,
            maximum_attempts=maximum_attempts,
        )

    def review(self, candidate: ReadinessQuestionCandidate) -> SearchQuestionReview:
        identity = {
            "version": READINESS_PROMPT_POPULATION_VERSION,
            "judge_id": self.judge_id,
            "judge_model": self.model_name,
            "candidate_id": candidate.candidate_id,
            "question_sha256": candidate.question_sha256,
        }
        cache_path = self.cache_directory / f"{_stable_hash(identity)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("terminal_parse_failure"):
                for failure in reversed(cached.get("failures", ())):
                    try:
                        recovered = parse_search_question_review(
                            str(failure.get("raw", "")),
                            candidate,
                            judge_id=self.judge_id,
                            judge_model=self.model_name,
                        )
                    except ValueError:
                        continue
                    _atomic_json(
                        cache_path,
                        {
                            "identity": identity,
                            "review": asdict(recovered),
                            "failures": cached.get("failures", []),
                            "recovered_terminal_parse_failure": True,
                        },
                    )
                    return recovered
            return SearchQuestionReview(**cached["review"])
        failures = []
        for attempt in range(self.maximum_attempts):
            raw = _generate_with_seed(
                self._ranker,
                render_search_validation_request(candidate),
                seed=candidate.generation_seed + 900_001 + attempt,
                max_new_tokens=220,
                temperature=0.0,
            )
            try:
                review = parse_search_question_review(
                    raw,
                    candidate,
                    judge_id=self.judge_id,
                    judge_model=self.model_name,
                )
            except ValueError as exc:
                failures.append({"attempt": attempt, "error": str(exc), "raw": raw})
                continue
            _atomic_json(
                cache_path,
                {"identity": identity, "review": asdict(review), "failures": failures},
            )
            return review
        review = SearchQuestionReview(
            candidate_id=candidate.candidate_id,
            judge_id=self.judge_id,
            judge_model=self.model_name,
            exact_keyword_present=candidate.keyword in candidate.question,
            single_question=(
                candidate.question.endswith("?") and candidate.question.count("?") == 1
            ),
            topic_relevant=False,
            search_intent=False,
            web_answerable=False,
            standalone=False,
            natural_language=False,
            relevance_score_1_5=1,
            accepted=False,
            concise_reason=(
                f"Validator output remained invalid after {self.maximum_attempts} attempts."
            ),
        )
        _atomic_json(
            cache_path,
            {
                "identity": identity,
                "review": asdict(review),
                "failures": failures,
                "terminal_parse_failure": True,
            },
        )
        return review


def load_readiness_embedding_map(path: str | Path) -> ReadinessEmbeddingMap:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readiness map must contain one JSON object")
    fitted = ReadinessEmbeddingMap(**payload)
    if fitted.map_version != "llm2vec-readiness-map-v3":
        raise ValueError(f"unsupported readiness map version: {fitted.map_version}")
    return fitted


def fit_reference_bounds(
    coordinate_rows: Sequence[Mapping[str, object]],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    reference_split: str = "development",
) -> ReadinessSubspaceBounds:
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("reference quantiles must satisfy 0 <= lower < upper <= 1")
    selected = [row for row in coordinate_rows if row.get("split") == reference_split]
    if len(selected) < 10:
        raise ValueError(f"at least ten {reference_split} coordinate rows are required")
    axis_1 = np.asarray([float(row["axis_1"]) for row in selected])
    axis_2 = np.asarray([float(row["axis_2"]) for row in selected])
    if not np.isfinite(axis_1).all() or not np.isfinite(axis_2).all():
        raise ValueError("reference coordinates must be finite")
    values = np.quantile(
        np.column_stack((axis_1, axis_2)),
        (lower_quantile, upper_quantile),
        axis=0,
    )
    if np.any(values[1] - values[0] <= 1e-12):
        raise ValueError("reference coordinates do not span both axes")
    return ReadinessSubspaceBounds(
        axis_1_low=float(values[0, 0]),
        axis_1_high=float(values[1, 0]),
        axis_2_low=float(values[0, 1]),
        axis_2_high=float(values[1, 1]),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_split=reference_split,
        reference_item_count=len(selected),
    )


def build_target_grid(
    bounds: ReadinessSubspaceBounds,
    *,
    axis_1_points: int = 6,
    axis_2_points: int = 5,
) -> tuple[ReadinessPromptTarget, ...]:
    if axis_1_points < 2 or axis_2_points < 2:
        raise ValueError("each target-grid axis requires at least two points")
    rows: list[ReadinessPromptTarget] = []
    index = 0
    for axis_1_index, normalized_axis_1 in enumerate(np.linspace(0.0, 1.0, axis_1_points)):
        axis_2_values = list(enumerate(np.linspace(0.0, 1.0, axis_2_points)))
        if axis_1_index % 2:
            axis_2_values.reverse()
        for axis_2_index, normalized_axis_2 in axis_2_values:
            rows.append(
                ReadinessPromptTarget(
                    target_id=f"readiness-cell:{axis_1_index:02d}-{axis_2_index:02d}",
                    target_index=index,
                    axis_1_index=axis_1_index,
                    axis_2_index=axis_2_index,
                    normalized_axis_1=float(normalized_axis_1),
                    normalized_axis_2=float(normalized_axis_2),
                    raw_axis_1=_denormalize(float(normalized_axis_1), bounds.axis_1_low, bounds.axis_1_high),
                    raw_axis_2=_denormalize(float(normalized_axis_2), bounds.axis_2_low, bounds.axis_2_high),
                )
            )
            index += 1
    return tuple(rows)


def build_support_aware_keyword_targets(
    coordinate_rows: Sequence[Mapping[str, object]],
    bounds: ReadinessSubspaceBounds,
    keywords: Sequence[tuple[str, str]],
    *,
    targets_per_keyword: int = 30,
    support_grid_resolution: int = 20,
    minimum_support_bin_count: int = 3,
    master_seed: int = 20260820,
    require_usable_for_axis: bool = True,
) -> tuple[
    dict[str, tuple[ReadinessPromptTarget, ...]],
    dict[str, object],
]:
    """Build deterministic area-balanced targets over empirical development support.

    Targets are balanced over occupied support cells rather than the empirical
    prompt density.  Coordinates are interpolated only between development
    points in the same cell, so seeded variation remains locally supported.
    """

    if not keywords or targets_per_keyword <= 0:
        raise ValueError("keywords and targets_per_keyword must be nonempty")
    if support_grid_resolution < 2 or minimum_support_bin_count <= 0:
        raise ValueError("invalid support-grid configuration")
    keyword_ids = [keyword_id for keyword_id, _ in keywords]
    if len(set(keyword_ids)) != len(keyword_ids):
        raise ValueError("support-aware keyword ids must be unique")

    support_rows = []
    for row in coordinate_rows:
        if row.get("split") != bounds.reference_split:
            continue
        if require_usable_for_axis and not bool(row.get("usable_for_axis", False)):
            continue
        values = np.asarray(
            [
                _normalize(float(row["axis_1"]), bounds.axis_1_low, bounds.axis_1_high),
                _normalize(float(row["axis_2"]), bounds.axis_2_low, bounds.axis_2_high),
            ],
            dtype=np.float64,
        )
        if np.isfinite(values).all() and np.all((values >= 0.0) & (values <= 1.0)):
            support_rows.append(values)
    if len(support_rows) < max(10, targets_per_keyword):
        raise ValueError("insufficient in-bounds development support coordinates")
    support = np.asarray(support_rows, dtype=np.float64)

    bin_members: dict[tuple[int, int], list[int]] = {}
    for index, values in enumerate(support):
        cell = tuple(
            int(np.clip(np.floor(value * support_grid_resolution), 0, support_grid_resolution - 1))
            for value in values
        )
        bin_members.setdefault(cell, []).append(index)
    eligible_cells = tuple(
        sorted(
            cell
            for cell, members in bin_members.items()
            if len(members) >= minimum_support_bin_count
        )
    )
    if not eligible_cells:
        raise ValueError("support design has no eligible support cells")

    rng = np.random.default_rng(master_seed)
    allocation_counts = np.zeros(len(eligible_cells), dtype=np.int64)
    targets_by_keyword: dict[str, tuple[ReadinessPromptTarget, ...]] = {}
    pooled_coordinates = []
    for keyword_id, keyword in keywords:
        if not keyword_id.strip() or not _single_line(keyword):
            raise ValueError("keyword ids and text must be nonempty")
        chosen_indices = []
        available = set(range(len(eligible_cells)))
        while len(chosen_indices) < targets_per_keyword:
            if not available:
                available = set(range(len(eligible_cells)))
            minimum_allocation = min(allocation_counts[index] for index in available)
            least_used = sorted(
                index
                for index in available
                if allocation_counts[index] == minimum_allocation
            )
            cell_index = int(rng.choice(least_used))
            chosen_indices.append(cell_index)
            available.remove(cell_index)
            allocation_counts[cell_index] += 1
        keyword_targets = []
        for target_index, cell_index in enumerate(chosen_indices):
            cell = eligible_cells[cell_index]
            member_indices = bin_members[cell]
            left_index, right_index = rng.choice(member_indices, size=2, replace=True)
            interpolation = float(rng.random())
            normalized = (
                interpolation * support[left_index]
                + (1.0 - interpolation) * support[right_index]
            )
            pooled_coordinates.append(normalized)
            keyword_targets.append(
                ReadinessPromptTarget(
                    target_id=f"readiness-support-target:{target_index:03d}",
                    target_index=target_index,
                    axis_1_index=cell[0],
                    axis_2_index=cell[1],
                    normalized_axis_1=float(normalized[0]),
                    normalized_axis_2=float(normalized[1]),
                    raw_axis_1=_denormalize(
                        float(normalized[0]), bounds.axis_1_low, bounds.axis_1_high
                    ),
                    raw_axis_2=_denormalize(
                        float(normalized[1]), bounds.axis_2_low, bounds.axis_2_high
                    ),
                )
            )
        targets_by_keyword[keyword_id] = tuple(keyword_targets)

    pooled = np.asarray(pooled_coordinates, dtype=np.float64)
    diagnostics = {
        "target_design": "support-aware-random",
        "master_seed": master_seed,
        "reference_split": bounds.reference_split,
        "require_usable_for_axis": require_usable_for_axis,
        "support_grid_resolution": support_grid_resolution,
        "minimum_support_bin_count": minimum_support_bin_count,
        "support_coordinate_count": len(support),
        "eligible_support_bin_count": len(eligible_cells),
        "keyword_count": len(keywords),
        "targets_per_keyword": targets_per_keyword,
        "pooled_target_count": len(pooled),
        "minimum_targets_per_eligible_bin": int(allocation_counts.min()),
        "maximum_targets_per_eligible_bin": int(allocation_counts.max()),
        "target_bin_count_range": int(allocation_counts.max() - allocation_counts.min()),
        "pooled_axis_1_range": [float(pooled[:, 0].min()), float(pooled[:, 0].max())],
        "pooled_axis_2_range": [float(pooled[:, 1].min()), float(pooled[:, 1].max())],
        "scientific_guard": (
            "Targets use development prompt geometry only. They describe generated "
            "question semantics and do not define the randomized policy variable B."
        ),
    }
    return targets_by_keyword, diagnostics


def build_generation_tasks(
    keywords: Sequence[tuple[str, str]],
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
    generator_ids: Sequence[str],
    *,
    round_index: int = 0,
    master_seed: int = 20260820,
    requested_candidate_count: int = 3,
    feedback_by_keyword_target: Mapping[tuple[str, str], str] | None = None,
) -> tuple[ReadinessGenerationTask, ...]:
    if not keywords or not targets or not generator_ids:
        raise ValueError("keywords, targets, and generator_ids must be nonempty")
    if len(set(generator_ids)) != len(generator_ids) or any(not value.strip() for value in generator_ids):
        raise ValueError("generator_ids must be unique and nonempty")
    if round_index < 0 or requested_candidate_count <= 0:
        raise ValueError("invalid round or candidate count")
    feedback_by_keyword_target = feedback_by_keyword_target or {}
    rows = []
    targets_by_keyword = _targets_by_keyword(keywords, targets)
    for keyword_index, (keyword_id, keyword) in enumerate(keywords):
        keyword = _single_line(keyword)
        if not keyword_id.strip() or not keyword:
            raise ValueError("keyword ids and text must be nonempty")
        for target in targets_by_keyword[keyword_id]:
            generator_id = generator_ids[
                (keyword_index + target.target_index + round_index) % len(generator_ids)
            ]
            identity = {
                "version": READINESS_PROMPT_POPULATION_VERSION,
                "keyword_id": keyword_id,
                "keyword": keyword,
                "target_id": target.target_id,
                "round": round_index,
                "generator_id": generator_id,
                "master_seed": master_seed,
            }
            if target.target_id.startswith("readiness-support-target:"):
                identity["support_target"] = asdict(target)
            digest = _stable_hash(identity)
            rows.append(
                ReadinessGenerationTask(
                    task_id=f"readiness-question-task:{digest[:24]}",
                    keyword_id=keyword_id,
                    keyword=keyword,
                    target=target,
                    round_index=round_index,
                    generator_id=generator_id,
                    generation_seed=int(digest[:16], 16) ^ master_seed,
                    requested_candidate_count=requested_candidate_count,
                    feedback=feedback_by_keyword_target.get(
                        (keyword_id, target.target_id), "No earlier measured candidate."
                    ),
                )
            )
    return tuple(rows)


def _targets_by_keyword(
    keywords: Sequence[tuple[str, str]],
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
) -> dict[str, tuple[ReadinessPromptTarget, ...]]:
    keyword_ids = [keyword_id for keyword_id, _ in keywords]
    if isinstance(targets, Mapping):
        if set(targets) != set(keyword_ids):
            raise ValueError("keyword-target mapping does not cover the exact keyword set")
        resolved = {
            keyword_id: tuple(targets[keyword_id]) for keyword_id in keyword_ids
        }
    else:
        shared = tuple(targets)
        resolved = {keyword_id: shared for keyword_id in keyword_ids}
    for keyword_id, keyword_targets in resolved.items():
        if not keyword_targets:
            raise ValueError(f"keyword has no targets: {keyword_id}")
        if len({target.target_id for target in keyword_targets}) != len(keyword_targets):
            raise ValueError(f"keyword target ids are not unique: {keyword_id}")
    return resolved


def render_generation_request(
    task: ReadinessGenerationTask, *, candidate_slot: int
) -> str:
    support_aware = task.target.target_id.startswith("readiness-support-target:")
    if support_aware:
        a1 = _continuous_axis_instruction(
            task.target.normalized_axis_1,
            (
                "purely understand or explain the topic",
                "investigate evidence, mechanisms, or implications",
                "evaluate concrete options or trade-offs",
                "prepare a decision, commitment, or practical plan",
                "request an immediate, concrete action or execution step",
            ),
        )
        a2 = _continuous_axis_instruction(
            task.target.normalized_axis_2,
            (
                "compare alternatives and decide which approach fits",
                "select an approach using explicit criteria",
                "translate a chosen approach into a practical procedure",
                "implement, configure, troubleshoot, or execute a chosen approach",
            ),
        )
        continuous_control = (
            "Treat each percentage as a graded semantic mixture, not a category. "
            "Preserve the difference between nearby targets through the question's "
            "actual information need."
        )
        surface_control = _surface_realization_instruction(
            task.generation_seed + candidate_slot * 1009
        )
    else:
        a1 = _axis_1_instruction(task.target.normalized_axis_1)
        a2 = _axis_2_instruction(task.target.normalized_axis_2)
        continuous_control = "Use the requested semantic destination."
        surface_control = "Make the wording natural and distinct."
    return f"""Write one standalone, natural search question about the exact keyword phrase below.

Exact keyword phrase (must appear verbatim): {task.keyword}

Semantic destination:
- Readiness stage: {a1}
- Decision mode: {a2}
- Control rule: {continuous_control}
- Surface realization: {surface_control}

Iteration feedback: {task.feedback}
Candidate slot: {candidate_slot}

Hard constraints:
- Ask one question that a search-capable LLM could research and answer.
- End with exactly one question mark.
- Use 8 to 60 words and one line.
- Include the exact keyword phrase verbatim.
- Do not answer the question.
- Do not mention axes, coordinates, readiness scores, embeddings, reranking,
  publishers, or source-preference policies.
- Make this candidate materially different from obvious generic phrasings.

Return only JSON: {{"question":"..."}}"""


def _continuous_axis_instruction(value: float, anchors: Sequence[str]) -> str:
    if not 0.0 <= value <= 1.0 or len(anchors) < 2:
        raise ValueError("continuous semantic control requires [0, 1] and two anchors")
    scaled = value * (len(anchors) - 1)
    lower = min(int(np.floor(scaled)), len(anchors) - 2)
    upper = lower + 1
    upper_weight = scaled - lower
    lower_weight = 1.0 - upper_weight
    return (
        f"{value:.3f} on a 0-to-1 continuum: "
        f"{lower_weight:.0%} '{anchors[lower]}' and "
        f"{upper_weight:.0%} '{anchors[upper]}'"
    )


def _surface_realization_instruction(seed: int) -> str:
    variants = (
        "use a concise direct interrogative with ordinary wording",
        "place a short context clause before the main interrogative",
        "state the main information need first and its qualifier near the end",
        "use neutral professional wording without unnecessary jargon",
        "use a natural first-person search question without inventing personal facts",
        "use an impersonal search question with a concrete but generic scope",
        "open with a conditional situation and ask what follows from it",
        "frame the question around evidence needed to resolve uncertainty",
        "ask through a contrast between two plausible approaches without naming brands",
        "use a concise how-or-why construction and avoid stock 'what are the best' wording",
        "frame the information need as a diagnostic question about causes or consequences",
        "ask for criteria that would distinguish a suitable approach from an unsuitable one",
        "use a natural scenario-first question with the core topic near the end",
        "ask what a careful reader should verify before proceeding",
        "use an outcome-first question that asks what would help achieve that outcome",
        "frame the question around a concrete obstacle without inventing personal details",
        "ask for a sequence or procedure only when the semantic destination calls for action",
        "use an uncommon but natural interrogative structure rather than a reusable template",
    )
    return variants[seed % len(variants)]


def delexicalize_question(question: str, keyword: str) -> str:
    """Normalize a question after replacing its topic phrase with one sentinel."""

    if not question.strip() or not keyword.strip():
        raise ValueError("question and keyword must be nonempty")
    replaced, count = re.subn(
        re.escape(keyword), " topicplaceholder ", question, flags=re.IGNORECASE
    )
    if count == 0:
        raise ValueError("question does not contain its keyword")
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", replaced.casefold())
    normalized = ["numberplaceholder" if token.isdigit() else token for token in tokens]
    return " ".join(normalized)


def audit_question_diversity(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_delexicalized_unique_fraction: float = 0.90,
    maximum_template_fraction: float = 0.01,
    minimum_median_keyword_unique_fraction: float = 0.90,
    minimum_keyword_unique_fraction: float = 0.70,
    maximum_opening_frame_fraction: float = 0.05,
    opening_frame_tokens: int = 5,
) -> dict[str, object]:
    """Audit lexical/template diversity without using semantic-map coordinates.

    Replacing the exact keyword before normalization detects the failure mode in
    which one question frame is copied across topics.  Per-keyword summaries
    additionally detect collapse among the targets for one topic.
    """

    thresholds = (
        minimum_delexicalized_unique_fraction,
        maximum_template_fraction,
        minimum_median_keyword_unique_fraction,
        minimum_keyword_unique_fraction,
        maximum_opening_frame_fraction,
    )
    if not rows:
        raise ValueError("diversity audit requires at least one question")
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("diversity thresholds must lie in [0, 1]")
    if opening_frame_tokens <= 0:
        raise ValueError("opening frame token count must be positive")

    exact_questions: list[str] = []
    templates: list[str] = []
    keyword_templates: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        keyword = str(row.get("keyword", "")).strip()
        keyword_id = str(row.get("keyword_id", "")).strip()
        if not question or not keyword or not keyword_id:
            raise ValueError(
                f"diversity row {index} requires question, keyword, and keyword_id"
            )
        template = delexicalize_question(question, keyword)
        exact_questions.append(_single_line(question).casefold())
        templates.append(template)
        keyword_templates.setdefault(keyword_id, []).append(template)

    row_count = len(rows)
    template_counts = Counter(templates)
    opening_counts = Counter(
        " ".join(template.split()[:opening_frame_tokens]) for template in templates
    )
    keyword_unique_fractions = {
        keyword_id: len(set(values)) / len(values)
        for keyword_id, values in sorted(keyword_templates.items())
    }
    delexicalized_unique_fraction = len(template_counts) / row_count
    maximum_observed_template_fraction = max(template_counts.values()) / row_count
    maximum_observed_opening_fraction = max(opening_counts.values()) / row_count
    median_keyword_unique_fraction = float(
        np.median(list(keyword_unique_fractions.values()))
    )
    minimum_observed_keyword_unique_fraction = min(keyword_unique_fractions.values())
    checks = {
        "all_exact_questions_unique": len(set(exact_questions)) == row_count,
        "delexicalized_unique_fraction_at_least_threshold": (
            delexicalized_unique_fraction >= minimum_delexicalized_unique_fraction
        ),
        "largest_delexicalized_template_fraction_at_most_threshold": (
            maximum_observed_template_fraction <= maximum_template_fraction
        ),
        "median_keyword_unique_fraction_at_least_threshold": (
            median_keyword_unique_fraction >= minimum_median_keyword_unique_fraction
        ),
        "minimum_keyword_unique_fraction_at_least_threshold": (
            minimum_observed_keyword_unique_fraction >= minimum_keyword_unique_fraction
        ),
        "largest_opening_frame_fraction_at_most_threshold": (
            maximum_observed_opening_fraction <= maximum_opening_frame_fraction
        ),
    }
    top_templates = [
        {"template": template, "count": count, "fraction": count / row_count}
        for template, count in template_counts.most_common(20)
    ]
    top_opening_frames = [
        {"frame": frame, "count": count, "fraction": count / row_count}
        for frame, count in opening_counts.most_common(20)
    ]
    return {
        "format_version": READINESS_PROMPT_POPULATION_VERSION,
        "row_count": row_count,
        "keyword_count": len(keyword_templates),
        "exact_question_unique_fraction": len(set(exact_questions)) / row_count,
        "delexicalized_template_count": len(template_counts),
        "delexicalized_unique_fraction": delexicalized_unique_fraction,
        "maximum_template_fraction": maximum_observed_template_fraction,
        "opening_frame_tokens": opening_frame_tokens,
        "maximum_opening_frame_fraction": maximum_observed_opening_fraction,
        "median_keyword_unique_fraction": median_keyword_unique_fraction,
        "minimum_keyword_unique_fraction": minimum_observed_keyword_unique_fraction,
        "thresholds": {
            "minimum_delexicalized_unique_fraction": minimum_delexicalized_unique_fraction,
            "maximum_template_fraction": maximum_template_fraction,
            "minimum_median_keyword_unique_fraction": minimum_median_keyword_unique_fraction,
            "minimum_keyword_unique_fraction": minimum_keyword_unique_fraction,
            "maximum_opening_frame_fraction": maximum_opening_frame_fraction,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "top_delexicalized_templates": top_templates,
        "top_opening_frames": top_opening_frames,
        "scientific_guard": (
            "This audit measures wording diversity only. It does not define B or "
            "either frozen semantic coordinate."
        ),
    }


def parse_generated_question(raw: str) -> str:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("question"), str):
            return _single_line(value["question"])
    raise ValueError("model output does not contain a JSON object with string question")


def validate_generated_question(question: str, keyword: str) -> None:
    if question != _single_line(question):
        raise ValueError("question must be one normalized line")
    words = question.split()
    if not 8 <= len(words) <= 60:
        raise ValueError("question must contain 8 to 60 words")
    if question.count("?") != 1 or not question.endswith("?"):
        raise ValueError("question must end with exactly one question mark")
    if keyword not in question:
        raise ValueError("question lost the exact keyword phrase")
    lowered = question.casefold()
    forbidden = (
        "axis_1", "axis 1", "axis_2", "axis 2", "coordinate", "embedding",
        "readiness score", "rerank", "source preference", "publisher preference",
        "{query}", "answer:",
    )
    if any(term in lowered for term in forbidden):
        raise ValueError("question contains forbidden meta-policy language")


def render_search_validation_request(candidate: ReadinessQuestionCandidate) -> str:
    return f"""Independently evaluate whether the candidate is a useful simulated online-search question.

Required topic phrase: {candidate.keyword}
Candidate question: {candidate.question}

Judge the text itself, not the generator. A valid candidate must:
- remain directly about the required topic phrase;
- express a genuine information need suitable for an online search;
- be answerable using information that could reasonably be found on the web;
- stand alone without hidden conversational context;
- read as one natural question rather than an answer, command to manipulate a model,
  or meta-comment about an experiment.

Return only one JSON object with exactly these fields:
{{"topic_relevant":true,"search_intent":true,"web_answerable":true,
"standalone":true,"natural_language":true,"relevance_score_1_5":5,
"concise_reason":"short reason"}}
"""


def parse_search_question_review(
    raw: str,
    candidate: ReadinessQuestionCandidate,
    *,
    judge_id: str,
    judge_model: str,
) -> SearchQuestionReview:
    raw = _normalize_byte_level_validator_text(raw)
    allowed_keys = (
        "topic_relevant",
        "search_intent",
        "web_answerable",
        "standalone",
        "natural_language",
        "relevance_score_1_5",
        "concise_reason",
    )
    key_pattern = "|".join(re.escape(key) for key in allowed_keys)
    raw = re.sub(rf'\*{{1,2}}(?="(?:{key_pattern})"\s*:)', "", raw)
    raw = re.sub(rf'("(?:{key_pattern})")\*{{1,2}}(?=\s*:)', r"\1", raw)
    decoder = json.JSONDecoder()
    payload = None
    for match in re.finditer(r"\{", raw):
        try:
            value, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload = value
            break
    required_booleans = (
        "topic_relevant",
        "search_intent",
        "web_answerable",
        "standalone",
        "natural_language",
    )
    if payload is None or any(type(payload.get(name)) is not bool for name in required_booleans):
        raise ValueError("validator output lacks required boolean fields")
    score_value = payload.get("relevance_score_1_5")
    score = None
    if type(score_value) is int:
        score = score_value
    elif type(score_value) is float and score_value.is_integer():
        score = int(score_value)
    elif isinstance(score_value, str):
        match = re.fullmatch(r"\s*([1-5])(?:\s*/\s*5)?\s*", score_value)
        if match:
            score = int(match.group(1))
    reason = " ".join(str(payload.get("concise_reason", "")).split())
    if score is None or not 1 <= score <= 5 or not reason:
        raise ValueError("validator output has invalid score or concise reason")
    reason = reason[:240].rstrip()
    exact_keyword = candidate.keyword in candidate.question
    single_question = candidate.question.endswith("?") and candidate.question.count("?") == 1
    accepted = (
        exact_keyword
        and single_question
        and all(bool(payload[name]) for name in required_booleans)
        and score >= 4
    )
    return SearchQuestionReview(
        candidate_id=candidate.candidate_id,
        judge_id=judge_id,
        judge_model=judge_model,
        exact_keyword_present=exact_keyword,
        single_question=single_question,
        topic_relevant=payload["topic_relevant"],
        search_intent=payload["search_intent"],
        web_answerable=payload["web_answerable"],
        standalone=payload["standalone"],
        natural_language=payload["natural_language"],
        relevance_score_1_5=score,
        accepted=accepted,
        concise_reason=reason,
    )


def _normalize_byte_level_validator_text(raw: str) -> str:
    """Undo GPT-2 byte-token display glyphs emitted by some slow tokenizers."""

    if not any(marker in raw for marker in ("Ġ", "Ċ", "ĉ")):
        return raw
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    extra_index = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra_index)
            extra_index += 1
    reverse = {
        chr(unicode_value): byte_value
        for byte_value, unicode_value in zip(byte_values, unicode_values)
    }
    encoded = bytearray()
    for character in raw:
        if character in reverse:
            encoded.append(reverse[character])
        else:
            encoded.extend(character.encode("utf-8"))
    try:
        return encoded.decode("utf-8")
    except UnicodeDecodeError:
        return raw


def generate_question_candidates(
    tasks: Sequence[ReadinessGenerationTask],
    generator: ReadinessQuestionGenerator,
) -> tuple[ReadinessQuestionCandidate, ...]:
    rows = []
    for task in tasks:
        if task.generator_id != generator.generator_id:
            continue
        questions = generator.generate(task)
        if len(questions) != task.requested_candidate_count:
            raise ValueError(f"generator returned wrong candidate count for {task.task_id}")
        for slot, question in enumerate(questions):
            question = _single_line(question)
            validate_generated_question(question, task.keyword)
            identity = {
                "version": READINESS_PROMPT_POPULATION_VERSION,
                "task_id": task.task_id,
                "slot": slot,
                "question": question,
                "model": generator.model_name,
            }
            digest = _stable_hash(identity)
            rows.append(
                ReadinessQuestionCandidate(
                    candidate_id=f"readiness-question:{digest[:24]}",
                    task_id=task.task_id,
                    keyword_id=task.keyword_id,
                    keyword=task.keyword,
                    target_id=task.target.target_id,
                    target_index=task.target.target_index,
                    target_normalized_axis_1=task.target.normalized_axis_1,
                    target_normalized_axis_2=task.target.normalized_axis_2,
                    target_raw_axis_1=task.target.raw_axis_1,
                    target_raw_axis_2=task.target.raw_axis_2,
                    round_index=task.round_index,
                    generator_id=generator.generator_id,
                    generator_model=generator.model_name,
                    candidate_slot=slot,
                    generation_seed=task.generation_seed + slot * 1009,
                    question=question,
                    question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                    proposal_kind=generator.proposal_kind,
                )
            )
    return tuple(rows)


def project_questions(
    fitted: ReadinessEmbeddingMap,
    bounds: ReadinessSubspaceBounds,
    candidates: Sequence[ReadinessQuestionCandidate],
    embeddings: np.ndarray,
) -> tuple[ProjectedReadinessQuestion, ...]:
    generic = project_text_embeddings(
        fitted,
        bounds,
        item_ids=[candidate.candidate_id for candidate in candidates],
        text_sha256s=[candidate.question_sha256 for candidate in candidates],
        embeddings=embeddings,
    )
    rows = []
    for candidate, projected in zip(candidates, generic):
        distance = float(
            np.hypot(
                projected.normalized_axis_1
                - candidate.target_normalized_axis_1,
                projected.normalized_axis_2
                - candidate.target_normalized_axis_2,
            )
        )
        rows.append(
            ProjectedReadinessQuestion(
                candidate_id=candidate.candidate_id,
                raw_axis_1=projected.raw_axis_1,
                raw_axis_2=projected.raw_axis_2,
                normalized_axis_1=projected.normalized_axis_1,
                normalized_axis_2=projected.normalized_axis_2,
                predicted_scalar_readiness_0_1=(
                    projected.predicted_scalar_readiness_0_1
                ),
                target_distance=distance,
            )
        )
    return tuple(rows)


def project_text_embeddings(
    fitted: ReadinessEmbeddingMap,
    bounds: ReadinessSubspaceBounds,
    *,
    item_ids: Sequence[str],
    text_sha256s: Sequence[str],
    embeddings: np.ndarray,
) -> tuple[ReadinessTextProjection, ...]:
    """Project arbitrary text embeddings without assigning them to target cells."""

    if len(item_ids) != len(text_sha256s) or len(set(item_ids)) != len(item_ids):
        raise ValueError("projection item ids must be aligned and unique")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(item_ids), fitted.dimension):
        raise ValueError("text embeddings do not match item count/map dimension")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(matrix).all():
        raise ValueError("candidate embeddings must be finite and nonzero")
    normalized = matrix / norms
    centered = normalized - np.asarray(fitted.embedding_mean, dtype=np.float64)
    axes = np.asarray(fitted.supervised_subspace_axes, dtype=np.float64)
    if axes.shape != (2, fitted.dimension):
        raise ValueError("readiness map must contain exactly two supervised axes")
    raw = centered @ axes.T
    scalar = fitted.label_mean + centered @ np.asarray(
        fitted.scalar_direction, dtype=np.float64
    )
    rows = []
    for item_id, text_sha256, values, scalar_value in zip(
        item_ids, text_sha256s, raw, scalar
    ):
        normalized_axis_1 = _normalize(values[0], bounds.axis_1_low, bounds.axis_1_high)
        normalized_axis_2 = _normalize(values[1], bounds.axis_2_low, bounds.axis_2_high)
        rows.append(
            ReadinessTextProjection(
                item_id=item_id,
                text_sha256=text_sha256,
                raw_axis_1=float(values[0]),
                raw_axis_2=float(values[1]),
                normalized_axis_1=normalized_axis_1,
                normalized_axis_2=normalized_axis_2,
                predicted_scalar_readiness_0_1=float(scalar_value),
            )
        )
    return tuple(rows)


def select_diverse_questions(
    candidates: Sequence[ReadinessQuestionCandidate],
    projections: Sequence[ProjectedReadinessQuestion],
    embeddings: np.ndarray,
    *,
    novelty_weight: float = 0.05,
    generator_balance_weight: float = 0.02,
) -> tuple[tuple[SelectedReadinessQuestion, ...], dict[str, object]]:
    if novelty_weight < 0 or generator_balance_weight < 0:
        raise ValueError("selection weights must be nonnegative")
    if len(candidates) != len(projections):
        raise ValueError("candidate and projection counts differ")
    by_projection = {row.candidate_id: row for row in projections}
    if len(by_projection) != len(projections):
        raise ValueError("projection candidate ids must be unique")
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(candidates):
        raise ValueError("selection embeddings do not align with candidates")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("selection embeddings contain a zero row")
    unit = matrix / norms
    index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    if len(index_by_id) != len(candidates):
        raise ValueError("candidate ids must be unique")
    grouped: dict[str, list[ReadinessQuestionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.keyword_id, []).append(candidate)

    selected: list[SelectedReadinessQuestion] = []
    missing_targets: dict[str, list[str]] = {}
    for keyword_id, keyword_candidates in sorted(grouped.items()):
        target_ids = sorted(
            {candidate.target_id for candidate in keyword_candidates},
            key=lambda target_id: min(
                candidate.target_index
                for candidate in keyword_candidates
                if candidate.target_id == target_id
            ),
        )
        selected_indices: list[int] = []
        selected_hashes: set[str] = set()
        generator_counts: Counter[str] = Counter()
        for target_id in target_ids:
            pool = [
                candidate
                for candidate in keyword_candidates
                if candidate.target_id == target_id
                and candidate.question_sha256 not in selected_hashes
            ]
            if not pool:
                missing_targets.setdefault(keyword_id, []).append(target_id)
                continue
            evaluated = []
            for candidate in pool:
                index = index_by_id[candidate.candidate_id]
                maximum_similarity = (
                    max(float(unit[index] @ unit[other]) for other in selected_indices)
                    if selected_indices
                    else 0.0
                )
                projection = by_projection[candidate.candidate_id]
                objective = (
                    projection.target_distance
                    + novelty_weight * max(0.0, maximum_similarity)
                    + generator_balance_weight * generator_counts[candidate.generator_id]
                )
                evaluated.append((objective, projection.target_distance, candidate.candidate_id, maximum_similarity, candidate, projection, index))
            objective, _, _, maximum_similarity, candidate, projection, index = min(evaluated)
            selection_id = "selected-readiness-question:" + _stable_hash(
                {"keyword_id": keyword_id, "target_id": target_id, "candidate_id": candidate.candidate_id}
            )[:24]
            selected.append(
                SelectedReadinessQuestion(
                    selection_id=selection_id,
                    keyword_id=candidate.keyword_id,
                    keyword=candidate.keyword,
                    target_id=candidate.target_id,
                    target_index=candidate.target_index,
                    candidate_id=candidate.candidate_id,
                    question=candidate.question,
                    generator_id=candidate.generator_id,
                    generator_model=candidate.generator_model,
                    round_index=candidate.round_index,
                    target_normalized_axis_1=candidate.target_normalized_axis_1,
                    target_normalized_axis_2=candidate.target_normalized_axis_2,
                    observed_normalized_axis_1=projection.normalized_axis_1,
                    observed_normalized_axis_2=projection.normalized_axis_2,
                    observed_raw_axis_1=projection.raw_axis_1,
                    observed_raw_axis_2=projection.raw_axis_2,
                    predicted_scalar_readiness_0_1=projection.predicted_scalar_readiness_0_1,
                    target_distance=projection.target_distance,
                    maximum_similarity_to_previously_selected=maximum_similarity,
                    selection_objective=float(objective),
                )
            )
            selected_indices.append(index)
            selected_hashes.add(candidate.question_sha256)
            generator_counts[candidate.generator_id] += 1

    distances = [row.target_distance for row in selected]
    diagnostics = {
        "format_version": READINESS_PROMPT_POPULATION_VERSION,
        "candidate_count": len(candidates),
        "keyword_count": len(grouped),
        "selected_count": len(selected),
        "missing_targets": missing_targets,
        "mean_target_distance": float(np.mean(distances)) if distances else None,
        "maximum_target_distance": max(distances) if distances else None,
        "selected_by_generator": dict(sorted(Counter(row.generator_id for row in selected).items())),
        "scientific_guard": (
            "Frozen LLM2Vec coordinates describe generated questions and do not define B."
        ),
    }
    return tuple(selected), diagnostics


def select_spatially_matched_questions(
    candidates: Sequence[ReadinessQuestionCandidate],
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
    coordinates_by_candidate: Mapping[str, Mapping[str, float]],
    *,
    accepted_candidate_ids: set[str],
    disagreement_weight: float = 0.10,
    distance_tolerance: float = 0.22,
    target_design: str = "rectangular-grid",
    require_both_views_within_tolerance: bool = False,
    require_delexicalized_template_uniqueness: bool = False,
) -> tuple[tuple[SpatiallySelectedReadinessQuestion, ...], dict[str, object]]:
    """Globally match validated candidates to planned two-view coordinates.

    When strict dual-view verification is enabled, a target-candidate edge is
    eligible only when the frozen reference projection and the independently
    aligned candidate projection both lie within ``distance_tolerance``.  The
    consensus coordinate remains useful for matching and coverage diagnostics,
    but opposing view errors can no longer cancel into a false acceptance.
    """

    if not targets or disagreement_weight < 0 or distance_tolerance < 0:
        raise ValueError("invalid spatial matching configuration")
    if target_design not in {"rectangular-grid", "support-aware-random"}:
        raise ValueError(f"unsupported target design: {target_design}")
    grouped: dict[str, list[ReadinessQuestionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.keyword_id, [])
        if candidate.candidate_id not in accepted_candidate_ids:
            continue
        coordinate = coordinates_by_candidate.get(candidate.candidate_id)
        if coordinate is None:
            raise ValueError(f"accepted candidate lacks aligned coordinates: {candidate.candidate_id}")
        values = np.asarray(
            [
                coordinate["reference_normalized_axis_1"],
                coordinate["reference_normalized_axis_2"],
                coordinate["candidate_aligned_normalized_axis_1"],
                coordinate["candidate_aligned_normalized_axis_2"],
                coordinate["consensus_normalized_axis_1"],
                coordinate["consensus_normalized_axis_2"],
                coordinate["cross_embedding_disagreement"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"candidate has nonfinite aligned coordinates: {candidate.candidate_id}")
        grouped[candidate.keyword_id].append(candidate)
    if not grouped:
        raise ValueError("no candidates are available for spatial matching")
    keyword_text = {
        candidate.keyword_id: candidate.keyword for candidate in candidates
    }
    keyword_targets = _targets_by_keyword(
        sorted(keyword_text.items()), targets
    )

    from scipy.optimize import linear_sum_assignment

    selected = []
    keyword_diagnostics = {}
    for keyword_id, pool in sorted(grouped.items()):
        planned_targets = keyword_targets[keyword_id]
        target_matrix = np.asarray(
            [
                [target.normalized_axis_1, target.normalized_axis_2]
                for target in planned_targets
            ],
            dtype=np.float64,
        )
        if not pool:
            keyword_diagnostics[keyword_id] = _spatial_coverage_diagnostics(
                (),
                targets=planned_targets,
                distance_tolerance=distance_tolerance,
                target_design=target_design,
            )
            continue
        pool = sorted(pool, key=lambda row: row.candidate_id)
        observed = np.asarray(
            [
                [
                    coordinates_by_candidate[row.candidate_id]["consensus_normalized_axis_1"],
                    coordinates_by_candidate[row.candidate_id]["consensus_normalized_axis_2"],
                ]
                for row in pool
            ],
            dtype=np.float64,
        )
        reference_observed = np.asarray(
            [
                [
                    coordinates_by_candidate[row.candidate_id][
                        "reference_normalized_axis_1"
                    ],
                    coordinates_by_candidate[row.candidate_id][
                        "reference_normalized_axis_2"
                    ],
                ]
                for row in pool
            ],
            dtype=np.float64,
        )
        candidate_observed = np.asarray(
            [
                [
                    coordinates_by_candidate[row.candidate_id][
                        "candidate_aligned_normalized_axis_1"
                    ],
                    coordinates_by_candidate[row.candidate_id][
                        "candidate_aligned_normalized_axis_2"
                    ],
                ]
                for row in pool
            ],
            dtype=np.float64,
        )
        disagreement = np.asarray(
            [
                coordinates_by_candidate[row.candidate_id]["cross_embedding_disagreement"]
                for row in pool
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm(
            target_matrix[:, None, :] - observed[None, :, :], axis=2
        )
        reference_distances = np.linalg.norm(
            target_matrix[:, None, :] - reference_observed[None, :, :], axis=2
        )
        candidate_distances = np.linalg.norm(
            target_matrix[:, None, :] - candidate_observed[None, :, :], axis=2
        )
        verified_pairs = (reference_distances <= distance_tolerance) & (
            candidate_distances <= distance_tolerance
        )
        base_costs = distances + disagreement_weight * disagreement[None, :]
        if require_both_views_within_tolerance:
            # One private dummy column per target permits unmatched targets.  A
            # penalty larger than every possible sum of real costs makes the
            # assignment maximize verified cardinality before minimizing cost.
            maximum_cost = float(np.max(base_costs)) if base_costs.size else 0.0
            unmatched_penalty = (maximum_cost + 1.0) * (len(planned_targets) + 1)
            costs = np.full(
                (len(planned_targets), len(pool) + len(planned_targets)),
                2.0 * unmatched_penalty,
                dtype=np.float64,
            )
            costs[:, : len(pool)] = np.where(
                verified_pairs, base_costs, 2.0 * unmatched_penalty
            )
            for target_index in range(len(planned_targets)):
                costs[target_index, len(pool) + target_index] = unmatched_penalty
        else:
            costs = base_costs
        target_indices, candidate_indices = linear_sum_assignment(costs)
        rows = []
        for target_index, candidate_index in zip(target_indices, candidate_indices):
            if candidate_index >= len(pool):
                continue
            target = planned_targets[int(target_index)]
            candidate = pool[int(candidate_index)]
            coordinate = coordinates_by_candidate[candidate.candidate_id]
            both_views_within_tolerance = bool(
                verified_pairs[target_index, candidate_index]
            )
            row = SpatiallySelectedReadinessQuestion(
                keyword_id=candidate.keyword_id,
                keyword=candidate.keyword,
                target_id=target.target_id,
                target_index=target.target_index,
                candidate_id=candidate.candidate_id,
                question=candidate.question,
                generator_id=candidate.generator_id,
                generator_model=candidate.generator_model,
                target_normalized_axis_1=target.normalized_axis_1,
                target_normalized_axis_2=target.normalized_axis_2,
                consensus_normalized_axis_1=float(observed[candidate_index, 0]),
                consensus_normalized_axis_2=float(observed[candidate_index, 1]),
                reference_normalized_axis_1=float(coordinate["reference_normalized_axis_1"]),
                reference_normalized_axis_2=float(coordinate["reference_normalized_axis_2"]),
                candidate_aligned_normalized_axis_1=float(
                    coordinate["candidate_aligned_normalized_axis_1"]
                ),
                candidate_aligned_normalized_axis_2=float(
                    coordinate["candidate_aligned_normalized_axis_2"]
                ),
                target_distance=float(distances[target_index, candidate_index]),
                reference_target_distance=float(
                    reference_distances[target_index, candidate_index]
                ),
                candidate_aligned_target_distance=float(
                    candidate_distances[target_index, candidate_index]
                ),
                both_views_within_tolerance=both_views_within_tolerance,
                cross_embedding_disagreement=float(disagreement[candidate_index]),
                assignment_cost=float(costs[target_index, candidate_index]),
            )
            rows.append(row)
            selected.append(row)
        keyword_diagnostics[keyword_id] = _spatial_coverage_diagnostics(
            rows,
            targets=planned_targets,
            distance_tolerance=distance_tolerance,
            target_design=target_design,
        )

    template_groups: dict[str, list[SpatiallySelectedReadinessQuestion]] = {}
    for row in selected:
        template_groups.setdefault(
            delexicalize_question(row.question, row.keyword), []
        ).append(row)
    duplicate_groups = {
        template: rows
        for template, rows in template_groups.items()
        if len(rows) > 1
    }
    template_duplicate_rejections = []
    if require_delexicalized_template_uniqueness:
        retained_ids = set()
        for rows in template_groups.values():
            retained = min(
                rows,
                key=lambda row: (
                    not row.both_views_within_tolerance,
                    max(
                        row.reference_target_distance,
                        row.candidate_aligned_target_distance,
                    ),
                    row.assignment_cost,
                    row.candidate_id,
                ),
            )
            retained_ids.add(retained.candidate_id)
            template_duplicate_rejections.extend(
                row for row in rows if row.candidate_id != retained.candidate_id
            )
        selected = [row for row in selected if row.candidate_id in retained_ids]

    # Recompute per-keyword gates after any cross-keyword template removals so
    # an omitted target necessarily becomes a refinement task.
    selected_by_keyword: dict[str, list[SpatiallySelectedReadinessQuestion]] = {
        keyword_id: [] for keyword_id in grouped
    }
    for row in selected:
        selected_by_keyword[row.keyword_id].append(row)
    keyword_diagnostics = {
        keyword_id: _spatial_coverage_diagnostics(
            selected_by_keyword[keyword_id],
            targets=keyword_targets[keyword_id],
            distance_tolerance=distance_tolerance,
            target_design=target_design,
        )
        for keyword_id in sorted(grouped)
    }

    all_distances = [row.target_distance for row in selected]
    all_reference_distances = [row.reference_target_distance for row in selected]
    all_candidate_distances = [
        row.candidate_aligned_target_distance for row in selected
    ]
    all_disagreement = [row.cross_embedding_disagreement for row in selected]
    verified_selected_count = sum(
        row.both_views_within_tolerance for row in selected
    )
    target_counts = {
        keyword_id: len(keyword_targets[keyword_id]) for keyword_id in grouped
    }
    unique_target_counts = set(target_counts.values())
    keyword_pass_fraction = float(
        np.mean(
            [
                item["spacing_gate_passed"]
                for item in keyword_diagnostics.values()
            ]
        )
    )
    pooled_coverage = _pooled_spatial_coverage(
        selected,
        keyword_targets,
        distance_tolerance=distance_tolerance,
    )
    all_keywords_pass = all(
        item["spacing_gate_passed"] for item in keyword_diagnostics.values()
    )
    overall_gate = (
        pooled_coverage["spacing_gate_passed"]
        and keyword_pass_fraction >= 0.80
        if target_design == "support-aware-random"
        else all_keywords_pass
    )
    diagnostics = {
        "format_version": READINESS_PROMPT_POPULATION_VERSION,
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted_candidate_ids),
        "selected_count": len(selected),
        "keyword_count": len(grouped),
        "target_count_per_keyword": (
            next(iter(unique_target_counts)) if len(unique_target_counts) == 1 else target_counts
        ),
        "target_design": target_design,
        "mean_target_distance": float(np.mean(all_distances)) if all_distances else None,
        "maximum_target_distance": float(np.max(all_distances)) if all_distances else None,
        "mean_reference_target_distance": (
            float(np.mean(all_reference_distances))
            if all_reference_distances
            else None
        ),
        "maximum_reference_target_distance": (
            float(np.max(all_reference_distances))
            if all_reference_distances
            else None
        ),
        "mean_candidate_aligned_target_distance": (
            float(np.mean(all_candidate_distances))
            if all_candidate_distances
            else None
        ),
        "maximum_candidate_aligned_target_distance": (
            float(np.max(all_candidate_distances))
            if all_candidate_distances
            else None
        ),
        "verified_selected_count": verified_selected_count,
        "verified_selected_fraction": (
            verified_selected_count / len(selected) if selected else 0.0
        ),
        "require_both_views_within_tolerance": (
            require_both_views_within_tolerance
        ),
        "require_delexicalized_template_uniqueness": (
            require_delexicalized_template_uniqueness
        ),
        "delexicalized_template_count_before_filter": len(template_groups),
        "delexicalized_duplicate_group_count_before_filter": len(
            duplicate_groups
        ),
        "template_duplicate_rejection_count": len(
            template_duplicate_rejections
        ),
        "selected_delexicalized_templates_are_unique": len(selected)
        == len(
            {
                delexicalize_question(row.question, row.keyword)
                for row in selected
            }
        ),
        "mean_cross_embedding_disagreement": (
            float(np.mean(all_disagreement)) if all_disagreement else None
        ),
        "keywords": keyword_diagnostics,
        "keyword_spacing_gate_pass_fraction": keyword_pass_fraction,
        "all_keywords_pass_spacing_gate": all_keywords_pass,
        "pooled_support_coverage": pooled_coverage,
        "overall_spacing_gate_passed": overall_gate,
        "selection_method": (
            "global-linear-assignment-with-strict-dual-view-tolerance"
            if require_both_views_within_tolerance
            else "global-linear-assignment-on-aligned-two-view-consensus"
        ),
    }
    return tuple(sorted(selected, key=lambda row: (row.keyword_id, row.target_index))), diagnostics


def _spatial_coverage_diagnostics(
    rows,
    *,
    targets: Sequence[ReadinessPromptTarget],
    distance_tolerance: float,
    target_design: str,
):
    target_count = len(targets)
    target_coordinates = np.asarray(
        [
            [target.normalized_axis_1, target.normalized_axis_2]
            for target in targets
        ],
        dtype=np.float64,
    )
    target_axis_spans = np.ptp(target_coordinates, axis=0)
    if target_count > 1:
        target_pairwise = np.linalg.norm(
            target_coordinates[:, None, :] - target_coordinates[None, :, :],
            axis=2,
        )
        np.fill_diagonal(target_pairwise, np.inf)
        target_nearest = target_pairwise.min(axis=1)
    else:
        target_nearest = np.asarray([0.0])
    target_occupied = {
        (
            int(np.clip(np.rint(value[0] * 5), 0, 5)),
            int(np.clip(np.rint(value[1] * 4), 0, 4)),
        )
        for value in target_coordinates
    }
    if not rows:
        gate_checks = _empty_spatial_gate_checks(target_design)
        return {
            "target_count": target_count,
            "selected_count": 0,
            "spacing_gate_passed": False,
            "uncovered_target_count": target_count,
            "within_distance_tolerance_fraction": 0.0,
            "axis_1_span": 0.0,
            "axis_2_span": 0.0,
            "median_nearest_neighbor_distance": 0.0,
            "occupied_grid_bin_count": 0,
            "target_axis_1_span": float(target_axis_spans[0]),
            "target_axis_2_span": float(target_axis_spans[1]),
            "target_median_nearest_neighbor_distance": float(
                np.median(target_nearest)
            ),
            "target_occupied_grid_bin_count": len(target_occupied),
            "gate_checks": gate_checks,
        }
    coordinates = np.asarray(
        [
            [row.consensus_normalized_axis_1, row.consensus_normalized_axis_2]
            for row in rows
        ],
        dtype=np.float64,
    )
    if len(rows) > 1:
        pairwise = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        np.fill_diagonal(pairwise, np.inf)
        nearest = pairwise.min(axis=1)
    else:
        nearest = np.asarray([0.0])
    distances = np.asarray([row.target_distance for row in rows])
    axis_spans = np.ptp(coordinates, axis=0)
    occupied = {
        (
            int(np.clip(np.rint(value[0] * 5), 0, 5)),
            int(np.clip(np.rint(value[1] * 4), 0, 4)),
        )
        for value in coordinates
    }
    within = int(np.sum(distances <= distance_tolerance))
    if target_design == "support-aware-random":
        gate_checks = {
            "complete_target_count": len(rows) == target_count,
            "mean_target_distance_at_most_0_25": float(np.mean(distances)) <= 0.25,
            "at_least_80_percent_within_tolerance": within
            >= int(np.ceil(0.80 * target_count)),
            "both_axis_spans_cover_80_percent_of_target": bool(
                np.all(axis_spans >= 0.80 * target_axis_spans)
            ),
            "median_spacing_at_least_60_percent_of_target": float(
                np.median(nearest)
            )
            >= 0.60 * float(np.median(target_nearest)),
            "at_least_80_percent_target_grid_bins_occupied": len(occupied)
            >= int(np.ceil(0.80 * len(target_occupied))),
        }
    else:
        gate_checks = {
            "complete_target_count": len(rows) == target_count,
            "mean_target_distance_at_most_0_25": float(np.mean(distances)) <= 0.25,
            "at_least_80_percent_within_tolerance": within
            >= int(np.ceil(0.80 * target_count)),
            "both_axis_spans_at_least_0_70": bool(np.all(axis_spans >= 0.70)),
            "median_nearest_neighbor_at_least_0_08": float(np.median(nearest))
            >= 0.08,
            "at_least_60_percent_grid_bins_occupied": len(occupied)
            >= int(np.ceil(0.60 * target_count)),
        }
    return {
        "target_count": target_count,
        "selected_count": len(rows),
        "uncovered_target_count": target_count - len(rows),
        "mean_target_distance": float(np.mean(distances)),
        "maximum_target_distance": float(np.max(distances)),
        "within_distance_tolerance_count": within,
        "within_distance_tolerance_fraction": within / target_count,
        "axis_1_span": float(axis_spans[0]),
        "axis_2_span": float(axis_spans[1]),
        "minimum_nearest_neighbor_distance": float(np.min(nearest)),
        "median_nearest_neighbor_distance": float(np.median(nearest)),
        "occupied_grid_bin_count": len(occupied),
        "target_axis_1_span": float(target_axis_spans[0]),
        "target_axis_2_span": float(target_axis_spans[1]),
        "target_median_nearest_neighbor_distance": float(np.median(target_nearest)),
        "target_occupied_grid_bin_count": len(target_occupied),
        "gate_checks": gate_checks,
        "spacing_gate_passed": all(gate_checks.values()),
    }


def _empty_spatial_gate_checks(target_design: str) -> dict[str, bool]:
    if target_design == "support-aware-random":
        return {
            "complete_target_count": False,
            "mean_target_distance_at_most_0_25": False,
            "at_least_80_percent_within_tolerance": False,
            "both_axis_spans_cover_80_percent_of_target": False,
            "median_spacing_at_least_60_percent_of_target": False,
            "at_least_80_percent_target_grid_bins_occupied": False,
        }
    return {
        "complete_target_count": False,
        "mean_target_distance_at_most_0_25": False,
        "at_least_80_percent_within_tolerance": False,
        "both_axis_spans_at_least_0_70": False,
        "median_nearest_neighbor_at_least_0_08": False,
        "at_least_60_percent_grid_bins_occupied": False,
    }


def _pooled_spatial_coverage(
    selected: Sequence[SpatiallySelectedReadinessQuestion],
    targets_by_keyword: Mapping[str, Sequence[ReadinessPromptTarget]],
    *,
    distance_tolerance: float,
    grid_resolution: int = 10,
) -> dict[str, object]:
    targets = [
        target
        for keyword_targets in targets_by_keyword.values()
        for target in keyword_targets
    ]
    target_coordinates = np.asarray(
        [
            [target.normalized_axis_1, target.normalized_axis_2]
            for target in targets
        ],
        dtype=np.float64,
    )
    observed_coordinates = np.asarray(
        [
            [row.consensus_normalized_axis_1, row.consensus_normalized_axis_2]
            for row in selected
        ],
        dtype=np.float64,
    ).reshape((-1, 2))

    def histogram(values: np.ndarray) -> np.ndarray:
        counts = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
        for coordinate in values:
            indices = tuple(
                int(np.clip(np.floor(value * grid_resolution), 0, grid_resolution - 1))
                for value in coordinate
            )
            counts[indices] += 1
        return counts

    target_histogram = histogram(target_coordinates)
    observed_histogram = histogram(observed_coordinates)
    target_probability = target_histogram / target_histogram.sum()
    observed_probability = (
        observed_histogram / observed_histogram.sum()
        if len(observed_coordinates)
        else observed_histogram
    )
    total_variation = float(
        0.5 * np.sum(np.abs(target_probability - observed_probability))
    )
    target_occupied = int(np.count_nonzero(target_histogram))
    observed_on_target = int(
        np.count_nonzero((target_histogram > 0) & (observed_histogram > 0))
    )
    target_spans = np.ptp(target_coordinates, axis=0)
    observed_spans = (
        np.ptp(observed_coordinates, axis=0)
        if len(observed_coordinates)
        else np.zeros(2)
    )
    distances = np.asarray([row.target_distance for row in selected])
    within_fraction = (
        float(np.mean(distances <= distance_tolerance)) if len(distances) else 0.0
    )
    gate_checks = {
        "complete_pooled_target_count": len(selected) == len(targets),
        "pooled_mean_target_distance_at_most_0_25": bool(
            len(distances) and float(np.mean(distances)) <= 0.25
        ),
        "pooled_at_least_80_percent_within_tolerance": within_fraction >= 0.80,
        "pooled_axis_spans_cover_80_percent_of_target": bool(
            np.all(observed_spans >= 0.80 * target_spans)
        ),
        "pooled_at_least_80_percent_target_bins_occupied": observed_on_target
        >= int(np.ceil(0.80 * target_occupied)),
        "pooled_histogram_total_variation_at_most_0_25": total_variation <= 0.25,
    }
    return {
        "target_count": len(targets),
        "selected_count": len(selected),
        "grid_resolution": grid_resolution,
        "within_distance_tolerance_fraction": within_fraction,
        "target_axis_1_span": float(target_spans[0]),
        "target_axis_2_span": float(target_spans[1]),
        "observed_axis_1_span": float(observed_spans[0]),
        "observed_axis_2_span": float(observed_spans[1]),
        "target_occupied_grid_bin_count": target_occupied,
        "observed_target_grid_bin_count": observed_on_target,
        "histogram_total_variation": total_variation,
        "gate_checks": gate_checks,
        "spacing_gate_passed": all(gate_checks.values()),
    }


def build_refinement_tasks(
    selected: Sequence[SelectedReadinessQuestion],
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
    generator_ids: Sequence[str],
    *,
    next_round_index: int,
    distance_tolerance: float = 0.22,
    master_seed: int = 20260820,
    requested_candidate_count: int = 3,
) -> tuple[ReadinessGenerationTask, ...]:
    if distance_tolerance < 0:
        raise ValueError("distance_tolerance must be nonnegative")
    keywords = sorted({(row.keyword_id, row.keyword) for row in selected})
    targets_by_keyword = _targets_by_keyword(keywords, targets)
    selected_by_key = {(row.keyword_id, row.target_id): row for row in selected}
    feedback = {}
    selected_targets = []
    for keyword_id, _ in keywords:
        for target in targets_by_keyword[keyword_id]:
            row = selected_by_key.get((keyword_id, target.target_id))
            if row is not None and row.target_distance <= distance_tolerance:
                continue
            selected_targets.append((keyword_id, target.target_id))
            feedback[(keyword_id, target.target_id)] = _refinement_feedback(row, target)
    all_tasks = build_generation_tasks(
        keywords,
        targets,
        generator_ids,
        round_index=next_round_index,
        master_seed=master_seed,
        requested_candidate_count=requested_candidate_count,
        feedback_by_keyword_target=feedback,
    )
    keep = set(selected_targets)
    return tuple(task for task in all_tasks if (task.keyword_id, task.target.target_id) in keep)


def _refinement_feedback(
    selected: SelectedReadinessQuestion | None,
    target: ReadinessPromptTarget,
) -> str:
    if selected is None:
        return "No valid candidate covered this cell; produce a clearly targeted alternative."
    delta_1 = target.normalized_axis_1 - selected.observed_normalized_axis_1
    delta_2 = target.normalized_axis_2 - selected.observed_normalized_axis_2
    direction_1 = "more action-ready" if delta_1 > 0 else "more exploratory"
    direction_2 = "more implementation-focused" if delta_2 > 0 else "more comparison-focused"
    return (
        f"The closest earlier question landed {abs(delta_1):.3f} away on the readiness "
        f"dimension and {abs(delta_2):.3f} away on the decision-mode dimension. "
        f"Make the new question {direction_1} and {direction_2}. Rewrite the closest "
        f"question with the smallest semantic change while preserving the exact "
        f"keyword: {selected.question}"
    )


def _axis_1_instruction(value: float) -> str:
    if value <= 0.2:
        return "purely understand or explain the topic; avoid choosing or acting"
    if value <= 0.4:
        return "investigate evidence, mechanisms, or implications"
    if value <= 0.6:
        return "evaluate concrete options or trade-offs"
    if value <= 0.8:
        return "prepare a decision, commitment, or practical plan"
    return "request an immediate, concrete action or execution step"


def _axis_2_instruction(value: float) -> str:
    if value <= 0.25:
        return "compare alternatives and decide which approach fits"
    if value <= 0.5:
        return "select an approach using explicit criteria"
    if value <= 0.75:
        return "translate a chosen approach into a practical procedure"
    return "implement, configure, troubleshoot, or execute a chosen approach"


def _generate_with_seed(ranker, prompt: str, *, seed: int, max_new_tokens: int, temperature: float) -> str:
    try:
        import torch
    except ImportError:
        return str(ranker.rank(prompt, max_tokens=max_new_tokens, temperature=temperature)).strip()
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            value = ranker.rank(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            value = ranker.rank(prompt, max_tokens=max_new_tokens, temperature=temperature)
    return str(value).strip()


def _normalize(value: float, low: float, high: float) -> float:
    return float((value - low) / (high - low))


def _denormalize(value: float, low: float, high: float) -> float:
    return float(low + value * (high - low))


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def _atomic_json(path: Path, payload: object) -> None:
    from .readiness_hf_dataset import atomic_json

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
