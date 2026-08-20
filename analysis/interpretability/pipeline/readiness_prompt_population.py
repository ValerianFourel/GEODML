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


READINESS_PROMPT_POPULATION_VERSION = "readiness-question-population-v1"


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


class ReadinessQuestionGenerator(Protocol):
    generator_id: str
    model_name: str
    proposal_kind: str

    def generate(
        self, task: ReadinessGenerationTask
    ) -> tuple[str, ...]: ...


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
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            questions = tuple(str(value) for value in payload["questions"])
            for question in questions:
                validate_generated_question(question, task.keyword)
            return questions

        accepted: list[str] = []
        failures: list[dict[str, object]] = []
        for slot in range(task.requested_candidate_count):
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
                    continue
                accepted.append(question)
                break
            else:
                raise RuntimeError(
                    f"question generation failed for {task.task_id} slot {slot}: "
                    f"{failures[-1]['error'] if failures else 'unknown error'}"
                )
        _atomic_json(
            cache_path,
            {"identity": identity, "questions": accepted, "failures": failures},
        )
        return tuple(accepted)


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


def build_generation_tasks(
    keywords: Sequence[tuple[str, str]],
    targets: Sequence[ReadinessPromptTarget],
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
    for keyword_index, (keyword_id, keyword) in enumerate(keywords):
        keyword = _single_line(keyword)
        if not keyword_id.strip() or not keyword:
            raise ValueError("keyword ids and text must be nonempty")
        for target in targets:
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


def render_generation_request(
    task: ReadinessGenerationTask, *, candidate_slot: int
) -> str:
    a1 = _axis_1_instruction(task.target.normalized_axis_1)
    a2 = _axis_2_instruction(task.target.normalized_axis_2)
    return f"""Write one standalone, natural search question about the exact keyword phrase below.

Exact keyword phrase (must appear verbatim): {task.keyword}

Semantic destination:
- Readiness stage: {a1}
- Decision mode: {a2}

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


def build_refinement_tasks(
    selected: Sequence[SelectedReadinessQuestion],
    targets: Sequence[ReadinessPromptTarget],
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
    selected_by_key = {(row.keyword_id, row.target_id): row for row in selected}
    feedback = {}
    selected_targets = []
    for keyword_id, _ in keywords:
        for target in targets:
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
        f"Make the new question {direction_1} and {direction_2}."
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
