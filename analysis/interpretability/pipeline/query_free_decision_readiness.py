"""Query-free decision-readiness direction contracts for frozen LLM2Vec space.

The scientific treatment is the assigned continuous readiness coordinate.  A
frozen embedding and external judges measure the realized prompt; neither may
reassign the treatment.  Fake providers in this module exist only for CPU
contract tests and smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence

import numpy as np


QUERY_FREE_AXIS_VERSION = "query-free-decision-readiness-v1"
CONTENT_MARKER = "[CONTENT]"
REPRESENTATION_VIEWS = ("intent-only", "content-masked", "full-content")
DEFAULT_SPEC_PATH = (
    Path(__file__).resolve().parent
    / "specs"
    / "query_free_decision_readiness_v1.json"
)

_FORBIDDEN_CRITERION = re.compile(
    r"\b(?:price|pricing|cost|budget|geograph(?:y|ic)|country|region|fresh(?:ness)?|"
    r"recen(?:t|cy)|popular(?:ity)?|prestige|authority|ratings?|sentiment|urgent|"
    r"publisher|source ownership|seller-controlled|seller-independent|first-party|"
    r"third-party|candidate count|company size)\b",
    re.IGNORECASE,
)
_INFORMATION_SEEKING = re.compile(
    r"\b(?:understand|learn|explore|explain|concepts?|fundamentals?|uses?)\w*\b",
    re.IGNORECASE,
)
_ACTION_READY = re.compile(
    r"\b(?:select|choose|adopt|acquire|deploy|execute|implement)\w*\b",
    re.IGNORECASE,
)
_NEGATED_ACTION = re.compile(
    r"\b(?:without|before|not)\s+(?:actively\s+)?(?:evaluat|select|choos|adopt|"
    r"acquir|deploy|execut|implement)\w*",
    re.IGNORECASE,
)


class ObjectiveGenerator(Protocol):
    backend_name: str
    model_name: str

    def generate(self, request: "QueryFreeGenerationRequest") -> str: ...


@dataclass(frozen=True, slots=True)
class ContentContext:
    context_id: str
    macrodomain: str
    payload: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in {"development", "confirmation"}:
            raise ValueError("context split must be development or confirmation")
        for name, value in (
            ("context_id", self.context_id),
            ("macrodomain", self.macrodomain),
            ("payload", self.payload),
        ):
            if not isinstance(value, str) or not " ".join(value.split()):
                raise ValueError(f"{name} must be a non-empty string")
        if CONTENT_MARKER in self.payload:
            raise ValueError("content payload must not contain the reserved marker")


@dataclass(frozen=True, slots=True)
class RealizationPlan:
    plan_id: str
    split: str
    tone: str
    sentence_length_band: str
    syntax: str
    clause_order: str
    directness: str
    formality: str
    response_form: str

    def __post_init__(self) -> None:
        if self.split not in {"development", "confirmation"}:
            raise ValueError("plan split must be development or confirmation")
        values = (
            self.plan_id,
            self.tone,
            self.sentence_length_band,
            self.syntax,
            self.clause_order,
            self.directness,
            self.formality,
            self.response_form,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("realization plan fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class QueryFreeGenerationRequest:
    request_id: str
    block_id: str
    context: ContentContext
    plan: RealizationPlan
    assigned_a1: float
    level_index: int
    generation_seed: int


@dataclass(frozen=True, slots=True)
class QueryFreeStimulus:
    stimulus_id: str
    stimulus_hash: str
    request_id: str
    block_id: str
    context_id: str
    macrodomain: str
    context_split: str
    content_payload: str
    realization_plan_id: str
    realization_plan_split: str
    assigned_a1: float
    level_index: int
    generation_seed: int
    generator_backend: str
    generator_model: str
    objective_clause: str
    intent_only_text: str
    content_masked_text: str
    full_content_text: str
    compiler_signature: str
    structural_valid: bool
    contract_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OrdinalJudgeTask:
    task_id: str
    stimulus_id: str
    prompt_text: str
    rubric_version: str


@dataclass(frozen=True, slots=True)
class PairwiseJudgeTask:
    task_id: str
    context_id: str
    left_stimulus_id: str
    right_stimulus_id: str
    left_prompt_text: str
    right_prompt_text: str
    comparison_kind: str
    presentation_order: str


@dataclass(frozen=True, slots=True)
class ViewDirection:
    representation_view: str
    unnormalized_direction: tuple[float, ...]
    unit_direction: tuple[float, ...]
    cosine_with_shared: float


@dataclass(frozen=True, slots=True)
class QueryFreeAxis:
    axis_id: str
    axis_version: str
    embedding_model: str
    dimension: int
    block_count: int
    stimulus_count: int
    representation_views: tuple[str, ...]
    shared_unnormalized_direction: tuple[float, ...]
    shared_unit_direction: tuple[float, ...]
    view_directions: tuple[ViewDirection, ...]


@dataclass(frozen=True, slots=True)
class AxisCoordinate:
    stimulus_id: str
    block_id: str
    context_id: str
    realization_plan_id: str
    representation_view: str
    assigned_a1: float
    raw_coordinate: float
    absolute_assigned_coordinate_error: float
    matched_off_axis_residual: float


@dataclass(frozen=True, slots=True)
class QueryFreeGeometryDiagnostics:
    stimulus_count: int
    block_count: int
    representation_count: int
    embedding_dimension: int
    treatment_first_component_share: float
    median_local_to_global_cosine: float
    positive_local_to_global_cosine_rate: float
    mean_block_spearman: float
    minimum_block_spearman: float
    mean_path_tortuosity: float
    mean_absolute_assigned_coordinate_error: float
    view_cosines_with_shared: tuple[tuple[str, float], ...]


def load_query_free_specification(
    path: str | Path = DEFAULT_SPEC_PATH,
) -> tuple[tuple[ContentContext, ...], tuple[RealizationPlan, ...], dict[str, object]]:
    """Load and validate the versioned query-free population specification."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("specification_version") != QUERY_FREE_AXIS_VERSION:
        raise ValueError("unexpected query-free specification version")
    contexts = tuple(ContentContext(**row) for row in payload.get("contexts", ()))
    plans = tuple(RealizationPlan(**row) for row in payload.get("realization_plans", ()))
    if len(contexts) != 64:
        raise ValueError("query-free specification must contain exactly 64 contexts")
    if len(plans) != 4:
        raise ValueError("query-free specification must contain exactly four plans")
    if len({item.context_id for item in contexts}) != len(contexts):
        raise ValueError("query-free context IDs must be unique")
    if len({item.plan_id for item in plans}) != len(plans):
        raise ValueError("query-free plan IDs must be unique")
    domains = {item.macrodomain for item in contexts}
    if len(domains) != 8:
        raise ValueError("query-free specification must contain eight macrodomains")
    for domain in domains:
        rows = [item for item in contexts if item.macrodomain == domain]
        if len(rows) != 8:
            raise ValueError("every macrodomain must contain eight contexts")
        if sum(item.split == "development" for item in rows) != 5:
            raise ValueError("every macrodomain must contain five development contexts")
        if sum(item.split == "confirmation" for item in rows) != 3:
            raise ValueError("every macrodomain must contain three confirmation contexts")
    if sum(item.split == "development" for item in plans) != 2:
        raise ValueError("two realization plans must be development plans")
    if sum(item.split == "confirmation" for item in plans) != 2:
        raise ValueError("two realization plans must be confirmation plans")
    return contexts, plans, payload


def stratified_random_a1_grid(
    *,
    master_seed: int,
    context_id: str,
    plan_id: str,
    level_count: int = 7,
    jitter_fraction: float = 0.35,
    minimum_separation: float = 0.04,
) -> tuple[float, ...]:
    """Return a stable, irregular, full-range grid for one matched block."""

    if isinstance(level_count, bool) or not isinstance(level_count, int):
        raise TypeError("level_count must be an integer")
    if level_count < 3:
        raise ValueError("level_count must be at least three")
    if not 0.0 <= jitter_fraction < 0.5:
        raise ValueError("jitter_fraction must be in [0, 0.5)")
    if not 0.0 < minimum_separation < 1.0:
        raise ValueError("minimum_separation must be in (0, 1)")
    step = 1.0 / (level_count - 1)
    values = [0.0]
    for index in range(1, level_count - 1):
        uniform = _stable_uniform(
            QUERY_FREE_AXIS_VERSION,
            master_seed,
            context_id,
            plan_id,
            index,
        )
        jitter = (2.0 * uniform - 1.0) * jitter_fraction * step
        values.append(index * step + jitter)
    values.append(1.0)
    if any(right - left < minimum_separation for left, right in zip(values, values[1:])):
        raise ValueError("jitter configuration violates minimum A1 separation")
    return tuple(values)


def build_generation_requests(
    contexts: Sequence[ContentContext],
    plans: Sequence[RealizationPlan],
    *,
    master_seed: int = 20260817,
    level_count: int = 7,
) -> tuple[QueryFreeGenerationRequest, ...]:
    """Cross contexts with same-split plans and assign irregular A1 grids."""

    context_rows = tuple(contexts)
    plan_rows = tuple(plans)
    if not context_rows or not plan_rows:
        raise ValueError("contexts and plans must be non-empty")
    requests: list[QueryFreeGenerationRequest] = []
    for context in sorted(context_rows, key=lambda item: item.context_id):
        matched_plans = [item for item in plan_rows if item.split == context.split]
        if len(matched_plans) != 2:
            raise ValueError(f"context split {context.split!r} must have exactly two plans")
        for plan in sorted(matched_plans, key=lambda item: item.plan_id):
            block_identity = _canonical_json(
                {
                    "version": QUERY_FREE_AXIS_VERSION,
                    "context_id": context.context_id,
                    "plan_id": plan.plan_id,
                }
            )
            block_id = f"qf-block:{_hash(block_identity)[:24]}"
            grid = stratified_random_a1_grid(
                master_seed=master_seed,
                context_id=context.context_id,
                plan_id=plan.plan_id,
                level_count=level_count,
            )
            for level_index, assigned_a1 in enumerate(grid):
                identity = _canonical_json(
                    {
                        "version": QUERY_FREE_AXIS_VERSION,
                        "block_id": block_id,
                        "assigned_a1": f"{assigned_a1:.17g}",
                        "level_index": level_index,
                        "master_seed": master_seed,
                    }
                )
                digest = _hash(identity)
                requests.append(
                    QueryFreeGenerationRequest(
                        request_id=f"qf-request:{digest[:24]}",
                        block_id=block_id,
                        context=context,
                        plan=plan,
                        assigned_a1=assigned_a1,
                        level_index=level_index,
                        generation_seed=int(digest[:16], 16) % (2**31 - 1),
                    )
                )
    return tuple(requests)


def render_query_free_generation_prompt(request: QueryFreeGenerationRequest) -> str:
    """Render the frozen generator instruction for one continuous A1 request."""

    plan = request.plan
    plan_payload = {
        "tone": plan.tone,
        "sentence_length_band": plan.sentence_length_band,
        "syntax": plan.syntax,
        "clause_order": plan.clause_order,
        "directness": plan.directness,
        "formality": plan.formality,
        "question_or_instruction_form": plan.response_form,
    }
    return f"""Generate one objective clause for a content-response instruction.

Subject marker: {CONTENT_MARKER}
The literal subject is withheld from your output; use the marker exactly once.
Assigned decision-readiness intensity: {request.assigned_a1:.12f}

Continuous semantic anchors:
- 0.00: understand or learn about the subject;
- 0.25: identify concepts, criteria, or possible uses;
- 0.50: compare or evaluate alternatives without choosing one;
- 0.75: shortlist, select, or recommend an alternative;
- 1.00: acquire, adopt, deploy, execute, or implement a solution.

Realization plan:
{json.dumps(plan_payload, ensure_ascii=False, sort_keys=True)}

Express only the assigned readiness state and interpolate smoothly between the
anchors. Do not state the number. Do not add facts, source preferences, price,
budget, geography, company details, recency, urgency, sentiment, popularity,
authority, ratings, or another decision criterion. Return JSON only:
{{"objective_clause":"..."}}
"""


def generate_query_free_stimuli(
    requests: Sequence[QueryFreeGenerationRequest],
    *,
    generator: ObjectiveGenerator,
) -> tuple[QueryFreeStimulus, ...]:
    """Generate and compile one objective for every frozen request."""

    rows = []
    for request in requests:
        objective = _one_line(generator.generate(request))
        failures = query_free_contract_checks(request, objective)
        intent_text, masked_text, full_text, compiler_signature = _compile_views(
            request,
            objective,
        )
        identity = _canonical_json(
            {
                "version": QUERY_FREE_AXIS_VERSION,
                "request_id": request.request_id,
                "objective_clause": objective,
                "full_content_text": full_text,
            }
        )
        digest = _hash(identity)
        rows.append(
            QueryFreeStimulus(
                stimulus_id=f"qf-stimulus:{digest[:24]}",
                stimulus_hash=_hash(full_text),
                request_id=request.request_id,
                block_id=request.block_id,
                context_id=request.context.context_id,
                macrodomain=request.context.macrodomain,
                context_split=request.context.split,
                content_payload=request.context.payload,
                realization_plan_id=request.plan.plan_id,
                realization_plan_split=request.plan.split,
                assigned_a1=request.assigned_a1,
                level_index=request.level_index,
                generation_seed=request.generation_seed,
                generator_backend=generator.backend_name,
                generator_model=generator.model_name,
                objective_clause=objective,
                intent_only_text=intent_text,
                content_masked_text=masked_text,
                full_content_text=full_text,
                compiler_signature=compiler_signature,
                structural_valid=not failures,
                contract_failures=failures,
            )
        )
    _validate_compiled_population(rows)
    return tuple(rows)


def query_free_contract_checks(
    request: QueryFreeGenerationRequest,
    objective_clause: str,
) -> tuple[str, ...]:
    """Return frozen, machine-auditable structural contract failures."""

    objective = _one_line(objective_clause)
    failures: list[str] = []
    if objective.count(CONTENT_MARKER) != 1:
        failures.append("content-marker-count")
    if request.context.payload.casefold() in objective.casefold():
        failures.append("literal-content-generated")
    if re.search(r"\d", objective):
        failures.append("numeric-coordinate-leak")
    if _FORBIDDEN_CRITERION.search(objective):
        failures.append("off-axis-criterion")
    action_ready = bool(_ACTION_READY.search(objective))
    negated_action = bool(_NEGATED_ACTION.search(objective))
    if request.assigned_a1 == 0.0 and (
        not _INFORMATION_SEEKING.search(objective)
        or (action_ready and not negated_action)
    ):
        failures.append("low-anchor-mismatch")
    if request.assigned_a1 == 1.0 and not _ACTION_READY.search(objective):
        failures.append("high-anchor-mismatch")
    return tuple(failures)


def representation_texts(
    stimuli: Sequence[QueryFreeStimulus],
) -> dict[str, tuple[str, ...]]:
    """Return ordered texts for the three frozen LLM2Vec views."""

    rows = tuple(stimuli)
    return {
        "intent-only": tuple(item.intent_only_text for item in rows),
        "content-masked": tuple(item.content_masked_text for item in rows),
        "full-content": tuple(item.full_content_text for item in rows),
    }


def build_ordinal_judge_tasks(
    stimuli: Sequence[QueryFreeStimulus],
) -> tuple[OrdinalJudgeTask, ...]:
    """Create blinded single-stimulus rubric tasks without assigned coordinates."""

    return tuple(
        OrdinalJudgeTask(
            task_id=f"qf-ordinal:{_hash(QUERY_FREE_AXIS_VERSION + ':' + item.stimulus_id)[:24]}",
            stimulus_id=item.stimulus_id,
            prompt_text=item.full_content_text,
            rubric_version="decision-readiness-five-item-v1",
        )
        for item in stimuli
    )


def build_pairwise_judge_tasks(
    stimuli: Sequence[QueryFreeStimulus],
    *,
    master_seed: int = 20260817,
) -> tuple[tuple[PairwiseJudgeTask, ...], dict[str, dict[str, object]]]:
    """Create blinded adjacent/endpoint/nonadjacent comparisons and codebook."""

    grouped = _stimuli_by_block(stimuli)
    tasks: list[PairwiseJudgeTask] = []
    codebook: dict[str, dict[str, object]] = {}
    for block_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.assigned_a1)
        pairs: list[tuple[QueryFreeStimulus, QueryFreeStimulus, str]] = [
            (left, right, "adjacent") for left, right in zip(ordered, ordered[1:])
        ]
        pairs.append((ordered[0], ordered[-1], "endpoint"))
        candidates = [
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 2, len(ordered))
            if (left, right) != (0, len(ordered) - 1)
        ]
        candidates.sort(
            key=lambda pair: _hash(
                f"{QUERY_FREE_AXIS_VERSION}:{master_seed}:{block_id}:"
                f"{pair[0].stimulus_id}:{pair[1].stimulus_id}"
            )
        )
        for left, right in candidates[:2]:
            pairs.append((left, right, "nonadjacent"))
        for pair_index, (lower, upper, kind) in enumerate(pairs):
            reverse = _stable_uniform(master_seed, block_id, pair_index, kind) < 0.5
            left, right = (upper, lower) if reverse else (lower, upper)
            identity = _canonical_json(
                {
                    "version": QUERY_FREE_AXIS_VERSION,
                    "block_id": block_id,
                    "left": left.stimulus_id,
                    "right": right.stimulus_id,
                    "kind": kind,
                }
            )
            task_id = f"qf-pair:{_hash(identity)[:24]}"
            tasks.append(
                PairwiseJudgeTask(
                    task_id=task_id,
                    context_id=left.context_id,
                    left_stimulus_id=left.stimulus_id,
                    right_stimulus_id=right.stimulus_id,
                    left_prompt_text=left.full_content_text,
                    right_prompt_text=right.full_content_text,
                    comparison_kind=kind,
                    presentation_order="reverse" if reverse else "forward",
                )
            )
            codebook[task_id] = {
                "block_id": block_id,
                "left_assigned_a1": left.assigned_a1,
                "right_assigned_a1": right.assigned_a1,
                "expected_winner_stimulus_id": upper.stimulus_id,
            }
    by_context: dict[str, list[QueryFreeStimulus]] = {}
    for item in stimuli:
        by_context.setdefault(item.context_id, []).append(item)
    for context_id, group in sorted(by_context.items()):
        for assigned_a1 in (0.0, 1.0):
            matched = sorted(
                (item for item in group if item.assigned_a1 == assigned_a1),
                key=lambda item: item.realization_plan_id,
            )
            if len(matched) != 2:
                raise ValueError(
                    f"context {context_id} must expose both plans at A1={assigned_a1}"
                )
            reverse = _stable_uniform(
                master_seed,
                context_id,
                "same-a1-cross-plan",
                assigned_a1,
            ) < 0.5
            left, right = (matched[1], matched[0]) if reverse else tuple(matched)
            identity = _canonical_json(
                {
                    "version": QUERY_FREE_AXIS_VERSION,
                    "context_id": context_id,
                    "left": left.stimulus_id,
                    "right": right.stimulus_id,
                    "kind": "same-a1-cross-plan",
                }
            )
            task_id = f"qf-pair:{_hash(identity)[:24]}"
            tasks.append(
                PairwiseJudgeTask(
                    task_id=task_id,
                    context_id=context_id,
                    left_stimulus_id=left.stimulus_id,
                    right_stimulus_id=right.stimulus_id,
                    left_prompt_text=left.full_content_text,
                    right_prompt_text=right.full_content_text,
                    comparison_kind="same-a1-cross-plan",
                    presentation_order="reverse" if reverse else "forward",
                )
            )
            codebook[task_id] = {
                "block_id": None,
                "left_assigned_a1": assigned_a1,
                "right_assigned_a1": assigned_a1,
                "expected_winner_stimulus_id": None,
            }
    return tuple(tasks), codebook


def fit_query_free_axis(
    stimuli: Sequence[QueryFreeStimulus],
    embeddings_by_view: Mapping[str, np.ndarray],
    *,
    embedding_model: str,
) -> QueryFreeAxis:
    """Fit shared and view-specific blocked continuous-treatment directions."""

    rows, matrices = _validated_embeddings(stimuli, embeddings_by_view)
    grouped = _block_indices(rows)
    shared_numerator = np.zeros(next(iter(matrices.values())).shape[1], dtype=np.float64)
    shared_denominator = 0.0
    view_coefficients: dict[str, np.ndarray] = {}
    for view in REPRESENTATION_VIEWS:
        numerator = np.zeros_like(shared_numerator)
        denominator = 0.0
        matrix = matrices[view]
        for indices in grouped.values():
            assigned = np.asarray([rows[index].assigned_a1 for index in indices])
            centered_a = assigned - np.mean(assigned)
            centered_z = matrix[indices] - np.mean(matrix[indices], axis=0)
            numerator += np.sum(centered_a[:, None] * centered_z, axis=0)
            denominator += float(centered_a @ centered_a)
        if denominator <= 1e-12:
            raise ValueError(f"view {view} has degenerate assigned A1 variation")
        coefficient = numerator / denominator
        if np.linalg.norm(coefficient) <= 1e-12:
            raise ValueError(f"view {view} does not identify a nonzero direction")
        view_coefficients[view] = coefficient
        shared_numerator += numerator
        shared_denominator += denominator
    shared = shared_numerator / shared_denominator
    shared_norm = float(np.linalg.norm(shared))
    if shared_norm <= 1e-12:
        raise ValueError("pooled views do not identify a nonzero shared direction")
    shared_unit = shared / shared_norm
    view_rows = []
    for view in REPRESENTATION_VIEWS:
        coefficient = view_coefficients[view]
        unit = coefficient / np.linalg.norm(coefficient)
        view_rows.append(
            ViewDirection(
                representation_view=view,
                unnormalized_direction=tuple(float(value) for value in coefficient),
                unit_direction=tuple(float(value) for value in unit),
                cosine_with_shared=float(unit @ shared_unit),
            )
        )
    identity = _canonical_json(
        {
            "version": QUERY_FREE_AXIS_VERSION,
            "embedding_model": embedding_model,
            "stimulus_ids": [item.stimulus_id for item in rows],
            "shared_hash": _hash(shared.astype("<f8").tobytes()),
        }
    )
    return QueryFreeAxis(
        axis_id=f"query-free-axis:{_hash(identity)[:24]}",
        axis_version=QUERY_FREE_AXIS_VERSION,
        embedding_model=embedding_model,
        dimension=int(shared.shape[0]),
        block_count=len(grouped),
        stimulus_count=len(rows),
        representation_views=REPRESENTATION_VIEWS,
        shared_unnormalized_direction=tuple(float(value) for value in shared),
        shared_unit_direction=tuple(float(value) for value in shared_unit),
        view_directions=tuple(view_rows),
    )


def project_query_free_axis(
    axis: QueryFreeAxis,
    stimuli: Sequence[QueryFreeStimulus],
    embeddings_by_view: Mapping[str, np.ndarray],
) -> tuple[AxisCoordinate, ...]:
    """Project complete matched blocks without redefining assigned A1."""

    rows, matrices = _validated_embeddings(stimuli, embeddings_by_view)
    if next(iter(matrices.values())).shape[1] != axis.dimension:
        raise ValueError("embedding dimension does not match the frozen axis")
    direction = np.asarray(axis.shared_unnormalized_direction, dtype=np.float64)
    scale = float(direction @ direction)
    if scale <= 1e-12:
        raise ValueError("frozen shared direction is degenerate")
    grouped = _block_indices(rows)
    coordinates: list[AxisCoordinate] = []
    for view in REPRESENTATION_VIEWS:
        matrix = matrices[view]
        for indices in grouped.values():
            assigned = np.asarray([rows[index].assigned_a1 for index in indices])
            baseline = np.mean(matrix[indices], axis=0)
            mean_a = float(np.mean(assigned))
            for index in indices:
                displacement = matrix[index] - baseline
                delta = float(direction @ displacement / scale)
                raw = mean_a + delta
                residual = displacement - delta * direction
                item = rows[index]
                coordinates.append(
                    AxisCoordinate(
                        stimulus_id=item.stimulus_id,
                        block_id=item.block_id,
                        context_id=item.context_id,
                        realization_plan_id=item.realization_plan_id,
                        representation_view=view,
                        assigned_a1=item.assigned_a1,
                        raw_coordinate=raw,
                        absolute_assigned_coordinate_error=abs(raw - item.assigned_a1),
                        matched_off_axis_residual=float(np.linalg.norm(residual)),
                    )
                )
    return tuple(coordinates)


def measure_query_free_geometry(
    axis: QueryFreeAxis,
    stimuli: Sequence[QueryFreeStimulus],
    embeddings_by_view: Mapping[str, np.ndarray],
) -> QueryFreeGeometryDiagnostics:
    """Measure collinearity, monotonicity, and path stability without claims."""

    rows, matrices = _validated_embeddings(stimuli, embeddings_by_view)
    grouped = _block_indices(rows)
    coordinates = project_query_free_axis(axis, rows, matrices)
    coordinate_map = {
        (item.representation_view, item.stimulus_id): item for item in coordinates
    }
    centered_rows = []
    local_cosines = []
    correlations = []
    tortuosities = []
    shared_unit = np.asarray(axis.shared_unit_direction, dtype=np.float64)
    for view in REPRESENTATION_VIEWS:
        matrix = matrices[view]
        for indices in grouped.values():
            block = matrix[indices]
            centered_rows.extend(block - np.mean(block, axis=0))
            ordered = sorted(indices, key=lambda index: rows[index].assigned_a1)
            steps = np.diff(matrix[ordered], axis=0)
            step_norms = np.linalg.norm(steps, axis=1)
            local_cosines.extend(
                float(step @ shared_unit / max(norm, 1e-12))
                for step, norm in zip(steps, step_norms)
            )
            direct = float(np.linalg.norm(matrix[ordered[-1]] - matrix[ordered[0]]))
            tortuosities.append(float(np.sum(step_norms) / max(direct, 1e-12)))
            assigned = [rows[index].assigned_a1 for index in ordered]
            observed = [
                coordinate_map[(view, rows[index].stimulus_id)].raw_coordinate
                for index in ordered
            ]
            correlations.append(_spearman(assigned, observed))
    centered = np.asarray(centered_rows, dtype=np.float64)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = singular_values**2
    first_share = float(variances[0] / max(np.sum(variances), 1e-12))
    errors = [item.absolute_assigned_coordinate_error for item in coordinates]
    view_cosines = tuple(
        (item.representation_view, item.cosine_with_shared)
        for item in axis.view_directions
    )
    return QueryFreeGeometryDiagnostics(
        stimulus_count=len(rows),
        block_count=len(grouped),
        representation_count=len(REPRESENTATION_VIEWS),
        embedding_dimension=axis.dimension,
        treatment_first_component_share=first_share,
        median_local_to_global_cosine=float(np.median(local_cosines)),
        positive_local_to_global_cosine_rate=float(np.mean(np.asarray(local_cosines) > 0)),
        mean_block_spearman=float(np.mean(correlations)),
        minimum_block_spearman=float(np.min(correlations)),
        mean_path_tortuosity=float(np.mean(tortuosities)),
        mean_absolute_assigned_coordinate_error=float(np.mean(errors)),
        view_cosines_with_shared=view_cosines,
    )


class FakeQueryFreeObjectiveGenerator:
    """Deterministic semantic clauses for contract tests only."""

    backend_name = "fake-query-free-generator"
    model_name = "fake-query-free-generator-v1"
    clauses = (
        "Understand the fundamentals and uses of [CONTENT] without evaluating or selecting an option.",
        "Explore the concepts and possible uses of [CONTENT] before evaluating alternatives.",
        "Learn about [CONTENT] and identify practical criteria for a later evaluation.",
        "Compare approaches to [CONTENT] without choosing an option.",
        "Evaluate approaches to [CONTENT] and narrow the plausible alternatives.",
        "Shortlist and select an approach to [CONTENT] while preparing to act.",
        "Select, adopt, and implement an appropriate approach to [CONTENT].",
    )

    def generate(self, request: QueryFreeGenerationRequest) -> str:
        index = min(int(round(request.assigned_a1 * (len(self.clauses) - 1))), 6)
        return self.clauses[index]


def fake_query_free_embeddings(
    stimuli: Sequence[QueryFreeStimulus],
    *,
    dimension: int = 12,
    noise: float = 0.002,
) -> dict[str, np.ndarray]:
    """Return smooth synthetic views for smoke tests; never scientific data."""

    if dimension < 6:
        raise ValueError("fake embedding dimension must be at least six")
    rows = tuple(stimuli)
    result = {}
    for view_index, view in enumerate(REPRESENTATION_VIEWS):
        matrix = []
        for item in rows:
            base = np.zeros(dimension, dtype=np.float64)
            base[1] = 5.0
            base[2] = 0.1 * (_stable_uniform(item.context_id) - 0.5)
            base[3] = 0.1 * (_stable_uniform(item.realization_plan_id) - 0.5)
            base[4] = 0.05 * view_index
            base[0] = item.assigned_a1 - 0.5
            if noise:
                for index in range(5, dimension):
                    base[index] = noise * (
                        2 * _stable_uniform(item.stimulus_id, view, index) - 1
                    )
            matrix.append(base)
        result[view] = np.asarray(matrix, dtype=np.float64)
    return result


def _compile_views(
    request: QueryFreeGenerationRequest,
    objective: str,
) -> tuple[str, str, str, str]:
    masked = _render_instruction(request.plan, CONTENT_MARKER, objective)
    full_objective = objective.replace(CONTENT_MARKER, request.context.payload)
    full = _render_instruction(request.plan, request.context.payload, full_objective)
    signature = _hash(
        _render_instruction(request.plan, "<SUBJECT>", "<OBJECTIVE>")
    )
    return objective, masked, full, signature


def _render_instruction(plan: RealizationPlan, subject: str, objective: str) -> str:
    if plan.response_form == "question":
        task = f"What response would best satisfy this objective: {objective}"
    elif plan.syntax == "imperative":
        task = f"Address this objective: {objective}"
    elif plan.syntax == "request":
        task = f"Please address this objective: {objective}"
    else:
        task = f"The response should address this objective: {objective}"
    if plan.directness == "indirect":
        task = f"Use an indirect formulation. {task}"
    if plan.formality == "formal":
        task = f"Use a formal register. {task}"
    subject_block = f"Subject: {subject}"
    contract = "Response contract: provide a concise, evidence-oriented answer."
    blocks = (subject_block, task, contract)
    if plan.clause_order == "objective-first":
        blocks = (task, subject_block, contract)
    return "\n\n".join(blocks)


def _validate_compiled_population(stimuli: Sequence[QueryFreeStimulus]) -> None:
    rows = tuple(stimuli)
    if not rows:
        raise ValueError("compiled stimulus population must be non-empty")
    if len({item.stimulus_id for item in rows}) != len(rows):
        raise ValueError("compiled stimulus IDs must be unique")
    grouped = _stimuli_by_block(rows)
    for block_id, group in grouped.items():
        if len({item.content_payload for item in group}) != 1:
            raise ValueError(f"content payload changed within block {block_id}")
        if len({item.realization_plan_id for item in group}) != 1:
            raise ValueError(f"realization plan changed within block {block_id}")
        if len({item.compiler_signature for item in group}) != 1:
            raise ValueError(f"compiler-owned fields changed within block {block_id}")
        assigned = sorted(item.assigned_a1 for item in group)
        if len(set(assigned)) != len(assigned) or assigned[0] != 0.0 or assigned[-1] != 1.0:
            raise ValueError(f"block {block_id} must have unique A1 values and endpoints")
        hashes = [item.stimulus_hash for item in group]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"block {block_id} contains duplicate full stimuli")


def _validated_embeddings(stimuli, embeddings_by_view):
    rows = tuple(stimuli)
    if not rows:
        raise ValueError("stimuli must be non-empty")
    if any(not item.structural_valid for item in rows):
        raise ValueError("cannot fit an axis from structurally invalid stimuli")
    if set(embeddings_by_view) != set(REPRESENTATION_VIEWS):
        raise ValueError("embeddings must contain exactly the three frozen views")
    matrices = {
        view: _unit_rows(np.asarray(embeddings_by_view[view], dtype=np.float64))
        for view in REPRESENTATION_VIEWS
    }
    dimensions = {matrix.shape[1] for matrix in matrices.values()}
    if len(dimensions) != 1:
        raise ValueError("representation views must share one embedding dimension")
    if any(len(matrix) != len(rows) for matrix in matrices.values()):
        raise ValueError("every embedding view must align with all stimuli")
    _block_indices(rows)
    return rows, matrices


def _block_indices(stimuli: Sequence[QueryFreeStimulus]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(stimuli):
        grouped.setdefault(item.block_id, []).append(index)
    for block_id, indices in grouped.items():
        assigned = [stimuli[index].assigned_a1 for index in indices]
        if len(indices) < 3 or len(set(assigned)) != len(assigned):
            raise ValueError(f"block {block_id} lacks unique continuous A1 variation")
    return grouped


def _stimuli_by_block(
    stimuli: Sequence[QueryFreeStimulus],
) -> dict[str, list[QueryFreeStimulus]]:
    grouped: dict[str, list[QueryFreeStimulus]] = {}
    for item in stimuli:
        grouped.setdefault(item.block_id, []).append(item)
    return grouped


def _unit_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a non-empty finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings contain a zero-norm row")
    return values / norms


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        raise ValueError("Spearman input is constant")
    return float(left_centered @ right_centered / denominator)


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _one_line(value: str) -> str:
    return " ".join(str(value).split())
