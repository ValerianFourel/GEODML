"""Closed-loop LLM2Vec-Gen proposals for frozen readiness targets.

The frozen readiness map and the decodable LLM2Vec-Gen reconstruction state are
different spaces.  This module therefore treats latent steering as a proposal
mechanism only: every decoded question is anchored to the exact keyword,
validated, and measured again in the frozen readiness map before it can be
selected.

Model loading is deliberately kept out of this module.  Small protocols make
the controller CPU-testable and allow the command-line adapter to own the GPU
and API dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Protocol, Sequence

import numpy as np

from .llm2vec_gen_axis import (
    build_realization_reconstruction_text,
    clean_decoded_realization,
    stable_array_hash,
)
from .readiness_prompt_population import validate_generated_question


LATENT_FEEDBACK_VERSION = "llm2vec-gen-readiness-feedback-v1"


class ReconstructionBackend(Protocol):
    """Minimal LLM2Vec-Gen reconstruction interface."""

    model_name: str

    def encode(
        self, texts: Sequence[str], *, batch_size: int, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def decode(self, state: np.ndarray, *, max_new_tokens: int) -> str: ...


class ReadinessCoordinateScorer(Protocol):
    """Frozen pooled-embedding scorer; coordinates must be normalized."""

    model_name: str

    def score(self, texts: Sequence[str]) -> np.ndarray: ...


class SemanticQuestionValidator(Protocol):
    """Independent semantic check after deterministic validation."""

    model_name: str

    def review(self, question: str, keyword: str) -> tuple[bool, str]: ...


@dataclass(frozen=True, slots=True)
class LatentCoordinateBridge:
    """Affine local bridge from readiness-coordinate changes to state changes."""

    bridge_version: str
    coordinate_mean: np.ndarray
    state_mean: np.ndarray
    directions: np.ndarray
    ridge_penalty: float
    calibration_item_count: int
    state_shape: tuple[int, ...]
    coordinate_condition_number: float
    state_reconstruction_rmse: float
    bridge_hash: str


@dataclass(frozen=True, slots=True)
class LatentFeedbackAttempt:
    """One decoded and re-measured proposal in the feedback loop."""

    round_index: int
    variant_index: int
    step_scale: float
    parent_question: str
    parent_normalized_axis_1: float
    parent_normalized_axis_2: float
    requested_step_axis_1: float
    requested_step_axis_2: float
    source_state_hash: str
    proposed_state_hash: str
    raw_decoded_text: str
    question: str | None
    hard_valid: bool
    semantic_valid: bool
    validation_reason: str
    observed_normalized_axis_1: float | None
    observed_normalized_axis_2: float | None
    target_distance: float | None


@dataclass(frozen=True, slots=True)
class LatentFeedbackResult:
    """Best valid proposal and the complete bounded search trace."""

    initial_question: str
    target_normalized_axis_1: float
    target_normalized_axis_2: float
    best_question: str | None
    best_normalized_axis_1: float | None
    best_normalized_axis_2: float | None
    best_target_distance: float | None
    accepted_within_tolerance: bool
    stop_reason: str
    completed_round_count: int
    attempts: tuple[LatentFeedbackAttempt, ...]


def fit_latent_coordinate_bridge(
    reconstruction_states: np.ndarray,
    normalized_coordinates: np.ndarray,
    *,
    ridge_penalty: float = 1e-3,
    minimum_items: int = 10,
) -> LatentCoordinateBridge:
    """Fit state changes associated with the two frozen readiness coordinates.

    Calibration rows must come from the frozen development corpus.  The fitted
    directions predict reconstruction state from coordinates; they do not claim
    that LLM2Vec-Gen and the frozen readiness map share a coordinate system.
    """

    states = np.asarray(reconstruction_states, dtype=np.float64)
    coordinates = np.asarray(normalized_coordinates, dtype=np.float64)
    if minimum_items < 3:
        raise ValueError("minimum_items must be at least three")
    if states.ndim < 2 or states.shape[0] < minimum_items:
        raise ValueError(f"at least {minimum_items} calibration states are required")
    if coordinates.shape != (states.shape[0], 2):
        raise ValueError("normalized coordinates must have shape (items, 2)")
    if ridge_penalty <= 0 or not math.isfinite(ridge_penalty):
        raise ValueError("ridge_penalty must be positive and finite")
    if not np.isfinite(states).all() or not np.isfinite(coordinates).all():
        raise ValueError("calibration arrays must be finite")
    if np.any((coordinates < 0.0) | (coordinates > 1.0)):
        raise ValueError("calibration coordinates must lie in [0, 1]")

    coordinate_mean = coordinates.mean(axis=0)
    centered_coordinates = coordinates - coordinate_mean
    gram = centered_coordinates.T @ centered_coordinates
    if np.linalg.matrix_rank(gram, tol=1e-10) < 2:
        raise ValueError("calibration coordinates do not identify two directions")
    regularized = gram + ridge_penalty * np.eye(2, dtype=np.float64)
    condition_number = float(np.linalg.cond(regularized))
    if not math.isfinite(condition_number):
        raise ValueError("calibration coordinate system is singular")

    state_shape = tuple(int(value) for value in states.shape[1:])
    flattened = states.reshape(states.shape[0], -1)
    state_mean_flat = flattened.mean(axis=0)
    centered_states = flattened - state_mean_flat
    coefficients = np.linalg.solve(
        regularized, centered_coordinates.T @ centered_states
    )
    fitted = centered_coordinates @ coefficients
    rmse = float(np.sqrt(np.mean((centered_states - fitted) ** 2)))
    directions = coefficients.reshape((2, *state_shape))
    state_mean = state_mean_flat.reshape(state_shape)

    digest = hashlib.sha256()
    digest.update(LATENT_FEEDBACK_VERSION.encode("utf-8"))
    digest.update(str(states.shape).encode("ascii"))
    digest.update(np.asarray(coordinate_mean, dtype="<f4").tobytes())
    digest.update(np.asarray(directions, dtype="<f4").tobytes())
    return LatentCoordinateBridge(
        bridge_version=LATENT_FEEDBACK_VERSION,
        coordinate_mean=coordinate_mean,
        state_mean=state_mean,
        directions=directions,
        ridge_penalty=float(ridge_penalty),
        calibration_item_count=int(states.shape[0]),
        state_shape=state_shape,
        coordinate_condition_number=condition_number,
        state_reconstruction_rmse=rmse,
        bridge_hash=digest.hexdigest(),
    )


def steer_reconstruction_state(
    state: np.ndarray,
    bridge: LatentCoordinateBridge,
    *,
    observed_coordinates: Sequence[float],
    target_coordinates: Sequence[float],
    gain: float = 1.0,
    coordinate_step_limit: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a bounded bridge step while retaining the state's residual content."""

    source = np.asarray(state, dtype=np.float64)
    observed = _coordinate_pair(
        observed_coordinates, name="observed_coordinates", require_unit=False
    )
    target = _coordinate_pair(target_coordinates, name="target_coordinates")
    if source.shape != bridge.state_shape or not np.isfinite(source).all():
        raise ValueError(f"state must be finite with shape {bridge.state_shape}")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be positive and finite")
    if not math.isfinite(coordinate_step_limit) or coordinate_step_limit <= 0:
        raise ValueError("coordinate_step_limit must be positive and finite")
    requested_step = np.clip(
        gain * (target - observed), -coordinate_step_limit, coordinate_step_limit
    )
    state_delta = np.einsum("a,a...->...", requested_step, bridge.directions)
    proposed = source + state_delta
    if not np.isfinite(proposed).all():
        raise ValueError("latent steering produced non-finite state values")
    return proposed, requested_step


def anchor_exact_keyword_question(decoded_text: str, keyword: str) -> str:
    """Return a one-line question containing the exact keyword phrase.

    The anchoring operation is deterministic and deliberately small.  It does
    not make the candidate acceptable: the final text is still passed through
    hard validation, an independent semantic review, and frozen re-embedding.
    """

    normalized_keyword = " ".join(str(keyword).split())
    if not normalized_keyword:
        raise ValueError("keyword must be non-empty")
    cleaned = clean_decoded_realization(decoded_text)
    if cleaned.startswith("{"):
        # LLM2Vec-Gen sometimes reconstructs a generator's JSON envelope.  Only
        # the simple one-field form is accepted here; malformed JSON fails later.
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if (
            isinstance(payload, dict)
            and set(payload) == {"question"}
            and isinstance(payload["question"], str)
        ):
            cleaned = payload["question"]
    question = " ".join(cleaned.split()).strip(' "')
    question = re.sub(r"^(?:question|search question)\s*:\s*", "", question, flags=re.I)
    if not question:
        raise ValueError("decoded text contains no question")
    if not question.endswith("?"):
        question = question.rstrip(".!;:") + "?"
    if normalized_keyword not in question:
        question = f"Regarding {normalized_keyword}, {question[0].lower()}{question[1:]}"
    return question


def run_latent_feedback(
    *,
    initial_question: str,
    keyword: str,
    target_coordinates: Sequence[float],
    bridge: LatentCoordinateBridge,
    reconstruction_backend: ReconstructionBackend,
    scorer: ReadinessCoordinateScorer,
    validator: SemanticQuestionValidator,
    maximum_rounds: int = 3,
    step_scales: Sequence[float] = (0.5, 1.0, 1.5),
    distance_tolerance: float = 0.12,
    coordinate_step_limit: float = 0.35,
    encode_batch_size: int = 8,
    encode_max_length: int = 512,
    decode_max_new_tokens: int = 96,
) -> LatentFeedbackResult:
    """Run bounded generate-measure-steer-decode-validate feedback for one seed."""

    validate_generated_question(initial_question, keyword)
    target = _coordinate_pair(target_coordinates, name="target_coordinates")
    if maximum_rounds < 0 or not step_scales:
        raise ValueError("maximum_rounds must be nonnegative and step_scales nonempty")
    scales = tuple(float(value) for value in step_scales)
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("step scales must be positive and finite")
    if distance_tolerance < 0 or coordinate_step_limit <= 0:
        raise ValueError("invalid feedback tolerances")

    initial_coordinate = _score_one(scorer, initial_question)
    initial_semantic_valid, initial_reason = validator.review(initial_question, keyword)
    best_question = initial_question if initial_semantic_valid else None
    best_coordinate = initial_coordinate if initial_semantic_valid else None
    best_distance = (
        float(np.linalg.norm(initial_coordinate - target))
        if initial_semantic_valid
        else None
    )
    if best_distance is not None and best_distance <= distance_tolerance:
        return LatentFeedbackResult(
            initial_question=initial_question,
            target_normalized_axis_1=float(target[0]),
            target_normalized_axis_2=float(target[1]),
            best_question=best_question,
            best_normalized_axis_1=float(best_coordinate[0]),
            best_normalized_axis_2=float(best_coordinate[1]),
            best_target_distance=best_distance,
            accepted_within_tolerance=True,
            stop_reason="initial-question-within-tolerance",
            completed_round_count=0,
            attempts=(),
        )

    current_question = initial_question
    current_coordinate = initial_coordinate
    current_semantic_valid = initial_semantic_valid
    attempts: list[LatentFeedbackAttempt] = []
    stop_reason = (
        "maximum-rounds-exhausted"
        if initial_semantic_valid
        else f"initial-semantic-review-failed: {initial_reason}"
    )
    completed_rounds = 0
    for round_index in range(1, maximum_rounds + 1):
        _, states = reconstruction_backend.encode(
            [build_realization_reconstruction_text(current_question)],
            batch_size=encode_batch_size,
            max_length=encode_max_length,
        )
        states = np.asarray(states, dtype=np.float64)
        if states.shape != (1, *bridge.state_shape):
            raise ValueError(
                "reconstruction backend returned unexpected state shape: "
                f"{states.shape} != {(1, *bridge.state_shape)}"
            )
        source_state = states[0]
        source_hash = stable_array_hash(source_state)
        decoded_rows: list[tuple[int, float, np.ndarray, np.ndarray, str]] = []
        for variant_index, scale in enumerate(scales):
            proposed_state, requested_step = steer_reconstruction_state(
                source_state,
                bridge,
                observed_coordinates=current_coordinate,
                target_coordinates=target,
                gain=scale,
                coordinate_step_limit=coordinate_step_limit,
            )
            raw = reconstruction_backend.decode(
                proposed_state, max_new_tokens=decode_max_new_tokens
            )
            decoded_rows.append(
                (variant_index, scale, proposed_state, requested_step, raw)
            )

        prepared: list[tuple[int, float, np.ndarray, np.ndarray, str, str]] = []
        failed: list[LatentFeedbackAttempt] = []
        for variant_index, scale, proposed_state, requested_step, raw in decoded_rows:
            try:
                question = anchor_exact_keyword_question(raw, keyword)
                validate_generated_question(question, keyword)
            except ValueError as exc:
                failed.append(
                    LatentFeedbackAttempt(
                        round_index=round_index,
                        variant_index=variant_index,
                        step_scale=scale,
                        parent_question=current_question,
                        parent_normalized_axis_1=float(current_coordinate[0]),
                        parent_normalized_axis_2=float(current_coordinate[1]),
                        requested_step_axis_1=float(requested_step[0]),
                        requested_step_axis_2=float(requested_step[1]),
                        source_state_hash=source_hash,
                        proposed_state_hash=stable_array_hash(proposed_state),
                        raw_decoded_text=raw,
                        question=None,
                        hard_valid=False,
                        semantic_valid=False,
                        validation_reason=str(exc),
                        observed_normalized_axis_1=None,
                        observed_normalized_axis_2=None,
                        target_distance=None,
                    )
                )
                continue
            prepared.append(
                (variant_index, scale, proposed_state, requested_step, raw, question)
            )
        attempts.extend(failed)
        if not prepared:
            stop_reason = "all-decoded-variants-failed-hard-validation"
            completed_rounds = round_index
            break

        measured = _score_many(scorer, [row[-1] for row in prepared])
        valid_this_round: list[tuple[float, str, np.ndarray]] = []
        for row, observed in zip(prepared, measured):
            variant_index, scale, proposed_state, requested_step, raw, question = row
            semantic_valid, reason = validator.review(question, keyword)
            distance = float(np.linalg.norm(observed - target))
            attempts.append(
                LatentFeedbackAttempt(
                    round_index=round_index,
                    variant_index=variant_index,
                    step_scale=scale,
                    parent_question=current_question,
                    parent_normalized_axis_1=float(current_coordinate[0]),
                    parent_normalized_axis_2=float(current_coordinate[1]),
                    requested_step_axis_1=float(requested_step[0]),
                    requested_step_axis_2=float(requested_step[1]),
                    source_state_hash=source_hash,
                    proposed_state_hash=stable_array_hash(proposed_state),
                    raw_decoded_text=raw,
                    question=question,
                    hard_valid=True,
                    semantic_valid=semantic_valid,
                    validation_reason=reason,
                    observed_normalized_axis_1=float(observed[0]),
                    observed_normalized_axis_2=float(observed[1]),
                    target_distance=distance,
                )
            )
            if semantic_valid:
                valid_this_round.append((distance, question, observed))

        completed_rounds = round_index
        if not valid_this_round:
            stop_reason = "all-decoded-variants-failed-semantic-validation"
            break
        round_distance, round_question, round_coordinate = min(
            valid_this_round, key=lambda row: (row[0], row[1])
        )
        if best_distance is None or round_distance < best_distance:
            best_question = round_question
            best_coordinate = round_coordinate
            best_distance = round_distance
        if round_distance <= distance_tolerance:
            stop_reason = "decoded-question-within-tolerance"
            break
        current_distance = float(np.linalg.norm(current_coordinate - target))
        if current_semantic_valid and round_distance >= current_distance - 1e-8:
            stop_reason = "no-target-distance-progress"
            break
        if float(np.linalg.norm(round_coordinate - current_coordinate)) <= 1e-8:
            stop_reason = "no-measured-coordinate-progress"
            break
        current_question = round_question
        current_coordinate = round_coordinate
        current_semantic_valid = True
        stop_reason = "maximum-rounds-exhausted"

    accepted = best_distance is not None and best_distance <= distance_tolerance
    return LatentFeedbackResult(
        initial_question=initial_question,
        target_normalized_axis_1=float(target[0]),
        target_normalized_axis_2=float(target[1]),
        best_question=best_question,
        best_normalized_axis_1=(
            float(best_coordinate[0]) if best_coordinate is not None else None
        ),
        best_normalized_axis_2=(
            float(best_coordinate[1]) if best_coordinate is not None else None
        ),
        best_target_distance=best_distance,
        accepted_within_tolerance=accepted,
        stop_reason=stop_reason,
        completed_round_count=completed_rounds,
        attempts=tuple(attempts),
    )


def _coordinate_pair(
    values: Sequence[float], *, name: str, require_unit: bool = True
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain two finite values")
    if require_unit and np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _score_one(scorer: ReadinessCoordinateScorer, text: str) -> np.ndarray:
    return _score_many(scorer, [text])[0]


def _score_many(
    scorer: ReadinessCoordinateScorer, texts: Sequence[str]
) -> np.ndarray:
    coordinates = np.asarray(scorer.score(texts), dtype=np.float64)
    if coordinates.shape != (len(texts), 2):
        raise ValueError("readiness scorer must return shape (texts, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("readiness scorer returned non-finite coordinates")
    return coordinates
