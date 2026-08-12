"""Decodable informational-to-transactional axes for LLM2Vec-Gen.

This module contains only small, deterministic numerical operations.  Loading
McGill-NLP/LLM2Vec-Gen-Qwen3-8B and running GPU inference are deliberately
owned by the command-line adapter in ``analysis/scripts``.

The representation manipulated here is the reconstruction hidden state with
shape ``(compression_tokens, hidden_dim)``.  The pooled retrieval embedding is
also diagnosed, but it is not decoded and does not define the experimental
coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

import numpy as np


LLM2VEC_GEN_AXIS_VERSION = "llm2vec-gen-search-purpose-axis-v1"
ENCODING_INSTRUCTION_VERSION = "reranking-template-reconstruction-v1"
QUERY_CONDITIONED_ENDPOINT_VERSION = "query-conditioned-search-purpose-v1"
ENCODING_INSTRUCTION = (
    "Generate the reusable listwise search-reranking instruction given below. "
    "Preserve its meaning and the literal placeholders {QUERY}, {CANDIDATES}, "
    "and {TOP_N}. Output only the instruction.\n\nInstruction:\n"
)

__all__ = [
    "ENCODING_INSTRUCTION",
    "ENCODING_INSTRUCTION_VERSION",
    "LLM2VEC_GEN_AXIS_VERSION",
    "QUERY_CONDITIONED_ENDPOINT_VERSION",
    "DecodableAxis",
    "axis_geometry_diagnostics",
    "build_decodable_axis",
    "build_encoding_text",
    "build_query_conditioned_requests",
    "decode_record_checks",
    "inject_query_after_decode",
    "interpolate_axis_centroids",
    "interpolate_endpoint_pair",
    "project_onto_axis",
    "stable_array_hash",
]


@dataclass(frozen=True, slots=True)
class DecodableAxis:
    """A calibrated line in reconstruction-state space.

    Arrays retain the native ``(compression_tokens, hidden_dim)`` shape so they
    can be passed back to LLM2Vec-Gen after interpolation.
    """

    axis_version: str
    informational_centroid: np.ndarray
    transactional_centroid: np.ndarray
    direction_unit: np.ndarray
    centroid_distance: float
    endpoint_pair_count: int
    state_shape: tuple[int, ...]
    axis_hash: str


def build_encoding_text(prompt_template: str) -> str:
    """Wrap an endpoint as the generation task expected by LLM2Vec-Gen."""

    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValueError("prompt_template must be a non-empty string")
    return ENCODING_INSTRUCTION + prompt_template.strip()


def build_query_conditioned_requests(query: str) -> tuple[str, str]:
    """Build a topic-matched pair with the exact query in both endpoints.

    These are direct generation requests rather than reusable reranking
    templates. Only search purpose changes: the informational endpoint asks
    to learn and understand, while the transactional endpoint asks to choose
    and begin acting now. Ranking structure is deliberately left for a later
    deterministic wrapper if this smaller latent-decoding test succeeds.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = " ".join(query.split())
    if '"' in normalized:
        raise ValueError("query must not contain double quotes")
    quoted = f'"{normalized}"'
    informational = (
        f"For the fixed search topic {quoted}, explain how it works and what "
        "approaches are available so the user can learn and understand it."
    )
    transactional = (
        f"For the fixed search topic {quoted}, help the user choose a suitable "
        "approach and begin implementing it now."
    )
    return informational, transactional


def _validated_paired_states(
    informational_states: np.ndarray,
    transactional_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    informational = np.asarray(informational_states, dtype=np.float64)
    transactional = np.asarray(transactional_states, dtype=np.float64)
    if informational.ndim < 2:
        raise ValueError("endpoint states must have shape (pairs, ...)")
    if informational.shape != transactional.shape:
        raise ValueError(
            "informational and transactional state shapes differ: "
            f"{informational.shape} != {transactional.shape}"
        )
    if informational.shape[0] == 0:
        raise ValueError("at least one endpoint pair is required")
    if not np.isfinite(informational).all() or not np.isfinite(transactional).all():
        raise ValueError("endpoint states contain non-finite values")
    return informational, transactional


def build_decodable_axis(
    informational_states: np.ndarray,
    transactional_states: np.ndarray,
    *,
    axis_version: str = LLM2VEC_GEN_AXIS_VERSION,
) -> DecodableAxis:
    """Estimate the paired mean direction in reconstruction-state space."""

    if axis_version != LLM2VEC_GEN_AXIS_VERSION:
        raise ValueError(f"unsupported axis version: {axis_version!r}")
    informational, transactional = _validated_paired_states(
        informational_states, transactional_states
    )
    informational_centroid = informational.mean(axis=0)
    transactional_centroid = transactional.mean(axis=0)
    direction = transactional_centroid - informational_centroid
    distance = float(np.linalg.norm(direction.reshape(-1)))
    if not math.isfinite(distance) or distance <= 1e-12:
        raise ValueError("endpoint states do not define a nonzero direction")
    direction_unit = direction / distance
    identity = np.concatenate(
        [
            informational_centroid.reshape(-1),
            transactional_centroid.reshape(-1),
        ]
    ).astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(axis_version.encode("utf-8"))
    digest.update(str(informational.shape).encode("ascii"))
    digest.update(identity.tobytes(order="C"))
    return DecodableAxis(
        axis_version=axis_version,
        informational_centroid=informational_centroid,
        transactional_centroid=transactional_centroid,
        direction_unit=direction_unit,
        centroid_distance=distance,
        endpoint_pair_count=int(informational.shape[0]),
        state_shape=tuple(int(value) for value in informational.shape[1:]),
        axis_hash=digest.hexdigest(),
    )


def interpolate_axis_centroids(axis: DecodableAxis, coordinate: float) -> np.ndarray:
    """Return the point at ``coordinate`` on the two endpoint centroids."""

    value = _coordinate(coordinate)
    return (
        axis.informational_centroid
        + value * (axis.transactional_centroid - axis.informational_centroid)
    )


def interpolate_endpoint_pair(
    informational_state: np.ndarray,
    transactional_state: np.ndarray,
    coordinate: float,
) -> np.ndarray:
    """Interpolate within a topic-matched endpoint pair."""

    informational = np.asarray(informational_state, dtype=np.float64)
    transactional = np.asarray(transactional_state, dtype=np.float64)
    if informational.shape != transactional.shape or informational.ndim < 1:
        raise ValueError("paired endpoint states must have the same non-empty shape")
    if not np.isfinite(informational).all() or not np.isfinite(transactional).all():
        raise ValueError("paired endpoint states contain non-finite values")
    value = _coordinate(coordinate)
    return informational + value * (transactional - informational)


def project_onto_axis(axis: DecodableAxis, states: np.ndarray) -> np.ndarray:
    """Project states to coordinates calibrated to centroid means 0 and 1."""

    array = np.asarray(states, dtype=np.float64)
    single = array.shape == axis.state_shape
    if single:
        array = array.reshape((1, *axis.state_shape))
    if array.ndim != len(axis.state_shape) + 1 or tuple(array.shape[1:]) != axis.state_shape:
        raise ValueError(
            f"expected state shape (n, {axis.state_shape}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("states contain non-finite values")
    centered = array - axis.informational_centroid
    projections = np.einsum(
        "ni,i->n", centered.reshape(array.shape[0], -1), axis.direction_unit.reshape(-1)
    )
    coordinates = projections / axis.centroid_distance
    return coordinates[0] if single else coordinates


def axis_geometry_diagnostics(
    informational_states: np.ndarray,
    transactional_states: np.ndarray,
) -> dict[str, object]:
    """Measure paired coherence and leave-one-pair-out generalization.

    A global positive gap is partly mechanical because it is measured along the
    mean paired displacement.  Leave-one-pair-out gaps are therefore the main
    geometric diagnostic: the held-out pair never contributes to its direction.
    """

    informational, transactional = _validated_paired_states(
        informational_states, transactional_states
    )
    pair_count = informational.shape[0]
    flattened_info = informational.reshape(pair_count, -1)
    flattened_trans = transactional.reshape(pair_count, -1)
    deltas = flattened_trans - flattened_info
    axis = build_decodable_axis(informational, transactional)
    unit = axis.direction_unit.reshape(-1)
    delta_norms = np.linalg.norm(deltas, axis=1)
    if np.any(delta_norms <= 1e-12):
        raise ValueError("at least one endpoint pair has zero displacement")
    cosines = (deltas @ unit) / delta_norms
    global_gaps = (deltas @ unit) / axis.centroid_distance

    loo_rows: list[dict[str, object]] = []
    if pair_count >= 2:
        for held_out in range(pair_count):
            training = np.delete(deltas, held_out, axis=0)
            training_direction = training.mean(axis=0)
            training_norm = float(np.linalg.norm(training_direction))
            if training_norm <= 1e-12:
                cosine = None
                gap = None
                positive = False
            else:
                training_unit = training_direction / training_norm
                delta = deltas[held_out]
                gap = float(delta @ training_unit)
                cosine = float(gap / delta_norms[held_out])
                positive = gap > 0.0
            loo_rows.append(
                {
                    "held_out_pair_index": held_out,
                    "cosine_to_training_direction": cosine,
                    "signed_gap": gap,
                    "positive_direction": positive,
                }
            )

    loo_positive = [bool(row["positive_direction"]) for row in loo_rows]
    loo_cosines = [
        float(row["cosine_to_training_direction"])
        for row in loo_rows
        if row["cosine_to_training_direction"] is not None
    ]
    return {
        "pair_count": int(pair_count),
        "state_shape": [int(value) for value in informational.shape[1:]],
        "flattened_dimension": int(flattened_info.shape[1]),
        "centroid_distance": axis.centroid_distance,
        "pair_direction_cosines": [float(value) for value in cosines],
        "pair_direction_cosine_mean": float(np.mean(cosines)),
        "pair_direction_cosine_min": float(np.min(cosines)),
        "pair_calibrated_gaps": [float(value) for value in global_gaps],
        "all_pairs_positive_on_global_axis": bool(np.all(global_gaps > 0.0)),
        "leave_one_pair_out": loo_rows,
        "leave_one_pair_out_positive_rate": (
            float(np.mean(loo_positive)) if loo_positive else None
        ),
        "leave_one_pair_out_cosine_mean": (
            float(np.mean(loo_cosines)) if loo_cosines else None
        ),
    }


def decode_record_checks(text: str) -> dict[str, object]:
    """Return structural observations without declaring semantic validity."""

    value = text.strip()
    placeholders = {
        name: token in value
        for name, token in (
            ("query", "{QUERY}"),
            ("candidates", "{CANDIDATES}"),
            ("top_n", "{TOP_N}"),
        )
    }
    lowered = value.lower()
    off_axis_terms = [
        label
        for label, pattern in (
            ("first-party", r"\bfirst[- ]party\b"),
            ("commerciality", r"\bcommercial(?:ity)?\b"),
            ("authority", r"\bauthorit(?:y|ative)\b"),
            ("freshness", r"\bfresh(?:ness)?\b|\brecen(?:t|cy)\b"),
            ("citations", r"\bcitations?\b"),
            ("popularity", r"\bpopular(?:ity)?\b"),
            ("brand-fame", r"\bbrand fame\b"),
            ("writing-quality", r"\bwriting quality\b"),
        )
        if re.search(pattern, lowered)
    ]
    informational_cues = _cue_count(
        lowered, ("learn", "understand", "explain", "overview", "information")
    )
    transactional_cues = _cue_count(
        lowered,
        (
            "select",
            "choose",
            "configure",
            "start",
            "deploy",
            "download",
            "register",
            "complete",
            "action",
            "now",
        ),
    )
    return {
        "nonempty": bool(value),
        "character_count": len(value),
        "placeholders": placeholders,
        "all_placeholders_preserved": all(placeholders.values()),
        "mentions_identifier_only": (
            "identifier" in lowered
            and ("only" in lowered or "nothing else" in lowered)
        ),
        "prohibits_explanation": (
            "no explanation" in lowered
            or "without explanation" in lowered
            or "do not explain" in lowered
        ),
        "detected_off_axis_terms": off_axis_terms,
        "informational_lexical_cue_count": informational_cues,
        "transactional_lexical_cue_count": transactional_cues,
        "lexical_cues_are_diagnostic_only": True,
    }


def inject_query_after_decode(decoded_template: str, query: str) -> str:
    """Insert the query only after latent reconstruction.

    Requiring the placeholder prevents a silent change from post-decoding
    injection to query-conditioned latent construction.
    """

    if "{QUERY}" not in decoded_template:
        raise ValueError("decoded template does not preserve {QUERY}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return decoded_template.replace("{QUERY}", query.strip())


def stable_array_hash(array: np.ndarray) -> str:
    """Hash a numeric array after canonical float32 little-endian conversion."""

    value = np.asarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _coordinate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("coordinate must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("coordinate must be in [0, 1]")
    return result


def _cue_count(text: str, cues: tuple[str, ...]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(cue)}\w*\b", text)) for cue in cues)
