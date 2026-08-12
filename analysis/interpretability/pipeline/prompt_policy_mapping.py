"""Locate policy prompts in latent space and link them to ranking permutations.

The experimental path implemented here is::

    (assigned B, surface seed S) -> deterministic prompt P = G(B, S)
        -> embed P and validate its latent location
        -> rerank one frozen candidate set with P
        -> record the strict candidate-ID permutation Y

The embedding projection is diagnostic.  It never replaces ``assigned_bias``
and no embedding vector is decoded into a final prompt.  Legacy prompt variants
and reranking code are intentionally left unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Protocol, Sequence

import numpy as np

from .prompt_continuum import PromptRecord, TemplatePromptGenerator
from .search_purpose_continuum import (
    CandidateBinding,
    RankingPermutation,
    SearchCandidate,
    SearchPurposePromptRecord,
    parse_ranking_permutation,
    render_search_purpose_prompt,
)

POLICY_LATENT_MAPPING_VERSION = "first-party-policy-latent-map-v1"

__all__ = [
    "FakePolicyPromptEmbedder",
    "PolicyLatentAxis",
    "PolicyLatentValidation",
    "PolicyPermutationOutcome",
    "PolicyPromptLocation",
    "PolicyPromptEmbedder",
    "PolicyStyleOrigin",
    "PolicyMappingArtifacts",
    "RenderedPolicyPrompt",
    "build_policy_latent_axis",
    "locate_policy_prompts",
    "map_policy_prompt_to_permutation",
    "render_policy_prompt",
    "validate_policy_prompt_locations",
    "write_policy_mapping_artifacts",
]


class PolicyPromptEmbedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one finite embedding row per prompt text."""


@dataclass(frozen=True, slots=True)
class PolicyStyleOrigin:
    """Matched midpoint for one surface seed, used only for drift checks."""

    style_seed: int
    midpoint: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PolicyLatentAxis:
    """Frozen first-party policy direction fitted from matched prompt endpoints."""

    axis_id: str
    axis_version: str
    embedding_model: str
    dimension: int
    direction: tuple[float, ...]
    neutral_anchor: float
    strong_preference_anchor: float
    neutral_centroid: tuple[float, ...]
    strong_preference_centroid: tuple[float, ...]
    endpoint_style_seeds: tuple[int, ...]
    style_origins: tuple[PolicyStyleOrigin, ...]
    pair_direction_cosines: tuple[float, ...]
    leave_one_style_out_positive_rate: float | None


@dataclass(frozen=True, slots=True)
class PolicyPromptLocation:
    """Observed embedding diagnostics for one prompt with assigned ``B``."""

    prompt_assignment_id: str
    prompt_id: str
    prompt_hash: str
    assigned_bias: float
    style_seed: int
    axis_id: str
    embedding_model: str
    observed_axis_coordinate: float
    absolute_assigned_coordinate_error: float
    matched_off_axis_residual: float
    prompt_embedding: tuple[float, ...]
    embedding_hash: str


@dataclass(frozen=True, slots=True)
class PolicyLatentValidation:
    """Pre-specified acceptance checks; not a redefinition of treatment."""

    max_absolute_coordinate_error: float
    max_matched_off_axis_residual: float
    monotonic_tolerance: float
    prompt_count: int
    rejected_prompt_ids: tuple[str, ...]
    nonmonotonic_style_seeds: tuple[int, ...]
    all_points_within_thresholds: bool
    all_style_trajectories_monotonic: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class RenderedPolicyPrompt:
    """One deterministic policy prompt bound to a frozen reranking problem."""

    prompt_assignment_id: str
    prompt_instance_id: str
    rendered_prompt_hash: str
    prompt_id: str
    prompt_hash: str
    assigned_bias: float
    style_seed: int
    keyword: str
    top_n: int
    candidate_set_id: str
    candidates: tuple[CandidateBinding, ...]
    rendered_prompt: str
    prompt_space_version: str


@dataclass(frozen=True, slots=True)
class PolicyPermutationOutcome:
    """Traceable mapping from assigned policy and latent location to outcome."""

    prompt_assignment_id: str
    outcome_id: str
    mapping_version: str
    reranker_run_id: str
    reranker_model: str
    axis_id: str
    prompt_instance_id: str
    prompt_id: str
    prompt_hash: str
    assigned_bias: float
    style_seed: int
    observed_axis_coordinate: float
    absolute_assigned_coordinate_error: float
    matched_off_axis_residual: float
    candidate_set_id: str
    ranking: RankingPermutation
    raw_model_output: str
    raw_model_output_hash: str
    prompt_embedding_hash: str


@dataclass(frozen=True, slots=True)
class PolicyMappingArtifacts:
    axis_path: Path
    locations_path: Path
    validation_path: Path
    outcomes_path: Path
    report_path: Path


def build_policy_latent_axis(
    embedder: PolicyPromptEmbedder,
    endpoint_prompts: Sequence[PromptRecord],
    *,
    axis_version: str = POLICY_LATENT_MAPPING_VERSION,
) -> PolicyLatentAxis:
    """Fit one direction from matched ``B=0``/``B=1`` prompts across styles."""

    if axis_version != POLICY_LATENT_MAPPING_VERSION:
        raise ValueError(f"unsupported policy latent mapping version: {axis_version!r}")
    grouped: dict[int, dict[float, PromptRecord]] = {}
    for record in endpoint_prompts:
        if not (math.isclose(record.assigned_bias, 0.0) or math.isclose(record.assigned_bias, 1.0)):
            raise ValueError("endpoint prompts must contain only assigned B=0 and B=1")
        endpoint = 0.0 if math.isclose(record.assigned_bias, 0.0) else 1.0
        by_bias = grouped.setdefault(record.style_seed, {})
        if endpoint in by_bias:
            raise ValueError(f"duplicate B={endpoint:g} endpoint for S={record.style_seed}")
        by_bias[endpoint] = record
    if len(grouped) < 2:
        raise ValueError("at least two matched surface styles are required")

    ordered: list[tuple[int, PromptRecord, PromptRecord]] = []
    for style_seed in sorted(grouped):
        pair = grouped[style_seed]
        if set(pair) != {0.0, 1.0}:
            raise ValueError(f"surface seed S={style_seed} lacks a matched B=0/B=1 pair")
        neutral, strong = pair[0.0], pair[1.0]
        if neutral.style_plan != strong.style_plan:
            raise ValueError(f"surface plan changes across B for S={style_seed}")
        ordered.append((style_seed, neutral, strong))

    texts = [record.prompt_template for _, pair0, pair1 in ordered for record in (pair0, pair1)]
    embeddings = _validated_embeddings(embedder, texts)
    neutral = embeddings[0::2]
    strong = embeddings[1::2]
    differences = strong - neutral
    mean_difference = np.mean(differences, axis=0)
    norm = float(np.linalg.norm(mean_difference))
    if norm <= 1e-12:
        raise ValueError("matched endpoints do not define a nonzero policy direction")
    direction = mean_difference / norm
    neutral_centroid = np.mean(neutral, axis=0)
    strong_centroid = np.mean(strong, axis=0)
    lower = float(neutral_centroid @ direction)
    upper = float(strong_centroid @ direction)
    if upper - lower <= 1e-12:
        raise ValueError("strong-preference centroid does not project above neutral")

    difference_norms = np.linalg.norm(differences, axis=1)
    if np.any(difference_norms <= 1e-12):
        raise ValueError("one matched endpoint pair has zero distance")
    pair_cosines = (differences @ direction) / difference_norms
    loo_positive: list[bool] = []
    for held_out in range(len(differences)):
        training = np.delete(differences, held_out, axis=0).mean(axis=0)
        training_norm = float(np.linalg.norm(training))
        if training_norm <= 1e-12:
            loo_positive.append(False)
        else:
            loo_positive.append(float(differences[held_out] @ (training / training_norm)) > 0.0)

    origins = tuple(
        PolicyStyleOrigin(
            style_seed=style_seed,
            midpoint=tuple(float(value) for value in (neutral[index] + strong[index]) / 2.0),
        )
        for index, (style_seed, _, _) in enumerate(ordered)
    )
    identity = {
        "axis_version": axis_version,
        "embedding_model": embedder.model_name,
        "endpoint_pairs": [
            {
                "style_seed": style_seed,
                "neutral_hash": neutral_record.prompt_hash,
                "strong_hash": strong_record.prompt_hash,
            }
            for style_seed, neutral_record, strong_record in ordered
        ],
        "direction": [round(float(value), 12) for value in direction],
    }
    axis_hash = _stable_hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return PolicyLatentAxis(
        axis_id=f"{axis_version}:{axis_hash[:20]}",
        axis_version=axis_version,
        embedding_model=embedder.model_name,
        dimension=int(direction.shape[0]),
        direction=tuple(float(value) for value in direction),
        neutral_anchor=lower,
        strong_preference_anchor=upper,
        neutral_centroid=tuple(float(value) for value in neutral_centroid),
        strong_preference_centroid=tuple(float(value) for value in strong_centroid),
        endpoint_style_seeds=tuple(style_seed for style_seed, _, _ in ordered),
        style_origins=origins,
        pair_direction_cosines=tuple(float(value) for value in pair_cosines),
        leave_one_style_out_positive_rate=sum(loo_positive) / len(loo_positive),
    )


def locate_policy_prompts(
    axis: PolicyLatentAxis,
    embedder: PolicyPromptEmbedder,
    prompts: Sequence[PromptRecord],
) -> tuple[PolicyPromptLocation, ...]:
    """Embed prompts and record location and matched-style off-axis drift."""

    if axis.embedding_model != embedder.model_name:
        raise ValueError("axis and prompt embedder models do not match")
    if not prompts:
        raise ValueError("at least one prompt is required")
    origins = {
        origin.style_seed: np.asarray(origin.midpoint, dtype=np.float64)
        for origin in axis.style_origins
    }
    unknown = sorted({record.style_seed for record in prompts} - set(origins))
    if unknown:
        raise ValueError("prompt styles are absent from the fitted axis: " + ", ".join(map(str, unknown)))
    embeddings = _validated_embeddings(embedder, [record.prompt_template for record in prompts])
    direction = np.asarray(axis.direction, dtype=np.float64)
    scale = axis.strong_preference_anchor - axis.neutral_anchor
    projections = (embeddings @ direction - axis.neutral_anchor) / scale
    locations: list[PolicyPromptLocation] = []
    for record, embedding, coordinate in zip(prompts, embeddings, projections):
        offset = embedding - origins[record.style_seed]
        residual = offset - float(offset @ direction) * direction
        locations.append(
            PolicyPromptLocation(
                prompt_assignment_id=_prompt_assignment_id(record),
                prompt_id=record.prompt_id,
                prompt_hash=record.prompt_hash,
                assigned_bias=record.assigned_bias,
                style_seed=record.style_seed,
                axis_id=axis.axis_id,
                embedding_model=axis.embedding_model,
                observed_axis_coordinate=float(coordinate),
                absolute_assigned_coordinate_error=abs(float(coordinate) - record.assigned_bias),
                matched_off_axis_residual=float(np.linalg.norm(residual) / scale),
                prompt_embedding=tuple(float(value) for value in embedding),
                embedding_hash=hashlib.sha256(np.asarray(embedding, dtype="<f8").tobytes()).hexdigest(),
            )
        )
    return tuple(locations)


def validate_policy_prompt_locations(
    locations: Sequence[PolicyPromptLocation],
    *,
    max_absolute_coordinate_error: float,
    max_matched_off_axis_residual: float,
    monotonic_tolerance: float = 0.0,
) -> PolicyLatentValidation:
    """Apply explicitly supplied thresholds to locations and within-S ordering."""

    for name, value in (
        ("max_absolute_coordinate_error", max_absolute_coordinate_error),
        ("max_matched_off_axis_residual", max_matched_off_axis_residual),
        ("monotonic_tolerance", monotonic_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if not locations:
        raise ValueError("at least one prompt location is required")

    rejected = tuple(
        location.prompt_id
        for location in locations
        if location.absolute_assigned_coordinate_error > max_absolute_coordinate_error
        or location.matched_off_axis_residual > max_matched_off_axis_residual
    )
    grouped: dict[int, list[PolicyPromptLocation]] = {}
    for location in locations:
        grouped.setdefault(location.style_seed, []).append(location)
    nonmonotonic: list[int] = []
    for style_seed, trajectory in grouped.items():
        ordered = sorted(trajectory, key=lambda item: item.assigned_bias)
        if len({item.assigned_bias for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate assigned B values for S={style_seed}")
        if any(
            right.observed_axis_coordinate + monotonic_tolerance < left.observed_axis_coordinate
            for left, right in zip(ordered, ordered[1:])
        ):
            nonmonotonic.append(style_seed)
    points_ok = not rejected
    trajectories_ok = not nonmonotonic
    return PolicyLatentValidation(
        max_absolute_coordinate_error=float(max_absolute_coordinate_error),
        max_matched_off_axis_residual=float(max_matched_off_axis_residual),
        monotonic_tolerance=float(monotonic_tolerance),
        prompt_count=len(locations),
        rejected_prompt_ids=rejected,
        nonmonotonic_style_seeds=tuple(sorted(nonmonotonic)),
        all_points_within_thresholds=points_ok,
        all_style_trajectories_monotonic=trajectories_ok,
        passed=points_ok and trajectories_ok,
    )


def render_policy_prompt(
    prompt: PromptRecord,
    *,
    keyword: str,
    candidates: Sequence[SearchCandidate],
    top_n: int,
) -> RenderedPolicyPrompt:
    """Bind a policy prompt to the fixed query, candidates, and output size."""

    bridge = SearchPurposePromptRecord(
        prompt_id=prompt.prompt_id,
        prompt_hash=prompt.prompt_hash,
        assigned_action_intensity=prompt.assigned_bias,
        style_seed=prompt.style_seed,
        top_n=top_n,
        style_plan=prompt.style_plan,
        purpose_level="first-party-policy",
        purpose_clause="",
        prompt_template=prompt.prompt_template,
        prompt_space_version=prompt.prompt_space_version,
        axis_specification_version=POLICY_LATENT_MAPPING_VERSION,
        generator_backend=prompt.generator_backend,
    )
    rendered = render_search_purpose_prompt(
        bridge, keyword=keyword, candidates=candidates, top_n=top_n
    )
    identity = {
        "prompt_id": prompt.prompt_id,
        "rendered_prompt_hash": rendered.rendered_prompt_hash,
        "candidate_set_id": rendered.candidate_set_id,
        "assigned_bias": prompt.assigned_bias,
        "style_seed": prompt.style_seed,
    }
    identity_hash = _stable_hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return RenderedPolicyPrompt(
        prompt_assignment_id=_prompt_assignment_id(prompt),
        prompt_instance_id=f"policy-prompt-instance:{identity_hash[:20]}",
        rendered_prompt_hash=rendered.rendered_prompt_hash,
        prompt_id=prompt.prompt_id,
        prompt_hash=prompt.prompt_hash,
        assigned_bias=prompt.assigned_bias,
        style_seed=prompt.style_seed,
        keyword=rendered.keyword,
        top_n=rendered.top_n,
        candidate_set_id=rendered.candidate_set_id,
        candidates=rendered.candidates,
        rendered_prompt=rendered.rendered_prompt,
        prompt_space_version=prompt.prompt_space_version,
    )


def map_policy_prompt_to_permutation(
    rendered_prompt: RenderedPolicyPrompt,
    location: PolicyPromptLocation,
    raw_model_output: str,
    *,
    reranker_run_id: str,
    reranker_model: str,
) -> PolicyPermutationOutcome:
    """Validate a response and join assigned B, latent diagnostics, and Y."""

    if rendered_prompt.prompt_id != location.prompt_id or rendered_prompt.prompt_hash != location.prompt_hash:
        raise ValueError("rendered prompt and latent location identities do not match")
    if rendered_prompt.prompt_assignment_id != location.prompt_assignment_id:
        raise ValueError("rendered prompt and latent location assignment IDs do not match")
    if rendered_prompt.style_seed != location.style_seed or not math.isclose(rendered_prompt.assigned_bias, location.assigned_bias):
        raise ValueError("rendered prompt and latent location assignments do not match")
    if not isinstance(reranker_run_id, str) or not reranker_run_id.strip():
        raise ValueError("reranker_run_id must be non-empty")
    if not isinstance(reranker_model, str) or not reranker_model.strip():
        raise ValueError("reranker_model must be non-empty")
    ranking = parse_ranking_permutation(raw_model_output, rendered_prompt)  # type: ignore[arg-type]
    raw_hash = _stable_hash(raw_model_output)
    identity = {
        "mapping_version": POLICY_LATENT_MAPPING_VERSION,
        "reranker_run_id": reranker_run_id,
        "reranker_model": reranker_model,
        "prompt_instance_id": rendered_prompt.prompt_instance_id,
        "axis_id": location.axis_id,
        "permutation_hash": ranking.permutation_hash,
        "raw_model_output_hash": raw_hash,
    }
    outcome_hash = _stable_hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return PolicyPermutationOutcome(
        prompt_assignment_id=location.prompt_assignment_id,
        outcome_id=f"policy-permutation:{outcome_hash[:20]}",
        mapping_version=POLICY_LATENT_MAPPING_VERSION,
        reranker_run_id=reranker_run_id,
        reranker_model=reranker_model,
        axis_id=location.axis_id,
        prompt_instance_id=rendered_prompt.prompt_instance_id,
        prompt_id=rendered_prompt.prompt_id,
        prompt_hash=rendered_prompt.prompt_hash,
        assigned_bias=location.assigned_bias,
        style_seed=location.style_seed,
        observed_axis_coordinate=location.observed_axis_coordinate,
        absolute_assigned_coordinate_error=location.absolute_assigned_coordinate_error,
        matched_off_axis_residual=location.matched_off_axis_residual,
        candidate_set_id=rendered_prompt.candidate_set_id,
        ranking=ranking,
        raw_model_output=raw_model_output,
        raw_model_output_hash=raw_hash,
        prompt_embedding_hash=location.embedding_hash,
    )


def write_policy_mapping_artifacts(
    output_directory: str | Path,
    *,
    axis: PolicyLatentAxis,
    locations: Sequence[PolicyPromptLocation],
    validation: PolicyLatentValidation,
    outcomes: Sequence[PolicyPermutationOutcome] = (),
    overwrite: bool = False,
) -> PolicyMappingArtifacts:
    """Atomically persist the axis, prompt locations, permutation map, and report."""

    output = Path(output_directory)
    artifacts = PolicyMappingArtifacts(
        axis_path=output / "policy_latent_axis.json",
        locations_path=output / "policy_prompt_locations.jsonl",
        validation_path=output / "policy_latent_validation.json",
        outcomes_path=output / "policy_permutation_outcomes.jsonl",
        report_path=output / "policy_mapping_report.md",
    )
    paths = tuple(asdict(artifacts).values())
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite policy mapping artifacts: " + ", ".join(map(str, existing)))
    if any(location.axis_id != axis.axis_id for location in locations):
        raise ValueError("all prompt locations must reference the supplied axis")
    if validation.prompt_count != len(locations):
        raise ValueError("validation prompt count does not match supplied locations")
    known_assignments = {location.prompt_assignment_id for location in locations}
    if len(known_assignments) != len(locations):
        raise ValueError("prompt assignment IDs must be unique")
    if any(
        outcome.prompt_assignment_id not in known_assignments
        or outcome.axis_id != axis.axis_id
        for outcome in outcomes
    ):
        raise ValueError("every outcome must reference a supplied prompt location and axis")

    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(artifacts.axis_path, json.dumps(asdict(axis), indent=2, sort_keys=True) + "\n")
    _atomic_text(artifacts.locations_path, _jsonl(asdict(location) for location in locations))
    _atomic_text(
        artifacts.validation_path,
        json.dumps(asdict(validation), indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(artifacts.outcomes_path, _jsonl(asdict(outcome) for outcome in outcomes))
    report = "\n".join(
        (
            "# Prompt latent-location to reranking-permutation map",
            "",
            f"- Assigned prompt records: `{len(locations)}`",
            f"- Latent validation passed: `{validation.passed}`",
            f"- Rejected prompt locations: `{len(validation.rejected_prompt_ids)}`",
            f"- Nonmonotonic surface trajectories: `{len(validation.nonmonotonic_style_seeds)}`",
            f"- Strict reranking permutations attached: `{len(outcomes)}`",
            "",
            "`assigned_bias` is the randomized treatment. `observed_axis_coordinate` and",
            "off-axis residuals validate the realized prompt but never redefine assignment.",
            "Final prompts are generated as text before embedding; latent vectors are not decoded.",
            "",
        )
    )
    _atomic_text(artifacts.report_path, report)
    return artifacts


class FakePolicyPromptEmbedder:
    """Deterministic first-party-policy embedder for CPU tests only."""

    model_name = "fake-first-party-policy-embedder-v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "a strong preference" in lowered:
                strength = 1.0
            elif "a clear preference" in lowered:
                strength = 0.75
            elif "a moderate preference" in lowered:
                strength = 0.5
            elif "a slight preference" in lowered:
                strength = 0.25
            else:
                strength = 0.0
            rows.append(
                [
                    strength,
                    float(lowered.startswith("please")),
                    float("your task is" in lowered),
                    float("query:\n{query}" in lowered),
                    float("response rule:" in lowered),
                ]
            )
        return np.asarray(rows, dtype=np.float64)


def _validated_embeddings(embedder: PolicyPromptEmbedder, texts: Sequence[str]) -> np.ndarray:
    embeddings = np.asarray(embedder.embed(texts), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] <= 0:
        raise ValueError(f"embedder returned invalid shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("prompt embeddings contain non-finite values")
    return embeddings


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_assignment_id(record: PromptRecord) -> str:
    identity = {
        "prompt_id": record.prompt_id,
        "assigned_bias": f"{record.assigned_bias:.12g}",
        "style_seed": record.style_seed,
        "prompt_space_version": record.prompt_space_version,
    }
    digest = _stable_hash(json.dumps(identity, sort_keys=True, separators=(",", ":")))
    return f"policy-assignment:{digest[:20]}"


def _jsonl(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
