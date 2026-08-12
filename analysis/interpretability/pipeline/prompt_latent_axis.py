"""Query-conditioned prompt generation along a learned semantic direction.

The scientific path is generate -> embed -> project -> select:

1. Build an informational-to-transactional direction from paired endpoint
   prompt examples.
2. Ask a provider for multiple prompt templates conditioned on one query and a
   target coordinate.
3. Embed every candidate and project it onto the frozen direction.
4. Select the candidate closest to the target coordinate.

An embedding vector is not directly decoded into text.  Candidate generation
and projection make that limitation explicit.  The assigned coordinate remains
the experimental variable; the observed embedding projection is a diagnostic.
This module performs no inference at import time and has deterministic fake
providers for CPU-only tests.
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

from .prompt_continuum import TemplatePromptGenerator, _normalize_template
from .search_purpose_continuum import (
    AXIS_SPECIFICATION_VERSION,
    RenderedSearchPurposePrompt,
    SearchCandidate,
    SearchPurposePromptRecord,
    render_search_purpose_prompt,
)

LATENT_AXIS_VERSION = "informational-transactional-latent-axis-v1"
LATENT_META_PROMPT_VERSION = "search-purpose-latent-meta-prompt-v1"
_META_PROMPT_PATH = (
    Path(__file__).with_name("specs") / "search_purpose_latent_meta_prompt_v1.txt"
)

_OFF_AXIS_PATTERNS = (
    r"\bfirst-party\b",
    r"\bcommercial(?:ity)?\b",
    r"\bauthorit(?:y|ative)\b",
    r"\bfresh(?:ness)?\b",
    r"\brecen(?:t|cy)\b",
    r"\bcitations?\b",
    r"\bpopular(?:ity)?\b",
    r"\bbrand fame\b",
    r"\bwriting quality\b",
)

__all__ = [
    "FakeLatentPromptProvider",
    "FakePromptEmbedder",
    "LatentPromptGenerationRequest",
    "LatentPromptRecord",
    "PromptEmbeddingProvider",
    "PromptLatentAxis",
    "PromptTextProvider",
    "RepositoryLocalLatentPromptProvider",
    "SentenceTransformerPromptEmbedder",
    "build_latent_prompt_request",
    "build_prompt_latent_axis",
    "generate_prompt_at_coordinate",
    "project_prompt_embeddings",
    "render_selected_latent_prompt",
]


class PromptEmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one finite embedding row per input text."""


class PromptTextProvider(Protocol):
    backend_name: str

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        """Return strict JSON containing a list of prompt templates."""


@dataclass(frozen=True, slots=True)
class PromptLatentAxis:
    axis_id: str
    axis_version: str
    embedding_model: str
    dimension: int
    direction: tuple[float, ...]
    informational_anchor: float
    transactional_anchor: float
    informational_centroid: tuple[float, ...]
    transactional_centroid: tuple[float, ...]
    endpoint_pair_count: int


@dataclass(frozen=True, slots=True)
class LatentPromptGenerationRequest:
    query: str
    target_coordinate: float
    style_seed: int
    generation_seed: int
    number_candidates: int
    generator_model: str
    axis_version: str = LATENT_AXIS_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        target = self.target_coordinate
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("target_coordinate must be numeric")
        if not math.isfinite(float(target)) or not 0.0 <= float(target) <= 1.0:
            raise ValueError("target_coordinate must be in [0, 1]")
        for name, value in (
            ("style_seed", self.style_seed),
            ("generation_seed", self.generation_seed),
            ("number_candidates", self.number_candidates),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.number_candidates <= 0:
            raise ValueError("number_candidates must be greater than zero")
        if not isinstance(self.generator_model, str) or not self.generator_model.strip():
            raise ValueError("generator_model must be non-empty")
        if self.axis_version != LATENT_AXIS_VERSION:
            raise ValueError(f"unsupported latent axis version: {self.axis_version!r}")


@dataclass(frozen=True, slots=True)
class PromptCandidateProjection:
    prompt_template: str
    prompt_hash: str
    observed_coordinate: float
    absolute_target_error: float


@dataclass(frozen=True, slots=True)
class LatentPromptRecord:
    prompt_id: str
    prompt_hash: str
    query: str
    assigned_target_coordinate: float
    observed_axis_coordinate: float
    absolute_target_error: float
    style_seed: int
    generation_seed: int
    prompt_template: str
    axis_id: str
    axis_version: str
    embedding_model: str
    generator_backend: str
    generator_model: str
    generation_parameters: dict[str, object]
    candidate_projections: tuple[PromptCandidateProjection, ...]
    raw_model_output: str
    validation_status: str


def build_prompt_latent_axis(
    embedder: PromptEmbeddingProvider,
    *,
    informational_endpoint_prompts: Sequence[str],
    transactional_endpoint_prompts: Sequence[str],
    axis_version: str = LATENT_AXIS_VERSION,
) -> PromptLatentAxis:
    """Fit a direction from paired endpoint prompts.

    Paired differences reduce topic/query offsets: each informational endpoint
    must describe the same underlying query as the transactional endpoint at
    the corresponding position.
    """

    if axis_version != LATENT_AXIS_VERSION:
        raise ValueError(f"unsupported latent axis version: {axis_version!r}")
    if not informational_endpoint_prompts:
        raise ValueError("at least one endpoint pair is required")
    if len(informational_endpoint_prompts) != len(transactional_endpoint_prompts):
        raise ValueError("informational and transactional endpoints must be paired")
    informational = _validated_embeddings(
        embedder, informational_endpoint_prompts, label="informational endpoints"
    )
    transactional = _validated_embeddings(
        embedder, transactional_endpoint_prompts, label="transactional endpoints"
    )
    if informational.shape != transactional.shape:
        raise ValueError("endpoint embedding shapes do not match")
    direction = np.mean(transactional - informational, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("endpoint prompts do not define a nonzero latent direction")
    direction = direction / norm
    informational_centroid = np.mean(informational, axis=0)
    transactional_centroid = np.mean(transactional, axis=0)
    lower = float(informational_centroid @ direction)
    upper = float(transactional_centroid @ direction)
    if upper - lower <= 1e-12:
        raise ValueError("transactional endpoint does not project above informational endpoint")
    identity = {
        "axis_version": axis_version,
        "embedding_model": embedder.model_name,
        "informational_hashes": [_stable_hash(text) for text in informational_endpoint_prompts],
        "transactional_hashes": [_stable_hash(text) for text in transactional_endpoint_prompts],
        "direction": [round(float(value), 12) for value in direction],
    }
    axis_hash = _stable_hash(
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
    )
    return PromptLatentAxis(
        axis_id=f"{axis_version}:{axis_hash[:20]}",
        axis_version=axis_version,
        embedding_model=embedder.model_name,
        dimension=int(direction.shape[0]),
        direction=tuple(float(value) for value in direction),
        informational_anchor=lower,
        transactional_anchor=upper,
        informational_centroid=tuple(float(value) for value in informational_centroid),
        transactional_centroid=tuple(float(value) for value in transactional_centroid),
        endpoint_pair_count=len(informational_endpoint_prompts),
    )


def project_prompt_embeddings(axis: PromptLatentAxis, embeddings: np.ndarray) -> np.ndarray:
    """Map embedding rows to calibrated coordinates (endpoints average to 0/1)."""

    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != axis.dimension:
        raise ValueError(
            f"expected embedding shape (n, {axis.dimension}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    direction = np.asarray(axis.direction, dtype=np.float64)
    projections = array @ direction
    scale = axis.transactional_anchor - axis.informational_anchor
    return (projections - axis.informational_anchor) / scale


def build_latent_prompt_request(request: LatentPromptGenerationRequest) -> str:
    """Build the exact query-conditioned meta-prompt used for candidate generation."""

    meta_prompt = _META_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        meta_prompt.replace("{{QUERY}}", request.query.strip())
        .replace("{{TARGET_COORDINATE}}", f"{request.target_coordinate:.6f}")
        .replace("{{STYLE_SEED}}", str(request.style_seed))
        .replace("{{NUMBER_CANDIDATES}}", str(request.number_candidates))
        .replace("{{AXIS_VERSION}}", request.axis_version)
    )


def generate_prompt_at_coordinate(
    request: LatentPromptGenerationRequest,
    *,
    axis: PromptLatentAxis,
    provider: PromptTextProvider,
    embedder: PromptEmbeddingProvider,
    generation_parameters: Mapping[str, object] | None = None,
) -> LatentPromptRecord:
    """Generate candidates and select the prompt nearest a target latent coordinate."""

    if axis.axis_version != request.axis_version:
        raise ValueError("request and latent-axis versions do not match")
    if axis.embedding_model != embedder.model_name:
        raise ValueError("latent axis and embedding provider models do not match")
    parameters = dict(
        generation_parameters
        or {"max_new_tokens": 900, "temperature": 0.9, "top_p": 1.0}
    )
    parameters["generation_seed"] = request.generation_seed
    request_text = build_latent_prompt_request(request)
    raw_output = provider.generate(request_text, parameters)
    candidates = _parse_prompt_candidates(raw_output, request)
    embeddings = _validated_embeddings(embedder, candidates, label="prompt candidates")
    coordinates = project_prompt_embeddings(axis, embeddings)
    diagnostics = tuple(
        PromptCandidateProjection(
            prompt_template=template,
            prompt_hash=_stable_hash(template),
            observed_coordinate=float(coordinate),
            absolute_target_error=abs(float(coordinate) - request.target_coordinate),
        )
        for template, coordinate in zip(candidates, coordinates)
    )
    selected = min(
        diagnostics,
        key=lambda item: (item.absolute_target_error, item.prompt_hash),
    )
    identity = {
        "axis_id": axis.axis_id,
        "query": request.query.strip(),
        "target": f"{request.target_coordinate:.6f}",
        "style_seed": request.style_seed,
        "generation_seed": request.generation_seed,
        "prompt_hash": selected.prompt_hash,
    }
    identity_hash = _stable_hash(
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
    )
    return LatentPromptRecord(
        prompt_id=f"latent-prompt:{identity_hash[:20]}",
        prompt_hash=selected.prompt_hash,
        query=request.query.strip(),
        assigned_target_coordinate=float(request.target_coordinate),
        observed_axis_coordinate=selected.observed_coordinate,
        absolute_target_error=selected.absolute_target_error,
        style_seed=request.style_seed,
        generation_seed=request.generation_seed,
        prompt_template=selected.prompt_template,
        axis_id=axis.axis_id,
        axis_version=axis.axis_version,
        embedding_model=axis.embedding_model,
        generator_backend=provider.backend_name,
        generator_model=request.generator_model,
        generation_parameters=parameters,
        candidate_projections=diagnostics,
        raw_model_output=raw_output,
        validation_status="latent-selected-unvalidated",
    )


def render_selected_latent_prompt(
    record: LatentPromptRecord,
    *,
    candidates: Sequence[SearchCandidate],
    top_n: int,
) -> RenderedSearchPurposePrompt:
    """Render a selected query-conditioned template with a frozen candidate set."""

    bridge = SearchPurposePromptRecord(
        prompt_id=record.prompt_id,
        prompt_hash=record.prompt_hash,
        assigned_action_intensity=record.assigned_target_coordinate,
        style_seed=record.style_seed,
        top_n=top_n,
        style_plan=TemplatePromptGenerator._build_style_plan(record.style_seed),
        purpose_level="latent-selected",
        purpose_clause="",
        prompt_template=record.prompt_template,
        prompt_space_version=record.axis_version,
        axis_specification_version=AXIS_SPECIFICATION_VERSION,
        generator_backend=record.generator_backend,
    )
    return render_search_purpose_prompt(
        bridge,
        keyword=record.query,
        candidates=candidates,
        top_n=top_n,
    )


class SentenceTransformerPromptEmbedder:
    """Lazy adapter around the repository's sentence-transformer convention."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", *, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        kwargs = {"device": device} if device else {}
        self._model = SentenceTransformer(model_name, **kwargs)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts), show_progress_bar=False, convert_to_numpy=True
            ),
            dtype=np.float64,
        )


class RepositoryLocalLatentPromptProvider:
    """Lazy adapter around the repository's configurable local HF ranker."""

    backend_name = "repository-local-ranker"

    def __init__(self, ranker) -> None:
        self._ranker = ranker

    @classmethod
    def from_model(
        cls, model: str, *, precision: str = "full"
    ) -> "RepositoryLocalLatentPromptProvider":
        from ..utils import make_ranker

        return cls(make_ranker("local", model, precision=precision))

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        import torch

        seed = int(generation_config["generation_seed"])
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self._ranker.rank(
                request_text,
                max_tokens=int(generation_config.get("max_new_tokens", 900)),
                temperature=float(generation_config.get("temperature", 0.9)),
            )


class FakePromptEmbedder:
    """Small deterministic semantic embedder for tests only."""

    model_name = "fake-search-purpose-embedder-v1"
    _INFO = ("learn", "understand", "explain", "information", "overview")
    _TRANSACTION = (
        "act",
        "complete",
        "download",
        "install",
        "register",
        "select",
        "start",
        "deploy",
        "now",
    )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            info = sum(lowered.count(term) for term in self._INFO)
            transaction = sum(lowered.count(term) for term in self._TRANSACTION)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append(
                [
                    float(info),
                    float(transaction),
                    float("compare" in lowered or "evaluate" in lowered),
                    len(text) / 500.0,
                    digest[0] / 2550.0,
                ]
            )
        return np.asarray(rows, dtype=np.float64)


class FakeLatentPromptProvider:
    """Deterministic candidate generator for unit tests; not scientific output."""

    backend_name = "fake-latent-prompt-provider"

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        match = re.search(r"Target coordinate:\s*([0-9.]+)", request_text)
        count_match = re.search(r"Number of candidates:\s*(\d+)", request_text)
        if match is None or count_match is None:
            raise ValueError("fake provider could not parse request")
        target = float(match.group(1))
        count = int(count_match.group(1))
        purposes = (
            "Help the user learn and understand the query with an explanatory overview.",
            "Help the user compare and evaluate approaches relevant to the query.",
            "Help the user select and complete the action implied by the query now.",
        )
        center = min(2, max(0, int(round(target * 2))))
        templates = []
        for offset in range(count):
            purpose = purposes[(center + offset) % len(purposes)]
            templates.append(
                "Rerank the candidates by relevance to the query and this search purpose: "
                + purpose
                + "\n\nQuery: {QUERY}\n\nCandidates:\n{CANDIDATES}\n\n"
                "Return exactly {TOP_N} candidate identifiers only, with no explanation."
            )
        return json.dumps({"prompt_templates": templates}, ensure_ascii=False)


def _parse_prompt_candidates(
    raw_output: str, request: LatentPromptGenerationRequest
) -> tuple[str, ...]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("prompt provider returned empty output")
    payload = _load_provider_json(raw_output)
    if not isinstance(payload, dict) or set(payload) != {"prompt_templates"}:
        raise ValueError("prompt provider JSON must contain only prompt_templates")
    templates = payload["prompt_templates"]
    if not isinstance(templates, list) or len(templates) < request.number_candidates:
        raise ValueError(
            f"provider must return at least {request.number_candidates} prompt templates"
        )
    normalized: list[str] = []
    for index, value in enumerate(templates):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt candidate {index} is empty")
        template = _normalize_template(value)
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            if placeholder not in template:
                raise ValueError(f"prompt candidate {index} lacks {placeholder}")
        lowered = template.lower()
        if not _has_identifier_only_contract(lowered):
            raise ValueError(
                f"prompt candidate {index} lacks identifier-only contract; "
                f"candidate preview={template[:240]!r}"
            )
        if not re.search(r"no explanation|do not (?:provide|include) an explanation", lowered):
            raise ValueError(f"prompt candidate {index} permits explanations")
        for pattern in _OFF_AXIS_PATTERNS:
            if re.search(pattern, lowered):
                raise ValueError(f"prompt candidate {index} introduces off-axis criterion")
        canonical_target = f"{request.target_coordinate:.6f}"
        compact_target = canonical_target.rstrip("0").rstrip(".")
        numeric_forms = {canonical_target}
        if 0.0 < request.target_coordinate < 1.0:
            numeric_forms.add(compact_target)
            numeric_forms.add(compact_target.lstrip("0"))
        if any(
            form and re.search(rf"(?<![\d.]){re.escape(form)}(?![\d.])", template)
            for form in numeric_forms
        ):
            raise ValueError(f"prompt candidate {index} exposes the target coordinate")
        normalized.append(template)
    unique = tuple(dict.fromkeys(normalized))
    if len(unique) < request.number_candidates:
        raise ValueError("provider returned too few unique prompt templates")
    return unique


def _has_identifier_only_contract(lowered_template: str) -> bool:
    """Recognize equivalent identifier-only output-contract wording."""
    identifier = r"(?:candidate\s+)?(?:identifiers?|ids?)"
    return bool(
        re.search(rf"\b{identifier}\s+only\b", lowered_template)
        or re.search(rf"\bonly\s+(?:the\s+)?{identifier}\b", lowered_template)
        or re.search(
            rf"\b{identifier}\b.{{0,80}}\b(?:nothing else|no (?:other|additional|extra) "
            r"(?:text|content))\b",
            lowered_template,
        )
    )


def _load_provider_json(raw_output: str) -> object:
    """Load one valid JSON object, allowing harmless model wrappers.

    Local instruction models sometimes wrap an otherwise valid JSON response
    in a Markdown fence or a short sentence despite an explicit no-commentary
    instruction. This accepts the embedded JSON object without relaxing any of
    the prompt-candidate semantic and structural validation that follows.
    """
    stripped = raw_output.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", stripped):
            try:
                payload, _ = decoder.raw_decode(stripped[match.start() :])
            except json.JSONDecodeError:
                continue
            return payload
        preview = re.sub(r"\s+", " ", stripped)[:240]
        raise ValueError(
            f"prompt provider returned invalid JSON; output preview={preview!r}"
        ) from direct_error


def _validated_embeddings(
    embedder: PromptEmbeddingProvider,
    texts: Sequence[str],
    *,
    label: str,
) -> np.ndarray:
    array = np.asarray(embedder.embed(texts), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] <= 0:
        raise ValueError(f"{label} produced invalid embedding shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contain non-finite embeddings")
    return array


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
