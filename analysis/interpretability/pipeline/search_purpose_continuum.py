"""Deterministic foundation for an informational-to-action search-purpose axis.

The assigned variable is ``I`` in ``[0, 1]``.  It changes only the user's
search purpose, from learning about a topic to completing the action implied by
the query.  Surface realization is controlled independently by ``style_seed``.

This module is an engineering scaffold for manifests, smoke tests, and strict
ranking-output validation.  Its finite phrase schedule is not a scientifically
validated continuous semantic generator, and it performs no model inference.
The legacy neutral/biased and first-party prompt pipelines remain separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

from .prompt_continuum import StylePlan, TemplatePromptGenerator, _normalize_template

__all__ = [
    "CandidateBinding",
    "PermutationValidationError",
    "RankingPermutation",
    "RenderedSearchPurposePrompt",
    "SearchCandidate",
    "SearchPurposeGenerationRequest",
    "SearchPurposePilotArtifacts",
    "SearchPurposePromptRecord",
    "SearchPurposeTemplateGenerator",
    "load_search_purpose_specification",
    "parse_ranking_permutation",
    "render_search_purpose_prompt",
    "write_search_purpose_pilot",
]


AXIS_SPECIFICATION_VERSION = "search-purpose-axis-v1"
DEFAULT_PROMPT_SPACE_VERSION = "search-purpose-template-v1"
DEFAULT_INTENT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PILOT_MANIFEST_VERSION = "search-purpose-pilot-manifest-v1"
_SPECIFICATION_PATH = Path(__file__).with_name("specs") / "search_purpose_axis_v1.json"


@dataclass(frozen=True, slots=True)
class SearchPurposeGenerationRequest:
    """Inputs identifying one deterministic search-purpose template."""

    assigned_action_intensity: float
    style_seed: int
    top_n: int
    prompt_space_version: str = DEFAULT_PROMPT_SPACE_VERSION

    def __post_init__(self) -> None:
        value = self.assigned_action_intensity
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("assigned_action_intensity must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("assigned_action_intensity must be in [0, 1]")
        if isinstance(self.style_seed, bool) or not isinstance(self.style_seed, int):
            raise TypeError("style_seed must be an integer")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int):
            raise TypeError("top_n must be an integer")
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if not isinstance(self.prompt_space_version, str) or not self.prompt_space_version.strip():
            raise ValueError("prompt_space_version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SearchPurposePromptRecord:
    """One prompt template on the assigned search-purpose axis."""

    prompt_id: str
    prompt_hash: str
    assigned_action_intensity: float
    style_seed: int
    top_n: int
    style_plan: StylePlan
    purpose_level: str
    purpose_clause: str
    prompt_template: str
    prompt_space_version: str
    axis_specification_version: str
    generator_backend: str


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Frozen evidence for one page in a keyword-specific candidate pool."""

    source_position: int
    title: str
    url: str
    snippet: str

    def __post_init__(self) -> None:
        if isinstance(self.source_position, bool) or not isinstance(self.source_position, int):
            raise TypeError("source_position must be an integer")
        if self.source_position <= 0:
            raise ValueError("source_position must be greater than zero")
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("candidate url must be a non-empty string")
        for name, value in (("title", self.title), ("snippet", self.snippet)):
            if not isinstance(value, str):
                raise TypeError(f"candidate {name} must be a string")


@dataclass(frozen=True, slots=True)
class CandidateBinding:
    candidate_id: str
    source_position: int
    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class RenderedSearchPurposePrompt:
    """A template rendered with one keyword and one frozen candidate set."""

    prompt_instance_id: str
    rendered_prompt_hash: str
    prompt_id: str
    prompt_hash: str
    assigned_action_intensity: float
    style_seed: int
    keyword: str
    top_n: int
    candidate_set_id: str
    candidates: tuple[CandidateBinding, ...]
    rendered_prompt: str
    prompt_space_version: str


@dataclass(frozen=True, slots=True)
class RankingPermutation:
    """A validated ordered top-k selection from one frozen candidate set."""

    prompt_instance_id: str
    candidate_set_id: str
    candidate_ids: tuple[str, ...]
    source_position_vector: tuple[int, ...]
    permutation_hash: str
    is_full_permutation: bool


@dataclass(frozen=True, slots=True)
class SearchPurposePilotArtifacts:
    manifest_path: Path
    report_path: Path
    prompt_count: int


class PermutationValidationError(ValueError):
    """Raised when a model response violates the candidate-ID contract."""


class SearchPurposeTemplateGenerator:
    """Generate deterministic templates using a finite intent phrase schedule."""

    backend_name = "search-purpose-template-scaffold"

    def generate(self, request: SearchPurposeGenerationRequest) -> SearchPurposePromptRecord:
        load_search_purpose_specification()
        style = TemplatePromptGenerator._build_style_plan(request.style_seed)
        purpose_level, purpose_clause = self._purpose_clause(
            float(request.assigned_action_intensity)
        )
        instruction = self._instruction_block(style)
        output = TemplatePromptGenerator._output_clause(style.output_contract)
        inputs = "Query or topic:\n{QUERY}\n\nCandidates:\n{CANDIDATES}"
        if style.clause_order == "instructions_first":
            blocks = (instruction, purpose_clause, output, inputs)
        else:
            blocks = (inputs, instruction, purpose_clause, output)
        template = _normalize_template("\n\n".join(blocks))
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            if placeholder not in template:
                raise RuntimeError(f"generated template lost {placeholder}")
        prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return SearchPurposePromptRecord(
            prompt_id=(
                f"{request.prompt_space_version}:top{request.top_n}:{prompt_hash[:16]}"
            ),
            prompt_hash=prompt_hash,
            assigned_action_intensity=float(request.assigned_action_intensity),
            style_seed=request.style_seed,
            top_n=request.top_n,
            style_plan=style,
            purpose_level=purpose_level,
            purpose_clause=purpose_clause,
            prompt_template=template,
            prompt_space_version=request.prompt_space_version,
            axis_specification_version=AXIS_SPECIFICATION_VERSION,
            generator_backend=self.backend_name,
        )

    @staticmethod
    def _instruction_block(style: StylePlan) -> str:
        verb = style.ranking_verb
        if style.syntax == "imperative":
            task = (
                f"{verb} the supplied candidates by relevance to the supplied query "
                "and the user's assigned search purpose."
            )
        elif style.syntax == "request":
            task = (
                f"Please {verb.lower()} the supplied candidates by relevance to the "
                "supplied query and the user's assigned search purpose."
            )
        else:
            task = (
                f"Your task is to {verb.lower()} the supplied candidates by relevance "
                "to the supplied query and the user's assigned search purpose."
            )
        if style.tone == "formal":
            task = f"Follow this ranking instruction. {task}"
        elif style.tone == "courteous":
            task = f"Please follow this ranking instruction carefully. {task}"
        if style.verbosity == "standard":
            task += " Consider every supplied candidate and preserve the candidate set."
        return task

    @staticmethod
    def _purpose_clause(assigned_action_intensity: float) -> tuple[str, str]:
        if assigned_action_intensity == 0.0:
            return (
                "understand",
                "The user's purpose is to learn and understand the topic, without "
                "selecting or carrying out a solution.",
            )
        if assigned_action_intensity <= 0.25:
            return (
                "explore",
                "The user's purpose is to explore possible approaches while primarily "
                "building an understanding of the topic.",
            )
        if assigned_action_intensity <= 0.50:
            return (
                "evaluate",
                "The user's purpose is to compare and evaluate relevant approaches "
                "before deciding what to do.",
            )
        if assigned_action_intensity <= 0.75:
            return (
                "prepare",
                "The user's purpose is to make a decision and prepare to carry out the "
                "action implied by the query.",
            )
        return (
            "act",
            "The user's purpose is to complete the concrete action implied by the query now.",
        )


def render_search_purpose_prompt(
    prompt: SearchPurposePromptRecord,
    *,
    keyword: str,
    candidates: Sequence[SearchCandidate],
    top_n: int,
) -> RenderedSearchPurposePrompt:
    """Insert a real keyword and frozen evidence without changing the policy axis."""

    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword must be a non-empty string")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if top_n > len(candidates):
        raise ValueError("top_n cannot exceed the number of candidates")
    if top_n != prompt.top_n:
        raise ValueError(
            f"top_n={top_n} does not match the template contract top_n={prompt.top_n}"
        )
    if "{TOP_N}" not in prompt.prompt_template:
        raise ValueError("prompt template lacks {TOP_N}")

    urls = [candidate.url for candidate in candidates]
    if len(set(urls)) != len(urls):
        raise ValueError("candidate URLs must be unique within a frozen candidate set")
    bindings = tuple(
        CandidateBinding(
            candidate_id=f"C{index:03d}",
            source_position=candidate.source_position,
            title=_single_line(candidate.title),
            url=candidate.url.strip(),
            snippet=_single_line(candidate.snippet),
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    candidate_rows = [
        {
            "candidate_id": binding.candidate_id,
            "source_position": binding.source_position,
            "title": binding.title,
            "url": binding.url,
            "snippet": binding.snippet,
        }
        for binding in bindings
    ]
    candidate_canonical = json.dumps(
        candidate_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    candidate_set_hash = hashlib.sha256(candidate_canonical.encode("utf-8")).hexdigest()
    candidate_set_id = f"candidate-set:{candidate_set_hash[:20]}"
    candidate_text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in candidate_rows
    )
    rendered = prompt.prompt_template
    rendered = rendered.replace("{QUERY}", keyword.strip())
    rendered = rendered.replace("{CANDIDATES}", candidate_text)
    rendered = rendered.replace("{TOP_N}", str(top_n))
    for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
        if placeholder in rendered:
            raise ValueError(f"rendered prompt retained placeholder {placeholder}")
    rendered = _normalize_template(rendered)
    rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    identity = {
        "prompt_id": prompt.prompt_id,
        "keyword": keyword.strip(),
        "candidate_set_id": candidate_set_id,
        "top_n": top_n,
        "rendered_prompt_hash": rendered_hash,
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RenderedSearchPurposePrompt(
        prompt_instance_id=f"search-purpose-instance:{identity_hash[:20]}",
        rendered_prompt_hash=rendered_hash,
        prompt_id=prompt.prompt_id,
        prompt_hash=prompt.prompt_hash,
        assigned_action_intensity=prompt.assigned_action_intensity,
        style_seed=prompt.style_seed,
        keyword=keyword.strip(),
        top_n=top_n,
        candidate_set_id=candidate_set_id,
        candidates=bindings,
        rendered_prompt=rendered,
        prompt_space_version=prompt.prompt_space_version,
    )


def parse_ranking_permutation(
    raw_model_output: str,
    prompt: RenderedSearchPurposePrompt,
) -> RankingPermutation:
    """Validate an identifiers-only model response with no silent fallback."""

    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        raise PermutationValidationError("model output is empty")
    tokens = tuple(token for token in re.split(r"[\s,]+", raw_model_output.strip()) if token)
    if any(re.fullmatch(r"C\d{3}", token) is None for token in tokens):
        raise PermutationValidationError(
            "model output must contain candidate identifiers only"
        )
    if len(tokens) != prompt.top_n:
        raise PermutationValidationError(
            f"expected exactly {prompt.top_n} identifiers, received {len(tokens)}"
        )
    if len(set(tokens)) != len(tokens):
        raise PermutationValidationError("model output contains duplicate identifiers")
    bindings = {binding.candidate_id: binding for binding in prompt.candidates}
    unknown = sorted(set(tokens) - set(bindings))
    if unknown:
        raise PermutationValidationError(
            "model output contains unknown identifiers: " + ", ".join(unknown)
        )
    canonical = "\n".join(tokens)
    permutation_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return RankingPermutation(
        prompt_instance_id=prompt.prompt_instance_id,
        candidate_set_id=prompt.candidate_set_id,
        candidate_ids=tokens,
        source_position_vector=tuple(bindings[token].source_position for token in tokens),
        permutation_hash=permutation_hash,
        is_full_permutation=len(tokens) == len(prompt.candidates),
    )


def load_search_purpose_specification(
    version: str = AXIS_SPECIFICATION_VERSION,
) -> dict[str, object]:
    """Load and version-check the human-readable axis definition."""

    if version != AXIS_SPECIFICATION_VERSION:
        raise ValueError(f"unsupported search-purpose specification: {version!r}")
    specification = json.loads(_SPECIFICATION_PATH.read_text(encoding="utf-8"))
    if specification.get("specification_version") != version:
        raise ValueError("search-purpose specification file/version mismatch")
    return specification


def write_search_purpose_pilot(
    output_directory: str | Path,
    *,
    keyword_candidates: Mapping[str, Sequence[SearchCandidate]],
    intent_grid: Sequence[float] = DEFAULT_INTENT_GRID,
    style_seeds: Sequence[int] = (0, 1),
    top_n: int = 10,
    prompt_space_version: str = DEFAULT_PROMPT_SPACE_VERSION,
    overwrite: bool = False,
) -> SearchPurposePilotArtifacts:
    """Write a prompt-only JSONL manifest and audit report atomically."""

    output_dir = Path(output_directory)
    manifest_path = output_dir / "search_purpose_prompt_instances.jsonl"
    report_path = output_dir / "search_purpose_pilot_report.md"
    existing = [path for path in (manifest_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite search-purpose artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    if not keyword_candidates:
        raise ValueError("keyword_candidates must not be empty")
    if not intent_grid:
        raise ValueError("intent_grid must not be empty")
    if not style_seeds:
        raise ValueError("style_seeds must not be empty")
    normalized_grid = tuple(float(value) for value in intent_grid)
    if tuple(sorted(set(normalized_grid))) != normalized_grid:
        raise ValueError("intent_grid must be strictly increasing and unique")

    generator = SearchPurposeTemplateGenerator()
    generated_at = _utc_now()
    rows: list[dict[str, object]] = []
    candidate_set_by_keyword: dict[str, set[str]] = {}
    for keyword, candidates in keyword_candidates.items():
        candidate_set_by_keyword[keyword] = set()
        for style_seed in style_seeds:
            for intensity in normalized_grid:
                template = generator.generate(
                    SearchPurposeGenerationRequest(
                        assigned_action_intensity=intensity,
                        style_seed=style_seed,
                        top_n=top_n,
                        prompt_space_version=prompt_space_version,
                    )
                )
                rendered = render_search_purpose_prompt(
                    template, keyword=keyword, candidates=candidates, top_n=top_n
                )
                candidate_set_by_keyword[keyword].add(rendered.candidate_set_id)
                rows.append(_manifest_row(template, rendered, generated_at))
    if any(len(ids) != 1 for ids in candidate_set_by_keyword.values()):
        raise RuntimeError("candidate set changed across the assigned intent trajectory")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        manifest_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )
    level_counts: dict[str, int] = {}
    for row in rows:
        level = str(row["purpose_level"])
        level_counts[level] = level_counts.get(level, 0) + 1
    report = "\n".join(
        [
            "# Search-purpose prompt pilot report",
            "",
            "> This is a prompt-only engineering scaffold. No model inference or",
            "> scientific result was produced.",
            "",
            "The assigned axis moves from informational understanding to immediate",
            "action. The current deterministic generator uses a finite phrase schedule",
            "and is not the final semantically validated continuum.",
            "",
            "## Summary",
            "",
            f"- Prompt instances: {len(rows)}",
            f"- Keywords: {len(keyword_candidates)}",
            f"- Assigned intensity values: {len(normalized_grid)}",
            f"- Style seeds: {len(style_seeds)}",
            f"- Top N: {top_n}",
            f"- Unique templates: {len({str(row['prompt_hash']) for row in rows})}",
            f"- Unique candidate sets: {len({str(row['candidate_set_id']) for row in rows})}",
            f"- Axis specification: {AXIS_SPECIFICATION_VERSION}",
            "",
            "## Finite purpose schedule",
            "",
            *[f"- `{level}`: {count}" for level, count in sorted(level_counts.items())],
            "",
            "DataForSEO intent labels may describe keywords, but classifier probabilities",
            "must not be interpreted as assigned action intensity.",
            "",
        ]
    )
    _atomic_text(report_path, report)
    return SearchPurposePilotArtifacts(manifest_path, report_path, len(rows))


def _manifest_row(
    template: SearchPurposePromptRecord,
    rendered: RenderedSearchPurposePrompt,
    generated_at: str,
) -> dict[str, object]:
    return {
        "prompt_instance_id": rendered.prompt_instance_id,
        "rendered_prompt_hash": rendered.rendered_prompt_hash,
        "prompt_id": template.prompt_id,
        "prompt_hash": template.prompt_hash,
        "assigned_action_intensity": template.assigned_action_intensity,
        "style_seed": template.style_seed,
        "top_n": template.top_n,
        "style_plan": asdict(template.style_plan),
        "purpose_level": template.purpose_level,
        "purpose_clause": template.purpose_clause,
        "keyword": rendered.keyword,
        "top_n": rendered.top_n,
        "candidate_set_id": rendered.candidate_set_id,
        "candidates": [asdict(candidate) for candidate in rendered.candidates],
        "prompt_template": template.prompt_template,
        "rendered_prompt": rendered.rendered_prompt,
        "prompt_space_version": template.prompt_space_version,
        "axis_specification_version": template.axis_specification_version,
        "generator_backend": template.generator_backend,
        "generated_at": generated_at,
        "pilot_manifest_version": PILOT_MANIFEST_VERSION,
    }


def _single_line(value: str) -> str:
    return " ".join(value.replace("\x00", "").split())


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
