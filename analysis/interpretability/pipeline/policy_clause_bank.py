"""Offline generation of unvalidated policy-clause candidates.

The module is importable without Torch, model weights, API credentials, or a
GPU.  Real model dependencies are loaded only when the repository-local
provider is explicitly constructed.
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
from typing import Callable, Mapping, Protocol

from .prompt_continuum import (
    PromptGenerationRequest,
    PromptRecord,
    StylePlan,
    TemplatePromptGenerator,
    _normalize_template,
)

SPECIFICATION_VERSION = "policy-clause-spec-v1"
META_PROMPT_VERSION = "policy-clause-meta-prompt-v1"
DEFAULT_MAX_CLAUSE_CHARS = 420
NEUTRAL_POLICY_CLAUSE = (
    "Apply no source-type preference and rank all candidates by query relevance."
)
_SPEC_DIR = Path(__file__).with_name("specs")
_SPEC_PATH = _SPEC_DIR / "policy_clause_spec_v1.json"
_META_PROMPT_PATH = _SPEC_DIR / "policy_clause_meta_prompt_v1.txt"

_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("freshness", r"\bfresh(?:ness)?\b"),
    ("recency", r"\brecen(?:t|cy)\b"),
    ("authority", r"\bauthorit(?:y|ative)\b"),
    ("credibility", r"\bcredib(?:le|ility)\b"),
    ("citations", r"\bcitat(?:ion|ions|e|ed)\b"),
    ("popularity", r"\bpopular(?:ity)?\b"),
    ("brand fame", r"\b(?:brand fame|famous brands?)\b"),
    ("page length", r"\b(?:page length|word count)\b"),
    ("statistics", r"\bstatistic(?:s|al)?\b"),
    ("numerical density", r"\bnumerical density\b"),
    ("writing quality", r"\bwriting quality\b"),
    ("comprehensiveness", r"\bcomprehensive(?:ness)?\b"),
    ("review scores", r"\b(?:review scores?|ratings?)\b"),
)

_EXCLUSION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("exclude", r"\bexclude\b"),
    ("forbid", r"\bforbid(?:den)?\b"),
    ("never rank", r"\bnever rank\b"),
    ("must not rank", r"\bmust not rank\b"),
    ("only rank", r"\bonly rank\b"),
    ("rank only", r"\brank only\b"),
    ("always above", r"\balways rank .+ above\b"),
    ("regardless of relevance", r"\bregardless of relevance\b"),
    ("categorical", r"\bcategorical(?:ly)?\b"),
)


def canonical_bias(assigned_bias: float) -> str:
    """Return the identity representation of B with exactly six decimals."""

    _validate_bias(assigned_bias)
    value = 0.0 if float(assigned_bias) == 0.0 else float(assigned_bias)
    return f"{value:.6f}"


@dataclass(frozen=True, slots=True)
class PolicyClauseGenerationRequest:
    assigned_bias: float
    style_seed: int
    generation_seed: int
    specification_version: str
    generator_model: str

    def __post_init__(self) -> None:
        _validate_bias(self.assigned_bias)
        for name, value in (
            ("style_seed", self.style_seed),
            ("generation_seed", self.generation_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name, value in (
            ("specification_version", self.specification_version),
            ("generator_model", self.generator_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PolicyClauseRecord:
    clause_id: str
    clause_hash: str
    assigned_bias: float
    style_seed: int
    generation_seed: int
    policy_clause: str
    raw_model_output: str
    specification_version: str
    generator_backend: str
    generator_model: str
    generation_parameters: dict[str, object]
    validation_status: str
    generated_at: str


@dataclass(frozen=True, slots=True)
class StructuralValidationResult:
    accepted: bool
    policy_clause: str | None
    rejection_reasons: tuple[str, ...]


class PolicyClauseProvider(Protocol):
    backend_name: str

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        """Return the provider's exact raw text."""


class PolicyClauseStructuralError(ValueError):
    def __init__(self, reasons: tuple[str, ...], raw_model_output: str):
        self.reasons = reasons
        self.raw_model_output = raw_model_output
        super().__init__("policy clause failed structural validation: " + ", ".join(reasons))


class FakePolicyClauseProvider:
    """Deterministic test provider; never loads or calls a model."""

    backend_name = "fake-deterministic"

    _LOW = (
        "Use first-party software-product status as a light tie-breaker while ranking primarily by query relevance.",
        "Give first-party software-product sources a limited edge only when candidates are similarly relevant.",
        "Let first-party software-product status provide a mild preference without overriding clear relevance differences.",
    )
    _MIDDLE = (
        "Prefer first-party software-product sources when candidates are comparably relevant, while allowing relevance differences to determine order.",
        "Give first-party software-product sources a meaningful preference among similarly relevant candidates while retaining query relevance.",
        "Favor first-party software-product sources when relevance is close, but continue to order candidates by query relevance.",
    )
    _HIGH = (
        "Strongly favor first-party software-product sources while allowing a substantially more relevant non-first-party source to rank higher.",
        "Give first-party software-product sources substantial priority, but preserve a higher rank for a clearly more relevant non-first-party candidate.",
        "Apply a strong first-party software-product preference without overriding a clear and substantial advantage in query relevance.",
    )

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        self.call_count += 1
        match = re.search(r"Assigned preference intensity B:\s*([0-9.]+)", request_text)
        if match is None:
            raise ValueError("fake provider could not locate assigned B")
        assigned_bias = float(match.group(1))
        seed = int(generation_config["generation_seed"])
        if assigned_bias == 0.0:
            clause = NEUTRAL_POLICY_CLAUSE
        elif assigned_bias < 1 / 3:
            clause = self._LOW[seed % len(self._LOW)]
        elif assigned_bias < 2 / 3:
            clause = self._MIDDLE[seed % len(self._MIDDLE)]
        else:
            clause = self._HIGH[seed % len(self._HIGH)]
        return json.dumps({"policy_clause": clause}, ensure_ascii=False)


class RepositoryLocalPolicyClauseProvider:
    """Lazy adapter around the repository's existing local HF ranker."""

    backend_name = "repository-local-ranker"

    def __init__(self, ranker) -> None:
        self._ranker = ranker

    @classmethod
    def from_model(
        cls, model: str, *, precision: str = "full"
    ) -> "RepositoryLocalPolicyClauseProvider":
        from ..utils import make_ranker

        return cls(make_ranker("local", model, precision=precision))

    def generate(
        self, request_text: str, generation_config: Mapping[str, object]
    ) -> str:
        top_p = float(generation_config.get("top_p", 1.0))
        if top_p != 1.0:
            raise ValueError(
                "repository LocalRanker fixes top_p=1.0; configure top_p=1.0"
            )
        seed = int(generation_config["generation_seed"])
        max_tokens = int(generation_config["max_new_tokens"])
        temperature = float(generation_config["temperature"])
        import torch

        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return self._ranker.rank(
                request_text,
                max_tokens=max_tokens,
                temperature=temperature,
            )


def load_semantic_specification(
    version: str = SPECIFICATION_VERSION,
) -> dict[str, object]:
    if version != SPECIFICATION_VERSION:
        raise ValueError(f"unsupported policy-clause specification: {version!r}")
    specification = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    if specification.get("specification_version") != version:
        raise ValueError("semantic specification file/version mismatch")
    return specification


def load_meta_prompt(version: str = META_PROMPT_VERSION) -> str:
    if version != META_PROMPT_VERSION:
        raise ValueError(f"unsupported policy-clause meta-prompt: {version!r}")
    return _META_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_policy_clause_request(request: PolicyClauseGenerationRequest) -> str:
    """Render the exact versioned generation request without any real ranking data."""

    load_semantic_specification(request.specification_version)
    style_plan = TemplatePromptGenerator._build_style_plan(request.style_seed)
    rendered = load_meta_prompt().replace(
        "{{SPECIFICATION_VERSION}}", request.specification_version
    ).replace(
        "{{ASSIGNED_BIAS}}", canonical_bias(request.assigned_bias)
    ).replace(
        "{{STYLE_PLAN_JSON}}",
        json.dumps(asdict(style_plan), sort_keys=True, separators=(",", ":")),
    )
    return rendered


def default_generation_parameters(
    *,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_p: float = 1.0,
    precision: str = "full",
) -> dict[str, object]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not precision:
        raise ValueError("precision must be nonempty")
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "do_sample": temperature > 0,
        "precision": precision,
        "meta_prompt_version": META_PROMPT_VERSION,
        "meta_prompt_hash": hashlib.sha256(
            load_meta_prompt().encode("utf-8")
        ).hexdigest(),
    }


def validate_policy_clause_output(
    raw_model_output: str,
    request: PolicyClauseGenerationRequest,
    *,
    max_clause_chars: int = DEFAULT_MAX_CLAUSE_CHARS,
) -> StructuralValidationResult:
    """Apply lexical/structural checks only; no semantic claim is made."""

    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        return StructuralValidationResult(False, None, ("empty_model_output",))
    try:
        payload = json.loads(raw_model_output)
    except json.JSONDecodeError:
        return StructuralValidationResult(False, None, ("invalid_json",))
    if not isinstance(payload, dict) or set(payload) != {"policy_clause"}:
        return StructuralValidationResult(False, None, ("invalid_json_shape",))
    clause = payload.get("policy_clause")
    if not isinstance(clause, str) or not clause.strip():
        return StructuralValidationResult(False, None, ("empty_policy_clause",))
    clause = _normalize_clause(clause)
    reasons: list[str] = []
    if len(clause) > max_clause_chars:
        reasons.append("clause_too_long")
    if _contains_numeric_bias(clause, request.assigned_bias):
        reasons.append("numeric_bias_exposed")
    lowered = clause.lower()
    if re.search(r"\brelevan(?:ce|t)\b", lowered) is None:
        reasons.append("missing_query_relevance")
    if request.assigned_bias > 0.0 and "first-party" not in lowered:
        reasons.append("missing_first_party_preference")
    for label, pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append(f"forbidden_criterion:{label}")
    for label, pattern in _EXCLUSION_PATTERNS:
        if re.search(pattern, lowered):
            reasons.append(f"hard_exclusion:{label}")
    if re.search(r"\b(?:explain|explanation|reasoning|commentary)\b", lowered):
        reasons.append("explanation_language")
    return StructuralValidationResult(not reasons, clause, tuple(dict.fromkeys(reasons)))


class PolicyClauseGenerator:
    """Generate validated-structurally but scientifically unvalidated candidates."""

    def __init__(
        self,
        provider: PolicyClauseProvider,
        *,
        generation_parameters: Mapping[str, object] | None = None,
        cache_directory: str | Path | None = None,
        max_clause_chars: int = DEFAULT_MAX_CLAUSE_CHARS,
        generated_at_factory: Callable[[], str] | None = None,
    ) -> None:
        self.provider = provider
        self.generation_parameters = dict(
            generation_parameters or default_generation_parameters()
        )
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self.max_clause_chars = max_clause_chars
        self.generated_at_factory = generated_at_factory or _utc_now

    def generate(self, request: PolicyClauseGenerationRequest) -> PolicyClauseRecord:
        request_text = build_policy_clause_request(request)
        parameters = dict(self.generation_parameters)
        parameters["generation_seed"] = request.generation_seed
        _validate_generation_metadata(parameters)
        cache_key = policy_clause_cache_key(
            request,
            provider_backend=self.provider.backend_name,
            generation_parameters=parameters,
            request_text=request_text,
        )
        cache_path = (
            self.cache_directory / f"{cache_key}.json"
            if self.cache_directory is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            return _record_from_dict(json.loads(cache_path.read_text(encoding="utf-8")))

        raw_output = self.provider.generate(request_text, parameters)
        validation = validate_policy_clause_output(
            raw_output, request, max_clause_chars=self.max_clause_chars
        )
        if not validation.accepted or validation.policy_clause is None:
            raise PolicyClauseStructuralError(validation.rejection_reasons, raw_output)
        clause_hash = hashlib.sha256(
            _normalize_clause(validation.policy_clause).encode("utf-8")
        ).hexdigest()
        clause_id = (
            f"{request.specification_version}:b{canonical_bias(request.assigned_bias)}:"
            f"s{request.style_seed}:g{request.generation_seed}:{clause_hash[:16]}"
        )
        record = PolicyClauseRecord(
            clause_id=clause_id,
            clause_hash=clause_hash,
            assigned_bias=float(request.assigned_bias),
            style_seed=request.style_seed,
            generation_seed=request.generation_seed,
            policy_clause=validation.policy_clause,
            raw_model_output=raw_output,
            specification_version=request.specification_version,
            generator_backend=self.provider.backend_name,
            generator_model=request.generator_model,
            generation_parameters=parameters,
            validation_status="unvalidated",
            generated_at=self.generated_at_factory(),
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(cache_path, policy_clause_record_to_dict(record))
        return record


class HybridPromptComposer:
    """Combine a persisted candidate clause with the Milestone 1 style wrapper."""

    backend_name = "hybrid-policy-clause-bank"

    def compose(
        self,
        *,
        assigned_bias: float,
        style_seed: int,
        policy_clause: str,
        top_n: int = 10,
        prompt_space_version: str = "hybrid-pilot-v1",
    ) -> PromptRecord:
        request = PromptGenerationRequest(
            assigned_bias=assigned_bias,
            style_seed=style_seed,
            top_n=top_n,
            prompt_space_version=prompt_space_version,
        )
        style_plan = TemplatePromptGenerator._build_style_plan(style_seed)
        clause = (
            NEUTRAL_POLICY_CLAUSE
            if float(assigned_bias) == 0.0
            else _normalize_clause(policy_clause)
        )
        if not clause:
            raise ValueError("policy_clause must be nonempty")
        instruction = TemplatePromptGenerator._instruction_block(style_plan)
        output = TemplatePromptGenerator._output_clause(style_plan.output_contract)
        inputs = "Query:\n{QUERY}\n\nCandidates:\n{CANDIDATES}"
        if style_plan.clause_order == "instructions_first":
            blocks = (instruction, clause, output, inputs)
        else:
            blocks = (inputs, instruction, clause, output)
        template = _normalize_template("\n\n".join(blocks))
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            if placeholder not in template:
                raise ValueError(f"composed prompt lost required placeholder {placeholder}")
        if _contains_numeric_bias(template, request.assigned_bias):
            raise ValueError("composed prompt exposes numeric assigned B")
        prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return PromptRecord(
            prompt_id=f"{prompt_space_version}:{prompt_hash[:16]}",
            prompt_hash=prompt_hash,
            assigned_bias=float(assigned_bias),
            style_seed=style_seed,
            style_plan=style_plan,
            prompt_template=template,
            prompt_space_version=prompt_space_version,
            generator_backend=self.backend_name,
        )


def policy_clause_cache_key(
    request: PolicyClauseGenerationRequest,
    *,
    provider_backend: str,
    generation_parameters: Mapping[str, object],
    request_text: str,
) -> str:
    payload = {
        "assigned_bias": canonical_bias(request.assigned_bias),
        "style_seed": request.style_seed,
        "generation_seed": request.generation_seed,
        "specification_version": request.specification_version,
        "generator_model": request.generator_model,
        "provider_backend": provider_backend,
        "generation_parameters": dict(generation_parameters),
        "request_text_hash": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def policy_clause_record_to_dict(record: PolicyClauseRecord) -> dict[str, object]:
    return {
        "clause_id": record.clause_id,
        "clause_hash": record.clause_hash,
        "assigned_bias": record.assigned_bias,
        "style_seed": record.style_seed,
        "generation_seed": record.generation_seed,
        "policy_clause": record.policy_clause,
        "raw_model_output": record.raw_model_output,
        "specification_version": record.specification_version,
        "generator_backend": record.generator_backend,
        "generator_model": record.generator_model,
        "generation_parameters": record.generation_parameters,
        "validation_status": record.validation_status,
        "generated_at": record.generated_at,
    }


def _record_from_dict(row: Mapping[str, object]) -> PolicyClauseRecord:
    record = PolicyClauseRecord(
        clause_id=str(row["clause_id"]),
        clause_hash=str(row["clause_hash"]),
        assigned_bias=float(row["assigned_bias"]),
        style_seed=int(row["style_seed"]),
        generation_seed=int(row["generation_seed"]),
        policy_clause=str(row["policy_clause"]),
        raw_model_output=str(row["raw_model_output"]),
        specification_version=str(row["specification_version"]),
        generator_backend=str(row["generator_backend"]),
        generator_model=str(row["generator_model"]),
        generation_parameters=dict(row["generation_parameters"]),  # type: ignore[arg-type]
        validation_status=str(row["validation_status"]),
        generated_at=str(row["generated_at"]),
    )
    expected_hash = hashlib.sha256(
        _normalize_clause(record.policy_clause).encode("utf-8")
    ).hexdigest()
    if record.clause_hash != expected_hash:
        raise ValueError("cached policy clause has an invalid clause_hash")
    expected_suffix = f":{expected_hash[:16]}"
    if not record.clause_id.endswith(expected_suffix):
        raise ValueError("cached policy clause has an invalid clause_id")
    if record.validation_status != "unvalidated":
        raise ValueError("candidate cache must contain unvalidated records")
    return record


def _validate_bias(assigned_bias: float) -> None:
    if isinstance(assigned_bias, bool) or not isinstance(assigned_bias, (int, float)):
        raise TypeError("assigned_bias must be numeric")
    if not math.isfinite(float(assigned_bias)) or not 0.0 <= float(assigned_bias) <= 1.0:
        raise ValueError("assigned_bias must be in [0, 1]")


def _validate_generation_metadata(parameters: Mapping[str, object]) -> None:
    required = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "do_sample",
        "precision",
        "meta_prompt_version",
        "meta_prompt_hash",
        "generation_seed",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError("incomplete generation metadata: " + ", ".join(missing))


def _contains_numeric_bias(text: str, assigned_bias: float) -> bool:
    value = float(assigned_bias)
    canonical = canonical_bias(value)
    compact = canonical.rstrip("0").rstrip(".")
    if compact.startswith("0."):
        compact_without_zero = compact[1:]
    else:
        compact_without_zero = compact
    representations = {canonical, compact, compact_without_zero}
    percent = value * 100.0
    percent_text = f"{percent:.6f}".rstrip("0").rstrip(".")
    patterns = []
    for representation in representations:
        if representation:
            patterns.append(
                rf"(?<![\d.]){re.escape(representation)}(?![\d.])"
            )
    patterns.append(rf"(?<![\d.]){re.escape(percent_text)}\s*(?:%|percent)\b")
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_clause(clause: str) -> str:
    normalized = clause.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
