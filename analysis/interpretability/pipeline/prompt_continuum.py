"""Deterministic scaffold for the randomized prompt continuum ``P = G(B, S)``.

This module is intentionally limited to tests and smoke runs.  The finite set
of hand-written policy phrases is not the final scientific prompt generator.
Legacy neutral and biased prompts remain in :mod:`.prompts` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

__all__ = [
    "PromptGenerationRequest",
    "PromptRecord",
    "StylePlan",
    "TemplatePromptGenerator",
]


@dataclass(frozen=True, slots=True)
class PromptGenerationRequest:
    """Inputs that identify one deterministic prompt-template request."""

    assigned_bias: float
    style_seed: int
    top_n: int
    prompt_space_version: str

    def __post_init__(self) -> None:
        if isinstance(self.assigned_bias, bool) or not isinstance(
            self.assigned_bias, (int, float)
        ):
            raise TypeError("assigned_bias must be a float in [0, 1]")
        if not math.isfinite(self.assigned_bias) or not 0.0 <= self.assigned_bias <= 1.0:
            raise ValueError("assigned_bias must be in [0, 1]")
        if isinstance(self.style_seed, bool) or not isinstance(self.style_seed, int):
            raise TypeError("style_seed must be an integer")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int):
            raise TypeError("top_n must be an integer")
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if not isinstance(self.prompt_space_version, str):
            raise TypeError("prompt_space_version must be a string")
        if not self.prompt_space_version.strip():
            raise ValueError("prompt_space_version must not be empty")


@dataclass(frozen=True, slots=True)
class StylePlan:
    """Surface choices derived only from ``style_seed``."""

    ranking_verb: str
    syntax: str
    clause_order: str
    tone: str
    verbosity: str
    output_contract: str


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """A generated prompt template and its reproducibility metadata."""

    prompt_id: str
    prompt_hash: str
    assigned_bias: float
    style_seed: int
    style_plan: StylePlan
    prompt_template: str
    prompt_space_version: str
    generator_backend: str


class TemplatePromptGenerator:
    """Generate deterministic prompt scaffolds without calling an LLM.

    Continuous ``assigned_bias`` is mapped monotonically onto a small set of
    fixed preference-strength phrases.  This makes the backend suitable for
    unit tests and CPU smoke runs, but not for the final scientific prompt
    family.
    """

    backend_name = "template"

    _RANKING_VERBS = ("Rerank", "Reorder", "Rank", "Arrange")
    _SYNTAXES = ("imperative", "request", "task_statement")
    _CLAUSE_ORDERS = ("instructions_first", "inputs_first")
    _TONES = ("direct", "formal", "courteous")
    _VERBOSITIES = ("compact", "standard")
    _OUTPUT_CONTRACTS = ("sentences", "single_sentence", "response_rule")

    def generate(self, request: PromptGenerationRequest) -> PromptRecord:
        """Return the deterministic record identified by ``request``."""

        style_plan = self._build_style_plan(request.style_seed)
        template = _normalize_template(
            self._render_template(style_plan, request.assigned_bias)
        )
        prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        prompt_id = f"{request.prompt_space_version}:{prompt_hash[:16]}"
        return PromptRecord(
            prompt_id=prompt_id,
            prompt_hash=prompt_hash,
            assigned_bias=float(request.assigned_bias),
            style_seed=request.style_seed,
            style_plan=style_plan,
            prompt_template=template,
            prompt_space_version=request.prompt_space_version,
            generator_backend=self.backend_name,
        )

    @classmethod
    def _build_style_plan(cls, style_seed: int) -> StylePlan:
        rng = random.Random(style_seed)
        return StylePlan(
            ranking_verb=rng.choice(cls._RANKING_VERBS),
            syntax=rng.choice(cls._SYNTAXES),
            clause_order=rng.choice(cls._CLAUSE_ORDERS),
            tone=rng.choice(cls._TONES),
            verbosity=rng.choice(cls._VERBOSITIES),
            output_contract=rng.choice(cls._OUTPUT_CONTRACTS),
        )

    @classmethod
    def _render_template(cls, style: StylePlan, assigned_bias: float) -> str:
        instruction_block = cls._instruction_block(style)
        inputs_block = "Query:\n{QUERY}\n\nCandidates:\n{CANDIDATES}"
        policy = cls._policy_clause(assigned_bias)
        output_contract = cls._output_clause(style.output_contract)

        trailing_clauses = [clause for clause in (policy, output_contract) if clause]
        if style.clause_order == "instructions_first":
            blocks = [instruction_block, *trailing_clauses, inputs_block]
        else:
            blocks = [inputs_block, instruction_block, *trailing_clauses]
        return "\n\n".join(blocks)

    @staticmethod
    def _instruction_block(style: StylePlan) -> str:
        verb = style.ranking_verb
        if style.syntax == "imperative":
            task = f"{verb} the supplied candidates for the supplied query by relevance."
        elif style.syntax == "request":
            task = (
                f"Please {verb.lower()} the supplied candidates for the supplied query "
                "by relevance."
            )
        else:
            task = (
                f"Your task is to {verb.lower()} the supplied candidates for the "
                "supplied query by relevance."
            )

        if style.tone == "formal":
            task = f"Follow this ranking instruction. {task}"
        elif style.tone == "courteous":
            task = f"Please follow this ranking instruction carefully. {task}"

        if style.verbosity == "standard":
            task += " Consider every supplied candidate and preserve the candidate set."
        return task

    @staticmethod
    def _policy_clause(assigned_bias: float) -> str:
        if assigned_bias == 0.0:
            return ""
        if assigned_bias <= 0.25:
            strength = "a slight"
        elif assigned_bias <= 0.50:
            strength = "a moderate"
        elif assigned_bias <= 0.75:
            strength = "a clear"
        else:
            strength = "a strong"
        return (
            f"Give {strength} preference to first-party software-product sources while "
            "continuing to rank all candidates by relevance."
        )

    @staticmethod
    def _output_clause(output_contract: str) -> str:
        if output_contract == "sentences":
            return (
                "Output exactly {TOP_N} candidate identifiers. Return identifiers only. "
                "Do not provide an explanation."
            )
        if output_contract == "single_sentence":
            return (
                "Return exactly {TOP_N} candidate identifiers only, with no explanation."
            )
        return (
            "Response rule: provide exactly {TOP_N} candidate identifiers and nothing "
            "else; do not include an explanation."
        )


def _normalize_template(template: str) -> str:
    """Canonicalize newlines and trailing whitespace before hashing."""

    normalized = template.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
