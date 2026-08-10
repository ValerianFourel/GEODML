"""Prompt-only calibration corpus generation, serialization, and auditing.

This module uses only the deterministic :class:`TemplatePromptGenerator`.  It
does not run a reranking model or perform semantic validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Iterable, Sequence

from .prompt_continuum import (
    PromptGenerationRequest,
    PromptRecord,
    StylePlan,
    TemplatePromptGenerator,
    _normalize_template,
)

__all__ = [
    "CALIBRATION_MANIFEST_VERSION",
    "DEFAULT_B_GRID",
    "DEFAULT_FIRST_STYLE_SEED",
    "DEFAULT_NUMBER_STYLE_SEEDS",
    "CalibrationArtifacts",
    "build_calibration_report",
    "generate_calibration_records",
    "load_calibration_manifest",
    "write_calibration_corpus",
]

CALIBRATION_MANIFEST_VERSION = "1"
DEFAULT_B_GRID = tuple(round(step / 10, 1) for step in range(11))
DEFAULT_NUMBER_STYLE_SEEDS = 20
DEFAULT_FIRST_STYLE_SEED = 0
DEFAULT_TOP_N = 10
DEFAULT_PROMPT_SPACE_VERSION = "template-v1"
MANIFEST_FILENAME = "prompt_calibration.jsonl"
REPORT_FILENAME = "prompt_calibration_report.md"

FORBIDDEN_CRITERIA = (
    "freshness",
    "recency",
    "authority",
    "citations",
    "popularity",
    "brand fame",
    "page length",
    "statistics density",
    "statistical density",
    "writing quality",
)

_STYLE_PLAN_FIELDS = tuple(StylePlan.__dataclass_fields__)
_REQUIRED_MANIFEST_FIELDS = (
    "prompt_id",
    "prompt_hash",
    "assigned_bias",
    "style_seed",
    "top_n",
    "prompt_space_version",
    "generator_backend",
    "style_plan",
    "prompt_template",
    "generated_at",
    "calibration_manifest_version",
)


@dataclass(frozen=True, slots=True)
class CalibrationArtifacts:
    """Paths and records produced by one calibration-corpus write."""

    manifest_path: Path
    report_path: Path
    records: tuple[PromptRecord, ...]


def generate_calibration_records(
    *,
    b_grid: Iterable[float] = DEFAULT_B_GRID,
    number_style_seeds: int = DEFAULT_NUMBER_STYLE_SEEDS,
    first_style_seed: int = DEFAULT_FIRST_STYLE_SEED,
    top_n: int = DEFAULT_TOP_N,
    prompt_space_version: str = DEFAULT_PROMPT_SPACE_VERSION,
) -> tuple[PromptRecord, ...]:
    """Generate one prompt trajectory over ``b_grid`` for every style seed.

    Seeds are the outer loop so rows for a fixed ``S`` form a contiguous
    trajectory from the smallest to largest requested ``B``.
    """

    biases = _validate_b_grid(b_grid)
    _validate_seed_range(number_style_seeds, first_style_seed)
    generator = TemplatePromptGenerator()
    records: list[PromptRecord] = []
    for style_seed in range(first_style_seed, first_style_seed + number_style_seeds):
        for assigned_bias in biases:
            request = PromptGenerationRequest(
                assigned_bias=assigned_bias,
                style_seed=style_seed,
                top_n=top_n,
                prompt_space_version=prompt_space_version,
            )
            records.append(generator.generate(request))
    return tuple(records)


def write_calibration_corpus(
    output_directory: str | Path,
    *,
    b_grid: Iterable[float] = DEFAULT_B_GRID,
    number_style_seeds: int = DEFAULT_NUMBER_STYLE_SEEDS,
    first_style_seed: int = DEFAULT_FIRST_STYLE_SEED,
    top_n: int = DEFAULT_TOP_N,
    prompt_space_version: str = DEFAULT_PROMPT_SPACE_VERSION,
    overwrite: bool = False,
    generated_at: str | None = None,
) -> CalibrationArtifacts:
    """Generate and atomically write the JSONL manifest and Markdown report."""

    output_dir = Path(output_directory)
    manifest_path = output_dir / MANIFEST_FILENAME
    report_path = output_dir / REPORT_FILENAME
    existing = [path for path in (manifest_path, report_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing calibration artifact(s): {paths}; "
            "pass overwrite=True or --overwrite explicitly"
        )

    biases = _validate_b_grid(b_grid)
    records = generate_calibration_records(
        b_grid=biases,
        number_style_seeds=number_style_seeds,
        first_style_seed=first_style_seed,
        top_n=top_n,
        prompt_space_version=prompt_space_version,
    )
    timestamp = generated_at or _utc_now()
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("generated_at must be a non-empty string")

    manifest_text = _serialize_manifest(records, top_n=top_n, generated_at=timestamp)
    report_text = build_calibration_report(
        records,
        b_grid=biases,
        number_style_seeds=number_style_seeds,
        first_style_seed=first_style_seed,
        top_n=top_n,
        prompt_space_version=prompt_space_version,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_temp: Path | None = None
    report_temp: Path | None = None
    try:
        manifest_temp = _write_temp_file(manifest_path, manifest_text)
        report_temp = _write_temp_file(report_path, report_text)
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
        os.replace(report_temp, report_path)
        report_temp = None
    finally:
        for temp_path in (manifest_temp, report_temp):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return CalibrationArtifacts(
        manifest_path=manifest_path,
        report_path=report_path,
        records=records,
    )


def load_calibration_manifest(path: str | Path) -> tuple[PromptRecord, ...]:
    """Load JSONL rows into validated typed records.

    The loader checks required fields, request bounds, normalized SHA-256
    hashes, the ``<version>:<16-char-hash>`` ID rule, metadata consistency,
    and conflicting reuse of a prompt ID.
    """

    manifest_path = Path(path)
    records: list[PromptRecord] = []
    prompt_id_content: dict[str, tuple[str, str]] = {}
    common_metadata: dict[str, object] = {}
    metadata_fields = (
        "top_n",
        "prompt_space_version",
        "generator_backend",
        "generated_at",
        "calibration_manifest_version",
    )

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"line {line_number}: blank JSONL rows are not allowed")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")

            missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"line {line_number}: missing required field(s): {', '.join(missing)}"
                )

            request = _request_from_row(row, line_number)
            style_plan = _style_plan_from_row(row, line_number)
            prompt_template = row["prompt_template"]
            prompt_hash = row["prompt_hash"]
            prompt_id = row["prompt_id"]
            if not isinstance(prompt_template, str):
                raise ValueError(f"line {line_number}: prompt_template must be a string")
            if not isinstance(prompt_hash, str):
                raise ValueError(f"line {line_number}: prompt_hash must be a string")
            if not isinstance(prompt_id, str):
                raise ValueError(f"line {line_number}: prompt_id must be a string")

            normalized = _normalize_template(prompt_template)
            expected_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if prompt_hash != expected_hash:
                raise ValueError(
                    f"line {line_number}: prompt_hash does not match normalized "
                    "prompt_template"
                )

            content_identity = (prompt_hash, normalized)
            previous_content = prompt_id_content.get(prompt_id)
            if previous_content is not None and previous_content != content_identity:
                raise ValueError(
                    f"line {line_number}: duplicate prompt_id {prompt_id!r} maps to "
                    "different prompt content"
                )
            prompt_id_content[prompt_id] = content_identity

            expected_id = f"{request.prompt_space_version}:{prompt_hash[:16]}"
            if prompt_id != expected_id:
                raise ValueError(
                    f"line {line_number}: prompt_id does not match documented ID rule; "
                    f"expected {expected_id!r}"
                )

            _validate_row_metadata(row, line_number, common_metadata, metadata_fields)
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt_hash=prompt_hash,
                    assigned_bias=float(request.assigned_bias),
                    style_seed=request.style_seed,
                    style_plan=style_plan,
                    prompt_template=prompt_template,
                    prompt_space_version=request.prompt_space_version,
                    generator_backend=row["generator_backend"],
                )
            )

    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return tuple(records)


def build_calibration_report(
    records: Sequence[PromptRecord],
    *,
    b_grid: Iterable[float],
    number_style_seeds: int,
    first_style_seed: int,
    top_n: int,
    prompt_space_version: str,
) -> str:
    """Build the human-readable calibration audit report."""

    biases = _validate_b_grid(b_grid)
    _validate_seed_range(number_style_seeds, first_style_seed)
    requested_count = len(biases) * number_style_seeds
    generated_count = len(records)
    unique_hashes = len({record.prompt_hash for record in records})
    duplicate_hash_count = generated_count - unique_hashes
    unique_style_plans = len({record.style_plan for record in records})
    lengths = [len(record.prompt_template) for record in records]
    counts_by_bias = Counter(record.assigned_bias for record in records)

    regenerated = generate_calibration_records(
        b_grid=biases,
        number_style_seeds=number_style_seeds,
        first_style_seed=first_style_seed,
        top_n=top_n,
        prompt_space_version=prompt_space_version,
    )
    reproducible = [
        (record.prompt_id, record.prompt_hash, record.prompt_template)
        for record in records
    ] == [
        (record.prompt_id, record.prompt_hash, record.prompt_template)
        for record in regenerated
    ]

    hashes_by_id: dict[str, set[str]] = defaultdict(set)
    plans_by_seed: dict[int, set[StylePlan]] = defaultdict(set)
    for record in records:
        hashes_by_id[record.prompt_id].add(record.prompt_hash)
        plans_by_seed[record.style_seed].add(record.style_plan)
    ids_are_unambiguous = all(len(hashes) == 1 for hashes in hashes_by_id.values())
    styles_are_bias_independent = all(len(plans) == 1 for plans in plans_by_seed.values())
    seed_level_plans = {
        seed: next(iter(plans)) for seed, plans in plans_by_seed.items() if plans
    }
    distinct_seed_plan_count = len(set(seed_level_plans.values()))
    seed_plan_ratio = (
        distinct_seed_plan_count / len(seed_level_plans) if seed_level_plans else 0.0
    )
    seeds_normally_differ = len(seed_level_plans) <= 1 or seed_plan_ratio >= 0.80

    policy_clauses = [TemplatePromptGenerator._policy_clause(bias) for bias in biases]
    distinct_policy_realizations = len(set(policy_clauses))
    distinct_nonempty_policy_phrases = len({clause for clause in policy_clauses if clause})
    policy_order = _policy_order()
    policy_is_monotonic = all(
        policy_order.get(left, -1) <= policy_order.get(right, -1)
        for left, right in zip(policy_clauses, policy_clauses[1:])
    ) and all(clause in policy_order for clause in policy_clauses)
    adjacent_policy_duplicates = any(
        left == right for left, right in zip(policy_clauses, policy_clauses[1:])
    )

    structure = _structural_checks(records)
    lines = [
        "# Prompt calibration report",
        "",
        "> The TemplatePromptGenerator is an engineering scaffold. Although assigned B",
        "> is continuous-valued, the current natural-language policy realization uses a",
        "> finite monotonic phrase schedule. It must not be used as the final scientific",
        "> prompt generator without further semantic generation and validation.",
        "",
        "The regular B grid in this report is diagnostic. It exposes the scaffold's",
        "piecewise-constant policy wording; it is not the assignment mechanism for the",
        "later confirmatory experiment, which will randomly sample continuous B values.",
        "",
        "## Corpus summary",
        "",
        f"- Requested prompts: {requested_count}",
        f"- Generated prompts: {generated_count}",
        f"- Unique prompt hashes: {unique_hashes}",
        f"- Duplicate-hash rows (generated minus unique): {duplicate_hash_count}",
        f"- Unique style plans: {unique_style_plans}",
        f"- Prompt length in characters: minimum {min(lengths)}, "
        f"mean {statistics.mean(lengths):.2f}, median {statistics.median(lengths):.2f}, "
        f"maximum {max(lengths)}",
        f"- Distinct policy realizations across B, including no preference: "
        f"{distinct_policy_realizations}",
        f"- Distinct non-empty preference phrases across B: "
        f"{distinct_nonempty_policy_phrases}",
        "",
        "### Counts by B value",
        "",
        "| B | Count |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {_format_bias(bias)} | {counts_by_bias.get(bias, 0)} |" for bias in biases
    )
    lines.extend(
        [
            "",
            "## Reproducibility checks",
            "",
            f"- Regeneration produces identical IDs, hashes, and templates: "
            f"{_pass_fail(reproducible)}",
            f"- Every prompt ID maps to exactly one prompt hash: "
            f"{_pass_fail(ids_are_unambiguous)}",
            f"- The same S has one style plan at every B: "
            f"{_pass_fail(styles_are_bias_independent)}",
            f"- Different S values normally yield different style plans: "
            f"{_pass_fail(seeds_normally_differ)} "
            f"({distinct_seed_plan_count}/{len(seed_level_plans)} distinct; "
            "the diagnostic threshold is 80%)",
            "",
            "## Axis diagnostics",
            "",
            "The current schedule is finite and piecewise constant. Adjacent B values",
            "can therefore produce byte-identical prompts for the same S.",
            "",
            "| S | B=0 has no preference | B=1 uses strongest preference | Monotonic | Adjacent B duplicates | Distinct policy realizations |",
            "|---:|:---:|:---:|:---:|:---:|---:|",
        ]
    )
    records_by_seed: dict[int, list[PromptRecord]] = defaultdict(list)
    for record in records:
        records_by_seed[record.style_seed].append(record)
    strongest_clause = TemplatePromptGenerator._policy_clause(1.0)
    for style_seed in sorted(records_by_seed):
        trajectory = sorted(records_by_seed[style_seed], key=lambda record: record.assigned_bias)
        trajectory_biases = [record.assigned_bias for record in trajectory]
        has_zero = any(math.isclose(bias, 0.0) for bias in trajectory_biases)
        has_one = any(math.isclose(bias, 1.0) for bias in trajectory_biases)
        zero_ok = has_zero and all(
            "first-party" not in record.prompt_template.lower()
            for record in trajectory
            if math.isclose(record.assigned_bias, 0.0)
        )
        one_ok = has_one and all(
            strongest_clause in record.prompt_template
            for record in trajectory
            if math.isclose(record.assigned_bias, 1.0)
        )
        lines.append(
            f"| {style_seed} | {_yes_no(zero_ok)} | {_yes_no(one_ok)} | "
            f"{_yes_no(policy_is_monotonic)} | {_yes_no(adjacent_policy_duplicates)} | "
            f"{distinct_policy_realizations} |"
        )

    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            f"- Every prompt contains `{{QUERY}}`: {_pass_fail(structure['query'])}",
            f"- Every prompt contains `{{CANDIDATES}}`: "
            f"{_pass_fail(structure['candidates'])}",
            f"- Every prompt contains `{{TOP_N}}`: {_pass_fail(structure['top_n'])}",
            f"- Every prompt requests candidate identifiers only: "
            f"{_pass_fail(structure['identifiers_only'])}",
            f"- Every prompt prohibits explanations: "
            f"{_pass_fail(structure['no_explanations'])}",
            f"- Every prompt excludes all forbidden ranking criteria: "
            f"{_pass_fail(structure['forbidden_criteria'])}",
            "- Forbidden criteria checked: freshness, recency, authority, citations,",
            "  popularity, brand fame, page length, statistics density, and writing quality.",
            "  Relevance is intentionally permitted.",
            "",
            "## Examples",
            "",
        ]
    )
    for bias, style_seed in ((0.0, 0), (0.5, 0), (1.0, 0), (0.5, 1)):
        lines.extend(_example_lines(records, bias, style_seed))
    return "\n".join(lines).rstrip() + "\n"


def _validate_b_grid(b_grid: Iterable[float]) -> tuple[float, ...]:
    values: list[float] = []
    for value in b_grid:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("every B-grid value must be numeric")
        numeric = float(value)
        PromptGenerationRequest(numeric, 0, 1, "validation")
        values.append(numeric)
    if not values:
        raise ValueError("B grid must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError("B grid must not contain duplicate values")
    if values != sorted(values):
        raise ValueError("B grid must be in increasing order")
    return tuple(values)


def _validate_seed_range(number_style_seeds: int, first_style_seed: int) -> None:
    if isinstance(number_style_seeds, bool) or not isinstance(number_style_seeds, int):
        raise TypeError("number_style_seeds must be an integer")
    if number_style_seeds <= 0:
        raise ValueError("number_style_seeds must be greater than zero")
    if isinstance(first_style_seed, bool) or not isinstance(first_style_seed, int):
        raise TypeError("first_style_seed must be an integer")


def _serialize_manifest(
    records: Sequence[PromptRecord], *, top_n: int, generated_at: str
) -> str:
    rows = []
    for record in records:
        row = {
            "prompt_id": record.prompt_id,
            "prompt_hash": record.prompt_hash,
            "assigned_bias": record.assigned_bias,
            "style_seed": record.style_seed,
            "top_n": top_n,
            "prompt_space_version": record.prompt_space_version,
            "generator_backend": record.generator_backend,
            "style_plan": asdict(record.style_plan),
            "prompt_template": record.prompt_template,
            "generated_at": generated_at,
            "calibration_manifest_version": CALIBRATION_MANIFEST_VERSION,
        }
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(rows) + "\n"


def _request_from_row(row: dict, line_number: int) -> PromptGenerationRequest:
    try:
        return PromptGenerationRequest(
            assigned_bias=row["assigned_bias"],
            style_seed=row["style_seed"],
            top_n=row["top_n"],
            prompt_space_version=row["prompt_space_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: invalid request metadata: {exc}") from exc


def _style_plan_from_row(row: dict, line_number: int) -> StylePlan:
    raw_style_plan = row["style_plan"]
    if not isinstance(raw_style_plan, dict):
        raise ValueError(f"line {line_number}: style_plan must be an object")
    missing = [field for field in _STYLE_PLAN_FIELDS if field not in raw_style_plan]
    extra = [field for field in raw_style_plan if field not in _STYLE_PLAN_FIELDS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError(f"line {line_number}: invalid style_plan ({'; '.join(details)})")
    if any(not isinstance(raw_style_plan[field], str) for field in _STYLE_PLAN_FIELDS):
        raise ValueError(f"line {line_number}: all style_plan values must be strings")
    return StylePlan(**raw_style_plan)


def _validate_row_metadata(
    row: dict,
    line_number: int,
    common_metadata: dict[str, object],
    metadata_fields: Sequence[str],
) -> None:
    for field in metadata_fields:
        value = row[field]
        if field != "top_n" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"line {line_number}: {field} must be a non-empty string")
        if (
            field == "calibration_manifest_version"
            and value != CALIBRATION_MANIFEST_VERSION
        ):
            raise ValueError(
                f"line {line_number}: unsupported calibration_manifest_version "
                f"{value!r}; expected {CALIBRATION_MANIFEST_VERSION!r}"
            )
        if field not in common_metadata:
            common_metadata[field] = value
        elif common_metadata[field] != value:
            raise ValueError(
                f"line {line_number}: {field} differs from earlier manifest rows"
            )


def _write_temp_file(destination: Path, content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _policy_order() -> dict[str, int]:
    representative_biases = (0.0, 0.25, 0.5, 0.75, 1.0)
    return {
        TemplatePromptGenerator._policy_clause(bias): level
        for level, bias in enumerate(representative_biases)
    }


def _structural_checks(records: Sequence[PromptRecord]) -> dict[str, bool]:
    templates = [record.prompt_template for record in records]
    return {
        "query": all("{QUERY}" in template for template in templates),
        "candidates": all("{CANDIDATES}" in template for template in templates),
        "top_n": all("{TOP_N}" in template for template in templates),
        "identifiers_only": all(_requests_identifiers_only(template) for template in templates),
        "no_explanations": all(_prohibits_explanations(template) for template in templates),
        "forbidden_criteria": all(not _find_forbidden_criteria(template) for template in templates),
    }


def _requests_identifiers_only(template: str) -> bool:
    lowered = template.lower()
    return "candidate identifiers" in lowered and (
        "identifiers only" in lowered or "and nothing else" in lowered
    )


def _prohibits_explanations(template: str) -> bool:
    lowered = template.lower()
    return any(
        wording in lowered
        for wording in (
            "no explanation",
            "do not provide an explanation",
            "do not include an explanation",
        )
    )


def _find_forbidden_criteria(template: str) -> tuple[str, ...]:
    lowered = template.lower()
    return tuple(criterion for criterion in FORBIDDEN_CRITERIA if criterion in lowered)


def _example_lines(
    records: Sequence[PromptRecord], assigned_bias: float, style_seed: int
) -> list[str]:
    matching = [
        record
        for record in records
        if math.isclose(record.assigned_bias, assigned_bias)
        and record.style_seed == style_seed
    ]
    heading = f"### B = {_format_bias(assigned_bias)}, S = {style_seed}"
    if not matching:
        return [heading, "", "Not present in this configured corpus.", ""]
    record = matching[0]
    return [heading, "", "```text", record.prompt_template, "```", ""]


def _format_bias(value: float) -> str:
    rendered = f"{value:.10f}".rstrip("0").rstrip(".")
    return rendered if "." in rendered else f"{rendered}.0"


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"
