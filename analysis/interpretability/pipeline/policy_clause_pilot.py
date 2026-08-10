"""Small prompt-generation pilot orchestration; no reranking or judging."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Mapping, Sequence

from .policy_clause_bank import (
    META_PROMPT_VERSION,
    SPECIFICATION_VERSION,
    HybridPromptComposer,
    PolicyClauseGenerationRequest,
    PolicyClauseGenerator,
    PolicyClauseProvider,
    PolicyClauseRecord,
    PolicyClauseStructuralError,
    build_policy_clause_request,
    canonical_bias,
    default_generation_parameters,
    policy_clause_record_to_dict,
)

PILOT_MANIFEST_VERSION = "policy-clause-pilot-v1"
REQUESTS_FILENAME = "policy_clause_requests.jsonl"
CANDIDATES_FILENAME = "policy_clause_candidates.jsonl"
FULL_PROMPTS_FILENAME = "candidate_full_prompts.jsonl"
REPORT_FILENAME = "policy_clause_pilot_report.md"


@dataclass(frozen=True, slots=True)
class PolicyClausePilotArtifacts:
    requests_path: Path
    candidates_path: Path | None
    full_prompts_path: Path | None
    report_path: Path | None
    requests: tuple[PolicyClauseGenerationRequest, ...]
    candidates: tuple[PolicyClauseRecord, ...]


def stratified_bias_schedule(
    *,
    number_bias_values: int = 8,
    master_seed: int = 20260810,
    include_anchors: bool = False,
) -> tuple[float, ...]:
    """Sample one jittered B per equal-width stratum using a local RNG."""

    if number_bias_values <= 0:
        raise ValueError("number_bias_values must be greater than zero")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    rng = random.Random(master_seed)
    values = [
        round((stratum + rng.random()) / number_bias_values, 6)
        for stratum in range(number_bias_values)
    ]
    if include_anchors:
        values.extend((0.0, 1.0))
    return tuple(sorted(set(values)))


def build_pilot_requests(
    *,
    generator_model: str,
    number_style_seeds: int = 8,
    first_style_seed: int = 0,
    number_bias_values: int = 8,
    master_seed: int = 20260810,
    include_anchors: bool = False,
    specification_version: str = SPECIFICATION_VERSION,
) -> tuple[PolicyClauseGenerationRequest, ...]:
    if number_style_seeds <= 0:
        raise ValueError("number_style_seeds must be greater than zero")
    schedule = stratified_bias_schedule(
        number_bias_values=number_bias_values,
        master_seed=master_seed,
        include_anchors=include_anchors,
    )
    requests = []
    for style_seed in range(first_style_seed, first_style_seed + number_style_seeds):
        for assigned_bias in schedule:
            generation_seed = _generation_seed(master_seed, assigned_bias, style_seed)
            requests.append(
                PolicyClauseGenerationRequest(
                    assigned_bias=assigned_bias,
                    style_seed=style_seed,
                    generation_seed=generation_seed,
                    specification_version=specification_version,
                    generator_model=generator_model,
                )
            )
    return tuple(requests)


def write_policy_clause_pilot(
    output_directory: str | Path,
    *,
    mode: str,
    provider: PolicyClauseProvider | None,
    generator_model: str,
    number_style_seeds: int = 8,
    first_style_seed: int = 0,
    number_bias_values: int = 8,
    master_seed: int = 20260810,
    include_anchors: bool = False,
    top_n: int = 10,
    prompt_space_version: str = "hybrid-pilot-v1",
    generation_parameters: Mapping[str, object] | None = None,
    max_clause_chars: int = 420,
    overwrite: bool = False,
) -> PolicyClausePilotArtifacts:
    if mode not in {"dry-run", "generate"}:
        raise ValueError("mode must be 'dry-run' or 'generate'")
    if mode == "generate" and provider is None:
        raise ValueError("generation mode requires a provider")
    output_dir = Path(output_directory)
    requests_path = output_dir / REQUESTS_FILENAME
    candidates_path = output_dir / CANDIDATES_FILENAME
    full_prompts_path = output_dir / FULL_PROMPTS_FILENAME
    report_path = output_dir / REPORT_FILENAME
    targets = [requests_path]
    if mode == "generate":
        targets.extend((candidates_path, full_prompts_path, report_path))
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite pilot artifact(s): "
            + ", ".join(str(path) for path in existing)
        )

    parameters = dict(generation_parameters or default_generation_parameters())
    requests = build_pilot_requests(
        generator_model=generator_model,
        number_style_seeds=number_style_seeds,
        first_style_seed=first_style_seed,
        number_bias_values=number_bias_values,
        master_seed=master_seed,
        include_anchors=include_anchors,
    )
    run_generated_at = _utc_now()
    backend_name = provider.backend_name if provider is not None else "not-invoked"
    request_rows = [
        _request_row(
            request,
            provider_backend=backend_name,
            generation_parameters=parameters,
            generated_at=run_generated_at,
        )
        for request in requests
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(requests_path, request_rows)
    if mode == "dry-run":
        return PolicyClausePilotArtifacts(
            requests_path=requests_path,
            candidates_path=None,
            full_prompts_path=None,
            report_path=None,
            requests=requests,
            candidates=(),
        )

    assert provider is not None
    generator = PolicyClauseGenerator(
        provider,
        generation_parameters=parameters,
        cache_directory=output_dir / ".candidate_cache",
        max_clause_chars=max_clause_chars,
    )
    composer = HybridPromptComposer()
    candidates: list[PolicyClauseRecord] = []
    full_prompt_rows: list[dict[str, object]] = []
    failures: list[tuple[PolicyClauseGenerationRequest, tuple[str, ...]]] = []
    for request in requests:
        try:
            candidate = generator.generate(request)
            prompt = composer.compose(
                assigned_bias=request.assigned_bias,
                style_seed=request.style_seed,
                policy_clause=candidate.policy_clause,
                top_n=top_n,
                prompt_space_version=prompt_space_version,
            )
        except PolicyClauseStructuralError as exc:
            failures.append((request, exc.reasons))
            continue
        except (TypeError, ValueError) as exc:
            failures.append((request, (f"composition_or_metadata:{exc}",)))
            continue
        except Exception as exc:
            failures.append(
                (request, (f"provider_error:{type(exc).__name__}:{exc}",))
            )
            continue
        candidates.append(candidate)
        full_prompt_rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt_hash": prompt.prompt_hash,
                "clause_id": candidate.clause_id,
                "clause_hash": candidate.clause_hash,
                "assigned_bias": prompt.assigned_bias,
                "style_seed": prompt.style_seed,
                "generation_seed": candidate.generation_seed,
                "top_n": top_n,
                "style_plan": asdict(prompt.style_plan),
                "prompt_template": prompt.prompt_template,
                "prompt_space_version": prompt.prompt_space_version,
                "generator_backend": prompt.generator_backend,
                "validation_status": candidate.validation_status,
                "generated_at": candidate.generated_at,
            }
        )
    _atomic_jsonl(
        candidates_path,
        [policy_clause_record_to_dict(record) for record in candidates],
    )
    _atomic_jsonl(full_prompts_path, full_prompt_rows)
    report = build_policy_clause_pilot_report(
        requests=requests,
        candidates=candidates,
        full_prompt_rows=full_prompt_rows,
        failures=failures,
        generator_backend=provider.backend_name,
        generator_model=generator_model,
        generation_parameters=parameters,
    )
    _atomic_text(report_path, report)
    return PolicyClausePilotArtifacts(
        requests_path=requests_path,
        candidates_path=candidates_path,
        full_prompts_path=full_prompts_path,
        report_path=report_path,
        requests=requests,
        candidates=tuple(candidates),
    )


def build_policy_clause_pilot_report(
    *,
    requests: Sequence[PolicyClauseGenerationRequest],
    candidates: Sequence[PolicyClauseRecord],
    full_prompt_rows: Sequence[Mapping[str, object]],
    failures: Sequence[tuple[PolicyClauseGenerationRequest, tuple[str, ...]]],
    generator_backend: str,
    generator_model: str,
    generation_parameters: Mapping[str, object],
) -> str:
    clause_lengths = [len(record.policy_clause) for record in candidates]
    region_requests = Counter(_bias_region(request.assigned_bias) for request in requests)
    region_successes = Counter(_bias_region(record.assigned_bias) for record in candidates)
    rejection_counts = Counter(reason for _, reasons in failures for reason in reasons)
    lines = [
        "# Policy-clause pilot report",
        "",
        "> These clauses are unvalidated candidates. They must not be used for reranking",
        "> or scientific inference until the semantic-validation milestone is complete.",
        "",
        "Structural checks do not establish semantic monotonicity, axis purity, or",
        "equivalence across surface realizations.",
        "",
        "## Summary",
        "",
        f"- Requests: {len(requests)}",
        f"- Successful generations: {len(candidates)}",
        f"- Failures: {len(failures)}",
        f"- Unique policy-clause hashes: {len({r.clause_hash for r in candidates})}",
        f"- Unique complete-prompt hashes: "
        f"{len({str(row['prompt_hash']) for row in full_prompt_rows})}",
        f"- Specification version: {SPECIFICATION_VERSION}",
        f"- Meta-prompt version: {META_PROMPT_VERSION}",
        f"- Generator backend: {generator_backend}",
        f"- Generator model: {generator_model}",
    ]
    if clause_lengths:
        lines.append(
            "- Clause length in characters: "
            f"minimum {min(clause_lengths)}, mean {statistics.mean(clause_lengths):.2f}, "
            f"median {statistics.median(clause_lengths):.2f}, maximum {max(clause_lengths)}"
        )
    else:
        lines.append("- Clause length in characters: unavailable (no accepted candidates)")
    lines.extend(
        [
            "",
            "### Counts by B region",
            "",
            "| Region | Requests | Successful |",
            "|---|---:|---:|",
        ]
    )
    for region in ("low", "middle", "high"):
        lines.append(
            f"| {region} | {region_requests[region]} | {region_successes[region]} |"
        )
    lines.extend(
        [
            "",
            "### Structural rejection reasons",
            "",
        ]
    )
    if rejection_counts:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(rejection_counts.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "### Generation parameters",
            "",
            "```json",
            json.dumps(dict(generation_parameters), indent=2, sort_keys=True),
            "```",
            "",
            "## Examples",
            "",
        ]
    )
    for region in ("low", "middle", "high"):
        matching = [record for record in candidates if _bias_region(record.assigned_bias) == region]
        lines.append(f"### {region.capitalize()} B")
        lines.append("")
        if matching:
            record = matching[len(matching) // 2]
            lines.append(f"- B: `{canonical_bias(record.assigned_bias)}`")
            lines.append(f"- S: `{record.style_seed}`")
            lines.append(f"- Clause: {record.policy_clause}")
        else:
            lines.append("No accepted candidate in this region.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _request_row(
    request: PolicyClauseGenerationRequest,
    *,
    provider_backend: str,
    generation_parameters: Mapping[str, object],
    generated_at: str,
) -> dict[str, object]:
    request_text = build_policy_clause_request(request)
    request_identity = {
        "assigned_bias": canonical_bias(request.assigned_bias),
        "style_seed": request.style_seed,
        "generation_seed": request.generation_seed,
        "specification_version": request.specification_version,
        "generator_model": request.generator_model,
        "provider_backend": provider_backend,
        "generation_parameters": dict(generation_parameters),
        "meta_prompt_hash": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
    }
    request_id_hash = hashlib.sha256(
        json.dumps(request_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "request_id": f"policy-request:{request_id_hash[:20]}",
        "assigned_bias": request.assigned_bias,
        "assigned_bias_canonical": canonical_bias(request.assigned_bias),
        "style_seed": request.style_seed,
        "generation_seed": request.generation_seed,
        "specification_version": request.specification_version,
        "generator_backend": provider_backend,
        "generator_model": request.generator_model,
        "generation_parameters": dict(generation_parameters),
        "request_text": request_text,
        "generated_at": generated_at,
        "pilot_manifest_version": PILOT_MANIFEST_VERSION,
    }


def _generation_seed(master_seed: int, assigned_bias: float, style_seed: int) -> int:
    payload = f"{master_seed}:{canonical_bias(assigned_bias)}:{style_seed}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _bias_region(assigned_bias: float) -> str:
    if assigned_bias < 1 / 3:
        return "low"
    if assigned_bias < 2 / 3:
        return "middle"
    return "high"


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    content = "\n".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    if rows:
        content += "\n"
    _atomic_text(path, content)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
