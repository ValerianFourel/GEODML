#!/usr/bin/env python3
"""Build and verify one compact Hugging Face dataset of readiness text tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.audit_fully_compliant_readiness_prompts import (  # noqa: E402
    audit_fully_compliant_prompts,
)
from analysis.scripts.build_readiness_hf_dataset import (  # noqa: E402
    _public_model_reference,
    _publish as _publish_likert_dataset,
    _verify_dataset as _verify_likert_dataset,
)
from analysis.interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    delexicalize_question,
    search_review_passes_contract,
)


FORMAT_VERSION = "axisgeo-readiness-text-hf-v1"
COUNTERFACTUAL_FORMAT_VERSION = "axisgeo-readiness-text-hf-v2"
SUPPORTED_FORMAT_VERSIONS = frozenset(
    {FORMAT_VERSION, COUNTERFACTUAL_FORMAT_VERSION}
)
COUNTERFACTUAL_SCENARIO = "search_trigger_v2_relaxed_tolerance"
COUNTERFACTUAL_CONFIG = "fully_compliant_prompts_relaxed_0035"
COUNTERFACTUAL_CONTRACT = "search-trigger-v2"
COUNTERFACTUAL_TOLERANCE = 0.035
LIKERT_CONFIGS = (
    "sources",
    "prompts",
    "annotations",
    "failures",
    "missing_tasks",
)
POPULATION_CONFIGS = (
    "generated_candidates",
    "candidate_compliance_annotations",
    "fully_compliant_prompts",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--likert-dataset-root", required=True)
    finalize.add_argument("--prompt-population-root", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--rows-per-shard", type=int, default=100_000)
    finalize.add_argument("--git-commit-sha")
    annotate = commands.add_parser("annotate-counterfactual")
    annotate.add_argument("--dataset-dir", required=True)
    annotate.add_argument("--counterfactual-root", required=True)
    annotate.add_argument("--output-dir", required=True)
    annotate.add_argument("--rows-per-shard", type=int, default=100_000)
    annotate.add_argument("--git-commit-sha")
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset-dir", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--dataset-dir", required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--confirm-repo-id", required=True)
    publish.add_argument("--public", action="store_true")
    publish.add_argument("--confirm-public", action="store_true")
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, object]]:
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _dataset_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _population_sources(root: Path) -> tuple[list[Path], Path, Path]:
    merged_candidates = root / "merged" / "candidates.jsonl"
    merged_validation = root / "merged" / "validation.jsonl"
    if merged_candidates.is_file() or merged_validation.is_file():
        if not merged_candidates.is_file() or not merged_validation.is_file():
            raise ValueError("prompt population has partial merged artifacts")
        candidates = [merged_candidates]
        validation = merged_validation
    else:
        candidate_list = root / "candidate-files.txt"
        validation = root / "validation.jsonl"
        if not candidate_list.is_file() or not validation.is_file():
            raise ValueError("prompt population is not an audited checkpoint")
        candidates = [
            Path(value).resolve()
            for value in candidate_list.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if not candidates or any(not path.is_file() for path in candidates):
            raise ValueError("checkpoint candidate file list is incomplete")
    selected = root / "strict-selection" / "spatially_selected_questions.jsonl"
    if not selected.is_file():
        raise ValueError("prompt population selection is missing")
    return candidates, validation, selected


def _sanitize_model_fields(row: Mapping[str, object]) -> dict[str, object]:
    sanitized = dict(row)
    for field, value in tuple(sanitized.items()):
        if (
            field.endswith("_seed")
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            # Hash-derived generation seeds may occupy the full unsigned
            # 64-bit range (and candidate-slot offsets can exceed it).  A
            # decimal string preserves the exact deterministic value across
            # Arrow implementations without signed-integer overflow.
            sanitized[field] = str(value)
    for field in ("generator_model", "judge_model", "model"):
        value = sanitized.get(field)
        if value:
            model_path = Path(str(value))
            revision = model_path.name if model_path.is_absolute() else ""
            sanitized[field] = _public_model_reference(str(value), revision)
    sanitized["split"] = "data"
    return sanitized


def _deduplicated_rows(
    paths: Sequence[Path],
    *,
    identity_field: str,
    seen: set[str],
) -> Iterator[dict[str, object]]:
    for row in _iter_jsonl(paths):
        identity = str(row.get(identity_field, "")).strip()
        if not identity:
            raise ValueError(f"row omits {identity_field}")
        if identity in seen:
            raise ValueError(f"duplicate {identity_field}: {identity}")
        seen.add(identity)
        yield _sanitize_model_fields(row)


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "text finalization requires pyarrow (install analysis/requirements.txt)"
        ) from exc
    return pa, pq


def _write_streaming_parquet(
    *,
    output: Path,
    config_name: str,
    rows: Iterable[Mapping[str, object]],
    rows_per_shard: int,
) -> tuple[dict[str, list[str]], int]:
    if rows_per_shard <= 0:
        raise ValueError("rows per shard must be positive")
    pa, pq = _pyarrow()
    directory = output / "data" / config_name
    directory.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, object]] = []
    paths: list[str] = []
    schema = None
    count = 0

    def flush() -> None:
        nonlocal batch, schema
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if schema is None:
            schema = table.schema
        path = directory / f"data-{len(paths):05d}.parquet"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
        paths.append(str(path.relative_to(output)))
        batch = []

    for row in rows:
        batch.append(dict(row))
        count += 1
        if len(batch) >= rows_per_shard:
            flush()
    flush()
    if not count:
        raise ValueError(f"configuration is empty: {config_name}")
    return {"data": paths}, count


def _copy_likert_configs(
    source: Path,
    output: Path,
) -> tuple[list[tuple[str, dict[str, list[str]], bool]], dict[str, int]]:
    manifest = _read_json(source / "dataset_manifest.json")
    source_counts = manifest.get("table_counts", {})
    configs = []
    counts = {}
    for source_name in LIKERT_CONFIGS:
        source_directory = source / "data" / source_name
        if not source_directory.is_dir():
            continue
        config_name = f"likert_{source_name}"
        destination = output / "data" / config_name
        destination.mkdir(parents=True, exist_ok=True)
        paths_by_split: dict[str, list[str]] = {}
        for path in sorted(source_directory.glob("*.parquet")):
            split = path.stem.rsplit("-", 1)[0]
            target = destination / path.name
            shutil.copy2(path, target)
            paths_by_split.setdefault(split, []).append(
                str(target.relative_to(output))
            )
        if not paths_by_split:
            continue
        configs.append((config_name, paths_by_split, source_name == "prompts"))
        counts[config_name] = int(source_counts[source_name])
    required = {"likert_prompts", "likert_annotations"}
    if not required.issubset(counts):
        raise ValueError("Likert dataset lacks prompt or annotation configuration")
    return configs, counts


def _parquet_rows(paths: Sequence[Path]) -> Iterator[dict[str, object]]:
    _, pq = _pyarrow()
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            yield dict(row)


def _likert_graded_rows(likert_root: Path) -> Iterator[dict[str, object]]:
    prompt_paths = sorted((likert_root / "data" / "prompts").glob("*.parquet"))
    annotation_paths = sorted(
        (likert_root / "data" / "annotations").glob("*.parquet")
    )
    prompts = {str(row["item_id"]): row for row in _parquet_rows(prompt_paths)}
    for annotation in _parquet_rows(annotation_paths):
        item_id = str(annotation["item_id"])
        prompt = prompts.get(item_id)
        if prompt is None:
            raise ValueError(f"Likert annotation references unknown prompt: {item_id}")
        row = dict(annotation)
        for key, value in prompt.items():
            if key in {"item_id", "split"}:
                continue
            row[f"prompt_{key}"] = value
        yield row


def _write_dataset_card(
    output: Path,
    configs: Sequence[tuple[str, Mapping[str, Sequence[str]], bool]],
    counts: Mapping[str, int],
    *,
    counterfactual_variants: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    lines = [
        "---",
        "pretty_name: AxisGEO Readiness Text Tables",
        "license: other",
        "language:",
        "- en",
        "tags:",
        "- llm",
        "- annotation",
        "- decision-readiness",
        "configs:",
    ]
    for name, paths_by_split, default in configs:
        lines.append(f"- config_name: {name}")
        if default:
            lines.append("  default: true")
        lines.append("  data_files:")
        for split, paths in sorted(paths_by_split.items()):
            lines.append(f"  - split: {split}")
            lines.append("    path:")
            lines.extend(f'    - "{path}"' for path in paths)
    lines.extend(
        [
            "---",
            "",
            "# AxisGEO readiness text tables",
            "",
            "This immutable snapshot keeps two measurement systems separate while",
            "making their text tables available in one Hugging Face repository.",
            "",
            f"- `{counts['likert_prompts']:,}` redistributable source prompts and",
            f"  `{counts['likert_annotations']:,}` parser-valid readiness judgments.",
            f"- `{counts['generated_candidates']:,}` generated search-question",
            f"  candidates and `{counts['candidate_compliance_annotations']:,}`",
            "  independent compliance",
            f"  reviews, and `{counts['fully_compliant_prompts']:,}` globally selected",
            "  prompts from one independently audited checkpoint.",
            "",
            "`likert_graded_prompts` joins each judge annotation to its exact source",
            "prompt. The generated candidates are not Likert graded: their target and",
            "observed coordinates are embedding-based measurements, and their",
            "compliance reviews use a separate Boolean/1--5 rubric. Embeddings",
            "describe",
            "text; they",
            "do not define the randomized policy variable `B`.",
            "",
            "WildChat and every local-only source artifact are excluded by the",
            "verified Likert publisher. Raw caches and `restricted-local` NPZ",
            "embeddings are not",
            "part of this text-only dataset.",
            "Hash-derived `*_seed` fields are stored as base-10 strings to preserve",
            "their exact values beyond the signed 64-bit integer range.",
        ]
    )
    for config_name, variant in sorted((counterfactual_variants or {}).items()):
        lines.extend(
            [
                "",
                f"## Existing-candidate counterfactual: `{config_name}`",
                "",
                f"This configuration contains `{int(variant['selected_count']):,}`",
                "globally selected prompts recomputed over the existing candidate",
                "union. It did not generate or embed new prompt text.",
                "",
                f"- Acceptance contract: `{variant['acceptance_contract_version']}`.",
                f"- Frozen dual-view tolerance: `{float(variant['distance_tolerance']):.3f}`.",
                f"- Historical selected count: `{int(variant['historical_selected_count']):,}`.",
                f"- Remaining target cells: `{int(variant['missing_count']):,}`.",
                "- Source texts retain their historical `question-v1` generation contract.",
                "",
                "This is a versioned prompt-space counterfactual, not a replacement",
                "for the historical Gold selection and not a definition of randomized",
                "policy variable `B`.",
            ]
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_text_dataset(
    *,
    likert_dataset_root: str | Path,
    prompt_population_root: str | Path,
    output_dir: str | Path,
    rows_per_shard: int = 100_000,
    git_commit_sha: str | None = None,
) -> dict[str, object]:
    likert_root = Path(likert_dataset_root).resolve()
    population_root = Path(prompt_population_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite text dataset: {output}")
    _verify_likert_dataset(likert_root)
    audit = audit_fully_compliant_prompts(population_root)
    if not audit["audit_passed"] or not audit["ready_to_export_count"]:
        raise ValueError("prompt population is not independently ready to export")
    candidate_paths, validation_path, selected_path = _population_sources(
        population_root
    )

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        configs, counts = _copy_likert_configs(likert_root, temporary)
        graded_paths, graded_count = _write_streaming_parquet(
            output=temporary,
            config_name="likert_graded_prompts",
            rows=_likert_graded_rows(likert_root),
            rows_per_shard=rows_per_shard,
        )
        configs.append(("likert_graded_prompts", graded_paths, False))
        counts["likert_graded_prompts"] = graded_count

        candidate_ids: set[str] = set()
        candidate_paths_out, candidate_count = _write_streaming_parquet(
            output=temporary,
            config_name="generated_candidates",
            rows=_deduplicated_rows(
                candidate_paths,
                identity_field="candidate_id",
                seen=candidate_ids,
            ),
            rows_per_shard=rows_per_shard,
        )
        configs.append(("generated_candidates", candidate_paths_out, False))
        counts["generated_candidates"] = candidate_count

        validation_ids: set[str] = set()
        validation_paths_out, validation_count = _write_streaming_parquet(
            output=temporary,
            config_name="candidate_compliance_annotations",
            rows=_deduplicated_rows(
                [validation_path],
                identity_field="candidate_id",
                seen=validation_ids,
            ),
            rows_per_shard=rows_per_shard,
        )
        if validation_ids != candidate_ids:
            raise ValueError("validation does not cover the exact candidate population")
        configs.append(
            ("candidate_compliance_annotations", validation_paths_out, False)
        )
        counts["candidate_compliance_annotations"] = validation_count

        selected_ids: set[str] = set()
        selected_paths_out, selected_count = _write_streaming_parquet(
            output=temporary,
            config_name="fully_compliant_prompts",
            rows=_deduplicated_rows(
                [selected_path],
                identity_field="candidate_id",
                seen=selected_ids,
            ),
            rows_per_shard=rows_per_shard,
        )
        if not selected_ids.issubset(candidate_ids):
            raise ValueError("selected prompt is absent from generated candidates")
        if selected_count != int(audit["ready_to_export_count"]):
            raise ValueError("selected count differs from independent audit")
        configs.append(("fully_compliant_prompts", selected_paths_out, False))
        counts["fully_compliant_prompts"] = selected_count

        _write_dataset_card(temporary, configs, counts)
        likert_manifest = _read_json(likert_root / "dataset_manifest.json")
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at": _now(),
            "git_commit_sha": git_commit_sha or _git_commit_sha(),
            "publication_safe": True,
            "text_only": True,
            "generated_candidates_are_likert_graded": False,
            "field_encodings": {
                "*_seed": "base-10 string preserving arbitrary-size integer",
            },
            "likert_source_manifest_sha256": _sha256(
                likert_root / "dataset_manifest.json"
            ),
            "restricted_sources_excluded": likert_manifest[
                "restricted_sources_excluded"
            ],
            "restricted_prompt_count_excluded": likert_manifest[
                "restricted_prompt_count_excluded"
            ],
            "prompt_population_audit": {
                "artifact_kind": audit["artifact_kind"],
                "audit_passed": audit["audit_passed"],
                "fully_compliant_prompt_count": audit[
                    "fully_compliant_prompt_count"
                ],
                "complete_30330_population_passed": audit[
                    "complete_30330_population_passed"
                ],
                "distance_tolerance": audit["distance_tolerance"],
            },
            "source_artifacts": {
                "candidate_files": [
                    _source_identity(path) for path in candidate_paths
                ],
                "validation": _source_identity(validation_path),
                "selection": _source_identity(selected_path),
            },
            "table_counts": dict(sorted(counts.items())),
            "excluded_artifacts": [
                "restricted-local prompt rows",
                "raw model caches",
                "NPZ embedding arrays",
                "section-local provisional selections",
            ],
        }
        _atomic_json(temporary / "dataset_manifest.json", manifest)
        _atomic_json(temporary / "checksums.json", _dataset_checksums(temporary))
        verify_text_dataset(temporary)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _read_json(output / "dataset_manifest.json")


def _discover_dataset_configs(
    root: Path,
) -> list[tuple[str, dict[str, list[str]], bool]]:
    configs = []
    for directory in sorted((root / "data").iterdir()):
        if not directory.is_dir():
            continue
        paths_by_split: dict[str, list[str]] = {}
        for path in sorted(directory.glob("*.parquet")):
            split = path.stem.rsplit("-", 1)[0]
            paths_by_split.setdefault(split, []).append(
                str(path.relative_to(root))
            )
        if paths_by_split:
            configs.append(
                (directory.name, paths_by_split, directory.name == "likert_prompts")
            )
    return configs


def _parquet_rows_for_ids(
    directory: Path,
    candidate_ids: set[str],
    columns: Sequence[str],
) -> dict[str, dict[str, object]]:
    _, pq = _pyarrow()
    rows: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.parquet")):
        for row in pq.read_table(path, columns=list(columns)).to_pylist():
            candidate_id = str(row["candidate_id"])
            if candidate_id not in candidate_ids:
                continue
            if candidate_id in rows:
                raise ValueError(
                    f"duplicate candidate_id in {directory.name}: {candidate_id}"
                )
            rows[candidate_id] = dict(row)
    missing = candidate_ids - set(rows)
    if missing:
        raise ValueError(
            f"{directory.name} omits selected candidate: {min(missing)}"
        )
    return rows


def _counterfactual_selection(
    dataset_root: Path,
    counterfactual_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    summary_path = counterfactual_root / "counterfactual_summary.json"
    scenario_root = counterfactual_root / "scenarios" / COUNTERFACTUAL_SCENARIO
    scenario_summary_path = scenario_root / "summary.json"
    selected_path = scenario_root / "selected.jsonl"
    summary = _read_json(summary_path)
    scenario_summary = _read_json(scenario_summary_path)
    if summary.get("format_version") != "readiness-search-trigger-counterfactual-v1":
        raise ValueError("unexpected counterfactual audit format")
    stored_scenario = summary.get("scenarios", {}).get(COUNTERFACTUAL_SCENARIO)
    if stored_scenario != scenario_summary:
        raise ValueError("counterfactual scenario summary identity differs")
    tolerance = float(scenario_summary.get("distance_tolerance", -1.0))
    if not math.isclose(tolerance, COUNTERFACTUAL_TOLERANCE, abs_tol=1e-12):
        raise ValueError("counterfactual selection does not use tolerance 0.035")
    if scenario_summary.get("require_template_uniqueness") is not True:
        raise ValueError("counterfactual selection omits template uniqueness")
    diagnostics = scenario_summary.get("selection_diagnostics", {})
    required_checks = (
        "require_both_views_within_tolerance",
        "require_delexicalized_template_uniqueness",
        "selected_delexicalized_templates_are_unique",
    )
    if any(diagnostics.get(key) is not True for key in required_checks):
        raise ValueError("counterfactual selection diagnostics fail a global contract")

    dataset_manifest = _read_json(dataset_root / "dataset_manifest.json")
    candidate_count = int(dataset_manifest["table_counts"]["generated_candidates"])
    if int(summary.get("candidate_count", -1)) != candidate_count:
        raise ValueError("counterfactual candidate population differs from dataset")
    selected = list(_iter_jsonl((selected_path,)))
    selected_count = int(scenario_summary.get("selected_count", -1))
    if not selected or len(selected) != selected_count:
        raise ValueError("counterfactual selected row count differs")
    if int(diagnostics.get("selected_count", -1)) != selected_count:
        raise ValueError("counterfactual selection diagnostics count differs")
    if int(diagnostics.get("verified_selected_count", -1)) != selected_count:
        raise ValueError("counterfactual selection is not fully dual-view verified")

    selected_ids: set[str] = set()
    target_pairs: set[tuple[str, str]] = set()
    templates: set[str] = set()
    for row in selected:
        candidate_id = str(row.get("candidate_id", ""))
        keyword_id = str(row.get("keyword_id", ""))
        target_id = str(row.get("target_id", ""))
        keyword = str(row.get("keyword", ""))
        question = str(row.get("question", ""))
        if not candidate_id or candidate_id in selected_ids:
            raise ValueError("counterfactual candidate ids are not unique")
        selected_ids.add(candidate_id)
        target_pair = (keyword_id, target_id)
        if not all(target_pair) or target_pair in target_pairs:
            raise ValueError("counterfactual target assignments are not unique")
        target_pairs.add(target_pair)
        if row.get("both_views_within_tolerance") is not True:
            raise ValueError("counterfactual row fails the dual-view contract")
        for field in (
            "reference_target_distance",
            "candidate_aligned_target_distance",
        ):
            if float(row.get(field, math.inf)) > tolerance:
                raise ValueError(f"counterfactual row exceeds {field}")
        template = delexicalize_question(
            question,
            keyword,
            require_keyword=False,
        )
        if template in templates:
            raise ValueError("counterfactual selected templates are not unique")
        templates.add(template)

    candidate_rows = _parquet_rows_for_ids(
        dataset_root / "data/generated_candidates",
        selected_ids,
        ("candidate_id", "keyword_id", "keyword", "question"),
    )
    review_rows = _parquet_rows_for_ids(
        dataset_root / "data/candidate_compliance_annotations",
        selected_ids,
        (
            "candidate_id",
            "topic_relevant",
            "search_intent",
            "web_answerable",
            "natural_language",
            "relevance_score_1_5",
        ),
    )
    for row in selected:
        candidate_id = str(row["candidate_id"])
        candidate = candidate_rows[candidate_id]
        for field in ("keyword_id", "keyword", "question"):
            if str(row[field]) != str(candidate[field]):
                raise ValueError(
                    f"counterfactual {field} differs from candidate: {candidate_id}"
                )
        if not search_review_passes_contract(
            review_rows[candidate_id],
            contract=COUNTERFACTUAL_CONTRACT,
        ):
            raise ValueError(
                f"counterfactual candidate fails search-trigger-v2: {candidate_id}"
            )

    historical = summary.get("scenarios", {}).get("question_v1_historical", {})
    target_count = selected_count + int(scenario_summary.get("missing_count", -1))
    metadata = {
        "scenario": COUNTERFACTUAL_SCENARIO,
        "acceptance_contract_version": COUNTERFACTUAL_CONTRACT,
        "source_text_contract": "question-v1",
        "distance_tolerance": tolerance,
        "selected_count": selected_count,
        "missing_count": int(scenario_summary["missing_count"]),
        "target_count": target_count,
        "historical_selected_count": int(historical.get("selected_count", -1)),
        "incremental_selected_count": selected_count
        - int(historical.get("selected_count", -1)),
        "existing_candidates_only": True,
        "new_generation_performed": False,
        "validation_recovered_count": int(summary["validation_recovered_count"]),
        "source_counterfactual_summary": _source_identity(summary_path),
        "source_scenario_summary": _source_identity(scenario_summary_path),
        "source_selection": _source_identity(selected_path),
    }
    return selected, metadata


def annotate_counterfactual_variant(
    *,
    dataset_dir: str | Path,
    counterfactual_root: str | Path,
    output_dir: str | Path,
    rows_per_shard: int = 100_000,
    git_commit_sha: str | None = None,
) -> dict[str, object]:
    source = Path(dataset_dir).resolve()
    counterfactual = Path(counterfactual_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite text dataset: {output}")
    verify_text_dataset(source)
    selected, variant = _counterfactual_selection(source, counterfactual)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    source_manifest_path = source / "dataset_manifest.json"
    try:
        shutil.copytree(source, temporary)
        annotated_rows = []
        for source_row in selected:
            row = dict(source_row)
            row.update(
                {
                    "acceptance_contract_version": COUNTERFACTUAL_CONTRACT,
                    "source_text_contract": "question-v1",
                    "selection_scenario": COUNTERFACTUAL_SCENARIO,
                    "selection_distance_tolerance": COUNTERFACTUAL_TOLERANCE,
                    "existing_candidates_only": True,
                    "new_generation_performed": False,
                }
            )
            annotated_rows.append(row)
        paths, count = _write_streaming_parquet(
            output=temporary,
            config_name=COUNTERFACTUAL_CONFIG,
            rows=(_sanitize_model_fields(row) for row in annotated_rows),
            rows_per_shard=rows_per_shard,
        )
        if count != int(variant["selected_count"]):
            raise ValueError("annotated counterfactual count differs")
        manifest = _read_json(temporary / "dataset_manifest.json")
        manifest["format_version"] = COUNTERFACTUAL_FORMAT_VERSION
        manifest["created_at"] = _now()
        manifest["git_commit_sha"] = git_commit_sha or _git_commit_sha()
        manifest["derived_from_dataset_manifest"] = _source_identity(
            source_manifest_path
        )
        manifest["counterfactual_prompt_variants"] = {
            COUNTERFACTUAL_CONFIG: variant
        }
        manifest["table_counts"][COUNTERFACTUAL_CONFIG] = count
        configs = _discover_dataset_configs(temporary)
        if not any(name == COUNTERFACTUAL_CONFIG for name, _, _ in configs):
            raise ValueError("annotated counterfactual configuration is missing")
        _write_dataset_card(
            temporary,
            configs,
            manifest["table_counts"],
            counterfactual_variants=manifest["counterfactual_prompt_variants"],
        )
        _atomic_json(temporary / "dataset_manifest.json", manifest)
        _atomic_json(temporary / "checksums.json", _dataset_checksums(temporary))
        verify_text_dataset(temporary)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _read_json(output / "dataset_manifest.json")


def _parquet_count(directory: Path) -> int:
    _, pq = _pyarrow()
    return sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in sorted(directory.glob("*.parquet"))
    )


def _parquet_id_set(directory: Path, field: str) -> set[str]:
    _, pq = _pyarrow()
    values: set[str] = set()
    count = 0
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path, columns=[field])
        for value in table[field].to_pylist():
            count += 1
            values.add(str(value))
    if len(values) != count:
        raise ValueError(f"duplicate {field} in {directory.name}")
    return values


def _require_true_column(directory: Path, field: str) -> None:
    _, pq = _pyarrow()
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path, columns=[field])
        if any(value is not True for value in table[field].to_pylist()):
            raise ValueError(f"{directory.name} contains a false {field}")


def _require_exact_column(directory: Path, field: str, expected: object) -> None:
    _, pq = _pyarrow()
    for path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(path, columns=[field])
        if any(value != expected for value in table[field].to_pylist()):
            raise ValueError(
                f"{directory.name} contains an unexpected {field}"
            )


def verify_text_dataset(dataset_dir: str | Path) -> None:
    root = Path(dataset_dir).resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError("unexpected text dataset format")
    if manifest.get("publication_safe") is not True:
        raise ValueError("text dataset is not marked publication safe")
    if manifest.get("generated_candidates_are_likert_graded") is not False:
        raise ValueError("generated prompt measurement type is mislabeled")
    if int(manifest.get("restricted_prompt_count_excluded", 0)) <= 0:
        raise ValueError("restricted-source exclusion was not preserved")
    expected = _read_json(root / "checksums.json")
    if expected != _dataset_checksums(root):
        raise ValueError("text dataset checksum verification failed")
    forbidden = [
        path
        for path in root.rglob("*")
        if path.is_symlink()
        or (
            path.is_file()
            and (
                path.suffix in {".jsonl", ".npz"}
                or "restricted-local" in path.name
            )
        )
    ]
    if forbidden:
        raise ValueError(f"forbidden intermediate leaked into dataset: {forbidden[0]}")
    counts = manifest["table_counts"]
    for config_name, expected_count in counts.items():
        actual = _parquet_count(root / "data" / config_name)
        if actual != int(expected_count):
            raise ValueError(f"table count mismatch: {config_name}")
    candidates = _parquet_id_set(root / "data/generated_candidates", "candidate_id")
    validations = _parquet_id_set(
        root / "data/candidate_compliance_annotations", "candidate_id"
    )
    selected = _parquet_id_set(
        root / "data/fully_compliant_prompts", "candidate_id"
    )
    if validations != candidates:
        raise ValueError("candidate validation identities differ")
    if not selected.issubset(candidates):
        raise ValueError("selected prompt identities differ")
    _require_true_column(
        root / "data/fully_compliant_prompts",
        "both_views_within_tolerance",
    )
    variants = manifest.get("counterfactual_prompt_variants", {})
    if manifest.get("format_version") == COUNTERFACTUAL_FORMAT_VERSION:
        if not isinstance(variants, dict) or not variants:
            raise ValueError("annotated dataset omits counterfactual variants")
    elif variants:
        raise ValueError("v1 dataset unexpectedly declares counterfactual variants")
    for config_name, variant in variants.items():
        if config_name != COUNTERFACTUAL_CONFIG or not isinstance(variant, dict):
            raise ValueError("unsupported counterfactual prompt variant")
        directory = root / "data" / config_name
        variant_ids = _parquet_id_set(directory, "candidate_id")
        if not variant_ids.issubset(candidates):
            raise ValueError("counterfactual prompt identities differ")
        if len(variant_ids) != int(variant.get("selected_count", -1)):
            raise ValueError("counterfactual prompt count differs")
        if variant.get("acceptance_contract_version") != COUNTERFACTUAL_CONTRACT:
            raise ValueError("counterfactual acceptance contract differs")
        if variant.get("source_text_contract") != "question-v1":
            raise ValueError("counterfactual source text contract differs")
        if not math.isclose(
            float(variant.get("distance_tolerance", -1.0)),
            COUNTERFACTUAL_TOLERANCE,
            abs_tol=1e-12,
        ):
            raise ValueError("counterfactual distance tolerance differs")
        if variant.get("existing_candidates_only") is not True:
            raise ValueError("counterfactual is not existing-candidates-only")
        if variant.get("new_generation_performed") is not False:
            raise ValueError("counterfactual generation provenance differs")
        _require_true_column(directory, "both_views_within_tolerance")
        _require_true_column(directory, "existing_candidates_only")
        _require_exact_column(
            directory,
            "acceptance_contract_version",
            COUNTERFACTUAL_CONTRACT,
        )
        _require_exact_column(
            directory,
            "source_text_contract",
            "question-v1",
        )
        _require_exact_column(
            directory,
            "selection_scenario",
            COUNTERFACTUAL_SCENARIO,
        )
        _require_exact_column(
            directory,
            "selection_distance_tolerance",
            COUNTERFACTUAL_TOLERANCE,
        )
        variant_rows = _parquet_rows_for_ids(
            directory,
            variant_ids,
            (
                "candidate_id",
                "keyword_id",
                "keyword",
                "question",
                "reference_target_distance",
                "candidate_aligned_target_distance",
            ),
        )
        candidate_rows = _parquet_rows_for_ids(
            root / "data/generated_candidates",
            variant_ids,
            ("candidate_id", "keyword_id", "keyword", "question"),
        )
        review_rows = _parquet_rows_for_ids(
            root / "data/candidate_compliance_annotations",
            variant_ids,
            (
                "candidate_id",
                "topic_relevant",
                "search_intent",
                "web_answerable",
                "natural_language",
                "relevance_score_1_5",
            ),
        )
        templates = set()
        for candidate_id, row in variant_rows.items():
            candidate = candidate_rows[candidate_id]
            if any(
                str(row[field]) != str(candidate[field])
                for field in ("keyword_id", "keyword", "question")
            ):
                raise ValueError("counterfactual text differs from candidate")
            if max(
                float(row["reference_target_distance"]),
                float(row["candidate_aligned_target_distance"]),
            ) > COUNTERFACTUAL_TOLERANCE:
                raise ValueError("counterfactual row exceeds tolerance")
            if not search_review_passes_contract(
                review_rows[candidate_id],
                contract=COUNTERFACTUAL_CONTRACT,
            ):
                raise ValueError("counterfactual row fails independent review")
            template = delexicalize_question(
                str(row["question"]),
                str(row["keyword"]),
                require_keyword=False,
            )
            if template in templates:
                raise ValueError("counterfactual templates are not unique")
            templates.add(template)


def _git_commit_sha() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = _parser().parse_args()
    if args.command == "verify":
        verify_text_dataset(args.dataset_dir)
        print("READINESS TEXT HF DATASET VERIFICATION: PASS")
        return 0
    if args.command == "publish":
        verify_text_dataset(args.dataset_dir)
        return _publish_likert_dataset(args)
    if args.command == "annotate-counterfactual":
        manifest = annotate_counterfactual_variant(
            dataset_dir=args.dataset_dir,
            counterfactual_root=args.counterfactual_root,
            output_dir=args.output_dir,
            rows_per_shard=args.rows_per_shard,
            git_commit_sha=args.git_commit_sha,
        )
        print(f"output: {Path(args.output_dir).resolve()}")
        print(
            f"{COUNTERFACTUAL_CONFIG}="
            f"{manifest['table_counts'][COUNTERFACTUAL_CONFIG]}"
        )
        print(
            "READINESS TEXT HF COUNTERFACTUAL ANNOTATION: PASS "
            "(no upload performed)"
        )
        return 0
    manifest = finalize_text_dataset(
        likert_dataset_root=args.likert_dataset_root,
        prompt_population_root=args.prompt_population_root,
        output_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        git_commit_sha=args.git_commit_sha,
    )
    print(f"output: {Path(args.output_dir).resolve()}")
    for name, count in manifest["table_counts"].items():
        print(f"{name}={count}")
    print("READINESS TEXT HF DATASET FINALIZATION: PASS (no upload performed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
