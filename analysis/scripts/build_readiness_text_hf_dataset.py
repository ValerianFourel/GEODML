#!/usr/bin/env python3
"""Build and verify one compact Hugging Face dataset of readiness text tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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


FORMAT_VERSION = "axisgeo-readiness-text-hf-v1"
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


def verify_text_dataset(dataset_dir: str | Path) -> None:
    root = Path(dataset_dir).resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION:
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
