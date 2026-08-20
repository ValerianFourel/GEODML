#!/usr/bin/env python3
"""Build, embed, fit, finalize, and explicitly publish the readiness HF dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    HUB_SCOPE,
    LOCAL_SCOPE,
    READINESS_HF_FORMAT_VERSION,
    assemble_readiness_export,
    atomic_json,
    embed_prompt_shards,
    read_json,
    read_jsonl,
    sha256_file,
)
from interpretability.pipeline.readiness_hf_subspace import (  # noqa: E402
    fit_readiness_hf_subspace,
)
from interpretability.pipeline.readiness_hf_subspace_comparison import (  # noqa: E402
    compare_readiness_hf_subspaces,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecGenPromptEmbedder,
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--corpus", required=True)
    assemble.add_argument("--tasks", required=True)
    assemble.add_argument("--codebook", required=True)
    assemble.add_argument("--queue-root", required=True)
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--transfer-spec")
    assemble.add_argument(
        "--expected-judge-slots",
        default=(
            "primary-frontier,replicate-frontier-a,replicate-frontier-b,"
            "replicate-frontier-c"
        ),
    )
    assemble.add_argument("--git-commit-sha")

    embed = commands.add_parser("embed")
    embed.add_argument("--prompts", required=True)
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--view-name", required=True)
    embed.add_argument("--backend", choices=("llm2vec", "llm2vec-gen"), required=True)
    embed.add_argument("--embedding-model", required=True)
    embed.add_argument("--embedding-model-id", required=True)
    embed.add_argument("--embedding-model-revision", required=True)
    embed.add_argument("--mntp-model")
    embed.add_argument("--mntp-model-id")
    embed.add_argument("--mntp-model-revision")
    embed.add_argument("--peft-model")
    embed.add_argument("--peft-model-id")
    embed.add_argument("--peft-model-revision")
    embed.add_argument("--pooling", default="mean")
    embed.add_argument("--batch-size", type=int, default=8)
    embed.add_argument("--max-length", type=int, default=512)
    embed.add_argument("--shard-size", type=int, default=512)
    embed.add_argument("--git-commit-sha")

    subspace = commands.add_parser("fit-subspace")
    subspace.add_argument("--prompts", required=True)
    subspace.add_argument("--annotations", required=True)
    subspace.add_argument("--embedding-dir", required=True)
    subspace.add_argument("--output-dir", required=True)
    subspace.add_argument(
        "--judge-slots",
        default="primary-frontier,replicate-frontier-a,replicate-frontier-b",
    )
    subspace.add_argument("--minimum-rating-judges", type=int, default=2)
    subspace.add_argument("--minimum-mean-confidence", type=float, default=0.60)
    subspace.add_argument("--maximum-global-mad", type=float, default=15.0)
    subspace.add_argument("--ridge-penalty", type=float, default=1.0)
    subspace.add_argument(
        "--compute-backend",
        choices=("numpy", "torch-cuda"),
        default="numpy",
    )
    subspace.add_argument("--bootstrap-replicates", type=int, default=500)
    subspace.add_argument("--bootstrap-seed", type=int, default=20260820)
    subspace.add_argument("--git-commit-sha")

    compare = commands.add_parser("compare-subspaces")
    compare.add_argument("--reference-dir", required=True)
    compare.add_argument("--candidate-dir", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--git-commit-sha")

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--bundle-root", required=True)
    finalize.add_argument("--embedding-dir", action="append", default=[])
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--parquet-rows-per-shard", type=int, default=10_000)
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


def main() -> int:
    args = _parser().parse_args()
    if args.command == "assemble":
        return _assemble(args)
    if args.command == "embed":
        return _embed(args)
    if args.command == "fit-subspace":
        return _fit_subspace(args)
    if args.command == "compare-subspaces":
        return _compare_subspaces(args)
    if args.command == "finalize":
        return _finalize(args)
    if args.command == "verify":
        _verify_dataset(Path(args.dataset_dir).resolve())
        print("HF DATASET VERIFICATION: PASS")
        return 0
    return _publish(args)


def _assemble(args) -> int:
    options = {}
    if args.transfer_spec:
        options["transfer_spec_path"] = args.transfer_spec
    manifest = assemble_readiness_export(
        corpus_path=args.corpus,
        tasks_path=args.tasks,
        codebook_path=args.codebook,
        queue_root=args.queue_root,
        output_dir=args.output_dir,
        expected_judge_slots=tuple(
            value.strip()
            for value in args.expected_judge_slots.split(",")
            if value.strip()
        ),
        git_commit_sha=args.git_commit_sha or _git_commit_sha(),
        **options,
    )
    print(f"output: {Path(args.output_dir).resolve()}")
    print(f"complete local prompts: {manifest['scopes'][LOCAL_SCOPE]['prompt_count']}")
    print(f"HF-safe prompts: {manifest['scopes'][HUB_SCOPE]['prompt_count']}")
    print(f"restricted prompts excluded from HF: {manifest['restricted_prompt_count']}")
    print("ASSEMBLY: PASS (no upload performed)")
    return 0


def _embed(args) -> int:
    if args.backend == "llm2vec-gen" and (args.mntp_model or args.peft_model):
        raise SystemExit("LLM2Vec-Gen does not accept MNTP or SimCSE adapters")
    if args.mntp_model and not args.mntp_model_revision:
        raise SystemExit("--mntp-model-revision is required with --mntp-model")
    if args.mntp_model and not args.mntp_model_id:
        raise SystemExit("--mntp-model-id is required with --mntp-model")
    if args.peft_model and not args.peft_model_revision:
        raise SystemExit("--peft-model-revision is required with --peft-model")
    if args.peft_model and not args.peft_model_id:
        raise SystemExit("--peft-model-id is required with --peft-model")
    view = {
        "view_name": args.view_name,
        "backend": args.backend,
        "embedding_model": str(Path(args.embedding_model).resolve()),
        "embedding_model_id": args.embedding_model_id,
        "embedding_model_revision": args.embedding_model_revision,
        "mntp_model": str(Path(args.mntp_model).resolve()) if args.mntp_model else None,
        "mntp_model_id": args.mntp_model_id,
        "mntp_model_revision": args.mntp_model_revision,
        "peft_model": str(Path(args.peft_model).resolve()) if args.peft_model else None,
        "peft_model_id": args.peft_model_id,
        "peft_model_revision": args.peft_model_revision,
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dtype": "float32-export-from-bfloat16-inference",
        "git_commit_sha": args.git_commit_sha or _git_commit_sha(),
    }

    def factory(frozen_view):
        if frozen_view["backend"] == "llm2vec":
            return LLM2VecPromptEmbedder(
                str(frozen_view["embedding_model"]),
                mntp_model_name_or_path=frozen_view.get("mntp_model"),
                peft_model_name_or_path=frozen_view.get("peft_model"),
                batch_size=int(frozen_view["batch_size"]),
                max_length=int(frozen_view["max_length"]),
            )
        return LLM2VecGenPromptEmbedder(
            str(frozen_view["embedding_model"]),
            batch_size=int(frozen_view["batch_size"]),
            max_length=int(frozen_view["max_length"]),
        )

    manifest = embed_prompt_shards(
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        view=view,
        shard_size=args.shard_size,
        embedder_factory=factory,
    )
    print(f"output: {Path(args.output_dir).resolve()}")
    print(
        f"view={args.view_name} items={manifest['completed_item_count']}/"
        f"{manifest['item_count']} dimension={manifest['embedding_dimension']}"
    )
    print("EMBEDDING VIEW: PASS")
    return 0


def _fit_subspace(args) -> int:
    manifest = fit_readiness_hf_subspace(
        prompts_path=args.prompts,
        annotations_path=args.annotations,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        judge_slots=tuple(
            value.strip() for value in args.judge_slots.split(",") if value.strip()
        ),
        git_commit_sha=args.git_commit_sha or _git_commit_sha(),
        ridge_penalty=args.ridge_penalty,
        minimum_rating_judges=args.minimum_rating_judges,
        minimum_mean_confidence=args.minimum_mean_confidence,
        maximum_global_mad=args.maximum_global_mad,
        compute_backend=args.compute_backend,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        progress=lambda message: print(f"[fit-subspace] {message}", flush=True),
    )
    print(f"output: {Path(args.output_dir).resolve()}")
    print(f"map_id={manifest['map_id']}")
    print(
        f"training={manifest['usable_development_count']} "
        f"confirmation={manifest['usable_confirmation_count']} "
        f"dimension={manifest['embedding_dimension']}"
    )
    print(
        "evidence_assessment="
        f"{manifest['evidence_assessment']['status']} "
        f"checks={manifest['evidence_assessment']['passed_check_count']}/"
        f"{manifest['evidence_assessment']['total_check_count']}"
    )
    print("READINESS SUBSPACE: PASS")
    return 0


def _compare_subspaces(args) -> int:
    manifest = compare_readiness_hf_subspaces(
        reference_dir=args.reference_dir,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        git_commit_sha=args.git_commit_sha or _git_commit_sha(),
    )
    print(f"output: {Path(args.output_dir).resolve()}")
    print(f"reference_map_id={manifest['reference_map_id']}")
    print(f"candidate_map_id={manifest['candidate_map_id']}")
    print("READINESS SUBSPACE COMPARISON: PASS")
    return 0


def _finalize(args) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "finalization requires pyarrow (install analysis/requirements.txt)"
        ) from exc

    bundle_root = Path(args.bundle_root).resolve()
    assembly = read_json(bundle_root / "assembly_manifest.json")
    if assembly.get("format_version") != READINESS_HF_FORMAT_VERSION:
        raise SystemExit("unexpected readiness bundle format")
    scope = assembly.get("scopes", {}).get(HUB_SCOPE, {})
    if not scope.get("redistributable"):
        raise SystemExit("the selected bundle is not marked redistributable")
    safe_root = bundle_root / HUB_SCOPE
    prompts = read_jsonl(safe_root / "prompts.jsonl")
    annotations = read_jsonl(safe_root / "annotations.jsonl")
    failures = read_jsonl(safe_root / "failures.jsonl")
    missing = read_jsonl(safe_root / "missing_tasks.jsonl")
    if len(prompts) != int(scope["prompt_count"]):
        raise SystemExit("HF-safe prompt count changed after assembly")
    if {row["source_name"] for row in prompts} & set(assembly["restricted_sources"]):
        raise SystemExit("restricted source leaked into HF-safe prompts")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite finalized dataset: {output}")
    output.mkdir(parents=True)
    prompt_by_id = {str(row["item_id"]): row for row in prompts}
    configs = []
    table_counts = {}
    source_catalog = _source_catalog(prompts)
    for name, rows in (
        ("sources", source_catalog),
        ("prompts", prompts),
        ("annotations", annotations),
        ("failures", failures),
        ("missing_tasks", missing),
    ):
        if not rows:
            continue
        enriched = []
        for row in rows:
            if name in {"sources", "prompts"}:
                enriched.append(row)
            else:
                item_id = str(row["item_id"])
                if item_id not in prompt_by_id:
                    raise SystemExit(f"table {name} references excluded item: {item_id}")
                public_row = dict(row)
                if public_row.get("model"):
                    public_row["model"] = _public_model_reference(
                        str(public_row["model"]),
                        str(public_row.get("model_revision", "")),
                    )
                enriched.append(
                    {"split": prompt_by_id[item_id]["split"], **public_row}
                )
        paths = _write_partitioned_parquet(
            output=output,
            config_name=name,
            rows=enriched,
            rows_per_shard=args.parquet_rows_per_shard,
            pa=pa,
            pq=pq,
        )
        configs.append((name, paths, name == "prompts"))
        table_counts[name] = len(enriched)

    safe_ids = set(prompt_by_id)
    embedding_manifests = {}
    seen_view_names = set()
    for embedding_argument in args.embedding_dir:
        embedding_root = Path(embedding_argument).resolve()
        embedding_manifest = read_json(embedding_root / "embedding_manifest.json")
        if not embedding_manifest.get("is_complete"):
            raise SystemExit(f"embedding view is incomplete: {embedding_root}")
        view = embedding_manifest["view"]
        view_name = str(view["view_name"])
        config_name = f"embeddings-{view_name}"
        if view_name in seen_view_names:
            raise SystemExit(f"duplicate embedding view: {view_name}")
        seen_view_names.add(view_name)
        paths, embedded_safe_count = _write_embedding_parquet_shards(
            output=output,
            config_name=config_name,
            embedding_root=embedding_root,
            embedding_manifest=embedding_manifest,
            prompt_by_id=prompt_by_id,
            pa=pa,
            pq=pq,
        )
        configs.append((config_name, paths, False))
        table_counts[config_name] = embedded_safe_count
        embedding_manifests[view_name] = _public_embedding_manifest(
            embedding_manifest
        )

    _write_dataset_card(
        output,
        configs=configs,
        prompt_count=len(prompts),
        annotation_count=len(annotations),
        source_catalog=source_catalog,
        embedding_views=embedding_manifests,
    )
    manifest = {
        "format_version": READINESS_HF_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_sha": args.git_commit_sha or _git_commit_sha(),
        "source_input_hashes": {
            name: identity["sha256"]
            for name, identity in assembly["inputs"].items()
            if isinstance(identity, dict) and "sha256" in identity
        },
        "safe_scope_input_hashes": {
            name: identity["sha256"]
            for name, identity in scope["artifacts"].items()
        },
        "restricted_sources_excluded": assembly["restricted_sources"],
        "restricted_prompt_count_excluded": assembly["restricted_prompt_count"],
        "included_prompt_count": len(prompts),
        "included_source_counts": scope["source_counts"],
        "table_counts": table_counts,
        "embedding_views": embedding_manifests,
        "publication_safe": True,
    }
    atomic_json(output / "dataset_manifest.json", manifest)
    checksums = _dataset_checksums(output)
    atomic_json(output / "checksums.json", checksums)
    _verify_dataset(output)
    print(f"output: {output}")
    print(f"HF-safe prompts: {len(prompts)}")
    print(f"valid annotations: {len(annotations)}")
    print(f"embedding views: {len(embedding_manifests)}")
    print("FINALIZATION: PASS (no upload performed)")
    return 0


def _write_partitioned_parquet(
    *,
    output: Path,
    config_name: str,
    rows: Sequence[Mapping[str, object]],
    rows_per_shard: int,
    pa,
    pq,
) -> dict[str, list[str]]:
    if rows_per_shard <= 0:
        raise SystemExit("--parquet-rows-per-shard must be positive")
    paths: dict[str, list[str]] = {}
    by_split = {}
    for row in rows:
        split = str(row.get("split", "data"))
        by_split.setdefault(split, []).append(dict(row))
    for split, split_rows in sorted(by_split.items()):
        split_paths = []
        for shard_index, start in enumerate(range(0, len(split_rows), rows_per_shard)):
            part = split_rows[start : start + rows_per_shard]
            path = output / "data" / config_name / f"{split}-{shard_index:05d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            table = pa.Table.from_pylist(part)
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(path)
            split_paths.append(str(path.relative_to(output)))
        paths[split] = split_paths
    return paths


def _write_embedding_parquet_shards(
    *,
    output: Path,
    config_name: str,
    embedding_root: Path,
    embedding_manifest: Mapping[str, object],
    prompt_by_id: Mapping[str, Mapping[str, object]],
    pa,
    pq,
) -> tuple[dict[str, list[str]], int]:
    expected_dimension = int(embedding_manifest["embedding_dimension"])
    expected_full_count = int(embedding_manifest["item_count"])
    safe_ids = set(prompt_by_id)
    seen_full = set()
    seen_safe = set()
    paths: dict[str, list[str]] = {}
    split_indices = {}
    for shard in embedding_manifest.get("shards", ()):
        source = embedding_root / str(shard["path"])
        if sha256_file(source) != shard["sha256"]:
            raise SystemExit(f"embedding shard checksum mismatch: {source}")
        with np.load(source, allow_pickle=False) as payload:
            item_ids = [str(value) for value in payload["item_ids"]]
            text_hashes = [str(value) for value in payload["text_sha256s"]]
            matrix = np.asarray(payload["embeddings"], dtype=np.float32)
        if matrix.shape != (len(item_ids), expected_dimension):
            raise SystemExit(f"embedding shard shape mismatch: {source}")
        by_split = {}
        for row_index, (item_id, text_hash) in enumerate(zip(item_ids, text_hashes)):
            if item_id in seen_full:
                raise SystemExit(f"duplicate embedding item ID: {item_id}")
            seen_full.add(item_id)
            if item_id not in safe_ids:
                continue
            if text_hash != prompt_by_id[item_id]["text_sha256"]:
                raise SystemExit(f"embedding text hash mismatch: {item_id}")
            seen_safe.add(item_id)
            split = str(prompt_by_id[item_id]["split"])
            by_split.setdefault(split, []).append(row_index)
        for split, row_indices in sorted(by_split.items()):
            shard_index = split_indices.get(split, 0)
            split_indices[split] = shard_index + 1
            path = (
                output
                / "data"
                / config_name
                / f"{split}-{shard_index:05d}.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            selected_ids = [item_ids[index] for index in row_indices]
            selected_hashes = [text_hashes[index] for index in row_indices]
            selected_matrix = matrix[row_indices]
            table = pa.Table.from_arrays(
                [
                    pa.array(selected_ids, type=pa.string()),
                    pa.array(selected_hashes, type=pa.string()),
                    pa.array(
                        selected_matrix.tolist(),
                        type=pa.list_(pa.float32(), expected_dimension),
                    ),
                ],
                names=("item_id", "text_sha256", "embedding"),
            )
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(path)
            paths.setdefault(split, []).append(str(path.relative_to(output)))
    if len(seen_full) != expected_full_count:
        raise SystemExit("embedding manifest/full item count mismatch")
    if seen_safe != safe_ids:
        raise SystemExit(
            f"embedding view misses {len(safe_ids - seen_safe)} HF-safe prompts"
        )
    return paths, len(seen_safe)


def _write_dataset_card(
    output: Path,
    *,
    configs: Sequence[tuple[str, Mapping[str, Sequence[str]], bool]],
    prompt_count: int,
    annotation_count: int,
    source_catalog: Sequence[Mapping[str, object]],
    embedding_views: Mapping[str, Mapping[str, object]],
) -> None:
    yaml = [
        "---",
        "pretty_name: GEODML Semantic Readiness Prompt Panel",
        "license: other",
        "language:",
        "- en",
        "tags:",
        "- llm",
        "- embeddings",
        "- annotation",
        "- decision-readiness",
        "configs:",
    ]
    for name, paths_by_split, default in configs:
        yaml.append(f"- config_name: {name}")
        if default:
            yaml.append("  default: true")
        yaml.append("  data_files:")
        for split, paths in sorted(paths_by_split.items()):
            yaml.append(f"  - split: {split}")
            yaml.append("    path:")
            for path in paths:
                yaml.append(f'    - "{path}"')
    yaml.append("---")
    sources = "\n".join(
        f"- [`{row['source_name']}`]({row['source_url']}): "
        f"{int(row['prompt_count']):,} prompts, `{row['license']}`"
        for row in source_catalog
    )
    views = (
        "\n".join(
            f"- `{name}`: {metadata['backend']}, dimension "
            f"{metadata['embedding_dimension']}, revision "
            f"`{metadata['embedding_model_revision']}`"
            for name, metadata in sorted(embedding_views.items())
        )
        or "- No embedding view was included in this finalized revision."
    )
    body = f"""

# GEODML Semantic Readiness Prompt Panel

This repository contains {prompt_count:,} exact-unique prompts and
{annotation_count:,} parser-valid, independently generated LLM judgments using
the frozen `decision-readiness-ordinal-abstention-v2` rubric. The annotation
schema includes the Likert dimensions, continuous readiness, category,
ambiguity, confidence, `not_applicable`, and `dont_know` outcomes, plus the raw
model response and retry audit.

The tables are separate Hugging Face configurations: `prompts`, `annotations`,
failure/missing-task audits when present, and one configuration per embedding
view. Join them on `item_id`; annotations additionally use `task_id` and
`judge_slot`.

## Included sources

{sources}

Each row preserves its upstream license and provenance. There is no single
relicense for this mixed-source collection; users must comply with every
upstream source's terms and attribution requirements.

## Embedding views

{views}

## Exclusions and limitations

Rows from sources marked local-only by the frozen GEODML source policy are
excluded in their entirety, including prompt text, judge responses, failed
attempts, rationales, and embeddings. In particular, the WildChat portion of
the local research panel is not in this Hub artifact. The full local panel must
not be reconstructed from this repository.

These are model-generated measurements, not human ground truth. Missing and
failed judgments are retained as audit records and must not be silently treated
as negative labels. Prompt embeddings are measurements and do not define the
experiment's assigned policy variable.
"""
    (output / "README.md").write_text("\n".join(yaml) + body, encoding="utf-8")


def _source_catalog(
    prompts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    fallback_urls = {
        "databricks-dolly-15k": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "anthropic-hh-helpful-base": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
    }
    rows = []
    for source_name in sorted({str(row["source_name"]) for row in prompts}):
        source_rows = [row for row in prompts if row["source_name"] == source_name]
        licenses = sorted({str(row["license"]) for row in source_rows})
        if len(licenses) != 1:
            raise SystemExit(f"source mixes licenses: {source_name}")
        urls = sorted(
            {
                str(row["source_url"])
                for row in source_rows
                if row.get("source_url")
            }
        )
        source_url = fallback_urls.get(source_name)
        if source_url is None and source_name.startswith("stackexchange:"):
            site = source_name.split(":", 1)[1]
            source_url = f"https://{site}.stackexchange.com"
            if site == "stackoverflow":
                source_url = "https://stackoverflow.com"
            elif site == "superuser":
                source_url = "https://superuser.com"
            elif site == "askubuntu":
                source_url = "https://askubuntu.com"
        if source_url is None:
            source_url = urls[0] if urls else None
        if not source_url:
            raise SystemExit(f"source lacks attribution URL: {source_name}")
        rows.append(
            {
                "split": "data",
                "source_name": source_name,
                "license": licenses[0],
                "source_url": source_url,
                "prompt_count": len(source_rows),
            }
        )
    return rows


def _public_embedding_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    view = dict(manifest["view"])
    revisions = {
        "embedding_model": ("embedding_model_id", "embedding_model_revision"),
        "mntp_model": ("mntp_model_id", "mntp_model_revision"),
        "peft_model": ("peft_model_id", "peft_model_revision"),
    }
    for field, (id_field, revision_field) in revisions.items():
        value = view.get(field)
        if value:
            view[field] = str(
                view.get(id_field)
                or _public_model_reference(
                    str(value), str(view.get(revision_field, ""))
                )
            )
    return {
        **view,
        "embedding_dimension": int(manifest["embedding_dimension"]),
        "item_count_before_publication_filter": int(manifest["item_count"]),
        "view_config_sha256": manifest["view_config_sha256"],
    }


def _public_model_reference(value: str, revision: str) -> str:
    path = Path(value)
    if path.is_absolute():
        if revision and path.name == revision and len(path.parents) >= 2:
            return f"{path.parent.parent.name}/{path.parent.name}"
        return path.name
    return value


def _dataset_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def _verify_dataset(root: Path) -> None:
    manifest = read_json(root / "dataset_manifest.json")
    if not manifest.get("publication_safe"):
        raise SystemExit("dataset manifest is not publication-safe")
    if int(manifest.get("restricted_prompt_count_excluded", 0)) <= 0:
        raise SystemExit("publication guard did not exclude any restricted prompts")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for source in manifest.get("restricted_sources_excluded", ()):
        if source in readme and source != "allenai-wildchat-1m":
            raise SystemExit(f"restricted source name leaked into dataset card: {source}")
    expected = read_json(root / "checksums.json")
    actual = _dataset_checksums(root)
    if expected != actual:
        raise SystemExit("finalized dataset checksum verification failed")
    forbidden_suffixes = {".jsonl", ".npz", ".failed.json"}
    leaked = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and any(str(path).endswith(suffix) for suffix in forbidden_suffixes)
    ]
    if leaked:
        raise SystemExit(f"local intermediate leaked into finalized dataset: {leaked[0]}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise SystemExit("finalized dataset may not contain symlinks")


def _publish(args) -> int:
    dataset_dir = Path(args.dataset_dir).resolve()
    _verify_dataset(dataset_dir)
    if args.repo_id != args.confirm_repo_id:
        raise SystemExit("--confirm-repo-id must exactly match --repo-id")
    if args.public and not args.confirm_public:
        raise SystemExit("public upload requires --confirm-public")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required; authenticate without printing the token")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("publish requires huggingface_hub") from exc
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=[".cache/**", ".DS_Store"],
    )
    print(f"published: https://huggingface.co/datasets/{args.repo_id}")
    return 0


def _git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
