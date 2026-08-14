#!/usr/bin/env python3
"""Download, validate, and freeze the Stage A surface-coverage corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.query_free_surface_corpus import (  # noqa: E402
    SURFACE_CORPUS_VERSION,
    build_surface_coverage_corpus,
    read_dolly_surface_prompts,
    read_hh_surface_prompts,
)


SOURCES = (
    {
        "source_id": "databricks-dolly-15k",
        "repository": "databricks/databricks-dolly-15k",
        "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "path": "databricks-dolly-15k.jsonl",
        "sha256": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        "license": "CC-BY-SA-3.0",
        "dataset_card": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
    },
    {
        "source_id": "anthropic-hh-helpful-base",
        "repository": "Anthropic/hh-rlhf",
        "revision": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
        "path": "helpful-base/train.jsonl.gz",
        "sha256": "518a5bf288456fc9f3b7c980c54116fba0c52f274d3f4d344675d83e4058f6f4",
        "license": "MIT",
        "dataset_card": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--master-seed", type=int, default=20260817)
    parser.add_argument("--maximum-per-source", type=int, default=2_000)
    parser.add_argument("--skip-download", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    cache = Path(args.cache_dir).resolve() if args.cache_dir else output / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    source_paths = {}
    source_manifests = []
    for source in SOURCES:
        destination = cache / Path(source["path"]).name
        if not destination.exists():
            if args.skip_download:
                raise SystemExit(f"missing cached source under --skip-download: {destination}")
            _download_source(source, destination)
        actual_hash = _sha256_file(destination)
        if actual_hash != source["sha256"]:
            raise SystemExit(
                f"source hash mismatch for {source['source_id']}: {actual_hash}"
            )
        source_paths[source["source_id"]] = destination
        source_manifests.append(
            {
                **source,
                "resolved_file": str(destination),
                "resolved_size_bytes": destination.stat().st_size,
                "verified_sha256": actual_hash,
            }
        )

    raw = (
        *read_dolly_surface_prompts(source_paths["databricks-dolly-15k"]),
        *read_hh_surface_prompts(source_paths["anthropic-hh-helpful-base"]),
    )
    records, diagnostics = build_surface_coverage_corpus(
        raw,
        master_seed=args.master_seed,
        maximum_per_source=args.maximum_per_source,
    )
    _atomic_jsonl(output / "surface_coverage_corpus.jsonl", map(asdict, records))
    _atomic_json(output / "surface_coverage_diagnostics.json", asdict(diagnostics))
    _atomic_json(
        output / "source_provenance.json",
        {
            "corpus_version": SURFACE_CORPUS_VERSION,
            "sources": source_manifests,
            "license_notice": (
                "Rows retain source_id and source_record_id. Redistribution must "
                "comply with each source license; no unified relicensing is asserted."
            ),
        },
    )
    _atomic_text(
        output / "README.md",
        """# Query-free surface-coverage corpus

This corpus is a nuisance/style reservoir for the decision-readiness Stage A
experiment. It is not an A1-labeled dataset. Do not train, fit, or validate the
semantic direction from naturally occurring source intent.

Use `surface_family_id` to select diverse development plans and wholly unseen
confirmation families. Use the structural fields, not source-text semantics,
to construct realization plans. `eligible_as_semantic_label` is always false.

The rows originate from pinned Databricks Dolly and Anthropic HH helpful-base
snapshots. See `source_provenance.json` for exact revisions, hashes, dataset
cards, and licenses. No unified relicensing is asserted. The raw snapshots are
local cache files and are excluded from Git.
""",
    )
    corpus_path = output / "surface_coverage_corpus.jsonl"
    _atomic_json(
        output / "run_manifest.json",
        {
            "artifact_version": SURFACE_CORPUS_VERSION,
            "git_commit_sha": _git_sha(),
            "generated_at": _now(),
            "master_seed": args.master_seed,
            "maximum_per_source": args.maximum_per_source,
            "source_provenance_file": "source_provenance.json",
            "corpus_file": corpus_path.name,
            "corpus_sha256": _sha256_file(corpus_path),
            "record_count": len(records),
            "development_count": diagnostics.development_count,
            "confirmation_count": diagnostics.confirmation_count,
            "assigned_a1_present": False,
            "semantic_labels_present": False,
            "intended_use": "surface-style-coverage-only",
            "scientific_result": False,
            "reranking_outcomes_observed": False,
            "environment": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
    )
    print(f"records: {len(records)}")
    print(f"development: {diagnostics.development_count}")
    print(f"confirmation: {diagnostics.confirmation_count}")
    print(f"output: {output}")
    return 0


def _download_source(source, destination: Path) -> None:
    url = (
        f"https://huggingface.co/datasets/{source['repository']}/resolve/"
        f"{source['revision']}/{source['path']}?download=true"
    )
    request = Request(url, headers={"User-Agent": "geodml-research-dataset/1.0"})
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if _sha256_file(temporary) != source["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded hash mismatch for {source['source_id']}")
    temporary.replace(destination)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
