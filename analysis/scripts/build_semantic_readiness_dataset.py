#!/usr/bin/env python3
"""Collect web texts, merge natural corpora, and export blinded label tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    LABEL_RUBRIC_VERSION,
    SEMANTIC_DATASET_VERSION,
    WebTextRecord,
    build_readiness_label_tasks,
    build_semantic_readiness_corpus,
    load_web_retrieval_specification,
    merge_web_records,
    parse_stackexchange_items,
)
from interpretability.pipeline.semantic_readiness_transfer import (  # noqa: E402
    DEFAULT_TRANSFER_SPEC,
    TRANSFER_PANEL_VERSION,
    TransferPromptRecord,
    build_transfer_prompt_panel,
    extend_semantic_readiness_corpus,
    load_transfer_source_specification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    collect = stages.add_parser("collect-web")
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--specification")
    collect.add_argument("--page-size", type=int, default=30)
    collect.add_argument("--request-delay-seconds", type=float, default=0.15)
    collect.add_argument(
        "--reuse-raw-responses",
        help="Reparse one complete prior raw_responses directory without network calls.",
    )

    merge = stages.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--surface-corpus", required=True)
    merge.add_argument("--web-records", required=True)

    transfer = stages.add_parser("collect-transfer")
    transfer.add_argument("--output-dir", required=True)
    transfer.add_argument("--source-specification")
    transfer.add_argument(
        "--source-input",
        action="append",
        required=True,
        metavar="SOURCE_ID=PATH",
        help="Local JSON/JSONL/JSONL.GZ/TSV/Parquet file or directory.",
    )
    transfer.add_argument(
        "--source-revision",
        action="append",
        required=True,
        metavar="SOURCE_ID=REVISION",
        help="Exact upstream commit or dataset revision for each source input.",
    )
    transfer.add_argument("--maximum-per-source", type=int, default=1_000)
    transfer.add_argument("--master-seed", type=int, default=20260817)

    extend = stages.add_parser("merge-transfer")
    extend.add_argument("--output-dir", required=True)
    extend.add_argument("--base-corpus", required=True)
    extend.add_argument("--transfer-records", required=True)
    extend.add_argument(
        "--additional-transfer-records",
        action="append",
        default=[],
        help="Additional disjoint transfer-record JSONL file; may be repeated.",
    )
    extend.add_argument("--source-specification")

    label = stages.add_parser("export-labeling")
    label.add_argument("--output-dir", required=True)
    label.add_argument("--corpus", required=True)
    label.add_argument("--judge-slots", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    if args.stage == "collect-web":
        _collect_web(args, output)
    elif args.stage == "merge":
        _merge(args, output)
    elif args.stage == "collect-transfer":
        _collect_transfer(args, output)
    elif args.stage == "merge-transfer":
        _merge_transfer(args, output)
    else:
        _export_labeling(args, output)
    print(f"output: {output}")
    return 0


def _collect_web(args, output: Path) -> None:
    specification = args.specification if args.specification else None
    probes = (
        load_web_retrieval_specification(specification)
        if specification
        else load_web_retrieval_specification()
    )
    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be in [1, 100]")
    raw_responses = output / "raw_responses"
    raw_responses.mkdir()
    reused_raw = (
        Path(args.reuse_raw_responses).resolve()
        if args.reuse_raw_responses
        else None
    )
    records = []
    quota_remaining = None
    for index, probe in enumerate(probes, 1):
        if reused_raw:
            source_path = reused_raw / f"{probe.probe_id}.json"
            if not source_path.exists():
                raise SystemExit(f"missing reused raw response: {source_path}")
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        else:
            parameters = {
                "site": probe.site,
                "q": probe.query,
                "pagesize": args.page_size,
                "order": "desc",
                "sort": "relevance",
            }
            url = "https://api.stackexchange.com/2.3/search/advanced?" + urlencode(
                parameters
            )
            request = Request(
                url,
                headers={"User-Agent": "geodml-semantic-readiness-research/1.0"},
            )
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        _atomic_json(raw_responses / f"{probe.probe_id}.json", payload)
        records.extend(parse_stackexchange_items(payload, probe))
        quota_remaining = payload.get("quota_remaining", quota_remaining)
        backoff = float(payload.get("backoff", 0.0))
        print(
            f"[{index}/{len(probes)}] {probe.site} {probe.query!r}: "
            f"{len(payload.get('items', ()))}",
            flush=True,
        )
        if index < len(probes) and not reused_raw:
            time.sleep(max(args.request_delay_seconds, backoff))
    merged = merge_web_records(records)
    _atomic_jsonl(output / "web_text_records.jsonl", map(asdict, merged))
    _atomic_json(
        output / "retrieval_specification.json",
        {
            "specification_version": SEMANTIC_DATASET_VERSION,
            "probes": [asdict(item) for item in probes],
        },
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            stage="collect-web",
            source_api="Stack Exchange API v2.3",
            source_license="per-record Stack Exchange API content_license",
            probe_count=len(probes),
            unique_web_record_count=len(merged),
            page_size=args.page_size,
            quota_remaining=quota_remaining,
            retrieval_regions_are_labels=False,
            reused_raw_responses=str(reused_raw) if reused_raw else None,
            scientific_result=False,
        ),
    )


def _merge(args, output: Path) -> None:
    surface_path = Path(args.surface_corpus).resolve()
    web_path = Path(args.web_records).resolve()
    surface_rows = _read_jsonl(surface_path)
    web_rows = []
    for row in _read_jsonl(web_path):
        for key in (
            "tags",
            "retrieval_probe_ids",
            "retrieval_sampling_regions",
        ):
            row[key] = tuple(row.get(key, ()))
        web_rows.append(WebTextRecord(**row))
    corpus = build_semantic_readiness_corpus(surface_rows, web_rows)
    _atomic_jsonl(output / "semantic_readiness_corpus.jsonl", map(asdict, corpus))
    counts = {}
    for item in corpus:
        key = f"{item.source_name}|{item.split}"
        counts[key] = counts.get(key, 0) + 1
    corpus_path = output / "semantic_readiness_corpus.jsonl"
    included_web_count = sum(
        item.source_kind == "public-web-question-title" for item in corpus
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            stage="merge",
            surface_corpus=str(surface_path),
            surface_corpus_sha256=_sha256_file(surface_path),
            web_records=str(web_path),
            web_records_sha256=_sha256_file(web_path),
            corpus_count=len(corpus),
            web_record_count=len(web_rows),
            web_included_count=included_web_count,
            web_excluded_count=len(web_rows) - included_web_count,
            corpus_sha256=_sha256_file(corpus_path),
            counts_by_source_and_split=counts,
            semantic_labels_present=False,
            scientific_result=False,
        ),
    )


def _collect_transfer(args, output: Path) -> None:
    specification_path = Path(
        args.source_specification or DEFAULT_TRANSFER_SPEC
    ).resolve()
    sources = load_transfer_source_specification(specification_path)
    source_inputs = _key_value_arguments(args.source_input, label="source input")
    source_revisions = _key_value_arguments(
        args.source_revision,
        label="source revision",
    )
    if set(source_inputs) != set(source_revisions):
        raise SystemExit(
            "--source-input and --source-revision must name the same source IDs"
        )
    source_ids = {item.source_id for item in sources}
    unknown = sorted(set(source_inputs) - source_ids)
    if unknown:
        raise SystemExit(f"source inputs not present in specification: {unknown}")
    resolved_inputs = {
        source_id: Path(path).resolve()
        for source_id, path in source_inputs.items()
    }
    missing_paths = sorted(
        str(path) for path in resolved_inputs.values() if not path.exists()
    )
    if missing_paths:
        raise SystemExit(f"missing source inputs: {missing_paths}")
    rows_by_source = {
        source_id: _iter_source_rows(path)
        for source_id, path in resolved_inputs.items()
    }
    records, diagnostics = build_transfer_prompt_panel(
        rows_by_source,
        source_revisions=source_revisions,
        sources=sources,
        maximum_per_source=args.maximum_per_source,
        master_seed=args.master_seed,
    )
    record_path = output / "semantic_readiness_transfer_records.jsonl"
    _atomic_jsonl(record_path, map(asdict, records))
    _atomic_json(
        output / "transfer_source_diagnostics.json",
        {
            "transfer_panel_version": TRANSFER_PANEL_VERSION,
            "sources": [asdict(item) for item in diagnostics],
        },
    )
    source_by_id = {item.source_id: item for item in sources}
    input_audit = []
    for source_id, path in sorted(resolved_inputs.items()):
        source = source_by_id[source_id]
        input_audit.append(
            {
                **asdict(source),
                "source_revision": source_revisions[source_id],
                "local_input": str(path),
                "local_input_sha256": _sha256_source_input(path),
                "local_input_size_bytes": _source_input_size(path),
            }
        )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            artifact_version=TRANSFER_PANEL_VERSION,
            stage="collect-transfer",
            source_specification=str(specification_path),
            source_specification_sha256=_sha256_file(specification_path),
            master_seed=args.master_seed,
            maximum_per_source=args.maximum_per_source,
            included_source_ids=sorted(resolved_inputs),
            omitted_source_ids=sorted(source_ids - set(resolved_inputs)),
            source_inputs=input_audit,
            transfer_record_count=len(records),
            transfer_records_sha256=_sha256_file(record_path),
            sampling_roles_are_labels=False,
            sampling_roles_visible_to_judges=False,
            restricted_source_text_may_require_local_only_outputs=any(
                "local-only" in source_by_id[item.source_id].redistribution_policy
                for item in records
            ),
            scientific_result=False,
        ),
    )


def _merge_transfer(args, output: Path) -> None:
    from interpretability.pipeline.semantic_readiness_dataset import (
        SemanticReadinessItem,
    )

    specification_path = Path(
        args.source_specification or DEFAULT_TRANSFER_SPEC
    ).resolve()
    sources = load_transfer_source_specification(specification_path)
    base_path = Path(args.base_corpus).resolve()
    transfer_path = Path(args.transfer_records).resolve()
    additional_transfer_paths = tuple(
        Path(value).resolve() for value in args.additional_transfer_records
    )
    transfer_paths = (transfer_path, *additional_transfer_paths)
    base = tuple(
        SemanticReadinessItem(**row) for row in _read_jsonl(base_path)
    )
    records = tuple(
        TransferPromptRecord(**row)
        for path in transfer_paths
        for row in _read_jsonl(path)
    )
    source_by_id = {item.source_id: item for item in sources}
    included_source_ids = {item.source_id for item in records}
    restricted_source_ids = sorted(
        source_id
        for source_id in included_source_ids
        if "local-only" in source_by_id[source_id].redistribution_policy
    )
    transfer, expanded, diagnostics = extend_semantic_readiness_corpus(
        base,
        records,
        sources=sources,
    )
    transfer_corpus_path = output / "semantic_readiness_transfer_corpus.jsonl"
    expanded_corpus_path = output / "semantic_readiness_expanded_corpus.jsonl"
    _atomic_jsonl(transfer_corpus_path, map(asdict, transfer))
    _atomic_jsonl(expanded_corpus_path, map(asdict, expanded))
    counts = {}
    for item in transfer:
        key = f"{item.source_name}|{item.split}"
        counts[key] = counts.get(key, 0) + 1
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            artifact_version=TRANSFER_PANEL_VERSION,
            stage="merge-transfer",
            source_specification=str(specification_path),
            source_specification_sha256=_sha256_file(specification_path),
            base_corpus=str(base_path),
            base_corpus_sha256=_sha256_file(base_path),
            transfer_records=str(transfer_path),
            transfer_records_sha256=_sha256_file(transfer_path),
            additional_transfer_records=[
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for path in additional_transfer_paths
            ],
            transfer_corpus_sha256=_sha256_file(transfer_corpus_path),
            expanded_corpus_sha256=_sha256_file(expanded_corpus_path),
            counts_by_source_and_split=counts,
            included_source_ids=sorted(included_source_ids),
            restricted_source_ids=restricted_source_ids,
            redistribution_allowed=not restricted_source_ids,
            redistribution_notice=(
                "Artifacts containing restricted-source text must remain local "
                "and must not be uploaded to GitHub, Hugging Face, or APIs."
                if restricted_source_ids
                else "Source-specific license and attribution requirements still apply."
            ),
            frozen_base_is_exact_expanded_prefix=True,
            label_rubric_changed=False,
            **asdict(diagnostics),
            semantic_labels_present=False,
            scientific_result=False,
        ),
    )


def _export_labeling(args, output: Path) -> None:
    corpus_path = Path(args.corpus).resolve()
    from interpretability.pipeline.semantic_readiness_dataset import (
        SemanticReadinessItem,
    )

    corpus = tuple(SemanticReadinessItem(**row) for row in _read_jsonl(corpus_path))
    slots = tuple(value.strip() for value in args.judge_slots.split(",") if value.strip())
    tasks, codebook = build_readiness_label_tasks(corpus, judge_slots=slots)
    _atomic_jsonl(output / "readiness_label_tasks_blinded.jsonl", map(asdict, tasks))
    _atomic_jsonl(
        output / "readiness_label_codebook_private.jsonl",
        ({"task_id": task_id, **row} for task_id, row in sorted(codebook.items())),
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            stage="export-labeling",
            corpus=str(corpus_path),
            corpus_sha256=_sha256_file(corpus_path),
            corpus_count=len(corpus),
            judge_slots=slots,
            task_count=len(tasks),
            rubric_version=LABEL_RUBRIC_VERSION,
            source_metadata_visible_to_judges=False,
            retrieval_regions_visible_to_judges=False,
            scientific_result=False,
        ),
    )


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iter_source_rows(path: Path):
    if path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and _supported_source_file(item)
        )
        if not files:
            raise SystemExit(f"source input directory contains no supported files: {path}")
        for item in files:
            yield from _iter_source_rows(item)
        return
    suffixes = path.suffixes
    if suffixes[-2:] == [".jsonl", ".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return
    if path.suffix == ".jsonl":
        yield from _read_jsonl(path)
        return
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload, dict):
            for key in ("data", "dialogs", "dialogues", "conversations", "questions"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            yield row
                    return
            yield payload
            return
        raise SystemExit(f"unsupported JSON root in source input: {path}")
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                columns = line.rstrip("\n").split("\t", 1)
                if len(columns) == 2:
                    yield {"query_id": columns[0] or str(index), "query": columns[1]}
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("pyarrow is required to read Parquet source inputs") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4_096):
            yield from batch.to_pylist()
        return
    raise SystemExit(f"unsupported source input: {path}")


def _supported_source_file(path: Path) -> bool:
    return (
        path.suffix in {".json", ".jsonl", ".tsv", ".parquet"}
        or path.suffixes[-2:] == [".jsonl", ".gz"]
    )


def _key_value_arguments(values, *, label: str) -> dict[str, str]:
    parsed = {}
    for value in values:
        key, separator, raw = str(value).partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key or not raw:
            raise SystemExit(f"invalid {label}; expected SOURCE_ID=VALUE: {value!r}")
        if key in parsed:
            raise SystemExit(f"duplicate {label} for {key}")
        parsed[key] = raw
    return parsed


def _source_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    return tuple(
        sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and _supported_source_file(item)
        )
    )


def _sha256_source_input(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for item in _source_files(path):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_input_size(path: Path) -> int:
    return sum(item.stat().st_size for item in _source_files(path))


def _manifest(**values):
    return {
        "artifact_version": values.pop("artifact_version", SEMANTIC_DATASET_VERSION),
        "git_commit_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reranking_outcomes_observed": False,
        "environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        **values,
    }


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
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
