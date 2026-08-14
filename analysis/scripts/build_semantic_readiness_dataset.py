#!/usr/bin/env python3
"""Collect web texts, merge natural corpora, and export blinded label tasks."""

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


def _manifest(**values):
    return {
        "artifact_version": SEMANTIC_DATASET_VERSION,
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
