#!/usr/bin/env python3
"""Freeze full-population registration inputs and deterministic keyword order."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.scripts.select_readiness_axis_pilot import _normalize


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not values or any(not isinstance(row, dict) for row in values):
        raise ValueError(f"invalid JSONL objects: {path}")
    return values


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _eligible_ids(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = {row.get("prompt_id") for row in rows if isinstance(row, dict)}
    if not ids or None in ids or len(ids) != len(rows):
        raise ValueError("recovery eligible-prompt table has invalid prompt IDs")
    return ids


def prepare(
    selection_manifest: Path,
    output: Path,
    *,
    axis_bins: int = 20,
    eligible_prompts_table: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite population registration: {output}")
    if axis_bins < 2:
        raise ValueError("axis_bins must be at least two")
    manifest = json.loads(selection_manifest.read_text())
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise TypeError("selection manifest lacks sources")
    verified: dict[str, Path] = {}
    for name in ("prompts", "axis_map"):
        entry = sources.get(name)
        if not isinstance(entry, dict):
            raise TypeError(f"selection source is missing: {name}")
        source = (selection_manifest.parent / entry["path"]).resolve()
        if _sha256(source) != entry.get("sha256"):
            raise ValueError(f"selection source checksum mismatch: {name}")
        if type(entry.get("rows")) is not int:
            raise TypeError(f"selection source row count is invalid: {name}")
        verified[name] = source
    prompts = _rows(verified["prompts"])
    axes = _rows(verified["axis_map"])
    if len(prompts) != sources["prompts"]["rows"] or len(axes) != sources["axis_map"]["rows"]:
        raise ValueError("selection source row count mismatch")
    population = _normalize(prompts, axes, axis_bins)
    eligible_ids = (
        {row.candidate_id for row in population}
        if eligible_prompts_table is None
        else _eligible_ids(eligible_prompts_table)
    )
    population_ids = {row.candidate_id for row in population}
    if not eligible_ids.issubset(population_ids):
        raise ValueError("eligible-prompt table contains IDs outside the population")
    excluded_ids = sorted(population_ids - eligible_ids)
    registered = [row for row in population if row.candidate_id in eligible_ids]
    source_to_id: dict[str, str] = {}
    id_to_source: dict[str, str] = {}
    for row in registered:
        source_keyword = row.prompt.get("keyword")
        if not isinstance(source_keyword, str) or not source_keyword.strip():
            raise ValueError(f"prompt lacks source keyword: {row.candidate_id}")
        if source_to_id.setdefault(source_keyword, row.keyword_id) != row.keyword_id:
            raise ValueError("one source keyword maps to multiple keyword IDs")
        if id_to_source.setdefault(row.keyword_id, source_keyword) != source_keyword:
            raise ValueError("one keyword ID maps to multiple source keywords")
    output.mkdir(parents=True)
    prompt_target = output / "population-prompts.jsonl"
    _jsonl(prompt_target, (
        row.prompt for row in sorted(registered, key=lambda item: item.candidate_id)
    ))
    records_target = output / "population-selection-records.jsonl"
    _jsonl(records_target, ({
        "candidate_id": row.candidate_id,
        "keyword_id": row.keyword_id,
        "axis_bin": row.axis_bin,
        "observed_axis_1_percentile_0_1": row.observed_percentile,
        "assigned_axis_1_0_1": row.assigned_coordinate,
    } for row in sorted(registered, key=lambda item: item.candidate_id)))
    excluded_target = output / "excluded-prompts.jsonl"
    _jsonl(excluded_target, ({
        "prompt_id": prompt_id,
        "reason": "prompt_transfer_provenance_unverified",
    } for prompt_id in excluded_ids))
    priority_target = output / "keyword-priority.json"
    priority = {
        "format_version": "geodml-keyword-priority-v1",
        "priority_policy": "frozen-keyword-id-lexical-v1",
        "keywords": [
            {
                "source_keyword": id_to_source[keyword_id],
                "keyword_id": keyword_id,
                "priority_rank": rank,
            }
            for rank, keyword_id in enumerate(sorted(id_to_source))
        ],
    }
    priority_target.write_text(json.dumps(priority, indent=2, sort_keys=True) + "\n")
    result = {
        "format_version": "geodml-population-registration-v1",
        "population_count": len(population),
        "registered_prompt_count": len(registered),
        "excluded_prompt_count": len(excluded_ids),
        "keyword_count": len(priority["keywords"]),
        "axis_bins": axis_bins,
        "source_selection_manifest": str(selection_manifest.resolve()),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (
                prompt_target, records_target, priority_target, excluded_target
            )
        },
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis-bins", type=int, default=20)
    parser.add_argument("--eligible-prompts-table", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        args.selection_manifest.resolve(), args.output.resolve(), axis_bins=args.axis_bins,
        eligible_prompts_table=(
            None if args.eligible_prompts_table is None
            else args.eligible_prompts_table.resolve()
        ),
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
