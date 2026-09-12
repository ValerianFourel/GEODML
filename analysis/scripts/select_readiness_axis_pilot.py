#!/usr/bin/env python3
"""Select a deterministic, axis-balanced pilot from audited readiness prompts."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT_VERSION = "readiness-axis-balanced-pilot-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(seed: int, *values: str) -> str:
    payload = "\0".join((str(seed), *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty values")

    def at(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "q10": at(0.10),
        "q25": at(0.25),
        "median": at(0.50),
        "q75": at(0.75),
        "q90": at(0.90),
        "maximum": ordered[-1],
    }


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    keyword_id: str
    observed_percentile: float
    assigned_coordinate: float
    axis_bin: int
    question_length: int
    prompt: Mapping[str, Any]
    axis: Mapping[str, Any]


@dataclass(slots=True)
class Edge:
    target: int
    reverse: int
    capacity: int
    initial_capacity: int


class FlowNetwork:
    def __init__(self, size: int) -> None:
        self.graph: list[list[Edge]] = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int) -> Edge:
        forward = Edge(target, len(self.graph[target]), capacity, capacity)
        reverse = Edge(source, len(self.graph[source]), 0, 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            parent: list[tuple[int, int] | None] = [None] * len(self.graph)
            parent[source] = (source, -1)
            queue = deque([source])
            while queue and parent[sink] is None:
                node = queue.popleft()
                for edge_index, edge in enumerate(self.graph[node]):
                    if edge.capacity > 0 and parent[edge.target] is None:
                        parent[edge.target] = (node, edge_index)
                        queue.append(edge.target)
            if parent[sink] is None:
                return total
            amount = math.inf
            node = sink
            while node != source:
                previous, edge_index = parent[node]  # type: ignore[misc]
                amount = min(amount, self.graph[previous][edge_index].capacity)
                node = previous
            increment = int(amount)
            node = sink
            while node != source:
                previous, edge_index = parent[node]  # type: ignore[misc]
                edge = self.graph[previous][edge_index]
                edge.capacity -= increment
                self.graph[node][edge.reverse].capacity += increment
                node = previous
            total += increment


def _quota(total: int, labels: Sequence[str], seed: int) -> dict[str, int]:
    base, remainder = divmod(total, len(labels))
    priority = sorted(labels, key=lambda label: (_stable_hash(seed, label), label))
    extra = set(priority[:remainder])
    return {label: base + int(label in extra) for label in labels}


def _axis_quota(sample_size: int, axis_bins: int) -> dict[int, int]:
    base, remainder = divmod(sample_size, axis_bins)
    return {
        index: base + int(index < remainder)
        for index in range(axis_bins)
    }


def _normalize(
    prompts: Sequence[Mapping[str, Any]],
    axes: Sequence[Mapping[str, Any]],
    axis_bins: int,
) -> list[Candidate]:
    axis_by_id: dict[str, Mapping[str, Any]] = {}
    for row in axes:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in axis_by_id:
            raise ValueError("axis map has missing or duplicate candidate IDs")
        axis_by_id[candidate_id] = row
    prompt_ids = [str(row.get("candidate_id", "")) for row in prompts]
    if not all(prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompts have missing or duplicate candidate IDs")
    if set(prompt_ids) != set(axis_by_id):
        raise ValueError("prompt and axis candidate ID sets differ")

    output = []
    for prompt in prompts:
        candidate_id = str(prompt["candidate_id"])
        axis = axis_by_id[candidate_id]
        question = str(prompt.get("question", ""))
        keyword_id = str(prompt.get("keyword_id", ""))
        if not question.strip() or not keyword_id:
            raise ValueError(f"prompt lacks question or keyword ID: {candidate_id}")
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if str(prompt.get("question_sha256", "")) != question_hash:
            raise ValueError(f"prompt question hash mismatch: {candidate_id}")
        if str(axis.get("text_sha256", "")) != question_hash:
            raise ValueError(f"axis text hash mismatch: {candidate_id}")
        observed = float(axis["axis_1_percentile_0_1"])
        assigned = float(prompt["target_normalized_axis_1"])
        if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
            raise ValueError(f"invalid observed axis percentile: {candidate_id}")
        if not math.isfinite(assigned) or not 0.0 <= assigned <= 1.0:
            raise ValueError(f"invalid assigned axis coordinate: {candidate_id}")
        axis_bin = min(axis_bins - 1, int(observed * axis_bins))
        output.append(Candidate(
            candidate_id=candidate_id,
            keyword_id=keyword_id,
            observed_percentile=observed,
            assigned_coordinate=assigned,
            axis_bin=axis_bin,
            question_length=len(question),
            prompt=prompt,
            axis=axis,
        ))
    return output


def select_candidates(
    prompts: Sequence[Mapping[str, Any]],
    axes: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    axis_bins: int,
    master_seed: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    if sample_size <= 0 or axis_bins <= 1 or sample_size < axis_bins:
        raise ValueError("sample size and axis-bin count are invalid")
    candidates = _normalize(prompts, axes, axis_bins)
    if sample_size > len(candidates):
        raise ValueError("sample size exceeds the audited population")
    keywords = sorted({row.keyword_id for row in candidates})
    keyword_quota = _quota(sample_size, keywords, master_seed)
    axis_quota = _axis_quota(sample_size, axis_bins)
    available = Counter((row.keyword_id, row.axis_bin) for row in candidates)

    source = 0
    keyword_offset = 1
    axis_offset = keyword_offset + len(keywords)
    sink = axis_offset + axis_bins
    network = FlowNetwork(sink + 1)
    keyword_nodes = {keyword: keyword_offset + index for index, keyword in enumerate(keywords)}
    cell_edges: dict[tuple[str, int], Edge] = {}
    for keyword in keywords:
        network.add_edge(source, keyword_nodes[keyword], keyword_quota[keyword])
        for axis_bin in range(axis_bins):
            capacity = available[(keyword, axis_bin)]
            if capacity:
                cell_edges[(keyword, axis_bin)] = network.add_edge(
                    keyword_nodes[keyword], axis_offset + axis_bin, capacity
                )
    for axis_bin, quota in axis_quota.items():
        network.add_edge(axis_offset + axis_bin, sink, quota)
    achieved = network.maximum_flow(source, sink)
    if achieved != sample_size:
        raise ValueError(
            f"axis and keyword quotas are infeasible: selected {achieved} of {sample_size}"
        )

    by_cell: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        by_cell.setdefault((candidate.keyword_id, candidate.axis_bin), []).append(candidate)
    selected: list[Candidate] = []
    selected_cell_counts: Counter[tuple[str, int]] = Counter()
    for cell, edge in sorted(cell_edges.items()):
        count = edge.initial_capacity - edge.capacity
        if not count:
            continue
        lower = cell[1] / axis_bins
        width = 1.0 / axis_bins
        pool = list(by_cell[cell])
        chosen: list[Candidate] = []
        for slot in range(count):
            target = lower + width * (slot + 0.5) / count
            best = min(
                pool,
                key=lambda row: (
                    abs(row.observed_percentile - target),
                    0.25 * abs(row.assigned_coordinate - target),
                    _stable_hash(master_seed, row.candidate_id),
                    row.candidate_id,
                ),
            )
            pool.remove(best)
            chosen.append(best)
        selected.extend(chosen)
        selected_cell_counts[cell] += len(chosen)
    selected.sort(key=lambda row: (row.axis_bin, row.keyword_id, row.candidate_id))
    if len(selected) != sample_size or len({row.candidate_id for row in selected}) != sample_size:
        raise AssertionError("selection is not a unique sample of the requested size")

    observed_axis_counts = Counter(row.axis_bin for row in selected)
    observed_keyword_counts = Counter(row.keyword_id for row in selected)
    if observed_axis_counts != Counter(axis_quota):
        raise AssertionError("selected axis-bin counts differ from quotas")
    if observed_keyword_counts != Counter(keyword_quota):
        raise AssertionError("selected keyword counts differ from quotas")

    diagnostics = {
        "population_size": len(candidates),
        "sample_size": sample_size,
        "axis_bins": axis_bins,
        "keyword_count": len(keywords),
        "axis_bin_population_counts": {
            str(index): sum(row.axis_bin == index for row in candidates)
            for index in range(axis_bins)
        },
        "axis_bin_sample_counts": {
            str(index): observed_axis_counts[index] for index in range(axis_bins)
        },
        "keyword_sample_count_minimum": min(observed_keyword_counts.values()),
        "keyword_sample_count_maximum": max(observed_keyword_counts.values()),
        "population_observed_axis_percentiles": _quantiles(
            [row.observed_percentile for row in candidates]
        ),
        "sample_observed_axis_percentiles": _quantiles(
            [row.observed_percentile for row in selected]
        ),
        "population_assigned_axis": _quantiles(
            [row.assigned_coordinate for row in candidates]
        ),
        "sample_assigned_axis": _quantiles(
            [row.assigned_coordinate for row in selected]
        ),
        "population_question_characters": _quantiles(
            [float(row.question_length) for row in candidates]
        ),
        "sample_question_characters": _quantiles(
            [float(row.question_length) for row in selected]
        ),
    }
    return selected, diagnostics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--axis-map-jsonl", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--axis-bins", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=20260912)
    parser.add_argument("--expected-population-size", type=int, default=26009)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    prompts_path = arguments.prompts_jsonl.resolve()
    axis_path = arguments.axis_map_jsonl.resolve()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    prompts = _read_jsonl(prompts_path)
    axes = _read_jsonl(axis_path)
    if len(prompts) != arguments.expected_population_size:
        raise SystemExit(
            f"expected {arguments.expected_population_size} prompts, found {len(prompts)}"
        )
    selected, diagnostics = select_candidates(
        prompts,
        axes,
        sample_size=arguments.sample_size,
        axis_bins=arguments.axis_bins,
        master_seed=arguments.master_seed,
    )

    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise SystemExit(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    selected_prompts = [row.prompt for row in selected]
    selected_axes = [row.axis for row in selected]
    population_cell_counts = Counter(
        (row.keyword_id, row.axis_bin)
        for row in _normalize(prompts, axes, arguments.axis_bins)
    )
    sample_cell_counts = Counter((row.keyword_id, row.axis_bin) for row in selected)
    selection_records = [
        {
            "candidate_id": row.candidate_id,
            "keyword_id": row.keyword_id,
            "axis_bin": row.axis_bin,
            "observed_axis_1_percentile_0_1": row.observed_percentile,
            "assigned_axis_1_0_1": row.assigned_coordinate,
            "stratum_population_count": population_cell_counts[
                (row.keyword_id, row.axis_bin)
            ],
            "stratum_sample_count": sample_cell_counts[(row.keyword_id, row.axis_bin)],
            "analysis_weight": population_cell_counts[(row.keyword_id, row.axis_bin)]
            / sample_cell_counts[(row.keyword_id, row.axis_bin)],
        }
        for row in selected
    ]
    prompt_output = temporary / "pilot-prompts.jsonl"
    axis_output = temporary / "pilot-axis.jsonl"
    records_output = temporary / "selection-records.jsonl"
    _write_jsonl(prompt_output, selected_prompts)
    _write_jsonl(axis_output, selected_axes)
    _write_jsonl(records_output, selection_records)
    selection_id = hashlib.sha256(_canonical({
        "format_version": FORMAT_VERSION,
        "master_seed": arguments.master_seed,
        "candidate_ids": [row.candidate_id for row in selected],
    })).hexdigest()
    manifest = {
        "format_version": FORMAT_VERSION,
        "scientific_result": False,
        "selection_id": selection_id,
        "source_git_commit": arguments.source_git_commit,
        "master_seed": arguments.master_seed,
        "selection_design": (
            "exact equal-mass observed-axis bins with near-equal keyword quotas "
            "and deterministic within-stratum selection"
        ),
        "interpretation_guard": (
            "The consensus embedding coordinate stratifies the audited prompt "
            "population. It does not define the assigned experimental variable."
        ),
        "sources": {
            "prompts": {
                "path": str(prompts_path),
                "sha256": _sha256_file(prompts_path),
                "rows": len(prompts),
            },
            "axis_map": {
                "path": str(axis_path),
                "sha256": _sha256_file(axis_path),
                "rows": len(axes),
            },
        },
        "artifacts": {
            "prompts": {
                "path": "pilot-prompts.jsonl",
                "sha256": _sha256_file(prompt_output),
                "rows": len(selected_prompts),
            },
            "axis_map": {
                "path": "pilot-axis.jsonl",
                "sha256": _sha256_file(axis_output),
                "rows": len(selected_axes),
            },
            "selection_records": {
                "path": "selection-records.jsonl",
                "sha256": _sha256_file(records_output),
                "rows": len(selection_records),
            },
        },
        "diagnostics": diagnostics,
    }
    manifest_path = temporary / "selection-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(output)
    print("READINESS_AXIS_PILOT=" + json.dumps({
        "selection_id": selection_id,
        "population_size": diagnostics["population_size"],
        "sample_size": diagnostics["sample_size"],
        "axis_bins": diagnostics["axis_bins"],
        "keyword_count": diagnostics["keyword_count"],
        "keyword_sample_count_minimum": diagnostics[
            "keyword_sample_count_minimum"
        ],
        "keyword_sample_count_maximum": diagnostics[
            "keyword_sample_count_maximum"
        ],
        "scientific_result": False,
    }, sort_keys=True))
    print(f"SELECTION_MANIFEST={output / 'selection-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
