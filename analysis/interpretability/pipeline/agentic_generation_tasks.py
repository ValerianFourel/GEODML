"""Frozen prompt selection and deterministic agentic generator cells."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agentic_search import (
    ExperimentalCondition,
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
)

METHODS = (ParallelExpansionV1, ReactiveSnippetLoopV1)
CONDITIONS = tuple(ExperimentalCondition)
ENGINES = ("duckduckgo", "searxng")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CalibrationPrompt:
    prompt_id: str
    prompt: str
    question_sha256: str
    axis_bin: int
    keyword: str


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _selection_key(seed: int, *values: str) -> tuple[str, ...]:
    payload = "\0".join((str(seed), *values)).encode("utf-8")
    return (hashlib.sha256(payload).hexdigest(), *values)


def load_calibration_prompts(
    prompts_path: Path,
    records_path: Path,
    *,
    prompt_count: int,
    seed: int,
) -> tuple[CalibrationPrompt, ...]:
    """Select a deterministic axis-balanced subset from a frozen pilot."""
    prompt_rows = _read_jsonl_objects(prompts_path)
    record_rows = _read_jsonl_objects(records_path)
    prompts_by_id: dict[str, Mapping[str, Any]] = {}
    for row in prompt_rows:
        prompt_id = str(row.get("candidate_id", ""))
        if not prompt_id or prompt_id in prompts_by_id:
            raise ValueError("prompt records have missing or duplicate candidate IDs")
        prompts_by_id[prompt_id] = row
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for row in record_rows:
        prompt_id = str(row.get("candidate_id", ""))
        if not prompt_id or prompt_id in records_by_id:
            raise ValueError(
                "selection records have missing or duplicate candidate IDs"
            )
        records_by_id[prompt_id] = row
    if set(prompts_by_id) != set(records_by_id):
        raise ValueError("prompt and selection-record candidate ID sets differ")
    if prompt_count > len(prompts_by_id):
        raise ValueError("prompt count exceeds the frozen pilot size")

    by_bin: dict[int, list[CalibrationPrompt]] = {}
    for prompt_id, row in prompts_by_id.items():
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"prompt {prompt_id} has no question text")
        actual_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        saved_hash = row.get("question_sha256")
        if saved_hash is not None and saved_hash != actual_hash:
            raise ValueError(f"prompt {prompt_id} question hash mismatch")
        axis_bin = records_by_id[prompt_id].get("axis_bin")
        if type(axis_bin) is not int or axis_bin < 0:
            raise ValueError(f"prompt {prompt_id} has an invalid axis bin")
        keyword = row.get("keyword")
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError(f"prompt {prompt_id} has no keyword")
        by_bin.setdefault(axis_bin, []).append(CalibrationPrompt(
            prompt_id=prompt_id,
            prompt=question,
            question_sha256=actual_hash,
            axis_bin=axis_bin,
            keyword=keyword,
        ))
    bins = sorted(by_bin)
    if prompt_count == len(prompts_by_id):
        selected = [prompt for rows in by_bin.values() for prompt in rows]
        selected.sort(key=lambda row: (row.axis_bin, row.prompt_id))
        return tuple(selected)
    if prompt_count < len(bins):
        raise ValueError(
            "prompt count must be at least the number of observed axis bins"
        )
    base, remainder = divmod(prompt_count, len(bins))
    extra_bins = set(sorted(
        bins,
        key=lambda axis_bin: _selection_key(seed, "axis-bin", str(axis_bin)),
    )[:remainder])
    selected: list[CalibrationPrompt] = []
    for axis_bin in bins:
        quota = base + int(axis_bin in extra_bins)
        candidates = sorted(
            by_bin[axis_bin],
            key=lambda row: _selection_key(seed, "prompt", row.prompt_id),
        )
        if quota > len(candidates):
            raise ValueError(f"axis bin {axis_bin} cannot satisfy its quota")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: (row.axis_bin, row.prompt_id))
    if len(selected) != prompt_count:
        raise AssertionError("calibration prompt selection has the wrong size")
    return tuple(selected)


def shard_prompts(
    prompts: Sequence[CalibrationPrompt],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[CalibrationPrompt, ...]:
    """Partition one frozen selection into disjoint, exhaustive stable shards."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid prompt shard")
    return tuple(prompts[shard_index::shard_count])


@dataclass(frozen=True, slots=True)
class SmokeCell:
    cell_id: str
    engine: str
    condition: ExperimentalCondition
    method_class: type[ParallelExpansionV1 | ReactiveSnippetLoopV1]
    prompt_id: str | None = None
    prompt: str | None = None
    prompt_sha256: str | None = None

    @property
    def core(self) -> dict[str, str]:
        core = {
            "method": self.method_class.method_id,
            "engine": self.engine,
            "condition": self.condition.value,
        }
        if self.prompt_id is not None:
            core["prompt_id"] = self.prompt_id
            if self.prompt_sha256 is None:
                raise AssertionError("calibration cell lacks its prompt hash")
            core["prompt_sha256"] = self.prompt_sha256
        return core


def build_cells(
    prompts: Sequence[CalibrationPrompt] | None = None,
) -> tuple[SmokeCell, ...]:
    cells: list[SmokeCell] = []
    prompt_values: Sequence[CalibrationPrompt | None] = prompts or (None,)
    for prompt in prompt_values:
        for engine in ENGINES:
            for condition in CONDITIONS:
                for method_class in METHODS:
                    core = {
                        "method": method_class.method_id,
                        "engine": engine,
                        "condition": condition.value,
                    }
                    if prompt is not None:
                        core["prompt_id"] = prompt.prompt_id
                        core["prompt_sha256"] = prompt.question_sha256
                    cells.append(SmokeCell(
                        cell_id=hashlib.sha256(_canonical(core)).hexdigest()[:20],
                        engine=engine,
                        condition=condition,
                        method_class=method_class,
                        prompt_id=None if prompt is None else prompt.prompt_id,
                        prompt=None if prompt is None else prompt.prompt,
                        prompt_sha256=(
                            None if prompt is None else prompt.question_sha256
                        ),
                    ))
    return tuple(cells)


def select_cells(
    cells: Sequence[SmokeCell], cell_ids_jsonl: Path | None
) -> tuple[SmokeCell, ...]:
    """Select an exact, canonically ordered subset from a frozen population."""

    if cell_ids_jsonl is None:
        return tuple(cells)
    requested: list[str] = []
    for row in _read_jsonl_objects(cell_ids_jsonl):
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell selection row lacks cell_id")
        requested.append(cell_id)
    if len(requested) != len(set(requested)):
        raise ValueError("cell selection contains duplicate cell IDs")
    indexed = {cell.cell_id: cell for cell in cells}
    unknown = set(requested) - set(indexed)
    if unknown:
        raise ValueError("cell selection contains unknown cell IDs")
    requested_set = set(requested)
    return tuple(cell for cell in cells if cell.cell_id in requested_set)
