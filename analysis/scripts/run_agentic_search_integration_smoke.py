#!/usr/bin/env python3
"""Run a resumable, non-scientific agentic-search integration smoke."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_search import (  # noqa: E402
    AgentExecutionError,
    ContextCompactor,
    ExperimentalCondition,
    LLMRequest,
    MemoizingSnippetScorer,
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
    SearchResponse,
    SentenceTransformersCrossEncoderScorer,
    Snippet,
    write_trace_atomic,
)
from analysis.scripts.run_acl_arr_vllm import VllmChatClient  # noqa: E402


METHODS = (ParallelExpansionV1, ReactiveSnippetLoopV1)
CONDITIONS = tuple(ExperimentalCondition)
ENGINES = ("duckduckgo", "searxng")
TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(value)}


def _read_snapshot(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing search snapshot: {path}")
    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("pyarrow is required for Parquet snapshots") from error
        rows = parquet.read_table(
            path, columns=["keyword", "position", "title", "url", "snippet"]
        ).to_pylist()
    else:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    required = {"keyword", "position", "title", "url", "snippet"}
    if not rows or any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise ValueError(f"search snapshot has invalid rows: {path}")
    return rows


def _normalize_usable_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    raw_position = row["position"]
    if isinstance(raw_position, bool):
        return None, "invalid_position"
    if isinstance(raw_position, int):
        position = raw_position
    elif isinstance(raw_position, float) and raw_position.is_integer():
        position = int(raw_position)
    elif isinstance(raw_position, str) and re.fullmatch(r"[0-9]+", raw_position):
        position = int(raw_position)
    else:
        return None, "invalid_position"
    if position < 1:
        return None, "invalid_position"

    for field in ("keyword", "title", "url", "snippet"):
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            return None, f"invalid_{field}"
    try:
        Snippet.from_mapping({
            "url": row["url"],
            "title": row["title"],
            "text": row["snippet"],
        })
    except ValueError:
        return None, "invalid_url"

    normalized = dict(row)
    normalized["position"] = position
    return normalized, None


class FrozenSnapshotSearchAdapter:
    """Deterministic lexical retrieval over one immutable search snapshot."""

    def __init__(self, engine: str, path: Path) -> None:
        if engine not in ENGINES:
            raise ValueError(f"unsupported search engine: {engine}")
        self.engine = engine
        self.path = path.resolve()
        self.snapshot_sha256 = _sha256_file(self.path)
        source_rows = _read_snapshot(self.path)
        rows: list[dict[str, Any]] = []
        exclusion_reasons: Counter[str] = Counter()
        for row in source_rows:
            normalized, reason = _normalize_usable_row(row)
            if normalized is None:
                if reason is None:
                    raise AssertionError("excluded snapshot row has no reason")
                exclusion_reasons[reason] += 1
            else:
                rows.append(normalized)
        if not rows:
            raise ValueError(f"search snapshot has no usable result rows: {self.path}")
        self.rows = rows
        self.snapshot_rows = {
            "total": len(source_rows),
            "usable": len(rows),
            "excluded": len(source_rows) - len(rows),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        }

    @property
    def keywords(self) -> set[str]:
        return {str(row["keyword"]) for row in self.rows if str(row["keyword"]).strip()}

    async def search(self, query: str, limit: int) -> SearchResponse:
        if limit < 1:
            raise ValueError("search limit must be positive")
        query_terms = _tokens(query)

        def key(item: tuple[int, Mapping[str, Any]]) -> tuple[Any, ...]:
            index, row = item
            keyword_terms = _tokens(str(row["keyword"]))
            evidence_terms = _tokens(f"{row['title']} {row['snippet']}")
            exact = int(str(row["keyword"]).casefold() == query.casefold())
            overlap = len(query_terms & keyword_terms) * 4 + len(query_terms & evidence_terms)
            stable = hashlib.sha256(f"{query}\0{row['url']}".encode()).hexdigest()
            return (-exact, -overlap, int(row["position"]), stable, index)

        selected = [row for _, row in sorted(enumerate(self.rows), key=key)[:limit]]
        snippets = tuple(
            Snippet.from_mapping({
                "url": str(row["url"]),
                "title": str(row["title"]),
                "text": str(row["snippet"]),
            })
            for row in selected
        )
        return SearchResponse(
            engine=self.engine,
            query=query,
            snippets=snippets,
            raw_payload={
                "format_version": "frozen-search-snapshot-response-v1",
                "snapshot": str(self.path),
                "snapshot_sha256": self.snapshot_sha256,
                "snapshot_rows": self.snapshot_rows,
                "query": query,
                "selection": "deterministic-lexical-v1",
                "rows": selected,
            },
        )


class SmokeConditionHook:
    """Visible smoke-only transformations, not scientific interventions."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def apply(
        self,
        condition: ExperimentalCondition,
        snippets: Sequence[Snippet],
    ) -> Sequence[Snippet]:
        rows = list(snippets)
        if condition is ExperimentalCondition.NATURAL:
            return rows
        if condition is ExperimentalCondition.ABLATED:
            return [row for index, row in enumerate(rows) if index % 4 != 0]
        rng = random.Random(self.seed)
        rng.shuffle(rows)
        return rows


class VllmAgentGenerator:
    def __init__(
        self,
        client: VllmChatClient,
        *,
        seed: int,
        max_tokens: int,
        request_semaphore: asyncio.Semaphore,
    ) -> None:
        self.client = client
        self.seed = seed
        self.max_tokens = max_tokens
        self.request_semaphore = request_semaphore
        self.call_index = 0
        self.diagnostics: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> str:
        self.call_index += 1
        call_index = self.call_index
        queued_at = time.perf_counter()
        async with self.request_semaphore:
            started_at = time.perf_counter()
            try:
                raw, usage = await self.client.complete(
                    prompt=request.prompt,
                    schema_name=f"agentic_{request.purpose}_{call_index}",
                    schema=request.response_schema,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    seed=self.seed + call_index,
                )
            except Exception as error:
                self.diagnostics.append({
                    "call_index": call_index,
                    "purpose": request.purpose,
                    "queue_seconds": started_at - queued_at,
                    "request_seconds": time.perf_counter() - started_at,
                    "error": f"{type(error).__name__}: {error}",
                    "usage": None,
                })
                raise
        self.diagnostics.append({
            "call_index": call_index,
            "purpose": request.purpose,
            "queue_seconds": started_at - queued_at,
            "request_seconds": time.perf_counter() - started_at,
            "error": None,
            "usage": json.loads(json.dumps(usage)),
        })
        return raw


@dataclass(frozen=True)
class SmokeInputs:
    output: Path
    base_url: str
    model_id: str
    model_revision: str
    cross_encoder_snapshot: Path
    cross_encoder_revision: str
    search_snapshots: Mapping[str, Path]
    seed: int
    max_tokens: int
    request_concurrency: int = 1
    disable_thinking: bool = False
    prompts_jsonl: Path | None = None
    selection_records_jsonl: Path | None = None
    prompt_count: int = 1
    prompt_selection_seed: int = 20260912

    def __post_init__(self) -> None:
        if (
            type(self.request_concurrency) is not int
            or not 1 <= self.request_concurrency <= 4
        ):
            raise ValueError("request_concurrency must be an integer from 1 to 4")
        if (self.prompts_jsonl is None) != (self.selection_records_jsonl is None):
            raise ValueError(
                "prompts and selection records must be configured together"
            )
        if type(self.prompt_count) is not int or self.prompt_count < 1:
            raise ValueError("prompt count must be a positive integer")


@dataclass(frozen=True, slots=True)
class CalibrationPrompt:
    prompt_id: str
    prompt: str
    question_sha256: str
    axis_bin: int


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
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _selection_key(seed: int, *values: str) -> tuple[str, ...]:
    payload = "\0".join((str(seed), *values)).encode("utf-8")
    return (hashlib.sha256(payload).hexdigest(), *values)


def _load_calibration_prompts(
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
        by_bin.setdefault(axis_bin, []).append(CalibrationPrompt(
            prompt_id=prompt_id,
            prompt=question,
            question_sha256=actual_hash,
            axis_bin=axis_bin,
        ))
    bins = sorted(by_bin)
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


def _cells(
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


def _config(
    inputs: SmokeInputs,
    keyword: str,
    prompts: Sequence[CalibrationPrompt] | None = None,
) -> dict[str, Any]:
    config = {
        "format_version": "agentic-search-integration-smoke-v1",
        "scientific_result": False,
        "git_commit": os.environ.get("GEODML_EXECUTION_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_id": inputs.model_id,
        "model_revision": inputs.model_revision,
        "base_url": inputs.base_url,
        "cross_encoder_snapshot": str(inputs.cross_encoder_snapshot.resolve()),
        "cross_encoder_revision": inputs.cross_encoder_revision,
        "disable_thinking": inputs.disable_thinking,
        "search_snapshots": {
            engine: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
            }
            for engine, path in sorted(inputs.search_snapshots.items())
        },
        "methods": [method.method_id for method in METHODS],
        "conditions": [condition.value for condition in CONDITIONS],
        "engines": list(ENGINES),
        "cell_count": len(METHODS) * len(CONDITIONS) * len(ENGINES),
        "keyword": keyword,
        "seed": inputs.seed,
        "max_tokens": inputs.max_tokens,
        "request_concurrency": inputs.request_concurrency,
        "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
        "condition_mode": "smoke-only-subset-and-order-v1",
    }
    if prompts is not None:
        if inputs.prompts_jsonl is None or inputs.selection_records_jsonl is None:
            raise AssertionError("calibration inputs lack source paths")
        prompt_identity = [
            {
                "prompt_id": prompt.prompt_id,
                "question_sha256": prompt.question_sha256,
                "axis_bin": prompt.axis_bin,
            }
            for prompt in prompts
        ]
        config.pop("slurm_job_id")
        config.update({
            "format_version": "agentic-search-execution-calibration-v1",
            "cell_count": len(prompts) * len(METHODS) * len(CONDITIONS) * len(ENGINES),
            "prompt_count": len(prompts),
            "prompt_selection_seed": inputs.prompt_selection_seed,
            "prompt_selection_sha256": hashlib.sha256(
                _canonical(prompt_identity)
            ).hexdigest(),
            "prompt_sources": {
                "prompts_jsonl": {
                    "path": str(inputs.prompts_jsonl.resolve()),
                    "sha256": _sha256_file(inputs.prompts_jsonl),
                },
                "selection_records_jsonl": {
                    "path": str(inputs.selection_records_jsonl.resolve()),
                    "sha256": _sha256_file(inputs.selection_records_jsonl),
                },
            },
            "axis_bin_counts": {
                str(axis_bin): count
                for axis_bin, count in sorted(Counter(
                    prompt.axis_bin for prompt in prompts
                ).items())
            },
            "keyword": None,
        })
    return config


def _cache_metrics(compactor: ContextCompactor) -> dict[str, int] | None:
    stats = getattr(compactor.scorer, "stats", None)
    if stats is None:
        return None
    return {
        "hits": int(stats.hits),
        "misses": int(stats.misses),
        "entries": int(stats.entries),
    }


def _validate_completed_cell(
    cell: SmokeCell,
    trace_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_result = {"cell_id": cell.cell_id, **cell.core}
    if any(result.get(key) != value for key, value in expected_result.items()):
        raise ValueError(f"completed result identity mismatch: {result_path}")
    if result.get("trace") != str(trace_path.resolve()):
        raise ValueError(f"completed result trace path mismatch: {result_path}")
    actual_hash = _validate_trace(cell, trace_path)
    if result.get("trace_sha256") != actual_hash:
        raise ValueError(f"completed result trace hash mismatch: {result_path}")
    return result


def _validate_trace(cell: SmokeCell, trace_path: Path) -> str:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    saved_hash = trace.pop("trace_sha256", None)
    actual_hash = hashlib.sha256(_canonical(trace)).hexdigest()
    if saved_hash != actual_hash:
        raise ValueError(f"trace hash mismatch: {trace_path}")
    if (
        trace.get("method_id") != cell.method_class.method_id
        or trace.get("condition") != cell.condition.value
        or trace.get("search_engine") != cell.engine
    ):
        raise ValueError(f"trace identity mismatch: {trace_path}")
    return actual_hash


def _recover_incomplete_trace(cell: SmokeCell, trace_path: Path) -> None:
    trace_hash = _validate_trace(cell, trace_path)
    destination = (
        trace_path.parents[1] / "failed_traces" / cell.cell_id
        / f"recovered-{trace_hash}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != trace_path.read_bytes():
            raise ValueError(f"recovered trace content mismatch: {destination}")
        trace_path.unlink()
        return
    os.replace(trace_path, destination)


def _load_completed_cells(
    output: Path,
    cells: Sequence[SmokeCell] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[SmokeCell]]:
    completed: dict[str, dict[str, Any]] = {}
    pending: list[SmokeCell] = []
    for cell in cells or _cells():
        trace_path = output / "traces" / f"{cell.cell_id}.json"
        result_path = output / "results" / f"{cell.cell_id}.json"
        trace_exists = trace_path.is_file()
        result_exists = result_path.is_file()
        if trace_exists and result_exists:
            completed[cell.cell_id] = _validate_completed_cell(
                cell, trace_path, result_path
            )
        elif result_exists:
            raise ValueError(f"result exists without immutable trace: {result_path}")
        else:
            if trace_exists:
                _recover_incomplete_trace(cell, trace_path)
            pending.append(cell)
    return completed, pending


async def run_smoke(
    inputs: SmokeInputs,
    *,
    client_context: Any | None = None,
    compactor: ContextCompactor | None = None,
) -> dict[str, Any]:
    if set(inputs.search_snapshots) != set(ENGINES):
        raise ValueError("both duckduckgo and searxng snapshots are required")
    adapters = {
        engine: FrozenSnapshotSearchAdapter(engine, inputs.search_snapshots[engine])
        for engine in ENGINES
    }
    shared_keywords = set.intersection(*(adapter.keywords for adapter in adapters.values()))
    if not shared_keywords:
        raise ValueError("search snapshots have no shared keyword")
    keyword = sorted(shared_keywords, key=lambda value: (value.casefold(), value))[0]
    prompts = (
        _load_calibration_prompts(
            inputs.prompts_jsonl,
            inputs.selection_records_jsonl,
            prompt_count=inputs.prompt_count,
            seed=inputs.prompt_selection_seed,
        )
        if inputs.prompts_jsonl is not None
        and inputs.selection_records_jsonl is not None
        else None
    )
    legacy_prompt = (
        f"Answer the following retrieval question: {keyword}. "
        "For the reactive method, perform at least one search before finishing."
    )
    cells = _cells(prompts)
    config = _config(inputs, keyword, prompts)
    config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    inputs.output.mkdir(parents=True, exist_ok=True)
    config_path = inputs.output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != {**config, "config_sha256": config_hash}:
            raise ValueError("existing smoke configuration differs")
    else:
        _write_json_atomic(config_path, {**config, "config_sha256": config_hash})

    completed, pending = _load_completed_cells(inputs.output, cells)
    manifest_path = inputs.output / "run_manifest.json"
    if not pending and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("config_sha256") != config_hash
            or existing_manifest.get("status") != "complete"
            or existing_manifest.get("completed_count") != config["cell_count"]
            or existing_manifest.get("remaining_count") != 0
        ):
            raise ValueError("completed smoke manifest is inconsistent")
        return existing_manifest

    if compactor is None:
        scorer = MemoizingSnippetScorer(
            SentenceTransformersCrossEncoderScorer(
                str(inputs.cross_encoder_snapshot),
                model_revision=None,
                device="cpu",
                batch_size=32,
                local_files_only=True,
            )
        )
        compactor = ContextCompactor(scorer)
    if client_context is None:
        client_context = VllmChatClient(
            base_url=inputs.base_url,
            api_key=None,
            server_model_name=inputs.model_id,
            timeout_seconds=300.0,
            maximum_attempts=2,
            chat_template_kwargs=(
                {"enable_thinking": False} if inputs.disable_thinking else None
            ),
        )
    async with client_context as client:
        request_semaphore = asyncio.Semaphore(inputs.request_concurrency)

        async def execute(cell: SmokeCell) -> tuple[str, dict[str, Any]]:
            cell_started = time.perf_counter()
            trace_path = inputs.output / "traces" / f"{cell.cell_id}.json"
            result_path = inputs.output / "results" / f"{cell.cell_id}.json"
            diagnostics_path = inputs.output / "diagnostics" / f"{cell.cell_id}.json"
            generator = VllmAgentGenerator(
                client,
                seed=inputs.seed + int(cell.cell_id[:8], 16),
                max_tokens=inputs.max_tokens,
                request_semaphore=request_semaphore,
            )
            method = cell.method_class(
                llm=generator,
                search=adapters[cell.engine],
                compactor=compactor,
                condition_hook=SmokeConditionHook(inputs.seed),
            )
            try:
                result = await method.run(
                    cell.prompt if cell.prompt is not None else legacy_prompt,
                    cell.condition,
                )
            except AgentExecutionError as error:
                failure_hash = error.trace.to_dict()["trace_sha256"]
                failure_path = (
                    inputs.output / "failed_traces" / cell.cell_id
                    / f"{failure_hash}.json"
                )
                if not failure_path.exists():
                    write_trace_atomic(failure_path, error.trace)
                _write_json_atomic(diagnostics_path, {
                    "cell_id": cell.cell_id,
                    **cell.core,
                    "status": "failed",
                    "elapsed_seconds": time.perf_counter() - cell_started,
                    "llm_calls": generator.diagnostics,
                })
                raise
            search_count = sum(
                event.event_type == "search" for event in result.trace.events
            )
            expected_searches = (
                3 if cell.method_class is ParallelExpansionV1 else None
            )
            if expected_searches is not None and search_count != expected_searches:
                raise RuntimeError(
                    f"{cell.method_class.method_id} made {search_count} searches, "
                    f"expected {expected_searches}"
                )
            if cell.method_class is ReactiveSnippetLoopV1 and search_count < 1:
                raise RuntimeError(
                    "Reactive-Snippet-Loop-v1 did not exercise retrieval"
                )
            trace_hash = write_trace_atomic(trace_path, result.trace)
            record = {
                "cell_id": cell.cell_id,
                **cell.core,
                "ranking": list(result.ranking),
                "answer": result.answer,
                "final_snippet_count": len(result.final_snippets),
                "search_count": search_count,
                "trace": str(trace_path.resolve()),
                "trace_sha256": trace_hash,
            }
            _write_json_atomic(diagnostics_path, {
                "cell_id": cell.cell_id,
                **cell.core,
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - cell_started,
                "llm_calls": generator.diagnostics,
            })
            _write_json_atomic(result_path, record)
            return cell.cell_id, record

        tasks = [asyncio.create_task(execute(cell)) for cell in pending]
        try:
            for future in asyncio.as_completed(tasks):
                cell_id, record = await future
                completed[cell_id] = record
                _write_json_atomic(inputs.output / "run_manifest.json", {
                    **config,
                    "config_sha256": config_hash,
                    "status": "checkpointed",
                    "completed_count": len(completed),
                    "remaining_count": config["cell_count"] - len(completed),
                    "compactor_cache_this_invocation": _cache_metrics(compactor),
                })
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    manifest = {
        **config,
        "config_sha256": config_hash,
        "status": "complete",
        "completed_count": len(completed),
        "remaining_count": 0,
        "compactor_cache_this_invocation": _cache_metrics(compactor),
    }
    _write_json_atomic(inputs.output / "run_manifest.json", manifest)
    return manifest


def _binding(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in ENGINES or not path:
        raise argparse.ArgumentTypeError("search snapshot must be ENGINE=PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--cross-encoder-snapshot", type=Path, required=True)
    parser.add_argument("--cross-encoder-revision", required=True)
    parser.add_argument("--search-snapshot", action="append", type=_binding, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-concurrency", type=int, default=1)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--selection-records-jsonl", type=Path)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--prompt-selection-seed", type=int, default=20260912)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false to the model chat template.",
    )
    arguments = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", arguments.model_revision) is None:
        parser.error("model revision must be a 40-character SHA")
    if re.fullmatch(r"[0-9a-f]{40}", arguments.cross_encoder_revision) is None:
        parser.error("cross-encoder revision must be a 40-character SHA")
    snapshots = dict(arguments.search_snapshot)
    if len(snapshots) != len(arguments.search_snapshot):
        parser.error("search engines must be unique")
    inputs = SmokeInputs(
        output=arguments.output,
        base_url=arguments.base_url,
        model_id=arguments.model_id,
        model_revision=arguments.model_revision,
        cross_encoder_snapshot=arguments.cross_encoder_snapshot,
        cross_encoder_revision=arguments.cross_encoder_revision,
        search_snapshots=snapshots,
        seed=arguments.seed,
        max_tokens=arguments.max_tokens,
        request_concurrency=arguments.request_concurrency,
        disable_thinking=arguments.disable_thinking,
        prompts_jsonl=arguments.prompts_jsonl,
        selection_records_jsonl=arguments.selection_records_jsonl,
        prompt_count=arguments.prompt_count,
        prompt_selection_seed=arguments.prompt_selection_seed,
    )
    manifest = asyncio.run(run_smoke(inputs))
    print("AGENTIC_INTEGRATION_SMOKE=" + json.dumps({
        key: manifest[key]
        for key in ("status", "cell_count", "completed_count", "remaining_count")
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
