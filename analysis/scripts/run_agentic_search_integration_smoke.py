#!/usr/bin/env python3
"""Run a resumable, non-scientific agentic-search integration smoke."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_generation_tasks import (  # noqa: E402
    CONDITIONS as CONDITIONS,
    ENGINES as ENGINES,
    METHODS as METHODS,
    CalibrationPrompt as CalibrationPrompt,
    SmokeCell as SmokeCell,
    _canonical as _canonical,
    _read_jsonl_objects as _read_jsonl_objects,
    _selection_key as _selection_key,
    build_cells as _cells,
    load_calibration_prompts as _load_calibration_prompts,
    select_cells as _select_cells,
    shard_prompts as _prompt_shard,
)
from analysis.interpretability.pipeline.agentic_search import (  # noqa: E402
    FINAL_ANSWER_MAX_CHARACTERS,
    SEARCH_RESULT_LIMIT,
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
from analysis.interpretability.pipeline.inference_budget import (  # noqa: E402
    AllocationBudget,
)
from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
    TaskClaim,
)
from analysis.scripts.run_acl_arr_vllm import VllmChatClient  # noqa: E402

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
QUERY_PURPOSES = frozenset(("parallel_query_expansion",))
ANSWER_PURPOSES = frozenset(
    ("parallel_final", "reactive_action", "reactive_forced_finish")
)
LEGACY_RESUME_COMMIT = "b561f1aaf54971a89fa2dabb7f3f9d32770ce8cc"
PREVIOUS_RESUME_COMMIT = "3426907232d98634a3502381e58708cc3c028f30"
EVIDENCE_ID_RESUME_COMMIT = "c5eba6036fc1bef032bfa0992007f396133b72e7"
REPAIR_RESUME_COMMIT = "3d0fa062094dca9571198ae8198be6a00b634f7e"
SERIAL_RESUME_COMMIT = "6a2469958df8355234982a45608c220780b855a0"
OPTIMIZED_RESUME_COMMIT = "40d848175ce327585b048efc293e7bfa5e4482c5"
MALFORMED_PREFIX_V1_RESUME_COMMITS = (
    "0b8902f23109b24e7f47fe5dcfc053e75cdfbaf5",
    "f301dc549107a855127f5be67fb95b523f6dc024",
)
FAILED_CELL_RETRY_PASSES = 1


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

    def urls_for_keyword(self, keyword: str) -> tuple[str, ...]:
        """Return the frozen natural URL order for one exact baseline query."""
        matches = sorted(
            (row for row in self.rows if row["keyword"] == keyword),
            key=lambda row: (row["position"], row["url"]),
        )
        output: list[str] = []
        seen: set[str] = set()
        for row in matches:
            url = str(row["url"])
            if url not in seen:
                seen.add(url)
                output.append(url)
        return tuple(output[:SEARCH_RESULT_LIMIT])

    def urls_for_query(self, query: str) -> tuple[str, ...]:
        """Return URLs from the same frozen lexical retrieval used by search."""
        output: list[str] = []
        seen: set[str] = set()
        for row in self._select_rows(query, SEARCH_RESULT_LIMIT):
            url = str(row["url"])
            if url not in seen:
                seen.add(url)
                output.append(url)
        return tuple(output)

    @property
    def keywords(self) -> set[str]:
        return {str(row["keyword"]) for row in self.rows if str(row["keyword"]).strip()}

    def _select_rows(self, query: str, limit: int) -> list[dict[str, Any]]:
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

        return [row for _, row in sorted(enumerate(self.rows), key=key)[:limit]]

    async def search(self, query: str, limit: int) -> SearchResponse:
        selected = self._select_rows(query, limit)
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


class TargetUrlConditionHook:
    """Apply the preregistered target removal and prompt-specific permutation."""

    def __init__(self, *, target_url: str, seed: int) -> None:
        if not target_url:
            raise ValueError("target URL must be nonempty")
        self.target_url = target_url
        self.seed = seed
        self.calls: list[dict[str, Any]] = []

    def apply(
        self,
        condition: ExperimentalCondition,
        snippets: Sequence[Snippet],
    ) -> Sequence[Snippet]:
        natural = list(snippets)
        target_count_before = sum(row.url == self.target_url for row in natural)
        if condition is ExperimentalCondition.NATURAL:
            output = natural
        elif condition is ExperimentalCondition.ABLATED:
            output = [row for row in natural if row.url != self.target_url]
        else:
            output = list(natural)
            random.Random(self.seed + len(self.calls)).shuffle(output)
        target_count_after = sum(row.url == self.target_url for row in output)
        self.calls.append({
            "condition": condition.value,
            "input_count": len(natural),
            "output_count": len(output),
            "target_count_before": target_count_before,
            "target_count_after": target_count_after,
            "target_removed_count": target_count_before - target_count_after,
            "input_urls_sha256": hashlib.sha256(
                _canonical([row.url for row in natural])
            ).hexdigest(),
            "output_urls_sha256": hashlib.sha256(
                _canonical([row.url for row in output])
            ).hexdigest(),
        })
        return output


def _build_target_urls(
    prompts: Sequence[CalibrationPrompt],
    adapters: Mapping[str, FrozenSnapshotSearchAdapter],
    *,
    seed: int,
    selection_audit: dict[str, dict[str, int]] | None = None,
) -> dict[tuple[str, str], str]:
    """Freeze balanced baseline target URLs before any shard is selected."""
    output: dict[tuple[str, str], str] = {}
    for engine, adapter in sorted(adapters.items()):
        groups: dict[int, list[CalibrationPrompt]] = {}
        urls_by_prompt: dict[str, tuple[str, ...]] = {}
        mode_counts: Counter[str] = Counter()
        for prompt in prompts:
            urls = adapter.urls_for_keyword(prompt.keyword)
            if not urls:
                urls = adapter.urls_for_query(prompt.keyword)
                mode_counts["deterministic_lexical_fallback"] += 1
            else:
                mode_counts["exact_keyword"] += 1
            if not urls:
                raise ValueError(f"{engine} snapshot has no usable baseline rows")
            urls_by_prompt[prompt.prompt_id] = urls
            groups.setdefault(len(urls), []).append(prompt)
        for count, group in groups.items():
            ordered = sorted(
                group,
                key=lambda prompt: _selection_key(
                    seed, "ablation-target", engine, prompt.prompt_id
                ),
            )
            for index, prompt in enumerate(ordered):
                output[(prompt.prompt_id, engine)] = urls_by_prompt[
                    prompt.prompt_id
                ][index % count]
        if selection_audit is not None:
            selection_audit[engine] = dict(sorted(mode_counts.items()))
    return output


class VllmAgentGenerator:
    def __init__(
        self,
        client: VllmChatClient,
        *,
        seed: int,
        query_max_tokens: int,
        final_max_tokens: int = 2048,
        request_semaphore: asyncio.Semaphore,
    ) -> None:
        self.client = client
        self.seed = seed
        self.max_tokens_by_purpose = {
            **{purpose: query_max_tokens for purpose in QUERY_PURPOSES},
            **{purpose: final_max_tokens for purpose in ANSWER_PURPOSES},
        }
        self.request_semaphore = request_semaphore
        self.call_index = 0
        self.diagnostics: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> str:
        try:
            max_tokens = self.max_tokens_by_purpose[request.purpose]
        except KeyError as error:
            raise ValueError(
                f"unsupported LLM request purpose: {request.purpose}"
            ) from error
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
                    max_tokens=max_tokens,
                    seed=self.seed + call_index,
                )
            except Exception as error:
                self.diagnostics.append({
                    "call_index": call_index,
                    "purpose": request.purpose,
                    "max_tokens": max_tokens,
                    "queue_seconds": started_at - queued_at,
                    "request_seconds": time.perf_counter() - started_at,
                    "error": f"{type(error).__name__}: {error}",
                    "usage": None,
                })
                raise
        self.diagnostics.append({
            "call_index": call_index,
            "purpose": request.purpose,
            "max_tokens": max_tokens,
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
    query_max_tokens: int | None = None
    final_max_tokens: int = 2048
    request_concurrency: int = 1
    cell_concurrency: int | None = None
    disable_thinking: bool = False
    prompts_jsonl: Path | None = None
    selection_records_jsonl: Path | None = None
    cell_ids_jsonl: Path | None = None
    prompt_count: int = 1
    prompt_selection_seed: int = 20260912
    prompt_shard_index: int = 0
    prompt_shard_count: int = 1
    production_conditions: bool = False
    shared_claim_root: Path | None = None
    worker_index: int = 0
    worker_count: int = 1

    def __post_init__(self) -> None:
        budgets = [
            ("max_tokens", self.max_tokens),
            ("final_max_tokens", self.final_max_tokens),
        ]
        if self.query_max_tokens is not None:
            budgets.append(("query_max_tokens", self.query_max_tokens))
        for name, value in budgets:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.request_concurrency) is not int
            or not 1 <= self.request_concurrency <= 4
        ):
            raise ValueError("request_concurrency must be an integer from 1 to 4")
        if self.cell_concurrency is not None and (
            type(self.cell_concurrency) is not int
            or not self.request_concurrency <= self.cell_concurrency <= 32
        ):
            raise ValueError(
                "cell_concurrency must be an integer from request_concurrency to 32"
            )
        if (self.prompts_jsonl is None) != (self.selection_records_jsonl is None):
            raise ValueError(
                "prompts and selection records must be configured together"
            )
        if type(self.prompt_count) is not int or self.prompt_count < 1:
            raise ValueError("prompt count must be a positive integer")
        if type(self.prompt_shard_count) is not int or self.prompt_shard_count < 1:
            raise ValueError("prompt shard count must be a positive integer")
        if (
            type(self.prompt_shard_index) is not int
            or not 0 <= self.prompt_shard_index < self.prompt_shard_count
        ):
            raise ValueError("prompt shard index must be in [0, prompt shard count)")
        if self.prompt_shard_count > 1 and self.prompts_jsonl is None:
            raise ValueError("prompt sharding requires frozen prompt inputs")
        if self.production_conditions and self.prompts_jsonl is None:
            raise ValueError("production conditions require frozen prompt inputs")
        if self.cell_ids_jsonl is not None and self.prompts_jsonl is None:
            raise ValueError("cell selection requires frozen prompt inputs")
        if type(self.worker_count) is not int or self.worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        if type(self.worker_index) is not int or not 0 <= self.worker_index < self.worker_count:
            raise ValueError("worker_index must be in [0, worker_count)")
        if self.shared_claim_root is None and (self.worker_index or self.worker_count != 1):
            raise ValueError("worker preferences require a shared claim root")

    @property
    def resolved_query_max_tokens(self) -> int:
        if self.query_max_tokens is None:
            return self.max_tokens
        return self.query_max_tokens

    @property
    def resolved_cell_concurrency(self) -> int:
        if self.cell_concurrency is None:
            return self.request_concurrency
        return self.cell_concurrency


def _config(
    inputs: SmokeInputs,
    keyword: str,
    prompts: Sequence[CalibrationPrompt] | None = None,
    cells: Sequence[SmokeCell] | None = None,
    target_urls: Mapping[tuple[str, str], str] | None = None,
    target_url_selection_audit: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    config = {
        "format_version": "agentic-search-integration-smoke-v2",
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
        "execution_policy": {
            "max_tokens_by_purpose": {
                **{
                    purpose: inputs.resolved_query_max_tokens
                    for purpose in sorted(QUERY_PURPOSES)
                },
                **{
                    purpose: inputs.final_max_tokens
                    for purpose in sorted(ANSWER_PURPOSES)
                },
            },
            "maximum_active_cells": inputs.resolved_cell_concurrency,
            "failed_cell_retry_passes": FAILED_CELL_RETRY_PASSES,
            "ranking_reference_mode": "evidence-id-v1",
            "final_answer_max_characters": FINAL_ANSWER_MAX_CHARACTERS,
            "final_attempt_repair_mode": (
                "evidence-projection-and-malformed-prefix-v2"
            ),
            "structured_output_schema_mode": "xgrammar-structural-v1",
        },
        "request_concurrency": inputs.request_concurrency,
        "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
        "condition_mode": "smoke-only-subset-and-order-v1",
    }
    if inputs.cell_concurrency is not None:
        config["cell_concurrency"] = inputs.cell_concurrency
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
            "format_version": "agentic-search-execution-calibration-v2",
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
        if inputs.prompt_shard_count > 1:
            config.update({
                "prompt_population_count": inputs.prompt_count,
                "prompt_shard_index": inputs.prompt_shard_index,
                "prompt_shard_count": inputs.prompt_shard_count,
            })
        if inputs.production_conditions:
            if target_urls is None:
                raise AssertionError("production conditions lack target URLs")
            target_identity = [
                {
                    "prompt_id": prompt_id,
                    "engine": engine,
                    "target_url": target_url,
                }
                for (prompt_id, engine), target_url in sorted(target_urls.items())
                if prompt_id in {prompt.prompt_id for prompt in prompts}
            ]
            config.update({
                "condition_mode": "frozen-target-url-and-stable-shuffle-v1",
                "ablation_target_source": (
                    "balanced position in each engine's deterministic frozen "
                    "baseline retrieval for the prompt keyword"
                ),
                "ablation_target_selection": (
                    "exact-keyword-else-deterministic-lexical-v1"
                ),
                "ablation_target_selection_counts": {
                    engine: dict(sorted(counts.items()))
                    for engine, counts in sorted(
                        (target_url_selection_audit or {}).items()
                    )
                },
                "target_url_map_sha256": hashlib.sha256(
                    _canonical(target_identity)
                ).hexdigest(),
                "target_url_count": len(target_identity),
            })
        if inputs.cell_ids_jsonl is not None:
            if cells is None:
                raise AssertionError("cell selection lacks selected cells")
            config.update({
                "cell_count": len(cells),
                "selected_prompt_count": len({
                    cell.prompt_id for cell in cells if cell.prompt_id is not None
                }),
                "cell_selection": {
                    "path": str(inputs.cell_ids_jsonl.resolve()),
                    "sha256": _sha256_file(inputs.cell_ids_jsonl),
                    "policy": "explicit-frozen-cell-id-set-v1",
                },
            })
    return config


def _legacy_resume_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    previous = _previous_resume_config(config)
    if (
        previous is None
        or previous.get("model_id") != "Qwen/Qwen3.8-27B"
        or previous.get("disable_thinking") is not True
    ):
        return None
    legacy = json.loads(json.dumps(previous))
    versions = {
        "agentic-search-integration-smoke-v2": "agentic-search-integration-smoke-v1",
        "agentic-search-execution-calibration-v2": (
            "agentic-search-execution-calibration-v1"
        ),
    }
    try:
        legacy["format_version"] = versions[legacy["format_version"]]
    except KeyError as error:
        raise ValueError(
            "unsupported configuration version for legacy resume"
        ) from error
    legacy["git_commit"] = LEGACY_RESUME_COMMIT
    legacy["disable_thinking"] = False
    legacy.pop("execution_policy")
    legacy["max_tokens"] = 1024
    return legacy


def _optimized_resume_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    policy = config.get("execution_policy", {})
    if policy.get("final_attempt_repair_mode") not in {
        "evidence-projection-and-malformed-prefix-v1",
        "evidence-projection-and-malformed-prefix-v2",
    }:
        return None
    previous = json.loads(json.dumps(config))
    previous["git_commit"] = OPTIMIZED_RESUME_COMMIT
    previous["execution_policy"]["final_attempt_repair_mode"] = (
        "evidence-projection-v1"
    )
    return previous


def _previous_resume_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    previous = _evidence_id_resume_config(config)
    if previous is None:
        return None
    previous["git_commit"] = PREVIOUS_RESUME_COMMIT
    previous["execution_policy"].pop("ranking_reference_mode")
    previous["execution_policy"].pop("final_answer_max_characters")
    return previous


def _evidence_id_resume_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    previous = _repair_resume_config(config)
    if previous is None:
        return None
    previous["git_commit"] = EVIDENCE_ID_RESUME_COMMIT
    for purpose in ANSWER_PURPOSES:
        previous["execution_policy"]["max_tokens_by_purpose"][purpose] = 2048
    return previous


def _repair_resume_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    previous = _serial_resume_config(config)
    if previous is None:
        return None
    policy = previous.get("execution_policy", {})
    if (
        policy.get("ranking_reference_mode") != "evidence-id-v1"
        or policy.get("final_answer_max_characters")
        != FINAL_ANSWER_MAX_CHARACTERS
        or policy.get("final_attempt_repair_mode") != "evidence-projection-v1"
    ):
        return None
    previous["git_commit"] = REPAIR_RESUME_COMMIT
    previous["request_concurrency"] = 4
    previous["execution_policy"]["maximum_active_cells"] = 4
    previous["execution_policy"].pop("final_attempt_repair_mode")
    return previous


def _serial_resume_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    optimized = _optimized_resume_config(config)
    source = optimized if optimized is not None else config
    policy = source.get("execution_policy", {})
    if policy.get("structured_output_schema_mode") != "xgrammar-structural-v1":
        return None
    previous = json.loads(json.dumps(source))
    previous["git_commit"] = SERIAL_RESUME_COMMIT
    previous["execution_policy"].pop("structured_output_schema_mode")
    previous.pop("cell_concurrency", None)
    previous["execution_policy"]["maximum_active_cells"] = previous[
        "request_concurrency"
    ]
    return previous


def _prepare_config(
    path: Path,
    config: Mapping[str, Any],
    config_hash: str,
    *,
    shared_mode: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    current = {**config, "config_sha256": config_hash}
    optimized_config = _optimized_resume_config(config)
    optimized_hash = (
        hashlib.sha256(_canonical(optimized_config)).hexdigest()
        if optimized_config
        else None
    )
    legacy = _legacy_resume_config(config)
    legacy_hash = hashlib.sha256(_canonical(legacy)).hexdigest() if legacy else None
    prior_config = _previous_resume_config(config)
    prior_hash = (
        hashlib.sha256(_canonical(prior_config)).hexdigest()
        if prior_config
        else None
    )
    evidence_id_config = _evidence_id_resume_config(config)
    evidence_id_hash = (
        hashlib.sha256(_canonical(evidence_id_config)).hexdigest()
        if evidence_id_config
        else None
    )
    repair_config = _repair_resume_config(config)
    repair_hash = (
        hashlib.sha256(_canonical(repair_config)).hexdigest()
        if repair_config
        else None
    )
    serial_config = _serial_resume_config(config)
    serial_hash = (
        hashlib.sha256(_canonical(serial_config)).hexdigest()
        if serial_config
        else None
    )
    candidates = [
        (
            optimized_config,
            optimized_hash,
            OPTIMIZED_RESUME_COMMIT,
            config["execution_policy"]["max_tokens_by_purpose"][
                "parallel_final"
            ],
            config["disable_thinking"],
        ),
        (
            legacy,
            legacy_hash,
            LEGACY_RESUME_COMMIT,
            1024,
            False,
        ),
        (
            prior_config,
            prior_hash,
            PREVIOUS_RESUME_COMMIT,
            2048,
            config["disable_thinking"],
        ),
        (
            evidence_id_config,
            evidence_id_hash,
            EVIDENCE_ID_RESUME_COMMIT,
            2048,
            config["disable_thinking"],
        ),
        (
            repair_config,
            repair_hash,
            REPAIR_RESUME_COMMIT,
            config["execution_policy"]["max_tokens_by_purpose"][
                "parallel_final"
            ],
            config["disable_thinking"],
        ),
        (
            serial_config,
            serial_hash,
            SERIAL_RESUME_COMMIT,
            config["execution_policy"]["max_tokens_by_purpose"][
                "parallel_final"
            ],
            config["disable_thinking"],
        ),
    ]
    if config["execution_policy"].get("final_attempt_repair_mode") == (
        "evidence-projection-and-malformed-prefix-v2"
    ):
        for source_commit in MALFORMED_PREFIX_V1_RESUME_COMMITS:
            previous = json.loads(json.dumps(config))
            previous["git_commit"] = source_commit
            previous["execution_policy"]["final_attempt_repair_mode"] = (
                "evidence-projection-and-malformed-prefix-v1"
            )
            candidates.append((
                previous,
                hashlib.sha256(_canonical(previous)).hexdigest(),
                source_commit,
                previous["execution_policy"]["max_tokens_by_purpose"][
                    "parallel_final"
                ],
                previous["disable_thinking"],
            ))
    compatible_sources = [
        {
            "config": candidate,
            "config_sha256": candidate_hash,
            "git_commit": source_commit,
            "max_tokens": source_tokens,
            "disable_thinking": source_thinking,
        }
        for (
            candidate,
            candidate_hash,
            source_commit,
            source_tokens,
            source_thinking,
        ) in candidates
        if candidate is not None and candidate_hash is not None
    ]
    if not path.exists():
        _write_json_atomic(path, current)
        return compatible_sources, None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored == current:
        return compatible_sources, None
    if shared_mode:
        # Scheduling-only commits/server relocations may resume the same local
        # backlog. Scientific settings and frozen source hashes remain exact.
        runtime_keys = {
            "git_commit", "slurm_job_id", "base_url", "config_sha256",
            "request_concurrency", "cell_concurrency",
        }

        def scientific_config(value):
            result = {k: v for k, v in value.items() if k not in runtime_keys}
            result["execution_policy"] = {
                k: v for k, v in value["execution_policy"].items() if k != "maximum_active_cells"
            }
            return result

        if (
            scientific_config(stored) == scientific_config(current)
            and stored.get("config_sha256") == hashlib.sha256(_canonical({
                k: v for k, v in stored.items() if k != "config_sha256"
            })).hexdigest()
        ):
            source = {
                "config": {k: v for k, v in stored.items() if k != "config_sha256"},
                "config_sha256": stored["config_sha256"],
                "git_commit": stored.get("git_commit"),
                "max_tokens": stored["execution_policy"]["max_tokens_by_purpose"]["parallel_final"],
                "disable_thinking": stored["disable_thinking"],
            }
            compatible_sources.append(source)
            _write_json_atomic(path, current)
            return compatible_sources, source
    for source in compatible_sources:
        if stored == {
            **source["config"],
            "config_sha256": source["config_sha256"],
        }:
            _write_json_atomic(path, current)
            return compatible_sources, source
    raise ValueError("existing smoke configuration differs")


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


def _generator_claim_identity(
    cell: SmokeCell, inputs: SmokeInputs, config: Mapping[str, Any],
    legacy_prompt: str, target_urls: Mapping[tuple[str, str], str] | None,
    *, method_source_sha256: str,
) -> ClaimIdentity:
    """Per-cell semantics, independent of cohort partition, paths and allocation."""
    policy = {k: v for k, v in config["execution_policy"].items()
              if k not in {"maximum_active_cells", "failed_cell_retry_passes"}}
    request = {
        "cell": cell.core,
        "prompt": cell.prompt if cell.prompt is not None else legacy_prompt,
        "seed": inputs.seed,
        "disable_thinking": inputs.disable_thinking,
        "execution_policy": policy,
        "condition_mode": config["condition_mode"],
        "retrieval_mode": config["retrieval_mode"],
        "snapshot_sha256": config["search_snapshots"][cell.engine]["sha256"],
        "cross_encoder_revision": inputs.cross_encoder_revision,
        "target_url": None if target_urls is None else target_urls[(cell.prompt_id, cell.engine)],
        "agentic_method_source_sha256": method_source_sha256,
    }
    return ClaimIdentity(
        task_id=cell.cell_id, model_id=inputs.model_id,
        model_revision=inputs.model_revision,
        protocol="agentic-generator-shared-v1",
        request_sha256=hashlib.sha256(_canonical(request)).hexdigest(),
    )


def _validate_shared_generator_bundle(
    cell: SmokeCell, prompt: str, value: dict[str, Any], *, failed: bool = False,
) -> None:
    required = {"trace", "diagnostics", "producer"} | ({"error", "attempts"} if failed else {"result"})
    if set(value) != required:
        raise ValueError("shared generator bundle fields differ")
    trace, diagnostic = value["trace"], value["diagnostics"]
    if not isinstance(trace, dict) or not isinstance(diagnostic, dict):
        raise ValueError("shared generator trace and diagnostics must be objects")  # noqa: TRY004
    core = {k: v for k, v in trace.items() if k != "trace_sha256"}
    digest = hashlib.sha256(_canonical(core)).hexdigest()
    if trace.get("trace_sha256") != digest:
        raise ValueError("shared generator trace hash mismatch")
    if (
        trace.get("method_id") != cell.method_class.method_id
        or trace.get("condition") != cell.condition.value
        or trace.get("search_engine") != cell.engine
        or trace.get("user_prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
    ):
        raise ValueError("shared generator trace identity mismatch")
    expected = {"cell_id": cell.cell_id, **cell.core}
    if any(diagnostic.get(k) != v for k, v in expected.items()):
        raise ValueError("shared generator diagnostics identity mismatch")
    if diagnostic.get("status") != ("failed" if failed else "complete"):
        raise ValueError("shared generator diagnostics status mismatch")
    if not isinstance(value["producer"], dict) or not {
        "git_commit", "slurm_job_id", "source_config_sha256",
    } <= value["producer"].keys():
        raise ValueError("shared generator producer provenance missing")
    if failed:
        if value["attempts"] != FAILED_CELL_RETRY_PASSES + 1 or not isinstance(value["error"], str):
            raise ValueError("shared generator bounded failure metadata invalid")
        return
    result = value["result"]
    if not isinstance(result, dict) or "trace" in result:
        raise ValueError("shared generator result must be path-independent")
    if any(result.get(k) != v for k, v in expected.items()) or result.get("trace_sha256") != digest:
        raise ValueError("shared generator result identity mismatch")
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("shared generator answer missing")
    ranking = result.get("ranking")
    if not isinstance(ranking, list) or any(not isinstance(url, str) or not url for url in ranking):
        raise ValueError("shared generator ranking malformed")
    if len(set(ranking)) != len(ranking):
        raise ValueError("shared generator ranking contains duplicates")


def _materialize_shared_generator_bundle(
    output: Path, cell: SmokeCell, bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared commit is authoritative after a crash between these local writes."""
    trace_path = output / "traces" / f"{cell.cell_id}.json"
    result_path = output / "results" / f"{cell.cell_id}.json"
    record = {**bundle["result"], "trace": str(trace_path.resolve())}
    provenance_path = output / "provenance" / f"{cell.cell_id}.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != bundle["producer"]:
            raise ValueError("local producer provenance conflicts with shared commit")
    else:
        _write_json_atomic(provenance_path, bundle["producer"])
    if result_path.exists():
        existing = _validate_completed_cell(cell, trace_path, result_path)
        if existing != record:
            raise ValueError("local completed result conflicts with shared commit")
        return existing
    for path, value in (
        (trace_path, bundle["trace"]),
        (output / "diagnostics" / f"{cell.cell_id}.json", bundle["diagnostics"]),
        (result_path, record),
    ):
        _write_json_atomic(path, value)
    return record


def validate_manifest_artifacts(
    manifest_path: Path, expected_cells: int,
) -> dict[str, Any]:
    """Validate a completed queue or a deadline checkpoint for cluster wrappers."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed_count = manifest.get("completed_count")
    remaining_count = manifest.get("remaining_count")
    if (
        manifest.get("cell_count") != expected_cells
        or manifest.get("scientific_result") is not False
        or type(completed_count) is not int
        or type(remaining_count) is not int
        or min(completed_count, remaining_count) < 0
        or completed_count + remaining_count != expected_cells
    ):
        raise ValueError("generator manifest counts are inconsistent")
    if manifest.get("status") == "checkpointed":
        if remaining_count == 0 or manifest.get("stop_reason") not in {
            "allocation_deadline", "shared_tasks_busy",
        }:
            raise ValueError("generator checkpoint does not record a deadline stop")
    elif manifest.get("status") != "complete" or remaining_count != 0:
        raise ValueError("generator queue did not complete or checkpoint at its deadline")

    output = manifest_path.parent
    result_paths = sorted((output / "results").glob("*.json"))
    result_ids = {path.stem for path in result_paths}
    trace_ids = {path.stem for path in (output / "traces").glob("*.json")}
    if len(result_ids) != completed_count or trace_ids != result_ids:
        raise ValueError("generator manifest counts differ from completed artifacts")
    failed_ids = manifest.get("failed_cell_ids", [])
    if len(set(failed_ids)) > remaining_count or set(failed_ids) & result_ids:
        raise ValueError("generator manifest failed cell IDs are inconsistent")
    methods = {method.method_id: method for method in METHODS}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        cell = SmokeCell(
            cell_id=path.stem,
            engine=result["engine"],
            condition=ExperimentalCondition(result["condition"]),
            method_class=methods[result["method"]],
            prompt_id=result.get("prompt_id"),
            prompt_sha256=result.get("prompt_sha256"),
        )
        _validate_completed_cell(cell, output / "traces" / path.name, path)
    return manifest


async def run_smoke(
    inputs: SmokeInputs,
    *,
    client_context: Any | None = None,
    compactor: ContextCompactor | None = None,
) -> dict[str, Any]:
    if inputs.shared_claim_root is None:
        return await _run_smoke(inputs, client_context=client_context, compactor=compactor)
    # Cell locks protect shared inference; a separate permanent output lock
    # protects this output's config, manifest and local materialized files.
    inputs.output.mkdir(parents=True, exist_ok=True)
    with (inputs.output / ".generator-output.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("generator output already has an active writer") from error
        try:
            return await _run_smoke(inputs, client_context=client_context, compactor=compactor)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


async def _run_smoke(
    inputs: SmokeInputs,
    *,
    client_context: Any | None = None,
    compactor: ContextCompactor | None = None,
) -> dict[str, Any]:
    allocation_budget = AllocationBudget.from_environment()
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
    prompt_population = (
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
    target_url_selection_audit: dict[str, dict[str, int]] = {}
    target_urls = (
        _build_target_urls(
            prompt_population,
            adapters,
            seed=inputs.prompt_selection_seed,
            selection_audit=target_url_selection_audit,
        )
        if prompt_population is not None and inputs.production_conditions
        else None
    )
    prompts = (
        _prompt_shard(
            prompt_population,
            shard_index=inputs.prompt_shard_index,
            shard_count=inputs.prompt_shard_count,
        )
        if prompt_population is not None
        else None
    )
    legacy_prompt = (
        f"Answer the following retrieval question: {keyword}. "
        "For the reactive method, perform at least one search before finishing."
    )
    cells = _select_cells(_cells(prompts), inputs.cell_ids_jsonl)
    config = _config(
        inputs,
        keyword,
        prompts,
        cells,
        target_urls,
        target_url_selection_audit,
    )
    config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    inputs.output.mkdir(parents=True, exist_ok=True)
    config_path = inputs.output / "config.json"
    prior_config = json.loads(config_path.read_text()) if config_path.is_file() else None
    compatible_sources, migrated_source = _prepare_config(
        config_path,
        config,
        config_hash,
        shared_mode=inputs.shared_claim_root is not None,
    )

    completed, pending = _load_completed_cells(inputs.output, cells)
    manifest_path = inputs.output / "run_manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    allowed_manifest_hashes = {config_hash}
    allowed_manifest_hashes.update(
        source["config_sha256"] for source in compatible_sources
    )
    if (
        existing_manifest is not None
        and existing_manifest.get("config_sha256") not in allowed_manifest_hashes
    ):
        raise ValueError("existing smoke manifest has an incompatible configuration")
    resume_migration = (
        existing_manifest.get("resume_migration")
        if existing_manifest is not None
        else None
    )
    resume_source = migrated_source
    if resume_source is None and existing_manifest is not None:
        resume_source = next(
            (
                source
                for source in compatible_sources
                if source["config_sha256"]
                == existing_manifest.get("config_sha256")
            ),
            None,
        )
    if resume_source is not None:
        resume_migration = {
            "source_config_sha256": resume_source["config_sha256"],
            "source_git_commit": resume_source["git_commit"],
            "source_max_tokens": resume_source["max_tokens"],
            "source_disable_thinking": resume_source["disable_thinking"],
            "preserved_completed_count": len(completed),
            "preserved_cell_ids_sha256": hashlib.sha256(
                _canonical(sorted(completed))
            ).hexdigest(),
        }
        if existing_manifest is not None and existing_manifest.get(
            "resume_migration"
        ) is not None:
            resume_migration["prior_resume_migration"] = existing_manifest[
                "resume_migration"
            ]
    claim_store = InferenceClaimStore(inputs.shared_claim_root) if inputs.shared_claim_root else None
    shared_identities = {}
    shared_failed: set[str] = set()
    shared_busy: set[str] = set()
    shared_stats = {
        "root": str(inputs.shared_claim_root.resolve()) if inputs.shared_claim_root else None,
        "worker_index": inputs.worker_index, "worker_count": inputs.worker_count,
        "dispatch_policy": "stable-slot-preference-then-steal-v1",
        "reused_count": 0, "exported_count": 0, "committed_count": 0,
    }

    def validate_bundle(cell: SmokeCell, value: dict[str, Any], *, failed: bool = False) -> None:
        _validate_shared_generator_bundle(
            cell, cell.prompt if cell.prompt is not None else legacy_prompt, value, failed=failed,
        )

    if claim_store is not None:
        method_source_sha256 = _sha256_file(
            REPOSITORY_ROOT / "analysis/interpretability/pipeline/agentic_search.py"
        )
        for cell in cells:
            identity = _generator_claim_identity(
                cell, inputs, config, legacy_prompt, target_urls,
                method_source_sha256=method_source_sha256,
            )
            shared_identities[cell.cell_id] = identity
            with claim_store.try_claim(
                identity, validate=lambda value, cell=cell: validate_bundle(cell, value),
                validate_failure=lambda value, cell=cell: validate_bundle(cell, value, failed=True),
            ) as claim:
                if cell.cell_id in completed:
                    record = completed[cell.cell_id]
                    diagnostic_path = inputs.output / "diagnostics" / f"{cell.cell_id}.json"
                    source = prior_config or {**config, "config_sha256": config_hash}
                    provenance_path = inputs.output / "provenance" / f"{cell.cell_id}.json"
                    bundle = {
                        "result": {k: v for k, v in record.items() if k != "trace"},
                        "trace": json.loads(Path(record["trace"]).read_text()),
                        "diagnostics": json.loads(diagnostic_path.read_text()),
                        "producer": {
                            # A legacy output can mix commits after a migration.
                            # Preserve its source record, do not invent a per-cell
                            # producer or stamp this importing allocation on it.
                            "git_commit": None if resume_migration else source["git_commit"],
                            "slurm_job_id": source.get("slurm_job_id"),
                            "source_config_sha256": source["config_sha256"],
                            "source_config_git_commit": source["git_commit"],
                            "imported_legacy_artifact": True,
                        },
                    }
                    if provenance_path.is_file():
                        bundle["producer"] = json.loads(provenance_path.read_text())
                    validate_bundle(cell, bundle)
                    if claim.status == "owned":
                        claim.commit(bundle)
                        shared_stats["exported_count"] += 1
                        _materialize_shared_generator_bundle(inputs.output, cell, bundle)
                    elif claim.status == "completed":
                        _materialize_shared_generator_bundle(inputs.output, cell, claim.outcome)
                    elif claim.status == "failed":
                        raise ValueError("local completed cell conflicts with terminal shared failure")
                    else:
                        raise RuntimeError("cannot export a completed cell while another worker owns it")
                elif claim.status == "completed":
                    completed[cell.cell_id] = _materialize_shared_generator_bundle(inputs.output, cell, claim.outcome)
                    shared_stats["reused_count"] += 1
                elif claim.status == "failed":
                    shared_failed.add(cell.cell_id)
        pending = [cell for cell in cells if cell.cell_id not in completed]
        pending.sort(key=lambda cell: (
            int(hashlib.sha256(cell.cell_id.encode()).hexdigest(), 16) % inputs.worker_count != inputs.worker_index,
            cell.cell_id,
        ))
    if not pending:
        if existing_manifest is not None and claim_store is None:
            if (
                existing_manifest.get("status") not in {"complete", "checkpointed"}
                or existing_manifest.get("completed_count") != config["cell_count"]
                or existing_manifest.get("remaining_count") != 0
            ):
                raise ValueError("completed smoke manifest is inconsistent")
            if (
                existing_manifest.get("config_sha256") == config_hash
                and existing_manifest.get("status") == "complete"
                and claim_store is None
            ):
                return existing_manifest
        finished = {
            **config,
            "config_sha256": config_hash,
            "status": "complete",
            "stop_reason": "queue_exhausted",
            "allocation_budget": allocation_budget.record(),
            "completed_count": len(completed),
            "remaining_count": 0,
            "compactor_cache_this_invocation": None,
            "peak_active_cells_this_invocation": 0,
            "failed_cell_retry_passes_completed": 0,
        }
        if resume_migration is not None:
            finished["resume_migration"] = resume_migration
        if claim_store is not None:
            finished["shared_backlog"] = shared_stats
        _write_json_atomic(manifest_path, finished)
        return finished

    pending_by_id = {cell.cell_id: cell for cell in pending}
    known_failed = {
        cell_id: pending_by_id[cell_id]
        for cell_id in (existing_manifest or {}).get("failed_cell_ids", [])
        if cell_id in pending_by_id
    }
    known_failed.update({cell_id: pending_by_id[cell_id] for cell_id in shared_failed})
    peak_active_cells = 0
    deadline_reached = False

    def checkpoint(
        retry_passes_completed: int,
        *,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        value = {
            **config,
            "config_sha256": config_hash,
            "status": "checkpointed",
            "stop_reason": stop_reason,
            "allocation_budget": allocation_budget.record(),
            "completed_count": len(completed),
            "remaining_count": config["cell_count"] - len(completed),
            "failed_cell_ids": sorted(known_failed),
            "failed_cell_retry_passes_completed": retry_passes_completed,
            "peak_active_cells_this_invocation": peak_active_cells,
            "compactor_cache_this_invocation": (
                _cache_metrics(compactor) if compactor is not None else None
            ),
        }
        if resume_migration is not None:
            value["resume_migration"] = resume_migration
        if claim_store is not None:
            value["shared_backlog"] = {**shared_stats, "busy_cell_ids": sorted(shared_busy)}
        _write_json_atomic(manifest_path, value)
        return value

    if not allocation_budget.can_start():
        return checkpoint(0, stop_reason="allocation_deadline")

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
    if not allocation_budget.can_start():
        return checkpoint(0, stop_reason="allocation_deadline")
    async with AsyncExitStack() as exit_stack:
        try:
            client = await asyncio.wait_for(
                exit_stack.enter_async_context(client_context),
                timeout=allocation_budget.work_seconds_left(),
            )
        except asyncio.TimeoutError:
            if allocation_budget.work_seconds_left() == 0:
                return checkpoint(0, stop_reason="allocation_deadline")
            raise
        request_semaphore = asyncio.Semaphore(inputs.request_concurrency)

        async def execute_once(
            cell: SmokeCell, claim: TaskClaim | None = None,
        ) -> tuple[str, dict[str, Any]]:
            cell_started = time.perf_counter()
            trace_path = inputs.output / "traces" / f"{cell.cell_id}.json"
            result_path = inputs.output / "results" / f"{cell.cell_id}.json"
            diagnostics_path = inputs.output / "diagnostics" / f"{cell.cell_id}.json"
            generator = VllmAgentGenerator(
                client,
                seed=inputs.seed + int(cell.cell_id[:8], 16),
                query_max_tokens=inputs.resolved_query_max_tokens,
                final_max_tokens=inputs.final_max_tokens,
                request_semaphore=request_semaphore,
            )
            condition_hook: SmokeConditionHook | TargetUrlConditionHook
            if inputs.production_conditions:
                if cell.prompt_id is None or target_urls is None:
                    raise AssertionError("production cell lacks target identity")
                condition_hook = TargetUrlConditionHook(
                    target_url=target_urls[(cell.prompt_id, cell.engine)],
                    seed=inputs.seed + int(cell.cell_id[:8], 16),
                )
            else:
                condition_hook = SmokeConditionHook(inputs.seed)
            method = cell.method_class(
                llm=generator,
                search=adapters[cell.engine],
                compactor=compactor,
                condition_hook=condition_hook,
            )
            try:
                result = await method.run(
                    cell.prompt if cell.prompt is not None else legacy_prompt,
                    cell.condition,
                )
                search_count = sum(
                    event.event_type == "search" for event in result.trace.events
                )
                expected_searches = (
                    3 if cell.method_class is ParallelExpansionV1 else None
                )
                if expected_searches is not None and search_count != expected_searches:
                    raise AgentExecutionError(
                        f"{cell.method_class.method_id} made {search_count} searches, "
                        f"expected {expected_searches}",
                        result.trace,
                    )
                if cell.method_class is ReactiveSnippetLoopV1 and search_count < 1:
                    raise AgentExecutionError(
                        "Reactive-Snippet-Loop-v1 did not exercise retrieval",
                        result.trace,
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
            trace_value = result.trace.to_dict()
            trace_hash = trace_value["trace_sha256"]
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
            if isinstance(condition_hook, TargetUrlConditionHook):
                record["condition_audit"] = {
                    "target_url": condition_hook.target_url,
                    "calls": condition_hook.calls,
                    "target_observed": any(
                        call["target_count_before"] > 0
                        for call in condition_hook.calls
                    ),
                    "target_removed_count": sum(
                        call["target_removed_count"]
                        for call in condition_hook.calls
                    ),
                }
            diagnostic = {
                "cell_id": cell.cell_id,
                **cell.core,
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - cell_started,
                "llm_calls": generator.diagnostics,
            }
            if claim is not None:
                claim.commit({
                    "result": {k: v for k, v in record.items() if k != "trace"},
                    "trace": trace_value, "diagnostics": diagnostic,
                    "producer": {
                        "git_commit": config["git_commit"],
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "source_config_sha256": config_hash,
                    },
                })
                shared_stats["committed_count"] += 1
                return cell.cell_id, _materialize_shared_generator_bundle(inputs.output, cell, claim.outcome)
            write_trace_atomic(trace_path, result.trace)
            _write_json_atomic(diagnostics_path, diagnostic)
            _write_json_atomic(result_path, record)
            return cell.cell_id, record

        class SharedFailure(RuntimeError):
            pass

        async def execute(cell: SmokeCell) -> tuple[str, dict[str, Any] | None]:
            if claim_store is None:
                return await execute_once(cell)
            with claim_store.try_claim(
                shared_identities[cell.cell_id],
                validate=lambda value: validate_bundle(cell, value),
                validate_failure=lambda value: validate_bundle(cell, value, failed=True),
            ) as claim:
                if claim.status == "busy":
                    shared_busy.add(cell.cell_id)
                    return cell.cell_id, None
                shared_busy.discard(cell.cell_id)
                if claim.status == "completed":
                    record = _materialize_shared_generator_bundle(inputs.output, cell, claim.outcome)
                    shared_stats["reused_count"] += 1
                    return cell.cell_id, record
                if claim.status == "failed":
                    raise SharedFailure("cell has a durable bounded failure")
                for attempt in range(FAILED_CELL_RETRY_PASSES + 1):
                    if not allocation_budget.can_start():
                        shared_busy.add(cell.cell_id)
                        return cell.cell_id, None
                    try:
                        return await execute_once(cell, claim)
                    except AgentExecutionError as error:
                        if attempt == FAILED_CELL_RETRY_PASSES:
                            claim.fail({
                                "trace": error.trace.to_dict(),
                                "diagnostics": json.loads((
                                    inputs.output / "diagnostics" / f"{cell.cell_id}.json"
                                ).read_text()),
                                "error": str(error), "attempts": attempt + 1,
                                "producer": {
                                    "git_commit": config["git_commit"],
                                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                    "source_config_sha256": config_hash,
                                },
                            })
                            raise
            raise AssertionError("shared generator attempt loop did not terminate")

        async def run_pass(
            cells_for_pass: Sequence[SmokeCell],
            *,
            retry_passes_completed: int,
        ) -> list[SmokeCell]:
            nonlocal peak_active_cells, deadline_reached
            next_cell_index = 0
            active: dict[asyncio.Task[tuple[str, dict[str, Any] | None]], SmokeCell] = {}
            failed: list[SmokeCell] = []

            def fill() -> None:
                nonlocal peak_active_cells, next_cell_index, deadline_reached
                while (
                    len(active) < inputs.resolved_cell_concurrency
                    and next_cell_index < len(cells_for_pass)
                ):
                    if not allocation_budget.can_start():
                        deadline_reached = True
                        break
                    cell = cells_for_pass[next_cell_index]
                    next_cell_index += 1
                    active[asyncio.create_task(execute(cell))] = cell
                peak_active_cells = max(peak_active_cells, len(active))

            fill()
            try:
                while active:
                    done, _ = await asyncio.wait(
                        active,
                        timeout=allocation_budget.work_seconds_left(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        deadline_reached = True
                        for task in active:
                            task.cancel()
                        await asyncio.gather(*active, return_exceptions=True)
                        break
                    for task in done:
                        cell = active.pop(task)
                        try:
                            cell_id, record = task.result()
                        except AgentExecutionError:
                            failed.append(cell)
                            known_failed[cell.cell_id] = cell
                        except SharedFailure:
                            known_failed[cell.cell_id] = cell
                        else:
                            if record is not None:
                                completed[cell_id] = record
                                known_failed.pop(cell_id, None)
                        checkpoint(retry_passes_completed)
                    fill()
            except BaseException:
                for task in active:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                raise
            order = {cell.cell_id: index for index, cell in enumerate(cells_for_pass)}
            failed.sort(key=lambda cell: order[cell.cell_id])
            return failed

        failed = await run_pass(pending, retry_passes_completed=0)
        retry_passes_completed = 0
        for retry_pass in range(1, (FAILED_CELL_RETRY_PASSES + 1) if claim_store is None else 1):
            if not failed:
                break
            if deadline_reached or not allocation_budget.can_start():
                deadline_reached = True
                break
            retry_passes_completed = retry_pass
            failed = await run_pass(
                failed,
                retry_passes_completed=retry_pass,
            )
        if claim_store is not None and shared_busy and allocation_budget.can_start():
            # A peer may have finished or crashed while we consumed other cells.
            # Revisit busy work once; if only owned work remains, leave a clear
            # checkpoint instead of claiming completion or idling out the job.
            await run_pass(
                [cell for cell in pending if cell.cell_id in shared_busy],
                retry_passes_completed=0,
            )
    remaining_count = config["cell_count"] - len(completed)
    stopped_at_deadline = (deadline_reached or not allocation_budget.can_start()) and remaining_count > 0
    stopped_busy = bool(shared_busy) and remaining_count > 0 and not stopped_at_deadline
    manifest = {
        **config,
        "config_sha256": config_hash,
        "status": (
            "checkpointed" if stopped_at_deadline or stopped_busy
            else "complete_with_failures" if known_failed
            else "complete"
        ),
        "stop_reason": (
            "allocation_deadline" if stopped_at_deadline
            else "shared_tasks_busy" if stopped_busy
            else "bounded_failures" if known_failed
            else "queue_exhausted"
        ),
        "allocation_budget": allocation_budget.record(),
        "completed_count": len(completed),
        "remaining_count": remaining_count,
        "failed_cell_ids": sorted(known_failed),
        "failed_cell_retry_passes_completed": retry_passes_completed,
        "peak_active_cells_this_invocation": peak_active_cells,
        "compactor_cache_this_invocation": _cache_metrics(compactor),
    }
    if resume_migration is not None:
        manifest["resume_migration"] = resume_migration
    if claim_store is not None:
        manifest["shared_backlog"] = {**shared_stats, "busy_cell_ids": sorted(shared_busy)}
    _write_json_atomic(inputs.output / "run_manifest.json", manifest)
    if known_failed and not stopped_at_deadline:
        raise RuntimeError(
            f"{len(known_failed)} agentic-search cells failed after bounded retry"
        )
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
    parser.add_argument("--query-max-tokens", type=int)
    parser.add_argument("--final-max-tokens", type=int, default=2048)
    parser.add_argument("--request-concurrency", type=int, default=1)
    parser.add_argument(
        "--cell-concurrency",
        type=int,
        help=(
            "Maximum active agent cells. Defaults to request concurrency; "
            "raising it overlaps retrieval and validation without increasing "
            "simultaneous vLLM requests."
        ),
    )
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--selection-records-jsonl", type=Path)
    parser.add_argument("--cell-ids-jsonl", type=Path)
    parser.add_argument("--shared-claim-root", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--prompt-selection-seed", type=int, default=20260912)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--prompt-shard-count", type=int, default=1)
    parser.add_argument(
        "--production-conditions",
        action="store_true",
        help=(
            "Use exact frozen baseline target-URL ablation and stable "
            "prompt-specific shuffling instead of smoke transformations."
        ),
    )
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
        query_max_tokens=arguments.query_max_tokens,
        final_max_tokens=arguments.final_max_tokens,
        request_concurrency=arguments.request_concurrency,
        cell_concurrency=arguments.cell_concurrency,
        disable_thinking=arguments.disable_thinking,
        prompts_jsonl=arguments.prompts_jsonl,
        selection_records_jsonl=arguments.selection_records_jsonl,
        cell_ids_jsonl=arguments.cell_ids_jsonl,
        prompt_count=arguments.prompt_count,
        prompt_selection_seed=arguments.prompt_selection_seed,
        prompt_shard_index=arguments.prompt_shard_index,
        prompt_shard_count=arguments.prompt_shard_count,
        production_conditions=arguments.production_conditions,
        shared_claim_root=arguments.shared_claim_root,
        worker_index=arguments.worker_index,
        worker_count=arguments.worker_count,
    )
    manifest = asyncio.run(run_smoke(inputs))
    print("AGENTIC_INTEGRATION_SMOKE=" + json.dumps({
        key: manifest[key]
        for key in ("status", "cell_count", "completed_count", "remaining_count")
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
