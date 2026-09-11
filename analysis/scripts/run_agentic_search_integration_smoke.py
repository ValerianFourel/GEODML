#!/usr/bin/env python3
"""Run a resumable, non-scientific agentic-search integration smoke."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_search import (  # noqa: E402
    AgentExecutionError,
    ContextCompactor,
    ExperimentalCondition,
    LLMRequest,
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


class FrozenSnapshotSearchAdapter:
    """Deterministic lexical retrieval over one immutable search snapshot."""

    def __init__(self, engine: str, path: Path) -> None:
        if engine not in ENGINES:
            raise ValueError(f"unsupported search engine: {engine}")
        self.engine = engine
        self.path = path.resolve()
        self.snapshot_sha256 = _sha256_file(self.path)
        self.rows = _read_snapshot(self.path)

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
    def __init__(self, client: VllmChatClient, *, seed: int, max_tokens: int) -> None:
        self.client = client
        self.seed = seed
        self.max_tokens = max_tokens
        self.call_index = 0

    async def generate(self, request: LLMRequest) -> str:
        self.call_index += 1
        raw, _ = await self.client.complete(
            prompt=request.prompt,
            schema_name=f"agentic_{request.purpose}_{self.call_index}",
            schema=request.response_schema,
            temperature=0.0,
            max_tokens=self.max_tokens,
            seed=self.seed + self.call_index,
        )
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


def _config(inputs: SmokeInputs, keyword: str) -> dict[str, Any]:
    return {
        "format_version": "agentic-search-integration-smoke-v1",
        "scientific_result": False,
        "git_commit": os.environ.get("GEODML_EXECUTION_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_id": inputs.model_id,
        "model_revision": inputs.model_revision,
        "base_url": inputs.base_url,
        "cross_encoder_snapshot": str(inputs.cross_encoder_snapshot.resolve()),
        "cross_encoder_revision": inputs.cross_encoder_revision,
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
        "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
        "condition_mode": "smoke-only-subset-and-order-v1",
    }


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
    prompt = (
        f"Answer the following retrieval question: {keyword}. "
        "For the reactive method, perform at least one search before finishing."
    )
    config = _config(inputs, keyword)
    config_hash = hashlib.sha256(_canonical(config)).hexdigest()
    inputs.output.mkdir(parents=True, exist_ok=True)
    config_path = inputs.output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != {**config, "config_sha256": config_hash}:
            raise ValueError("existing smoke configuration differs")
    else:
        _write_json_atomic(config_path, {**config, "config_sha256": config_hash})

    if compactor is None:
        scorer = SentenceTransformersCrossEncoderScorer(
            str(inputs.cross_encoder_snapshot),
            model_revision=None,
            device="cpu",
            batch_size=32,
            local_files_only=True,
        )
        compactor = ContextCompactor(scorer)
    completed: list[dict[str, Any]] = []
    if client_context is None:
        client_context = VllmChatClient(
            base_url=inputs.base_url,
            api_key=None,
            server_model_name=inputs.model_id,
            timeout_seconds=300.0,
            maximum_attempts=2,
        )
    async with client_context as client:
        for engine in ENGINES:
            for condition in CONDITIONS:
                for method_class in METHODS:
                    cell_core = {
                        "method": method_class.method_id,
                        "engine": engine,
                        "condition": condition.value,
                    }
                    cell_id = hashlib.sha256(_canonical(cell_core)).hexdigest()[:20]
                    trace_path = inputs.output / "traces" / f"{cell_id}.json"
                    result_path = inputs.output / "results" / f"{cell_id}.json"
                    if result_path.is_file() and trace_path.is_file():
                        completed.append(json.loads(result_path.read_text(encoding="utf-8")))
                        continue
                    generator = VllmAgentGenerator(
                        client,
                        seed=inputs.seed + int(cell_id[:8], 16),
                        max_tokens=inputs.max_tokens,
                    )
                    method = method_class(
                        llm=generator,
                        search=adapters[engine],
                        compactor=compactor,
                        condition_hook=SmokeConditionHook(inputs.seed),
                    )
                    try:
                        result = await method.run(prompt, condition)
                    except AgentExecutionError as error:
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        write_trace_atomic(trace_path, error.trace)
                        raise
                    search_count = sum(
                        event.event_type == "search" for event in result.trace.events
                    )
                    expected_searches = (
                        3 if method_class is ParallelExpansionV1 else None
                    )
                    if expected_searches is not None and search_count != expected_searches:
                        raise RuntimeError(
                            f"{method_class.method_id} made {search_count} searches, "
                            f"expected {expected_searches}"
                        )
                    if method_class is ReactiveSnippetLoopV1 and search_count < 1:
                        raise RuntimeError(
                            "Reactive-Snippet-Loop-v1 did not exercise retrieval"
                        )
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_hash = write_trace_atomic(trace_path, result.trace)
                    record = {
                        "cell_id": cell_id,
                        **cell_core,
                        "ranking": list(result.ranking),
                        "answer": result.answer,
                        "final_snippet_count": len(result.final_snippets),
                        "search_count": search_count,
                        "trace": str(trace_path.resolve()),
                        "trace_sha256": trace_hash,
                    }
                    _write_json_atomic(result_path, record)
                    completed.append(record)
                    _write_json_atomic(inputs.output / "run_manifest.json", {
                        **config,
                        "config_sha256": config_hash,
                        "status": "checkpointed",
                        "completed_count": len(completed),
                        "remaining_count": config["cell_count"] - len(completed),
                    })
    manifest = {
        **config,
        "config_sha256": config_hash,
        "status": "complete",
        "completed_count": len(completed),
        "remaining_count": 0,
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
    )
    manifest = asyncio.run(run_smoke(inputs))
    print("AGENTIC_INTEGRATION_SMOKE=" + json.dumps({
        key: manifest[key]
        for key in ("status", "cell_count", "completed_count", "remaining_count")
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
