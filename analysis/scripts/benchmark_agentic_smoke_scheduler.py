#!/usr/bin/env python3
"""Measure bounded agentic smoke scheduling without model or network access."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_search import (  # noqa: E402
    ContextCompactor,
    LLMRequest,
    MemoizingSnippetScorer,
    Snippet,
)
from analysis.scripts.run_agentic_search_integration_smoke import (  # noqa: E402
    SmokeInputs,
    run_smoke,
)


class DelayedClient:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.active_calls = 0
        self.peak_active_calls = 0
        self.call_count = 0

    async def __aenter__(self) -> "DelayedClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def complete(
        self,
        *,
        prompt: str,
        schema_name: str,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int]]:
        del kwargs
        self.call_count += 1
        self.active_calls += 1
        self.peak_active_calls = max(self.peak_active_calls, self.active_calls)
        try:
            await asyncio.sleep(self.delay_seconds)
            return json.dumps(_response(prompt, schema_name)), {"total_tokens": 1}
        finally:
            self.active_calls -= 1


class CountingScorer:
    model_id = "benchmark-counting-scorer"
    model_revision = "v1"
    deterministic = True

    def __init__(self) -> None:
        self.snippet_count = 0

    def score(self, query: str, snippets: list[Snippet]) -> list[float]:
        self.snippet_count += len(snippets)
        terms = set(query.casefold().split())
        return [
            float(len(terms.intersection(f"{row.title} {row.text}".casefold().split())))
            for row in snippets
        ]


def _response(prompt: str, schema_name: str) -> dict[str, Any]:
    if "parallel_query_expansion" in schema_name:
        return {"queries": ["Berlin people", "Berlin census", "Berlin residents"]}
    if "parallel_final" in schema_name:
        return {
            "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
            "answer": "Parallel answer",
        }
    if "reactive_action" in schema_name and "OBSERVATIONS:\n[]" in prompt:
        return {"action": "search", "query": "Berlin population"}
    if "reactive_action" in schema_name:
        return {
            "action": "finish",
            "ranking": [re.findall(r"https://[^\" ]+", prompt)[0]],
            "answer": "Reactive answer",
        }
    raise AssertionError(schema_name)


def _snapshots(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Path] = {}
    for engine in ("duckduckgo", "searxng"):
        path = root / f"{engine}.jsonl"
        path.write_text("".join(
            json.dumps({
                "keyword": "Berlin population",
                "position": index + 1,
                "title": f"Berlin source {index}",
                "url": f"https://{engine}.test/{index}",
                "snippet": f"Berlin population evidence {index}",
            }) + "\n"
            for index in range(20)
        ), encoding="utf-8")
        snapshots[engine] = path
    return snapshots


async def _run(
    root: Path,
    delay_seconds: float,
    concurrency: int,
    snapshots: dict[str, Path],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cross_encoder = root / ("e" * 40)
    cross_encoder.mkdir(exist_ok=True)
    client = DelayedClient(delay_seconds)
    underlying = CountingScorer()
    scorer = MemoizingSnippetScorer(underlying)
    inputs = SmokeInputs(
        output=root / f"output-{concurrency}",
        base_url="http://127.0.0.1:8010/v1",
        model_id="benchmark/model",
        model_revision="a" * 40,
        cross_encoder_snapshot=cross_encoder,
        cross_encoder_revision="e" * 40,
        search_snapshots=snapshots,
        seed=20260912,
        max_tokens=128,
        request_concurrency=concurrency,
    )
    started = time.perf_counter()
    manifest = await run_smoke(
        inputs,
        client_context=client,
        compactor=ContextCompactor(scorer),
    )
    elapsed = time.perf_counter() - started
    logical_records = []
    for path in sorted((inputs.output / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("trace")
        logical_records.append(record)
    logical_result_sha256 = hashlib.sha256(json.dumps(
        logical_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    trace_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((inputs.output / "traces").glob("*.json"))
    ]
    logical_trace_sha256 = hashlib.sha256(json.dumps(
        trace_payloads,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "concurrency": concurrency,
        "elapsed_seconds": round(elapsed, 6),
        "cell_count": manifest["completed_count"],
        "llm_calls": client.call_count,
        "peak_active_llm_calls": client.peak_active_calls,
        "reranker_input_snippets": scorer.stats.hits + scorer.stats.misses,
        "reranker_scored_snippets": underlying.snippet_count,
        "reranker_cache_hits": scorer.stats.hits,
        "logical_result_sha256": logical_result_sha256,
        "logical_trace_sha256": logical_trace_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-delay-seconds", type=float, default=0.02)
    arguments = parser.parse_args()
    if arguments.llm_delay_seconds < 0:
        parser.error("llm delay must be nonnegative")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snapshots = _snapshots(root / "snapshots")
        serial = asyncio.run(
            _run(root / "serial", arguments.llm_delay_seconds, 1, snapshots)
        )
        concurrent = asyncio.run(
            _run(root / "concurrent", arguments.llm_delay_seconds, 4, snapshots)
        )
    if (
        serial["llm_calls"] != concurrent["llm_calls"]
        or serial["cell_count"] != 12
        or serial["logical_result_sha256"] != concurrent["logical_result_sha256"]
        or serial["logical_trace_sha256"] != concurrent["logical_trace_sha256"]
    ):
        raise RuntimeError("scheduler benchmark changed the logical workload")
    result = {
        "format_version": "agentic-smoke-scheduler-benchmark-v1",
        "scientific_result": False,
        "serial": serial,
        "concurrent": concurrent,
        "speedup": round(
            serial["elapsed_seconds"] / concurrent["elapsed_seconds"], 3
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
