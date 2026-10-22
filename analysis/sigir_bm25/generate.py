#!/usr/bin/env python3
"""Generation rerun with BM25 (or the original lexical search), natural condition, resumable.

  generate.py run    --bundle B --run-dir R --model {qwen38,llama4} --retriever {bm25,orig} --snapshot ENGINE=PATH ...
  generate.py status --run-dir R --bundle B --model M

Each cell runs the unchanged pipeline methods (ParallelExpansionV1, ReactiveSnippetLoopV1), the original
VllmAgentGenerator/VllmChatClient (seed 20260911 + int(cell_id[:8], 16)) and the original cross-encoder setup; only the
search adapter changes. Parallel Expansion replays the cell's three logged queries instead of asking the generator
again (the generator's call counter still advances, so the final call keeps the original seed); the Reactive Loop
runs live from the prompt, since its later queries depend on what the search returns.

Output: run-dir/parts/part-<host>-<job|pid>-<utc>.jsonl, one fsync'd line per finished cell (status complete or
failed; the result fields of the original controller, the inline trace and diagnostics). A rerun with the same
run-dir skips finished cells and ignores a torn last line. A validation failure is retried once, then recorded;
transport failures are never recorded as results, and --breaker consecutive ones stop the run.
Exit codes: 0 all cells done, 4 deadline or stop file (rerun to continue), 3 breaker, 2 refused, 1 error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd

from analysis.interpretability.pipeline.agentic_search import (
    AgentExecutionError,
    ContextCompactor,
    ExperimentalCondition,
    IdentityConditionHook,
    LexicalOverlapScorer,
    LLMRequest,
    MemoizingSnippetScorer,
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
    SentenceTransformersCrossEncoderScorer,
)
from analysis.interpretability.pipeline.inference_budget import AllocationBudget
from analysis.scripts.run_acl_arr_vllm import VllmChatClient
from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter, VllmAgentGenerator
from analysis.sigir_bm25.bm25 import Bm25SnapshotSearchAdapter
from analysis.sigir_bm25.reranker import cuda_pool
from analysis.sigir_bm25.score_cache import CachedScorer, load_cache

FORMAT = "sigir-bm25-generation-v1"
SEED = 20260911
METHODS = {ParallelExpansionV1.method_id: ParallelExpansionV1, ReactiveSnippetLoopV1.method_id: ReactiveSnippetLoopV1}
# Infrastructure failures (the methods' own messages): never recorded as results, the cell runs again later.
INFRA = frozenset(("LLM transport failed", "context compaction failed", "parallel search failed",
                   "reactive search failed"))


class ReplayQueriesLLM:
    """Answers the Parallel query-expansion request with the cell's logged queries; everything else goes to inner."""

    def __init__(self, inner, generator: VllmAgentGenerator, queries: list[str] | None) -> None:
        self.inner, self.generator, self.queries = inner, generator, queries

    async def generate(self, request: LLMRequest) -> str:
        if request.purpose == "parallel_query_expansion" and self.queries is not None:
            self.generator.call_index += 1  # the original's query call; the final call keeps its seed
            self.generator.diagnostics.append({"call_index": self.generator.call_index, "purpose": request.purpose,
                                               "replayed": True, "error": None, "usage": None})
            return json.dumps({"queries": list(self.queries)}, ensure_ascii=False)
        return await self.inner.generate(request)


class TransportRetryLLM:
    """Retries a failed request (HTTP error, timeout, connection) with backoff before the method sees the error."""

    def __init__(self, inner, retries: int, base_delay: float = 5.0) -> None:
        self.inner, self.retries, self.base_delay = inner, retries, base_delay
        self.retried = 0

    async def generate(self, request: LLMRequest) -> str:
        for attempt in range(self.retries + 1):
            try:
                return await self.inner.generate(request)
            except Exception:
                if attempt == self.retries:
                    raise
                self.retried += 1
                await asyncio.sleep(min(300.0, self.base_delay * 3 ** attempt))
        raise AssertionError("unreachable")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def packages() -> dict[str, str | None]:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "sentence-transformers", "aiohttp", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def load_cells(bundle: Path, model: str) -> tuple[pd.DataFrame, dict[str, str], dict]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    cells = pd.read_parquet(bundle / f"cells-{model}.parquet").sort_values("order").reset_index(drop=True)
    prompts = pd.read_parquet(bundle / "prompts.parquet")
    return cells, dict(zip(prompts["prompt_id"], prompts["text"])), manifest


def finished(run_dir: Path) -> dict[str, str]:
    """cell_id -> status over every part file; a torn line (an interrupted write) is ignored."""
    out: dict[str, str] = {}
    for part in sorted((run_dir / "parts").glob("part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("format_version") == FORMAT and rec.get("status") in ("complete", "failed"):
                out[rec["cell_id"]] = rec["status"]
    return out


def scientific_config(a, manifest: dict, adapters: dict) -> dict:
    return {"format_version": FORMAT, "model": a.model, "retriever": a.retriever, "served_model": a.served_model,
            "base_url": a.base_url, "seed": SEED, "query_max_tokens": a.query_max_tokens,
            "final_max_tokens": a.final_max_tokens, "disable_thinking": a.disable_thinking,
            "condition": "natural", "bundle_sha256": manifest.get("cells_sha256", {}).get(a.model),
            "snapshots": {e: ad.snapshot_sha256 for e, ad in sorted(adapters.items())},
            "bm25_fingerprints": {e: getattr(getattr(ad, "index", None), "fingerprint", None)
                                  for e, ad in sorted(adapters.items())},
            "cross_encoder": a.cross_encoder or "lexical-overlap (test only)",
            "score_cache": [Path(p).name for p in a.score_cache]}


def make_adapters(a) -> dict:
    snapshots = dict(s.split("=", 1) for s in a.snapshot)
    cls = Bm25SnapshotSearchAdapter if a.retriever == "bm25" else FrozenSnapshotSearchAdapter
    return {e: cls(e, Path(p)) for e, p in snapshots.items()}


def make_scorer(a):
    if a.cross_encoder and a.device == "cuda-pool":
        base = cuda_pool(a.cross_encoder)
    elif a.cross_encoder:
        base = SentenceTransformersCrossEncoderScorer(a.cross_encoder, model_revision=None, device=a.device,
                                                      batch_size=32, local_files_only=True)
    else:
        base = LexicalOverlapScorer()
    if a.score_cache:
        return CachedScorer(base, load_cache(Path(p) for p in a.score_cache))
    return base


def make_client(a) -> VllmChatClient:
    loopback = a.base_url.startswith(("http://127.0.0.1:", "http://localhost:"))
    if a.api_key_env:
        if os.environ.get("GEODML_INFERENCE_AUTH_REQUIRED") == "1":
            raise SystemExit("refused: --api-key-env is for an external API, not the managed local vLLM server")
        if not a.base_url.startswith("https://"):
            raise SystemExit("refused: an external API must use https")
        key = os.environ.get(a.api_key_env, "")
        if not key:
            raise SystemExit(f"refused: environment variable {a.api_key_env} is empty")
    elif not loopback:
        raise SystemExit("refused: a non-local base URL needs --api-key-env")
    else:
        key = None  # the managed vLLM stage passes VLLM_API_KEY in the environment
    return VllmChatClient(base_url=a.base_url, api_key=key, server_model_name=a.served_model,
                          timeout_seconds=a.timeout, maximum_attempts=2,
                          chat_template_kwargs={"enable_thinking": False} if a.disable_thinking else None)


async def run_cells(a, cells: pd.DataFrame, prompts: dict[str, str], adapters: dict, scorer, client,
                    budget: AllocationBudget, part: Path) -> int:
    request_semaphore = asyncio.Semaphore(a.request_concurrency)
    producer = {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(),
                **git_commit()}
    stats: Counter[str] = Counter()
    streak = {"transport": 0}
    stop = {"reason": None}
    queue = list(cells.itertuples(index=False))
    queue.reverse()
    out = part.open("x", encoding="utf-8")

    def write(record: dict) -> None:
        out.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())

    async def attempt(cell) -> tuple[str, dict]:
        generator = VllmAgentGenerator(client, seed=SEED + int(cell.cell_id[:8], 16), query_max_tokens=a.query_max_tokens,
                                       final_max_tokens=a.final_max_tokens, request_semaphore=request_semaphore)
        retry = TransportRetryLLM(generator, a.transport_retries)
        queries = list(cell.queries) if cell.queries is not None and len(cell.queries) else None
        llm = ReplayQueriesLLM(retry, generator, queries if cell.method == ParallelExpansionV1.method_id else None)
        method = METHODS[cell.method](llm=llm, search=adapters[cell.engine],
                                      compactor=ContextCompactor(MemoizingSnippetScorer(scorer)),
                                      condition_hook=IdentityConditionHook())
        started = time.perf_counter()
        identity = {"format_version": FORMAT, "cell_id": cell.cell_id, "answer_id": cell.answer_id, "model": a.model,
                    "retriever": a.retriever, "method": cell.method, "engine": cell.engine, "condition": "natural",
                    "prompt_id": cell.prompt_id, "prompt_sha256": cell.prompt_sha256}
        diagnostics = {"queries_source": "logged" if queries and cell.method == ParallelExpansionV1.method_id
                       else "generated", "llm_calls": generator.diagnostics}
        try:
            result = await method.run(prompts[cell.prompt_id], ExperimentalCondition.NATURAL)
            searches = sum(e.event_type == "search" for e in result.trace.events)
            if cell.method == ParallelExpansionV1.method_id and searches != 3:
                raise AgentExecutionError(f"{cell.method} made {searches} searches, expected 3", result.trace)
            if cell.method == ReactiveSnippetLoopV1.method_id and searches < 1:
                raise AgentExecutionError(f"{cell.method} did not exercise retrieval", result.trace)
        except AgentExecutionError as error:
            trace = error.trace.to_dict()
            kind = "transport" if str(error) in INFRA else "failed"
            return kind, {**identity, "status": "failed", "error": str(error), "trace": trace,
                          "trace_sha256": trace["trace_sha256"],
                          "diagnostics": {**diagnostics, "elapsed_seconds": time.perf_counter() - started,
                                          "transport_retries": retry.retried}, "producer": producer,
                          "finished_at": utc()}
        trace = result.trace.to_dict()
        return "complete", {**identity, "status": "complete", "ranking": list(result.ranking),
                            "answer": result.answer, "final_snippet_count": len(result.final_snippets),
                            "search_count": searches, "trace": trace, "trace_sha256": trace["trace_sha256"],
                            "diagnostics": {**diagnostics, "elapsed_seconds": time.perf_counter() - started,
                                            "transport_retries": retry.retried},
                            "producer": producer, "finished_at": utc()}

    async def worker() -> None:
        while queue and stop["reason"] is None:
            reason = budget.admission_stop_reason()
            if reason is not None:
                stop["reason"] = stop["reason"] or reason
                return
            cell = queue.pop()
            kind, record = await attempt(cell)
            if kind == "failed":  # one more pass, as the original controller's failed-cell retry
                kind, record = await attempt(cell)
            if kind == "transport":
                stats["transport"] += 1
                streak["transport"] += 1
                if streak["transport"] >= a.breaker:
                    stop["reason"] = "breaker"
                continue
            streak["transport"] = 0
            stats[kind] += 1
            write(record)

    tasks = [asyncio.create_task(worker()) for _ in range(a.cell_concurrency)]
    left = budget.work_seconds_left()
    done, pending = await asyncio.wait(tasks, timeout=left)
    for t in pending:
        t.cancel()
    for t in done:
        t.result()
    out.close()
    if pending:
        stop["reason"] = stop["reason"] or "allocation_deadline"
    summary = {"finished_at": utc(), "written": dict(stats), "stop_reason": stop["reason"],
               "remaining_in_queue": len(queue), "score_cache": dict(getattr(scorer, "counts", {})),
               "reranker_device": a.device,
               "reranker_calls_by_device": dict(getattr(getattr(scorer, "base", scorer), "counts", {})),
               "packages": packages()}
    (part.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if stop["reason"] == "breaker":
        return 3
    if stop["reason"] is not None or queue or pending:
        return 4
    return 0


def cmd_run(a) -> int:
    cells, prompts, manifest = load_cells(Path(a.bundle), a.model)
    run_dir = Path(a.run_dir)
    (run_dir / "parts").mkdir(parents=True, exist_ok=True)
    adapters = make_adapters(a)
    if set(adapters) != set(cells["engine"]):
        raise SystemExit(f"refused: snapshots {sorted(adapters)} do not cover the engines {sorted(set(cells['engine']))}")
    config = scientific_config(a, manifest, adapters)
    path = run_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != config:
            diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            print(f"refused: run-dir {run_dir} was started with different settings: {diff}", file=sys.stderr)
            return 2
    else:
        path.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    if not 0 <= a.shard < a.shards:
        raise SystemExit("refused: --shard must be in [0, --shards)")
    cells = cells[cells["order"] % a.shards == a.shard]  # disjoint random slices for parallel allocations
    if a.prompt_ids:  # a pilot subset (runtime selection: a later run without it continues the same run-dir)
        keep = {x.strip() for x in Path(a.prompt_ids).read_text().splitlines() if x.strip()}
        cells = cells[cells["prompt_id"].isin(keep)]
    mine = set(cells["cell_id"])
    done = {c: s for c, s in finished(run_dir).items() if c in mine}
    todo = cells[~cells["cell_id"].isin(done)]
    if a.limit:
        todo = todo.head(max(0, a.limit - len(done)))
    print(json.dumps({"cells": len(cells), "finished": len(done), "to_run": len(todo), "retriever": a.retriever,
                      "model": a.model, "shard": f"{a.shard}/{a.shards}", "started_at": utc()}), flush=True)
    if todo.empty:
        return 0
    budget = AllocationBudget.from_environment()
    scorer = make_scorer(a)
    part = run_dir / "parts" / (f"part-{socket.gethostname().split('.')[0]}-"
                                f"{os.environ.get('SLURM_JOB_ID') or os.getpid()}-"
                                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.jsonl")

    async def main() -> int:
        async with make_client(a) as client:
            return await run_cells(a, todo, prompts, adapters, scorer, client, budget, part)

    status = asyncio.run(main())
    if status == 0 and a.limit == 0 and len(set(finished(run_dir)) & mine) < len(cells):
        return 4
    return status


def cmd_status(a) -> int:
    cells, _, _ = load_cells(Path(a.bundle), a.model)
    done = finished(Path(a.run_dir))
    c = cells.assign(status=cells["cell_id"].map(done).fillna("pending"))
    table = c.groupby(["method", "engine", "status"]).size().unstack(fill_value=0)
    print(table.to_string())
    print(json.dumps({"cells": len(cells), "complete": int((c.status == "complete").sum()),
                      "failed": int((c.status == "failed").sum()), "pending": int((c.status == "pending").sum())}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--model", required=True, choices=("qwen38", "llama4"))
    r.add_argument("--retriever", required=True, choices=("bm25", "orig"))
    r.add_argument("--snapshot", action="append", required=True, help="ENGINE=PATH, once per engine")
    r.add_argument("--base-url", required=True)
    r.add_argument("--served-model", required=True)
    r.add_argument("--api-key-env", help="environment variable holding an external API key (https only)")
    r.add_argument("--cross-encoder", help="local bge-reranker-v2-m3 snapshot; omitted only in tests")
    r.add_argument("--device", default="cpu", help="cross-encoder device: cpu (the original), cuda, cuda:N, or "
                   "cuda-pool (one replica per visible GPU next to vLLM, CPU fallback on out-of-memory)")
    r.add_argument("--score-cache", action="append", default=[], help="score-cache parquet (repeatable)")
    r.add_argument("--query-max-tokens", type=int, default=256)
    r.add_argument("--final-max-tokens", type=int, required=True)
    r.add_argument("--disable-thinking", action="store_true")
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--request-concurrency", type=int, default=4)
    r.add_argument("--cell-concurrency", type=int, default=12)
    r.add_argument("--transport-retries", type=int, default=0)
    r.add_argument("--breaker", type=int, default=20, help="consecutive transport-failed cells that stop the run")
    r.add_argument("--limit", type=int, default=0, help="stop once this many cells are finished (pilots)")
    r.add_argument("--prompt-ids", help="file with one prompt_id per line: run only these prompts' cells")
    r.add_argument("--shard", type=int, default=0, help="this allocation's slice: cells with order %% shards == shard")
    r.add_argument("--shards", type=int, default=1, help="number of slices run by parallel allocations")
    s = sub.add_parser("status")
    s.add_argument("--bundle", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--model", required=True, choices=("qwen38", "llama4"))
    a = p.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_status(a)


if __name__ == "__main__":
    raise SystemExit(main())
