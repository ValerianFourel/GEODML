#!/usr/bin/env python3
"""Measure client scheduling with controlled service delays, not GPU inference."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.scripts import run_acl_arr_vllm as runner


async def measure(mode, concurrency, count):
    raw = '{"ranked_document_ids":["C001"]}'
    items = [{"base": {"task_id": str(i)}, "prompt": str(i),
              "schema": runner._rerank_schema(1), "schema_name": "acl_arr_rerank",
              "temperature": 0.0, "max_tokens": 256, "seed": i,
              "validator": lambda text: runner.validate_rerank_output(
                  text, allowed_document_ids=["C001"], output_count=1)}
             for i in range(count)]
    calls, results = [], []
    active = peak = 0
    started = time.perf_counter()

    class Client:
        async def complete(self, **request):
            nonlocal active, peak
            calls.append(request)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.12 if int(request["prompt"]) % (4 * concurrency) == 0 else 0.003)
            active -= 1
            return raw, {}

    first_result = None
    if mode == "chunked":
        for offset in range(0, count, concurrency * 4):
            batch = await runner._execute(items[offset:offset + concurrency * 4],
                client=Client(), maximum_concurrency=concurrency, fake=False)
            if first_result is None:
                first_result = time.perf_counter() - started
            results.extend(batch)
    else:
        async for result in runner._iter_execute(iter(items), client=Client(),
                maximum_concurrency=concurrency, fake=False):
            if first_result is None:
                first_result = time.perf_counter() - started
            results.append(result)
    elapsed = time.perf_counter() - started
    assert len(results) == count and all(r["ok"] for r in results)
    assert len({r["base"]["task_id"] for r in results}) == count
    assert peak <= concurrency
    payloads = sorted(calls, key=lambda r: int(r["prompt"]))
    return {"scheduler": mode, "tasks": count, "concurrency": concurrency,
            "elapsed_seconds": elapsed, "first_result_seconds": first_result,
            "validated_mock_tasks_per_second": count / elapsed, "peak_active_requests": peak,
            "request_payloads_sha256": hashlib.sha256(json.dumps(payloads, sort_keys=True).encode()).hexdigest(),
            "gpu_performance_measured": False, "gpu_hours": None, "scientific_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=("chunked", "rolling", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    modes = ("chunked", "rolling") if args.scheduler == "both" else (args.scheduler,)
    results = [asyncio.run(measure(mode, 8, 128))
               for _ in range(args.repetitions) for mode in modes]
    assert len({r["request_payloads_sha256"] for r in results}) == 1
    runner._atomic_json(args.output, {"benchmark": "controlled-client-service-delays-v1", "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
