#!/usr/bin/env python3
"""Smoke-test the SI-v1 judge on a few finished cells; diagnostic only, no scientific result.

Runs inside `search_vllm_stage.py run` (authenticated loopback Nemotron server).
Reads a generator dataset read-only, picks cells deterministically (alternating
methods), freezes their tasks exactly as prepare_source_importance_tasks.py does
(complete answers, every observed source, J1), runs every task once through the
production retry path, and derives rank groups and alignment metrics in code.
Writes results.jsonl, audit.jsonl, cells.jsonl, example.json and summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as si
from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs, iter_cells
from analysis.scripts.prepare_source_importance_tasks import build_cell


def pick(refs: list[dict], *, count: int, seed: int) -> list[dict]:
    """Deterministic sample alternating Parallel and Reactive cells."""

    by_method = collections.defaultdict(list)
    for ref in sorted(refs, key=lambda r: hashlib.sha256(f"{seed}:{r['fingerprint']}".encode()).hexdigest()):
        by_method[ref.get("method")].append(ref)
    queues, chosen = [by_method[m] for m in sorted(by_method, key=str)], []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return chosen


def summarize(results: list[dict], cells: list[dict]) -> dict:
    by_task = collections.defaultdict(list)
    for row in results:
        by_task[row["task"]].append(row)
    tasks = {}
    for name, rows in sorted(by_task.items()):
        tokens = [r.get("usage", {}).get("completion_tokens") for r in rows]
        tokens = [t for t in tokens if isinstance(t, int)]
        tasks[name] = {
            "requests": len(rows), "ok": sum(bool(r.get("ok")) for r in rows),
            "failure_categories": dict(collections.Counter(c for r in rows for c in r.get("failure_categories", []))),
            "final_failures": dict(collections.Counter(r.get("failure_category") for r in rows if not r.get("ok"))),
            "first_error": next((r["error"] for r in rows if r.get("error")), None),
            "finish_reasons": dict(collections.Counter(r.get("usage", {}).get("finish_reason") for r in rows)),
            "completion_tokens_median": statistics.median(tokens) if tokens else None,
            "completion_tokens_max": max(tokens) if tokens else None,
            "seconds_median": statistics.median(r.get("duration_seconds", 0) for r in rows),
        }
    grades = [r["parsed_output"]["importance"] for r in by_task.get("source_importance", []) if r.get("ok")]
    return {"scientific_result": False, "cells": len(cells), "tasks": tasks,
            "grade_distribution": dict(sorted(collections.Counter(grades).items())),
            "complete_cells": sum(bool(c.get("metrics", {}).get("complete")) for c in cells),
            "top_source_alignment": [c.get("metrics", {}).get("top_source_alignment") for c in cells],
            "truncated_cells": sum(bool(c.get("truncated")) for c in cells)}


async def main_async(args) -> int:
    from analysis.scripts.run_acl_arr_vllm import VllmChatClient, _execute_one
    refs, selection = completed_generator_refs(args.dataset_root, model=args.model)
    chosen = pick(refs, count=args.count, seed=args.seed)
    print(f"SELECTED {len(chosen)} cells of {selection.get('completed', 0)} completed", flush=True)
    frozen = []
    for cell in iter_cells(args.dataset_root, chosen):
        record, tasks = build_cell(cell, max_tokens=args.max_tokens, j1_max_tokens=64, sensitivity_fraction=0.0)
        frozen.append((record, {t["judge_task_id"]: t for t in tasks}))
    args.output.mkdir(parents=True)
    audit = (args.output / "audit.jsonl").open("a", encoding="utf-8")
    results_file = (args.output / "results.jsonl").open("a", encoding="utf-8")
    results: list[dict] = []
    example: dict = {}
    started = time.time()
    async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=args.server_model_name,
                              timeout_seconds=args.request_timeout, maximum_attempts=3,
                              audit_callback=lambda e: audit.write(json.dumps(e, default=str) + "\n"),
                              chat_template_kwargs={"enable_thinking": False}) as client:
        gate = asyncio.Semaphore(args.concurrency)

        async def one(record_task: dict) -> dict:
            item = si.item_from_record(record_task)
            async with gate:
                result = await _execute_one(item, client=client, fake=False)
            row = {**{k: v for k, v in result.items() if k != "base"}, "judge_task_id": record_task["judge_task_id"],
                   "task": record_task["task"], "prompt_sha256": item["prompt_sha256"]}
            results_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            results_file.flush()
            if record_task["task"] == "source_importance" and result.get("ok") and not example:
                example.update(prompt=item["prompt"], raw_output=result.get("raw_output"),
                               parsed_output=result.get("parsed_output"))
            return row

        unique = {tid: task for _, tasks in frozen for tid, task in tasks.items()}
        rows = await asyncio.gather(*(one(task) for task in unique.values()))
        results.extend(rows)
    audit.close()
    results_file.close()
    by_id = {r["judge_task_id"]: r for r in results}
    cells = []
    for record, _ in frozen:
        if record.get("status") == "ok":
            grades = {s["url"]: (by_id[s["judge_task_id"]]["parsed_output"]["importance"]
                                 if s.get("judge_task_id") and by_id[s["judge_task_id"]].get("ok") else None)
                      for s in record["sources"]}
            record["grades"] = grades
            record["metrics"] = si.cell_metrics(grades, generator_list=record["generator_ranking"],
                                                presented=record["presented"])
            j1 = by_id.get(record["j1_task_id"], {})
            record["j1"] = (j1.get("parsed_output") or {}).get("request_fulfillment") if j1.get("ok") else None
            listed = set(record["generator_ranking"])
            print("CELL " + json.dumps({
                "cell": record["cell_id"][:12], "method": record["method"], "condition": record["condition"],
                "truncated": record["truncated"], "answer_chars": record["judged_answer_chars"],
                "grades_in_presented_order": [grades[u] for u in record["presented"]],
                "listed_by_generator": [u in listed for u in record["presented"]],
                "generator_order_positions": [record["presented"].index(u) for u in record["generator_ranking"]],
                "rank_groups": record["metrics"].get("rank_groups"),
                "top_source_alignment": record["metrics"]["top_source_alignment"],
                "first_presented_alignment": record["metrics"]["first_presented_alignment"],
                "j1": record["j1"], "reasons": record["metrics"]["reasons"]}), flush=True)
        cells.append(record)
    report = {**summarize(results, cells), "seconds": round(time.time() - started, 1),
              "settings": {"seed": args.seed, "max_tokens": args.max_tokens, "concurrency": args.concurrency,
                           "model_filter": args.model, "server_model_name": args.server_model_name,
                           "protocol": si.PROTOCOL, "temperature": 0.0, "enable_thinking": False}}
    (args.output / "cells.jsonl").write_text("".join(json.dumps(c, default=str) + "\n" for c in cells))
    (args.output / "example.json").write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    if example:
        print("EXAMPLE_REQUEST_CASE_BLOCK\n" + example["prompt"].split("Return JSON only.")[1].split("\n\n", 1)[1],
              flush=True)
        print("EXAMPLE_RAW_OUTPUT " + str(example["raw_output"]), flush=True)
    print("SUMMARY " + json.dumps(report, default=str), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-model-name", required=True)
    parser.add_argument("--model", choices=["qwen38", "llama4"], default="qwen38")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-tokens", type=int, default=si.DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a new path")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
