#!/usr/bin/env python3
"""Smoke-test SI-v2 on a few finished cells; diagnostic only, no scientific result.

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


def replay_refs(refs: list[dict], saved: list[dict], *, model: str, count: int) -> list[dict]:
    """Resolve exactly the saved generators, never replace a missing or changed cell."""
    if len(saved) != count or len({r["fingerprint"] for r in saved}) != count:
        raise ValueError("replay must contain exactly --count distinct cells")
    available = {r["fingerprint"]: r for r in refs}
    chosen = []
    for row in saved:
        ref = available.get(row["fingerprint"])
        if row.get("model") != model or ref is None:
            raise ValueError(f"replay cell unavailable for {model}: {row['fingerprint']}")
        for key, reference in (("generation_record_id", "generation_ref"), ("trace_record_id", "trace_ref")):
            if row.get(key) != ref[reference]["record_id"]:
                raise ValueError(f"replay {key} changed: {row['fingerprint']}")
        chosen.append(ref)
    return chosen


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


async def main_async(args) -> int:
    from analysis.scripts.run_acl_arr_vllm import VllmChatClient, _execute_one, _iter_execute
    args.output.mkdir(parents=True)
    frozen = []
    try:
        saved = None
        if args.cells_from:
            content = args.cells_from.read_bytes()
            if args.cells_from_sha256 and hashlib.sha256(content).hexdigest() != args.cells_from_sha256:
                raise ValueError("replay selection file checksum changed")
            saved = [json.loads(line) for line in content.splitlines() if line.strip()]
        refs, selection = completed_generator_refs(args.dataset_root, model=args.model,
            prompt_ids={r["prompt_id"] for r in saved} if saved is not None else None)
        chosen = (replay_refs(refs, saved, model=args.model, count=args.count) if saved is not None
                  else pick(refs, count=args.count, seed=args.seed))
        if len(chosen) != args.count:
            raise ValueError(f"requested {args.count} cells, only {len(chosen)} available")
        print(f"SELECTED {len(chosen)} cells of {selection.get('completed', 0)} completed", flush=True)
        for cell in iter_cells(args.dataset_root, chosen):
            record, tasks = build_cell(cell, max_tokens=args.max_tokens, j1_max_tokens=64, sensitivity_fraction=0.0)
            frozen.append((record, {t["judge_task_id"]: t for t in tasks}))
    except Exception as exc:
        report = {"scientific_result": False, "status": "failed", "requested_cells": args.count,
                  "error": f"{type(exc).__name__}: {exc}", "cells": len(frozen), "complete_cells": 0}
        write_json(args.output / "summary.json", report)
        print("SUMMARY " + json.dumps(report), flush=True)
        return 1
    unique = {tid: task for _, tasks in frozen for tid, task in tasks.items()}
    # Freeze exact inputs and the per-cell provenance before opening a connection.
    (args.output / "tasks.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n"
                                                  for t in unique.values()))
    (args.output / "cells.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c, _ in frozen))
    audit = (args.output / "audit.jsonl").open("a", encoding="utf-8")
    results_file = (args.output / "results.jsonl").open("a", encoding="utf-8")
    results: list[dict] = []
    example: dict = {}
    started = time.time()
    execution_error = None
    def record_audit(event):
        audit.write(json.dumps(event, default=str) + "\n")
        audit.flush()

    try:
        async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=args.server_model_name,
                                  timeout_seconds=args.request_timeout, maximum_attempts=3,
                                  audit_callback=record_audit,
                                  chat_template_kwargs={"enable_thinking": False}) as client:
            async def one(record_task: dict, **kwargs) -> dict:
                item = si.item_from_record(record_task)
                result = await _execute_one(item, client=client, fake=False)
                row = {**{k: v for k, v in result.items() if k != "base"}, "judge_task_id": record_task["judge_task_id"],
                       "task": record_task["task"], "prompt_sha256": item["prompt_sha256"]}
                results_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                results_file.flush()
                results.append(row)
                if record_task["task"] == "source_importance" and result.get("ok") and not example:
                    example.update(prompt=item["prompt"], raw_output=result.get("raw_output"),
                                   parsed_output=result.get("parsed_output"))
                return row

            # The shared scheduler drains/cancels peers if one task raises unexpectedly.
            async for _ in _iter_execute(unique.values(), client=client, maximum_concurrency=args.concurrency,
                                         fake=False, execute_one=one):
                pass
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        audit.close()
        results_file.close()
    by_id = {r["judge_task_id"]: r for r in results}
    cells = []
    for record, _ in frozen:
        if record.get("status") == "ok":
            grades = {s["url"]: (by_id[s["judge_task_id"]]["parsed_output"]["importance"]
                                 if by_id.get(s.get("judge_task_id"), {}).get("ok") else None)
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
    passed = (execution_error is None and len(cells) == args.count and bool(unique)
              and len(results) == len(unique) and all(r.get("ok") for r in results)
              and all(c.get("metrics", {}).get("complete") and c.get("j1") is not None for c in cells))
    report = {**summarize(results, cells), "seconds": round(time.time() - started, 1),
              "status": "passed" if passed else "failed", "requested_cells": args.count,
              "execution_error": execution_error,
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
    return 0 if passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-schema", action="store_true", help="CPU-only check with installed xgrammar; no server")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--server-model-name")
    parser.add_argument("--cells-from", type=Path, help="replay exactly these saved cells.jsonl identities")
    parser.add_argument("--cells-from-sha256", help="expected checksum of the saved replay selection")
    parser.add_argument("--model", choices=["qwen38", "llama4"], default="qwen38")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-tokens", type=int, default=si.DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.check_schema:
        import importlib.metadata
        import xgrammar
        schema = si.source_importance_schema(["a1", "a2"])
        xgrammar.Grammar.from_json_schema(json.dumps(schema))
        print(json.dumps({"status": "accepted", "protocol": si.PROTOCOL,
                          "xgrammar_version": importlib.metadata.version("xgrammar"),
                          "schema_sha256": si._digest(schema)}))
        return 0
    for name in ("dataset_root", "output", "base_url", "server_model_name"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.count <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        parser.error("count, concurrency and max-tokens must be positive")
    if args.cells_from_sha256 and not args.cells_from:
        parser.error("--cells-from-sha256 requires --cells-from")
    if args.output.exists():
        parser.error("output already exists; use a new path")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
