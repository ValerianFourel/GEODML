#!/usr/bin/env python3
"""Try the claims-v3 judge on a few finished cells; diagnostic only, no scientific result.

Runs inside `search_vllm_stage.py run` (authenticated loopback Nemotron server).
Reads a dataset (never writes it), picks cells deterministically, runs claim
extraction, J1, J3 and J2 in two evidence orders, and writes raw results plus a
summary to a new output folder.
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
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows, verify_record_reference
from analysis.interpretability.pipeline.agentic_judging import (
    _trace_evidence,
    aggregate_claim_attribution,
    claims_evidence,
    prepare_attribution,
    prepare_claim_extraction,
    prepare_fulfilment,
    prepare_relevance,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity


def _record(root: Path, reference: dict) -> dict:
    if not verify_record_reference(root, reference):
        raise ValueError(f"record reference failed verification: {reference.get('record_id')}")
    path = root / "data" / reference["table"] / f"part-{reference['writer_id']}-{reference['shard_sequence']:06d}.jsonl"
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if number == reference["line_number"]:
                return json.loads(line)["row"]
    raise ValueError("record line missing")


def select_cells(root: Path, *, count: int, seed: int, model: str | None, stripes: int = 256,
                 skipped: list | None = None) -> list[dict]:
    """Deterministic sample of verified completed generator cells with their trace and prompt.

    Cells whose record references do not verify (e.g. shards a live writer has not
    sealed yet) are skipped, like the other dataset readers do; their record IDs go
    to `skipped` when given.
    """
    tasks = {identity_fingerprint(ClaimIdentity(**row["claim_identity"])): row
             for row in iter_sealed_rows(root, "task_definitions", required=True)
             if row.get("model") in {"qwen38", "llama4"} and (model is None or row.get("model") == model)}
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    done = sorted((fp for fp, event in latest.items() if fp in tasks and event.get("state") == "completed"),
                  key=lambda fp: hashlib.sha256(f"{seed}:{fp}".encode()).hexdigest())
    prompts = {row["prompt_id"]: row for row in iter_sealed_rows(root, "prompts")}
    cells = []
    for fp in done:
        refs = {ref["table"]: ref for ref in latest[fp].get("record_references", [])}
        if "generations" not in refs or "traces" not in refs:
            continue
        unverified = [ref.get("record_id") for ref in (refs["generations"], refs["traces"])
                      if not verify_record_reference(root, ref)]
        if unverified:
            if skipped is not None:
                skipped.extend(unverified)
            continue
        generation, trace = _record(root, refs["generations"]), _record(root, refs["traces"])
        prompt = prompts.get(generation.get("prompt_id"), {})
        if not prompt.get("prompt_text") or not generation.get("answer"):
            continue
        cells.append({"fingerprint": fp, "model": tasks[fp].get("model"), "generation": generation,
                      "trace": trace, "prompt_text": prompt["prompt_text"]})
        if len(cells) == count:
            break
    return cells


def _public(result: dict, item: dict) -> dict:
    return {**{k: v for k, v in result.items()}, "prompt_sha256": item["prompt_sha256"],
            "schema_sha256": item["schema_sha256"], "max_tokens": item["max_tokens"]}


async def judge_cell(cell: dict, *, client, execute, master_seed: int, max_tokens: int, write) -> dict:
    generation, trace = cell["generation"], cell["trace"]
    rows, _ = _trace_evidence(trace, generation["method"])
    evidence = claims_evidence(rows, master_seed=master_seed)
    items = {"claim_extraction": prepare_claim_extraction(generation["answer"], max_tokens=max_tokens),
             "fulfilment": prepare_fulfilment(prompt_text=cell["prompt_text"], answer=generation["answer"], max_tokens=64),
             "ideal_relevance": prepare_relevance(prompt_text=cell["prompt_text"], evidence=evidence,
                                                  master_seed=master_seed, max_tokens=max_tokens)}
    first = await asyncio.gather(*(execute(item, client=client, fake=False) for item in items.values()))
    results = dict(zip(items, first))
    for name, result in results.items():
        write({"cell": cell["fingerprint"], "task": name, **_public(result, items[name])})
    summary = {"cell": cell["fingerprint"], "model": cell["model"], "method": generation["method"],
               "evidence_count": len(evidence), "answer_chars": len(generation["answer"]),
               "ok": {name: bool(result.get("ok")) for name, result in results.items()}}
    extraction = results["claim_extraction"]
    if not extraction.get("ok"):
        return summary
    claims = extraction["parsed_output"]["claims"]
    summary["claim_count"] = len(claims)
    if not claims or not evidence:
        return summary
    for variant in ("primary", "reverse"):
        item = prepare_attribution(claims, evidence, master_seed=master_seed, max_tokens=max_tokens, variant=variant)
        result = await execute(item, client=client, fake=False)
        write({"cell": cell["fingerprint"], "task": f"attribution_{variant}", **_public(result, item)})
        summary["ok"][f"attribution_{variant}"] = bool(result.get("ok"))
        if result.get("ok"):
            aggregate = aggregate_claim_attribution(result["parsed_output"],
                                                    evidence_ids=[row.evidence_id for row in evidence])
            summary[f"tiers_{variant}"] = aggregate["realized_support_tiers"]
            summary[f"fractions_{variant}"] = {k: aggregate[k] for k in (
                "fully_supported_fraction", "partial_only_fraction", "unsupported_fraction", "contradiction_fraction")}
    return summary


def summarize(records: list[dict], cells: list[dict]) -> dict:
    by_task = collections.defaultdict(list)
    for row in records:
        by_task[row["task"]].append(row)
    tasks = {}
    for name, rows in sorted(by_task.items()):
        tokens = [row.get("usage", {}).get("completion_tokens") for row in rows]
        tokens = [value for value in tokens if isinstance(value, int)]
        tasks[name] = {
            "requests": len(rows), "ok": sum(bool(row.get("ok")) for row in rows),
            "failure_categories": dict(collections.Counter(
                c for row in rows for c in row.get("failure_categories", []))),
            "finish_reasons": dict(collections.Counter(row.get("usage", {}).get("finish_reason") for row in rows)),
            "completion_tokens_median": statistics.median(tokens) if tokens else None,
            "completion_tokens_max": max(tokens) if tokens else None,
            "seconds_median": statistics.median(row.get("duration_seconds", 0) for row in rows),
        }
    paired = [c for c in cells if "tiers_primary" in c and "tiers_reverse" in c]
    top1 = [bool(c["tiers_primary"]) and bool(c["tiers_reverse"]) and c["tiers_primary"][0] == c["tiers_reverse"][0]
            for c in paired]
    return {"scientific_result": False, "cells": len(cells), "tasks": tasks,
            "claims_per_answer": [c.get("claim_count") for c in cells],
            "order_pairs": len(paired), "order_top1_tier_agreement": sum(top1),
            "order_identical_tiers": sum(c["tiers_primary"] == c["tiers_reverse"] for c in paired)}


async def main_async(args) -> int:
    from analysis.scripts.run_acl_arr_vllm import VllmChatClient, _execute_one
    skipped: list = []
    cells = select_cells(args.dataset_root, count=args.count, seed=args.seed, model=args.model, skipped=skipped)
    print(f"SELECTED {len(cells)} cells; SKIPPED {len(skipped)} unverified references {skipped[:3]}", flush=True)
    if not cells:
        raise SystemExit("no verified completed cells to judge")
    args.output.mkdir(parents=True)
    records: list[dict] = []
    audit = (args.output / "audit.jsonl").open("a", encoding="utf-8")
    results = (args.output / "results.jsonl").open("a", encoding="utf-8")

    def write(row):
        records.append(row)
        results.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        results.flush()

    started = time.time()
    async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=args.server_model_name,
                              timeout_seconds=args.request_timeout, maximum_attempts=3,
                              audit_callback=lambda e: audit.write(json.dumps(e, default=str) + "\n"),
                              chat_template_kwargs={"enable_thinking": False}) as client:
        gate = asyncio.Semaphore(args.concurrency)

        async def one(cell):
            async with gate:
                summary = await judge_cell(cell, client=client, execute=_execute_one, master_seed=args.master_seed,
                                           max_tokens=args.max_tokens, write=write)
                print("CELL " + json.dumps(summary, default=str), flush=True)
                return summary

        summaries = await asyncio.gather(*(one(cell) for cell in cells))
    audit.close()
    results.close()
    report = {**summarize(records, summaries), "seconds": round(time.time() - started, 1),
              "skipped_unverified_references": skipped,
              "settings": {"master_seed": args.master_seed, "seed": args.seed, "max_tokens": args.max_tokens,
                           "concurrency": args.concurrency, "model_filter": args.model,
                           "server_model_name": args.server_model_name, "temperature": 0.0,
                           "enable_thinking": False}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (args.output / "cells.jsonl").write_text("".join(json.dumps(s, default=str) + "\n" for s in summaries))
    print("SUMMARY " + json.dumps(report, default=str), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-model-name", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--master-seed", type=int, default=20260915)
    parser.add_argument("--model", choices=["qwen38", "llama4"])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a new path")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
