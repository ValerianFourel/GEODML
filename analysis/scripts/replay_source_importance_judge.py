#!/usr/bin/env python3
"""Replay frozen SI-v3 pilot tasks twice against one already running judge server."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as si
from analysis.scripts import try_source_importance_judge as trial


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def indexed(records):
    out = {}
    for record in records:
        key = record["judge_task_id"]
        if key in out:
            raise ValueError(f"duplicate task/result: {key}")
        out[key] = record
    return out


def frozen_cells(bundle):
    tasks = indexed(bundle["tasks"])
    for task in tasks.values():
        si.item_from_record(task)
    frozen, used, seen = [], set(), set()
    for original in bundle["cells"]:
        cell = copy.deepcopy(original)
        if cell["cell_id"] in seen or cell.get("status") != "ok":
            raise ValueError("duplicate or incomplete baseline cell")
        seen.add(cell["cell_id"])
        ids = {s["judge_task_id"] for s in cell["sources"]} | {cell["j1_task_id"]}
        if not ids <= tasks.keys():
            raise ValueError("cell references missing frozen tasks")
        used |= ids
        for key in ("grades", "metrics", "j1"):
            cell.pop(key, None)
        frozen.append((cell, {tid: tasks[tid] for tid in sorted(ids)}))
    if used != tasks.keys() or not frozen:
        raise ValueError("orphan tasks or empty frozen pilot")
    return frozen


def freeze_baseline(root):
    """Read the four saved trials, refusing missing, changed or ambiguous attempts."""
    bundles = {}
    files = {}
    for pass_id in (1, 2):
        tasks, cells, results = {}, [], []
        for model in ("qwen38", "llama4"):
            found = sorted((root / "runs" / f"pass{pass_id}-{model}" / "attempts").glob("job*/trial"))
            if len(found) != 1:
                raise ValueError(f"expected one saved trial for pass{pass_id}-{model}: {found}")
            path = found[0]
            for name in ("tasks.jsonl", "cells.jsonl", "results.jsonl", "summary.json"):
                file = path / name
                files[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            if json.loads((path / "summary.json").read_text()).get("status") != "passed":
                raise ValueError(f"baseline did not pass: {path}")
            selected = rows(path / "cells.jsonl")
            if len(selected) != 10 or any(c.get("model") != model for c in selected):
                raise ValueError("expected the frozen ten cells for each generator")
            cells.extend(selected)
            for record in rows(path / "tasks.jsonl"):
                tid = record["judge_task_id"]
                if tid in tasks and tasks[tid] != record:
                    raise ValueError("conflicting shared task")
                tasks[tid] = record
            results.extend(rows(path / "results.jsonl"))
        bundles[pass_id] = {"tasks": list(tasks.values()), "cells": cells, "results": results}
        frozen_cells(bundles[pass_id])
        # Tasks may be shared between corpora; retain both observed judgments.
        if {r["judge_task_id"] for r in results} != tasks.keys() or not all(r.get("ok") for r in results):
            raise ValueError("incomplete baseline judgments")
    one, two = bundles[1], bundles[2]
    if indexed(one["tasks"]) != indexed(two["tasks"]) or frozen_cells(one) != frozen_cells(two):
        raise ValueError("baseline passes have different inputs or cell mappings")
    return {"protocol": si.PROTOCOL, "tasks": one["tasks"], "cells": one["cells"],
            "nemotron": {"pass1": one["results"], "pass2": two["results"]}, "source_files": files}


def compare(bundle, passes):
    def value(row):
        if not row or not row.get("ok"):
            return None
        return row["parsed_output"].get("importance", row["parsed_output"].get("request_fulfillment"))
    by_pass = {name: indexed(records) for name, records in passes.items()}
    return [{"judge_task_id": task["judge_task_id"], "task": task["task"],
             "nemotron": {name: [value(r) for r in records if r["judge_task_id"] == task["judge_task_id"]]
                          for name, records in bundle["nemotron"].items()},
             "gemma": {name: value(records.get(task["judge_task_id"])) for name, records in by_pass.items()}}
            for task in bundle["tasks"]]


async def run(args):
    content = args.inputs.read_bytes()
    if hashlib.sha256(content).hexdigest() != args.inputs_sha256:
        raise ValueError("frozen replay bundle checksum changed")
    bundle = json.loads(content)
    frozen = frozen_cells(bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    passes, codes = {}, []
    for pass_id in (1, 2):
        name = f"pass{pass_id}"
        options = SimpleNamespace(output=args.output / name, cells_from=None, count=len(frozen),
            base_url=args.base_url, server_model_name=args.server_model_name, request_timeout=300,
            concurrency=4, seed=2026093010, max_tokens=640, model="qwen38+llama4")
        codes.append(await trial.main_async(options, frozen_input=copy.deepcopy(frozen)))
        passes[name] = rows(options.output / "results.jsonl")
        trial.write_json(args.output / "comparison.json", {"scientific_result": False,
            "scores": compare(bundle, passes)})
        if codes[-1]:
            break
    trial.write_json(args.output / "summary.json", {"scientific_result": False,
        "status": "passed" if codes == [0, 0] else "failed", "pass_exit_codes": codes,
        "inputs_sha256": args.inputs_sha256, "judge_model": args.server_model_name})
    return 0 if codes == [0, 0] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-model-name", required=True)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
