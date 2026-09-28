#!/usr/bin/env python3
"""Audit whether Shuffled/Ablated changed the generator's input, from recorded traces.

Read-only. Groups completed cells by (generator, prompt, method, engine), classifies
each Natural/Shuffled and Natural/Ablated pair (see condition_manipulation.py), and
writes per-group rows plus a summary by generator, method and engine.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs, iter_cells
from analysis.interpretability.pipeline.condition_manipulation import audit_group


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=["qwen38", "llama4"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--limit-prompts", type=int, help="only the first N prompts in sorted order")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a new folder")
    prompt_ids = ({line.strip() for line in args.prompt_ids.read_text().splitlines() if line.strip()}
                  if args.prompt_ids else None)
    refs, selection = completed_generator_refs(args.dataset_root, model=args.model, prompt_ids=prompt_ids)
    by_key: dict[tuple, list] = defaultdict(list)
    for ref in refs:
        by_key[(ref["prompt_id"], ref["method"], ref["engine"])].append(ref)
    keys = sorted(by_key)
    if args.limit_prompts is not None:
        allowed = set(sorted({k[0] for k in keys})[:args.limit_prompts])
        keys = [k for k in keys if k[0] in allowed]

    def groups(batch: int = 300):
        # Read traces a batch of groups at a time; never the whole corpus in memory.
        for start in range(0, len(keys), batch):
            chunk = keys[start:start + batch]
            cells: dict[tuple, list] = defaultdict(list)
            for cell in iter_cells(args.dataset_root, [r for k in chunk for r in by_key[k]]):
                g = cell["generation"]
                cells[(g["prompt_id"], g["method"], g["engine"])].append(
                    {"condition": g["condition"], "method": g["method"], "trace": cell["trace"],
                     "condition_audit": g.get("condition_audit")})
            for key in chunk:
                yield key, cells[key]

    args.output.mkdir(parents=True)
    summary: dict[str, Counter] = defaultdict(Counter)
    with gzip.open(args.output / "groups.jsonl.gz", "wt", encoding="utf-8") as stream:
        for key, cells in groups():
            result = audit_group(cells)
            prompt_id, method, engine = key
            stream.write(json.dumps({"model": args.model, "prompt_id": prompt_id, "method": method,
                                     "engine": engine, **result}, sort_keys=True) + "\n")
            slice_key = f"{method}|{engine}"
            summary[slice_key][f"status:{result['status']}"] += 1
            for condition in ("shuffled", "ablated"):
                if condition in result:
                    summary[slice_key][f"{condition}:{result[condition]['change']}"] += 1
                    if result[condition]["erased_by_reranking"]:
                        summary[slice_key][f"{condition}:erased_by_reranking"] += 1
            if "ablated" in result:
                summary[slice_key][f"ablated_exposure:{result['ablated'].get('exposure')}"] += 1
    report = {"model": args.model, "dataset_root": str(args.dataset_root), "cell_selection": dict(selection),
              "groups": len(keys), "by_method_engine": {k: dict(sorted(v.items())) for k, v in sorted(summary.items())},
              "scientific_result": False}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
