#!/usr/bin/env python3
"""List the cells of a finished Gemma SI-v4 run whose answer map failed, for a labelled map-recovery run.

Reads each shard's results database read-only (answer_map tasks whose saved result has ok = false) and the
shard's frozen cells, and writes the fingerprints of every cell that uses a failed map, one per line.
Prints counts by failure, including how many validation attempts each failed map used. Changes nothing.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sqlite3
from pathlib import Path


def select(run: Path) -> tuple[list[str], dict]:
    plan = json.loads((run / "plan.json").read_text())
    failed, attempts, cells = set(), collections.Counter(), set()
    for shard in plan["shards"]:
        directory = Path(shard["directory"])
        db = directory / "results/control/index.sqlite"
        if not db.exists():
            raise ValueError(f"shard has no results yet: {shard['id']}")
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for tid, raw in con.execute("SELECT id,result FROM tasks WHERE kind='answer_map' AND result IS NOT NULL"):
                result = json.loads(raw)
                if not result.get("ok"):
                    failed.add(tid)
                    attempts[len(result.get("validation_attempts") or [])] += 1
        finally:
            con.close()
        with gzip.open(directory / "inputs/cells.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                cell = json.loads(line)
                if cell.get("map_task_id") in failed:
                    cells.add(cell["fingerprint"])
    report = {"plan_id": plan["plan_id"], "failed_maps": len(failed), "cells": len(cells),
              "failed_maps_by_attempts": dict(sorted(attempts.items()))}
    return sorted(cells), report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cells, report = select(args.run.resolve())
    if not cells:
        raise ValueError("no cell uses a failed map; nothing to recover")
    args.output.write_text("".join(f + "\n" for f in cells))
    print(json.dumps({**report, "output": str(args.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
