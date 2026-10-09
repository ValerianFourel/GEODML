#!/usr/bin/env python3
"""List cells of a Gemma SI-v4 run for a labelled recovery run.

Default: cells whose answer map failed (saved result ok = false), with counts by validation attempts.
--unfinished: cells Gemma never fully judged; their answer map is not done, or a source task is neither
done nor blocked. Cells with a failed map are not unfinished.
Reads each shard's results database read-only and the shard's frozen cells, and writes one fingerprint per
line. Changes nothing.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sqlite3
from pathlib import Path


def select(run: Path, unfinished: bool = False) -> tuple[list[str], dict]:
    plan = json.loads((run / "plan.json").read_text())
    failed, attempts, cells, why = set(), collections.Counter(), set(), collections.Counter()
    for shard in plan["shards"]:
        directory = Path(shard["directory"])
        db = directory / "results/control/index.sqlite"
        if not db.exists():
            raise ValueError(f"shard has no results yet: {shard['id']}")
        con = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            states = dict(con.execute("SELECT id,state FROM tasks")) if unfinished else {}
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
                if not unfinished:
                    if cell.get("map_task_id") in failed:
                        cells.add(cell["fingerprint"])
                    continue
                if states.get(cell.get("map_task_id")) != "done":
                    why["map_not_done"] += 1
                elif any(states.get(s.get("dependency_id")) not in ("done", "blocked") for s in cell.get("sources", [])):
                    why["sources_not_done"] += 1
                else:
                    continue
                cells.add(cell["fingerprint"])
    if unfinished:
        return sorted(cells), {"plan_id": plan["plan_id"], "selection": "unfinished", "cells": len(cells),
                               "unfinished_by_reason": dict(sorted(why.items()))}
    report = {"plan_id": plan["plan_id"], "failed_maps": len(failed), "cells": len(cells),
              "failed_maps_by_attempts": dict(sorted(attempts.items()))}
    return sorted(cells), report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unfinished", action="store_true", help="cells never fully judged instead of failed-map cells")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    cells, report = select(args.run.resolve(), args.unfinished)
    if not cells:
        raise ValueError("no selected cell; nothing to recover")
    args.output.write_text("".join(f + "\n" for f in cells))
    print(json.dumps({**report, "output": str(args.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
