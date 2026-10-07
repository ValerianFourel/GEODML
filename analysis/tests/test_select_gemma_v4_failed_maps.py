"""Failed-map selection: only cells whose map result has ok = false; refuses unfinished shards."""
import gzip
import json
import sqlite3

import pytest

from analysis.scripts import select_gemma_v4_failed_maps as sel


def make_run(tmp_path, maps, cells, *, missing_db=False):
    root = tmp_path / "run"
    shard = root / "shards/shard-0000"
    (shard / "inputs").mkdir(parents=True)
    with gzip.open(shard / "inputs/cells.jsonl.gz", "wt") as f:
        for fp, mid in cells:
            f.write(json.dumps({"fingerprint": fp, "map_task_id": mid}) + "\n")
    if not missing_db:
        (shard / "results/control").mkdir(parents=True)
        con = sqlite3.connect(shard / "results/control/index.sqlite")
        con.execute("CREATE TABLE tasks (id TEXT, kind TEXT, result TEXT)")
        for mid, ok, attempts in maps:
            con.execute("INSERT INTO tasks VALUES (?,?,?)",
                        (mid, "answer_map", json.dumps({"ok": ok, "validation_attempts": [{}] * attempts})))
        con.execute("INSERT INTO tasks VALUES (?,?,?)", ("s1", "source_dependency", json.dumps({"ok": False})))
        con.commit(); con.close()
    (root / "plan.json").write_text(json.dumps({"plan_id": "gemma-v4-x", "shards": [{"id": "shard-0000", "directory": str(shard)}]}))
    return root


def test_selects_cells_of_failed_maps_only(tmp_path):
    root = make_run(tmp_path, [("m1", False, 2), ("m2", True, 1), ("m3", False, 1)],
                    [("c1", "m1"), ("c2", "m1"), ("c3", "m2"), ("c4", "m3")])
    cells, report = sel.select(root)
    assert cells == ["c1", "c2", "c4"]
    assert report["failed_maps"] == 2 and report["failed_maps_by_attempts"] == {1: 1, 2: 1}
    out = tmp_path / "cells.txt"
    assert sel.main(["--run", str(root), "--output", str(out)]) == 0
    assert out.read_text() == "c1\nc2\nc4\n"
    with pytest.raises(FileExistsError):
        sel.main(["--run", str(root), "--output", str(out)])


def test_refuses_a_shard_without_results(tmp_path):
    with pytest.raises(ValueError, match="no results"):
        sel.select(make_run(tmp_path, [], [("c1", "m1")], missing_db=True))
