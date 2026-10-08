"""Pooling of the stored per-answer alignment measures across Gemma runs (diagnostic)."""

import gzip
import json

from analysis.scripts import gemma_ranking_agreement as g


def make_run(root, cells, status="finished"):
    shard = root / "shards/s1"
    report = shard / "results/reports/r1"
    report.mkdir(parents=True)
    (shard / "results/reports/latest.json").write_text(json.dumps({"directory": "r1"}))
    (report / "summary.json").write_text(json.dumps({"status": status}))
    with gzip.open(report / "cells.jsonl.gz", "wt") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    (root / "plan.json").write_text(json.dumps({"plan_id": root.name, "shards": [{"id": "s1", "directory": str(shard), "cells": len(cells)}]}))
    return root


def cell(fp, model, top, complete=True, prompt="p1"):
    return {"fingerprint": fp, "status": "ok", "model": model, "method": "Reactive", "engine": "duckduckgo",
            "condition": "natural", "prompt_id": prompt, "sources": [{"status": "scored"}],
            "metrics": {"complete": complete, "top_source_alignment": top, "generator_chance_baseline": 0.5,
                        "ordering_tau_b": 0.2 if top else -0.2}}


def test_pools_complete_cells_once_and_audits(tmp_path):
    a = make_run(tmp_path / "llama", [cell("f1", "llama4", True), cell("f2", "llama4", False, prompt="p2"),
                                      cell("f3", "llama4", True, complete=False), {"fingerprint": "f4", "status": "missing_request_or_answer"}])
    b = make_run(tmp_path / "qwen", [cell("q1", "qwen38", True), cell("f1", "llama4", False)])      # f1 again: skipped
    c = make_run(tmp_path / "pending", [cell("x", "qwen38", True)], status="running")                  # not finished: ignored
    out = tmp_path / "out"
    assert g.main(["--run", str(a), "--run", str(b), "--run", str(c), "--output", str(out), "--draws", "20"]) == 0
    r = json.loads((out / "agreement.json").read_text())
    llama = r["groups"]["model"]["llama4"]
    assert llama["answers"] == 2 and llama["top_source_alignment"]["mean"] == 0.5
    assert r["groups"]["model"]["qwen38"]["answers"] == 1
    assert r["audit"]["llama"]["cells"] == {"complete": 2, "incomplete": 1, "input_missing_request_or_answer": 1}
    assert r["audit"]["qwen"]["duplicates_skipped"] == 1 and r["audit"]["pending"]["shards"] == {"running": 1}
    assert r["scientific_result"] is False and (out / "agreement.md").exists()
