"""Gemma-vs-generator agreement: hand-checked cell metrics, deduplication, exclusions and run-root input."""
import csv
import gzip
import json

import pytest

from analysis.scripts import analyze_gemma_generator_agreement as agree


def cell(fp, presented, ranking, grades, *, prompt="p1", model="qwen38", status="ok", source_status=None):
    statuses = source_status or {u: "scored" if g is not None else "global_absence_only" for u, g in grades.items()}
    return {"fingerprint": fp, "cell_id": fp, "prompt_id": prompt, "model": model, "method": "m", "engine": "e",
            "condition": "c", "status": status, "map_eligibility": "eligible", "presented": presented,
            "generator_ranking": ranking, "grades": grades,
            "sources": [{"url": u, "status": statuses[u]} for u in presented]}


def write(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")


def test_cell_metrics_by_hand():
    row = agree.compact(cell("f", ["a", "b", "c", "d"], ["c", "a"], {"a": 3, "b": 0, "c": 5, "d": 4}), "x")
    m = agree.cell_metrics(row)
    # kept grades {5,3} vs dropped {0,4}: pairs 5>0,5>4,3>0,3<4 -> 3/4
    assert m["keep_auc"] == 0.75
    # positions kept {2,0} vs dropped {1,3} (earlier wins): 2<1? no; 2<3 yes; 0<1 yes; 0<3 yes -> 3/4
    assert m["position_keep_auc"] == 0.75 and m["keep_auc_minus_position"] == 0
    assert m["kept_zero_share"] == 0 and m["dropped_used_share"] == 0.5
    assert m["important_dropped_share"] == 0.5  # grade>=4: c kept, d dropped
    assert m["order_tau_b"] == 1.0  # generator puts c(5) before a(3)
    assert m["position_order_tau_b"] == -1.0  # shown order puts a before c
    assert m["generator_follows_shown_order_tau_b"] == -1.0
    assert m["order_tau_minus_position"] == 2.0
    assert m["top_source_alignment"] == 1.0 and m["top_source_chance"] == 0.5


def test_undefined_metrics_stay_none():
    m = agree.cell_metrics(agree.compact(cell("f", ["a", "b"], ["a", "b"], {"a": 0, "b": 0}), "x"))
    assert m["keep_auc"] is None and m["order_tau_b"] is None and m["top_source_alignment"] is None
    assert m["kept_zero_share"] == 1.0
    with pytest.raises(ValueError, match="outside the shown sources"):
        agree.cell_metrics(agree.compact(cell("g", ["a"], ["z"], {"a": 1}), "x"))


def test_bootstrap_resamples_whole_clusters():
    out = agree.bootstrap([1, 1, 0, None], ["p", "p", "q", "q"], replicates=500, seed=1)
    assert out["estimate"] == pytest.approx(2 / 3) and out["n"] == 3 and out["clusters"] == 2
    assert 0 <= out["low"] <= out["estimate"] <= out["high"] <= 1
    assert agree.bootstrap([None], ["p"], replicates=10, seed=1)["estimate"] is None


def test_analysis_deduplicates_and_reports(tmp_path):
    complete = cell("f1", ["a", "b"], ["a"], {"a": 4, "b": 0}, prompt="p1")
    partial = cell("f1", ["a", "b"], ["a"], {"a": None, "b": None}, prompt="p1")
    other = cell("f2", ["a", "b", "c"], ["b", "a"], {"a": 2, "b": 3, "c": 0}, prompt="p2", model="llama4")
    absent = cell("f3", ["a"], ["a"], {"a": None}, prompt="p3")
    failed = cell("f4", [], [], {}, prompt="p4", status="ranking_outside_evidence")
    write(tmp_path / "hub/plan-a/shards/shard-0001/cells.jsonl.gz", [partial, absent])
    write(tmp_path / "hub/plan-b/shards/shard-0001/cells.jsonl.gz", [complete, other, failed])
    report = agree.analyze([tmp_path / "hub"], tmp_path / "out", replicates=50)
    assert report["counts"]["cells_read"] == 5 and report["counts"]["duplicate_cells"] == 1
    assert report["counts"]["unique_cells"] == 4 and report["counts"]["complete_cells"] == 2
    assert report["excluded_cells_by_reason"] == {"source_status:global_absence_only": 1,
                                                  "cell_status:ranking_outside_evidence": 1}
    overall = report["strata"]["all"]
    assert overall["keep_auc"]["estimate"] == 1.0 and overall["keep_auc"]["n"] == 2
    assert overall["order_tau_b"]["n"] == 1 and overall["order_tau_b"]["estimate"] == 1.0
    assert set(report["strata"]) >= {"model:qwen38", "model:llama4", "method:llama4:m"}
    assert report["kept_share_by_grade"]["0"] == {"sources": 2, "kept": 0, "kept_share": 0.0}
    assert report["source_level_kappa"]["kept_vs_graded_1_or_more"]["kappa"] == 1.0
    with gzip.open(tmp_path / "out/sources.csv.gz", "rt") as f:
        sources = list(csv.DictReader(f))
    assert len(sources) == 2 + 3 + 1  # every shown source of the kept copies, incomplete cells included
    assert {r["fingerprint"] for r in sources if r["cell_complete"] == "0"} == {"f3"}
    with pytest.raises(FileExistsError):
        agree.analyze([tmp_path / "hub"], tmp_path / "out", replicates=0)


def test_run_root_reads_each_shards_latest_report(tmp_path):
    shard = tmp_path / "run/shards/shard-0001"
    write(shard / "results/reports/old/cells.jsonl.gz", [cell("f1", ["a", "b"], ["b"], {"a": 5, "b": 0})])
    write(shard / "results/reports/new/cells.jsonl.gz", [cell("f1", ["a", "b"], ["a"], {"a": 5, "b": 0})])
    (shard / "results/reports/latest.json").write_text(json.dumps({"directory": "new"}))
    (tmp_path / "run/plan.json").write_text(json.dumps({"shards": [{"id": "shard-0001", "directory": str(shard)},
                                                                   {"id": "shard-0002", "directory": str(tmp_path / "none")}]}))
    report = agree.analyze([tmp_path / "run"], tmp_path / "out", replicates=0)
    assert len(report["files"]) == 1 and report["files"][0]["path"].endswith("new/cells.jsonl.gz")
    assert report["strata"]["all"]["keep_auc"]["estimate"] == 1.0
    (shard / "results/reports/latest.json").write_text(json.dumps({"directory": "../x"}))
    with pytest.raises(ValueError, match="invalid report pointer"):
        agree.analyze([tmp_path / "run"], tmp_path / "out2", replicates=0)
