"""Public freeze-partition contracts: coverage, ownership, integrity and memory."""
import copy
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import tracemalloc

import pytest

from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts.partition_si_v4_tasks import partition
from analysis.scripts.run_source_importance_judge import Coordinator, file_hash, rows


def make_records(specs):
    cells, tasks = [], {}
    for spec in specs:
        request, answer = spec.get("request", "Which tool?"), spec.get("answer", spec["id"] + " works.")
        url = spec.get("url", "https://source.example/" + spec["id"])
        masked, mask_spans = v4.mask_answer(answer, [url])
        mapper = v4.prepare_map_task(request=request, answer=masked)["record"]
        text = spec.get("text", "Café exports files without changing ‘quotes’ or whitespace.  Two spaces.")
        dependency = v4.source_dependency(mapper["judge_task_id"], "Exact source title", text)
        j1 = v3.task_record(v3.prepare_fulfilment_task(request=request, answer=answer, max_tokens=64))
        for task in (mapper, dependency, j1):
            tasks[task["judge_task_id"]] = task
        cell = {"fingerprint": "fingerprint-" + spec["id"], "cell_id": spec["id"], "status": "ok",
                "model": spec.get("model", "qwen38"), "prompt_id": spec.get("prompt", spec["id"]),
                "method": "Parallel-Expansion-v1", "engine": "ddg", "condition": "natural",
                "answer_source": "stored", "map_task_id": mapper["judge_task_id"], "j1_task_id": j1["judge_task_id"],
                "masked_answer_sha256": hashlib.sha256(masked.encode()).hexdigest(), "answer_mask_spans": mask_spans,
                "judged_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "provenance": {"status": "verified", "source_trace_sha256": "a" * 64},
                "presented": [url], "generator_ranking": [],
                "sources": [{"url": url,
                             "dependency_id": dependency["judge_task_id"], "map_task_id": mapper["judge_task_id"],
                             "status": "awaiting_map", "presented_position": 0,
                             "source_sha256": v4._digest({"title": "Exact source title", "text": text})}]}
        if "rank" in spec:
            cell["keyword_memberships"] = {"primary_priority_rank": spec["rank"],
                                           "primary_keyword_id": spec.get("keyword", f"keyword-{spec['rank']}"),
                                           "keyword_ids": [spec.get("keyword", f"keyword-{spec['rank']}")]}
        elif "metadata" in spec:
            cell["task_metadata"] = spec["metadata"]
        cells.append(cell)
    return cells, list(tasks.values())


def freeze(path, cells, tasks, **changes):
    path.mkdir()
    files = {}
    for name, records in (("cells", cells), ("tasks", tasks)):
        target = path / f"{name}.jsonl.gz"
        with gzip.open(target, "wb") as stream:
            for record in records:
                # Noncanonical formatting is intentional: repartitioning must
                # retain the exact original JSONL record, including provenance.
                stream.write(b"  " + json.dumps(record, ensure_ascii=False).encode() + b" \n")
        files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": v4.PROTOCOL,
                "task_version": v4.TASK_VERSION, "retry_contract": v4.RETRY_CONTRACT,
                "max_tokens": 4096, "map_max_tokens": 4096, "j1_max_tokens": 64,
                "preprocessing": v4.PREPROCESSING_VERSION, "eligibility_version": v4.ELIGIBILITY_VERSION,
                "truncation_sensitivity": {"fraction": 0}, "counts": {}, "files": files,
                "scientific_result": False, **changes}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def raw_records(path, key):
    with gzip.open(path, "rb") as stream:
        return {json.loads(line)[key]: line for line in stream if line.strip()}


def test_shared_maps_preserve_all_cells_tasks_bytes_and_coordinator_contract(tmp_path):
    cells, tasks = make_records([
        {"id": "late-shared", "answer": "Shared answer.", "model": "llama4", "rank": 30},
        {"id": "middle", "rank": 10},
        {"id": "early-shared", "answer": "Shared answer.", "rank": 0},
        {"id": "later", "model": "llama4", "metadata": {"priority_rank": 20, "keyword_id": "keyword-20"}},
        {"id": "unranked"},
    ])
    blocked = {"cell_id": "blocked", "fingerprint": "blocked", "status": "evidence_count_mismatch", "model": "qwen38"}
    inputs = freeze(tmp_path / "inputs", cells + [blocked], tasks)
    result = partition(inputs, tmp_path / "output", max_shards=2, minimum_cells_per_shard=2)
    assert [s["cells"] for s in result] == [2, 3]
    originals = raw_records(inputs / "cells.jsonl.gz", "cell_id")
    original_tasks = raw_records(inputs / "tasks.jsonl.gz", "judge_task_id")
    seen_cells, seen_tasks, seen_maps = [], set(), set()
    for shard in result:
        assert set(shard) == {"id", "directory", "cells"}
        assert Path(shard["directory"]).is_absolute()
        target = Path(shard["directory"]) / "inputs"
        shard_cells = raw_records(target / "cells.jsonl.gz", "cell_id")
        shard_tasks = raw_records(target / "tasks.jsonl.gz", "judge_task_id")
        seen_cells.extend(shard_cells)
        assert all(raw == originals[cid] for cid, raw in shard_cells.items())
        assert all(raw == original_tasks[tid] for tid, raw in shard_tasks.items())
        maps = {r["map_task_id"] for r in rows(target / "cells.jsonl.gz")}
        assert not seen_maps & maps
        seen_maps |= maps
        non_j1 = {r["judge_task_id"] for r in rows(target / "tasks.jsonl.gz") if r["task"] != "fulfilment"}
        assert not seen_tasks & non_j1
        seen_tasks |= set(shard_tasks)
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["cells"] == manifest["counts"]["cells_ok"] == shard["cells"]
        assert manifest["source_manifest_sha256"] == file_hash(inputs / "manifest.json")
        with closing_coordinator(target, tmp_path / (shard["id"] + "-run")) as coordinator:
            report = coordinator.report()
            assert report["counts"]["cells"] == shard["cells"]
            assert report["states"]["not_requested"] == manifest["counts"]["unique_fulfilment_tasks"]
    assert seen_cells == ["early-shared", "late-shared", "middle", "later", "unranked"]
    assert seen_tasks == set(original_tasks)
    summary = json.loads((tmp_path / "output/partition.json").read_text())
    assert summary["counts"]["ready_cells"] == 5
    assert summary["blocked_input_counts"] == {"evidence_count_mismatch": 1}
    assert raw_records(tmp_path / "output/blocked-cells.jsonl.gz", "cell_id") == {"blocked": originals["blocked"]}
    for shard in summary["shards"]:
        assert shard["manifest_sha256"] == file_hash(Path(shard["directory"]) / "inputs/manifest.json")


@contextmanager
def closing_coordinator(inputs, output):
    coordinator = Coordinator(inputs, output, {"source_importance_only": True})
    try:
        yield coordinator
    finally:
        coordinator.close()


def test_shared_j1_can_repeat_without_joining_distinct_map_components(tmp_path):
    # The same original answer can mask differently for distinct evidence hosts,
    # producing different SI maps while retaining the same unrequested J1 task.
    answer = "See https://first.example for details."
    cells, tasks = make_records([
        {"id": "one", "answer": answer, "url": "https://first.example", "rank": 1},
        {"id": "two", "answer": answer, "url": "https://second.example", "rank": 2},
    ])
    assert cells[0]["j1_task_id"] == cells[1]["j1_task_id"]
    assert cells[0]["map_task_id"] != cells[1]["map_task_id"]
    inputs = freeze(tmp_path / "inputs", cells, tasks)
    result = partition(inputs, tmp_path / "output", minimum_cells_per_shard=1)
    assert len(result) == 2
    for shard in result:
        target = Path(shard["directory"]) / "inputs"
        assert [r["judge_task_id"] for r in rows(target / "tasks.jsonl.gz") if r["task"] == "fulfilment"] == [cells[0]["j1_task_id"]]
        with closing_coordinator(target, tmp_path / (shard["id"] + "-run")) as coordinator:
            assert coordinator.report()["states"]["not_requested"] == 1


def test_order_and_compressed_records_are_deterministic_across_input_order(tmp_path):
    cells, tasks = make_records([
        {"id": "z-prompt", "rank": 2, "keyword": "b", "model": "llama4"},
        {"id": "a-prompt", "rank": 2, "keyword": "b", "model": "qwen38"},
        {"id": "keyword-first", "rank": 2, "keyword": "a"},
        {"id": "rank-first", "rank": 1, "keyword": "z"},
    ])
    a = partition(freeze(tmp_path / "a", cells, tasks), tmp_path / "out-a", minimum_cells_per_shard=2)
    b = partition(freeze(tmp_path / "b", list(reversed(cells)), list(reversed(tasks))), tmp_path / "out-b", minimum_cells_per_shard=2)
    flattened = []
    for left, right in zip(a, b):
        assert left["id"] == right["id"] and left["cells"] == right["cells"] == 2
        for name in ("cells.jsonl.gz", "tasks.jsonl.gz"):
            assert (Path(left["directory"]) / "inputs" / name).read_bytes() == (Path(right["directory"]) / "inputs" / name).read_bytes()
        flattened += [c["cell_id"] for c in rows(Path(left["directory"]) / "inputs/cells.jsonl.gz")]
    assert flattened == ["rank-first", "keyword-first", "z-prompt", "a-prompt"]


@pytest.mark.parametrize("name", ["cells.jsonl.gz", "tasks.jsonl.gz"])
def test_stale_frozen_file_hash_prevents_promotion(tmp_path, name):
    cells, tasks = make_records([{"id": "one", "rank": 0}])
    inputs = freeze(tmp_path / "inputs", cells, tasks)
    with (inputs / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        partition(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fault,match", [
    ("missing_map", "missing/non-map"), ("missing_source", "missing/non-source"),
    ("foreign_source", "different answer map"), ("source_hash", "source hash"),
    ("foreign_j1", "differs from map request"), ("orphan", "unreferenced/foreign"),
    ("sensitivity_cell", "sensitivity"), ("sensitivity_manifest", "sensitivity"),
])
def test_reference_and_single_pass_failures_preserve_unpromoted_inputs(tmp_path, fault, match):
    cells, tasks = make_records([{"id": "one", "rank": 0}, {"id": "two", "rank": 1, "request": "Another request?"}])
    changes = {}
    if fault == "missing_map":
        cells[0]["map_task_id"] = "missing"
    elif fault == "missing_source":
        cells[0]["sources"][0]["dependency_id"] = "missing"
    elif fault == "foreign_source":
        cells[0]["sources"][0] = copy.deepcopy(cells[1]["sources"][0])
    elif fault == "source_hash":
        cells[0]["sources"][0]["source_sha256"] = "b" * 64
    elif fault == "foreign_j1":
        cells[0]["j1_task_id"] = cells[1]["j1_task_id"]
    elif fault == "orphan":
        tasks.append(v4.prepare_map_task(request="Unreferenced?", answer="Nothing references this.")["record"])
    elif fault == "sensitivity_cell":
        cells[0]["stored_answer_sensitivity"] = {}
    else:
        changes["truncation_sensitivity"] = {"fraction": 0.1}
    inputs = freeze(tmp_path / "inputs", cells, tasks, **changes)
    with pytest.raises(ValueError, match=match):
        partition(inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_output_is_unchanged_and_shared_component_limits_shard_count(tmp_path):
    cells, tasks = make_records([{"id": "one", "answer": "Same."}, {"id": "two", "answer": "Same.", "model": "llama4"}])
    inputs = freeze(tmp_path / "inputs", cells, tasks)
    result = partition(inputs, tmp_path / "output", minimum_cells_per_shard=1)
    assert len(result) == 1 and result[0]["cells"] == 2
    summary = tmp_path / "output/partition.json"
    before = summary.read_bytes()
    assert json.loads(before)["requested_shards"] == 2
    with pytest.raises(ValueError, match="already exists"):
        partition(inputs, tmp_path / "output")
    assert summary.read_bytes() == before


def test_no_ready_cells_records_blocked_work_without_empty_shards(tmp_path):
    cell = {"cell_id": "blocked", "fingerprint": "blocked", "status": "missing_request_or_answer"}
    inputs = freeze(tmp_path / "inputs", [cell], [])
    assert partition(inputs, tmp_path / "output") == []
    summary = json.loads((tmp_path / "output/partition.json").read_text())
    assert summary["counts"]["ready_cells"] == 0
    assert summary["counts"]["blocked_input_cells"] == 1
    assert list(rows(tmp_path / "output/blocked-cells.jsonl.gz")) == [cell]


def test_partition_does_not_retain_corpus_task_contents_in_python_memory(tmp_path):
    payload = "unchanged source content " * 11000
    cells, tasks = make_records([{"id": f"cell-{i:03d}", "rank": i, "text": payload} for i in range(96)])
    inputs = freeze(tmp_path / "inputs", cells, tasks)
    tracemalloc.start()
    try:
        result = partition(inputs, tmp_path / "output", max_shards=4, minimum_cells_per_shard=24)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert sum(s["cells"] for s in result) == 96
    # A whole-corpus list of the 25 MB source texts exceeds this allowance;
    # streaming one record at a time leaves room for parsing and validation.
    assert peak < len(payload.encode()) * 96 // 2
