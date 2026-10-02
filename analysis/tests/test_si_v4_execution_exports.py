"""Saved-map repeats preserve repair evidence, failures and constructed boundaries."""
import asyncio
import copy
import gzip
import hashlib
import json

import pytest

from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts import run_source_importance_judge as runner
from analysis.tests.test_source_importance_v4 import ANSWER, CONFIG, MAP, SOURCE, freeze, marker_conflict


def inputs_for(tmp_path, *, answer=ANSWER, diagnostic_seed=None, constructed=False):
    inputs, _ = freeze(tmp_path)
    mapper = v4.prepare_map_task(request="Which offline CSV exporter?", answer=answer,
                                 diagnostic_seed=diagnostic_seed)["record"]
    dependency = v4.source_dependency(mapper["judge_task_id"], "", SOURCE)
    j1 = v3.task_record(v3.prepare_fulfilment_task(request="Which offline CSV exporter?", answer=answer))
    cell = next(runner.rows(inputs / "cells.jsonl.gz"))
    cell.update(map_task_id=mapper["judge_task_id"], j1_task_id=j1["judge_task_id"],
                judged_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                masked_answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
    cell["sources"][0]["dependency_id"] = dependency["judge_task_id"]
    manifest = json.loads((inputs / "manifest.json").read_text())
    manifest["constructed"] = constructed
    for name, rows in (("tasks", [mapper, dependency, j1]), ("cells", [cell])):
        path = inputs / (name + ".jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        manifest["files"][path.name] = runner.file_hash(path)
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    return inputs, mapper


class MapResponse:
    def __init__(self, raw, *, map_issue=False):
        self.raw, self.map_issue, self.calls = raw, map_issue, []

    async def complete(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "answer_map_v4":
            output = self.raw
        else:
            output = {"status": "map_issue" if self.map_issue else "scored", "findings": [],
                      "importance": None if self.map_issue else 0,
                      "note": "Map role needs independent review." if self.map_issue else
                              "The unrelated source supports none of this answer."}
        return json.dumps(output), {"finish_reason": "stop", "completion_tokens": 30, "prompt_tokens": 100}


def run_report(inputs, output, client, *, fixed_maps=None):
    coordinator = runner.Coordinator(inputs, output, {**CONFIG, "source_importance_only": True},
                                     fixed_maps=fixed_maps)
    try:
        asyncio.run(coordinator.execute(client))
        summary = coordinator.report()
        latest = json.loads((output / "reports/latest.json").read_text())
        report = output / "reports" / latest["directory"]
        cell = next(runner.rows(report / "cells.jsonl.gz"))
        return report, summary, cell
    finally:
        coordinator.close()


def test_repaired_map_exports_into_fixed_repeat_without_repairing_or_generating_again(tmp_path):
    answer, raw = marker_conflict()
    inputs, _ = inputs_for(tmp_path, answer=answer)
    report, _, first = run_report(inputs, tmp_path / "first", MapResponse(raw))
    exported = next(runner.rows(report / "maps.jsonl"))
    assert first["map_result"]["raw_output"] == exported["raw_output"] == json.dumps(raw)
    assert exported["deterministic_repairs"] == first["map_result"]["deterministic_repairs"]
    assert exported["repaired_output"] == first["map_result"]["repaired_output"]
    assert json.loads(first["sources"][0]["raw_output"])["importance"] == first["sources"][0]["parsed_output"]["importance"]
    assert hashlib.sha256(first["sources"][0]["raw_output"].encode()).hexdigest() == first["sources"][0]["raw_output_sha256"]

    client = MapResponse(raw)
    _, summary, repeated = run_report(inputs, tmp_path / "repeat", client, fixed_maps=report / "maps.jsonl")
    assert client.calls == ["source_importance_v4"]
    assert repeated["map_sha256"] == first["map_sha256"]
    for field in ("raw_output", "repaired_output", "deterministic_repairs"):
        assert repeated["map_result"][field] == first["map_result"][field]
    assert summary["request_totals"].get("answer_map_requests", 0) == 0

    exported["deterministic_repairs"][0]["word_id"] = "a1w3"
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(json.dumps(exported) + "\n")
    with pytest.raises(ValueError, match="repair does not reproduce"):
        runner.Coordinator(inputs, tmp_path / "tampered-output", CONFIG, fixed_maps=tampered)


@pytest.mark.parametrize("kind", ["failed", "quarantined"])
def test_failed_and_quarantined_maps_stay_missing_in_fixed_repeats(tmp_path, kind):
    if kind == "failed":
        answer, raw = ANSWER, copy.deepcopy(MAP)
        raw["excluded"] = [{"span": {"first": "a1w3", "last": "a1w3"}, "reason": "non_substantive"}]
    else:
        answer, raw = marker_conflict()
    inputs, _ = inputs_for(tmp_path, answer=answer)
    report, _, first = run_report(inputs, tmp_path / "first", MapResponse(raw, map_issue=kind == "quarantined"))
    exported = next(runner.rows(report / "maps.jsonl"))
    assert exported["ok"] is False
    assert exported["status"] == "map_" + kind
    client = MapResponse(raw)
    _, summary, repeated = run_report(inputs, tmp_path / "repeat", client, fixed_maps=report / "maps.jsonl")
    assert client.calls == []
    assert summary["counts"]["cells_complete"] == 0
    assert repeated["sources"][0]["importance"] is None
    assert repeated["sources"][0]["status"] == "map_" + kind
    assert repeated["map_result"]["raw_output"] == first["map_result"]["raw_output"]
    if kind == "quarantined":
        assert repeated["map_result"]["deterministic_repairs"] == first["map_result"]["deterministic_repairs"]
        assert repeated["map_result"]["repaired_output"] == first["map_result"]["repaired_output"]


@pytest.mark.parametrize("constructed", [False, True])
def test_matched_map_seed_is_allowed_only_for_constructed_inputs(tmp_path, constructed):
    inputs, mapper = inputs_for(tmp_path, diagnostic_seed=20261002, constructed=constructed)
    if not constructed:
        with pytest.raises(ValueError, match="diagnostic interventions require a constructed"):
            runner.Coordinator(inputs, tmp_path / "out", CONFIG)
    else:
        coordinator = runner.Coordinator(inputs, tmp_path / "out", CONFIG)
        try:
            item = next(coordinator.ready())
            assert item["seed"] == mapper["seed"] == 20261002
        finally:
            coordinator.close()
