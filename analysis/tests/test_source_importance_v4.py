"""Observable v4 contracts: span fidelity, frozen tasks, failures and resume."""
import asyncio
import copy
import gzip
import hashlib
import json
import sqlite3

import pytest

from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts import prepare_source_importance_tasks as prepare
from analysis.scripts import run_source_importance_judge as runner
from analysis.scripts.run_acl_arr_vllm import _execute_one

ANSWER = "Choose Acme for offline CSV export. It exports CSV without an internet connection. Its interface is blue."
MAP = {"status": "ready", "claims": [
    {"spans": [{"first": "a1w1", "last": "a1w6"}], "kind": "recommendation", "roles": ["central"]},
    {"spans": [{"first": "a2w1", "last": "a2w7"}], "kind": "assertion", "roles": ["major"]},
    {"spans": [{"first": "a3w1", "last": "a3w4"}], "kind": "assertion", "roles": ["peripheral"]}],
    "excluded": [], "note": ""}
SOURCE = "Choose Acme when you need offline CSV export. Acme exports CSV without an internet connection."
CONFIG = {"model_id": "fixture", "model_revision": "a" * 40, "chat_template_sha256": "b" * 64,
          "chat_template_kwargs": {"enable_thinking": False}, "precision": "bfloat16", "serving_version": "fixture",
          "hardware": "CPU fixture, no scientific result", "gpu_count": 4, "tensor_parallel_size": 4,
          "concurrency": 2, "context_length": 73728, "tokenizer_path": "/unused-fixture",
          "request_timeout": 300, "repetition_id": "first"}


def mapped():
    return v4.validate_map(MAP, answer=ANSWER)


def support(grade=5):
    return {"status": "scored", "findings": [
        {"claim_id": "c1", "relation": "full", "spans": MAP["claims"][0]["spans"],
         "evidence_ids": ["text1"], "note": ""},
        {"claim_id": "c2", "relation": "full", "spans": MAP["claims"][1]["spans"],
         "evidence_ids": ["text2"], "note": ""}], "importance": grade,
        "note": "Supports the recommendation and its main justification."}


def item():
    return v4.prepare_source_task(request="Which offline CSV exporter?", answer=ANSWER,
                                  answer_map=mapped(), title="", text=SOURCE)


def test_map_and_source_resolve_original_spans_and_preserve_rubric():
    resolved = mapped()
    assert resolved["claims"][1]["spans"][0]["text"] == "It exports CSV without an internet connection."
    result = item()["validator"](support())
    assert result["importance"] == 5
    assert result["claim_relations"]["c3"] == ["unsupported"]
    assert result["findings"][0]["evidence"][0]["text"] == "Choose Acme when you need offline CSV export."
    for line in v3.INSTRUCTIONS.splitlines():
        if line[:1] in "012345" and " = " in line:
            assert line in v4.SOURCE_INSTRUCTIONS


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["claims"].pop(), "uncovered"),
    (lambda m: m["claims"][0]["roles"].append("peripheral"), "adjacent"),
    (lambda m: m["claims"][0]["spans"][0].update(last="a2w1"), "range"),
    (lambda m: m["claims"][0]["spans"][0].update(first="a99w1"), "unknown"),
    (lambda m: m["claims"].append(copy.deepcopy(m["claims"][0])), "duplicate"),
    (lambda m: m["excluded"].append({"span": {"first": "a1w1", "last": "a1w1"}, "reason": "non_substantive"}), "overlaps"),
])
def test_invalid_maps_do_not_produce_a_usable_map(mutate, match):
    raw = copy.deepcopy(MAP)
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        v4.validate_map(raw, answer=ANSWER)


def test_unicode_shared_context_and_canonical_map_identity():
    answer = "Café exports A and B."
    raw = {"status": "ready", "claims": [
        {"spans": [{"first": "a1w1", "last": "a1w3"}], "kind": "assertion", "roles": ["central"]},
        {"spans": [{"first": "a1w1", "last": "a1w2"}, {"first": "a1w4", "last": "a1w5"}],
         "kind": "assertion", "roles": ["secondary", "major"]}], "excluded": [], "note": ""}
    first = v4.validate_map(raw, answer=answer)
    raw["claims"].reverse()
    second = v4.validate_map(raw, answer=answer)
    assert first == second
    assert first["claims"][0]["spans"][0]["text"] == "Café exports"
    v4.verify_map(first, answer)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(importance=True), "integer"),
    (lambda r: r.update(importance=0), "disagree"),
    (lambda r: r.update(findings=[]), "disagree"),
    (lambda r: r["findings"][0]["evidence_ids"].append("text1"), "duplicate"),
    (lambda r: r["findings"][0].update(evidence_ids=["text999"]), "invalid evidence"),
    (lambda r: r["findings"][0]["spans"][0].update(last="a1w2"), "full finding"),
    (lambda r: r.update(status="uncertain"), "null"),
])
def test_invalid_source_results_are_not_retained(mutate, match):
    raw = copy.deepcopy(support())
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        item()["validator"](raw)


def test_partial_support_and_uncertainty_preserve_distinct_meanings():
    raw = support(2)
    raw["findings"] = [copy.deepcopy(raw["findings"][1])]
    raw["findings"][0].update(relation="partial", spans=[{"first": "a2w1", "last": "a2w3"}],
                               note="CSV export is supported; offline operation is not.")
    result = item()["validator"](raw)
    assert result["findings"][0]["spans"][0]["text"] == "It exports CSV"
    uncertain = item()["validator"]({"status": "uncertain", "findings": [], "importance": None,
                                     "note": "Scope cannot be resolved."})
    assert uncertain["claim_relations"]["c1"] == ["unresolved"]


def test_global_absence_and_pure_abstention_are_not_zero():
    answer = "The supplied evidence contains no answer."
    raw = {"status": "ready", "claims": [{"spans": [{"first": "a1w1", "last": "a1w6"}],
           "kind": "global_absence", "roles": ["central"]}], "excluded": [], "note": ""}
    result = v4.validate_map(raw, answer=answer)
    assert result["eligibility"] == "global_absence_only"
    with pytest.raises(ValueError, match="ineligible"):
        v4.prepare_source_task(request="Q?", answer=answer, answer_map=result, title="", text="Evidence.")
    abstain = v4.validate_map({"status": "ready", "claims": [], "excluded": [
        {"span": {"first": "a1w1", "last": "a1w3"}, "reason": "pure_abstention"}], "note": ""},
        answer="I cannot answer.")
    assert abstain["eligibility"] == "no_substantive_content"


def test_freeze_and_map_hash_prevent_semantic_cache_collisions():
    first = item()
    assert v4.item_from_record(first["record"])["prompt"] == first["prompt"]
    corrupt = copy.deepcopy(first["record"])
    corrupt["inputs"]["answer_map"]["claims"][0]["spans"][0]["text"] = "Invented"
    with pytest.raises(ValueError, match="map does not reproduce"):
        v4.item_from_record(corrupt)
    alternate = copy.deepcopy(MAP)
    alternate["claims"][1]["roles"] = ["secondary"]
    second = v4.prepare_source_task(request="Which offline CSV exporter?", answer=ANSWER,
        answer_map=v4.validate_map(alternate, answer=ANSWER), title="", text=SOURCE)
    assert first["record"]["judge_task_id"] != second["record"]["judge_task_id"]
    new_mask = v4.prepare_map_task(request="Q?", answer=ANSWER, preprocessing=v4.PROSE_MASK_VERSION)
    old_mask = v4.prepare_map_task(request="Q?", answer=ANSWER)
    assert new_mask["record"]["judge_task_id"] != old_mask["record"]["judge_task_id"]


def test_prose_mask_is_versioned_reversible_and_keeps_narration():
    answer = "Snippet S1 says Café works [S2]. Vitamin S1 is unrelated."
    old, _ = v4.mask_answer(answer, [])
    new, spans = v4.mask_answer(answer, [], v4.PROSE_MASK_VERSION)
    assert old.startswith("Snippet S1")
    assert new == "Snippet [ref] says Café works [ref]. Vitamin S1 is unrelated."
    assert v3.unmask_answer(new, spans) == answer


def freeze(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    mapper = v4.prepare_map_task(request="Which offline CSV exporter?", answer=ANSWER)["record"]
    dep = v4.source_dependency(mapper["judge_task_id"], "", SOURCE)
    j1 = v3.task_record(v3.prepare_fulfilment_task(request="Which offline CSV exporter?", answer=ANSWER))
    tasks = [mapper, dep, j1]
    cells = [{"cell_id": "cell1", "status": "ok", "model": "qwen38", "method": "Parallel-Expansion-v1",
              "engine": "ddg", "condition": "natural", "map_task_id": mapper["judge_task_id"],
              "j1_task_id": j1["judge_task_id"], "presented": ["https://acme.example"],
              "judged_answer_sha256": hashlib.sha256(ANSWER.encode()).hexdigest(),
              "masked_answer_sha256": hashlib.sha256(ANSWER.encode()).hexdigest(),
              "generator_ranking": ["https://acme.example"], "provenance": {"status": "verified"},
              "sources": [{"url": "https://acme.example", "dependency_id": dep["judge_task_id"], "status": "awaiting_map",
                           "source_sha256": v4._digest({"title": "", "text": SOURCE})}]}]
    files = {}
    for name, data in (("tasks", tasks), ("cells", cells)):
        path = root / f"{name}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            for row in data:
                stream.write(json.dumps(row) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"protocol": v4.PROTOCOL, "files": files}))
    return root, mapper


class Responses:
    def __init__(self, *, bad_map=False, map_issue=False, interrupt_source=False):
        self.calls = []
        self.bad_map, self.map_issue, self.interrupt_source = bad_map, map_issue, interrupt_source

    async def complete(self, **kw):
        self.calls.append(kw)
        if kw["schema_name"] == "answer_map_v4":
            raw = {"status": "unusable", "claims": [], "excluded": [], "note": "Unresolved meaning."} if self.bad_map else MAP
        elif kw["schema_name"] == "source_importance_v4":
            if self.interrupt_source:
                raise asyncio.CancelledError()
            raw = {"status": "map_issue", "findings": [], "importance": None, "note": "Central claim misrepresented."} if self.map_issue else support()
        else:
            raw = {"request_fulfillment": 5}
        return json.dumps(raw), {"finish_reason": "stop", "completion_tokens": 30, "prompt_tokens": 100}


def test_end_to_end_uses_one_map_then_source_and_resumes_without_calls(tmp_path):
    inputs, _ = freeze(tmp_path)
    output = tmp_path / "output"
    client = Responses()
    coordinator = runner.Coordinator(inputs, output, CONFIG)
    asyncio.run(coordinator.execute(client))
    report = coordinator.report()
    assert report["counts"]["cells_complete"] == 1
    assert [c["schema_name"] for c in client.calls].count("answer_map_v4") == 1
    assert [c["schema_name"] for c in client.calls].index("answer_map_v4") < [c["schema_name"] for c in client.calls].index("source_importance_v4")
    coordinator.close()
    resumed = runner.Coordinator(inputs, output, CONFIG)
    again = Responses()
    asyncio.run(resumed.execute(again))
    assert again.calls == []
    resumed.close()
    with pytest.raises(ValueError, match="immutable contract"):
        runner.Coordinator(inputs, output, {**CONFIG, "repetition_id": "second"})


@pytest.mark.parametrize("bad_map,map_issue,status", [(True, False, "map_unusable"), (False, True, "map_quarantined")])
def test_failed_or_challenged_map_keeps_scores_missing(tmp_path, bad_map, map_issue, status):
    inputs, _ = freeze(tmp_path)
    coordinator = runner.Coordinator(inputs, tmp_path / "output", CONFIG)
    client = Responses(bad_map=bad_map, map_issue=map_issue)
    asyncio.run(coordinator.execute(client))
    report = coordinator.report()
    assert report["counts"]["sources_" + status] == 1
    assert report["counts"]["cells_complete"] == 0
    if bad_map:
        assert all(c["schema_name"] != "source_importance_v4" for c in client.calls)
    coordinator.close()


def test_interrupted_source_does_not_repeat_saved_map(tmp_path):
    inputs, _ = freeze(tmp_path)
    output = tmp_path / "output"
    coordinator = runner.Coordinator(inputs, output, CONFIG)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(coordinator.execute(Responses(interrupt_source=True)))
    coordinator.close()
    resumed = runner.Coordinator(inputs, output, CONFIG)
    client = Responses()
    asyncio.run(resumed.execute(client))
    assert all(c["schema_name"] != "answer_map_v4" for c in client.calls)
    assert resumed.report()["counts"]["cells_complete"] == 1
    resumed.close()


def test_context_overflow_is_rejected_before_transport():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return list(range(30))
    client = Responses()
    checked = runner.ContextCheckedClient(client, Tokenizer(), {**CONFIG, "context_length": 40})
    with pytest.raises(ValueError, match="not truncated"):
        asyncio.run(checked.complete(prompt="data", max_tokens=20))
    assert client.calls == []


def test_v4_truncation_is_terminal_without_repeating_same_budget():
    class Truncated:
        calls = 0
        async def complete(self, **kwargs):
            self.calls += 1
            return json.dumps(MAP), {"finish_reason": "length"}
    client = Truncated()
    result = asyncio.run(_execute_one(v4.prepare_map_task(request="Q?", answer=ANSWER), client=client, fake=False))
    assert not result["ok"] and client.calls == 1
    assert "parsed_output" not in result


@pytest.mark.parametrize("kind", ["map", "source"])
def test_corrective_retry_reports_missing_words_or_grade_support_conflict(kind):
    good = copy.deepcopy(MAP if kind == "map" else support())
    bad = copy.deepcopy(good)
    if kind == "map":
        bad["claims"].pop()
    else:
        bad["findings"] = []
    class Responses:
        def __init__(self):
            self.calls = []
        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            return json.dumps(bad if len(self.calls) == 1 else good), {"finish_reason": "stop"}
    client = Responses()
    task = v4.prepare_map_task(request="Q?", answer=ANSWER) if kind == "map" else item()
    result = asyncio.run(_execute_one(task, client=client, fake=False))
    assert result["ok"] and len(client.calls) == 2
    feedback = json.loads(client.calls[1]["prompt"].split("VALIDATION_FEEDBACK_JSON=", 1)[1])["error"]
    if kind == "map":
        assert all(word in feedback for word in ("a3w1", "a3w2", "a3w3", "a3w4"))
    else:
        assert "importance=5" in feedback and "full or partial" in feedback
        assert "do not invent" in feedback


def test_exhausted_source_validation_stays_missing_and_is_not_retried_on_resume(tmp_path):
    class InvalidSource(Responses):
        async def complete(self, **kw):
            if kw["schema_name"] == "source_importance_v4":
                self.calls.append(kw)
                return json.dumps(support(0)), {"finish_reason": "stop"}
            return await super().complete(**kw)
    inputs, _ = freeze(tmp_path)
    output = tmp_path / "output"
    coordinator = runner.Coordinator(inputs, output, CONFIG)
    client = InvalidSource()
    asyncio.run(coordinator.execute(client))
    report = coordinator.report()
    assert report["counts"]["sources_inference_failed"] == 1
    assert report["status"] == "finished_with_failures"
    assert len([r for r in client.calls if r["schema_name"] == "source_importance_v4"]) == 2
    assert any(e["state"] == "terminal_failed" for e in coordinator.ledger.snapshot()["latest"].values())
    coordinator.close()
    resumed = runner.Coordinator(inputs, output, CONFIG)
    again = Responses()
    asyncio.run(resumed.execute(again))
    assert not again.calls
    resumed.close()


def test_reporting_preserves_flat_ties_and_independent_presentation_baseline():
    result = v4.cell_metrics({"a": 2, "b": 2}, generator_list=[], presented=["a", "b"])
    assert result["top_source_alignment"] is None
    assert result["first_presented_alignment"] is True
    assert result["presented_chance_baseline"] == 1
    assert result["top_group_size"] == 2
    assert result["ordering_tau_b"] is None and result["important_source_omission"] is None
    incomplete = v4.cell_metrics({"a": 2, "b": None}, generator_list=["a"], presented=["a", "b"])
    assert incomplete["top_source_alignment"] is None


def test_fixed_map_repetition_bypasses_mapping_but_not_source_calls(tmp_path):
    inputs, mapper = freeze(tmp_path)
    fixed = tmp_path / "maps.jsonl"
    fixed.write_text(json.dumps({"judge_task_id": mapper["judge_task_id"], "raw_output": json.dumps(MAP)}) + "\n")
    coordinator = runner.Coordinator(inputs, tmp_path / "fixed", CONFIG, fixed_maps=fixed)
    client = Responses()
    asyncio.run(coordinator.execute(client))
    assert all(c["schema_name"] != "answer_map_v4" for c in client.calls)
    assert any(c["schema_name"] == "source_importance_v4" for c in client.calls)
    assert coordinator.report()["fixed_maps_sha256"] == hashlib.sha256(fixed.read_bytes()).hexdigest()
    coordinator.close()


def test_unindexed_durable_response_is_reconciled_without_another_model_call(tmp_path):
    inputs, _ = freeze(tmp_path)
    output = tmp_path / "output"
    coordinator = runner.Coordinator(inputs, output, CONFIG)
    original = coordinator.db

    class FailIndexOnce:
        failed = False
        def execute(self, sql, params=()):
            if "SET state='saved',result=" in sql and not self.failed:
                self.failed = True
                raise sqlite3.OperationalError("simulated index write failure after durable response")
            return original.execute(sql, params)
        def __getattr__(self, name):
            return getattr(original, name)

    coordinator.db = FailIndexOnce()
    first = Responses()
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(coordinator.execute(first))
    writer_id = coordinator.writer_id
    coordinator.close()
    with pytest.raises(ValueError, match="terminal-owner"):
        runner.Coordinator(inputs, output, CONFIG)
    confirmation = tmp_path / "terminal.json"
    confirmation.write_text(json.dumps([{"writer_id": writer_id, "slurm_job_id": None,
        "owner_terminal": True, "evidence": "CPU fixture coordinator has been closed"}]))
    resumed = runner.Coordinator(inputs, output, CONFIG, recovery_evidence=confirmation)
    client = Responses()
    asyncio.run(resumed.execute(client))
    assert all(c["schema_name"] != "answer_map_v4" for c in client.calls)
    assert resumed.report()["counts"]["cells_complete"] == 1
    resumed.close()


def test_admission_failure_preserves_queue_without_sending_requests(tmp_path):
    inputs, _ = freeze(tmp_path)
    coordinator = runner.Coordinator(inputs, tmp_path / "output", CONFIG,
        admission_check=lambda: {"safe_to_admit": False, "quota_verified": False, "reasons": ["stale quota"]})
    client = Responses()
    asyncio.run(coordinator.execute(client))
    assert not client.calls
    assert coordinator.report()["status"] == "incomplete"
    coordinator.close()


def test_diagnostic_renaming_preserves_evidence_and_creates_new_identity():
    base = item()
    changed = v4.prepare_source_task(request="Which offline CSV exporter?", answer=ANSWER, answer_map=mapped(),
        title="", text=SOURCE, diagnostic_passage_ids={"text1": "text101", "text2": "text102"})
    raw = copy.deepcopy(support())
    raw["findings"][0]["evidence_ids"] = ["text101"]
    raw["findings"][1]["evidence_ids"] = ["text102"]
    assert changed["validator"](raw)["findings"][0]["evidence"][0]["text"] == "Choose Acme when you need offline CSV export."
    assert changed["record"]["judge_task_id"] != base["record"]["judge_task_id"]
    assert v4.item_from_record(changed["record"])["prompt"] == changed["prompt"]


def test_constructed_freezes_are_disjoint_and_importable(tmp_path):
    from analysis.scripts.prepare_si_v4_diagnostics import freeze as diagnostics
    manifests = [diagnostics(tmp_path / split, split) for split in ("development", "heldout")]
    assert [m["source_pairs"] for m in manifests] == [48, 48]
    ids = [{c["cell_id"] for c in runner.rows(tmp_path / split / "cells.jsonl.gz")}
           for split in ("development", "heldout")]
    assert not ids[0] & ids[1]
    coordinator = runner.Coordinator(tmp_path / "development", tmp_path / "output", CONFIG)
    assert coordinator.db.execute("SELECT COUNT(*) FROM tasks WHERE kind='source_dependency'").fetchone()[0] > 0
    coordinator.close()


def test_comparison_binds_inputs_references_and_uncached_repetition(tmp_path):
    from analysis.scripts.compare_source_importance_runs import compare
    inputs, _ = freeze(tmp_path)
    reports = []
    for repetition in ("one", "two", "three"):
        coordinator = runner.Coordinator(inputs, tmp_path / repetition, {**CONFIG, "repetition_id": repetition})
        asyncio.run(coordinator.execute(Responses()))
        coordinator.report()
        reports.append(coordinator.output / "reports" / coordinator.writer_id)
        coordinator.close()
    cell = next(runner.rows(reports[1] / "cells.jsonl.gz"))
    source = cell["sources"][0]
    reference = {"cell_id": cell["cell_id"], "url": source["url"], "source_sha256": source["source_sha256"],
                 "masked_answer_sha256": cell["masked_answer_sha256"], "reviewer_models": ["fixture-reference"],
                 "frozen_before_grade_review": True, "acceptable_grade_range": [4, 5], "keyword_id": "k1",
                 "candidate_raw_output_sha256": source["raw_output_sha256"], "candidate_pair_support": ["supported", "supported"]}
    refs = tmp_path / "references.jsonl"
    refs.write_text(json.dumps(reference) + "\n")
    result = compare(reports[0], reports[1], references=refs, repetitions=[reports[2]], replicates=20)
    assert result["candidate_range_agreement"]["fraction"] == 1
    assert result["resolved_pair_support"]["denominator"] == 2
    assert result["repeats"]["exact"] == result["repeats"]["grade_pairs"] == 1
    assert result["paired_keyword_interval"]["clusters"] == 1
    with pytest.raises(ValueError, match="reused an execution"):
        compare(reports[0], reports[1], repetitions=[reports[1]])
    reference["source_sha256"] = "different"
    refs.write_text(json.dumps(reference) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(reports[0], reports[1], references=refs)
