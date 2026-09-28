"""SI-v1 task freezing, condition-manipulation audit and keyword-cluster bootstrap on fixture data."""

import gzip
import hashlib
import json
from dataclasses import asdict

import pytest

from analysis.interpretability.pipeline import source_importance as si
from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.cluster_bootstrap import cluster_bootstrap, mean_of, paired_difference
from analysis.interpretability.pipeline.condition_manipulation import classify_pair
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import audit_condition_manipulation as audit
from analysis.scripts import prepare_source_importance_tasks as prepare

A = {"url": "https://a.example/free", "title": "A", "text": "Asana has a free plan."}
B = {"url": "https://b.example/boards", "title": "B", "text": "Trello uses boards."}
C = {"url": "https://c.example/x", "title": "C", "text": "Monday offers a trial."}
ANSWER = "Asana has a free plan. Trello uses boards."


def trace(incoming, selected, *, queries=("free project tools",), raw_ranking=("S2", "S1")):
    events = [{"event_type": "search", "payload": {"query": q, "snippets": [A, B, C]}} for q in queries]
    events += [
        {"event_type": "condition", "payload": {"input_snippets": [A, B, C], "output_snippets": incoming,
                                                "output_urls": [r["url"] for r in incoming]}},
        {"event_type": "compaction", "payload": {"selected_snippets": selected}},
        {"event_type": "llm_call", "payload": {
            "purpose": "parallel_final", "request": {"prompt": "FINAL " + json.dumps(selected)},
            "raw_output": json.dumps({"ranking": list(raw_ranking), "answer": ANSWER}), "validation_error": None}},
    ]
    return {"events": events}


CELLS = {  # condition -> (incoming order after the hook, selected evidence, answer ranking URLs)
    "natural": ([A, B, C], [A, B], [B["url"], A["url"]]),
    "shuffled": ([C, B, A], [A, B], [B["url"], A["url"]]),
    "ablated": ([B, C], [B, C], [B["url"]]),
}


def dataset(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="p", acceptance_policy_id="a")
    writer = FinalDatasetWriter(root, writer_id="fixture")
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=256)
    writer.append("prompts", {"prompt_id": "q1", "prompt_text": "Free project tools?"}, transaction_id="q1")
    pending = []
    for condition, (incoming, selected, ranking) in CELLS.items():
        task_id = f"cell-{condition}"
        identity = ClaimIdentity(task_id=task_id, model_id="m", model_revision="a" * 40, protocol="p",
                                 request_sha256=hashlib.sha256(task_id.encode()).hexdigest())
        writer.append("task_definitions", {"task_id": task_id, "prompt_id": "q1", "model": "qwen38",
                                           "stage": "generation", "method": "Parallel-Expansion-v1",
                                           "engine": "ddg", "condition": condition,
                                           "claim_identity": asdict(identity)}, transaction_id="d" + task_id)
        claim = ledger.claim(identity, owner_id="fixture").claim
        t = writer.append("traces", trace(incoming, selected), transaction_id=task_id, record_id="trace-" + task_id)
        audit_row = {"target_url": A["url"], "target_observed": True,
                     "target_removed_count": int(condition == "ablated")}
        g = writer.append("generations", {
            "cell_id": task_id, "prompt_id": "q1", "method": "Parallel-Expansion-v1", "engine": "ddg",
            "condition": condition, "answer": ANSWER, "ranking": ranking, "final_snippet_count": len(selected),
            "condition_audit": audit_row}, transaction_id=task_id, record_id="generation-" + task_id)
        pending.append((claim, [t, g]))
    writer.seal()
    for claim, refs in pending:
        ledger.transition(claim, state="completed", record_references=refs)
    return root


def read_gz(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def test_freezing_judges_every_observed_source_and_shares_identical_tasks(tmp_path):
    root = dataset(tmp_path)
    out = tmp_path / "si"
    assert prepare.main(["--source", f"{root}:qwen38", "--output", str(out)]) == 0
    cells = {c["condition"]: c for c in read_gz(out / "cells.jsonl.gz")}
    tasks = read_gz(out / "tasks.jsonl.gz")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["cells_ok"] == 3 and manifest["protocol"] == si.PROTOCOL
    natural = cells["natural"]
    assert natural["presented"] == [A["url"], B["url"]] and natural["generator_ranking"] == [B["url"], A["url"]]
    assert natural["generator_output"]["raw_ranking"] == ["S2", "S1"]
    assert [s["presented_position"] for s in natural["sources"]] == [0, 1]
    # Natural and Shuffled show the same answer and sources: their source tasks are shared.
    assert [s["judge_task_id"] for s in natural["sources"]] == [s["judge_task_id"] for s in cells["shuffled"]["sources"]]
    ids = [t["judge_task_id"] for t in tasks]
    assert len(ids) == len(set(ids)) == manifest["counts"]["unique_source_importance_tasks"] + 1  # + one J1
    for record in tasks:
        item = si.item_from_record(record)
        assert "a.example" not in item["prompt"].split("ANSWER UNITS:")[-1] or record["task"] == "fulfilment"
    assert not (tmp_path / "si.partial").exists()
    with pytest.raises(SystemExit):
        prepare.main(["--source", f"{root}:qwen38", "--output", str(out)])


def test_stored_task_records_refuse_to_drift(tmp_path):
    item = si.prepare_source_task(request="Q?", answer=ANSWER, source=A, observed_urls=[A["url"]])
    record = si.task_record(item)
    assert si.item_from_record(record)["prompt"] == item["prompt"]
    with pytest.raises(ValueError, match="reproduce"):
        si.item_from_record({**record, "source_text": "Asana is paid only."})


def test_audit_classifies_erased_shuffle_and_ablation_exposure(tmp_path):
    root = dataset(tmp_path)
    out = tmp_path / "audit"
    assert audit.main(["--dataset-root", str(root), "--model", "qwen38", "--output", str(out)]) == 0
    (row,) = read_gz(out / "groups.jsonl.gz")
    assert row["shuffled"] == {"change": "none_at_generator_input", "incoming_order_changed": True,
                               "erased_by_reranking": True}
    ablated = row["ablated"]
    assert ablated["change"] == "membership_or_content" and ablated["exposure"] == "shown"
    assert ablated["replacement_urls"] == [C["url"]] and ablated["removed_urls"] == [A["url"]]
    assert ablated["target_in_ablated_evidence"] is False


def view(queries=("q",), evidence=(("u1", "t", "x"), ("u2", "t", "y")), prompt="P", conditioned=(("u1", "u2"),)):
    return {"queries": list(queries), "evidence": list(evidence), "final_prompt": prompt,
            "conditioned_urls": [list(c) for c in conditioned]}


@pytest.mark.parametrize("other, change", [
    (view(queries=("q", "q2")), "trajectory_divergence"),
    (view(evidence=(("u1", "t", "x"), ("u3", "t", "z"))), "membership_or_content"),
    (view(evidence=(("u1", "t", "CHANGED"), ("u2", "t", "y"))), "membership_or_content"),
    (view(evidence=(("u2", "t", "y"), ("u1", "t", "x"))), "order"),
    (view(prompt="P2"), "prompt_text_only"),
    (view(), "none_at_generator_input"),
    (view(prompt=None), "not_assessable"),
])
def test_change_classes_follow_the_largest_observed_change(other, change):
    assert classify_pair(view(), other)["change"] == change


def test_keyword_cluster_bootstrap_resamples_keywords_not_rows():
    rows = [{"kw": "k1", "a": True, "f": False}, {"kw": "k1", "a": True, "f": True},
            {"kw": "k2", "a": False, "f": False}, {"kw": "k3", "a": None, "f": True}]
    result = cluster_bootstrap(rows, cluster="kw", statistic=mean_of("a"), replicates=200, seed=1)
    assert result["estimate"] == pytest.approx(2 / 3) and result["clusters"] == 3
    assert result == cluster_bootstrap(rows, cluster="kw", statistic=mean_of("a"), replicates=200, seed=1)
    assert result["replicates"] + result["undefined_replicates"] == 200
    assert 0.0 <= result["low"] <= result["estimate"] <= result["high"] <= 1.0
    assert paired_difference("a", "f")(rows) == pytest.approx((0 + 1 + 0) / 3)
    flat = cluster_bootstrap([{"kw": k, "v": 1} for k in "abc"], cluster="kw", statistic=mean_of("v"))
    assert flat["low"] == flat["high"] == 1.0
