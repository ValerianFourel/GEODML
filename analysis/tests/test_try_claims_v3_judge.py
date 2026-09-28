"""The claims-v3 trial driver selects real cells deterministically and runs every task path."""

import asyncio
import hashlib
import json
from dataclasses import asdict

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import try_claims_v3_judge as trial
from analysis.scripts.run_acl_arr_vllm import _execute_one

ANSWER = "Asana has a free plan. Trello uses boards."


def dataset(tmp_path, cells=3):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="p", acceptance_policy_id="a")
    writer = FinalDatasetWriter(root, writer_id="fixture")
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=256)
    pending = []
    writer.append("prompts", {"prompt_id": "q1", "prompt_text": "Free project tools?"}, transaction_id="q1")
    for index in range(cells):
        task_id = f"cell-{index}"
        identity = ClaimIdentity(task_id=task_id, model_id="m", model_revision="a" * 40, protocol="p",
                                 request_sha256=hashlib.sha256(task_id.encode()).hexdigest())
        writer.append("task_definitions", {"task_id": task_id, "prompt_id": "q1", "model": "llama4",
                                           "claim_identity": asdict(identity)}, transaction_id="d" + task_id)
        claim = ledger.claim(identity, owner_id="fixture").claim
        trace = {"events": [{"event_type": "compaction", "payload": {"selected_snippets": [
            {"url": "https://a.example", "title": "A", "text": "Asana has a free plan."},
            {"url": "https://b.example", "title": "B", "text": "Trello uses boards."}]}}]}
        t = writer.append("traces", trace, transaction_id=task_id, record_id="trace-" + task_id)
        g = writer.append("generations", {"cell_id": task_id, "prompt_id": "q1", "method": "Parallel-Expansion-v1",
                                          "answer": ANSWER, "ranking": []}, transaction_id=task_id,
                          record_id="generation-" + task_id)
        pending.append((claim, [t, g]))
    writer.seal()
    for claim, refs in pending:
        ledger.transition(claim, state="completed", record_references=refs)
    return root


def test_selection_is_deterministic_and_reads_trace_and_prompt(tmp_path):
    root = dataset(tmp_path)
    first = trial.select_cells(root, count=2, seed=1, model="llama4")
    assert [c["fingerprint"] for c in first] == [c["fingerprint"] for c in trial.select_cells(root, count=2, seed=1, model="llama4")]
    assert len(first) == 2 and first[0]["prompt_text"] == "Free project tools?"
    assert first[0]["trace"]["events"][0]["event_type"] == "compaction"
    assert trial.select_cells(root, count=2, seed=1, model="qwen38") == []


class Scripted:
    """Answers each v3 task by its schema name, like a well-behaved judge."""

    async def complete(self, *, prompt, schema_name, schema, temperature, max_tokens, seed):
        if "extraction" in schema_name:
            out = {"claims": [{"sentence_id": "S001", "claim_text": "Asana has a free plan"},
                              {"sentence_id": "S002", "claim_text": "Trello uses boards"}]}
        elif "fulfilment" in schema_name:
            out = {"request_fulfillment": 4}
        elif "relevance" in schema_name:
            out = {"ideal_relevance_ranking": schema["properties"]["ideal_relevance_ranking"]["items"]["enum"]}
        else:
            ids = schema["properties"]["claims"]["items"]["properties"]["full_support_evidence_ids"]["items"]["enum"]
            out = {"claims": [{"claim_id": cid, "full_support_evidence_ids": sorted(ids)[:1] if cid == "C001" else sorted(ids)[1:],
                               "partial_support_evidence_ids": [], "contradicting_evidence_ids": []}
                              for cid in schema["properties"]["claims"]["items"]["properties"]["claim_id"]["enum"]]}
        return json.dumps(out), {"completion_tokens": 20, "finish_reason": "stop"}


def test_every_task_runs_and_both_orders_agree(tmp_path):
    root = dataset(tmp_path, cells=1)
    cell = trial.select_cells(root, count=1, seed=1, model=None)[0]
    rows = []
    summary = asyncio.run(trial.judge_cell(cell, client=Scripted(), execute=_execute_one, master_seed=7,
                                           max_tokens=512, write=rows.append))
    assert set(summary["ok"]) == {"claim_extraction", "fulfilment", "ideal_relevance",
                                  "attribution_primary", "attribution_reverse"}
    assert all(summary["ok"].values()) and summary["claim_count"] == 2
    assert summary["tiers_primary"] == summary["tiers_reverse"]
    report = trial.summarize(rows, [summary])
    assert report["scientific_result"] is False and report["order_identical_tiers"] == 1
    assert report["tasks"]["fulfilment"]["finish_reasons"] == {"stop": 1}
