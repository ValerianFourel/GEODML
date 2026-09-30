"""Fresh diagnostic samples preserve sealed evidence, task identity and prior runs."""
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime

import pytest

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import fresh_gemma_si_review as fresh
from analysis.scripts import verify_inference_allocation
from analysis.tests.test_source_importance_pipeline import A, B, ANSWER, cell_for, prepare, trace


def sealed_sample(root, model, *, maximum_shard_bytes=128 * 1024 * 1024):
    initialize_dataset(root, population_id="p", acceptance_policy_id="a")
    writer = FinalDatasetWriter(root, writer_id="fixture", maximum_shard_bytes=maximum_shard_bytes)
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=256)
    pending = []
    # Includes the excluded prompt and two conditions per fresh prompt, so
    # choosing twenty cells without enforcing distinct prompts would be wrong.
    for i in range(13):
        prompt = "q1" if i == 0 else f"{model}-{i}"
        writer.append("prompts", {"prompt_id": prompt, "prompt_text": f"Free project tools for {prompt}?"},
                      transaction_id=prompt)
        for condition in ("natural", "shuffled"):
            tid = f"{prompt}-{condition}"
            identity = ClaimIdentity(task_id=tid, model_id=model, model_revision="a" * 40, protocol="p",
                                     request_sha256=hashlib.sha256(tid.encode()).hexdigest())
            cell = cell_for(ANSWER, trace([A, B], [A, B]), cell_id=tid)
            generation = {**cell["generation"], "prompt_id": prompt, "condition": condition}
            writer.append("task_definitions", {"task_id": tid, "prompt_id": prompt, "model": model,
                "stage": "generation", "method": "Parallel-Expansion-v1", "engine": "ddg",
                "condition": condition, "claim_identity": asdict(identity)}, transaction_id="d" + tid)
            claim = ledger.claim(identity, owner_id="fixture").claim
            refs = [writer.append("generations", generation, transaction_id=tid, record_id="g" + tid),
                    writer.append("traces", cell["trace"], transaction_id=tid, record_id="t" + tid)]
            pending.append((claim, refs))
    writer.seal()
    for claim, refs in pending:
        ledger.transition(claim, state="completed", record_references=refs)
    return root


def test_freeze_excludes_prior_prompts_is_repeatable_and_rejects_corrupt_evidence(tmp_path, monkeypatch):
    qwen = sealed_sample(tmp_path / "qwen", "qwen38")
    llama = sealed_sample(tmp_path / "llama", "llama4")
    record, tasks = prepare.build_cell(cell_for(ANSWER, trace([A, B], [A, B])), max_tokens=640, j1_max_tokens=64)
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"cells": [record], "tasks": tasks}))
    before = previous.read_bytes()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(verify_inference_allocation, "verify", lambda cluster: {"verified": True})
    end = datetime.fromtimestamp(time.time() + 2400).isoformat()
    def scheduler(command, **kwargs):
        assert command == ["scontrol", "show", "job", "123", "-o"]
        return f"JobState=RUNNING JobName=geodml-gemma-si-replay TimeLimit=01:00:00 EndTime={end}"
    monkeypatch.setattr(fresh.subprocess, "check_output", scheduler)
    output = tmp_path / "fresh.json"
    args = ["freeze", "--qwen", str(qwen), "--llama", str(llama), "--exclude-inputs", str(previous),
            "--existing-job-id", "123", "--output", str(output)]
    assert fresh.main(args) == 0
    bundle = json.loads(output.read_text())
    assert len(bundle["cells"]) == len({c["prompt_id"] for c in bundle["cells"]}) == 20
    assert all(c["prompt_id"] != "q1" for c in bundle["cells"])
    assert sorted(c["model"] for c in bundle["cells"]) == ["llama4"] * 10 + ["qwen38"] * 10
    assert len(fresh.frozen_cells(bundle)) == 20
    saved = output.read_bytes()
    assert fresh.main(args) == 0 and output.read_bytes() == saved
    second = tmp_path / "second.json"
    assert fresh.main(args[:-1] + [str(second)]) == 0
    assert second.read_bytes() == saved and previous.read_bytes() == before
    bundle["cells"][0]["prompt_id"] = "q1"
    output.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="distinct fresh prompts"):
        fresh.main(args)
    # Changing a selected trace's sealed shard must fail rather than replace
    # its cell with a different candidate or freeze unverified content.
    shard = next((qwen / "data/traces").glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes().replace(b"Asana", b"Other"))
    third = tmp_path / "third.json"
    with pytest.raises(ValueError, match="verification|verified fresh prompts"):
        fresh.main(args[:-1] + [str(third)])
    assert not third.exists()


def test_selection_records_unavailable_candidate_and_uses_verified_fresh_prompts(tmp_path):
    root = sealed_sample(tmp_path / "qwen", "qwen38", maximum_shard_bytes=1)
    initial = fresh.sample_refs(root, "qwen38", {"q1"}, deadline=time.time() + 120)
    # Support the pre-fix list return to reproduce the original bad selection.
    chosen = initial[0] if isinstance(initial, tuple) else initial
    bad = chosen[0]
    ref = bad["generation_ref"]
    shard = root / "data/generations" / f"part-{ref['writer_id']}-{ref['shard_sequence']:06d}.jsonl"
    shard.unlink()  # ledger completion survives a missing local shard
    result = fresh.sample_refs(root, "qwen38", {"q1"}, deadline=time.time() + 120)
    selected, rejected = result if isinstance(result, tuple) else (result, [])
    assert len(selected) == len({r["prompt_id"] for r in selected}) == 10
    assert bad["prompt_id"] not in {r["prompt_id"] for r in selected}
    assert rejected[0]["fingerprint"] == bad["fingerprint"]
    assert rejected[0]["failed_references"] == [ref]
    assert len(list(fresh.iter_cells(root, selected))) == 10
