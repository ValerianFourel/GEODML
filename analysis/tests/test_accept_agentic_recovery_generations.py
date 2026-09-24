from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.accept_agentic_recovery_generations import accept


def _gzip(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_accepts_only_exact_registered_native_identity(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    prompt_text = "Question"
    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    identity = ClaimIdentity("cell", "Qwen/Qwen3.8-27B", "a" * 40,
                             "agentic-generator-shared-v1", "b" * 64)
    writer = FinalDatasetWriter(root, writer_id="register")
    writer.append("prompts", {"prompt_id": "p", "prompt_text": prompt_text,
                               "prompt_sha256": prompt_sha}, transaction_id="p")
    writer.append("keyword_memberships", {
        "prompt_id": "p", "keyword_ids": ["k"], "primary_keyword_id": "k",
        "primary_priority_rank": 0,
    }, transaction_id="p")
    writer.append("task_definitions", {
        "task_id": "cell", "prompt_id": "p", "model": "qwen38",
        "stage": "generation", "method": "method", "engine": "duckduckgo",
        "condition": "natural", "claim_identity": asdict(identity),
        "dependency_fingerprints": [], "runnable_task": {"cell_id": "cell"},
    }, transaction_id="cell")
    writer.seal()
    hub = tmp_path / "hub"
    (hub / "data").mkdir(parents=True)
    trace = {"trace_sha256": "c" * 64, "user_prompt_sha256": prompt_sha, "events": []}
    result = {"cell_id": "cell", "prompt_id": "p", "method": "method",
              "engine": "duckduckgo", "condition": "natural",
              "trace_sha256": "c" * 64, "answer": "answer", "ranking": []}
    _gzip(hub / "data/generations.jsonl.gz", [{
        "generation_id": "g", "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "validation": "payload_valid_protocol_acceptance_pending",
        "result": result, "trace": trace,
    }])
    _gzip(hub / "data/generation_aliases.jsonl.gz", [{
        "generation_id": "g", "native_identity": asdict(identity),
    }])
    files = {
        str(path.relative_to(hub)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in hub.rglob("*") if path.is_file()
    }
    (hub / "publication-manifest.json").write_text(json.dumps({
        "format_version": "geodml-recovery-publication-v1", "files": files,
    }))
    receipt = accept(root, hub, model="qwen38", stripe_count=8)
    assert receipt["ledger_completions_created"] == 1
    assert accept(root, hub, model="qwen38", stripe_count=8) == receipt
    assert StripedTaskLedger(root / "control/task-ledger", stripe_count=8).inspect(
        identity
    )["state"] == "completed"

