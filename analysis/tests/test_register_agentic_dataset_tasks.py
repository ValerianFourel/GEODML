"""Frozen task registration feeds incremental audits and segment queues."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
    iter_sealed_rows,
)
from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeEvidence,
    AgenticJudgeTask,
)
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.audit_agentic_dataset import build_audit
from analysis.scripts.register_agentic_dataset_tasks import (
    register_generator_tasks,
    register_nemotron_tasks,
)
from analysis.scripts.run_agentic_search_integration_smoke import SmokeInputs


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _generator_inputs(tmp_path: Path, *, model_id: str, revision: str) -> SmokeInputs:
    prompts = []
    selections = []
    for number, keyword in enumerate(("alpha", "beta")):
        question = f"Question for {keyword}?"
        prompts.append({
            "candidate_id": f"prompt-{keyword}",
            "question": question,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "keyword": keyword,
        })
        selections.append({"candidate_id": f"prompt-{keyword}", "axis_bin": number})
    snapshots = {}
    for engine in ("duckduckgo", "searxng"):
        rows = []
        for keyword in ("alpha", "beta"):
            rows.extend({
                "keyword": keyword,
                "position": position,
                "title": f"{keyword} source {position}",
                "url": f"https://{engine}.test/{keyword}/{position}",
                "snippet": f"Evidence for {keyword} {position}",
            } for position in range(1, 21))
        snapshots[engine] = _jsonl(tmp_path / f"{engine}.jsonl", rows)
    cross_encoder = tmp_path / ("e" * 40)
    cross_encoder.mkdir(exist_ok=True)
    return SmokeInputs(
        output=tmp_path / f"registration-{model_id.split('/')[-1]}",
        base_url="http://127.0.0.1:1/v1",
        model_id=model_id,
        model_revision=revision,
        cross_encoder_snapshot=cross_encoder,
        cross_encoder_revision="e" * 40,
        search_snapshots=snapshots,
        seed=20260911,
        max_tokens=1024,
        query_max_tokens=256,
        final_max_tokens=2048,
        request_concurrency=1,
        prompts_jsonl=_jsonl(tmp_path / "prompts.jsonl", prompts),
        selection_records_jsonl=_jsonl(tmp_path / "selections.jsonl", selections),
        prompt_count=2,
        prompt_selection_seed=20260912,
        production_conditions=True,
    )


def _priority(tmp_path: Path) -> Path:
    path = tmp_path / "keyword-priority.json"
    path.write_text(json.dumps({
        "format_version": "geodml-keyword-priority-v1",
        "keywords": [
            {"source_keyword": "alpha", "keyword_id": "alpha", "priority_rank": 0},
            {"source_keyword": "beta", "keyword_id": "beta", "priority_rank": 1},
        ],
    }), encoding="utf-8")
    return path


def test_generator_registration_is_idempotent_and_keeps_models_separate(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    qwen = _generator_inputs(
        tmp_path, model_id="Qwen/Qwen3.8-27B", revision="a" * 40
    )
    llama = _generator_inputs(
        tmp_path, model_id="meta-llama/Llama-4", revision="b" * 40
    )
    qwen_result = register_generator_tasks(
        dataset_root=root,
        model_slug="qwen38",
        inputs=qwen,
        keyword_priority_path=_priority(tmp_path),
        writer_id="register-qwen",
    )
    llama_result = register_generator_tasks(
        dataset_root=root,
        model_slug="llama4",
        inputs=llama,
        keyword_priority_path=_priority(tmp_path),
        writer_id="register-llama",
    )
    repeated = register_generator_tasks(
        dataset_root=root,
        model_slug="qwen38",
        inputs=qwen,
        keyword_priority_path=_priority(tmp_path),
        writer_id="register-qwen-again",
    )
    tasks = list(iter_sealed_rows(root, "task_definitions", required=True))
    assert qwen_result["appended"] == {
        "prompts": 2, "keyword_memberships": 2, "task_definitions": 24,
    }
    assert llama_result["appended"] == {
        "prompts": 0, "keyword_memberships": 0, "task_definitions": 24,
    }
    assert repeated["appended"] == {
        "prompts": 0, "keyword_memberships": 0, "task_definitions": 0,
    }
    assert len(tasks) == 48
    qwen_ids = {row["task_id"] for row in tasks if row["model"] == "qwen38"}
    llama_ids = {row["task_id"] for row in tasks if row["model"] == "llama4"}
    assert qwen_ids == llama_ids
    assert build_audit(root, model="qwen38", stripe_count=8)[
        "eligible_remaining"
    ] == 24
    assert build_audit(root, model="llama4", stripe_count=8)[
        "eligible_remaining"
    ] == 24


def test_nemotron_registration_links_exact_generator_fingerprint(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    qwen = _generator_inputs(
        tmp_path, model_id="Qwen/Qwen3.8-27B", revision="a" * 40
    )
    register_generator_tasks(
        dataset_root=root,
        model_slug="qwen38",
        inputs=qwen,
        keyword_priority_path=_priority(tmp_path),
        writer_id="register-qwen",
    )
    generator = next(iter_sealed_rows(root, "task_definitions", required=True))
    task = AgenticJudgeTask(
        judge_task_id="judge-1",
        format_version="agentic-search-judge-v1",
        blind_case_id="blind-1",
        prompt_text="Question for alpha?",
        evidence=(AgenticJudgeEvidence(
            evidence_id="evidence-1",
            url="https://example.test/1",
            title="Title",
            text="Evidence",
        ),),
        answer="Answer",
    )
    tasks_path = _jsonl(tmp_path / "bulk_tasks.jsonl", [task.to_dict()])
    mapping_path = _jsonl(tmp_path / "private_mapping.jsonl", [{
        "judge_task_id": task.judge_task_id,
        "source_cell_id": generator["task_id"],
        "prompt_id": generator["prompt_id"],
        "generator_model_id": generator["claim_identity"]["model_id"],
        "method": generator["method"],
        "engine": generator["engine"],
        "condition": generator["condition"],
    }])
    manifest_path = tmp_path / "judge-manifest.json"
    manifest_path.write_text(json.dumps({
        "judge_plan_id": "judge-plan-1",
        "format_version": "agentic-search-judge-v1",
        "bulk_model": {
            "role": "bulk",
            "model_id": "nvidia/Nemotron",
            "model_revision": "c" * 40,
        },
        "artifacts": {
            "bulk_tasks": {"sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest()},
            "private_mapping": {
                "sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest()
            },
        },
    }), encoding="utf-8")
    result = register_nemotron_tasks(
        dataset_root=root,
        manifest_path=manifest_path,
        tasks_path=tasks_path,
        mapping_path=mapping_path,
        writer_id="register-nemotron",
    )
    assert result["appended"] == {"task_definitions": 1}
    judge = next(
        row for row in iter_sealed_rows(root, "task_definitions")
        if row["model"] == "nemotron"
    )
    source_identity = ClaimIdentity(**generator["claim_identity"])
    assert judge["dependency_fingerprints"] == [
        identity_fingerprint(source_identity)
    ]
    blocked = build_audit(root, model="nemotron", stripe_count=8)
    assert blocked["blocked"] == 1
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    results = FinalDatasetWriter(root, writer_id="generator-results")
    generation_reference = results.append(
        "generations", {"answer": "saved"}, transaction_id="generator"
    )
    results.seal()
    claim = ledger.claim(source_identity, owner_id="generator").claim
    ledger.transition(
        claim,
        state="completed",
        record_references=[generation_reference],
    )
    eligible = build_audit(root, model="nemotron", stripe_count=8)
    assert eligible["eligible_remaining"] == 1
