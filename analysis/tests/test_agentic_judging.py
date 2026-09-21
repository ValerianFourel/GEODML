"""Contracts for blinded bulk, validation, and adjudication judging."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeModel,
    build_adjudication_plan,
    build_agentic_judge_plan,
    validate_agentic_judgment,
    write_agentic_judge_plan,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_result(
    root: Path,
    *,
    cell_id: str,
    prompt_id: str,
    condition: str,
    answer: str,
) -> Path:
    evidence = [
        {
            "url": "https://first.example/item",
            "title": "First",
            "text": "First frozen snippet.",
        },
        {
            "url": "https://second.example/item",
            "title": "Second",
            "text": "Second frozen snippet.",
        },
    ]
    trace = {
        "format_version": "agentic-search-trace-v1",
        "method_id": "Parallel-Expansion-v1",
        "condition": condition,
        "user_prompt_sha256": hashlib.sha256(
            f"Question for {prompt_id}".encode()
        ).hexdigest(),
        "search_engine": "duckduckgo",
        "compactor_model_id": "cross-encoder",
        "compactor_model_revision": "revision",
        "bounds": {},
        "events": [
            {
                "event_index": 0,
                "event_type": "compaction",
                "payload": {
                    "selected_snippets": [
                        {**row, "score": 1.0, "source_index": index}
                        for index, row in enumerate(evidence)
                    ]
                },
            }
        ],
    }
    trace_hash = hashlib.sha256(_canonical(trace)).hexdigest()
    trace_path = root / "traces" / f"{cell_id}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps({**trace, "trace_sha256": trace_hash}), encoding="utf-8"
    )
    result = {
        "cell_id": cell_id,
        "prompt_id": prompt_id,
        "prompt_sha256": trace["user_prompt_sha256"],
        "method": "Parallel-Expansion-v1",
        "engine": "duckduckgo",
        "condition": condition,
        "ranking": [row["url"] for row in evidence],
        "answer": answer,
        "final_snippet_count": 2,
        "search_count": 3,
        "trace": str(trace_path.resolve()),
        "trace_sha256": trace_hash,
    }
    result_path = root / "results" / f"{cell_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def _models() -> tuple[AgenticJudgeModel, AgenticJudgeModel]:
    return (
        AgenticJudgeModel(
            role="bulk",
            model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            model_revision="a" * 40,
        ),
        AgenticJudgeModel(
            role="validation",
            model_id="zai-org/GLM-5.3-Flash",
            model_revision="b" * 40,
        ),
    )


def test_builds_blinded_bulk_and_stratified_validation_tasks(tmp_path: Path) -> None:
    result_paths = []
    prompts = []
    for index in range(4):
        prompt_id = f"prompt-{index}"
        question = f"Question for {prompt_id}"
        prompts.append(
            {
                "candidate_id": prompt_id,
                "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "axis_bin": "middle",
            }
        )
        result_paths.append(
            _write_result(
                tmp_path / "qwen38" / "shard-0",
                cell_id=f"cell-{index}",
                prompt_id=prompt_id,
                condition="natural",
                answer=f"Answer {index}.",
            )
        )

    bulk, validation = _models()
    plan = build_agentic_judge_plan(
        result_paths,
        prompt_rows=prompts,
        generator_model_by_root={str(tmp_path / "qwen38"): "Qwen/Qwen3.8-27B"},
        bulk_model=bulk,
        validation_model=validation,
        validation_fraction=0.5,
        master_seed=17,
    )

    assert len(plan.bulk_tasks) == 4
    assert len(plan.validation_tasks) == 2
    assert {task.blind_case_id for task in plan.validation_tasks}.issubset(
        {task.blind_case_id for task in plan.bulk_tasks}
    )
    assert len(plan.mappings) == 4
    for task in plan.bulk_tasks:
        public = task.to_dict()
        serialized = json.dumps(public)
        assert "Qwen" not in serialized
        assert "natural" not in serialized
        assert "duckduckgo" not in serialized
        assert "Parallel-Expansion" not in serialized
        assert "generated_ranking" not in public
        assert [row["url"] for row in public["evidence"]] == [
            "https://second.example/item",
            "https://first.example/item",
        ]

    artifacts = write_agentic_judge_plan(tmp_path / "judge-plan", plan=plan)
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["scientific_result"] is False
    assert manifest["blinding"] == "generator-treatment-and-ranking-hidden-v1"
    assert manifest["summary"]["bulk_task_count"] == 4
    assert manifest["summary"]["validation_task_count"] == 2
    assert len(artifacts.bulk_tasks_path.read_text().splitlines()) == 4
    assert len(artifacts.validation_tasks_path.read_text().splitlines()) == 2

    bulk_only = build_agentic_judge_plan(
        result_paths, prompt_rows=prompts,
        generator_model_by_root={str(tmp_path / "qwen38"): "Qwen/Qwen3.8-27B"},
        bulk_model=bulk, validation_model=None, validation_fraction=0, master_seed=17,
    )
    assert bulk_only.bulk_tasks == plan.bulk_tasks
    assert bulk_only.validation_tasks == ()
    bulk_artifacts = write_agentic_judge_plan(tmp_path / "bulk-only-plan", plan=bulk_only)
    bulk_manifest = json.loads(bulk_artifacts.manifest_path.read_text())
    assert bulk_manifest["validation_model"] is None
    assert bulk_manifest["validation_sampling"] == "not_configured"
    assert bulk_manifest["summary"]["validation_task_count"] == 0
    from analysis.scripts.run_acl_arr_vllm import _agentic_judge_context
    tasks, _, model, revision = _agentic_judge_context(
        bulk_artifacts.manifest_path, bulk_artifacts.bulk_tasks_path, judge_role="bulk",
    )
    assert len(tasks) == 4
    assert (model, revision) == (bulk.model_id, bulk.model_revision)
    with pytest.raises(ValueError, match="validation_fraction=0"):
        build_agentic_judge_plan(
            result_paths, prompt_rows=prompts,
            generator_model_by_root={str(tmp_path / "qwen38"): "Qwen/Qwen3.8-27B"},
            bulk_model=bulk, validation_model=None,
        )


def test_rejects_tampered_trace(tmp_path: Path) -> None:
    prompt_id = "prompt-0"
    question = f"Question for {prompt_id}"
    result_path = _write_result(
        tmp_path / "qwen38" / "shard-0",
        cell_id="cell-0",
        prompt_id=prompt_id,
        condition="natural",
        answer="Answer.",
    )
    result = json.loads(result_path.read_text())
    trace_path = Path(result["trace"])
    trace = json.loads(trace_path.read_text())
    trace["events"][0]["payload"]["selected_snippets"][0]["text"] = "tampered"
    trace_path.write_text(json.dumps(trace))

    bulk, validation = _models()
    with pytest.raises(ValueError, match="trace hash mismatch"):
        build_agentic_judge_plan(
            [result_path],
            prompt_rows=[
                {
                    "candidate_id": prompt_id,
                    "question": question,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "axis_bin": "middle",
                }
            ],
            generator_model_by_root={str(tmp_path / "qwen38"): "generator"},
            bulk_model=bulk,
            validation_model=validation,
            validation_fraction=1.0,
        )


def test_same_cell_id_from_two_generators_produces_two_blind_cases(
    tmp_path: Path,
) -> None:
    prompt_id = "prompt-0"
    question = f"Question for {prompt_id}"
    qwen = tmp_path / "qwen38"
    llama = tmp_path / "llama4"
    paths = [
        _write_result(
            root / "shard-0",
            cell_id="shared-cell",
            prompt_id=prompt_id,
            condition="natural",
            answer=f"Answer from {name}.",
        )
        for root, name in ((qwen, "Qwen"), (llama, "Llama"))
    ]
    bulk, validation = _models()
    plan = build_agentic_judge_plan(
        paths,
        prompt_rows=[
            {
                "candidate_id": prompt_id,
                "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "axis_bin": "middle",
            }
        ],
        generator_model_by_root={str(qwen): "qwen", str(llama): "llama"},
        bulk_model=bulk,
        validation_model=validation,
        validation_fraction=1.0,
    )
    assert len(plan.bulk_tasks) == 2
    assert {mapping.generator_model_id for mapping in plan.mappings} == {
        "qwen",
        "llama",
    }


def test_validates_judgment_and_routes_predefined_adjudication() -> None:
    allowed = ("E1", "E2")
    nano = validate_agentic_judgment(
        {
            "request_fulfillment": 4,
            "evidence_grounding": 4,
            "ideal_relevance_ranking": ["E1", "E2"],
            "realized_support_ranking": [{"evidence_id": "E1", "use_score": 4}],
            "unsupported_claim_count": 0,
            "judge_confidence": 2,
        },
        allowed_evidence_ids=allowed,
    )
    glm = validate_agentic_judgment(
        {
            "request_fulfillment": 2,
            "evidence_grounding": 4,
            "ideal_relevance_ranking": ["E2", "E1"],
            "realized_support_ranking": [{"evidence_id": "E2", "use_score": 3}],
            "unsupported_claim_count": 2,
            "judge_confidence": 5,
        },
        allowed_evidence_ids=allowed,
    )

    routed = build_adjudication_plan(
        bulk_outcomes={"case-a": nano},
        validation_outcomes={"case-a": glm},
        all_case_ids=("case-a", "case-b"),
        low_confidence_threshold=2,
        score_disagreement_threshold=2,
        unsupported_claim_disagreement_threshold=2,
    )

    assert routed["case-a"]["requires_adjudication"] is True
    assert set(routed["case-a"]["reasons"]) == {
        "bulk_low_confidence",
        "ideal_top_choice_disagreement",
        "request_fulfillment_disagreement",
        "unsupported_claim_disagreement",
    }
    assert routed["case-a"]["resolution"] == "use_existing_validation_judgment"
    assert routed["case-b"] == {
        "requires_adjudication": True,
        "reasons": ["bulk_judgment_missing"],
        "resolution": "queue_validation_judgment",
    }


def test_judgment_rejects_unknown_or_duplicate_evidence() -> None:
    base = {
        "request_fulfillment": 3,
        "evidence_grounding": 3,
        "ideal_relevance_ranking": ["E1", "E1"],
        "realized_support_ranking": [],
        "unsupported_claim_count": 0,
        "judge_confidence": 3,
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_agentic_judgment(base, allowed_evidence_ids=("E1", "E2"))
    with pytest.raises(ValueError, match="unknown"):
        validate_agentic_judgment(
            {**base, "ideal_relevance_ranking": ["E1", "E3"]},
            allowed_evidence_ids=("E1", "E2"),
        )


def test_prepare_cli_requires_complete_shards_and_writes_plan(tmp_path: Path) -> None:
    model_root = tmp_path / "models" / "qwen38"
    shard_root = model_root / "shard-0"
    result_path = _write_result(
        shard_root,
        cell_id="cell-0",
        prompt_id="prompt-0",
        condition="natural",
        answer="Answer.",
    )
    (shard_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "starting",
                "model_id": "Qwen/Qwen3.8-27B",
                "cell_count": 1,
                "completed_count": 1,
                "remaining_count": 0,
            }
        ),
        encoding="utf-8",
    )
    prompts_path = tmp_path / "prompts.jsonl"
    question = "Question for prompt-0"
    prompts_path.write_text(
        json.dumps(
            {
                "candidate_id": "prompt-0",
                "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selections_path = tmp_path / "selections.jsonl"
    selections_path.write_text(
        json.dumps({"candidate_id": "prompt-0", "axis_bin": "middle"}) + "\n",
        encoding="utf-8",
    )
    generation_tasks = tmp_path / "generation-tasks.jsonl"
    generation_tasks.write_text(json.dumps({"cell_id": "cell-0"}) + "\n")
    output = tmp_path / "plan"
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "analysis/scripts/prepare_agentic_judge_tasks.py",
        "--generator-root",
        f"Qwen/Qwen3.8-27B={model_root}",
        "--prompts-jsonl",
        str(prompts_path),
        "--selection-records-jsonl",
        str(selections_path),
        "--generation-tasks",
        str(generation_tasks),
        "--bulk-model-id",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "--bulk-model-revision",
        "a" * 40,
        "--validation-model-id",
        "zai-org/GLM-5.3-Flash",
        "--validation-model-revision",
        "b" * 40,
        "--validation-fraction",
        "1",
        "--output-dir",
        str(output),
    ]

    incomplete = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    assert incomplete.returncode != 0
    assert "status is invalid" in incomplete.stderr

    manifest_path = shard_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    complete = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    assert complete.returncode == 0, complete.stderr
    assert "BULK_TASKS=1" in complete.stdout
    assert "VALIDATION_TASKS=1" in complete.stdout
    assert result_path.is_file()
    assert (output / "run_manifest.json").is_file()

    bulk_command = list(command)
    for flag in ("--validation-model-id", "--validation-model-revision", "--validation-fraction"):
        index = bulk_command.index(flag)
        del bulk_command[index:index + 2]
    bulk_output = tmp_path / "bulk-only-cli"
    bulk_command[bulk_command.index("--output-dir") + 1] = str(bulk_output)
    missing_model = subprocess.run(bulk_command, cwd=repository, text=True, capture_output=True, check=False)
    assert missing_model.returncode != 0
    assert "explicitly use --bulk-only" in missing_model.stderr
    bulk_command.append("--bulk-only")
    bulk_run = subprocess.run(bulk_command, cwd=repository, text=True, capture_output=True, check=False)
    assert bulk_run.returncode == 0, bulk_run.stderr
    bulk_manifest = json.loads((bulk_output / "run_manifest.json").read_text())
    assert bulk_manifest["validation_model"] is None
    assert bulk_manifest["summary"] == {**json.loads((output / "run_manifest.json").read_text())["summary"], "validation_task_count": 0}
    fake_output = tmp_path / "bulk-only-fake"
    fake_bulk = subprocess.run(
        [sys.executable, "analysis/scripts/run_acl_arr_vllm.py", "agentic-judge",
         "--tasks", str(bulk_output / "bulk_tasks.jsonl"),
         "--judge-manifest", str(bulk_output / "run_manifest.json"),
         "--judge-role", "bulk", "--output-dir", str(fake_output), "--fake"],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    assert fake_bulk.returncode == 0, fake_bulk.stderr
    fake_manifest = json.loads((fake_output / "run_manifest.json").read_text())
    assert fake_manifest["completed_count"] == 1
    assert fake_manifest["scientific_result"] is False

    for role in ("bulk", "validation"):
        run_output = tmp_path / f"{role}-run"
        run = subprocess.run(
            [
                sys.executable,
                "analysis/scripts/run_acl_arr_vllm.py",
                "agentic-judge",
                "--tasks",
                str(output / f"{role}_tasks.jsonl"),
                "--judge-manifest",
                str(output / "run_manifest.json"),
                "--judge-role",
                role,
                "--output-dir",
                str(run_output),
                "--fake",
            ],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        run_manifest = json.loads((run_output / "run_manifest.json").read_text())
        assert run_manifest["status"] == "complete"
        assert run_manifest["judge_role"] == role
        assert run_manifest["completed_count"] == 1

    empty_bulk = tmp_path / "empty-bulk"
    empty_bulk.mkdir()
    adjudication_output = tmp_path / "adjudication"
    adjudication = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/prepare_agentic_adjudication.py",
            "--judge-manifest",
            str(output / "run_manifest.json"),
            "--bulk-results-root",
            str(empty_bulk),
            "--output-dir",
            str(adjudication_output),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert adjudication.returncode == 0, adjudication.stderr
    assert "ADJUDICATION_TASKS=1" in adjudication.stdout
    assert (
        len(
            (adjudication_output / "validation_adjudication_tasks.jsonl")
            .read_text()
            .splitlines()
        )
        == 1
    )

    resolved_output = tmp_path / "resolved"
    resolved = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/prepare_agentic_adjudication.py",
            "--judge-manifest",
            str(output / "run_manifest.json"),
            "--bulk-results-root",
            str(tmp_path / "bulk-run"),
            "--validation-results-root",
            str(tmp_path / "validation-run"),
            "--output-dir",
            str(resolved_output),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert "ADJUDICATION_TASKS=0" in resolved.stdout
    assert "RESOLVED_JUDGMENTS=1" in resolved.stdout
