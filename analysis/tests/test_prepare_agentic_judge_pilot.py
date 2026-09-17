"""Bounded judge pilot preparation from existing completed generator artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "analysis/scripts/prepare_agentic_judge_pilot.py"
FACTORS = list(
    itertools.product(
        ("Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"),
        ("duckduckgo", "searxng"),
        ("natural", "ablated", "shuffled"),
    )
)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _fixture(tmp_path: Path):
    shard = tmp_path / "qwen38/shard-0"
    (shard / "results").mkdir(parents=True)
    (shard / "traces").mkdir()
    prompts = []
    selections = []
    evidence = [
        {
            "url": "https://example.org/source",
            "title": "Source",
            "text": "The frozen evidence text.",
        }
    ]
    for prompt_index, axis_bin in enumerate((0, 4, 9)):
        prompt_id = f"prompt-{prompt_index}"
        question = f"Question {prompt_index}?"
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        prompts.append(
            {
                "candidate_id": prompt_id,
                "question": question,
                "question_sha256": question_hash,
            }
        )
        selections.append({"candidate_id": prompt_id, "axis_bin": axis_bin})
        for factor_index, (method, engine, condition) in enumerate(FACTORS):
            cell_id = f"cell-{prompt_index}-{factor_index}"
            event = {
                "event_type": "compaction",
                "payload": {"selected_snippets": evidence},
            }
            if method == "Reactive-Snippet-Loop-v1":
                event = {"event_type": "observation", "payload": {"snippets": evidence}}
            trace = {
                "method_id": method,
                "search_engine": engine,
                "condition": condition,
                "user_prompt_sha256": question_hash,
                "events": [event],
            }
            trace_hash = hashlib.sha256(_canonical(trace)).hexdigest()
            trace_path = shard / "traces" / f"{cell_id}.json"
            trace_path.write_text(json.dumps({**trace, "trace_sha256": trace_hash}))
            result = {
                "cell_id": cell_id,
                "prompt_id": prompt_id,
                "prompt_sha256": question_hash,
                "method": method,
                "engine": engine,
                "condition": condition,
                "ranking": [evidence[0]["url"]],
                "answer": "A frozen answer.",
                "final_snippet_count": 1,
                "trace": str(trace_path),
                "trace_sha256": trace_hash,
            }
            (shard / "results" / f"{cell_id}.json").write_text(json.dumps(result))
    (shard / "run_manifest.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.8-27B",
                "status": "complete",
                "completed_count": 36,
                "cell_count": 36,
                "remaining_count": 0,
                "failed_cell_ids": [],
                "git_commit": "a" * 40,
                "prompt_shard_index": 0,
            }
        )
    )
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
    selection_path = tmp_path / "selections.jsonl"
    selection_path.write_text("".join(json.dumps(row) + "\n" for row in selections))
    command = [
        sys.executable,
        str(SCRIPT),
        "--generator-root",
        str(shard),
        "--prompts-jsonl",
        str(prompt_path),
        "--selection-records-jsonl",
        str(selection_path),
        "--output-dir",
        str(tmp_path / "plan"),
    ]
    return shard, command


def _run(command):
    return subprocess.run(
        command, cwd=REPOSITORY, capture_output=True, text=True, check=False
    )


def test_pilot_blinds_two_extreme_bin_prompts_and_runs_fake_worker(tmp_path):
    shard, command = _fixture(tmp_path)
    before = {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in shard.rglob("*.json")
    }
    run = _run(command)
    assert run.returncode == 0, run.stderr
    output = tmp_path / "plan"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["summary"]["bulk_task_count"] == 24
    assert manifest["pilot"]["selected_prompt_ids"] == ["prompt-0", "prompt-2"]
    assert manifest["scientific_result"] is False
    assert "validation_model" not in manifest
    assert not (output / "validation_tasks.jsonl").exists()
    assert (
        manifest["bulk_model"]["model_revision"]
        == "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
    )
    assert manifest["sources"]["generator_manifest"]["sha256"]
    assert len(manifest["git_commit"]) == 40
    public = (output / "bulk_tasks.jsonl").read_text()
    assert len(public.splitlines()) == 24
    for private in ("Qwen", "duckduckgo", "searxng", "condition", "generated_ranking"):
        assert private not in public
    assert before == {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in shard.rglob("*.json")
    }
    worker = _run(
        [
            sys.executable,
            "analysis/scripts/run_acl_arr_vllm.py",
            "agentic-judge",
            "--tasks",
            str(output / "bulk_tasks.jsonl"),
            "--judge-manifest",
            str(output / "run_manifest.json"),
            "--judge-role",
            "bulk",
            "--output-dir",
            str(tmp_path / "judgments"),
            "--fake",
            "--pilot-only",
        ]
    )
    assert worker.returncode == 0, worker.stderr
    judged = json.loads((tmp_path / "judgments/run_manifest.json").read_text())
    assert judged["completed_count"] == 24
    assert judged["scientific_result"] is False
    again = _run(command)
    assert again.returncode != 0
    assert "overwrite" in again.stderr


@pytest.mark.parametrize(
    "mode", ["unfinished", "count", "factor", "trace", "prompt", "budget"]
)
def test_pilot_rejects_bad_sources_before_writing(tmp_path, mode):
    shard, command = _fixture(tmp_path)
    manifest_path = shard / "run_manifest.json"
    if mode in ("unfinished", "count"):
        value = json.loads(manifest_path.read_text())
        value["status" if mode == "unfinished" else "completed_count"] = (
            "checkpointed" if mode == "unfinished" else 35
        )
        manifest_path.write_text(json.dumps(value))
    elif mode in ("factor", "prompt"):
        path = shard / "results/cell-0-0.json"
        value = json.loads(path.read_text())
        value["condition" if mode == "factor" else "prompt_sha256"] = (
            "shuffled" if mode == "factor" else "wrong"
        )
        path.write_text(json.dumps(value))
    elif mode == "trace":
        path = shard / "traces/cell-0-0.json"
        value = json.loads(path.read_text())
        value["events"][0]["payload"]["selected_snippets"][0]["text"] = "tampered"
        path.write_text(json.dumps(value))
    else:
        command.extend(["--prompt-count", "3"])
    run = _run(command)
    assert run.returncode != 0
    assert not (tmp_path / "plan").exists()


def test_pilot_queue_is_deterministic(tmp_path):
    _, command = _fixture(tmp_path)
    assert _run(command).returncode == 0
    alternate = command[:-1] + [str(tmp_path / "other-plan")]
    assert _run(alternate).returncode == 0
    for name in ("bulk_tasks.jsonl", "private_mapping.jsonl"):
        assert (tmp_path / "plan" / name).read_bytes() == (
            tmp_path / "other-plan" / name
        ).read_bytes()


def _saved_judgments(root, plan):
    """Construct valid saved journal fixtures without making model requests."""
    from analysis.interpretability.pipeline.agentic_judging import AgenticJudgeTask
    from analysis.scripts.run_acl_arr_vllm import (
        _prepare_agentic_judge,
        _request_sha256,
    )

    root.mkdir()
    plan_path = plan / "run_manifest.json"
    tasks_path = plan / "bulk_tasks.jsonl"
    source = json.loads(plan_path.read_text())
    identity = {
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "pipeline": "agentic-judge-bulk", "judge_role": "bulk",
        "model_id": source["bulk_model"]["model_id"],
        "model_revision": source["bulk_model"]["model_revision"],
        "fake_backend": False, "pilot_only": True, "maximum_attempts": 3,
        "request_timeout": 120, "max_output_tokens": 2048, "disable_thinking": True,
    }
    rows = []
    for line in tasks_path.read_text().splitlines():
        task = AgenticJudgeTask.from_dict(json.loads(line))
        item = _prepare_agentic_judge(task, max_tokens=2048)
        raw = item["fake_output"]
        rows.append({
            **item["base"], "scientific_result": False, "eligible_for_analysis": False,
            "request_sha256": _request_sha256(item), "raw_output": raw,
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "parsed_output": item["validator"](raw),
        })
    (root / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for name in ("failures.jsonl", "attempts.jsonl"):
        (root / name).write_text("")
    runtime = {
        **identity, "resume_identity": identity, "source_manifest": str(plan_path),
        "tasks": {"path": str(tasks_path), "sha256": identity["tasks_sha256"]},
        "status": "complete", "completed_count": len(rows), "remaining_count": 0,
    }
    (root / "run_manifest.json").write_text(json.dumps(runtime))
    return root / "outcomes.jsonl"


def test_all_available_freezes_every_complete_prompt_group(tmp_path):
    _, command = _fixture(tmp_path)
    run = _run(command + ["--all-available"])
    assert run.returncode == 0, run.stderr
    manifest = json.loads((tmp_path / "plan/run_manifest.json").read_text())
    assert manifest["pilot"]["selected_prompt_ids"] == ["prompt-0", "prompt-1", "prompt-2"]
    assert manifest["pilot"]["execution_mode"] == "throughput"
    assert manifest["pilot"]["scope"] == "allocation-filling-throughput-not-scientific-validation"
    assert manifest["summary"] == {
        "bulk_task_count": 36, "validation_task_count": 0, "available_task_count": 36,
        "excluded_task_count": 0, "pending_task_count": 36,
    }
    assert manifest["scientific_result"] is False


def test_all_available_accepts_checkpoint_and_skips_incomplete_prompt_group(tmp_path):
    shard, command = _fixture(tmp_path)
    for path in sorted((shard / "results").glob("cell-1-*.json"))[:3]:
        path.unlink()
    manifest_path = shard / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        status="checkpointed",
        completed_count=33,
        remaining_count=3,
        cell_count=36,
    )
    manifest.pop("prompt_shard_index")
    manifest_path.write_text(json.dumps(manifest))

    run = _run(command + ["--all-available"])
    assert run.returncode == 0, run.stderr
    plan = json.loads((tmp_path / "plan/run_manifest.json").read_text())
    assert plan["pilot"]["selected_prompt_ids"] == ["prompt-0", "prompt-2"]
    assert plan["pilot"]["source_completed_count"] == 33
    assert plan["pilot"]["skipped_incomplete_prompt_count"] == 1
    assert plan["pilot"]["skipped_incomplete_cell_count"] == 9
    assert plan["summary"] == {
        "bulk_task_count": 24,
        "validation_task_count": 0,
        "available_task_count": 24,
        "excluded_task_count": 0,
        "pending_task_count": 24,
    }


def test_throughput_excludes_verified_previous_24_and_records_coverage(tmp_path):
    _, command = _fixture(tmp_path)
    assert _run(command).returncode == 0
    outcomes = _saved_judgments(tmp_path / "judged", tmp_path / "plan")
    before = outcomes.read_bytes()
    queue_command = command[:-1] + [str(tmp_path / "queue"), "--all-available", "--exclude-outcomes", str(outcomes)]
    run = _run(queue_command)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((tmp_path / "queue/run_manifest.json").read_text())
    assert manifest["summary"] == {
        "bulk_task_count": 12, "validation_task_count": 0, "available_task_count": 36,
        "excluded_task_count": 24, "pending_task_count": 12,
    }
    excluded = json.loads((tmp_path / "queue/excluded_coverage.json").read_text())
    assert len(excluded["excluded_task_ids"]) == 24
    assert len(excluded["excluded_mappings"]) == 24
    assert excluded["sources"][0]["outcomes"]["sha256"] == hashlib.sha256(before).hexdigest()
    pending = [json.loads(line) for line in (tmp_path / "queue/bulk_tasks.jsonl").read_text().splitlines()]
    assert len(pending) == 12
    assert not set(excluded["excluded_task_ids"]) & {row["judge_task_id"] for row in pending}
    assert outcomes.read_bytes() == before


@pytest.mark.parametrize("corruption", [
    "duplicate", "duplicate_source", "model", "seed", "raw_hash", "parsed", "request",
    "identity", "unknown", "fake", "invalid_raw",
])
def test_throughput_rejects_invalid_exclusions_without_output(tmp_path, corruption):
    _, command = _fixture(tmp_path)
    assert _run(command).returncode == 0
    outcomes = _saved_judgments(tmp_path / "judged", tmp_path / "plan")
    runtime_path = outcomes.parent / "run_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    rows = [json.loads(line) for line in outcomes.read_text().splitlines()]
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption in ("model", "fake"):
        key, value = ("model_revision", "b" * 40) if corruption == "model" else ("fake_backend", True)
        runtime["resume_identity"][key] = value
        runtime[key] = value
    elif corruption == "seed":
        plan_path = tmp_path / "plan/run_manifest.json"
        plan = json.loads(plan_path.read_text())
        plan["master_seed"] += 1
        plan_path.write_text(json.dumps(plan))
        runtime["resume_identity"]["source_manifest_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    elif corruption == "raw_hash":
        rows[0]["raw_output_sha256"] = "b" * 64
    elif corruption == "parsed":
        rows[0]["parsed_output"]["request_fulfillment"] = 1
    elif corruption == "request":
        rows[0]["request_sha256"] = "b" * 64
    elif corruption == "identity":
        rows[0]["blind_case_id"] = "different-case"
    elif corruption == "unknown":
        rows[0]["judge_task_id"] = "different-task"
    elif corruption == "invalid_raw":
        raw = json.loads(rows[0]["raw_output"])
        raw["ideal_relevance_ranking"] = ["unknown"]
        rows[0]["raw_output"] = json.dumps(raw)
        rows[0]["raw_output_sha256"] = hashlib.sha256(rows[0]["raw_output"].encode()).hexdigest()
        rows[0]["parsed_output"] = raw
    runtime_path.write_text(json.dumps(runtime))
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    queue_command = command[:-1] + [str(tmp_path / "queue"), "--all-available", "--exclude-outcomes", str(outcomes)]
    if corruption == "duplicate_source":
        queue_command += ["--exclude-outcomes", str(outcomes)]
    run = _run(queue_command)
    assert run.returncode != 0
    assert not (tmp_path / "queue").exists()


def test_fixed_pilot_keeps_one_prompt_option_and_rejects_exclusions(tmp_path):
    _, command = _fixture(tmp_path)
    run = _run(command + ["--prompt-count", "1"])
    assert run.returncode == 0, run.stderr
    manifest = json.loads((tmp_path / "plan/run_manifest.json").read_text())
    assert manifest["summary"]["bulk_task_count"] == 12
    assert manifest["pilot"]["execution_mode"] == "fixed"
    run = _run(command[:-1] + [str(tmp_path / "other"), "--exclude-outcomes", "absent"])
    assert run.returncode != 0
    assert "requires all-available" in run.stderr


def test_multiple_disjoint_exclusions_can_exhaust_the_available_queue(tmp_path):
    _, command = _fixture(tmp_path)
    assert _run(command + ["--all-available"]).returncode == 0
    first = _saved_judgments(tmp_path / "first", tmp_path / "plan")
    second = _saved_judgments(tmp_path / "second", tmp_path / "plan")
    rows = first.read_text().splitlines()
    first.write_text("\n".join(rows[:18]) + "\n")
    second.write_text("\n".join(rows[18:]) + "\n")
    for path in (first, second):
        runtime_path = path.parent / "run_manifest.json"
        runtime = json.loads(runtime_path.read_text())
        runtime.update(status="checkpointed", completed_count=18, remaining_count=18)
        runtime_path.write_text(json.dumps(runtime))
    run = _run(command[:-1] + [str(tmp_path / "queue"), "--all-available",
                              "--exclude-outcomes", str(first), "--exclude-outcomes", str(second)])
    assert run.returncode == 0, run.stderr
    manifest = json.loads((tmp_path / "queue/run_manifest.json").read_text())
    assert manifest["summary"]["available_task_count"] == 36
    assert manifest["summary"]["excluded_task_count"] == 36
    assert manifest["summary"]["pending_task_count"] == 0
    assert (tmp_path / "queue/bulk_tasks.jsonl").read_bytes() == b""
    coverage = json.loads((tmp_path / "queue/excluded_coverage.json").read_text())
    assert len(coverage["sources"]) == 2
    assert len(coverage["excluded_task_ids"]) == 36
