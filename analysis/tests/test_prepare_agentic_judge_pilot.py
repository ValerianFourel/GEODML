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
