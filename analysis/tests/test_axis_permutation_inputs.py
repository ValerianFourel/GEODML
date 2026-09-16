"""Read-only adapters must not turn malformed journals into observations."""

import hashlib
import json
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeModel,
    build_agentic_judge_plan,
    write_agentic_judge_plan,
)
from analysis.interpretability.pipeline.axis_permutation_inputs import (
    _judges,
    _Reader,
    load_study,
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def identity(path):
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def direct(tmp_path):
    root = tmp_path / "direct"
    candidates = [{"candidate_id": f"C{i:03d}", "source_position": i,
                   "url": f"https://example.org/{i}", "title": f"Title {i}", "snippet": f"Evidence {i}"}
                  for i in (1, 2)]
    pool_core = {"engine": "duckduckgo", "search_snapshot_sha256": "a" * 64,
                 "query": "keyword", "candidates": candidates}
    pool = {"candidate_set_id": "candidate-set:" + digest(canonical(pool_core))[:20],
            "search_engine": "duckduckgo", "search_snapshot_sha256": "a" * 64,
            "search_query": "keyword", "keyword": "keyword", "candidates": candidates}
    model = {"model_id": "model", "model_revision": "b" * 40, "backend": "api", "precision": "bf16",
             "temperature": 0.0, "max_new_tokens": 20}
    task = {"prompt_candidate_id": "prompt", "prompt_sha256": digest("Question?"),
            "candidate_set_id": pool["candidate_set_id"], "top_n": 2,
            "reranker_configuration_id": "reranker-config:" + digest(canonical(model))[:20],
            "rendered_prompt_sha256": digest("Rendered prompt")}
    task["task_id"] = "readiness-rerank-task:" + digest(canonical(task))[:24]
    task.update(format_version="readiness-to-permutation-plan-v1", question="Question?", keyword="keyword",
                rendered_prompt="Rendered prompt", reranker_model="model", reranker_model_revision="b" * 40,
                reranker_backend="api", reranker_precision="bf16", temperature=0.0, max_new_tokens=20,
                model_native_web_search=False, search_engine="duckduckgo", candidate_ids=["C001", "C002"],
                search_query="keyword", search_snapshot_sha256="a" * 64,
                candidate_source_positions=[1, 2], readiness_coordinates={"target_normalized_axis_1": 0.5})
    tasks = lines(root / "tasks.jsonl", [task])
    pools = lines(root / "pools.jsonl", [pool])
    manifest = write(root / "run_manifest.json", {"format_version": task["format_version"], "plan_id": "plan",
        "rerankers": [model], "artifacts": {"rerank_tasks": identity(tasks), "frozen_candidate_sets": identity(pools)}})
    raw = "C002\nC001"
    outcome = {k: task[k] for k in ("task_id", "candidate_set_id", "prompt_candidate_id",
               "reranker_configuration_id", "reranker_model", "reranker_model_revision")}
    outcome.update(candidate_ids=["C002", "C001"], source_position_vector=[2, 1], raw_model_output=raw,
                   raw_model_output_sha256=digest(raw), permutation_sha256=digest(raw), fake_backend=False)
    output = lines(root / "run/permutation_outcomes.jsonl", [outcome])
    write(output.parent / "run_manifest.json", {"tasks": identity(tasks), "plan": {"plan_id": "plan"},
          "reranker": {"configuration_id": task["reranker_configuration_id"], "model": "model", "revision": "b" * 40},
          "fake_backend": False})
    return {"plan_manifest": str(manifest), "outcome_files": [str(output)]}, task, outcome


def agentic(tmp_path):
    root = tmp_path / "agentic"
    question = "Question?"
    prompt = {"candidate_id": "prompt", "question": question, "question_sha256": digest(question),
              "keyword": "keyword", "axis_bin": "middle", "target_normalized_axis_1": 0.5}
    prompts = lines(root / "prompts.jsonl", [prompt])
    core = {"prompt_id": "prompt", "prompt_sha256": digest(question), "method": "Parallel-Expansion-v1",
            "engine": "duckduckgo", "condition": "natural"}
    cell = {"cell_id": digest(canonical(core))[:20], **core}
    pending_core = {**core, "condition": "ablated"}
    pending = {"cell_id": digest(canonical(pending_core))[:20], **pending_core}
    tasks = lines(root / "tasks.jsonl", [cell, pending])
    plan = write(root / "run_manifest.json", {"format_version": "agentic-paired-new-cohort-trial-v1",
                 "tasks": identity(tasks), "models": {"qwen": {"model_id": "model", "model_revision": "b" * 40}}})
    output = root / "models/qwen/outputs/worker-00000"
    write(output / "run_manifest.json", {"model_id": "model", "model_revision": "b" * 40, "config_sha256": "e" * 64})
    evidence = [{"url": f"https://example.org/{i}", "title": f"Title {i}", "text": f"Evidence {i}"}
                for i in (1, 2)]
    trace = {"format_version": "agentic-search-trace-v1", "method_id": core["method"],
             "condition": core["condition"], "search_engine": core["engine"], "user_prompt_sha256": digest(question),
             "events": [{"event_type": "compaction", "payload": {"selected_snippets": evidence}}]}
    trace["trace_sha256"] = digest(canonical(trace))
    trace_path = write(output / "traces" / (cell["cell_id"] + ".json"), trace)
    result = {**cell, "ranking": [e["url"] for e in evidence], "answer": "Answer", "final_snippet_count": 2,
              "trace": str(trace_path), "trace_sha256": trace["trace_sha256"]}
    result_path = write(output / "results" / (cell["cell_id"] + ".json"), result)
    config = {"prompts_jsonl": str(prompts), "task_manifest": str(plan), "generator_roots": [str(root / "models/qwen")]}
    return config, prompt, result_path


def judge(tmp_path, generator_spec, prompt, result_path, *, recorded_conversation=False, extra_results=()):
    model = AgenticJudgeModel(role="bulk", model_id="nvidia/nemotron", model_revision="c" * 40)
    validation = AgenticJudgeModel(role="validation", model_id="glm", model_revision="d" * 40)
    plan = build_agentic_judge_plan([result_path, *extra_results], prompt_rows=[prompt],
        generator_model_by_root={generator_spec["generator_roots"][0]: "model"}, bulk_model=model,
        validation_model=validation, validation_fraction=1.0, master_seed=10,
        recorded_conversation=recorded_conversation)
    artifact = write_agentic_judge_plan(tmp_path / "judge/plan", plan=plan)
    task = plan.bulk_tasks[0]
    ids = [e.evidence_id for e in task.evidence]
    parsed = {"request_fulfillment": 4, "evidence_grounding": 4, "judge_confidence": 4,
              "unsupported_claim_count": 0, "ideal_relevance_ranking": ids,
              "realized_support_ranking": [{"evidence_id": ids[0], "use_score": 4}]}
    raw = json.dumps(parsed)
    outcome = {"judge_task_id": task.judge_task_id, "blind_case_id": task.blind_case_id,
               "evidence_ids": ids, "raw_output": raw, "raw_output_sha256": digest(raw),
               "parsed_output": parsed, "fake_backend": False}
    outcome_path = lines(tmp_path / "judge/run/outcomes.jsonl", [outcome])
    write(outcome_path.parent / "run_manifest.json", {"model_id": model.model_id, "model_revision": model.model_revision,
          "pipeline": "agentic-judge-bulk", "fake_backend": False, "tasks": identity(artifact.bulk_tasks_path),
          "resume_identity": {"source_manifest_sha256": digest(artifact.manifest_path.read_bytes()),
                              "tasks_sha256": digest(artifact.bulk_tasks_path.read_bytes())}})
    return {"plan_manifest": str(artifact.manifest_path), "outcome_files": [str(outcome_path)]}, outcome


def test_direct_and_agentic_share_only_exact_evidence_pool(tmp_path):
    direct_spec, _, _ = direct(tmp_path)
    agentic_spec, _, _ = agentic(tmp_path)
    result = load_study({"direct": [direct_spec], "agentic": [agentic_spec]}, tmp_path)
    completed = [t for t in result["tasks"] if t["ranking"] is not None]
    assert len(result["tasks"]) == 3
    assert len(completed) == 2
    assert completed[0]["pool_id"] == completed[1]["pool_id"]
    assert completed[0]["ranking"] == ["https://example.org/2", "https://example.org/1"]
    assert not result["issues"]


def test_pending_direct_tasks_survive_missing_journal(tmp_path):
    spec, _, _ = direct(tmp_path)
    spec["outcome_files"] = ["not-started/outcomes.jsonl"]
    result = load_study({"direct": [spec]}, tmp_path)
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["ranking"] is None
    assert result["issues"][0]["code"] == "pending_journal"


def test_partial_tail_duplicate_and_conflict_are_visible(tmp_path):
    spec, _, outcome = direct(tmp_path)
    output = Path(spec["outcome_files"][0])
    lines(output, [outcome, outcome])
    with output.open("a") as stream:
        stream.write('{"task_id":')
    result = load_study({"direct": [spec]}, tmp_path)
    assert len(result["tasks"]) == 1
    assert {r["code"] for r in result["issues"]} == {"partial_journal_tail", "duplicate_identical_outcome"}
    reverse = {**outcome, "candidate_ids": ["C001", "C002"], "source_position_vector": [1, 2],
               "raw_model_output": "C001\nC002", "raw_model_output_sha256": digest("C001\nC002"),
               "permutation_sha256": digest("C001\nC002")}
    lines(output, [outcome, reverse, outcome])
    result = load_study({"direct": [spec]}, tmp_path)
    assert result["tasks"][0]["ranking"] is None
    assert result["issues"][0]["code"] == "conflicting_outcomes_quarantined"


def test_changed_frozen_tasks_are_fatal(tmp_path):
    spec, _, _ = direct(tmp_path)
    manifest = json.loads(Path(spec["plan_manifest"]).read_text())
    Path(manifest["artifacts"]["rerank_tasks"]["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_study({"direct": [spec]}, tmp_path)


def test_wrong_raw_parsed_and_unknown_direct_outcomes_are_not_counted(tmp_path):
    spec, _, outcome = direct(tmp_path)
    lines(Path(spec["outcome_files"][0]), [{**outcome, "candidate_ids": ["C001", "C002"]}, {**outcome, "task_id": "unknown"}])
    result = load_study({"direct": [spec]}, tmp_path)
    assert result["tasks"][0]["ranking"] is None
    assert len(result["issues"]) == 2


def test_judgments_verified_by_rebuilding_source_tasks(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    judge_spec, _ = judge(tmp_path, spec, prompt, result_path)
    config = {"agentic": [spec], "judges": [judge_spec]}
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = load_study(config, tmp_path)
    assert len(report["judge_tasks"]) == len(report["judgments"]) == 1
    judgment = report["judgments"][0]
    assert set(judgment["ideal_ranking"]) == {"https://example.org/1", "https://example.org/2"}
    assert judgment["ranking_visible"] is False
    assert judgment["judge_model_id"] == "nvidia/nemotron"
    assert not report["issues"]
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_pending_judgment_is_in_denominator(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    judge_spec, _ = judge(tmp_path, spec, prompt, result_path)
    judge_spec["outcome_files"] = []
    report = load_study({"agentic": [spec], "judges": [judge_spec]}, tmp_path)
    assert len(report["judge_tasks"]) == 1
    assert report["judgments"] == []


def test_tampered_trace_quarantines_generator_and_judge(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    judge_spec, _ = judge(tmp_path, spec, prompt, result_path)
    result = json.loads(result_path.read_text())
    trace_path = Path(result["trace"])
    trace = json.loads(trace_path.read_text())
    trace["events"][0]["payload"]["selected_snippets"][0]["text"] = "Changed"
    write(trace_path, trace)
    report = load_study({"agentic": [spec], "judges": [judge_spec]}, tmp_path)
    assert all(t["ranking"] is None for t in report["tasks"])
    assert len(report["judge_tasks"]) == 1
    assert report["judge_tasks"][0]["task_key"] is None
    assert report["judge_tasks"][0]["source_verified"] is False
    assert report["judgments"] == []
    assert {r["code"] for r in report["issues"]} >= {"invalid_agentic_result", "invalid_judge_task"}


def test_bad_axis_join_is_rejected(tmp_path):
    spec, _, _ = agentic(tmp_path)
    path = lines(tmp_path / "axes.jsonl", [{"candidate_id": "prompt", "question_sha256": "x", "target_normalized_axis_1": 0.3}])
    spec["axis_jsonl"] = str(path)
    with pytest.raises(ValueError, match="axis join"):
        load_study({"agentic": [spec]}, tmp_path)


def test_fake_outputs_are_flagged_not_scientifically_promoted(tmp_path):
    spec, _, outcome = direct(tmp_path)
    path = Path(spec["outcome_files"][0])
    lines(path, [{**outcome, "fake_backend": True}])
    runtime_path = path.parent / "run_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    write(runtime_path, {**runtime, "fake_backend": True})
    report = load_study({"direct": [spec]}, tmp_path)
    assert report["tasks"][0]["fake"] is True
    assert report["tasks"][0]["ranking"] is not None


def test_legacy_manifest_reconstructs_exact_planned_cells(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    output = result_path.parent.parent
    prompt["axis_bin"] = 0
    prompt_path = lines(Path(spec["prompts_jsonl"]), [prompt])
    records = lines(tmp_path / "records.jsonl", [{"candidate_id": "prompt", "axis_bin": 0}])
    selection = [{"prompt_id": "prompt", "question_sha256": digest(prompt["question"]), "axis_bin": 0}]
    write(output / "run_manifest.json", {"format_version": "agentic-search-execution-calibration-v2",
          "condition_mode": "frozen-target-url-and-stable-shuffle-v1", "model_id": "model", "model_revision": "b" * 40,
          "config_sha256": "e" * 64,
          "prompt_sources": {"prompts_jsonl": identity(prompt_path), "selection_records_jsonl": identity(records)},
          "prompt_selection_seed": 0, "prompt_count": 1, "prompt_selection_sha256": digest(canonical(selection)),
          "methods": ["Parallel-Expansion-v1"], "engines": ["duckduckgo"], "conditions": ["natural", "ablated"], "cell_count": 2})
    report = load_study({"agentic": [{"prompts_jsonl": str(prompt_path), "generator_roots": [str(output)]}]}, tmp_path)
    assert len(report["tasks"]) == 2
    assert sum(t["ranking"] is not None for t in report["tasks"]) == 1
    assert not report["issues"]


def test_full_conversation_judge_is_kept_separate_and_ranking_visible(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    result = json.loads(result_path.read_text())
    trace_path = Path(result["trace"])
    trace = json.loads(trace_path.read_text())
    trace.pop("trace_sha256")
    trace["events"][0]["event_index"] = 0
    evidence = trace["events"][0]["payload"]["selected_snippets"]
    trace["events"].append({"event_index": 1, "event_type": "llm_call", "payload": {
        "request": {"purpose": "parallel_final", "prompt": "Question?\n\nCOMPACTED SNIPPETS:\n" + json.dumps(evidence),
                    "response_schema": {}, "force_finish": True}, "raw_output": "Answer"}})
    trace["trace_sha256"] = digest(canonical(trace))
    write(trace_path, trace)
    write(result_path, {**result, "trace_sha256": trace["trace_sha256"]})
    judge_spec, _ = judge(tmp_path, spec, prompt, result_path, recorded_conversation=True)
    report = load_study({"agentic": [spec], "judges": [judge_spec]}, tmp_path)
    assert len(report["judgments"]) == 1
    assert report["judgments"][0]["ranking_visible"] is True
    assert report["judgments"][0]["protocol"] == "agentic-search-judge-recorded-conversation-v2"
    assert not report["issues"]


def test_wrong_judge_raw_parsed_is_not_counted(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    judge_spec, outcome = judge(tmp_path, spec, prompt, result_path)
    outcome["parsed_output"]["request_fulfillment"] = 1
    lines(Path(judge_spec["outcome_files"][0]), [outcome])
    report = load_study({"agentic": [spec], "judges": [judge_spec]}, tmp_path)
    assert len(report["judge_tasks"]) == 1
    assert report["judgments"] == []
    assert report["issues"][0]["code"] == "invalid_judge_outcome"


def test_not_started_model_from_frozen_outer_plan_has_pending_cells(tmp_path):
    spec, _, _ = agentic(tmp_path)
    manifest_path = Path(spec["task_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["models"]["llama"] = {"model_id": "llama", "model_revision": "c" * 40}
    write(manifest_path, manifest)
    spec["generator_roots"].append(str(manifest_path.parent / "models/llama"))
    report = load_study({"agentic": [spec]}, tmp_path)
    llama = [t for t in report["tasks"] if t["model_id"] == "llama"]
    assert len(llama) == 2
    assert all(t["ranking"] is None for t in llama)


def test_shared_result_copies_count_once_and_judge_links_the_copy(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    result = json.loads(result_path.read_text())
    original_output = result_path.parent.parent
    second_output = original_output.parent / "worker-00001"
    trace = json.loads(Path(result["trace"]).read_text())
    trace_path = write(second_output / "traces" / Path(result["trace"]).name, trace)
    second_result = write(second_output / "results" / result_path.name, {**result, "trace": str(trace_path)})
    write(second_output / "run_manifest.json", json.loads((original_output / "run_manifest.json").read_text()))
    judge_spec, _ = judge(tmp_path, spec, prompt, second_result)
    report = load_study({"agentic": [spec], "judges": [judge_spec]}, tmp_path)
    completed = [t for t in report["tasks"] if t["ranking"] is not None]
    assert len(completed) == 1
    assert len(completed[0]["source_result_sha256s"]) == 2
    assert len(report["judgments"]) == 1
    assert [i["code"] for i in report["issues"]] == ["duplicate_identical_outcome"]


def test_duplicate_cell_in_frozen_generator_queue_is_fatal(tmp_path):
    spec, _, _ = agentic(tmp_path)
    manifest_path = Path(spec["task_manifest"])
    manifest = json.loads(manifest_path.read_text())
    tasks_path = Path(manifest["tasks"]["path"])
    first = json.loads(tasks_path.read_text().splitlines()[0])
    lines(tasks_path, [first, first])
    manifest["tasks"] = identity(tasks_path)
    write(manifest_path, manifest)
    with pytest.raises(ValueError, match="duplicate cell_id"):
        load_study({"agentic": [spec]}, tmp_path)


def test_changed_answer_in_shared_copy_is_a_conflict(tmp_path):
    spec, _, result_path = agentic(tmp_path)
    result = json.loads(result_path.read_text())
    original_output = result_path.parent.parent
    second_output = original_output.parent / "worker-00001"
    write(second_output / "results" / result_path.name, {**result, "answer": "Different answer"})
    write(second_output / "run_manifest.json", json.loads((original_output / "run_manifest.json").read_text()))
    report = load_study({"agentic": [spec]}, tmp_path)
    assert all(task["ranking"] is None for task in report["tasks"])
    assert [issue["code"] for issue in report["issues"]] == ["conflicting_outcomes_quarantined"]


def test_reader_byte_cache_is_bounded_and_rechecks_evicted_sources(tmp_path):
    reader = _Reader(tmp_path, cache_limit_bytes=8)
    one, two, large = (tmp_path / name for name in ("one", "two", "large"))
    one.write_bytes(b"12345")
    two.write_bytes(b"67890")
    large.write_bytes(b"x" * 20)
    assert reader.bytes(one) == b"12345"
    assert reader.bytes(two) == b"67890"
    assert one not in reader.cache
    assert reader.cache_size_bytes <= 8
    assert reader.bytes(one) == b"12345"
    assert reader.bytes(large) == b"x" * 20
    assert large not in reader.cache
    assert reader.cache_size_bytes <= 8
    reader.bytes(two)
    one.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during"):
        reader.bytes(one)


def test_judge_source_index_scans_generated_tasks_once(tmp_path):
    spec, prompt, result_path = agentic(tmp_path)
    result = json.loads(result_path.read_text())
    core = {key: result[key] for key in ("prompt_id", "prompt_sha256", "method", "engine", "condition")}
    core["condition"] = "ablated"
    cell_id = digest(canonical(core))[:20]
    trace = json.loads(Path(result["trace"]).read_text())
    trace.pop("trace_sha256")
    trace["condition"] = "ablated"
    trace["trace_sha256"] = digest(canonical(trace))
    trace_path = write(result_path.parent.parent / "traces" / (cell_id + ".json"), trace)
    second_result = write(result_path.parent / (cell_id + ".json"), {
        **result, **core, "cell_id": cell_id, "trace": str(trace_path), "trace_sha256": trace["trace_sha256"]})
    judge_spec, _ = judge(tmp_path, spec, prompt, result_path, extra_results=(second_result,))
    generated = load_study({"agentic": [spec]}, tmp_path)["tasks"]

    class CountedTasks(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    counted = CountedTasks(generated)
    reader = _Reader(tmp_path)
    planned, saved = _judges(reader, judge_spec, counted)
    assert len(planned) == 2
    assert len(saved) == 1
    assert counted.scans == 1
