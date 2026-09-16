"""Incremental reports preserve history and do not manufacture matched rankings."""

import copy
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.axis_permutation_report import (
    CONFIG_VERSION,
    FORMAT_VERSION,
    build_report,
    render_markdown,
    save_report,
)


def config():
    return {"format_version": CONFIG_VERSION, "study_id": "test-study",
            "axis_fields": {"B": {"role": "assigned", "construct": "first-party-source-preference", "domain": [0, 1]},
                            "latent": {"role": "observed", "construct": "measured-coordinate", "domain": [-1, 1]}},
            "fit": {"min_train": 2, "min_test": 1}}


def task(key="t1", **changes):
    question = "Which source fits?"
    row = {"task_key": key, "source_kind": "agentic", "cohort_id": "cohort", "protocol": "agentic",
           "configuration_id": "configuration", "model_id": "qwen", "model_revision": "revision",
           "execution_configuration_id": "execution-config",
           "method": "parallel", "engine": "duckduckgo", "condition": "natural", "keyword": "sources",
           "prompt_id": "prompt1", "question": question, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
           "axes": {"B": 0.5}, "candidate_ids": ["url-a", "url-b"], "pool_id": "pool", "fit_pool_id": "ordered-pool",
           "ranking": ["url-a", "url-b"], "source_cell_id": key,
           "source_result_sha256": "result", "source_trace_sha256": "trace", "fake": False}
    row.update(changes)
    return row


def judgment(task_key="t1", **changes):
    row = {"judge_task_id": "j1", "task_key": task_key, "judge_model_id": "nemotron", "judge_model_revision": "jrev",
           "protocol": "judge-v2", "ranking_visible": True, "ideal_ranking": ["url-b", "url-a"],
           "support_ranking": ["url-a"], "scores": {"request_fulfillment": 3}, "fake": False}
    row.update(changes)
    return row


def study(tasks=None, judgments=None, plans=None):
    return {"tasks": tasks if tasks is not None else [task()], "judgments": judgments or [],
            "judge_tasks": plans if plans is not None else judgments or [], "sources": [], "issues": []}


def test_missing_direct_study_is_unavailable_not_zero_agreement():
    report = build_report(study(judgments=[judgment()]), config())
    assert report["study_availability"]["direct"] == "unavailable"
    assert report["study_availability"]["agentic"] == "planned"
    assert "Fixed-candidate direct study: unavailable" in render_markdown(report)
    assert not any(row["family"] == "fixed_candidate_direct_vs_judge" for row in report["agreement"])
    direct = task("direct", source_kind="direct", protocol="fixed-candidate", ranking=None)
    report = build_report(study([task(), direct]), config())
    assert report["study_availability"]["direct"] == "planned"


def test_snapshot_increment_deduplicates_and_preserves_existing(tmp_path):
    pending = task("t2", ranking=None, candidate_ids=[], pool_id=None, fit_pool_id=None,
                   source_result_sha256=None, source_trace_sha256=None)
    first = build_report(study([task(), pending]), config())
    old = save_report(first, tmp_path)
    before = (old / "report.json").read_bytes()
    latest_bytes = (tmp_path / "latest.json").read_bytes()
    assert save_report(first, tmp_path) == old
    assert (tmp_path / "latest.json").read_bytes() == latest_bytes
    second = build_report(study([task(), task("t2")], [judgment()]), config())
    new = save_report(second, tmp_path)
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert old != new
    assert (old / "report.json").read_bytes() == before
    assert latest["new_completed"] == latest["new_judgments"] == 1
    assert latest["previous_snapshot_id"] == old.name
    assert second["generation"]["completed"] == 2
    assert "Ranking visible" in (new / "report.md").read_text()
    question = json.loads((new / "questions.jsonl").read_text().splitlines()[0])
    assert len(question["tasks"]) == 2
    assert len(question["tasks"][0]["judge_comparisons"]) == 2


def test_late_coordinates_allowed_existing_coordinates_frozen(tmp_path):
    save_report(build_report(study(), config()), tmp_path)
    augmented = task(axes={"B": 0.5, "latent": -0.25})
    save_report(build_report(study([augmented]), config()), tmp_path)
    augmented["axes"]["B"] = 0.6
    with pytest.raises(ValueError, match="coordinate changed"):
        save_report(build_report(study([augmented]), config()), tmp_path)


@pytest.mark.parametrize("change", [
    {"ranking": ["url-b", "url-a"]}, {"source_result_sha256": "new-result"},
    {"source_trace_sha256": "new-trace"}, {"ranking": None}, {"fake": True},
])
def test_completed_observation_cannot_change(tmp_path, change):
    save_report(build_report(study(), config()), tmp_path)
    before = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="completed ranking changed"):
        save_report(build_report(study([task(**change)]), config()), tmp_path)
    assert (tmp_path / "latest.json").read_bytes() == before


def test_removal_configuration_or_analysis_change_refused(tmp_path):
    save_report(build_report(study(), config()), tmp_path)
    for changed in (study([]), study([task(configuration_id="different")])):
        with pytest.raises(ValueError, match="planned task changed"):
            save_report(build_report(changed, config()), tmp_path)
    changed_config = config()
    changed_config["fit"]["ridge"] = 2.0
    with pytest.raises(ValueError, match="analysis settings changed"):
        save_report(build_report(study(), changed_config), tmp_path)


def test_changed_judgment_is_not_a_new_observation(tmp_path):
    save_report(build_report(study(judgments=[judgment()]), config()), tmp_path)
    changed = judgment(ideal_ranking=["url-a", "url-b"])
    with pytest.raises(ValueError, match="completed judgment changed"):
        save_report(build_report(study(judgments=[changed]), config()), tmp_path)


def test_pending_judge_plan_cannot_disappear_but_source_can_arrive(tmp_path):
    planned = judgment(task_key=None, source_verified=False)
    save_report(build_report(study(plans=[planned]), config()), tmp_path)
    save_report(build_report(study(plans=[judgment(source_verified=True)]), config()), tmp_path)
    with pytest.raises(ValueError, match="planned judgment changed or disappeared"):
        save_report(build_report(study(plans=[]), config()), tmp_path)
    with pytest.raises(ValueError, match="planned judgment changed or disappeared"):
        save_report(build_report(study(plans=[planned]), config()), tmp_path)


def test_report_writer_lock_fails_without_modifying_snapshot(tmp_path):
    with (tmp_path / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            save_report(build_report(study(), config()), tmp_path)
    assert not (tmp_path / "latest.json").exists()


def test_publish_recovers_snapshot_written_before_pointer(tmp_path):
    report = build_report(study(), config())
    destination = save_report(report, tmp_path)
    (tmp_path / "latest.json").unlink()
    assert save_report(report, tmp_path) == destination
    assert json.loads((tmp_path / "latest.json").read_text())["snapshot_id"] == destination.name


def test_untrusted_or_modified_latest_is_rejected(tmp_path):
    report = build_report(study(), config())
    destination = save_report(report, tmp_path)
    (destination / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="changed on disk"):
        save_report(report, tmp_path)
    (tmp_path / "latest.json").write_text(json.dumps({"format_version": FORMAT_VERSION, "snapshot_id": "../../bad"}))
    with pytest.raises(ValueError, match="snapshot identity"):
        save_report(report, tmp_path)


def test_pending_fake_empty_and_judge_denominators():
    rows = [task(), task("missing", ranking=None), task("fake", fake=True), task("empty", ranking=[])]
    planned = [judgment(), judgment("missing", judge_task_id="j2")]
    report = build_report(study(rows, [judgment()], planned), config())
    assert report["generation"] == {"expected": 4, "completed": 2, "remaining": 2, "percent": 50.0,
                                     "fake_excluded": 1, "empty_rankings": 1}
    assert report["judging"]["expected_materialized"] == 2
    assert report["judging"]["completed"] == report["judging"]["remaining_materialized"] == 1
    assert len(report["pending"]) == 3
    empty = build_report(study([task(ranking=[])], [judgment(ideal_ranking=[], support_ranking=[])]), config())
    assert empty["comparisons"][0]["metrics"]["exact_match"] is None


def test_exact_direct_matching_and_protocol_separation():
    direct = task("direct", source_kind="direct", protocol="direct", method="direct", condition="fixed")
    report = build_report(study([task(), direct], [judgment()]), config())
    assert len(report["comparisons"]) == 4
    assert {r["family"] for r in report["comparisons"]} == {"source_generator_vs_judge", "fixed_candidate_direct_vs_judge"}
    assert report["judging"]["without_exact_fixed_candidate_direct_match"] == 0
    for change in ({"pool_id": "changed-evidence"}, {"question_sha256": "a" * 64},
                   {"model_revision": "different"}, {"engine": "different"}):
        changed = {**direct, **change}
        result = build_report(study([task(), changed], [judgment()]), config())
        assert len(result["comparisons"]) == 2
        assert result["judging"]["without_exact_fixed_candidate_direct_match"] == 1


def test_visible_and_hidden_rankings_not_pooled():
    judges = [judgment(), judgment(judge_task_id="j2", protocol="judge-v1", ranking_visible=False)]
    report = build_report(study(judgments=judges), config())
    assert len(report["agreement"]) == 4
    assert {r["ranking_visible"] for r in report["agreement"]} == {True, False}


def test_fit_config_order_query_and_model_pools_are_separate():
    rows = [task(), task("cfg", configuration_id="cfg2"), task("rev", model_revision="rev2"),
            task("order", fit_pool_id="different-order"), task("query", keyword="another query")]
    report = build_report(study(rows), config())
    assert len(report["fits"]) == len(rows) * 2
    assert {r["status"] for r in report["fits"]} == {"insufficient_data"}
    assert report["scientific_result"] is False


def test_full_fits_and_question_predictions_when_support_arrives(tmp_path):
    rows = []
    for index in range(80):
        question = f"Variant {index}?"
        axis = index / 79
        rows.append(task(str(index), question=question, question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                         prompt_id=str(index), axes={"B": axis},
                         ranking=["url-a", "url-b"] if axis < .5 else ["url-b", "url-a"]))
    original = copy.deepcopy(rows)
    report = build_report(study(rows), config())
    assert rows == original
    fit = next(r for r in report["fits"] if r["axis_field"] == "B")
    assert fit["status"] == "fit"
    assert fit["heldout_improvement"]["log_loss_reduction"] > 0
    assert len(fit["predictions"]) == 80
    assert all(q["tasks"][0]["fits"] for q in report["questions"])
    destination = save_report(report, tmp_path)
    assert "Held-out loss improvement" in (destination / "report.md").read_text()
    assert json.loads((destination / "report.json").read_text()) == report


def test_config_requires_named_construct_and_axis_role():
    changed = config()
    changed["axis_fields"]["B"]["role"] = "embedding_is_treatment"
    with pytest.raises(ValueError, match="requires a role"):
        build_report(study(), changed)


def test_mixed_unknown_configurations_excluded_without_losing_progress():
    report = build_report(study([task(configuration_mixed=True),
                                 task("unknown", execution_configuration_id=None)]), config())
    assert report["generation"]["completed"] == 2
    assert report["fits"] == []
    assert report["fit_exclusions"] == {"mixed_execution_configuration": 1, "unknown_execution_configuration": 1}


def test_report_deterministic_across_judge_and_axis_specification_order():
    judges = [judgment(judge_model_id="z-model"), judgment(judge_model_id="a-model")]
    first = build_report(study(judgments=judges), config())
    changed = config()
    changed["axis_fields"] = dict(reversed(list(changed["axis_fields"].items())))
    second = build_report(study(judgments=list(reversed(judges))), changed)
    assert first == second


def test_file_backed_cli_updates_are_read_only_and_match_exact_sources(tmp_path, capsys):
    from analysis.scripts.report_axis_permutation_study import main
    from analysis.tests.test_axis_permutation_inputs import (
        agentic,
        direct,
        judge,
        lines,
        write,
    )

    direct_spec, _, direct_outcome = direct(tmp_path)
    agentic_spec, prompt, result_path = agentic(tmp_path)
    judge_spec, judge_outcome = judge(tmp_path, agentic_spec, prompt, result_path)
    direct_file = Path(direct_spec["outcome_files"][0])
    judge_file = Path(judge_spec["outcome_files"][0])
    lines(direct_file, [])
    lines(judge_file, [])
    settings = config()
    settings["axis_fields"] = {"target_normalized_axis_1": {"role": "assigned", "construct": "decision-readiness"}}
    settings["sources"] = {"direct": [direct_spec], "agentic": [agentic_spec], "judges": [judge_spec]}
    config_path = write(tmp_path / "study.json", settings)
    output = tmp_path / "report"
    inputs = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    argv = ["--config", str(config_path), "--output-dir", str(output)]
    assert main(argv) == 0
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    latest = json.loads((output / "latest.json").read_text())
    first = json.loads((output / "snapshots" / latest["snapshot_id"] / "report.json").read_text())
    assert first["generation"]["completed"] == 1
    assert first["judging"]["expected_materialized"] == 1
    assert first["judging"]["completed"] == 0

    lines(direct_file, [direct_outcome])
    lines(judge_file, [judge_outcome])
    assert main(argv) == 0
    latest = json.loads((output / "latest.json").read_text())
    second = json.loads((output / "snapshots" / latest["snapshot_id"] / "report.json").read_text())
    assert second["generation"]["completed"] == 2
    assert second["judging"]["completed"] == 1
    assert len(second["comparisons"]) == 4
    assert second["issues"] == []
    assert latest["new_completed"] == latest["new_judgments"] == 1
    assert "SCIENTIFIC_RESULT=false" in capsys.readouterr().out
