"""Preparation verifies frozen metadata without touching generator results."""

import json
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.axis_permutation_inputs import load_study
from analysis.interpretability.pipeline.axis_permutation_report import analysis_settings
from analysis.scripts.prepare_axis_permutation_report import main, prepare_report_config
from analysis.tests.test_axis_permutation_inputs import (
    agentic,
    digest,
    identity,
    lines,
    write,
)


def selection(root, *, question="Question?", include_axes=True):
    prompts = lines(root / "pilot-prompts.jsonl", [{
        "candidate_id": "prompt", "question": question, "question_sha256": digest(question),
        "target_normalized_axis_1": 0.5,
    }])
    axes = lines(root / "pilot-axis.jsonl", [{
        "candidate_id": "prompt", "text_sha256": digest(question),
        **({"consensus_normalized_axis_1": 0.4} if include_axes else {}),
    }])
    records = lines(root / "selection-records.jsonl", [{"candidate_id": "prompt", "axis_bin": 0}])
    manifest = write(root / "selection-manifest.json", {
        "format_version": "agentic-new-prompt-cohort-v1",
        "artifacts": {"prompts": identity(prompts), "axis_map": identity(axes),
                      "selection_records": identity(records)},
    })
    return prompts, axes, records, manifest


def legacy(root, selected):
    prompts, _, records, _ = selected
    write(root / "run_manifest.json", {
        "format_version": "agentic-search-execution-calibration-v2",
        "prompt_sources": {"prompts_jsonl": identity(prompts),
                           "selection_records_jsonl": identity(records)},
    })
    return root


def paired(tmp_path):
    spec, _, _ = agentic(tmp_path)
    root = Path(spec["task_manifest"]).parent
    selected = selection(root / "cohort")
    plan = json.loads((root / "run_manifest.json").read_text())
    plan["cohort_manifest"] = identity(selected[3])
    write(root / "run_manifest.json", plan)
    return root, selected


def test_legacy_coordinates_use_exact_hash_join_and_leave_sources_unchanged(tmp_path):
    selected = selection(tmp_path / "selection")
    root = legacy(tmp_path / "shard-0", selected)
    before = {path: path.read_bytes() for path in selected}
    output = tmp_path / "report/config.json"
    config = prepare_report_config(output=output, study_id="ready", legacy_generator_roots=[root])
    assert analysis_settings(config)["axis_fields"]["target_normalized_axis_1"]["role"] == "assigned"
    assert config["axis_fields"]["consensus_normalized_axis_1"]["role"] == "observed"
    assert config["sources"]["direct"] == []
    assert config["preparation"]["result_scan_performed"] is False
    normalized = [json.loads(line) for line in Path(config["sources"]["agentic"][0]["axis_jsonl"]).read_text().splitlines()]
    assert normalized == [{"candidate_id": "prompt", "question_sha256": digest("Question?"),
                           "consensus_normalized_axis_1": 0.4}]
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert any(row.get("join") == "candidate_id+exact-text-sha256"
               for row in config["preparation"]["verified_artifacts"])


def test_prepared_paired_config_runs_real_report_input_adapter(tmp_path):
    root, _ = paired(tmp_path)
    output = tmp_path / "report/config.json"
    config = prepare_report_config(output=output, study_id="ready", cohort_run_roots=[root])
    study = load_study(config["sources"], output.parent)
    assert len(study["tasks"]) == 2
    assert len([task for task in study["tasks"] if task["ranking"] is not None]) == 1
    assert study["tasks"][0]["axes"] == {"target_normalized_axis_1": 0.5,
                                        "consensus_normalized_axis_1": 0.4}


def test_report_rechecks_prepared_coordinate_hash_before_first_snapshot(tmp_path):
    from analysis.scripts.report_axis_permutation_study import main as report_main

    root, _ = paired(tmp_path)
    output = tmp_path / "report/config.json"
    config = prepare_report_config(output=output, study_id="ready", cohort_run_roots=[root])
    sidecar = Path(config["sources"]["agentic"][0]["axis_jsonl"])
    row = json.loads(sidecar.read_text())
    row["consensus_normalized_axis_1"] = 0.99
    lines(sidecar, [row])
    with pytest.raises(ValueError, match="prepared artifact hash mismatch"):
        report_main(["--config", str(output), "--output-dir", str(tmp_path / "snapshots")])
    assert not (tmp_path / "snapshots/latest.json").exists()


def test_backlog_frozen_cohort_manifest_and_tasks_are_verified(tmp_path):
    root, _ = paired(tmp_path)
    plan = json.loads((root / "run_manifest.json").read_text())
    plan["format_version"] = "agentic-four-generator-backlog-v1"
    plan.pop("cohort_manifest")
    plan.pop("tasks")
    plan["frozen_files"] = {relative: digest((root / relative).read_bytes()) for relative in (
        "tasks.jsonl", "cohort/selection-manifest.json", "cohort/pilot-prompts.jsonl", "cohort/pilot-axis.jsonl")}
    write(root / "run_manifest.json", plan)
    config = prepare_report_config(output=tmp_path / "report.json", study_id="ready", cohort_run_roots=[root])
    assert len(load_study(config["sources"], tmp_path)["tasks"]) == 2
    (root / "tasks.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_report_config(output=tmp_path / "other.json", study_id="ready", cohort_run_roots=[root])


def test_no_result_scan_during_preparation(tmp_path, monkeypatch):
    root, _ = paired(tmp_path)
    def refuse_scan(*args, **kwargs):
        raise AssertionError("preparation scanned results")
    monkeypatch.setattr(Path, "glob", refuse_scan)
    monkeypatch.setattr(Path, "rglob", refuse_scan)
    prepare_report_config(output=tmp_path / "report.json", study_id="ready", cohort_run_roots=[root])


def test_missing_legacy_axis_manifest_does_not_invent_coordinates(tmp_path):
    selected = selection(tmp_path / "selection")
    root = legacy(tmp_path / "shard", selected)
    selected[3].unlink()
    config = prepare_report_config(output=tmp_path / "report.json", study_id="ready", legacy_generator_roots=[root])
    assert "axis_jsonl" not in config["sources"]["agentic"][0]
    assert config["preparation"]["coordinate_status"][0]["observed_coordinates"] == "missing_frozen_selection_manifest"


def test_missing_consensus_is_reported_without_substituting_percentile(tmp_path):
    selected = selection(tmp_path / "selection", include_axes=False)
    root = legacy(tmp_path / "shard", selected)
    config = prepare_report_config(output=tmp_path / "report.json", study_id="ready", legacy_generator_roots=[root])
    assert config["preparation"]["coordinate_status"][0]["observed_coordinates"] == "partial_or_missing_consensus_axis_1"
    rows = Path(config["sources"]["agentic"][0]["axis_jsonl"]).read_text()
    assert "normalized_axis_" not in rows


def test_conflicting_prompt_coordinate_is_not_silently_overwritten(tmp_path):
    selected = selection(tmp_path / "selection")
    row = json.loads(selected[0].read_text())
    row["consensus_normalized_axis_1"] = 0.9
    lines(selected[0], [row])
    manifest = json.loads(selected[3].read_text())
    manifest["artifacts"]["prompts"] = identity(selected[0])
    write(selected[3], manifest)
    root = legacy(tmp_path / "shard", selected)
    with pytest.raises(ValueError, match="disagree on observed coordinates"):
        prepare_report_config(output=tmp_path / "report.json", study_id="ready", legacy_generator_roots=[root])


@pytest.mark.parametrize("change", ["prompts", "axis", "question_hash"])
def test_changed_frozen_or_mismatched_coordinate_inputs_rejected(tmp_path, change):
    selected = selection(tmp_path / "selection")
    root = legacy(tmp_path / "shard", selected)
    if change == "question_hash":
        axes = lines(selected[1], [{"candidate_id": "prompt", "text_sha256": "wrong"}])
        manifest = json.loads(selected[3].read_text())
        manifest["artifacts"]["axis_map"] = identity(axes)
        write(selected[3], manifest)
    else:
        selected[0 if change == "prompts" else 1].write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        prepare_report_config(output=tmp_path / "report.json", study_id="ready", legacy_generator_roots=[root])
    assert not (tmp_path / "report.json").exists()


def test_immutable_idempotent_config_and_sidecar(tmp_path):
    selected = selection(tmp_path / "selection")
    root = legacy(tmp_path / "shard", selected)
    output = tmp_path / "report.json"
    kwargs = {"output": output, "study_id": "ready", "legacy_generator_roots": [root]}
    config = prepare_report_config(**kwargs)
    original = output.read_bytes(), output.stat().st_mtime_ns
    assert prepare_report_config(**kwargs) == config
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original
    with pytest.raises(FileExistsError, match="different report configuration"):
        prepare_report_config(**{**kwargs, "study_id": "changed"})
    Path(config["sources"]["agentic"][0]["axis_jsonl"]).write_text("corrupt")
    with pytest.raises(FileExistsError, match="different report input"):
        prepare_report_config(**kwargs)


def test_judge_plan_requires_pair_and_verifies_frozen_artifacts(tmp_path):
    root, _ = paired(tmp_path)
    args = {"output": tmp_path / "report.json", "study_id": "ready", "cohort_run_roots": [root]}
    with pytest.raises(ValueError, match="supplied together"):
        prepare_report_config(**args, judge_plan_manifest=tmp_path / "judge.json")
    tasks = lines(tmp_path / "judge/tasks.jsonl", [{"candidate_id": "placeholder"}])
    mapping = lines(tmp_path / "judge/private.jsonl", [{"candidate_id": "placeholder"}])
    plan = write(tmp_path / "judge/plan.json", {"format_version": "agentic-search-judge-recorded-conversation-v2",
        "artifacts": {"bulk_tasks": identity(tasks), "private_mapping": identity(mapping)}})
    config = prepare_report_config(**args, judge_plan_manifest=plan, judge_outcomes=tmp_path / "pending.jsonl")
    assert config["sources"]["judges"][0]["outcome_files"] == [str(tmp_path / "pending.jsonl")]


def test_cli_writes_config_and_prints_nonallocating_status(tmp_path, capsys):
    root, _ = paired(tmp_path)
    output = tmp_path / "report.json"
    assert main(["--cohort-run-root", str(root), "--study-id", "ready", "--output", str(output)]) == 0
    assert output.is_file()
    assert "INFERENCE_OR_ALLOCATION_STARTED=false" in capsys.readouterr().out
