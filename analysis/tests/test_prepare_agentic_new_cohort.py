"""CPU contracts for a shared new-prompt cohort with no pilot overlap."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from analysis.scripts.run_agentic_search_integration_smoke import (
    _cells,
    _load_calibration_prompts,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "analysis/scripts/prepare_agentic_new_cohort.py"
COMMIT = "a" * 40


def _write_rows(path: Path, rows: list[dict]) -> dict:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
    }


def _fixture(root: Path) -> Path:
    prompts = []
    axes = []
    for keyword in range(5):
        for axis_bin in range(4):
            for copy in range(10):
                identity = f"p-{keyword}-{axis_bin}-{copy}"
                question = f"Question {identity}?"
                digest = hashlib.sha256(question.encode()).hexdigest()
                percentile = (axis_bin + (copy + 1) / 11) / 4
                prompts.append(
                    {
                        "candidate_id": identity,
                        "question": question,
                        "question_sha256": digest,
                        "keyword": f"topic {keyword}",
                        "keyword_id": str(keyword),
                        "target_normalized_axis_1": percentile,
                    }
                )
                axes.append(
                    {
                        "candidate_id": identity,
                        "text_sha256": digest,
                        "axis_1_percentile_0_1": percentile,
                    }
                )
    sources = {
        "prompts": _write_rows(root / "population.jsonl", prompts),
        "axis_map": _write_rows(root / "axes.jsonl", axes),
    }
    selection = root / "original"
    selection.mkdir()
    old_prompts = prompts[::10]
    old_axes = axes[::10]
    artifacts = {
        "prompts": _write_rows(selection / "pilot-prompts.jsonl", old_prompts),
        "axis_map": _write_rows(selection / "pilot-axis.jsonl", old_axes),
        "selection_records": _write_rows(
            selection / "selection-records.jsonl",
            [
                {"candidate_id": row["candidate_id"], "axis_bin": index % 4}
                for index, row in enumerate(old_prompts)
            ],
        ),
    }
    for artifact in artifacts.values():
        artifact["path"] = Path(artifact["path"]).name
    (selection / "selection-manifest.json").write_text(
        json.dumps(
            {
                "format_version": "readiness-axis-balanced-pilot-v1",
                "sources": sources,
                "artifacts": artifacts,
                "diagnostics": {"axis_bins": 4},
            }
        )
    )
    return selection


def _run(selection: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection-root",
            str(selection),
            "--prompt-count",
            "20",
            "--axis-bins",
            "4",
            "--expected-population-count",
            "200",
            "--expected-excluded-count",
            "20",
            "--master-seed",
            "20260916",
            "--source-git-commit",
            COMMIT,
            "--output-dir",
            str(output),
            *extra,
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shared_new_cohort_is_disjoint_reproducible_and_runner_compatible(tmp_path):
    selection = _fixture(tmp_path)
    preserved = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "selection-manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (output / "pilot-prompts.jsonl").read_text().splitlines()
    ]
    old_ids = {
        json.loads(line)["candidate_id"]
        for line in (selection / "pilot-prompts.jsonl").read_text().splitlines()
    }
    assert len(rows) == 20
    assert not old_ids.intersection(row["candidate_id"] for row in rows)
    assert manifest["excluded_prompt_count"] == 20
    assert manifest["eligible_population_count"] == 180
    assert manifest["scientific_result"] is False
    assert manifest["source_git_commit"] == COMMIT
    assert manifest["expected_cells_per_model"] == 240
    loaded = _load_calibration_prompts(
        output / "pilot-prompts.jsonl",
        output / "selection-records.jsonl",
        prompt_count=20,
        seed=20260912,
    )
    assert Counter(row.axis_bin for row in loaded) == {index: 5 for index in range(4)}
    assert len(_cells(loaded)) == 240
    assert all(path.read_bytes() == content for path, content in preserved.items())
    repeated = tmp_path / "repeated"
    assert _run(selection, repeated).returncode == 0
    for filename in (
        "pilot-prompts.jsonl",
        "pilot-axis.jsonl",
        "selection-records.jsonl",
    ):
        assert (output / filename).read_bytes() == (repeated / filename).read_bytes()
    assert _run(selection, output).returncode != 0


@pytest.mark.parametrize(
    "relative",
    [
        "population.jsonl",
        "axes.jsonl",
        "original/pilot-prompts.jsonl",
        "original/pilot-axis.jsonl",
        "original/selection-records.jsonl",
    ],
)
def test_source_or_exclusion_hash_mismatch_fails_before_output(tmp_path, relative):
    selection = _fixture(tmp_path)
    path = tmp_path / relative
    path.write_text(path.read_text() + "\n")
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode != 0
    assert "hash mismatch" in run.stderr
    assert not output.exists()


def test_non_divisible_cohort_is_rejected_before_output(tmp_path):
    selection = _fixture(tmp_path)
    output = tmp_path / "cohort"
    run = _run(selection, output, "--prompt-count", "21")
    assert run.returncode != 0
    assert "multiple of axis bins" in run.stderr
    assert not output.exists()


def test_new_ids_with_old_question_text_are_also_excluded(tmp_path):
    selection = _fixture(tmp_path)
    prompts = [
        json.loads(line)
        for line in (tmp_path / "population.jsonl").read_text().splitlines()
    ]
    axes = [
        json.loads(line) for line in (tmp_path / "axes.jsonl").read_text().splitlines()
    ]
    prompts[1]["question"] = "  " + prompts[0]["question"].upper() + "  \n"
    digest = hashlib.sha256(prompts[1]["question"].encode()).hexdigest()
    prompts[1]["question_sha256"] = digest
    axes[1]["text_sha256"] = digest
    original_path = selection / "selection-manifest.json"
    original = json.loads(original_path.read_text())
    original["sources"]["prompts"] = _write_rows(tmp_path / "population.jsonl", prompts)
    original["sources"]["axis_map"] = _write_rows(tmp_path / "axes.jsonl", axes)
    original_path.write_text(json.dumps(original))
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "selection-manifest.json").read_text())
    assert manifest["additional_text_overlap_count"] == 1
    assert manifest["eligible_population_count"] == 179
    rows = [
        json.loads(line)
        for line in (output / "pilot-prompts.jsonl").read_text().splitlines()
    ]
    assert prompts[1]["candidate_id"] not in {row["candidate_id"] for row in rows}
    axis_by_id = {row["candidate_id"]: row for row in axes}
    selected_axes = [
        json.loads(line)
        for line in (output / "pilot-axis.jsonl").read_text().splitlines()
    ]
    assert all(row == axis_by_id[row["candidate_id"]] for row in selected_axes)


def test_wrong_expected_source_count_fails_before_output(tmp_path):
    selection = _fixture(tmp_path)
    output = tmp_path / "cohort"
    run = _run(selection, output, "--expected-population-count", "201")
    assert run.returncode != 0
    assert "count differs" in run.stderr
    assert not output.exists()


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _prior_cohort(selection: Path, root: Path) -> Path:
    result = _run(selection, root)
    assert result.returncode == 0, result.stderr
    return root


def test_extra_cohorts_exclude_id_union_and_record_verified_provenance(tmp_path):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    second = _prior_cohort(selection, tmp_path / "same-prior-selection")
    output = tmp_path / "cohort"
    result = _run(
        selection, output,
        "--exclude-cohort-root", str(prior),
        "--exclude-cohort-root", str(second),
    )
    assert result.returncode == 0, result.stderr
    excluded = {
        row["candidate_id"]
        for root in (selection, prior, second)
        for row in _read_rows(root / "pilot-prompts.jsonl")
    }
    rows = _read_rows(output / "pilot-prompts.jsonl")
    assert len(rows) == 20
    assert not excluded.intersection(row["candidate_id"] for row in rows)
    manifest = json.loads((output / "selection-manifest.json").read_text())
    assert manifest["original_excluded_prompt_count"] == 20
    assert manifest["additional_excluded_prompt_count"] == 20
    assert manifest["excluded_prompt_count"] == 40
    assert manifest["eligible_population_count"] == 160
    assert len(manifest["additional_exclusions"]) == 2
    for provenance, root in zip(manifest["additional_exclusions"], (prior, second)):
        assert provenance["prompt_count"] == 20
        for key, filename in (
            ("selection_manifest", "selection-manifest.json"),
            ("prompts", "pilot-prompts.jsonl"),
            ("axis_map", "pilot-axis.jsonl"),
            ("selection_records", "selection-records.jsonl"),
        ):
            path = root / filename
            assert provenance[key]["path"] == str(path.resolve())
            assert provenance[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    repeated = tmp_path / "repeated"
    assert _run(
        selection, repeated,
        "--exclude-cohort-root", str(prior),
        "--exclude-cohort-root", str(second),
    ).returncode == 0
    assert (output / "selection-manifest.json").read_bytes() == (
        repeated / "selection-manifest.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "filename",
    ["pilot-prompts.jsonl", "pilot-axis.jsonl", "selection-records.jsonl"],
)
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_extra_exclusion_artifacts_fail_closed(tmp_path, filename, damage):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / filename
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n")
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "No such file" in result.stderr or "hash mismatch" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("field", ["prompts", "axis_map"])
@pytest.mark.parametrize("changed", ["sha256", "rows"])
def test_extra_exclusion_from_different_population_fails_closed(tmp_path, field, changed):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sources"][field][changed] = "0" * 64 if changed == "sha256" else 199
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "population source differs" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["prompts", "axis_map", "selection_records"])
@pytest.mark.parametrize("invalid_id", ["duplicate", "outside-population"])
def test_extra_exclusion_candidate_ids_fail_closed(tmp_path, artifact, invalid_id):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    manifest = json.loads(path.read_text())
    artifact_path = prior / manifest["artifacts"][artifact]["path"]
    rows = _read_rows(artifact_path)
    rows[0]["candidate_id"] = (
        rows[1]["candidate_id"] if invalid_id == "duplicate" else "unknown-candidate"
    )
    manifest["artifacts"][artifact] = _write_rows(artifact_path, rows)
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "IDs are duplicated or outside" in result.stderr or "ID sets differ" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("field", ["prompt_count", "population_count", "axis_bins"])
def test_extra_exclusion_manifest_counts_fail_closed(tmp_path, field):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    manifest = json.loads(path.read_text())
    target = manifest["diagnostics"] if field == "axis_bins" else manifest
    target[field] += 1
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "count differs" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("damage", ["missing", "wrong-format"])
def test_extra_exclusion_requires_cohort_manifest(tmp_path, damage):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    if damage == "missing":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        manifest["format_version"] = "unknown"
        path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "No such file" in result.stderr or "agentic-new-prompt-cohort-v1" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["prompts", "axis_map", "selection_records"])
def test_extra_exclusion_rows_must_match_population_even_with_valid_hash(tmp_path, artifact):
    selection = _fixture(tmp_path)
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    manifest = json.loads(path.read_text())
    artifact_path = prior / manifest["artifacts"][artifact]["path"]
    rows = _read_rows(artifact_path)
    if artifact == "prompts":
        rows[0]["question"] += " changed"
    elif artifact == "axis_map":
        rows[0]["axis_1_percentile_0_1"] = 0.987
    else:
        rows[0]["axis_bin"] = (rows[0]["axis_bin"] + 1) % 4
    manifest["artifacts"][artifact] = _write_rows(artifact_path, rows)
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode != 0
    assert "frozen population" in result.stderr
    assert not output.exists()


def test_extra_exclusion_also_removes_equivalent_text_under_other_ids(tmp_path):
    selection = _fixture(tmp_path)
    prompts = _read_rows(tmp_path / "population.jsonl")
    axes = _read_rows(tmp_path / "axes.jsonl")
    prompts[2]["question"] = "  " + prompts[1]["question"].upper() + "  \n"
    digest = hashlib.sha256(prompts[2]["question"].encode()).hexdigest()
    prompts[2]["question_sha256"] = digest
    axes[2]["text_sha256"] = digest
    original_path = selection / "selection-manifest.json"
    original = json.loads(original_path.read_text())
    original["sources"]["prompts"] = _write_rows(tmp_path / "population.jsonl", prompts)
    original["sources"]["axis_map"] = _write_rows(tmp_path / "axes.jsonl", axes)
    original_path.write_text(json.dumps(original))
    prior = _prior_cohort(selection, tmp_path / "prior")
    path = prior / "selection-manifest.json"
    manifest = json.loads(path.read_text())
    # Use a known, valid population subset containing the first duplicate text.
    chosen = list(range(1, 200, 10))
    for artifact, filename, rows in (
        ("prompts", "pilot-prompts.jsonl", [prompts[i] for i in chosen]),
        ("axis_map", "pilot-axis.jsonl", [axes[i] for i in chosen]),
        ("selection_records", "selection-records.jsonl", [
            {"candidate_id": prompts[i]["candidate_id"], "axis_bin": (i // 10) % 4}
            for i in chosen
        ]),
    ):
        manifest["artifacts"][artifact] = _write_rows(prior / filename, rows)
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cohort"
    result = _run(selection, output, "--exclude-cohort-root", str(prior))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "selection-manifest.json").read_text())
    assert manifest["excluded_prompt_count"] == 40
    assert manifest["additional_text_overlap_count"] == 1
    assert manifest["eligible_population_count"] == 159
    assert prompts[2]["candidate_id"] not in {
        row["candidate_id"] for row in _read_rows(output / "pilot-prompts.jsonl")
    }


def test_no_extra_cohort_preserves_historical_artifact_bytes_and_manifest_schema(tmp_path):
    selection = _fixture(tmp_path)
    output = _prior_cohort(selection, tmp_path / "cohort")
    expected = {
        "selection-records.jsonl": "4637827ce1ad51a53c039e85b280f2b672d1cc6f3abd92a4c837a6b2ebffde0e",
        "pilot-prompts.jsonl": "bca25191c41539f3fcf6c044e503d041ab0c985d939faf9220303ff77c646299",
        "pilot-axis.jsonl": "9e9a9a8b8083c34e2ea7ab85750b84f11532d330eb3c4c3511cecb2572595ee1",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    manifest = json.loads((output / "selection-manifest.json").read_text())
    assert "additional_exclusions" not in manifest
    assert "original_excluded_prompt_count" not in manifest
    assert "additional_excluded_prompt_count" not in manifest
