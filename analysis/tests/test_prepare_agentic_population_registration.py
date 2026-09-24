from __future__ import annotations

import gzip
import hashlib
import json

from analysis.scripts.prepare_agentic_population_registration import prepare


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_prepares_full_population_records_and_keyword_priority(tmp_path):
    prompts = []
    axes = []
    for number, (keyword_id, keyword) in enumerate((("z", "Zulu"), ("a", "Alpha"))):
        question = f"Question {number}"
        digest = hashlib.sha256(question.encode()).hexdigest()
        prompts.append({
            "candidate_id": f"p{number}", "question": question,
            "question_sha256": digest, "keyword": keyword,
            "keyword_id": keyword_id, "target_normalized_axis_1": number,
        })
        axes.append({
            "candidate_id": f"p{number}", "text_sha256": digest,
            "axis_1_percentile_0_1": number,
        })
    prompt_path = _jsonl(tmp_path / "prompts.jsonl", prompts)
    axis_path = _jsonl(tmp_path / "axes.jsonl", axes)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"sources": {
        "prompts": {"path": prompt_path.name, "rows": 2,
                    "sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()},
        "axis_map": {"path": axis_path.name, "rows": 2,
                     "sha256": hashlib.sha256(axis_path.read_bytes()).hexdigest()},
    }}))
    result = prepare(manifest, tmp_path / "out", axis_bins=20)
    assert result["population_count"] == 2
    priority = json.loads((tmp_path / "out/keyword-priority.json").read_text())
    assert [(row["keyword_id"], row["priority_rank"]) for row in priority["keywords"]] == [
        ("a", 0), ("z", 1)
    ]
    records = [json.loads(line) for line in (
        tmp_path / "out/population-selection-records.jsonl"
    ).read_text().splitlines()]
    assert {row["candidate_id"] for row in records} == {"p0", "p1"}


def test_filters_only_explicit_recovery_transfer_eligible_ids(tmp_path):
    prompts = []
    axes = []
    for number in range(2):
        question = f"Question {number}"
        digest = hashlib.sha256(question.encode()).hexdigest()
        prompts.append({
            "candidate_id": f"p{number}", "question": question,
            "question_sha256": digest, "keyword": f"Keyword {number}",
            "keyword_id": f"k{number}", "target_normalized_axis_1": number,
        })
        axes.append({"candidate_id": f"p{number}", "text_sha256": digest,
                     "axis_1_percentile_0_1": number})
    prompt_path = _jsonl(tmp_path / "prompts.jsonl", prompts)
    axis_path = _jsonl(tmp_path / "axes.jsonl", axes)
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({"sources": {
        "prompts": {"path": prompt_path.name, "rows": 2,
                    "sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()},
        "axis_map": {"path": axis_path.name, "rows": 2,
                     "sha256": hashlib.sha256(axis_path.read_bytes()).hexdigest()},
    }}))
    eligible = tmp_path / "eligible.jsonl.gz"
    with gzip.open(eligible, "wt") as stream:
        stream.write(json.dumps({"prompt_id": "p0"}) + "\n")
    result = prepare(
        manifest, tmp_path / "out", axis_bins=20,
        eligible_prompts_table=eligible,
    )
    assert result["registered_prompt_count"] == 1
    assert result["excluded_prompt_count"] == 1
    assert json.loads((tmp_path / "out/excluded-prompts.jsonl").read_text()) == {
        "prompt_id": "p1", "reason": "prompt_transfer_provenance_unverified"
    }
