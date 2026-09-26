"""The real-data checker must pass correct code and catch a wrong or incomplete report."""

import hashlib
import json
from dataclasses import asdict

import pytest

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import check_axis_ranking_change as checker

U = [f"https://www.a.example/{i}/" for i in range(10)]
PROMPTS = {  # keyword -> [(prompt, observed coordinate, ranking)]
    "kw-a": [("p0", 0.0, U[0:6]),
             ("p1", 0.2, [U[1], U[0], U[2], U[3], U[4], U[7]]),
             ("p2", 0.2, U[0:3]),                          # ties p1; too short for top 5
             ("p3", 0.5, [U[9], U[8]]),                    # too short for top 3 and 5
             ("p4", 0.8, []),                              # empty ranking
             ("p5", 1.0, ["http://a.example/0", *U[1:5]])],  # differs from p0 only by URL spelling
    "kw-b": [("q0", 0.1, U[0:5]), ("q1", 0.6, U[5:10]), ("q2", 0.9, [U[0], U[5], U[1], U[6], U[2]])],
}


def dataset(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="policy")
    writer = FinalDatasetWriter(root, writer_id="fixture")
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
    coordinates, pending = [], []
    for rank, (keyword, prompts) in enumerate(PROMPTS.items()):
        for number, (prompt, observed, ranking) in enumerate(prompts):
            coordinates.append({"candidate_id": prompt, "axis_bin": number,
                                "assigned_axis_1_0_1": number / 10, "observed_axis_1_percentile_0_1": observed})
            writer.append("keyword_memberships", {"prompt_id": prompt, "keyword_ids": [keyword],
                                                  "primary_keyword_id": keyword, "primary_priority_rank": rank},
                          transaction_id="membership-" + prompt)
            task_id = f"task-{prompt}-natural"
            identity = ClaimIdentity(task_id=task_id, model_id="qwen/model", model_revision="a" * 40,
                                     protocol="agentic-generator-shared-v1",
                                     request_sha256=hashlib.sha256(task_id.encode()).hexdigest())
            writer.append("task_definitions", {
                "task_id": task_id, "prompt_id": prompt, "model": "qwen38", "method": "Parallel-Expansion-v1",
                "engine": "duckduckgo", "condition": "natural", "claim_identity": asdict(identity),
                "dependency_fingerprints": [], "runnable_task": {"cell_id": task_id}}, transaction_id="definition-" + task_id)
            claim = ledger.claim(identity, owner_id="fixture").claim
            pending.append((claim, writer.append("generations", {"cell_id": task_id, "ranking": ranking},
                                                 transaction_id=task_id)))
    writer.seal()
    for claim, reference in pending:
        ledger.transition(claim, state="completed", record_references=[reference])
    registration = root / "local-only/population-registration-v1"
    registration.mkdir(parents=True)
    raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in coordinates)
    (registration / "population-selection-records.jsonl").write_bytes(raw)
    (registration / "manifest.json").write_text(json.dumps({"files": {"population-selection-records.jsonl": {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}))
    return root


def test_checker_passes_and_accounts_for_every_dropped_or_short_pair(tmp_path):
    result = checker.check(dataset(tmp_path), models=("qwen38",), stripes=4)
    assert result["status"] == "PASS", result["failures"]
    (group,) = result["groups"]
    assert group["keywords"] == 2 and group["prompt_pairs"] == 15 + 3
    assert group["empty_ranking_pairs"] == 5 and group["zero_gap_pairs"] == 1
    assert (group["top3_usable_pairs"], group["top3_short_pairs"]) == (5 + 3, 4)
    assert (group["top5_usable_pairs"], group["top5_short_pairs"]) == (3 + 3, 6)
    assert group["top3_usable_keywords"] == group["top5_usable_keywords"] == 2
    assert group["top3_string_only_membership_changes"] == 3  # p0, p1 and p2 against p5
    assert group["top3_full_set_distance"]["pairs"] == 8
    # The report counts short lists as excluded; empty-list and tied pairs never reach it.
    assert group["top3_full_set_distance"]["excluded_pairs"] == group["top3_short_pairs"] == 4
    data = result["data"]
    assert data["ranking_length_counts"] == {"0": 1, "2": 1, "3": 1, "5": 4, "6": 2}
    assert data["prompts_per_keyword_any_cell"] == {"n": 2, "min": 3, "median": 4.5, "max": 6}
    assert data["coordinates"]["rows"] == 9 and data["accounting"]["verified_completed_tasks"] == {"qwen38": 9}


@pytest.mark.parametrize("defect,kind", [("wrong_measure", "measure_mismatch"), ("drop_pair", "report_missing_pair")])
def test_checker_fails_when_the_report_code_is_wrong(tmp_path, monkeypatch, defect, kind):
    real = checker.compare_prompts

    def broken(rows, **kwargs):
        pairs = real(rows, **kwargs)
        target = next(p for p in pairs if p["coordinate_role"] == "observed_latent")
        if defect == "drop_pair":
            return [p for p in pairs if p is not target]
        target["top3_full_set_distance"] = 0.5 if target["top3_full_set_distance"] != 0.5 else 0.25
        return pairs

    monkeypatch.setattr(checker, "compare_prompts", broken)
    result = checker.check(dataset(tmp_path), models=("qwen38",), stripes=4)
    assert result["status"] == "FAIL"
    assert kind in {failure["kind"] for failure in result["failures"]}


def test_cli_writes_a_new_result_and_exits_nonzero_only_on_failure(tmp_path, capsys):
    root = dataset(tmp_path)
    output = tmp_path / "check.json"
    assert checker.main(["--dataset-root", str(root), "--models", "qwen38", "--stripes", "4",
                         "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    assert capsys.readouterr().out.startswith("CHECK=PASS")
    with pytest.raises(SystemExit):
        checker.main(["--dataset-root", str(root), "--models", "qwen38", "--stripes", "4", "--output", str(output)])


def test_independent_kendall_counts_only_shared_urls():
    assert checker.top_k_measures(list("abcde"), list("baxyz"), 3)["kendall_distance_common"] == 1.0
    assert checker.top_k_measures(list("abcde"), list("axyzb"), 5)["kendall_distance_common"] == 0.0
    assert checker.top_k_measures(list("abcde"), list("avwxy"), 5)["kendall_distance_common"] is None
    assert checker.top_k_measures(list("ab"), list("abc"), 3) is None
