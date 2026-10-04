"""Generator-output Markdown report on synthetic sealed datasets (no scientific result)."""
import csv
import json

import pytest

from analysis.scripts import report_generator_outputs as report
from analysis.tests.test_source_importance_pipeline import dataset


def axis_map(tmp_path, position=0.4):
    path = tmp_path / "final-axis-map.jsonl"
    path.write_text(json.dumps({"candidate_id": "q1", report.AXIS: position}) + "\n")
    return path


def test_report_counts_verified_cells_and_matches_pairs_across_conditions_and_models(tmp_path):
    roots = {model: dataset(tmp_path / model, model=model) for model in ("qwen38", "llama4")}
    output = tmp_path / "report"
    assert report.main([*(f"--dataset={m}={r}" for m, r in roots.items()), "--axis-map", str(axis_map(tmp_path)),
                        "--output-dir", str(output), "--permutations", "20"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scientific_result"] is False
    for model in roots:
        entry = summary["inventory"][model]
        assert entry["registered_tasks"] == entry["verified_cells"] == 3
        assert entry["latest_states"] == {"completed": 3}
        assert entry["expected_cells"] == 12 and entry["prompts_without_axis"] == 0
    pairs = list(csv.DictReader((output / "matched-pairs.csv").open()))
    by_comparison = {}
    for row in pairs:
        by_comparison.setdefault(row["comparison"], []).append(row)
    # Fixture: natural and shuffled rankings agree; both generators wrote identical rankings.
    assert len(by_comparison["natural vs shuffled evidence order"]) == 2
    assert len(by_comparison["llama4 vs qwen38 on the same cell"]) == 3
    assert {float(r["top1_change"]) for r in pairs} == {0.0}
    descriptives = list(csv.DictReader((output / "descriptives.csv").open()))
    ablated = next(r for r in descriptives if r["model"] == "qwen38" and r["condition"] == "ablated")
    assert float(ablated["mean_ranking_length"]) == 1.0
    assert float(ablated["target_selected_if_seen"]) == 0.0  # target removed, never ranked
    natural = next(r for r in descriptives if r["model"] == "qwen38" and r["condition"] == "natural")
    assert float(natural["target_mrr_if_seen"]) == 0.5  # target A ranked second
    text = (output / "report.md").read_text()
    assert "not a randomized treatment" in text and "## 4. Along the information-seeking" in text
    with pytest.raises(ValueError, match="already exists"):
        report.build(roots, output)


def test_unfinished_tasks_are_counted_but_not_analysed(tmp_path):
    from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
    from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
    from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter
    from dataclasses import asdict
    root = dataset(tmp_path, model="qwen38")
    identity = ClaimIdentity(task_id="cell-extra", model_id="m", model_revision="a" * 40, protocol="p",
                             request_sha256="b" * 64)
    writer = FinalDatasetWriter(root, writer_id="extra")
    writer.append("task_definitions", {"task_id": "cell-extra", "prompt_id": "q1", "model": "qwen38",
                                       "stage": "generation", "method": "M", "engine": "ddg",
                                       "condition": "natural", "claim_identity": asdict(identity)},
                  transaction_id="extra")
    writer.seal()
    StripedTaskLedger(root / "control/task-ledger", stripe_count=256).claim(identity, owner_id="fixture")
    cells, entry = report.load_cells(root, "qwen38")
    assert entry["registered_tasks"] == 4 and entry["verified_cells"] == len(cells) == 3
    assert sum(entry["latest_states"].values()) == 4 and entry["latest_states"]["completed"] == 3


def test_axis_association_uses_prompt_means_and_is_reproducible():
    cells = []
    for index in range(12):
        for repeat in range(3):  # several cells per prompt must not inflate n
            cells.append({"model": "llama4", "prompt_id": f"p{index}", "keyword_id": f"k{index % 3}",
                          report.AXIS: index / 11, "ranking_length": index + repeat})
    first = report.associations(cells, ("ranking_length",), ("model",), scope="all", permutations=50, seed=1)
    second = report.associations(cells, ("ranking_length",), ("model",), scope="all", permutations=50, seed=1)
    assert first == second
    assert first[0]["n"] == 12 and first[0]["spearman_rho"] == pytest.approx(1.0)
    assert first[0]["blocked_permutation_p_two_sided"] < 0.1


def test_axis_map_rejects_out_of_range_positions(tmp_path):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        report.load_axis(axis_map(tmp_path, position=1.5))
