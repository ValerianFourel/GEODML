"""The paper results bundle copies every input it finds, reads the known formats, and names what is missing."""

import json

import pytest

from analysis.scripts import paper_results as pr


def test_bundle_copies_inputs_reads_known_formats_and_reports_missing(tmp_path):
    feature = {"block": "A1", "beta_per_sd": 0.1, "odds_ratio_per_sd": 1.105, "ci95": [0.05, 0.15], "permutation_p": 0.02}
    stage = {"features": {"intent_alignment": feature, "intent_x_prompt": feature, "topic_similarity": {**feature, "block": "A2"},
                          "dfs_organic_count": {**feature, "block": "C1"}}}
    funnel = {"settings": {"split": "confirmation", "bootstrap": 200, "permutations": 200, "specs": ["main"], "exploratory": False},
              "git_commit": "abc", "models": {"main": {"llama4 · Parallel": {"R|U": stage, "K|P": stage}}},
              "negative_control_null": {"llama4 · Parallel": {"R|U": {"quantile": 0.1}, "K|P": {"quantile": 0.05}}},
              "confirmatory": {"P1 alignment at R|U": {"replicates": True, "strata": [{"stratum": "llama4 · Parallel", "estimate": 0.1, "ci95": [0.05, 0.15]}]}},
              "decomposition": {"llama4 · Parallel": {"intent_alignment": {"answers": 10, "log_rr": {"retrieval|U": 0.1, "reranker_candidate|R": 0.0,
                                                                                                        "shortlist|C": 0.2, "ranking|P": 0.05},
                                                                             "total_log_rr_K_given_U": 0.35, "ci95_total": [0.3, 0.4]}}}}
    (tmp_path / "funnel").mkdir()
    (tmp_path / "funnel" / "results.json").write_text(json.dumps(funnel))
    decisions = {"split": "confirmation", "bootstrap": 100, "permutations": 100, "git_commit": "abc",
                 "strata": {"llama4 · Parallel": {"counts": {"shown": 70, "kept": 28, "share_kept": 0.4},
                                                  "keep": {"pseudo_r2": 0.5, "block_shares": {"A2": 0.8, "slot": 0.1, "A1": 0.0}, "features": stage["features"]},
                                                  "order": {"skipped": "2 sets"}}}}
    (tmp_path / "dec").mkdir()
    (tmp_path / "dec" / "decisions.json").write_text(json.dumps(decisions))
    (tmp_path / "dec" / "manifest.json").write_text(json.dumps({"stage": "decisions", "git_commit": "abc", "counts": {"a": 1}}))
    out = tmp_path / "bundle"
    assert pr.main(["--input", f"funnel=dummy-label={tmp_path / 'funnel'}".split("=", 1)[1] if False else f"funnel={tmp_path / 'funnel'}",
                    "--input", f"decisions={tmp_path / 'dec'}", "--input", f"gemma={tmp_path / 'nowhere'}", "--output", str(out)]) == 0
    report = (out / "report.md").read_text()
    assert "## funnel" in report and "P1 alignment at R|U: replicates = True" in report and "+0.350 [+0.300, +0.400]" in report
    assert "## decisions" in report and "| llama4 · Parallel | 70 → 28 (40%) | keep | 0.500 | 80.0% | 10.0%" in report and "2 sets" in report
    assert "## gemma" in report and "**missing**" in report
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["files"]) == {"funnel/results.json", "decisions/decisions.json", "decisions/manifest.json"}
    assert (out / "funnel" / "results.json").exists() and manifest["inputs"]["gemma"]["found"] == []
    with pytest.raises(ValueError, match="overwrite"):
        pr.main(["--input", f"funnel={tmp_path / 'funnel'}", "--output", str(out)])
