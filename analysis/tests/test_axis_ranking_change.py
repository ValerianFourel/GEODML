import pytest

from analysis.scripts.report_axis_ranking_change import compare_prompts, summarize
from analysis.scripts.summarize_axis_ranking_change import format_top_k_summary


def test_cross_prompt_change_and_stable_condition():
    rows = []
    for condition in ("natural", "shuffled"):
        for i, ranking in enumerate((["a", "b", "c"], ["a", "c", "b"], ["c", "b", "a"])):
            rows.append(dict(model="qwen38", method="m", engine="e", condition=condition,
                             keyword_id="k", prompt_id=str(i),
                             assigned_axis_1_0_1=i / 2,
                             observed_axis_1_percentile_0_1=i / 2,
                             ranking=ranking if condition == "natural" else ["a", "b", "c"]))
    pairs = compare_prompts(rows)
    endpoint = next(r for r in pairs if r["condition"] == "natural"
                    and r["coordinate_role"] == "observed_latent" and r["axis_gap"] == 1)
    assert endpoint["kendall_distance_common"] == 1
    assert endpoint["top1_change"] == 1
    assert endpoint["url_set_distance"] == 0
    summary, _ = summarize(pairs)
    changing = next(r for r in summary if r["condition"] == "natural"
                    and r["coordinate_role"] == "observed_latent" and r["metric"] == "kendall_distance_common")
    assert changing["rho_axis_gap_vs_ranking_distance"] > 0.8
    stable = next(r for r in summary if r["condition"] == "shuffled"
                  and r["coordinate_role"] == "observed_latent" and r["metric"] == "kendall_distance_common")
    assert stable["mean_distance"] == 0
    assert stable["rho_axis_gap_vs_ranking_distance"] is None
    # A separate keyword must never produce a cross-keyword pair.
    other = dict(rows[0], keyword_id="other", prompt_id="other", ranking=["x", "y"])
    assert compare_prompts([*rows, other]) == pairs


def test_top_three_and_five_distinguish_membership_order_and_short_lists():
    rows = []
    for i, ranking in enumerate((
        ["a", "b", "c", "d", "e"],
        ["b", "a", "c", "d", "f"],
        ["a", "b", "c"],
        ["a", "b"],
    )):
        rows.append(dict(model="qwen38", method="Parallel", engine="engine", condition="natural",
                         keyword_id="k", prompt_id=str(i), assigned_axis_1_0_1=i / 3,
                         observed_axis_1_percentile_0_1=i / 3, ranking=ranking))
    pairs = compare_prompts(rows)
    complete = next(r for r in pairs if r["low_prompt"] == "0" and r["high_prompt"] == "1")
    assert complete["top3_full_set_change"] == 0
    assert complete["top3_full_ordered_change"] == 1
    assert complete["top3_full_set_distance"] == 0
    assert complete["top3_full_kendall_distance_common"] == pytest.approx(1 / 3)
    assert complete["top5_full_set_change"] == 1
    assert complete["top5_full_set_distance"] == pytest.approx(1 / 3)
    assert complete["top5_full_kendall_distance_common"] == pytest.approx(1 / 6)
    summary, _ = summarize(pairs)
    def metric(name):
        return next(r for r in summary if r["coordinate_role"] == "observed_latent" and r["metric"] == name)
    assert metric("top3_full_set_distance")["pairs"] == 3
    assert metric("top3_full_set_distance")["excluded_pairs"] == 3
    assert metric("top5_full_set_distance")["pairs"] == 1
    assert metric("top5_full_set_distance")["excluded_pairs"] == 5
    assert metric("top3_full_set_distance")["mean_distance"] == 0
    assert metric("top5_full_set_distance")["mean_distance"] == pytest.approx(1 / 3)
    text = format_top_k_summary(dict(summaries=summary, observations=4,
                                    accounting={"verified_completed_tasks": {"qwen38": 4}}))
    assert "TOP 3 | full lists only" in text
    assert "TOP 5 | full lists only" in text
    assert "Top1" not in text
    assert "0.333" in text


def test_top_k_order_is_missing_for_disjoint_results():
    rows = [dict(model="qwen38", method="m", engine="e", condition="natural", keyword_id="k",
                 prompt_id=str(i), assigned_axis_1_0_1=i, observed_axis_1_percentile_0_1=i,
                 ranking=ranking) for i, ranking in enumerate((list("abcde"), list("vwxyz")))]
    pair = compare_prompts(rows)[0]
    for k in (3, 5):
        assert pair[f"top{k}_full_set_distance"] == 1
        assert pair[f"top{k}_full_kendall_distance_common"] is None
