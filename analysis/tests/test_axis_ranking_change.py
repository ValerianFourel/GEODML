from analysis.scripts.report_axis_ranking_change import compare_prompts, summarize


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
