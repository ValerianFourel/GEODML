"""ACL figures: example picks stay inside their bands; every figure renders from its inputs."""
import json

import numpy as np
import pytest

from analysis.interpretability.pipeline import intent_stages as stages
from analysis.scripts import acl_figures as figs


def population(n_keywords=5, per=30, seed=1):
    rng = np.random.default_rng(seed)
    axis, prompts = [], []
    for k in range(n_keywords):
        name = "robinhood vs etrade" if k == 0 else f"kw-{k}"
        for j in range(per):
            cid = f"c-{k}-{j}"
            axis.append({"candidate_id": cid, "axis_1_percentile_0_1": float(rng.uniform())})
            prompts.append({"candidate_id": cid, "keyword": name, "question": f"question {k} {j}"})
    # a keyword that never reaches the action end is never chosen
    axis.append({"candidate_id": "low", "axis_1_percentile_0_1": 0.01})
    prompts.append({"candidate_id": "low", "keyword": "only-low", "question": "q"})
    return axis, prompts


def test_axis_examples_pick_one_prompt_per_band_and_repeat_with_the_seed():
    axis, prompts = population()
    first = figs.axis_examples(axis, prompts, seed=4)
    assert first == figs.axis_examples(axis, prompts, seed=4)
    names = [k["keyword"] for k in first["keywords"]]
    assert names[0] == "robinhood vs etrade" and len(set(names)) == 2 and "only-low" not in names
    for keyword in first["keywords"]:
        for band, (lo, hi) in figs.BANDS.items():
            assert lo <= keyword["picks"][band]["position"] <= hi
    with pytest.raises(ValueError, match="every band"):
        figs.axis_examples(axis, prompts, keywords=["only-low"])


def test_render_writes_all_four_figures(tmp_path):
    axis, prompts = population()
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps(figs.axis_examples(axis, prompts)))
    rng = np.random.default_rng(0)
    n = 3000
    x, keyword = rng.uniform(0, 1, n), rng.integers(0, 30, n)
    values = {t: 0.3 + 0.2 * x + rng.normal(0, 0.05, n) for t in ("R", "C", "P", "K")}
    block = stages.stage_curves(x, keyword, values, np.ones(n, bool))
    groups = {f"{m} · {e}": {"natural": {"u": block, "z": block}} for m in ("llama4", "qwen38")
              for e in ("duckduckgo", "searxng", "both engines")}
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"curves": {"bins": 20, "groups": groups}}))
    assert figs.main(["render", "--axis-examples", str(examples), "--results", str(results),
                      "--output-dir", str(tmp_path / "out")]) == 0
    for name in ("fig1-prompt-axis", "fig2-mini-internet", "fig3-pipeline", "fig4-cited-vs-prompt"):
        assert (tmp_path / "out" / f"{name}.pdf").stat().st_size > 1000
    results.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="no curves"):
        figs.main(["render", "--results", str(results), "--output-dir", str(tmp_path / "out2")])
