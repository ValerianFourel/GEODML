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


def test_cited_from_generations_weights_ranks_dedupes_and_recovers_the_slope():
    rng = np.random.default_rng(2)
    pages = [{"urls": [f"u{i}"], "prompt_scale_percentile_0_1": i / 99} for i in range(100)]
    pages.append({"urls": ["u0"], "prompt_scale_percentile_0_1": 1.0})  # a URL with two texts takes their mean
    prompts = [{"candidate_id": f"p{j}", "axis_1_percentile_0_1": j / 199, "keyword": f"k{j % 20}"} for j in range(200)]
    rows = []
    for j in range(200):
        x = j / 199
        top = int(np.clip(round(30 + 40 * x + rng.normal(0, 3)), 1, 98))
        for engine in ("duckduckgo", "searxng"):
            rows.append(("llama4", {"prompt_id": f"p{j}", "engine": engine, "method": "M", "condition": "natural",
                                    "ranking": [f"u{top}", f"u{top + 1}", "unknown-url"]}))
    rows.append(rows[0])  # duplicate cell
    rows.append(("llama4", {"prompt_id": "missing", "engine": "duckduckgo", "method": "M", "condition": "natural", "ranking": ["u1"]}))
    out = figs.cited_from_generations(iter(rows), prompts, pages, bootstrap=20)
    assert out["counts"]["duplicates"] == 1 and out["counts"]["unknown_prompt"] == 1
    assert out["counts"]["ranked_urls_matched"] == 2 * 400
    slope = out["slopes_natural"]["llama4 · both engines"]
    assert abs(slope["slope_K_on_x"] - 40 / 99) < 0.05 and slope["ci95"][0] < slope["slope_K_on_x"] < slope["ci95"][1]
    single = {"curves": out["curves"]}
    import json, tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "r.json").write_text(json.dumps(single))
    assert set(out["curves"]["groups_by_method"]) == {"llama4 · M"}
    assert figs.main(["render", "--results", str(d / "r.json"), "--output-dir", str(d)]) == 0
    assert (d / "fig4-cited-vs-prompt-by-method.pdf").exists()
    assert figs.main(["render", "--results", str(d / "r.json"), "--facet", "engine", "--output-dir", str(d)]) == 0
    assert (d / "fig4-cited-vs-prompt.pdf").exists()


def test_funnel_figures_render_from_their_results_and_refuse_empty_input(tmp_path):
    def feature(block):
        return {"block": block, "beta_per_sd": 0.1, "odds_ratio_per_sd": 1.105, "ci95": [0.05, 0.15], "permutation_p": None}
    stage = {"features": {name: feature(block) for name, _, block in figs.FUNNEL_FEATURES}}
    strata = ["qwen38 · Parallel", "qwen38 · Reactive"]
    steps = {"retrieval|U": 0.1, "reranker_candidate|R": 0.0, "shortlist|C": 0.5, "ranking|P": 0.1}
    funnel = {"settings": {"split": "exploration", "bootstrap": 10},
              "models": {"main": {s: {st: stage for st in ("R|U", "R0|U", "P|C", "K|P")} for s in strata}},
              "decomposition": {s: {"topic_similarity": {"answers": 10, "log_rr": steps, "ci95_total": [0.5, 0.9]}} for s in strata}}
    terms = ("R_u", "P_u", "K_u", "R0_u", "R_offtopic", "P_offtopic", "K_offtopic")
    block = {"bin_lower": [0.0, 0.5], "bin_upper": [0.5, 1.0],
             "terms": {t: {"mean": [0.3, 0.4], "se_keyword_cluster": [0.01, 0.01], "n": [5, 5]} for t in terms}}
    explore = {"keywords": 3, "natural": 10, "curves": {**{s: block for s in strata}, "qwen38 · searxng": block}}
    (tmp_path / "funnel.json").write_text(json.dumps(funnel))
    (tmp_path / "explore.json").write_text(json.dumps(explore))
    out = tmp_path / "out"
    assert figs.main(["render", "--funnel-results", str(tmp_path / "funnel.json"), "--explore", str(tmp_path / "explore.json"),
                      "--output-dir", str(out)]) == 0
    for name in ("fig5-funnel-odds-ratios", "fig6-decomposition", "fig7-offtopic", "fig8-stage-intent"):
        assert (out / f"{name}.pdf").exists() and (out / f"{name}.png").exists()
    (tmp_path / "empty.json").write_text(json.dumps({"models": {"main": {}}, "decomposition": {}}))
    with pytest.raises(ValueError, match="no fitted strata"):
        figs.main(["render", "--funnel-results", str(tmp_path / "empty.json"), "--output-dir", str(tmp_path / "out2")])


def test_fig9_search_methods_renders_pdf_png_and_a_compiling_tikz_twin(tmp_path, monkeypatch):
    import shutil
    import subprocess
    assert figs.main(["render", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "fig9-search-methods.pdf").stat().st_size > 1000
    assert (tmp_path / "fig9-search-methods.png").stat().st_size > 1000
    tex = (tmp_path / "fig9-search-methods-tikz.tex").read_text(encoding="utf-8")
    for needed in ("(a) Parallel Expansion", "(b) Reactive Loop", "against the PROMPT", "against THIS QUERY",
                   "BGE cross-encoder", "23,893 snippet rows", r"\end{tikzpicture}"):
        assert needed in tex
    layout = figs.fig9_layout()
    assert min(b["size"] for b in layout["boxes"]) >= figs.FIG9_MIN_FONT
    assert min(s["size"] for s in layout["texts"]) >= figs.FIG9_MIN_FONT
    # The renderer itself refuses text below the readability floor.
    monkeypatch.setattr(figs, "FIG9_MIN_FONT", 9.0)
    with pytest.raises(ValueError, match="below"):
        figs.fig9_search_methods(figs._setup(), tmp_path / "small")
    if shutil.which("pdflatex"):
        done = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig9-search-methods-tikz.tex"],
                              cwd=tmp_path, capture_output=True, text=True, timeout=180)
        assert done.returncode == 0, done.stdout[-2000:]
