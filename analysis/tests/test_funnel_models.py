"""Funnel estimators: Chamberlain conditional logit, block designs with shared draws, exact decomposition."""

from itertools import combinations
import json

import numpy as np
import pytest

from analysis.interpretability.pipeline import funnel_models as fm
from analysis.interpretability.pipeline import geo_drivers as geo
from analysis.interpretability.pipeline import intent_stages as stages
from analysis.interpretability.pipeline.page_readiness_ordering import fit_choice_model


def brute_loglik(theta, X, groups, y):
    total = 0.0
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        v = X[rows] @ theta
        m = int(y[rows].sum())
        norm = np.log(sum(np.exp(v[list(s)].sum()) for s in combinations(range(len(rows)), m)))
        total += v[y[rows] == 1].sum() - norm
    return total


def test_chamberlain_likelihood_and_gradient_match_brute_force():
    rng = np.random.default_rng(0)
    sizes = [3, 5, 6, 4, 6]
    answer = np.repeat(np.arange(len(sizes)), sizes)
    X = rng.normal(size=(len(answer), 3)) * 2
    y = np.zeros(len(answer), int)
    for g, n in enumerate(sizes):
        y[np.flatnonzero(answer == g)[rng.choice(n, rng.integers(1, n), replace=False)]] = 1
    y = np.r_[y, [0, 0, 1, 1]]
    answer = np.r_[answer, [9, 9, 10, 10]]  # uninformative: none / all admitted
    X = np.r_[X, rng.normal(size=(4, 3))]
    data, keep = fm.admission_data(answer, y)
    assert data.groups == 5 and data.dropped == {"answers_none_admitted": 1, "answers_all_admitted": 1}
    theta = np.array([0.7, -1.2, 0.3])
    loss, grad, pi = fm.admission_loss(theta, X[keep], data)
    assert loss == pytest.approx(-brute_loglik(theta, X[keep], data.group, data.admitted) / data.groups, rel=1e-10)
    assert np.allclose(np.bincount(data.group, pi), data.m)  # inclusion probabilities sum to the admitted count
    eps = 1e-6
    numeric = [(fm.admission_loss(theta + eps * e, X[keep], data)[0] - fm.admission_loss(theta - eps * e, X[keep], data)[0]) / (2 * eps)
               for e in np.eye(3)]
    assert np.allclose(grad, numeric, atol=1e-7)


def planted_admission(rng, answers=1500, items=12, theta=(1.0, -0.6)):
    answer = np.repeat(np.arange(answers), items)
    X = rng.normal(size=(len(answer), 2))
    alpha = rng.normal(0, 2.0, answers)[answer]  # answer intercepts the conditional logit removes
    p = 1 / (1 + np.exp(-(alpha + X @ np.asarray(theta))))
    return answer, X, (rng.random(len(p)) < p).astype(int)


def test_chamberlain_recovers_coefficients_that_a_pooled_logit_misses():
    rng = np.random.default_rng(1)
    answer, X, y = planted_admission(rng)
    data, keep = fm.admission_data(answer, y)
    fit = fm.fit_admission(X[keep], data, ridge=0)
    assert np.allclose(fit.x, [1.0, -0.6], atol=0.08)


def gumbel_topk(utility, sets, k):
    """Indices chosen in order by a Plackett–Luce draw (Gumbel max trick) within each set."""
    rng = np.random.default_rng(3)
    order = {}
    for s in np.unique(sets):
        rows = np.flatnonzero(sets == s)
        noisy = utility[rows] + rng.gumbel(size=len(rows))
        order[s] = rows[np.argsort(-noisy)[:k]]
    return order


def choice_rows_from(order, answer_of_item, positions=None):
    """Top-k Plackett–Luce choice sets: each pick among the items not picked yet."""
    rows, sets, chosen, pos = [], [], [], []
    count = 0
    for s, picked in order.items():
        pool = list(np.flatnonzero(answer_of_item == s))
        for item in picked:
            for alt in pool:
                rows.append(alt); sets.append(count); chosen.append(alt == item)
                pos.append(0 if positions is None else positions[alt])
            pool.remove(item)
            count += 1

    class Rows:
        pass
    r = Rows()
    r.z = np.zeros(len(rows)); r.set = np.asarray(sets); r.chosen = np.asarray(chosen); r.sets = count
    r.position = np.asarray(pos, np.int64); r.levels = int(r.position.max()) + 1
    return np.asarray(rows), r


def test_invisible_feature_is_null_when_visible_text_is_controlled_and_not_otherwise():
    rng = np.random.default_rng(4)
    answers, items = 2500, 8
    answer = np.repeat(np.arange(answers), items)
    text = rng.normal(size=len(answer))
    brand = rng.normal(size=len(answer))
    hidden = 0.6 * text + 0.3 * brand + 0.7 * rng.normal(size=len(answer))  # page-body feature nobody sees
    order = gumbel_topk(1.0 * text + 0.7 * brand, answer, 3)                # the generator uses text and brand
    rows, data = choice_rows_from(order, answer)
    full = fit_choice_model(np.column_stack([text[rows], brand[rows], hidden[rows]]), data)
    assert abs(full.x[0] - 1.0) < 0.1 and abs(full.x[1] - 0.7) < 0.1 and abs(full.x[2]) < 0.08
    no_text = fit_choice_model(np.column_stack([brand[rows], hidden[rows]]), data)
    assert no_text.x[1] > 0.3  # without the visible text, the hidden feature absorbs it (the negative-control logic)


def _toy_choice_stage():
    rng = np.random.default_rng(5)
    keywords, prompts_per = 40, 6
    answers = keywords * prompts_per
    answer_keyword = np.repeat(np.arange(keywords), prompts_per)
    answer_x = rng.uniform(0, 1, answers)
    items = 8
    answer = np.repeat(np.arange(answers), items)
    u = rng.uniform(0, 1, len(answer))
    text = rng.normal(size=len(answer))
    alignment = -np.abs(u - answer_x[answer])
    order = gumbel_topk(2.5 * alignment + 0.8 * text, answer, 3)
    rows, data = choice_rows_from(order, answer)
    design = fm.Design([fm.Feature("text", "A3"), fm.Feature("alignment", "A1", "alignment", "u")],
                       {"text": text[rows], "u": u[rows]})
    row_answer = answer[rows]
    design.fit_stats(answer_x[row_answer])
    set_answer = np.asarray([answer[rows][data.set == s][0] for s in range(data.sets)])
    stage = fm.Stage("choice", design, data, row_answer, set_answer)
    draws = stages.keyword_draws(keywords, 12, 7)
    prompt_x = answer_x  # one answer per prompt here
    shuffles = stages.shuffle_draws(prompt_x, answer_keyword, 19, 8)
    return stage, answer_x, answer_keyword, draws, shuffles


def test_estimate_blocks_reports_odds_ratios_intervals_and_x_permutations():
    stage, answer_x, answer_keyword, draws, shuffles = _toy_choice_stage()
    out = fm.estimate_blocks(stage, answer_x=answer_x, answer_keyword=answer_keyword, draws=draws, shuffles=shuffles)
    f = out["features"]
    assert f["alignment"]["beta_per_sd"] > 0.3 and f["alignment"]["permutation_p"] == pytest.approx(1 / 20)
    assert f["text"]["permutation_p"] is None and f["text"]["ci95"][0] < f["text"]["beta_per_sd"] < f["text"]["ci95"][1]
    assert set(out["blocks"]) == {"A3", "A1"} and abs(sum(b["fit_share"] for b in out["blocks"].values()) - 1) < 1e-9
    assert len(out["replicates"]) == 12 and out["failed_replicates"] == 0
    pooled = fm.estimate_blocks(stage, answer_x=answer_x, answer_keyword=answer_keyword, draws=draws, shuffles=shuffles, workers=2)
    assert pooled == out  # forked workers (drop-block refits, bootstrap on the shared matrix) change nothing


def test_design_standardises_once_and_rebuilds_x_terms():
    design = fm.Design([fm.Feature("z", "A1"), fm.Feature("zx", "A1", "interaction", "z"), fm.Feature("al", "A1", "alignment", "u")],
                       {"z": [1.0, 2.0, 3.0], "u": [0.1, 0.5, 0.9]})
    x = np.array([0.2, 0.5, 0.8])
    design.fit_stats(x)
    a = design.matrix(x)
    b = design.matrix(np.array([0.8, 0.5, 0.2]))
    assert np.allclose(a[:, 0], b[:, 0]) and not np.allclose(a[:, 2], b[:, 2])
    assert design.matrix(x, drop_block="A1").shape == (3, 0) and design.x_dependent == {"zx", "al"}


def test_funnel_decomposition_is_exact_and_refuses_non_nested_stages():
    rng = np.random.default_rng(9)
    n_answers, items = 300, 10
    answer = np.repeat(np.arange(n_answers), items)
    r = rng.random(len(answer)) < 0.6
    c = r & (rng.random(len(answer)) < 0.9)
    p = c & (rng.random(len(answer)) < 0.5)
    k = p & (rng.random(len(answer)) < 0.6)
    group = rng.choice([1, -1, 0], len(answer))
    kw = rng.integers(0, 20, n_answers)
    draws = stages.keyword_draws(20, 25, 1)
    out = fm.rr_decomposition(answer, {"retrieved": r, "scored": c, "presented": p, "ranked": k}, group, kw, draws)
    # the stage logs add up to the overall log risk ratio of reaching K from U
    hi, lo = group == 1, group == -1
    both = np.bincount(answer[hi], minlength=n_answers) * np.bincount(answer[lo], minlength=n_answers) > 0
    with np.errstate(divide="ignore"):  # answers without a group are masked out below
        w_hi = 1 / np.bincount(answer[hi], minlength=n_answers)[answer]
        w_lo = 1 / np.bincount(answer[lo], minlength=n_answers)[answer]
    sel_hi, sel_lo = hi & both[answer], lo & both[answer]
    direct = np.log(np.sum(w_hi[sel_hi] * k[sel_hi]) / np.sum(w_hi[sel_hi])) - np.log(np.sum(w_lo[sel_lo] * k[sel_lo]) / np.sum(w_lo[sel_lo]))
    assert out["total_log_rr_K_given_U"] == pytest.approx(direct, abs=1e-12)
    assert set(out["ci95"]) == set(out["log_rr"]) and abs(sum(out["share"].values()) - 1) < 1e-9
    with pytest.raises(ValueError, match="nested"):
        fm.rr_decomposition(answer, {"retrieved": r, "scored": c, "presented": p | ~r, "ranked": k}, group, kw)


def test_contrast_and_empirical_null():
    a = {"features": {"brand": {"beta_per_sd": 0.7, "block": "C1"}, "h": {"beta_per_sd": 0.02, "block": "C2"},
                      "g": {"beta_per_sd": -0.04, "block": "C2"}}, "replicates": [[0.7, 0.0, 0.0], [0.8, 0.0, 0.0], [0.6, 0, 0]],
         "replicate_columns": ["brand", "h", "g"]}
    b = {"features": {"brand": {"beta_per_sd": 0.05, "block": "C1"}}, "replicates": [[0.05], [0.1], [0.0]], "replicate_columns": ["brand"]}
    c = fm.contrast_from_replicates(a, b, "brand")
    assert c["estimate"] == pytest.approx(0.65) and c["ci95"][0] > 0.5
    assert fm.empirical_null(a, "C2")["features"] == 2
    with pytest.raises(ValueError, match="replicate_columns"):
        fm.contrast_from_replicates({k: v for k, v in a.items() if k != "replicate_columns"}, b, "brand")


def test_contrasts_survive_a_sorted_json_round_trip():
    """The analysis cache writes JSON with sorted keys; replicate columns must still pair by design order."""
    from analysis.scripts import page_readiness_ordering as readiness
    stage, answer_x, answer_keyword, draws, shuffles = _toy_choice_stage()
    out = fm.estimate_blocks(stage, answer_x=answer_x, answer_keyword=answer_keyword, draws=draws, shuffles=[])
    assert list(out["features"]) == ["text", "alignment"] and sorted(out["features"]) != list(out["features"])
    other = json.loads(json.dumps(out))
    for f in other["features"].values():
        f["beta_per_sd"] = 0.0
    other["replicates"] = [[0.0, 0.0] for _ in other["replicates"]]
    direct = fm.contrast_from_replicates(out, other, "alignment")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        readiness.write_json(Path(d) / "a.json", out)
        loaded = json.loads((Path(d) / "a.json").read_text())
    assert list(loaded["features"]) == ["alignment", "text"]  # sorted by the writer
    assert fm.contrast_from_replicates(loaded, other, "alignment") == direct
    assert direct["ci95"][0] < direct["estimate"] < direct["ci95"][1]


@pytest.mark.parametrize("workers", [1, 2])
def test_a_task_stopped_at_the_deadline_resumes_from_its_checkpoint(tmp_path, monkeypatch, workers):
    stage, answer_x, answer_keyword, draws, shuffles = _toy_choice_stage()
    args = dict(answer_x=answer_x, answer_keyword=answer_keyword, draws=draws, shuffles=shuffles, workers=workers)
    reference = fm.estimate_blocks(stage, **args)
    clock = iter(range(10 ** 6))
    monkeypatch.setattr(fm.time, "monotonic", lambda: next(clock))  # each deadline check advances the clock by one
    parts = tmp_path / "task.parts.jsonl"
    with pytest.raises(fm.DeadlineReached):
        fm.estimate_blocks(stage, parts=parts, deadline=6, **args)
    saved = [json.loads(line)["key"] for line in parts.read_text().splitlines()]
    assert saved[0] == "full-0" and 1 < len(saved) < 1 + 2 + len(draws) + len(shuffles)
    with open(parts, "a") as stream:
        stream.write('{"key": "bootstrap-9", "val')  # a torn line from a killed writer is ignored
    monkeypatch.setattr(fm, "_fit", _counting(fm._fit, calls := []))
    resumed = fm.estimate_blocks(stage, parts=parts, **args)
    assert resumed == reference
    if workers == 1:  # the full fit and the saved units are not refitted
        assert len(calls) == 2 + len(draws) + len(shuffles) - (len(saved) - 1)


def _counting(function, calls):
    def wrapped(*a, **k):
        calls.append(1)
        return function(*a, **k)
    return wrapped
