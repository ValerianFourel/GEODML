import numpy as np
import pytest

from analysis.interpretability.pipeline import intent_stages as stages
from analysis.interpretability.pipeline.geo_drivers import within_keyword_slope
from analysis.steelman import chain as ch
from analysis.steelman import tables as T


def tiny_items():
    # answer 0: 4 retrieved rows, 3 scored, 3 shown (slots 0..2), cites slot 2 then slot 0
    # answer 1: 3 retrieved, 2 scored, 2 shown, cites both in shown order
    offsets = np.array([0, 4, 7])
    row = np.array([0, 1, 2, 3, 4, 5, 6])
    scored = np.array([1, 1, 1, 0, 1, 1, 0])
    presented = np.array([0, 1, 2, -1, 0, 1, -1])
    ranked = np.array([1, -1, 0, -1, 0, 1, -1])
    u = np.array([0.1, 0.2, 0.9, 0.5, 0.3, 0.6, 0.0])
    return offsets, row, scored, presented, ranked, u


def test_stage_values_by_hand():
    v = T.stage_values(*tiny_items())
    w = 1 / np.log2(np.arange(2) + 2)
    assert v["R"][0] == pytest.approx((0.1 + 0.2 + 0.9 + 0.5) / 4)
    assert v["C"][0] == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    assert v["P"][1] == pytest.approx(0.45)
    assert v["K"][0] == pytest.approx((w[0] * 0.9 + w[1] * 0.1) / w.sum())   # cited order: slot 2 first
    assert v["I"][0] == pytest.approx((w[0] * 0.1 + w[1] * 0.2) / w.sum())   # first L=2 shown, shown order
    assert v["I"][1] == pytest.approx(v["K"][1])                              # identity ranking: I == K
    assert list(v["L"]) == [2, 2] and list(v["n_shown"]) == [3, 2]


def test_replay_values():
    rows = np.array([[0, 1], [2, 3]])
    u = np.array([0.0, 1.0, 0.5, 0.5])
    got = T.replay_values(["a", "b", "c"], ["e", "e", "e"], [{"prompt_id": "a", "engine": "e"}, {"prompt_id": "b", "engine": "e"}], rows, u)
    assert got[0] == 0.5 and got[1] == 0.5 and np.isnan(got[2])


def world(seed=0, n_kw=30, per=40):
    rng = np.random.default_rng(seed)
    keyword = np.repeat(np.arange(n_kw), per)
    prompt = np.arange(len(keyword))
    x = rng.random(len(keyword))
    vals = {}
    base = rng.normal(size=n_kw)[keyword]
    level = 0.0
    for name, step in zip(T.CHAIN, (0.01, 0.02, 0.0, 0.02, 0.004, 0.006)):
        level += step
        vals[name] = base + level * x + rng.normal(scale=0.05, size=len(x))
    return vals, x, keyword, prompt


def test_ols_reduces_to_existing_slopes():
    vals, x, k, _ = world()
    assert ch.within_keyword_ols(x[:, None], vals["K"], k)[0] == pytest.approx(within_keyword_slope(x, vals["K"], k))
    direct, _ = stages.mediation(vals["K"], vals["P"], x, k)
    assert ch.within_keyword_ols(np.column_stack([x, vals["P"]]), vals["K"], k)[0] == pytest.approx(direct)


@pytest.mark.parametrize("controls", [False, True])
def test_chain_increments_sum_exactly(controls):
    vals, x, k, p = world()
    c = np.random.default_rng(1).normal(size=(len(x), 2)) if controls else None
    draws = stages.keyword_draws(int(k.max()) + 1, 5, 3)
    res = ch.chain(vals, x, k, p, draws=draws, shuffles=[], controls=c)
    inc = {t: v["estimate"] for t, v in res["increments"].items()}
    total = sum(inc[t] for t in ("prompt_words_R0", "query_rewriting", "dedup_condition", "reranker", "shown_order", "generator"))
    assert total == pytest.approx(res["slopes"]["K"]["estimate"], abs=1e-12)
    assert inc["generator_vs_random"] == pytest.approx(inc["shown_order"] + inc["generator"], abs=1e-12)
    shares = sum(res["shares"][t]["estimate"] for t in ("prompt_words_R0", "query_rewriting", "dedup_condition", "reranker", "shown_order", "generator"))
    assert shares == pytest.approx(1.0, abs=1e-12)


def test_chain_recovers_planted_slopes_and_null():
    vals, x, k, p = world(n_kw=60, per=60)
    prompt_x, prompt_kw = x.copy(), k.copy()
    shuffles = stages.shuffle_draws(prompt_x, prompt_kw, 49, 2)
    res = ch.chain(vals, x, k, p, draws=stages.keyword_draws(60, 50, 1), shuffles=shuffles)
    assert res["slopes"]["K"]["estimate"] == pytest.approx(0.06, abs=0.01)
    assert abs(res["increments"]["dedup_condition"]["estimate"]) < 0.015      # planted 0
    assert res["increments"]["reranker"]["ci95"][0] > 0                         # planted 0.02
    assert res["slopes"]["K"]["permutation_p"] <= 0.02


def test_loko_and_noise_floor_shapes():
    vals, x, k, p = world()
    lo = ch.leave_one_keyword_out(vals, x, k)
    assert lo["generator"]["min"] <= lo["generator"]["max"]
    cell = np.asarray([f"{a}|{i % 5}" for i, a in enumerate(k)])
    nf = ch.noise_floor(vals, x, k, cell, stages.keyword_draws(int(k.max()) + 1, 5, 1))
    assert nf["cells_with_two_or_more"] > 0 and set(nf["terms"]) == set(T.CHAIN)
