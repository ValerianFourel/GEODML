"""Local exploratory review helpers: per-keyword slopes and shrinkage, query lexicon, per-answer metrics."""

import numpy as np
import pytest

from analysis.scripts import funnel_explore as explore
from analysis.scripts import funnel_keywords as kw
from analysis.scripts import funnel_review as review


def test_keyword_slopes_match_a_direct_regression():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 60)
    k = np.repeat([0, 1, 2], 20)
    y = np.r_[0.1 * x[:20], -0.2 * x[20:40], 0.3 * x[40:]] + rng.normal(0, 0.01, 60)
    slope, se, n = kw.keyword_slopes(x, y, k, 4)
    for j, start in enumerate((0, 20, 40)):
        b, a = np.polyfit(x[start:start + 20], y[start:start + 20], 1)
        assert slope[j] == pytest.approx(b, rel=1e-9) and se[j] > 0 and n[j] == 20
    assert np.isnan(slope[3]) and n[3] == 0


def test_shrinkage_pulls_noisy_keywords_and_detects_heterogeneity():
    rng = np.random.default_rng(1)
    same = 0.08 + rng.normal(0, 0.03, 400)          # identical true slopes, sampling noise only
    eb = kw.shrink(same, np.full(400, 0.03))
    assert eb["tau2"] < 2e-4 and abs(eb["mu"] - 0.08) < 0.01 and np.all(eb["shrinkage"] < 0.5)
    true = rng.normal(0.08, 0.1, 400)               # real between-keyword spread
    observed = true + rng.normal(0, 0.03, 400)
    eb2 = kw.shrink(observed, np.full(400, 0.03))
    assert 0.006 < eb2["tau2"] < 0.014 and eb2["i2"] > 0.8
    noisy = np.full(400, 0.03)
    noisy[0] = 0.3
    eb3 = kw.shrink(observed, noisy)
    assert abs(eb3["mean"][0] - eb3["mu"]) < abs(observed[0] - eb3["mu"])  # a noisy keyword is pulled to the mean


def test_query_lexicon_metrics():
    m = explore.query_metrics(["buy crm software price", "crm software guide", "crm software guide"],
                              "Where can I buy CRM software today?", "crm software")
    assert m["q_count"] == 3 and m["q_keyword_inclusion"] == 1.0 and m["q_distinct_share"] == pytest.approx(2 / 3)
    assert m["q_action_share"] > 0 and m["q_information_share"] > 0 and 0 < m["q_prompt_reuse"] <= 1
    assert explore.query_metrics([], "p", "k") == {}


def test_answer_metrics_weight_ranks_and_compare_with_the_keyword_rows():
    table = {"https://a.com/1": {"u": 0.9, "glued": 0.0, "domain": "a.com", "url_normalized": "https://a.com/1",
                                 **{f: 1.0 for f in review.URL_FEATURES + review.DOMAIN_FEATURES}},
             "https://b.com/2": {"u": 0.1, "glued": 1.0, "domain": "b.com", "url_normalized": "https://b.com/2",
                                 **{f: 0.0 for f in review.URL_FEATURES + review.DOMAIN_FEATURES}}}
    t = {"url": table, "keyword_rows": {("searxng", "crm"): {"https://a.com/1"}}, "keyword_u": {("searxng", "crm"): [0.2, 0.4]},
         "google_url": {("crm", "https://a.com/1")}, "google_domain": set(), "r0": {("p1", "searxng"): ({"https://b.com/2"}, 0.3)}}
    prompts = {"p1": ("crm", 0.8, "buy crm now")}
    row = {"prompt_id": "p1", "engine": "searxng", "method": "Parallel-Expansion-v1", "condition": "natural",
           "ranking": ["https://a.com/1", "https://b.com/2", "https://unknown.example"], "search_count": 3,
           "final_snippet_count": 7, "answer": "1) Go to the site. 2) Click sign up. It costs $5."}
    m = review.answer_metrics("llama4", row, prompts, t)
    w = np.array([1.0, 1 / np.log2(3)])
    assert m["cited_u"] == pytest.approx((w[0] * 0.9 + w[1] * 0.1) / w.sum())
    assert m["cited_on_topic"] == pytest.approx(w[0] / w.sum()) and m["cited_google_url"] == pytest.approx(w[0] / w.sum())
    assert m["top1_gap"] == pytest.approx(0.1) and m["null_gap"] == pytest.approx(np.mean([0.6, 0.4]))
    assert m["r0_overlap"] == pytest.approx(1 / 3) and m["r0_u"] == 0.3
    assert m["answer_steps"] == 1 and m["answer_imperatives"] >= 2 and m["answer_currency"] == 1 and m["ranking_len"] == 3
