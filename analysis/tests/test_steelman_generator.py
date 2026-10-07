import itertools

import numpy as np
import pytest

from analysis.interpretability.pipeline import funnel_models as fm
from analysis.steelman import generator as gen
from analysis.steelman.tables import rank_weights


def exact_expected_k(w_keep, v_order, u, L):
    """Enumerate subsets of size L (prob ∝ Π w) and Plackett–Luce orders (utilities exp(v))."""
    n = len(u)
    w = rank_weights(L)
    total, mass = 0.0, 0.0
    for S in itertools.combinations(range(n), L):
        ps = float(np.prod([w_keep[j] for j in S]))
        for perm in itertools.permutations(S):
            rest, po = list(S), 1.0
            for j in perm:
                po *= np.exp(v_order[j]) / sum(np.exp(v_order[r]) for r in rest)
                rest.remove(j)
            total += ps * po * float(np.dot(w, u[list(perm)]) / w.sum())
        mass += ps
    return total / mass


def test_subset_sampler_matches_inclusion_probabilities():
    rng = np.random.default_rng(0)
    W = np.exp(rng.normal(size=(3, 5)))
    W[2, 4] = 0.0                                   # padding
    L = np.array([2, 3, 1])
    P, _ = fm._inclusion(W, L, 3)
    draws = gen.sample_subsets(W, L, rng.random((3, 5, 40000)))
    assert (draws.sum(axis=1) == L[:, None]).all()
    assert draws[2, 4].sum() == 0
    np.testing.assert_allclose(draws.mean(axis=2), P, atol=0.01)


def test_expected_k_matches_enumeration():
    rng = np.random.default_rng(1)
    n, G = 4, 3
    keep = rng.normal(size=(G, n))
    order = rng.normal(size=(G, n))
    u = rng.random((G, n))
    L = np.array([1, 2, 3])
    got = gen.expected_cited_intent(keep, order, u, L, None, keep_model=True, M=40000, seed=5, chunk=10000)
    want = [exact_expected_k(np.exp(keep[g]), order[g], u[g], L[g]) for g in range(G)]
    np.testing.assert_allclose(got, want, atol=0.004)


def test_fixed_keep_set_and_crn():
    rng = np.random.default_rng(2)
    G, n = 50, 5
    u = rng.random((G, n))
    kept = np.zeros((G, n), bool)
    kept[:, :3] = True
    order = rng.normal(size=(G, n))
    L = kept.sum(axis=1)
    a = gen.expected_cited_intent(np.zeros((G, n)), order, u, L, kept, keep_model=False, M=200, seed=9)
    b = gen.expected_cited_intent(np.zeros((G, n)), order, u, L, kept, keep_model=False, M=200, seed=9)
    np.testing.assert_array_equal(a, b)          # common random numbers: identical inputs, identical output
    assert np.all(a <= u[:, :3].max(axis=1) + 1e-12) and np.all(a >= u[:, :3].min(axis=1) - 1e-12)


def test_planted_intent_sensitivity_moves_expected_slope():
    rng = np.random.default_rng(3)
    G, n = 3000, 4
    x = rng.random(G)
    u = rng.random((G, n))
    L = np.full(G, 2)
    align = -np.abs(u - x[:, None])
    keep_intent = 2.0 * align
    zero = np.zeros((G, n))
    e_full = gen.expected_cited_intent(keep_intent, zero, u, L, None, keep_model=True, M=100, seed=1)
    e_blind = gen.expected_cited_intent(zero, zero, u, L, None, keep_model=True, M=100, seed=1)
    slope = lambda y: np.polyfit(x, y, 1)[0]
    assert slope(e_full) - slope(e_blind) > 0.1     # intent-sensitive keep pulls cited intent toward x
    assert abs(slope(e_blind)) < 0.05               # blind keep: no slope


def test_two_way_fe_equals_dummy_ols():
    rng = np.random.default_rng(4)
    a = np.repeat(np.arange(10), 4)
    r = rng.integers(0, 6, len(a))
    X = rng.normal(size=(len(a), 2))
    y = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(size=10)[a] + rng.normal(size=6)[r] + 0.01 * rng.normal(size=len(a))
    got = gen.two_way_fe(y, X, a, r)
    D = np.column_stack([X, (a[:, None] == np.arange(10)).astype(float), (r[:, None] == np.arange(1, 6)).astype(float)])
    want = np.linalg.lstsq(D, y, rcond=None)[0][:2]
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(gen.two_way_fe(y, X, a, r, np.ones(len(a))), got, atol=1e-10)
