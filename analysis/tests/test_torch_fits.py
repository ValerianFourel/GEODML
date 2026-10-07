"""Equivalence of the PyTorch backend with the CPU reference fits (gates 1 and 2 of the GPU plan; torch-cpu here)."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from analysis.interpretability.pipeline import funnel_models as fm  # noqa: E402
from analysis.interpretability.pipeline import page_readiness_ordering as pro  # noqa: E402
from analysis.interpretability.pipeline import torch_fits as tf  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture()
def tight(monkeypatch):
    """scipy's L-BFGS-B with very tight tolerances: the reference optimum."""
    import scipy.optimize
    original = scipy.optimize.minimize

    def minimize(*a, **k):
        k["options"] = {"gtol": 1e-12, "ftol": 1e-16, "maxiter": 50000, "maxcor": 50}
        result = original(*a, **k)
        # L-BFGS-B ends "ABNORMAL" when the line search cannot improve further at such tolerances; accept the point
        # as the reference only if its own gradient is tiny (independent of the code under test)
        if not result.success and np.max(np.abs(result.jac)) < 1e-8:
            result.success = True
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", minimize)


def choice_data(seed=0, sets=600, k=5, levels=4):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 8, sets)
    set_ = np.repeat(np.arange(sets), sizes)
    X = rng.normal(size=(len(set_), k))
    pos = np.concatenate([np.minimum(np.arange(s), levels - 1) for s in sizes])
    beta = rng.normal(scale=0.7, size=k)
    delta = np.r_[0, -0.4 * np.arange(1, levels)]
    u = X @ beta + delta[pos] + rng.gumbel(size=len(set_))
    chosen = np.zeros(len(set_), bool)
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    for s, (a, n) in enumerate(zip(start, sizes)):
        chosen[a + int(np.argmax(u[a:a + n]))] = True
    data = SimpleNamespace(z=np.zeros(len(set_)), set=set_, chosen=chosen, position=pos, sets=sets, levels=levels)
    return X, data


def admission_problem(seed=1, groups=400, k=4):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 12, groups)
    answer = np.repeat(np.arange(groups), sizes)
    X = rng.normal(size=(len(answer), k))
    eta = X @ rng.normal(size=k)
    admitted = (rng.random(len(answer)) < 1 / (1 + np.exp(-eta))).astype(int)
    data, keep = fm.admission_data(answer, admitted)
    return X[keep], data


def test_choice_matches_the_tight_cpu_optimum(tight):
    X, data = choice_data()
    w = np.random.default_rng(5).integers(0, 3, data.sets).astype(float)
    for weight in (None, w):
        ref = pro.fit_choice_model(X, data, set_weight=weight)
        got = tf.ChoiceProblem(X, data, CPU).fit(set_weight=weight)
        assert got.success
        assert np.max(np.abs(got.x - ref.x)) < 1e-7
        assert abs(got.fun - ref.fun) < 1e-10


def test_choice_against_default_cpu_fit_and_chunks(monkeypatch):
    X, data = choice_data(seed=2)
    ref = pro.fit_choice_model(X, data)
    got = tf.ChoiceProblem(X, data, CPU).fit()
    assert got.fun <= ref.fun + 1e-12                      # Newton is at least as good an optimum
    assert np.max(np.abs(got.x - ref.x)) < 1e-4            # agreement to the CPU optimizer's own tolerance
    monkeypatch.setattr(tf, "CHUNK_ROWS", 97)              # chunking aligned to set boundaries changes nothing
    chunked = tf.ChoiceProblem(X, data, CPU).fit()
    assert np.max(np.abs(chunked.x - got.x)) < 1e-10


def test_choice_unsorted_sets_and_drop_mask():
    X, data = choice_data(seed=3, levels=1)
    perm = np.random.default_rng(0).permutation(len(data.set))
    shuffled = SimpleNamespace(**{**data.__dict__, "set": data.set[perm], "chosen": data.chosen[perm], "position": data.position[perm]})
    a = tf.ChoiceProblem(X, data, CPU).fit()
    b = tf.ChoiceProblem(X[perm], shuffled, CPU).fit()
    assert np.max(np.abs(a.x - b.x)) < 1e-10
    free = np.array([True, False, True, True, False])
    masked = tf.ChoiceProblem(X, data, CPU).fit(free=free)
    reduced = tf.ChoiceProblem(X[:, free], data, CPU).fit()
    assert np.allclose(masked.x[~free], 0) and np.max(np.abs(masked.x[free] - reduced.x)) < 1e-10
    assert abs(masked.fun - reduced.fun) < 1e-12


def test_admission_matches_the_tight_cpu_optimum_and_inclusion(tight):
    X, data = admission_problem()
    w = np.random.default_rng(7).integers(0, 3, data.groups).astype(float)
    for weight in (None, w):
        ref = fm.fit_admission(X, data, weight=weight)
        problem = tf.AdmissionProblem(X, data, CPU)
        got = problem.fit(weight=weight)
        assert got.success
        assert np.max(np.abs(got.x - ref.x)) < 1e-7
        assert abs(got.fun - ref.fun) < 1e-10
    _, _, pi_ref = fm.admission_loss(ref.x, X, data)
    assert np.max(np.abs(problem.inclusion(ref.x) - pi_ref)) < 1e-12


def test_admission_drop_mask_and_group_chunks():
    X, data = admission_problem(seed=4)
    free = np.array([True, False, True, True])
    masked = tf.AdmissionProblem(X, data, CPU).fit(free=free)
    reduced = tf.AdmissionProblem(X[:, free], data, CPU).fit()
    assert np.max(np.abs(masked.x[free] - reduced.x)) < 1e-10 and abs(masked.fun - reduced.fun) < 1e-12
    chunked = tf.AdmissionProblem(X, data, CPU, group_chunk=37).fit()
    whole = tf.AdmissionProblem(X, data, CPU).fit()
    assert np.max(np.abs(chunked.x - whole.x)) < 1e-10


def test_monte_carlo_twins_reproduce_numpy_exactly():
    from analysis.steelman import generator as gen
    rng = np.random.default_rng(3)
    G, n, M = 40, 6, 25
    W = np.exp(rng.normal(size=(G, n)))
    W[:5, 5] = 0.0
    L = rng.integers(1, 5, G)
    U, Gm = rng.random((G, n, M)), rng.gumbel(size=(G, n, M))
    ref = gen.sample_subsets(W, L, U)
    got = tf.sample_subsets(W, L, U, CPU)
    assert np.array_equal(got.numpy(), ref)
    V, u = rng.normal(size=(G, n)), rng.random((G, n))
    assert np.max(np.abs(tf.expected_k(got, V, u, L, Gm, CPU) - gen.expected_k(ref, V, u, L, Gm))) < 1e-14


def test_two_way_fe_batch_matches_the_cpu_fits():
    from analysis.steelman import generator as gen
    rng = np.random.default_rng(4)
    a = np.repeat(np.arange(40), 5)
    r = rng.integers(0, 25, len(a))
    X = rng.normal(size=(len(a), 3))
    y = X @ np.array([0.5, -0.2, 0.1]) + rng.normal(size=40)[a] + rng.normal(size=25)[r] + 0.1 * rng.normal(size=len(a))
    weights = np.stack([np.ones(len(a))] + [rng.integers(0, 3, 40)[a].astype(float) for _ in range(6)])
    got = tf.two_way_fe_batch(y, X, a, r, weights, CPU)
    for b in range(len(weights)):
        assert np.max(np.abs(got[b] - gen.two_way_fe(y, X, a, r, weights[b]))) < 1e-8


def _toy_admission_stage():
    from analysis.interpretability.pipeline import intent_stages as stages
    rng = np.random.default_rng(11)
    keywords, per, items = 30, 8, 10
    answers = keywords * per
    answer_keyword = np.repeat(np.arange(keywords), per)
    answer_x = rng.uniform(0, 1, answers)
    answer = np.repeat(np.arange(answers), items)
    u = rng.uniform(0, 1, len(answer))
    text = rng.normal(size=len(answer))
    eta = 1.2 * text + 2.0 * (-np.abs(u - answer_x[answer])) + rng.normal(0, 1.5, answers)[answer]
    admitted = (rng.random(len(answer)) < 1 / (1 + np.exp(-eta))).astype(int)
    data, keep = fm.admission_data(answer, admitted)
    design = fm.Design([fm.Feature("text", "A3"), fm.Feature("alignment", "A1", "alignment", "u")], {"text": text[keep], "u": u[keep]})
    row_answer = answer[keep]
    design.fit_stats(answer_x[row_answer])
    stage = fm.Stage("admission", design, data, row_answer)
    return stage, answer_x, answer_keyword, stages.keyword_draws(keywords, 10, 7), stages.shuffle_draws(answer_x, answer_keyword, 9, 8)


@pytest.mark.parametrize("which", ["choice", "admission"])
def test_estimate_blocks_torch_branch_matches_cpu(which, tmp_path):
    from analysis.tests.test_funnel_models import _toy_choice_stage
    stage, x, kw, draws, shuffles = _toy_choice_stage() if which == "choice" else _toy_admission_stage()
    cpu = fm.estimate_blocks(stage, answer_x=x, answer_keyword=kw, draws=draws, shuffles=shuffles)
    fm.set_backend("torch-cpu")
    try:
        got = fm.estimate_blocks(stage, answer_x=x, answer_keyword=kw, draws=draws, shuffles=shuffles, parts=tmp_path / "t.parts.jsonl")
    finally:
        fm.set_backend("cpu")
    assert got["failed_replicates"] == cpu["failed_replicates"] == 0
    for name, f in cpu["features"].items():
        g = got["features"][name]
        assert abs(g["beta_per_sd"] - f["beta_per_sd"]) < 1e-4
        assert max(abs(a - b) for a, b in zip(g["ci95"], f["ci95"])) < 1e-4
        if f["permutation_p"] is not None:
            assert abs(g["permutation_p"] - f["permutation_p"]) <= 1 / (len(shuffles) + 1) + 1e-12
    for b, v in cpu["blocks"].items():
        assert abs(got["blocks"][b]["fit_share"] - v["fit_share"]) < 1e-3
    assert np.max(np.abs(np.asarray(got["replicates"]) - np.asarray(cpu["replicates"]))) < 1e-4


def test_torch_unit_ranges_assemble_like_cpu_ranges(tmp_path):
    from analysis.tests.test_funnel_models import _toy_choice_stage
    stage, x, kw, draws, shuffles = _toy_choice_stage()
    fm.set_backend("torch-cpu")
    try:
        parts = tmp_path / "r.parts.jsonl"
        common = dict(answer_x=x, answer_keyword=kw, draws=draws, shuffles=shuffles, parts=parts)
        for units in (("full", 0, 1), ("drop", 0, 10 ** 9), ("bootstrap", 0, 6), ("bootstrap", 6, 12), ("permutation", 0, 19)):
            with pytest.raises(fm.UnitsDone):
                fm.estimate_blocks(stage, **common, units=units)
        assembled = fm.estimate_blocks(stage, **common)
        whole = fm.estimate_blocks(stage, **{**common, "parts": tmp_path / "w.parts.jsonl"})
    finally:
        fm.set_backend("cpu")
    assert assembled["features"] == whole["features"]   # the torch path is deterministic: identical


def test_expected_cited_intent_same_on_torch_and_numpy():
    from analysis.steelman import generator as gen
    rng = np.random.default_rng(9)
    G, n = 30, 5
    keep_eta = rng.normal(size=(G, n))
    keep_eta[:4, 4] = -np.inf
    order_eta, u = rng.normal(size=(G, n)), rng.random((G, n))
    L = rng.integers(1, 4, G)
    kept = np.zeros((G, n), bool)
    kept[:, :3] = True
    for keep_model in (True, False):
        ref = gen.expected_cited_intent(keep_eta, order_eta, u, L, kept, keep_model=keep_model, M=60, seed=3, chunk=25)
        fm.set_backend("torch-cpu")
        try:
            got = gen.expected_cited_intent(keep_eta, order_eta, u, L, kept, keep_model=keep_model, M=60, seed=3, chunk=25)
        finally:
            fm.set_backend("cpu")
        assert np.max(np.abs(got - ref)) < 1e-13


def test_two_way_fe_batch_fails_only_singular_draws():
    from analysis.steelman import generator as gen
    rng = np.random.default_rng(6)
    a = np.repeat(np.arange(30), 4)
    r = rng.integers(0, 20, len(a))
    X = np.column_stack([rng.normal(size=len(a)), (a < 10).astype(float) * rng.integers(0, 2, len(a))])
    y = X @ np.array([0.4, 0.3]) + rng.normal(size=len(a))
    ok = np.ones(len(a))
    singular = np.where(a < 10, 0.0, 1.0)          # drops every row where the second column varies
    got = tf.two_way_fe_batch(y, X, a, r, np.stack([ok, singular, ok]), CPU)
    assert np.all(np.isfinite(got[[0, 2]])) and np.allclose(got[0], gen.two_way_fe(y, X, a, r, ok), atol=1e-8)
    with np.errstate(all="ignore"):
        cpu = gen.two_way_fe(y, X, a, r, singular)
    assert np.isnan(got[1]).all() == np.isnan(cpu).all()
