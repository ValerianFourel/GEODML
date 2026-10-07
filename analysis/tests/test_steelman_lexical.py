from collections import Counter

import numpy as np

from analysis.steelman import lexical as lx


def test_action_lexicon_uses_only_given_keywords_and_x():
    rows = []
    rng = np.random.default_rng(0)
    for k in ("a", "b", "c"):
        for i in range(300):
            x = rng.random()
            words = ["buy"] if x > 0.5 else ["history"]
            rows.append({"keyword": k, "question": " ".join(words + ["about", k]), "x": x})
    lex = lx.action_lexicon(rows, lambda k: k != "c")
    assert lex[0] == "buy" and "a" not in lex


def test_lexical_selector_top_k_and_ties():
    docs = [Counter(["buy", "shoes"]), Counter(["shoes"]), Counter(["history", "shoes"]), Counter(["buy", "buy"])]
    lens = np.array([2, 1, 2, 2])
    idf = {"buy": 1.0, "shoes": 0.5, "history": 1.0}
    event = np.array([0, 0, 0, 0])
    row = np.array([0, 1, 2, 3])
    pos = np.array([4, 1, 3, 2])
    kept = lx.lexical_selector(event, np.zeros(4, int), row, [{"buy"}], np.array([2]), docs, lens, idf, 1.75, pos)
    assert kept.tolist() == [True, False, False, True]
    # all scores tie: stored position decides
    kept = lx.lexical_selector(event, np.zeros(4, int), row, [{"zzz"}], np.array([2]), docs, lens, idf, 1.75, pos)
    assert kept.tolist() == [False, True, False, True]


def test_shortlist_value_deduplicates_rows():
    u = np.array([0.0, 1.0, 0.5])
    v = lx.shortlist_value(np.array([0, 0, 0, 1]), np.array([0, 1, 1, 2]), np.array([True, True, True, False]), u, 2)
    assert v[0] == 0.5 and np.isnan(v[1])


def test_adjacent_pairs_respect_ties_blocks_and_gaps():
    from analysis.steelman.pairs import adjacent_pairs, logit_fit
    answer = np.array([0, 0, 0, 0, 1, 1])
    slot = np.array([0, 1, 2, 3, 0, 1])
    logit = np.array([2.0, 1.95, 1.95, 1.0, 0.5, 0.45])
    block = np.array([0, 0, 0, 1, 0, 0])
    a, b = adjacent_pairs(answer, slot, logit, block, 0.1)
    assert a.tolist() == [0, 4] and b.tolist() == [1, 5]    # exact tie (1,2) dropped, cross-block (2,3) dropped
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 1))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.3 + 1.0 * X[:, 0])))).astype(float)
    assert np.allclose(logit_fit(y, X), [0.3, 1.0], atol=0.12)


def test_frozen_search_selector_reproduces_the_frozen_search_order():
    from analysis.interpretability.pipeline import funnel_rows as fr
    from analysis.scripts.run_agentic_search_integration_smoke import _tokens
    rng = np.random.default_rng(1)
    words = ["buy", "cheap", "shoes", "history", "guide", "running", "price", "review"]
    keywords = ["running shoes", "shoe history", "buy shoes"]
    n = 30
    kw = [keywords[i % 3] for i in range(n)]
    title = [" ".join(rng.choice(words, 3)) for _ in range(n)]
    snippet = [" ".join(rng.choice(words, 5)) for _ in range(n)]
    pos = np.asarray([i // 3 + 1 for i in range(n)])
    url = [f"https://site{i}.example/p" for i in range(n)]
    rows = fr.SnapshotRows(engine=["e"] * n, keyword=kw, position=pos, url=url, title=title, snippet=snippet,
                           doc_id=[str(i) for i in range(n)], engine_offset={"e": 0}, snapshot_sha256={"e": "x"}, exclusions={})
    index = fr.LexicalIndex(rows, "e")
    for query in ("buy cheap running shoes", "shoe history", "price review guide"):
        k = 7
        want = index.select(query, limit=k)
        kept = lx.frozen_search_selector(np.zeros(n, np.int64), np.arange(n), [query], np.array([k]), kw, url, pos,
                                         [_tokens(x) for x in kw], [_tokens(f"{a} {b}") for a, b in zip(title, snippet)])
        assert sorted(np.flatnonzero(kept).tolist()) == sorted(want)
