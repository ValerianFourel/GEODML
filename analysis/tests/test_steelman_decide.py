from analysis.steelman import decide as D


def stratum(gen_ci, ctrl_ci=(0.0, 0.2), qr=(0.01, 0.02), rr=(0.01, 0.02)):
    s = lambda ci: {"ci95": list(ci)}
    return {"chain": {"common": {"shares": {"generator": s(gen_ci)}, "increments": {"query_rewriting": s(qr), "reranker": s(rr)}},
                      "controls": {"shares": {"generator": s(ctrl_ci)}}}}


def eng(ci):
    return {"e1": {"shares": {"generator": {"ci95": list(ci)}}}}


def test_c1_verdicts():
    assert D.verdict_c1(stratum((0.05, 0.2)), eng((0.0, 0.3)))["verdict"] == "supported"
    assert D.verdict_c1(stratum((0.05, 0.4)), eng((0.0, 0.3)))["verdict"] == "narrowed"
    assert D.verdict_c1(stratum((0.05, 0.5)), eng((0.0, 0.3)))["verdict"] == "failed"
    r = D.verdict_c1(stratum((0.05, 0.2), qr=(-0.01, 0.02)), eng((0.0, 0.3)))
    assert r["verdict"] == "narrowed" and "mainly the reranker" in r["reasons"][0]
    assert D.verdict_c1(stratum((0.05, 0.2)), eng((0.0, 0.34)))["verdict"] == "narrowed"
    assert D.verdict_c1(stratum((0.05, 0.2), ctrl_ci=(0.1, 0.34)), eng((0.0, 0.3)))["verdict"] == "narrowed"


def d(ci90, ci95):
    return {"delta_gen": {"ci90": list(ci90), "ci95": list(ci95)}}


def test_c2_verdicts():
    ok = d((-0.01, 0.01), (-0.012, 0.012))
    assert D.verdict_c2({"a": ok, "b": ok})["verdict"] == "supported"
    wide = d((-0.016, 0.01), (-0.02, 0.012))
    assert D.verdict_c2({"a": ok, "b": wide})["verdict"] == "narrowed"
    assert D.verdict_c2({"a": ok, "b": d((0.02, 0.03), (0.016, 0.035))})["verdict"] == "failed"
    assert D.verdict_c2({"a": ok, "b": d((-0.03, 0.01), (-0.04, 0.02))})["verdict"] == "undetermined"
    assert D.abs_bounds((-0.01, 0.02)) == (0.0, 0.02) and D.abs_bounds((-0.03, -0.02)) == (0.02, 0.03)
