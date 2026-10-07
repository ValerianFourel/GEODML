"""Verdicts of PREREG.md, as pure functions of the result files."""

from __future__ import annotations

SESOI = 0.015
C1_SUPPORT, C1_NARROW = 1 / 3, 1 / 2


def _upper(ci):
    return ci[1] if ci and ci[1] is not None else float("inf")


def _lower(ci):
    return ci[0] if ci and ci[0] is not None else float("-inf")


def verdict_c1(stratum: dict, engine_chains: dict) -> dict:
    """``stratum``: chain.json strata entry; ``engine_chains``: the engine strata chains of the same method."""
    common = stratum["chain"]["common"]
    up = _upper(common["shares"]["generator"]["ci95"])
    a_common = up < C1_SUPPORT
    a_controls = _upper(stratum["chain"]["controls"]["shares"]["generator"]["ci95"]) < C1_SUPPORT
    a_engines = {e: _upper(c["shares"]["generator"]["ci95"]) < C1_SUPPORT for e, c in engine_chains.items()}
    query = _lower(common["increments"]["query_rewriting"]["ci95"]) > 0
    reranker = _lower(common["increments"]["reranker"]["ci95"]) > 0
    checks = {"generator_share_upper95": up, "a_common_below_1/3": a_common, "a_controls_below_1/3": a_controls,
              "a_engines_below_1/3": a_engines, "b_query_rewriting_positive": query, "b_reranker_positive": reranker}
    if up >= C1_NARROW:
        verdict = "failed"
    elif a_common and a_controls and all(a_engines.values()) and query and reranker:
        verdict = "supported"
    else:
        verdict = "narrowed"
    reasons = []
    if verdict == "narrowed":
        if not a_common:
            reasons.append("generator share upper bound between 1/3 and 1/2")
        if not a_controls:
            reasons.append("generator share bound fails in the controls variant")
        if not all(a_engines.values()):
            reasons.append("generator share bound fails in an engine")
        if reranker and not query:
            reasons.append("query rewriting not distinguishable from zero: mainly the reranker")
        if not reranker:
            reasons.append("reranker increment not above zero")
    return {"verdict": verdict, "checks": checks, "reasons": reasons}


def abs_bounds(ci):
    lo, hi = ci
    lower = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return lower, max(abs(lo), abs(hi))


def verdict_c2(deltas: dict) -> dict:
    """``deltas``: method -> generator.json delta["main"] entry."""
    rows = {}
    for method, d in deltas.items():
        lo90, hi90 = d["delta_gen"]["ci90"]
        low95, up95 = abs_bounds(d["delta_gen"]["ci95"])
        rows[method] = {"tost_inside": -SESOI < lo90 and hi90 < SESOI, "abs_upper95": up95, "abs_lower95": low95}
    if any(r["abs_lower95"] > SESOI for r in rows.values()):
        verdict = "failed"
    elif all(r["tost_inside"] for r in rows.values()):
        verdict = "supported"
    elif all(r["abs_upper95"] < 2 * SESOI for r in rows.values()):
        verdict = "narrowed"
    else:
        verdict = "undetermined"
    return {"verdict": verdict, "checks": rows}
