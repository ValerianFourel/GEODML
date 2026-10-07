"""Stage chain of the cited-intent slope on one common sample, with shared keyword draws (PREREG.md).

β_K = β_R0 + (β_R − β_R0) + (β_C − β_R) + (β_P − β_C) + (β_I − β_P) + (β_K − β_I), exact by linearity in
every replicate, also with controls (the same design for every stage). Null selectors: identity (I) and
random (E[K] = P). Observational: x is a measured prompt property.
"""

from __future__ import annotations

import numpy as np

from analysis.interpretability.pipeline import intent_stages as stages
from analysis.interpretability.pipeline.geo_drivers import _p_value

from .tables import CHAIN, Tables, strata

TERMS = {  # increment name -> (stage, minus stage)
    "prompt_words_R0": ("R0", None),
    "query_rewriting": ("R", "R0"),
    "dedup_condition": ("C", "R"),
    "reranker": ("P", "C"),
    "shown_order": ("I", "P"),
    "generator": ("K", "I"),
    "generator_vs_random": ("K", "P"),
    "admission_P": ("P", None),
}


def within_keyword_ols(X: np.ndarray, y: np.ndarray, keyword: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Weighted least squares of y on the columns of X after demeaning everything within keyword."""
    X = np.asarray(X, float).reshape(len(y), -1)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    size = int(keyword.max()) + 1
    ws = np.bincount(keyword, w, minlength=size)
    safe = np.where(ws > 0, ws, 1.0)

    def centred(v):
        return v - (np.bincount(keyword, w * v, minlength=size) / safe)[keyword]

    Xt = np.column_stack([centred(X[:, j]) for j in range(X.shape[1])])
    yt = centred(np.asarray(y, float))
    A = (Xt * w[:, None]).T @ Xt
    b = (Xt * w[:, None]).T @ yt
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


def design(x: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    return x[:, None] if controls is None else np.column_stack([x, controls])


def slopes(values: dict, x, keyword, controls=None, weight=None, names=CHAIN) -> dict:
    X = design(x, controls)
    return {n: float(within_keyword_ols(X, values[n], keyword, weight)[0]) for n in names}


def derived(b: dict) -> dict:
    inc = {t: b[a] - (b[m] if m else 0.0) for t, (a, m) in TERMS.items()}
    k = b["K"]
    share = {t: (v / k if k != 0 else np.nan) for t, v in inc.items()}
    return inc, share


def _ci(values, level=0.95):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [None, None]
    lo = (1 - level) / 2
    return [float(np.quantile(v, lo)), float(np.quantile(v, 1 - lo))]


def summarise(observed: float, boot: list, null: list | None) -> dict:
    out = {"estimate": float(observed), "ci95": _ci(boot), "ci90": _ci(boot, 0.90)}
    if null is not None:
        out["permutation_p"] = _p_value(np.asarray(null, float)[np.isfinite(null)], observed)
    return out


def chain(values: dict, x, keyword, prompt, *, draws, shuffles, controls=None) -> dict:
    """Stage slopes, increments and shares with keyword-bootstrap intervals and shuffle p-values.
    ``values`` are per-answer arrays already restricted to the sample; ``shuffles`` are prompt-indexed x arrays."""
    b0 = slopes(values, x, keyword, controls)
    inc0, sh0 = derived(b0)
    B = {"slopes": {n: [] for n in CHAIN}, "inc": {t: [] for t in TERMS}, "share": {t: [] for t in TERMS}}
    for counts in draws:
        b = slopes(values, x, keyword, controls, counts[keyword])
        inc, sh = derived(b)
        for n in CHAIN:
            B["slopes"][n].append(b[n])
        for t in TERMS:
            B["inc"][t].append(inc[t])
            B["share"][t].append(sh[t])
    N = {"slopes": {n: [] for n in CHAIN}, "inc": {t: [] for t in TERMS}}
    for s in shuffles:
        b = slopes(values, s[prompt], keyword, controls)
        inc, _ = derived(b)
        for n in CHAIN:
            N["slopes"][n].append(b[n])
        for t in TERMS:
            N["inc"][t].append(inc[t])
    return {"answers": int(len(x)), "keywords": int(len(np.unique(keyword))),
            "slopes": {n: summarise(b0[n], B["slopes"][n], N["slopes"][n] if shuffles else None) for n in CHAIN},
            "increments": {t: summarise(inc0[t], B["inc"][t], N["inc"][t] if shuffles else None) for t in TERMS},
            "shares": {t: summarise(sh0[t], B["share"][t], None) for t in TERMS}}


def leave_one_keyword_out(values: dict, x, keyword) -> dict:
    """Point shares with each keyword left out: range and jackknife standard error."""
    out = {t: [] for t in TERMS}
    for k in np.unique(keyword):
        m = keyword != k
        _, sh = derived(slopes({n: v[m] for n, v in values.items()}, x[m], keyword[m]))
        for t in TERMS:
            out[t].append(sh[t])
    res = {}
    for t, v in out.items():
        v = np.asarray(v, float)
        n = len(v)
        res[t] = {"min": float(v.min()), "max": float(v.max()),
                  "jackknife_se": float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))}
    return res


def simple_slope(y, x, keyword, draws, shuffles=None, prompt=None) -> dict:
    ok = np.isfinite(y)
    y, x_, k = y[ok], x[ok], keyword[ok]
    est = float(within_keyword_ols(x_[:, None], y, k)[0])
    boot = [float(within_keyword_ols(x_[:, None], y, k, d[k])[0]) for d in draws]
    null = None
    if shuffles is not None:
        null = [float(within_keyword_ols(s[prompt[ok]][:, None], y, k)[0]) for s in shuffles]
    out = summarise(est, boot, null)
    out["answers"] = int(ok.sum())
    out["mean"] = float(y.mean())
    return out


def noise_floor(values: dict, x, keyword, cell, draws) -> dict:
    """Within (keyword, lattice target) cells: pooled within-cell SD of each stage value versus the
    within-keyword SD, and the slope on x identified only from prompts sharing a cell."""
    out = {"cells_with_two_or_more": 0, "answers_in_such_cells": 0, "terms": {}}
    _, cell_code = np.unique(cell, return_inverse=True)
    counts = np.bincount(cell_code)
    multi = counts[cell_code] >= 2
    out["cells_with_two_or_more"] = int(np.sum(counts >= 2))
    out["answers_in_such_cells"] = int(multi.sum())
    xm = x[multi]
    cm = np.unique(cell_code[multi], return_inverse=True)[1]
    km = keyword[multi]
    for n in CHAIN:
        v = values[n]

        def within_sd(v, g):
            means = np.bincount(g, v) / np.bincount(g)
            return float(np.sqrt(np.mean((v - means[g]) ** 2)))

        est = float(within_keyword_ols(xm[:, None], v[multi], cm)[0])
        boot = [float(within_keyword_ols(xm[:, None], v[multi], cm, d[km])[0]) for d in draws]
        out["terms"][n] = {"sd_within_cell": within_sd(v[multi], cm), "sd_within_keyword": within_sd(v, keyword),
                           "slope_within_cell": summarise(est, boot, None)}
    return out


def published_rows(review_parquet, draws_seed: int, n_draws: int, exploration=None) -> dict:
    """R0 and K (cited_u) slopes on x and keep rates per model · method from the published answers
    (both models, all keywords and the exploration keywords). Llama traces are not on the Mac."""
    import pandas as pd
    a = pd.read_parquet(review_parquet, columns=["model", "method", "condition", "keyword", "x", "ranking_len", "shown",
                                                 "search_count", "cited_u", "r0_u"])
    a = a[a["condition"] == "natural"]
    out = {}
    for split in ("all", "exploration"):
        g_all = a if split == "all" else a[a["keyword"].map(exploration)]
        codes = {k: i for i, k in enumerate(sorted(g_all["keyword"].unique()))}
        draws = stages.keyword_draws(len(codes), n_draws, draws_seed)
        for (model, method), g in g_all.groupby(["model", "method"]):
            k = g["keyword"].map(codes).to_numpy(np.int64)
            x = g["x"].to_numpy(float)
            name = f"{model} · {method.split('-')[0]} · {split}"
            out[name] = {"K_cited_u": simple_slope(g["cited_u"].to_numpy(float), x, k, draws),
                         "R0": simple_slope(g["r0_u"].to_numpy(float), x, k, draws),
                         "cited_count": simple_slope(g["ranking_len"].to_numpy(float), x, k, draws),
                         "mean_cited": float(g["ranking_len"].mean()), "mean_shown_published": float(g["shown"].mean()),
                         "share_answers_keeping_all_shown": float((g["ranking_len"] >= g["shown"]).mean())}
    return out


def as_stages(t: Tables, items_path) -> "stages.Stages":
    """The trace items as ``intent_stages.Stages`` (doc = snapshot row), for the cross-check with analyse_stratum."""
    it = np.load(items_path)
    off, row = it["offsets"], it["row"]
    n = len(off) - 1
    who = np.repeat(np.arange(n), np.diff(off))

    def csr(mask, key=None):
        idx = np.flatnonzero(mask)
        order = np.lexsort((key[idx], who[idx])) if key is not None else np.argsort(who[idx], kind="stable")
        idx = idx[order]
        return np.r_[0, np.cumsum(np.bincount(who[idx], minlength=n))], row[idx]

    ro, rd = csr(np.ones(len(row), bool))
    co, cd = csr(it["scored"] == 1)
    po, pd_ = csr(it["presented"] >= 0, it["presented"])
    ko, kd = csr(it["ranked"] >= 0, it["ranked"])
    return stages.Stages(model=t.model, engine=t.engine, method=t.method, condition=t.condition, keyword=t.keyword,
                         prompt=t.prompt, x=t.x, ret_offsets=ro, ret_docs=rd, cand_offsets=co, cand_docs=cd,
                         pool_offsets=po, pool_docs=pd_, rank_offsets=ko, rank_docs=kd,
                         q_offsets=np.zeros(n + 1, np.int64), q_ids=np.zeros(0, np.int64))


def control_matrix(t: Tables, mask: np.ndarray, kind: str) -> np.ndarray | None:
    m = t.meta
    if kind == "controls":
        slot = m["candidate_slot"][mask].astype(int)
        levels = sorted(set(slot.tolist()))[1:]
        cols = [np.log(m["words"][mask]), m["keyword_in_prompt"][mask]] + [(slot == s).astype(float) for s in levels]
        return np.column_stack(cols)
    if kind == "target":
        return m["target"][mask][:, None]
    return None


def run(t: Tables, *, bootstrap: int, permutations: int, seed: int, shuffle_seed: int, items_path=None,
        review_parquet=None) -> dict:
    prompt_x, prompt_kw = np.full(t.prompt.max() + 1, np.nan), np.full(t.prompt.max() + 1, -1, np.int64)
    prompt_x[t.prompt], prompt_kw[t.prompt] = t.x, t.keyword
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, bootstrap, seed)
    shuffles = stages.shuffle_draws(prompt_x, prompt_kw, permutations, shuffle_seed)
    common = t.common()
    out = {"strata": {}, "engine_strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        c = m & common
        vals = {n: t.values[n][c] for n in CHAIN}
        x, k, p = t.x[c], t.keyword[c], t.prompt[c]
        entry = {"natural_answers": int(m.sum()), "common_sample": int(c.sum())}
        entry["chain"] = {variant: chain(vals, x, k, p, draws=draws, shuffles=shuffles if variant == "common" else [],
                                         controls=control_matrix(t, c, variant))
                          for variant in ("common", "controls", "target")}
        entry["loko"] = leave_one_keyword_out(vals, x, k)
        drop = c & (t.values["L"] < t.values["n_shown"])
        entry["droppers_only"] = chain({n: t.values[n][drop] for n in CHAIN}, t.x[drop], t.keyword[drop], t.prompt[drop],
                                       draws=draws, shuffles=[])
        entry["droppers_share_of_answers"] = float(drop.sum() / max(c.sum(), 1))
        entry["cited_count"] = simple_slope(t.values["L"][m], t.x[m], t.keyword[m], draws, shuffles, t.prompt[m])
        frac = np.divide(t.values["L"], t.values["n_shown"], out=np.full(len(t.x), np.nan), where=t.values["n_shown"] > 0)
        entry["cited_share_of_shown"] = simple_slope(frac[m], t.x[m], t.keyword[m], draws, shuffles, t.prompt[m])
        entry["shown_count"] = simple_slope(t.values["n_shown"][m], t.x[m], t.keyword[m], draws)
        cell = np.asarray([f"{a}|{b}|{e}" for a, b, e in zip(t.keyword[c], t.meta["target_index"][c].astype(int), t.engine[c])])
        entry["noise_floor"] = noise_floor(vals, x, k, cell, draws)
        if items_path is not None:
            st = as_stages(t, items_path)
            sv = stages.stage_values(st, t.data.rf["u"])
            ref = stages.analyse_stratum(st, c, sv, stages.oracle_values(st, t.data.rf["u"], t.data.rf["u"],
                                                                            np.zeros(len(st.pool_docs))),
                                         draws=[], shuffles=[])["derived"]
            mine = entry["chain"]["common"]["shares"]["generator_vs_random"]["estimate"]
            entry["cross_check"] = {"analyse_stratum_share_reordering": ref["share_of_ranking:reordering"]["estimate"],
                                    "steelman_share_K_minus_P": mine,
                                    "abs_difference": abs(ref["share_of_ranking:reordering"]["estimate"] - mine)}
        out["strata"][name] = entry
    for name, m in strata(t, by_engine=True).items():
        c = m & common
        if c.sum() < 100:
            continue
        out["engine_strata"][name] = chain({n: t.values[n][c] for n in CHAIN}, t.x[c], t.keyword[c], t.prompt[c],
                                           draws=draws, shuffles=[])
    if review_parquet is not None:
        from analysis.scripts.funnel_study import exploration_keyword
        out["published_rows"] = published_rows(review_parquet, seed, bootstrap, exploration_keyword)
    return out


def followup(t: Tables, *, bootstrap: int, seed: int) -> dict:
    """Exploratory, added after the first run (not pre-registered): which prompt control absorbs the query-rewriting
    share? One control at a time, plus the slope of each control on x."""
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, bootstrap, seed)
    common = t.common()
    out = {"exploratory_after_first_run": True, "strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        c = m & common
        vals = {n: t.values[n][c] for n in CHAIN}
        x, k, p = t.x[c], t.keyword[c], t.prompt[c]
        slot = t.meta["candidate_slot"][c].astype(int)
        controls = {"keyword_in_prompt": t.meta["keyword_in_prompt"][c][:, None], "log_prompt_words": np.log(t.meta["words"][c])[:, None],
                    "candidate_slot": np.column_stack([(slot == s).astype(float) for s in sorted(set(slot.tolist()))[1:]])}
        entry = {"control_on_x": {n: simple_slope(v[:, 0], x, k, draws) for n, v in controls.items() if v.shape[1] == 1}}
        entry["single_control"] = {n: chain(vals, x, k, p, draws=draws, shuffles=[], controls=v)["shares"] for n, v in controls.items()}
        kip = t.meta["keyword_in_prompt"][c] > 0
        entry["within_keyword_named"] = {lab: chain({n: v[mm] for n, v in vals.items()}, x[mm], k[mm], p[mm], draws=draws, shuffles=[])["shares"]
                                         for lab, mm in (("prompt_names_keyword", kip), ("prompt_omits_keyword", ~kip))}
        out["strata"][name] = entry
    return out
