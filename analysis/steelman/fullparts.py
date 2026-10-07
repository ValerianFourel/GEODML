"""Full-run parts of PREREG addendum B3: census, supply of action-ready pages, the ablated condition, and the
keyword share of queries among prompts that name their keyword. Existing tables only; no inference.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from analysis.interpretability.pipeline import intent_stages as stages

from .chain import _ci, summarise, within_keyword_ols
from .tables import Tables, rank_weights, strata

SUPPLY_THRESHOLDS = (0.6, 0.7, 0.8)
TERCILE_THRESHOLD = 0.6
TOP_X = 0.9
QUERY_TOKEN = re.compile(r"[\w-]+")  # funnel_explore.tokens, so the keyword share matches the deck's metric


# ---------------------------------------------------------------- oracles

def oracle_k(answer: np.ndarray, row: np.ndarray, x: np.ndarray, u: np.ndarray, L: np.ndarray, n_answers: int) -> np.ndarray:
    """Per answer: rank-weighted mean u of the L rows of its pool (answer, row pairs) closest to x on u (ties by row id).
    NaN when the answer has no pool row or L = 0. ``x`` and ``L`` are indexed by answer; ``u`` by row."""
    answer, row = np.asarray(answer, np.int64), np.asarray(row, np.int64)
    gap = np.abs(u[row] - x[answer])
    order = np.lexsort((row, gap, answer))
    a_sorted = answer[order]
    start = np.r_[0, np.flatnonzero(np.diff(a_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    keep = rank < L[a_sorted]
    w = np.where(keep, 1.0 / np.log2(rank + 2.0), 0.0)
    tot = np.bincount(a_sorted, w, minlength=n_answers)
    num = np.bincount(a_sorted, w * u[row[order]], minlength=n_answers)
    return np.divide(num, tot, out=np.full(n_answers, np.nan), where=tot > 0)


def oracle_k_snapshot(x: np.ndarray, L: np.ndarray, u_sorted: np.ndarray) -> np.ndarray:
    """Oracle over a whole sorted pool shared by every answer (one engine's snapshot): the L values closest to x."""
    out = np.full(len(x), np.nan)
    if not len(u_sorted):
        return out
    width = int(max(L.max(), 1)) if len(L) else 1
    pos = np.searchsorted(u_sorted, x)
    offsets = np.arange(-width, width)
    idx = np.clip(pos[:, None] + offsets[None, :], 0, len(u_sorted) - 1)
    # duplicate indices at the edges must not be counted twice
    idx = np.sort(idx, axis=1)
    dup = np.zeros_like(idx, bool)
    dup[:, 1:] = idx[:, 1:] == idx[:, :-1]
    vals = u_sorted[idx]
    gap = np.where(dup, np.inf, np.abs(vals - x[:, None]))
    order = np.argsort(gap, axis=1, kind="stable")
    v = np.take_along_axis(vals, order, axis=1)
    g = np.take_along_axis(gap, order, axis=1)
    ranks = np.arange(v.shape[1])[None, :]
    w = np.where((ranks < L[:, None]) & np.isfinite(g), rank_weights(v.shape[1])[None, :], 0.0)
    tot = w.sum(axis=1)
    return np.divide((w * v).sum(axis=1), tot, out=out, where=tot > 0)


# ---------------------------------------------------------------- helpers

def _boot(fn, draws, keyword):
    return [fn(d[keyword]) for d in draws]


def _slope(x, y, k, w=None):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        return float("nan")
    return float(within_keyword_ols(x[ok][:, None], y[ok], k[ok], None if w is None else w[ok])[0])


def _mean(v, w=None):
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    ww = np.ones(ok.sum()) if w is None else w[ok]
    return float(np.sum(ww * v[ok]) / np.sum(ww)) if ww.sum() > 0 else float("nan")


def keyword_supply_table(rows_parquet: Path, u: np.ndarray) -> "object":
    """Per keyword × engine and per keyword: rows, share of rows with u ≥ thresholds, max and SD of u."""
    import pandas as pd
    rows = pd.read_parquet(rows_parquet, columns=["row_id", "engine", "keyword"])
    rows["u"] = u[rows["row_id"].to_numpy()]
    agg = {"rows": ("u", "size"), "u_max": ("u", "max"), "u_sd": ("u", "std"), "u_mean": ("u", "mean")}
    for t in SUPPLY_THRESHOLDS:
        rows[f"ge_{t}"] = rows["u"] >= t
        agg[f"share_u_ge_{t}"] = (f"ge_{t}", "mean")
    by_engine = rows.groupby(["keyword", "engine"]).agg(**agg).reset_index()
    by_keyword = rows.groupby("keyword").agg(**agg).reset_index()
    return by_engine, by_keyword


# ---------------------------------------------------------------- supply

def supply(t: Tables, *, rows_parquet: Path, bootstrap: int, seed: int) -> dict:
    u = t.data.rf["u"]
    by_engine, by_keyword = keyword_supply_table(rows_parquet, u)
    # terciles by rank of (share of rows with u ≥ .6, then mean u): about 40% of keywords have no such row, so cut points
    # on the share alone leave the lowest tercile empty (PREREG addendum B3, note of 2026-10-08)
    order = np.lexsort((by_keyword["u_mean"].to_numpy(), by_keyword[f"share_u_ge_{TERCILE_THRESHOLD}"].to_numpy()))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    kw_tercile = dict(zip(by_keyword["keyword"].str.casefold(), np.minimum(3 * rank // len(order), 2)))
    cuts = [float(by_keyword[f"share_u_ge_{TERCILE_THRESHOLD}"].to_numpy()[order[len(order) * q // 3]]) for q in (1, 2)]
    tercile = np.asarray([kw_tercile.get(k.casefold(), -1) for k in t.keyword_text], np.int64)

    n = len(t.x)
    L = t.values["L"].astype(np.int64)
    items = np.load(Path(t.manifest["paths"]["extract"]) / "items.npz")
    who = np.repeat(np.arange(n), np.diff(items["offsets"]))
    pools = {"R": (who, items["row"]), "C": (who[items["scored"] == 1], items["row"][items["scored"] == 1]),
             "P": (who[items["presented"] >= 0], items["row"][items["presented"] >= 0]),
             "U": (t.data.u["answer"], t.data.u["row"])}
    oracle = {name: oracle_k(a, r, t.x, u, L, n) for name, (a, r) in pools.items()}
    import pandas as pd
    engines = pd.read_parquet(rows_parquet, columns=["row_id", "engine"])
    snap = np.full(n, np.nan)
    for engine in np.unique(t.engine):
        ids = engines["row_id"][engines["engine"] == engine].to_numpy()
        m = t.engine == engine
        snap[m] = oracle_k_snapshot(t.x[m], L[m], np.sort(u[ids]))
    oracle["snapshot"] = snap

    draws = stages.keyword_draws(int(t.keyword.max()) + 1, bootstrap, seed)
    common = t.common()
    supply_threshold_shares = {f"share_rows_u_ge_{th}": float((u >= th).mean()) for th in SUPPLY_THRESHOLDS}
    keyword_any = {f"share_keywords_any_u_ge_{th}": float((by_keyword[f"share_u_ge_{th}"] > 0).mean()) for th in SUPPLY_THRESHOLDS}
    out = {"tercile_threshold_u": TERCILE_THRESHOLD, "tercile_cuts_share": cuts, "tercile_rule": "rank of (share u >= .6, mean u) over all keywords", "rows": supply_threshold_shares,
           "keywords": keyword_any, "keyword_engine_median_max_u": by_engine.groupby("engine")["u_max"].median().to_dict(),
           "strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        c = m & common & np.isfinite(oracle["U"])
        x, k, K = t.x[c], t.keyword[c], t.values["K"][c]
        entry = {"answers": int(c.sum()), "slopes": {}, "utilisation": {}}
        bK = _slope(x, K, k)
        entry["slopes"]["K"] = summarise(bK, _boot(lambda w: _slope(x, K, k, w), draws, k), None)
        for pool, values in oracle.items():
            o = values[c]
            b = _slope(x, o, k)
            entry["slopes"][f"oracle_{pool}"] = summarise(b, _boot(lambda w: _slope(x, o, k, w), draws, k), None)
            entry["utilisation"][pool] = summarise(bK / b if b else np.nan,
                                                   _boot(lambda w: _slope(x, K, k, w) / _slope(x, o, k, w), draws, k), None)
        top = x >= TOP_X
        oU = oracle["U"][c]
        gap = lambda w: (_mean(x[top], None if w is None else w[top]) - _mean(oU[top], None if w is None else w[top]))  # noqa: E731
        entry["ceiling_gap_top_x"] = summarise(gap(None), _boot(gap, draws, k), None)
        entry["ceiling_gap_top_x"]["answers"] = int(top.sum())
        dec = np.minimum((x * 10).astype(int), 9)
        entry["deciles"] = [{"decile": d, "answers": int((dec == d).sum()), "mean_x": _mean(x[dec == d]), "mean_K": _mean(K[dec == d]),
                             "mean_oracle_U": _mean(oU[dec == d]), "mean_oracle_P": _mean(oracle["P"][c][dec == d])}
                            for d in range(10)]
        tc = tercile[c]
        per = {}
        for q in range(3):
            mq = tc == q
            per[q] = (lambda mq=mq: (lambda w: _slope(x[mq], K[mq], k[mq], None if w is None else w[mq])))()
            entry.setdefault("tercile_slopes", {})[str(q)] = summarise(per[q](None), _boot(per[q], draws, k), None)
            entry["tercile_slopes"][str(q)]["answers"] = int(mq.sum())
        contrast = lambda w: per[2](w) - per[0](w)  # noqa: E731
        entry["tercile_contrast_top_minus_bottom"] = summarise(contrast(None), _boot(contrast, draws, k), None)
        out["strata"][name] = entry
    return out


# ---------------------------------------------------------------- ablation

def ablation(t: Tables, *, bootstrap: int, seed: int) -> dict:
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, bootstrap, seed)
    key = {}
    for i, (mo, p, e, me, c) in enumerate(zip(t.model, t.prompt_id, t.engine, t.method, t.condition)):
        key[(mo, p, e, me, c)] = i
    out = {"strata": {}}
    for name, m in strata(t).items():
        nat = np.flatnonzero(m)
        pairs = [(i, key.get((t.model[i], t.prompt_id[i], t.engine[i], t.method[i], "ablated"))) for i in nat]
        pairs = np.asarray([(i, j) for i, j in pairs if j is not None], np.int64).reshape(-1, 2)
        if len(pairs) < 50:
            out["strata"][name] = {"skipped": int(len(pairs))}
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        dP = t.values["P"][j] - t.values["P"][i]
        dK = t.values["K"][j] - t.values["K"][i]
        ok = np.isfinite(dP) & np.isfinite(dK)
        dP, dK, k, x = dP[ok], dK[ok], t.keyword[i][ok], t.x[i][ok]
        changed = np.abs(dP) > 1e-12
        through = lambda w: float(within_keyword_ols(dP[:, None], dK, k, w)[0])  # noqa: E731
        out["strata"][name] = {
            "pairs": int(ok.sum()), "share_shortlist_intent_changed": float(changed.mean()),
            "mean_delta_P": summarise(_mean(dP), _boot(lambda w: _mean(dP, w), draws, k), None),
            "mean_delta_K": summarise(_mean(dK), _boot(lambda w: _mean(dK, w), draws, k), None),
            "pass_through_dK_on_dP": summarise(through(None), _boot(through, draws, k), None),
            "slope_delta_K_on_x": summarise(_slope(x, dK, k), _boot(lambda w: _slope(x, dK, k, w), draws, k), None)}
    return out


# ---------------------------------------------------------------- queries among keyword-naming prompts

def keyword_naming_queries(t: Tables, *, bootstrap: int, seed: int) -> dict:
    if t.queries is None:
        return {"skipped": "agent queries unavailable"}
    tok = lambda s: {w.casefold() for w in QUERY_TOKEN.findall(s or "")}  # noqa: E731
    share = np.full(len(t.x), np.nan)
    for i, qs in enumerate(t.queries):
        if qs:
            kw = tok(t.keyword_text[i])
            if kw:
                share[i] = float(np.mean([kw <= tok(q) for q in qs]))
    names = t.meta["keyword_in_prompt"] > 0
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, bootstrap, seed)
    out = {"strata": {}}
    for name, m in strata(t).items():
        res = {}
        for label, mm in (("all_prompts", m), ("prompts_naming_keyword", m & names), ("prompts_omitting_keyword", m & ~names)):
            ok = mm & np.isfinite(share)
            if ok.sum() < 50:
                res[label] = {"skipped": int(ok.sum())}
                continue
            x, y, k = t.x[ok], share[ok], t.keyword[ok]
            res[label] = {"answers": int(ok.sum()), "mean_x_below_0.2": _mean(y[x < 0.2]), "mean_x_at_least_0.8": _mean(y[x >= 0.8]),
                          "slope": summarise(_slope(x, y, k), _boot(lambda w: _slope(x, y, k, w), draws, k), None)}
        out["strata"][name] = res
    return out


# ---------------------------------------------------------------- census

def census(t: Tables) -> dict:
    """Counts of the cells the analysis sees, by model × method × engine × condition × split, and what each stage
    keeps; the planned design from the prompt file; the extraction manifest's own counts and skip reasons."""
    planned_prompts = Counter()
    from analysis.scripts.funnel_study import exploration_keyword
    for p in t.prompts.values():
        planned_prompts["exploration" if exploration_keyword(p.get("keyword") or "") else "confirmation"] += 1
    split_of = np.asarray(["exploration" if exploration_keyword(k) else "confirmation" for k in t.keyword_text])
    finite_K = np.isfinite(t.values["K"])
    common = t.common()
    rows = Counter()
    for mo, me, e, c, s, fk, cm, L, n in zip(t.model, t.method, t.engine, t.condition, split_of, finite_K, common,
                                             t.values["L"], t.values["n_shown"]):
        key = (mo, me.split("-")[0], e, c, s)
        rows[key + ("extracted",)] += 1
        rows[key + ("cited_at_least_one",)] += int(fk)
        rows[key + ("common_sample",)] += int(cm)
        rows[key + ("dropped_a_shown_link",)] += int(L < n)
    table = [{"model": k[0], "method": k[1], "engine": k[2], "condition": k[3], "split": k[4], "measure": k[5], "count": v}
             for k, v in sorted(rows.items())]
    extract_manifest = json.loads((Path(t.manifest["paths"]["extract"]) / "manifest.json").read_text())
    keywords = {s: int(len(set(t.keyword_text[split_of == s]))) for s in ("exploration", "confirmation")}
    return {"planned_cells_per_model": {s: n * 12 for s, n in planned_prompts.items()}, "prompts_by_split": dict(planned_prompts),
            "keywords_with_answers": keywords, "table": table,
            "extract_counts": extract_manifest.get("counts", {}), "assembled_counts": t.manifest.get("assembled_counts"),
            "queries": t.manifest.get("queries")}


def run(t: Tables, args, part: str) -> dict:
    if part == "census":
        return census(t)
    if part == "supply":
        rows = Path(args.features or Path(args.input_root) / "funnel-features-v1") / "rows.parquet"
        return supply(t, rows_parquet=rows, bootstrap=args.bootstrap, seed=args.seed)
    if part == "ablation":
        return ablation(t, bootstrap=args.bootstrap, seed=args.seed)
    if part == "queries":
        return keyword_naming_queries(t, bootstrap=args.bootstrap, seed=args.seed)
    raise ValueError(part)
