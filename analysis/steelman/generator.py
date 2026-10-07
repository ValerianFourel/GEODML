"""The generator's own choices among the shown links (PREREG.md, C2), Qwen.

* Keep: Chamberlain conditional logit within each answer given its cited count L (Reactive only; Qwen keeps
  99% of shown links under Parallel). Order: Plackett–Luce over the cited links with shown-slot effects.
  Features: the main specification of ``funnel_study`` (blocks A1 intent, A2 topic, A3, A4, B, C1, C2).
* Δ_gen = β_x(E_full[K]) − β_x(E_blind[K]): expected rank-weighted cited intent under the fitted models,
  computed by Monte Carlo with common random numbers (exact sequential sampling of the conditional-logit
  subset from elementary symmetric polynomials; Gumbel-max for Plackett–Luce). "Blind" refits without A1
  (primary) or zeroes the A1 coefficients (secondary). Keyword-bootstrap refits give the intervals.
* Variants (reported): no slot (keep without slot dummies, order without slot effects), score control
  (the reranker's logit score as a feature).
* Two-way fixed effects (answer and snapshot row) linear probability models for keep and for rank credit.
* Within-answer correlation of the shown slot with intent, topic and the reranker score.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from analysis.interpretability.pipeline import funnel_models as fm
from analysis.interpretability.pipeline import intent_stages as stages

from .chain import summarise, within_keyword_ols
from .tables import PARALLEL, Tables, rank_weights, strata

SLOT_CAP = 10
SESOI = 0.015
VARIANTS = ("main", "no_slot", "score")
A1 = "A1"


# ---------------------------------------------------------------- design rows of the shown links

def shown_rows(t: Tables, mask: np.ndarray) -> SimpleNamespace:
    """Shown rows of the answers in ``mask``, sorted by (answer, slot)."""
    p = t.pres
    idx = np.flatnonzero(mask[p["answer"]])
    idx = idx[np.lexsort((p["slot"][idx], p["answer"][idx]))]
    return SimpleNamespace(idx=idx, answer=p["answer"][idx], row=p["row"][idx], slot=np.minimum(p["slot"][idx], SLOT_CAP),
                           rank=p["rank"][idx], kept=p["rank"][idx] >= 0, topic=p["topic"][idx],
                           on_keyword=p["on_keyword"][idx], logit=p["logit"][idx], block=p["block"][idx])


def feature_columns(t: Tables, s: SimpleNamespace) -> dict:
    from analysis.scripts import funnel_study as study
    rf = t.data.rf
    cols = {name: rf[name][s.row] for name, *_ in study.ROW_FEATURES}
    cols.update({k: rf[k][s.row] for k in study.MISSING_INDICATORS})
    cols["u"] = rf["u"][s.row]
    cols["topic_similarity"], cols["on_keyword"] = s.topic, s.on_keyword
    cols["reranker_logit"] = s.logit
    return cols


def base_features(variant: str, drop_a1: bool) -> list:
    from analysis.scripts import funnel_study as study
    feats = [f for f in study._features("main", "K|P") if not (drop_a1 and f.block == A1)]
    if variant == "score":
        feats.append(fm.Feature("reranker_logit", "S"))
    return feats


def keep_parts(s, cols, stats, variant: str, drop_a1: bool):
    """(fitting Stage on informative answers, prediction Design on all shown rows)."""
    adm, informative = fm.admission_data(s.answer, s.kept)
    feats = base_features(variant, drop_a1)
    all_cols = dict(cols)
    st = dict(stats)
    if variant != "no_slot":
        rows = np.flatnonzero(informative)
        for v in sorted(set(s.slot[rows].tolist()) - {0}):
            name = f"shown_slot_{v}"
            all_cols[name] = (s.slot == v).astype(float)
            sd = float(all_cols[name][rows].std())
            st[name] = (float(all_cols[name][rows].mean()), sd if sd > 0 else 1.0)
            feats.append(fm.Feature(name, "slot"))
    fit_cols = {k: v[informative] for k, v in all_cols.items()}
    stage = fm.Stage("admission", fm.Design(feats, fit_cols, st), adm, s.answer[informative])
    return stage, fm.Design(feats, all_cols, st)


def order_parts(s, cols, stats, variant: str, drop_a1: bool):
    """(fitting Stage: Plackett–Luce over the cited links in cited order, prediction Design on all shown rows)."""
    feats = base_features(variant, drop_a1)
    rows_out, sets, chosen, position, set_answer = [], [], [], [], []
    starts = np.flatnonzero(np.r_[True, s.answer[1:] != s.answer[:-1]])
    ends = np.r_[starts[1:], len(s.answer)]
    for a0, a1 in zip(starts, ends):
        members = [r for r in range(a0, a1) if s.kept[r]]
        if len(members) < 2:
            continue
        remaining = sorted(members, key=lambda r: s.rank[r])
        while len(remaining) > 1:
            pick = remaining[0]
            for r in remaining:
                rows_out.append(r); sets.append(len(set_answer)); chosen.append(r == pick)
                position.append(int(s.slot[r]) if variant != "no_slot" else 0)
            set_answer.append(int(s.answer[a0]))
            remaining = remaining[1:]
    rows_out = np.asarray(rows_out, np.int64)
    position = np.asarray(position, np.int64)
    data = SimpleNamespace(z=np.zeros(len(rows_out)), set=np.asarray(sets, np.int64), chosen=np.asarray(chosen, bool),
                           position=position, sets=len(set_answer), levels=int(position.max()) + 1 if len(position) else 1)
    stage = fm.Stage("choice", fm.Design(feats, {k: v[rows_out] for k, v in cols.items()}, dict(stats)), data,
                     s.answer[rows_out], np.asarray(set_answer, np.int64))
    return stage, fm.Design(feats, cols, dict(stats))


# ---------------------------------------------------------------- expected cited intent (Monte Carlo)

def grid(answer: np.ndarray, values: np.ndarray, fill=0.0):
    """(G, n) padded matrix of per-row values by answer (rows sorted by answer, slot), the answer ids and widths."""
    ids, start, size = np.unique(answer, return_index=True, return_counts=True)
    n = int(size.max())
    G = len(ids)
    col = np.arange(len(answer)) - np.repeat(start, size)
    out = np.full((G, n), fill, float)
    out[np.repeat(np.arange(G), size), col] = values
    return out, ids, size


def sample_subsets(W: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Exact draws of the conditional-logit subset of size L_g with weights W (padding = 0), one per column
    of U (G, n, M uniforms): P(take j | r still needed) = w_j e_{r−1}(w_{j+1:}) / e_r(w_{j:})."""
    G, n = W.shape
    M = U.shape[2]
    top = int(L.max()) if len(L) else 0
    W = W / np.maximum(W.max(axis=1, keepdims=True), 1e-300)
    S = np.zeros((n + 1, G, top + 1))
    S[n][:, 0] = 1.0
    for j in range(n - 1, -1, -1):
        S[j] = S[j + 1]
        S[j][:, 1:] = S[j + 1][:, 1:] + W[:, j:j + 1] * S[j + 1][:, :-1]
    need = np.repeat(L[:, None].astype(np.int64), M, axis=1)
    g = np.arange(G)[:, None]
    out = np.zeros((G, n, M), bool)
    for j in range(n):
        den = S[j][g, need]
        num = W[:, j][:, None] * S[j + 1][g, np.maximum(need - 1, 0)]
        prob = np.where((need > 0) & (den > 0), num / np.where(den > 0, den, 1.0), 0.0)
        take = U[:, j, :] < prob
        out[:, j, :] = take
        need = need - take
    return out


def expected_k(include: np.ndarray, V: np.ndarray, u: np.ndarray, L: np.ndarray, gumbel: np.ndarray) -> np.ndarray:
    """Mean over draws of the rank-weighted intent of the included links ordered by Plackett–Luce utilities V
    (Gumbel-max). include: (G, n, M) bool; V, u: (G, n); gumbel: (G, n, M)."""
    score = np.where(include, V[:, :, None] + gumbel, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")
    u_sorted = np.take_along_axis(np.broadcast_to(u[:, :, None], score.shape), order, axis=1)
    n = V.shape[1]
    w = rank_weights(n)
    wmask = (np.arange(n)[None, :] < L[:, None]) * w[None, :]
    num = np.einsum("gn,gnm->gm", wmask, u_sorted)
    return (num / wmask.sum(axis=1, keepdims=True)).mean(axis=1)


def crn(G: int, n: int, M: int, seed: int, chunk: int):
    rng = np.random.default_rng([seed, chunk])
    return rng.random((G, n, M)), rng.gumbel(size=(G, n, M))


def expected_cited_intent(keep_eta, order_eta, u, L, kept_obs, *, keep_model: bool, M: int, seed: int, chunk=50) -> np.ndarray:
    """E[K] per answer. keep_eta/order_eta/u/kept_obs: (G, n) padded (keep_eta −inf on padding)."""
    G, n = u.shape
    W = np.where(np.isfinite(keep_eta), np.exp(keep_eta - np.nanmax(np.where(np.isfinite(keep_eta), keep_eta, -np.inf), axis=1, keepdims=True)), 0.0)
    total = np.zeros(G)
    done = 0
    c = 0
    while done < M:
        m = min(chunk, M - done)
        U, Gm = crn(G, n, m, seed, c)
        include = sample_subsets(W, L, U) if keep_model else np.repeat(kept_obs[:, :, None], m, axis=2)
        total += expected_k(include, order_eta, u, L, Gm) * m
        done += m
        c += 1
    return total / M


# ---------------------------------------------------------------- one stratum

class Stratum:
    """Everything one stratum needs for refits and predictions under a keyword weight vector."""

    def __init__(self, t: Tables, mask: np.ndarray, stats: dict, method: str, M: int, seed: int):
        self.t, self.M, self.seed = t, M, seed
        self.keep_model = method != PARALLEL
        s = shown_rows(t, mask)
        self.s = s
        cols = feature_columns(t, s)
        sd = float(np.nanstd(cols["reranker_logit"]))
        stats = {**stats, "reranker_logit": (float(np.nanmean(cols["reranker_logit"])), sd if sd > 0 else 1.0)}
        self.x_row = t.x[s.answer]
        self.parts = {}
        for v in VARIANTS:
            for blind in (False, True):
                self.parts[(v, blind)] = {"order": order_parts(s, cols, stats, v, blind),
                                          "keep": keep_parts(s, cols, stats, v, blind) if self.keep_model else None}
        self.u_grid, self.ids, self.width = grid(s.answer, cols["u"])
        self.kept_grid = grid(s.answer, s.kept.astype(float))[0] > 0
        self.slot_grid = grid(s.answer, s.slot.astype(float))[0].astype(np.int64)
        self.valid = grid(s.answer, np.ones(len(s.answer)))[0] > 0
        self.L = self.kept_grid.sum(axis=1)
        self.use = self.L >= 1
        self.keyword = t.keyword[self.ids]
        self.K_obs = t.values["K"][self.ids]
        self.X = {key: {name: part[1].matrix(self.x_row) for name, part in (("order", d["order"]), ("keep", d["keep"])) if part}
                  for key, d in self.parts.items()}

    def fit(self, key, which, weight=None, start=None) -> np.ndarray:
        stage, _ = self.parts[key][which]
        X = stage.design.matrix(self.t.x[stage.row_answer])
        if weight is not None:
            g = stage.data.answer if stage.kind == "admission" else stage.set_answer
            weight = weight[self.t.keyword[g]]
        r = fm._fit(stage, X, weight=weight, start=start)
        return np.asarray(r.x, float)

    def eta(self, key, which, theta: np.ndarray, zero_a1: bool = False) -> np.ndarray:
        stage, design = self.parts[key][which]
        k = len(design.features)
        beta = theta[:k].copy()
        if zero_a1:
            beta[[i for i, f in enumerate(design.features) if f.block == A1]] = 0.0
        eta = self.X[key][which] @ beta
        g = grid(self.s.answer, eta)[0]
        if which == "order" and len(theta) > k:
            effects = np.r_[0.0, theta[k:]]
            g = g + effects[np.minimum(self.slot_grid, len(effects) - 1)]
        return np.where(self.valid, g, -np.inf)

    def expected(self, key, thetas: dict, zero_a1=False) -> np.ndarray:
        keep_eta = self.eta(key, "keep", thetas["keep"], zero_a1) if self.keep_model else np.where(self.valid, 0.0, -np.inf)
        order_eta = self.eta(key, "order", thetas["order"], zero_a1)
        u = np.where(self.valid, self.u_grid, 0.0)
        out = np.full(len(self.ids), np.nan)
        out[self.use] = expected_cited_intent(keep_eta[self.use], order_eta[self.use], u[self.use], self.L[self.use],
                                              self.kept_grid[self.use], keep_model=self.keep_model, M=self.M, seed=self.seed)
        return out

    def slope(self, y, weight=None) -> float:
        m = self.use & np.isfinite(y)
        w = None if weight is None else weight[self.keyword[m]]
        return float(within_keyword_ols(self.t.x[self.ids[m]][:, None], y[m], self.keyword[m], w)[0])

    def replicate(self, counts=None, starts=None) -> dict:
        """Refit every variant (full and blind) under keyword weights ``counts`` and return the slopes."""
        out = {}
        for v in VARIANTS:
            th = {}
            for blind in (False, True):
                key = (v, blind)
                th[blind] = {w: self.fit(key, w, counts, None if starts is None else starts[v][blind][w])
                             for w in ("order", "keep") if self.parts[key][w] is not None}
            e_full = self.expected((v, False), th[False])
            e_blind = self.expected((v, True), th[True])
            e_zero = self.expected((v, False), th[False], zero_a1=True)
            sf, sb, sz, so = self.slope(e_full, counts), self.slope(e_blind, counts), self.slope(e_zero, counts), self.slope(self.K_obs, counts)
            out[v] = {"delta_gen": sf - sb, "delta_gen_zeroed": sf - sz, "slope_expected_full": sf, "slope_observed_K": so,
                      "model_check": sf - so,
                      "theta": {str(b): {w: th[b][w].tolist() for w in th[b]} for b in (False, True)}}
        return out


_STATE: dict = {}


def _job(i):
    st, draws, starts = _STATE["st"], _STATE["draws"], _STATE["starts"]
    try:
        return st.replicate(draws[i], starts)
    except RuntimeError as error:
        return {"failed": str(error)}


def coefficient_table(st: Stratum, full: dict, reps: list) -> dict:
    out = {}
    for v in VARIANTS:
        for which in ("keep", "order"):
            part = st.parts[(v, False)][which]
            if part is None:
                continue
            names = [f.name for f in part[1].features]
            obs = np.asarray(full[v]["theta"]["False"][which])
            boot = np.asarray([r[v]["theta"]["False"][which] for r in reps if "failed" not in r])
            out[f"{v}|{which}"] = {n: summarise(obs[j], boot[:, j] if len(boot) else [], None)
                                   for j, n in enumerate(names) if n in ("intent_alignment", "intent_x_prompt", "page_intent_z",
                                                                         "topic_similarity", "on_keyword", "reranker_logit")}
            if which == "order" and len(obs) > len(names):
                out[f"{v}|{which}"]["slot_effects"] = [0.0, *obs[len(names):].tolist()]
    return out


def delta_summary(full: dict, reps: list) -> dict:
    ok = [r for r in reps if "failed" not in r]
    out = {"replicates": len(reps), "failed_replicates": len(reps) - len(ok)}
    for v in VARIANTS:
        e = {}
        for q in ("delta_gen", "delta_gen_zeroed", "slope_expected_full", "slope_observed_K", "model_check"):
            e[q] = summarise(full[v][q], [r[v][q] for r in ok], None)
        k = full[v]["slope_observed_K"]
        e["delta_gen_share_of_K"] = summarise(full[v]["delta_gen"] / k, [r[v]["delta_gen"] / r[v]["slope_observed_K"] for r in ok], None)
        lo, hi = e["delta_gen"]["ci90"]
        e["tost_inside_sesoi"] = bool(lo is not None and -SESOI < lo and hi < SESOI)
        out[v] = e
    return out


# ---------------------------------------------------------------- two-way fixed effects and slot correlations

def demean_two_way(v: np.ndarray, a: np.ndarray, r: np.ndarray, w: np.ndarray, tol=1e-10, iters=1000) -> np.ndarray:
    """Residual of v after weighted projection on answer and row dummies (alternating projections)."""
    out = v.astype(float).copy()
    na, nr = int(a.max()) + 1, int(r.max()) + 1
    wa = np.maximum(np.bincount(a, w, minlength=na), 1e-300)
    wr = np.maximum(np.bincount(r, w, minlength=nr), 1e-300)
    for _ in range(iters):
        before = out.copy()
        out -= (np.bincount(a, w * out, minlength=na) / wa)[a]
        out -= (np.bincount(r, w * out, minlength=nr) / wr)[r]
        if np.max(np.abs(out - before)) < tol:
            break
    return out


def two_way_fe(y, X, a, r, w=None) -> np.ndarray:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    keep = w > 0
    y, X, a, r, w = y[keep], X[keep], a[keep], r[keep], w[keep]
    a = np.unique(a, return_inverse=True)[1]
    r = np.unique(r, return_inverse=True)[1]
    yt = demean_two_way(y, a, r, w)
    Xt = np.column_stack([demean_two_way(X[:, j], a, r, w) for j in range(X.shape[1])])
    A = (Xt * w[:, None]).T @ Xt
    try:
        return np.linalg.solve(A, (Xt * w[:, None]).T @ yt)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


def fe_analysis(t: Tables, mask: np.ndarray, draws: list) -> dict:
    """Keep (informative answers) and rank credit (all answers citing ≥ 1) on intent alignment, page intent ×
    (x − ½), topic and slot dummies, with answer and snapshot-row fixed effects; rows shown at least twice.
    Coefficients per SD of each regressor in the estimation sample."""
    s = shown_rows(t, mask)
    rf = t.data.rf
    x = t.x[s.answer]
    align = -np.abs(rf["u"][s.row] - x)
    inter = rf["page_intent_z"][s.row] * (x - 0.5)
    L = np.bincount(s.answer, s.kept, minlength=len(t.x))[s.answer]
    n = np.bincount(s.answer, minlength=len(t.x))[s.answer]
    W = np.array([rank_weights(int(l)).sum() if l > 0 else np.nan for l in range(SLOT_CAP + 2)])
    credit = np.where(s.kept, 1.0 / np.log2(np.maximum(s.rank, 0) + 2.0), 0.0) / W[np.minimum(L, SLOT_CAP + 1)]
    out = {}
    for outcome, base in (("keep", (L > 0) & (L < n)), ("credit", L > 0)):
        counts = np.bincount(s.row[base], minlength=int(s.row.max()) + 1)
        m = base & (counts[s.row] >= 2)
        if m.sum() < 200:
            out[outcome] = {"skipped": int(m.sum())}
            continue
        slots = sorted(set(s.slot[m].tolist()) - {0})
        regs = {"intent_alignment": align[m], "intent_x_prompt": inter[m], "topic_similarity": s.topic[m]}
        sds = {k: float(v.std()) for k, v in regs.items()}
        X = np.column_stack([v / sds[k] for k, v in regs.items()] + [(s.slot[m] == v).astype(float) for v in slots])
        y = (s.kept[m].astype(float) if outcome == "keep" else credit[m])
        kw = t.keyword[s.answer[m]]
        est = two_way_fe(y, X, s.answer[m], s.row[m])
        boot = np.asarray([two_way_fe(y, X, s.answer[m], s.row[m], d[kw]) for d in draws])
        # identifying variation of x within rows
        rows = np.unique(s.row[m], return_inverse=True)[1]
        xr = x[m] - (np.bincount(rows, x[m]) / np.bincount(rows))[rows]
        out[outcome] = {"rows_used": int(m.sum()), "distinct_snapshot_rows": int(rows.max() + 1),
                        "answers": int(len(np.unique(s.answer[m]))), "outcome_mean": float(y.mean()),
                        "within_row_sd_of_x": float(xr.std()), "regressor_sd": sds,
                        "coefficients_per_sd": {k: summarise(est[j], boot[:, j], None) for j, k in enumerate(regs)}}
    return out


def slot_correlations(t: Tables, mask: np.ndarray, draws: list) -> dict:
    """Mean within-answer correlation of the shown slot with page intent, alignment, the intent interaction,
    topic similarity and the reranker logit; keyword-bootstrap intervals over answers."""
    s = shown_rows(t, mask)
    rf = t.data.rf
    x = t.x[s.answer]
    series = {"u": rf["u"][s.row], "intent_alignment": -np.abs(rf["u"][s.row] - x),
              "intent_x_prompt": rf["page_intent_z"][s.row] * (x - 0.5), "topic_similarity": s.topic, "reranker_logit": s.logit}
    a = np.unique(s.answer, return_inverse=True)[1]
    na = int(a.max()) + 1
    cnt = np.bincount(a, minlength=na).astype(float)

    def centred(v):
        return v - (np.bincount(a, v, minlength=na) / cnt)[a]

    sc = centred(s.slot.astype(float))
    out = {}
    kw = t.keyword[np.unique(s.answer)]
    for name, v in series.items():
        vc = centred(v)
        num = np.bincount(a, sc * vc, minlength=na)
        den = np.sqrt(np.bincount(a, sc * sc, minlength=na) * np.bincount(a, vc * vc, minlength=na))
        r = np.divide(num, den, out=np.full(na, np.nan), where=den > 0)
        ok = np.isfinite(r)
        est = float(r[ok].mean())
        boot = [float(np.average(r[ok], weights=d[kw[ok]])) if d[kw[ok]].sum() > 0 else np.nan for d in draws]
        out[name] = summarise(est, boot, None)
    return out


# ---------------------------------------------------------------- driver

def run(t: Tables, args, *, cache_root: Path, deadline: float):
    from analysis.scripts import funnel_study as study
    stats = study.standardisation(t.data, t.ans)
    draws_all = stages.keyword_draws(int(t.keyword.max()) + 1, args.bootstrap, args.seed)
    draws = draws_all[:args.model_draws]
    common = t.common()
    cache_root.mkdir(parents=True, exist_ok=True)
    out = {"sesoi": SESOI, "mc_draws_per_answer": args.mc, "model_draws": len(draws), "strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        method = PARALLEL if "Parallel" in name else "Reactive"
        mask = m & common
        entry = {"slot_correlations": slot_correlations(t, mask, draws_all), "fixed_effects": fe_analysis(t, mask, draws_all)}
        st = Stratum(t, mask, stats, method, args.mc, args.seed)
        entry["answers"] = int(st.use.sum())
        entry["keep_informative_answers"] = int(st.parts[("main", False)]["keep"][0].data.groups) if st.keep_model else 0
        digest = hashlib.sha256(name.encode()).hexdigest()[:12]
        parts = cache_root / f"{digest}.parts.jsonl"
        done = {}
        if parts.exists():
            for line in parts.read_text().splitlines():
                try:
                    unit = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[unit["key"]] = unit["value"]
        if "full" not in done:
            full = st.replicate(None, None)
            fm._keep(parts, done, "full", full)
        full = done["full"]
        starts = {v: {b: {w: np.asarray(full[v]["theta"][str(b)][w]) for w in full[v]["theta"][str(b)]} for b in (False, True)}
                  for v in VARIANTS}
        todo = [i for i in range(len(draws)) if f"draw-{i}" not in done]
        _STATE.update(st=st, draws=draws, starts=starts)
        try:
            with multiprocessing.get_context("fork").Pool(args.workers) as pool:
                for i, res in zip(todo, pool.imap(_job, todo)):
                    fm._keep(parts, done, f"draw-{i}", res)
                    if time.monotonic() > deadline:
                        print(json.dumps({"stopped_in": name, "draws_done": sum(k.startswith("draw-") for k in done)}), flush=True)
                        pool.terminate()
                        return None
        finally:
            _STATE.clear()
        reps = [done[f"draw-{i}"] for i in range(len(draws))]
        entry["delta"] = delta_summary(full, reps)
        entry["coefficients"] = coefficient_table(st, full, reps)
        out["strata"][name] = entry
        print(json.dumps({"stratum": name, "delta_gen": entry["delta"]["main"]["delta_gen"]}), flush=True)
    return out
