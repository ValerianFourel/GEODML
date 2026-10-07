"""Estimators of the funnel study (``analysis/docs/funnel_study.md``).

Stages of one answer i over the keyword's own snapshot rows U_i: retrieved R, scored by the reranker
C, presented P, ranked K. This module provides

* ``fit_admission``: the Chamberlain conditional logit for "which items of U_i were admitted"
  (several admitted items per answer; the answer's own intercept is conditioned away exactly):
  Pr(S_i | m_i) = prod_{j in S_i} w_ij / e_{m_i}(w_i), w_ij = exp(theta' x_ij), e_m the elementary
  symmetric polynomial of order m.
* ``Design``: named features grouped into blocks, standardised once (fixed mean/sd) and rebuilt for
  each shuffle of the prompt position for the features that depend on it.
* ``estimate_blocks``: full fit, drop-one-block fit shares, keyword-bootstrap and within-keyword
  shuffle replicates (shared draws so contrasts between stages and strata have valid intervals).
  Every fit is one work unit: units can be checkpointed to a JSON-lines file and a deadline stops
  admitting new units (``DeadlineReached``), so a long task resumes in the next allocation.
* ``rr_decomposition``: the exact identity log RR(K|U) = log RR(R|U) + log RR(C|R) + log RR(P|C)
  + log RR(K|P) between two groups of items, in every bootstrap replicate.
Observational throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .geo_drivers import _interval, _p_value
from .page_readiness_ordering import fit_choice_model


# ---------------------------------------------------------------- Chamberlain conditional logit

@dataclass
class AdmissionData:
    """Items of every informative answer (0 < admitted < items), padded to a G x n grid."""

    group: np.ndarray       # answer of each item row
    slot: np.ndarray        # column of the item in its group's padded row
    admitted: np.ndarray    # 0/1 per item
    groups: int
    width: int
    m: np.ndarray           # admitted items per group
    answer: np.ndarray      # original answer index of each group
    dropped: dict = field(default_factory=dict)

    @property
    def z(self):  # the fit interface only needs the row count
        return self.admitted


def admission_data(answer: np.ndarray, admitted: np.ndarray) -> tuple[AdmissionData, np.ndarray]:
    """Group item rows by answer; drop answers with none or all items admitted (no information).
    Returns the data and the boolean mask of kept item rows (to subset the feature matrix)."""
    answer, admitted = np.asarray(answer, np.int64), np.asarray(admitted, np.int64)
    order_ok = np.all(np.diff(answer) >= 0)
    if not order_ok:
        raise ValueError("item rows must be grouped by answer (sorted)")
    ids, start, size = np.unique(answer, return_index=True, return_counts=True)
    m = np.add.reduceat(admitted, start) if len(answer) else np.zeros(0, np.int64)
    informative = (m > 0) & (m < size)
    keep_group = np.repeat(informative, size)
    sub_answer = answer[keep_group]
    gids, gstart, gsize = np.unique(sub_answer, return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(gids)), gsize)
    slot = np.arange(len(sub_answer)) - np.repeat(gstart, gsize)
    data = AdmissionData(group=group, slot=slot, admitted=admitted[keep_group], groups=len(gids),
                         width=int(gsize.max()) if len(gsize) else 0, m=m[informative], answer=gids,
                         dropped={"answers_none_admitted": int(np.sum(m == 0)), "answers_all_admitted": int(np.sum(m == size))})
    return data, keep_group


def _esp(W: np.ndarray, top: int) -> np.ndarray:
    """Elementary symmetric polynomials e_0..e_top of each row of W (padding = 0)."""
    E = np.zeros((W.shape[0], top + 1))
    E[:, 0] = 1.0
    for j in range(W.shape[1]):
        E[:, 1:] = E[:, 1:] + W[:, j:j + 1] * E[:, :-1]
    return E


def _inclusion(W: np.ndarray, m: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """Conditional inclusion probabilities pi_j = w_j e_{m-1}(w_{-j}) / e_m(w) from prefix and suffix
    polynomials (all terms positive: no cancellation), and e_m(w) per row."""
    G, n = W.shape
    rows = np.arange(G)
    S = np.zeros((n + 1, G, top + 1))
    S[n][:, 0] = 1.0
    for j in range(n - 1, -1, -1):
        S[j] = S[j + 1]
        S[j][:, 1:] = S[j + 1][:, 1:] + W[:, j:j + 1] * S[j + 1][:, :-1]
    em = S[0][rows, m]
    prefix = np.zeros((G, top + 1))
    prefix[:, 0] = 1.0
    P = np.zeros((G, n))
    for j in range(n):
        acc = np.zeros(G)
        for a in range(top):
            index = m - 1 - a
            valid = index >= 0
            acc[valid] += prefix[valid, a] * S[j + 1][rows[valid], index[valid]]
        P[:, j] = W[:, j] * acc / em
        prefix[:, 1:] = prefix[:, 1:] + W[:, j:j + 1] * prefix[:, :-1]
    return P, em


def admission_loss(theta: np.ndarray, X: np.ndarray, data: AdmissionData, weight: np.ndarray | None = None):
    """Mean negative conditional log-likelihood, its gradient and the inclusion probabilities."""
    utility = X @ theta if X.shape[1] else np.zeros(len(data.group))
    G, n, top = data.groups, data.width, int(data.m.max())
    V = np.full((G, n), -np.inf)
    V[data.group, data.slot] = utility
    peak = V.max(axis=1, keepdims=True)
    W = np.exp(V - peak)                     # padding -> 0, largest -> 1
    P, em = _inclusion(W, data.m, top)
    w = np.ones(G) if weight is None else np.asarray(weight, float)
    loglik = np.bincount(data.group, utility * data.admitted, minlength=G) - (np.log(em) + data.m * peak[:, 0])
    pi = P[data.group, data.slot]
    residual = (data.admitted - pi) * w[data.group]
    total = float(w.sum())
    return -float(np.dot(w, loglik)) / total, -(X.T @ residual) / total, pi


def fit_admission(X: np.ndarray, data: AdmissionData, *, weight: np.ndarray | None = None, ridge: float = 1e-4,
                  start: np.ndarray | None = None):
    from scipy.optimize import minimize
    X = np.asarray(X, float)
    k = X.shape[1]

    def objective(theta):
        loss, grad, _ = admission_loss(theta, X, data, weight)
        return loss + 0.5 * ridge * theta @ theta, grad + ridge * theta

    result = minimize(objective, np.zeros(k) if start is None else np.asarray(start, float), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"admission fit did not converge: {result.message}")
    return result


# ---------------------------------------------------------------- designs with named feature blocks

@dataclass
class Feature:
    name: str
    block: str
    kind: str = "static"        # static | alignment (-|u - x|) | interaction (base * (x - 0.5))
    base: str | None = None     # source column for alignment/interaction


class Design:
    """Named features with fixed standardisation; x-dependent features are rebuilt from the prompt
    position of each row so shuffles of x change exactly those columns."""

    def __init__(self, features: list[Feature], columns: dict, stats: dict | None = None):
        self.features = list(features)
        self.columns = {k: np.asarray(v, float) for k, v in columns.items()}
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        self.stats = dict(stats or {})

    def raw(self, f: Feature, x_row: np.ndarray) -> np.ndarray:
        if f.kind == "static":
            return self.columns[f.name]
        if f.kind == "alignment":
            return -np.abs(self.columns[f.base] - x_row)
        if f.kind == "interaction":
            return self.columns[f.base] * (x_row - 0.5)
        raise ValueError(f.kind)

    def fit_stats(self, x_row: np.ndarray, mask: np.ndarray | None = None) -> dict:
        for f in self.features:
            column = self.raw(f, x_row)
            values = column if mask is None else column[mask]
            values = values[np.isfinite(values)]
            mean, sd = (float(values.mean()), float(values.std())) if len(values) else (0.0, 1.0)
            self.stats[f.name] = (mean, sd if sd > 0 else 1.0)
        return self.stats

    def _column(self, f: Feature, x_row: np.ndarray) -> np.ndarray:
        mean, sd = self.stats[f.name]
        z = (self.raw(f, x_row) - mean) / sd
        return np.where(np.isfinite(z), z, 0.0)

    def matrix(self, x_row: np.ndarray, *, drop_block: str | None = None, rows: np.ndarray | None = None) -> np.ndarray:
        """Standardised design; missing values become 0 (the mean) — missingness enters through the
        block indicators the caller adds as static features. Column-major, so that the choice fit's
        per-column arrays are views and a refresh of the x-dependent columns touches only those."""
        kept = [f for f in self.features if drop_block is None or f.block != drop_block]
        n = len(x_row) if rows is None else len(rows)
        out = np.empty((n, len(kept)), order="F")
        for j, f in enumerate(kept):
            z = self._column(f, x_row)
            out[:, j] = z if rows is None else z[rows]
        return out

    def refresh(self, X: np.ndarray, x_row: np.ndarray) -> None:
        """Overwrite, in place, the x-dependent columns of a full ``matrix`` with those of ``x_row``."""
        for j, f in enumerate(self.features):
            if f.kind != "static":
                X[:, j] = self._column(f, x_row)

    @property
    def names(self):
        return [f.name for f in self.features]

    @property
    def blocks(self):
        return list(dict.fromkeys(f.block for f in self.features))

    @property
    def x_dependent(self):
        return {f.name for f in self.features if f.kind != "static"}


# ---------------------------------------------------------------- block estimation with shared draws

@dataclass
class Stage:
    """One stage model: ``kind`` is 'admission' (Chamberlain) or 'choice' (Plackett–Luce via
    fit_choice_model). ``row_answer`` maps design rows to answers; ``data`` is AdmissionData or a
    ChoiceRows-like object; ``set_answer`` maps choice sets to answers (choice stages)."""

    kind: str
    design: Design
    data: object
    row_answer: np.ndarray
    set_answer: np.ndarray | None = None


_STATE: dict = {}


class DeadlineReached(Exception):
    """The deadline passed with work units of a task still unstarted; finished units are checkpointed."""


class UnitsDone(Exception):
    """A unit range of a task (``estimate_blocks(units=...)``) is finished; the task itself is not assembled yet."""


def parse_units(spec: str | None):
    """"full", "drop", "bootstrap:LO-HI" or "permutation:LO-HI" (HI exclusive) -> (kind, lo, hi); None passes through."""
    if not spec:
        return None
    kind, _, rng = spec.partition(":")
    if kind not in ("full", "drop", "bootstrap", "permutation"):
        raise ValueError(f"unknown unit kind: {kind}")
    lo, hi = (int(v) for v in rng.split("-")) if rng else (0, 10 ** 9)
    return kind, lo, hi


def _fit(stage: Stage, X: np.ndarray, weight=None, start=None):
    if stage.kind == "admission":
        return fit_admission(X, stage.data, weight=weight, start=start)
    return fit_choice_model(X, stage.data, set_weight=weight, start=start)


def _group_keyword(stage: Stage, answer_keyword: np.ndarray) -> np.ndarray:
    if stage.kind == "admission":
        return answer_keyword[stage.data.answer]
    return answer_keyword[stage.set_answer]


def _replicate(job):
    kind, _, value = job
    stage, x, keyword, start = _STATE["stage"], _STATE["x"], _STATE["keyword"], _STATE["start"]
    if kind == "drop":  # drop-one-block refit (no start, no weight): its failure is an error, as for the full fit
        return [float(_fit(stage, stage.design.matrix(x[stage.row_answer], drop_block=value)).fun)]
    k = len(stage.design.features)
    # the full-fit matrix, inherited by forked workers; each unit first writes the x-dependent columns
    # it needs (the observed x for the bootstrap, a shuffle for the null), so only those pages are copied
    X = _STATE["X"]
    stage.design.refresh(X, (x if kind == "bootstrap" else value)[stage.row_answer])
    try:
        if kind == "bootstrap":
            result = _fit(stage, X, weight=value[_group_keyword(stage, keyword)], start=start)
        else:
            result = _fit(stage, X, start=start)
        return [float(v) for v in result.x[:k]]
    except RuntimeError:
        return [float("nan")] * k  # counted as a failed replicate, never dropped silently


def _load_parts(parts: Path | None) -> dict:
    """Finished units of an interrupted task (a torn last line from a killed writer is ignored), including the
    unit-range files ``<parts>.<kind>-<lo>-<hi>`` written by other processes or nodes."""
    done = {}
    if parts is None:
        return done
    parts = Path(parts)
    files = ([parts] if parts.exists() else []) + sorted(parts.parent.glob(parts.name + ".*"))
    for path in files:
        for line in path.read_text().splitlines():
            try:
                unit = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.setdefault(unit["key"], unit["value"])
    return done


def _keep(parts: Path | None, done: dict, key: str, value) -> None:
    done[key] = value
    if parts is not None:
        with open(parts, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"key": key, "value": value}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _run(jobs, workers, parts: Path | None = None, deadline: float | None = None, done: dict | None = None):
    """Results of ``jobs`` (kind, index, value) in order. Units already in ``done``/``parts`` are reused;
    after ``deadline`` (time.monotonic) no new unit starts and DeadlineReached is raised once the
    running ones are saved."""
    done = _load_parts(parts) if done is None else done
    todo = [job for job in jobs if f"{job[0]}-{job[1]}" not in done]
    late = lambda: deadline is not None and time.monotonic() > deadline  # noqa: E731
    if workers <= 1 or len(todo) < 2:
        for job in todo:
            if late():
                raise DeadlineReached
            _keep(parts, done, f"{job[0]}-{job[1]}", _replicate(job))
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        queue = iter(todo)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            running = {pool.submit(_replicate, job): job for job in [next(queue) for _ in range(min(workers, len(todo)))]}
            submitted = len(running)
            while running:
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = running.pop(future)
                    _keep(parts, done, f"{job[0]}-{job[1]}", future.result())
                while len(running) < workers and submitted < len(todo) and not late():
                    job = next(queue)
                    running[pool.submit(_replicate, job)] = job
                    submitted += 1
        if submitted < len(todo):
            raise DeadlineReached
    return [done[f"{job[0]}-{job[1]}"] for job in jobs]


def estimate_blocks(stage: Stage, *, answer_x: np.ndarray, answer_keyword: np.ndarray, draws: list,
                    shuffles: list, workers: int = 1, parts: Path | None = None, deadline: float | None = None,
                    units: tuple | None = None) -> dict:
    """Coefficients per SD (odds ratio per SD), 95% keyword-bootstrap intervals, within-keyword
    permutation p for x-dependent features, drop-one-block fit shares and the replicate matrix.
    ``draws``: keyword resample counts (indexed by keyword code); ``shuffles``: answer-level x arrays.
    ``units`` = (kind, lo, hi) computes only that range of work units into ``<parts>.<kind>-<lo>-<hi>`` and raises
    UnitsDone; a later call without ``units`` assembles the task from every unit file (ranges need ``full-0`` first)."""
    design = stage.design
    X = design.matrix(answer_x[stage.row_answer])
    done = _load_parts(parts)
    if units is not None and units[0] != "full" and "full-0" not in done:
        raise ValueError("unit ranges need the full fit first (units=full)")
    if "full-0" not in done:
        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineReached
        fitted = _fit(stage, X)
        _keep(parts, done, "full-0", {"x": [float(v) for v in fitted.x], "fun": float(fitted.fun)})
    if units is not None and units[0] == "full":
        raise UnitsDone
    full = SimpleNamespace(x=np.asarray(done["full-0"]["x"], float), fun=done["full-0"]["fun"])
    k = len(design.features)
    _STATE.update(stage=stage, x=answer_x, keyword=answer_keyword, start=full.x, X=X)
    blocks = list(design.blocks)
    jobs = ([("drop", i, b) for i, b in enumerate(blocks)] + [("bootstrap", i, d) for i, d in enumerate(draws)]
            + [("permutation", i, s) for i, s in enumerate(shuffles)])
    if units is not None:
        kind, lo, hi = units
        subset = [j for j in jobs if j[0] == kind and lo <= j[1] < hi]
        target = Path(parts).with_name(f"{Path(parts).name}.{kind}-{lo}-{hi}") if parts is not None else None
        try:
            _run(subset, workers, target, deadline, done)
        finally:
            _STATE.clear()
        raise UnitsDone
    try:
        done = _run(jobs, workers, parts, deadline, done)
    finally:
        _STATE.clear()
    gains = {b: max(0.0, done[i][0] - float(full.fun)) for i, b in enumerate(blocks)}
    total_gain = sum(gains.values())
    replicates = np.asarray(done[len(blocks):], float).reshape(len(jobs) - len(blocks), k)
    boot, null = replicates[:len(draws)], replicates[len(draws):]
    out = {"features": {}, "blocks": {b: {"fit_share": gains[b] / total_gain if total_gain > 0 else None} for b in design.blocks},
           "mean_negative_log_likelihood": float(full.fun),
           "failed_replicates": int(np.sum(~np.isfinite(replicates).all(axis=1))),
           "replicates": boot.tolist(), "replicate_columns": [f.name for f in design.features]}
    if stage.kind == "choice" and len(full.x) > k:
        out["position_effects"] = [0.0, *map(float, full.x[k:])]
    if stage.kind == "admission":
        out.update(groups=int(stage.data.groups), items=int(len(stage.data.group)), dropped=stage.data.dropped)
    else:
        out.update(choice_sets=int(stage.data.sets), alternatives=int(len(stage.data.chosen)))
    for j, f in enumerate(design.features):
        beta = float(full.x[j])
        out["features"][f.name] = {
            "block": f.block, "beta_per_sd": beta, "odds_ratio_per_sd": float(np.exp(beta)),
            "ci95": _interval(boot[:, j]) if len(draws) else [None, None],
            "permutation_p": _p_value(null[:, j][np.isfinite(null[:, j])], beta)
            if (len(shuffles) and f.name in design.x_dependent) else None}
    return out


def contrast_from_replicates(a: dict, b: dict, name: str) -> dict:
    """Difference of one feature's coefficient between two fits that used the same bootstrap draws.
    Replicate columns follow each fit's design order (``replicate_columns``), not the order of the
    ``features`` mapping: JSON writers sort keys, which once paired the wrong columns."""
    for result in (a, b):
        if "replicate_columns" not in result:
            raise ValueError("fit result without replicate_columns: the order of its replicate columns is unknown")
    ia = a["replicate_columns"].index(name)
    ib = b["replicate_columns"].index(name)
    ra, rb = np.asarray(a["replicates"], float)[:, ia], np.asarray(b["replicates"], float)[:, ib]
    estimate = a["features"][name]["beta_per_sd"] - b["features"][name]["beta_per_sd"]
    return {"estimate": estimate, "ci95": _interval(ra - rb)}


def empirical_null(result: dict, block: str, level: float = 0.95) -> dict:
    """Spread of the standardised coefficients of a block that no component can see (negative controls)."""
    values = [abs(v["beta_per_sd"]) for v in result["features"].values() if v["block"] == block]
    return {"features": len(values), "quantile": float(np.quantile(values, level)) if values else None}


# ---------------------------------------------------------------- the exact funnel decomposition

STAGE_FLAGS = ("retrieved", "scored", "presented", "ranked")


def rr_decomposition(answer: np.ndarray, flags: dict, group: np.ndarray, answer_keyword: np.ndarray,
                     draws: list | None = None) -> dict:
    """log RR(K|U) = log RR(R|U) + log RR(C|R) + log RR(P|C) + log RR(K|P) between items of group +1
    and group -1 (0 = neither), each answer giving each group equal total weight. Exact in every
    bootstrap replicate because the draws only rescale the item weights."""
    answer = np.asarray(answer, np.int64)
    group = np.asarray(group, np.int64)
    stages = [np.ones(len(answer), bool)] + [np.asarray(flags[f], bool) for f in STAGE_FLAGS]
    for a, b in zip(stages[1:], stages[2:]):
        if np.any(b & ~a):
            raise ValueError("stage indicators are not nested")
    size = max(int(answer.max()) + 1, 1) if len(answer) else 1
    count = {g: np.bincount(answer[group == g], minlength=size) for g in (1, -1)}
    both = (count[1] > 0) & (count[-1] > 0)

    def logs(scale):
        out = []
        for g in (1, -1):
            sel = (group == g) & both[answer]
            w = scale[answer[sel]] / count[g][answer[sel]]
            totals = [float(np.sum(w * s[sel])) for s in stages]
            out.append(totals)
        steps = []
        for s in range(1, 5):
            hi, lo = out[0], out[1]
            if min(hi[s - 1], lo[s - 1], hi[s], lo[s]) <= 0:
                steps.append(float("nan"))
            else:
                steps.append(np.log(hi[s] / hi[s - 1]) - np.log(lo[s] / lo[s - 1]))
        return steps

    point = logs(np.ones(size))
    names = ["retrieval|U", "reranker_candidate|R", "shortlist|C", "ranking|P"]
    result = {"answers": int(both.sum()), "log_rr": dict(zip(names, point)), "total_log_rr_K_given_U": float(np.nansum(point))}
    if draws:
        reps = np.asarray([logs(np.asarray(d, float)[answer_keyword]) for d in draws])
        result["ci95"] = {n: _interval(reps[:, j]) for j, n in enumerate(names)}
        result["ci95_total"] = _interval(np.nansum(reps, axis=1))
        total = np.nansum(reps, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):  # a zero total gives a non-finite share, dropped by _interval
            result["share_ci95"] = {n: _interval(reps[:, j] / total) for j, n in enumerate(names)}
    total = result["total_log_rr_K_given_U"]
    result["share"] = {n: (v / total if total else None) for n, v in result["log_rr"].items()}
    return result
