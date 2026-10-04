"""GEO drivers: prompt intent → ranking intent (E1) and what drives a page's rank (E2).

Protocol: ``analysis/docs/geo_drivers_study.md``. Observational throughout: the prompt's
axis position is a measured text property, and page positions and similarities describe
evidence text. Answers are stored as compressed rows (CSR): presented pages per answer in
presented order, with per-pair on-keyword and topic-similarity values, and ranked pages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import multiprocessing

import numpy as np

from .page_readiness_ordering import MAX_POSITION, fit_choice_model

DRIVERS = ("on_keyword", "topic_similarity", "page_intent", "intent_alignment")
X_TERMS = {"intent_alignment", "ranking", "pool", "reordering"}  # estimates involving the prompt position


@dataclass
class Answers:
    """All answers of a study, as flat arrays."""

    model: np.ndarray        # int code per answer
    engine: np.ndarray
    keyword: np.ndarray
    prompt: np.ndarray
    x: np.ndarray            # prompt axis percentile
    p_offsets: np.ndarray    # presented pages: CSR offsets, doc index, pair features
    p_docs: np.ndarray
    p_on_keyword: np.ndarray
    p_topic: np.ndarray
    r_offsets: np.ndarray    # ranked pages: CSR offsets, doc index
    r_docs: np.ndarray
    doc_z: np.ndarray        # per document: consensus axis-1 z and prompt-scale percentile
    doc_u: np.ndarray

    def subset(self, mask: np.ndarray) -> "Answers":
        index = np.flatnonzero(mask)

        def take(offsets, *columns):
            counts = offsets[index + 1] - offsets[index]
            new_offsets = np.r_[0, np.cumsum(counts)]
            rows = np.repeat(offsets[index] - new_offsets[:-1], counts) + np.arange(new_offsets[-1])
            return (new_offsets, *(c[rows] for c in columns))

        p_offsets, p_docs, p_on, p_topic = take(self.p_offsets, self.p_docs, self.p_on_keyword, self.p_topic)
        r_offsets, r_docs = take(self.r_offsets, self.r_docs)
        return Answers(self.model[index], self.engine[index], self.keyword[index], self.prompt[index], self.x[index],
                       p_offsets, p_docs, p_on, p_topic, r_offsets, r_docs, self.doc_z, self.doc_u)


def _owner(offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def answer_intent(a: Answers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per answer: rank-weighted ranking intent y, pool intent y0, and a mask of answers with a ranking."""

    n = len(a.x)
    ranked_owner = _owner(a.r_offsets)
    rank = np.arange(len(a.r_docs)) - a.r_offsets[:-1][ranked_owner]
    weight = 1.0 / np.log2(rank + 2.0)
    total = np.bincount(ranked_owner, weight, minlength=n)
    y = np.divide(np.bincount(ranked_owner, weight * a.doc_z[a.r_docs], minlength=n), total,
                  out=np.full(n, np.nan), where=total > 0)
    presented_owner = _owner(a.p_offsets)
    count = np.bincount(presented_owner, minlength=n).astype(float)
    y0 = np.divide(np.bincount(presented_owner, a.doc_z[a.p_docs], minlength=n), count,
                   out=np.full(n, np.nan), where=count > 0)
    return y, y0, (total > 0) & (count > 0)


def within_keyword_slope(x, y, keyword, weight=None) -> float:
    """Keyword-demeaned least-squares slope of y on x."""

    w = np.ones(len(x)) if weight is None else np.asarray(weight, float)
    size = int(keyword.max()) + 1
    w_sum = np.bincount(keyword, w, minlength=size)
    safe = np.where(w_sum > 0, w_sum, 1.0)
    xt = x - (np.bincount(keyword, w * x, minlength=size) / safe)[keyword]
    yt = y - (np.bincount(keyword, w * y, minlength=size) / safe)[keyword]
    denominator = float(np.sum(w * xt * xt))
    return float(np.sum(w * xt * yt) / denominator) if denominator > 0 else float("nan")


def permuted_x(a: Answers, rng: np.random.Generator) -> np.ndarray:
    """Shuffle prompt positions among the prompts of each keyword."""

    prompts, first = np.unique(a.prompt, return_index=True)
    values, keywords = a.x[first], a.keyword[first]
    shuffled = values.copy()
    for keyword in np.unique(keywords):
        members = np.flatnonzero(keywords == keyword)
        shuffled[members] = values[rng.permutation(members)]
    return shuffled[np.searchsorted(prompts, a.prompt)]


def keyword_weights(keyword: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    keywords = np.unique(keyword)
    counts = np.bincount(rng.integers(0, len(keywords), len(keywords)), minlength=len(keywords))
    return counts[np.searchsorted(keywords, keyword)].astype(float)


def _interval(values) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if len(values) else [None, None]


def _p_value(null, observed) -> float:
    null = np.asarray(null, float)
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))


def estimate_e1(a: Answers, *, bootstrap: int, permutations: int, seed: int, bins: int = 10) -> dict:
    y, y0, ok = answer_intent(a)
    sub = a.subset(ok)
    y, y0 = y[ok], y0[ok]
    series = {"ranking": y, "pool": y0, "reordering": y - y0}
    rng = np.random.default_rng(seed)
    result = {"answers": int(len(y)), "keywords": int(len(np.unique(sub.keyword)))}
    boot = defaultdict(list)
    for _ in range(bootstrap):
        w = keyword_weights(sub.keyword, rng)
        for name, values in series.items():
            boot[name].append(within_keyword_slope(sub.x, values, sub.keyword, w))
    null = defaultdict(list)
    for _ in range(permutations):
        x = permuted_x(sub, rng)
        for name, values in series.items():
            null[name].append(within_keyword_slope(x, values, sub.keyword))
    for name, values in series.items():
        slope = within_keyword_slope(sub.x, values, sub.keyword)
        result[name] = {"slope": slope, "ci95": _interval(boot[name]),
                        "permutation_p": _p_value(null[name], slope) if permutations else None}
    edges = np.minimum((sub.x * bins).astype(int), bins - 1)
    result["bins"] = [{"low": b / bins, "high": (b + 1) / bins, "answers": int(np.sum(edges == b)),
                       **{name: float(np.mean(values[edges == b])) for name, values in series.items()}}
                      for b in range(bins) if np.any(edges == b)]
    return result


@dataclass
class DriverRows:
    """Alternatives of every Plackett–Luce choice set (the attribute names fit_choice_model reads)."""

    z: np.ndarray            # page intent of the alternative
    u: np.ndarray            # page percentile on the prompt scale
    on_keyword: np.ndarray
    topic: np.ndarray
    position: np.ndarray
    set: np.ndarray
    generation: np.ndarray   # answer index (within the Answers object)
    chosen: np.ndarray
    sets: int
    levels: int


def choice_rows(a: Answers) -> DriverRows:
    """Each ranked page is chosen from the presented pages not yet ranked; a page that appears in two
    presented slots (identical text) consumes the first remaining slot."""

    z, u, on, topic, position, choice_set, generation, chosen = ([] for _ in range(8))
    count = 0
    for i in range(len(a.x)):
        start, stop = a.p_offsets[i], a.p_offsets[i + 1]
        remaining = list(range(start, stop))
        for doc in a.r_docs[a.r_offsets[i]:a.r_offsets[i + 1]]:
            taken = next((k for k, slot in enumerate(remaining) if a.p_docs[slot] == doc), None)
            if taken is None:
                raise ValueError(f"answer {i}: a ranked page is not among its remaining presented pages")
            for k, slot in enumerate(remaining):
                d = a.p_docs[slot]
                z.append(a.doc_z[d]); u.append(a.doc_u[d]); on.append(a.p_on_keyword[slot]); topic.append(a.p_topic[slot])
                position.append(min(slot - start, MAX_POSITION)); choice_set.append(count); generation.append(i)
                chosen.append(k == taken)
            del remaining[taken]
            count += 1
    position = np.asarray(position, np.int64)
    return DriverRows(np.asarray(z, float), np.asarray(u, float), np.asarray(on, float), np.asarray(topic, float),
                      position, np.asarray(choice_set, np.int64), np.asarray(generation, np.int64),
                      np.asarray(chosen, bool), count, int(position.max()) + 1 if len(position) else 1)


def restrict(rows: DriverRows, keep: np.ndarray) -> DriverRows:
    """Rows of the answers in ``keep`` (sorted answer indices), renumbered for ``Answers.subset(keep)``."""

    mask = np.isin(rows.generation, keep)
    _, choice_set = np.unique(rows.set[mask], return_inverse=True)
    position = rows.position[mask]
    return DriverRows(rows.z[mask], rows.u[mask], rows.on_keyword[mask], rows.topic[mask], position,
                      choice_set.astype(np.int64), np.searchsorted(keep, rows.generation[mask]), rows.chosen[mask],
                      int(choice_set.max()) + 1 if len(choice_set) else 0, rows.levels)


def _standardize(column: np.ndarray, stats=None):
    mean, sd = (float(np.mean(column)), float(np.std(column))) if stats is None else stats
    return (column - mean) / (sd if sd > 0 else 1.0), (mean, sd)


_STATE: dict = {}


def _driver_features(rows: DriverRows, x: np.ndarray, stats: dict, drop: str | None = None) -> np.ndarray:
    raw = {"on_keyword": rows.on_keyword, "topic_similarity": rows.topic, "page_intent": rows.z,
           "intent_alignment": -np.abs(rows.u - x[rows.generation])}
    return np.column_stack([_standardize(raw[name], stats[name])[0] for name in DRIVERS if name != drop])


def _replicate(job):
    kind, value = job
    rows, x, stats, start = _STATE["rows"], _STATE["x"], _STATE["stats"], _STATE["start"]
    if kind == "bootstrap":
        result = fit_choice_model(_driver_features(rows, x, stats), rows, set_weight=value, start=start)
    else:
        result = fit_choice_model(_driver_features(rows, value, stats), rows, start=start)
    return [float(v) for v in result.x[:len(DRIVERS)]]


def _run(jobs, workers):
    if workers <= 1:
        return [_replicate(job) for job in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(_replicate, jobs, chunksize=1))


def estimate_e2(a: Answers, *, bootstrap: int, permutations: int, seed: int, workers: int = 1,
                rows: DriverRows | None = None) -> dict:
    rows = choice_rows(a) if rows is None else rows
    if not rows.sets:
        return {"answers": 0}
    raw = {"on_keyword": rows.on_keyword, "topic_similarity": rows.topic, "page_intent": rows.z,
           "intent_alignment": -np.abs(rows.u - a.x[rows.generation])}
    stats = {name: _standardize(column)[1] for name, column in raw.items()}
    full = fit_choice_model(_driver_features(rows, a.x, stats), rows)
    result = {"answers": int(len(np.unique(rows.generation))), "choice_sets": rows.sets,
              "alternatives": int(len(rows.z)), "mean_negative_log_likelihood": float(full.fun),
              "feature_means_sd": {k: list(v) for k, v in stats.items()},
              "position_effects": [0.0, *map(float, full.x[len(DRIVERS):])]}
    gains = {}
    for name in DRIVERS:
        dropped = fit_choice_model(_driver_features(rows, a.x, stats, drop=name), rows)
        gains[name] = max(0.0, float(dropped.fun - full.fun))
    total_gain = sum(gains.values())
    rng = np.random.default_rng(seed)
    set_keyword = a.keyword[rows.generation[np.r_[0, np.flatnonzero(np.diff(rows.set)) + 1]]]
    jobs = [("bootstrap", keyword_weights(set_keyword, rng)) for _ in range(bootstrap)]
    jobs += [("permutation", permuted_x(a, rng)) for _ in range(permutations)]
    _STATE.update(rows=rows, x=a.x, stats=stats, start=full.x)
    replicates = _run(jobs, workers)
    boot, null = np.asarray(replicates[:bootstrap]), np.asarray(replicates[bootstrap:])
    drivers = {}
    for j, name in enumerate(DRIVERS):
        beta = float(full.x[j])
        drivers[name] = {"beta_per_sd": beta, "odds_ratio_per_sd": float(np.exp(beta)),
                         "ci95": _interval(boot[:, j]) if bootstrap else [None, None],
                         "fit_share": gains[name] / total_gain if total_gain > 0 else None,
                         "permutation_p": _p_value(null[:, j], beta) if (permutations and name in X_TERMS) else None}
    result["drivers"] = drivers
    # Robustness: the earlier interaction form with the two keyword drivers.
    centered = a.x[rows.generation] - 0.5
    robust = fit_choice_model(np.column_stack((_standardize(rows.on_keyword, stats["on_keyword"])[0],
                                               _standardize(rows.topic, stats["topic_similarity"])[0],
                                               rows.z, rows.z * centered)), rows)
    result["robustness_interaction_form"] = dict(zip(("on_keyword", "topic_similarity", "page_z", "page_z_x_prompt_axis"),
                                                     map(float, robust.x[:4])))
    return result


def replication(strata: dict, *, section: str, terms, x_terms=X_TERMS) -> dict:
    """A term replicates if every stratum has the same sign, a 95% CI excluding 0 and,
    for terms involving the prompt position (``x_terms``), permutation p < 0.05."""

    summary = {}
    for term in terms:
        rows = []
        for name, values in strata.items():
            block = values.get(section) or {}
            entry = (block.get("drivers") or {}).get(term) if section == "e2" else block.get(term)
            if entry is None:
                continue
            estimate = entry.get("beta_per_sd", entry.get("slope"))
            low, high = entry["ci95"]
            rows.append({"stratum": name, "estimate": estimate, "ci95": [low, high], "p": entry.get("permutation_p")})
        signs = {np.sign(r["estimate"]) for r in rows if r["estimate"] is not None}
        excludes = all(r["ci95"][0] is not None and (r["ci95"][0] > 0 or r["ci95"][1] < 0) for r in rows)
        significant = all(r["p"] is not None and r["p"] < 0.05 for r in rows) if term in x_terms else True
        summary[term] = {"replicates": bool(rows) and len(signs) == 1 and excludes and significant,
                         "strata": rows}
    return summary
