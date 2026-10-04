"""Place evidence pages on the frozen prompt readiness axis and relate rankings to it.

The information-seeking to action-readiness axis was fitted on prompt text. Pages
are projected through the same frozen LLM2Vec maps, development z-scaling and
Procrustes alignment, then expressed against the 26,009-prompt distribution.
Page coordinates are an out-of-domain description of page text, never a
treatment or confounder. The prompt axis is a measured prompt property, so
every result here is an observational association.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math

import numpy as np

FORMAT_VERSION = "page-readiness-ordering-v1"
VIEWS = ("consensus", "qwen", "mistral")
MAX_POSITION = 20  # presented-position fixed effects; later positions share one level


def page_text(title: str, text: str) -> str:
    """The exact text embedded for one evidence item (what the generator saw)."""

    return f"{title.strip()}\n{text.strip()}"


def page_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptScale:
    """Map a consensus axis-1 z value onto the archived prompt percentile scale."""

    def __init__(self, final_axis_rows: Sequence[Mapping[str, object]]):
        values = np.sort(np.asarray([float(r["consensus_axis_1_z"]) for r in final_axis_rows]))
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("prompt scale needs at least two finite consensus values")
        self.values = values
        self.levels = np.linspace(0.0, 1.0, len(values))

    def percentile(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.float64)
        outside = (z < self.values[0]) | (z > self.values[-1])
        return np.interp(z, self.values, self.levels), outside


def consensus(reference_axis_1_z: np.ndarray, aligned_axis_1_z: np.ndarray) -> np.ndarray:
    """The archived final-audit rule: mean of the Qwen and aligned Mistral axis-1 z."""

    return (np.asarray(reference_axis_1_z, float) + np.asarray(aligned_axis_1_z, float)) / 2.0


def relocation_audit(final_axis_rows, aligned_rows, *, tolerance: float = 1e-9) -> dict:
    """Recompute the archived prompt coordinates from archived per-view projections."""

    final = {str(r["candidate_id"]): r for r in final_axis_rows}
    aligned = {str(r["candidate_id"]): r for r in aligned_rows}
    if len(final) != len(final_axis_rows) or len(aligned) != len(aligned_rows):
        raise ValueError("duplicate candidate ids in relocation inputs")
    ids = sorted(final)
    report = {"final_rows": len(final), "aligned_rows": len(aligned),
              "identities_equal": set(final) == set(aligned)}
    if not report["identities_equal"]:
        return {**report, "passed": False}
    recomputed = consensus([aligned[i]["reference_axis_1_z"] for i in ids],
                           [aligned[i]["candidate_aligned_axis_1_z"] for i in ids])
    archived = np.asarray([float(final[i]["consensus_axis_1_z"]) for i in ids])
    order = np.lexsort((np.asarray(ids), recomputed))  # final audit sorts by (z, id)
    percentile = np.empty(len(ids))
    percentile[order] = np.arange(len(ids)) / max(1, len(ids) - 1)
    archived_percentile = np.asarray([float(final[i]["axis_1_percentile_0_1"]) for i in ids])
    z_error = float(np.max(np.abs(recomputed - archived)))
    p_error = float(np.max(np.abs(percentile - archived_percentile)))
    return {**report, "max_abs_consensus_z_difference": z_error,
            "max_abs_percentile_difference": p_error, "tolerance": tolerance,
            "passed": z_error <= tolerance and p_error <= tolerance}


def projection_agreement(archived: Mapping[str, float], fresh: Mapping[str, float],
                         *, minimum_spearman: float = 0.999) -> dict:
    """Fresh re-embedding of archived prompts versus their archived raw axis-1 value."""

    ids = sorted(set(archived) & set(fresh))
    if len(ids) < 3:
        raise ValueError("projection agreement needs at least three shared items")
    left = np.asarray([archived[i] for i in ids], float)
    right = np.asarray([fresh[i] for i in ids], float)
    spearman = _spearman(left, right)
    return {"items": len(ids), "missing": len(set(archived) ^ set(fresh)),
            "pearson": float(np.corrcoef(left, right)[0, 1]), "spearman": spearman,
            "max_abs_difference": float(np.max(np.abs(left - right))),
            "minimum_spearman": minimum_spearman, "passed": spearman >= minimum_spearman}


def _ranks(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(values)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def generation_statistics(observation: Mapping, page_z: Mapping[str, float]) -> dict:
    """Descriptive within-answer contrasts of ranked versus presented pages."""

    presented = [page_z[p] for p in observation["presented"]]
    ranked = [page_z[p] for p in observation["ranking"]]
    row = {"n_presented": len(presented), "n_ranked": len(ranked),
           "presented_z_mean": float(np.mean(presented)) if presented else None,
           "presented_z_sd": float(np.std(presented)) if presented else None}
    if presented and ranked:
        row["ranked_minus_presented_z"] = float(np.mean(ranked) - np.mean(presented))
        row["top1_minus_presented_z"] = float(ranked[0] - np.mean(presented))
    if len(ranked) >= 3:
        rho = _spearman(-np.arange(len(ranked), dtype=float), np.asarray(ranked))
        row["rank_order_spearman"] = None if math.isnan(rho) else rho
    return row


class ChoiceData:
    """Flattened top-k Plackett-Luce choice sets: each ranked page is chosen from the
    presented pages not yet ranked. Unranked pages stay available and are never
    treated as explicitly rejected beyond the observed top-k prefix."""

    def __init__(self, observations: Sequence[Mapping], page_z: Mapping[str, float]):
        z, position, choice_set, generation, chosen = [], [], [], [], []
        keywords, prompts, axis = [], [], []
        set_count = 0
        for g, obs in enumerate(observations):
            keywords.append(obs["keyword"])
            prompts.append(obs["prompt_id"])
            axis.append(float(obs["prompt_axis"]))
            remaining = list(enumerate(obs["presented"]))
            for pick in obs["ranking"]:
                for slot, page in remaining:
                    z.append(page_z[page])
                    position.append(min(slot, MAX_POSITION))
                    choice_set.append(set_count)
                    generation.append(g)
                    chosen.append(page == pick)
                remaining = [(s, p) for s, p in remaining if p != pick]
                set_count += 1
        if not set_count:
            raise ValueError("no ranked choices to fit")
        self.z = np.asarray(z, float)
        self.position = np.asarray(position, np.int64)
        self.set = np.asarray(choice_set, np.int64)
        self.generation = np.asarray(generation, np.int64)
        self.chosen = np.asarray(chosen, bool)
        self.sets = set_count
        self.set_generation = self.generation[np.r_[0, np.flatnonzero(np.diff(self.set)) + 1]]
        self.keywords = np.asarray(keywords)
        self.prompts = np.asarray(prompts)
        self.axis = np.asarray(axis, float)
        self.levels = int(self.position.max()) + 1
        if np.count_nonzero(self.chosen) != self.sets:
            raise ValueError("each choice set must contain exactly one chosen page")


def fit_plackett_luce(data: ChoiceData, *, axis: np.ndarray | None = None,
                      set_weight: np.ndarray | None = None, ridge: float = 1e-4,
                      start: np.ndarray | None = None) -> dict:
    """Utility = b_page*z + b_interaction*z*(axis-0.5) + position effect (position 0 = 0)."""

    from scipy.optimize import minimize

    centered = (data.axis if axis is None else axis)[data.generation] - 0.5
    interaction = data.z * centered
    weight = np.ones(data.sets) if set_weight is None else np.asarray(set_weight, float)
    row_weight = weight[data.set]
    total = float(weight.sum())

    def objective(theta):
        b_page, b_int, gamma = theta[0], theta[1], np.r_[0.0, theta[2:]]
        utility = b_page * data.z + b_int * interaction + gamma[data.position]
        peak = np.full(data.sets, -np.inf)
        np.maximum.at(peak, data.set, utility)
        expo = np.exp(utility - peak[data.set])
        denominator = np.bincount(data.set, expo, minlength=data.sets)
        probability = expo / denominator[data.set]
        log_norm = peak + np.log(denominator)
        loss = -(np.sum(weight * (np.bincount(data.set, utility * data.chosen, minlength=data.sets) - log_norm)))
        residual = (data.chosen - probability) * row_weight
        gradient_gamma = -np.bincount(data.position, residual, minlength=data.levels)[1:]
        gradient = np.r_[-np.dot(residual, data.z), -np.dot(residual, interaction), gradient_gamma]
        loss += 0.5 * ridge * np.dot(theta, theta) * total
        gradient += ridge * theta * total
        return loss / total, gradient / total

    start = np.zeros(1 + data.levels) if start is None else np.asarray(start, float)
    result = minimize(objective, start, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Plackett-Luce fit did not converge: {result.message}")
    return {"page_z": float(result.x[0]), "page_z_x_prompt_axis": float(result.x[1]),
            "position_effects": [0.0, *map(float, result.x[2:])],
            "theta": [float(v) for v in result.x],
            "mean_negative_log_likelihood": float(result.fun), "choice_sets": data.sets,
            "alternatives": int(len(data.z))}


_SHARED: dict = {}


def _replicate(arguments):
    """One refit in a forked worker; the choice data is inherited, not pickled."""

    kind, value = arguments
    data, start = _SHARED["data"], _SHARED["start"]
    if kind == "weight":
        return fit_plackett_luce(data, set_weight=value, start=start)
    return fit_plackett_luce(data, axis=value, start=start)


def _map(data, start, jobs, workers):
    if workers <= 1:
        _SHARED.update(data=data, start=start)
        return [_replicate(job) for job in jobs]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    _SHARED.update(data=data, start=start)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(_replicate, jobs, chunksize=1))


def keyword_bootstrap(data: ChoiceData, *, replicates: int, seed: int,
                      start: np.ndarray | None = None, workers: int = 1) -> dict:
    """Resample keywords (clusters) with replacement via choice-set weights."""

    keywords = np.unique(data.keywords)
    set_keyword = np.searchsorted(keywords, data.keywords[data.set_generation])
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(keywords), len(keywords)), minlength=len(keywords))
        weight = counts[set_keyword].astype(float)
        if weight.any():
            jobs.append(("weight", weight))
    fits = _map(data, start, jobs, workers)
    return {key: {"replicates": len(fits),
                  "ci95": [float(np.quantile([f[key] for f in fits], 0.025)),
                           float(np.quantile([f[key] for f in fits], 0.975))]}
            for key in ("page_z", "page_z_x_prompt_axis")}


def within_keyword_permutation(data: ChoiceData, *, observed: float, replicates: int, seed: int,
                               start: np.ndarray | None = None, workers: int = 1) -> dict:
    """Null for the interaction: shuffle prompt axis values among prompts of one keyword."""

    rng = np.random.default_rng(seed)
    prompt_axis = {}
    for prompt, keyword, value in zip(data.prompts, data.keywords, data.axis):
        prompt_axis.setdefault((keyword, prompt), value)
    by_keyword = defaultdict(list)
    for (keyword, prompt), value in prompt_axis.items():
        by_keyword[keyword].append((prompt, value))
    jobs = []
    for _ in range(replicates):
        shuffled = {}
        for keyword, items in by_keyword.items():
            values = rng.permutation([v for _, v in items])
            shuffled.update({(keyword, p): v for (p, _), v in zip(items, values)})
        jobs.append(("axis", np.asarray([shuffled[(k, p)] for k, p in zip(data.keywords, data.prompts)])))
    null = np.asarray([f["page_z_x_prompt_axis"] for f in _map(data, start, jobs, workers)])
    exceed = int(np.sum(np.abs(null) >= abs(observed)))
    return {"replicates": replicates, "two_sided_p": (1 + exceed) / (replicates + 1),
            "null_q025": float(np.quantile(null, 0.025)), "null_q975": float(np.quantile(null, 0.975))}


def binned_contrasts(observations, statistics, *, field: str, bins: int = 10,
                     replicates: int = 200, seed: int = 20261004) -> list[dict]:
    """Mean within-answer contrast by prompt-axis bin with keyword-cluster bootstrap CIs."""

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(o["keyword"], s[field]) for o, s in zip(observations, statistics)
                   if s.get(field) is not None and (low <= o["prompt_axis"] < high or (index == bins - 1 and o["prompt_axis"] == 1.0))]
        row = {"bin": index, "prompt_axis_low": low, "prompt_axis_high": high, "answers": len(members)}
        if members:
            values = np.asarray([v for _, v in members], float)
            row["mean"] = float(values.mean())
            groups = defaultdict(list)
            for keyword, value in members:
                groups[keyword].append(value)
            sums = np.asarray([sum(v) for v in groups.values()])
            sizes = np.asarray([len(v) for v in groups.values()])
            means = []
            for _ in range(replicates):
                pick = rng.integers(0, len(sums), len(sums))
                means.append(sums[pick].sum() / sizes[pick].sum())
            row["ci95"] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
        rows.append(row)
    return rows


def page_scale_diagnostics(page_z: Mapping[str, float], percentile: Mapping[str, float],
                           outside: Mapping[str, bool], observations) -> dict:
    """How pages sit on the prompt scale, and how much of their variation is within an answer."""

    values = np.asarray(list(page_z.values()), float)
    within = [np.var([page_z[p] for p in o["presented"]]) for o in observations if len(o["presented"]) > 1]
    total = float(np.var(values)) if len(values) > 1 else float("nan")
    return {"pages": int(len(values)), "share_outside_prompt_range": float(np.mean(list(outside.values()))),
            "page_percentile_quantiles": {str(q): float(np.quantile(list(percentile.values()), q))
                                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "page_z_variance": total,
            "mean_within_answer_variance_z": float(np.mean(within)) if within else None,
            "warning": "Axis fitted on prompts; page coordinates are out-of-domain descriptions."}
