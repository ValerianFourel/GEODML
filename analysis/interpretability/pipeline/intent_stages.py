"""Where intent surfaces: prompt axis → queries → retrieved → candidates → shortlist → ranking → answer.

Protocol: ``analysis/docs/intent_surfacing_study.md``. Every stage value is a consensus axis-1 z
per answer; estimands are within-keyword slopes on the prompt position ``x``. All slopes share the
same keyword resamples and the same within-keyword shuffles of ``x``, so increments, shares and
contrasts get valid intervals. Observational throughout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .geo_drivers import DriverRows, _interval, _p_value, within_keyword_slope

PARALLEL, REACTIVE = "Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"
CHAIN = ("R", "C", "P", "K")  # retrieved, reranker candidates, shortlist, ranking
STAGES = ("Q", "R0", "Rk", "R", "C", "P", "K", "A")
INCREMENTS = ("C-R", "P-C", "K-P", "R-R0")


# ---------------------------------------------------------------- traces

def rows_digest(rows) -> str:
    """Identity of an ordered list of (url, title, text) search rows."""
    return hashlib.sha256(json.dumps([list(r[:3]) for r in rows], ensure_ascii=False).encode()).hexdigest()


def trace_stages(trace: dict, method: str, presented_urls: list[str]) -> dict:
    """The AI's queries, every retrieved row, each reranker event and consistency checks of one trace.

    Rows are (url, title, text). An event holds its scored candidates as (url, title, text, score,
    selection order), the order being −1 for candidates the reranker did not keep."""
    events = [e for e in trace["events"] if isinstance(e, dict)]
    searches = [e["payload"] for e in events if e.get("event_type") == "search"]
    compactions = [e["payload"] for e in events if e.get("event_type") == "compaction"]
    retrieved = [(r["url"], r["title"], r["text"]) for s in searches for r in s["snippets"]]
    parsed, top_k_rule = [], True
    for c in compactions:
        scored = c["scored_snippets"]
        best = sorted(range(len(scored)), key=lambda i: (-float(scored[i]["score"]), scored[i]["source_index"]))
        kept = [r["source_index"] for r in c["selected_snippets"]]
        top_k_rule &= (kept == [scored[i]["source_index"] for i in best[:c["top_k"]]]
                       and [r["source_index"] for r in scored] == list(range(len(scored))))
        order = {source: j for j, source in enumerate(kept)}
        parsed.append({"query": c["query"], "rows": [(r["url"], r["title"], r["text"], float(r["score"]),
                                                      order.get(r["source_index"], -1)) for r in scored]})
    shortlist, seen = [], set()
    for c in (compactions if method == REACTIVE else compactions[-1:]):
        for r in c["selected_snippets"]:
            if r["url"] not in seen:
                seen.add(r["url"])
                shortlist.append(r["url"])
    retrieved_urls = {r[0] for r in retrieved}
    raw = [s.get("raw_payload") or {} for s in searches]
    return {"queries": [s["query"] for s in searches], "retrieved": retrieved, "events": parsed,
            "searches": [(s["engine"], s["query"], rows_digest([(r["url"], r["title"], r["text"]) for r in s["snippets"]]))
                         for s in searches],
            "snapshots": {(s["engine"], p.get("snapshot"), p.get("snapshot_sha256")) for s, p in zip(searches, raw)},
            "checks": {"selection_is_top_k_by_score": top_k_rule,
                       "shortlist_equals_presented": shortlist == list(presented_urls),
                       "candidates_within_retrieved": all(r[0] in retrieved_urls for e in parsed for r in e["rows"]),
                       "one_compaction_if_parallel": method != PARALLEL or len(compactions) == 1}}


# ---------------------------------------------------------------- arrays

@dataclass
class Stages:
    """Per answer: codes and prompt position; CSR lists of corpus rows per stage."""

    model: np.ndarray
    engine: np.ndarray
    method: np.ndarray
    condition: np.ndarray
    keyword: np.ndarray
    prompt: np.ndarray
    x: np.ndarray
    ret_offsets: np.ndarray   # distinct pages any of the AI's searches returned
    ret_docs: np.ndarray
    cand_offsets: np.ndarray  # distinct pages the reranker scored
    cand_docs: np.ndarray
    pool_offsets: np.ndarray  # presented pages, presented order
    pool_docs: np.ndarray
    rank_offsets: np.ndarray  # ranked pages, rank order
    rank_docs: np.ndarray
    q_offsets: np.ndarray     # the AI's queries (rows of the query table)
    q_ids: np.ndarray


@dataclass
class Events:
    """Reranker compaction events, grouped by answer; scored candidates grouped by event."""

    answer: np.ndarray        # answer of each event
    row_offsets: np.ndarray   # scored candidates of each event
    doc: np.ndarray
    score: np.ndarray
    selected: np.ndarray      # selection order within the event (0 = best score), −1 if not kept


def owner(offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def csr_mean(offsets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of each answer's entries; NaN for answers without entries or with a NaN entry."""
    n = len(offsets) - 1
    who = owner(offsets)
    count = np.bincount(who, minlength=n).astype(float)
    total = np.bincount(who, np.nan_to_num(values, nan=0.0), minlength=n)
    mean = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    mean[np.bincount(who, np.isnan(values).astype(float), minlength=n) > 0] = np.nan
    return mean


def top_weighted(offsets: np.ndarray, values: np.ndarray, sort_key: np.ndarray | None = None,
                 length: np.ndarray | None = None) -> np.ndarray:
    """Rank-weighted mean (weights 1/log2(r+1) for rank r = 1, 2, ...) of each answer's entries.
    With ``sort_key`` entries are first ordered ascending by it within the answer (ties keep the
    stored order); ``length`` keeps each answer's first L."""
    n = len(offsets) - 1
    who = owner(offsets)
    order = np.arange(len(values)) if sort_key is None else np.lexsort((sort_key, who))
    who_sorted, values_sorted = who[order], values[order]
    rank = np.arange(len(values)) - offsets[:-1][who_sorted]
    keep = rank < (length[who_sorted] if length is not None else np.inf)
    weight = np.where(keep, 1.0 / np.log2(rank + 2.0), 0.0)
    total = np.bincount(who_sorted, weight, minlength=n)
    return np.divide(np.bincount(who_sorted, weight * values_sorted, minlength=n), total,
                     out=np.full(n, np.nan), where=total > 0)


def stage_values(st: Stages, doc_z: np.ndarray, *, query_z=None, answer_z=None, r0=None, rk=None) -> dict:
    """Per answer value of every available stage (NaN where an answer lacks it)."""
    values = {"R": csr_mean(st.ret_offsets, doc_z[st.ret_docs]),
              "C": csr_mean(st.cand_offsets, doc_z[st.cand_docs]),
              "P": csr_mean(st.pool_offsets, doc_z[st.pool_docs]),
              "K": top_weighted(st.rank_offsets, doc_z[st.rank_docs])}
    if query_z is not None:
        values["Q"] = csr_mean(st.q_offsets, np.asarray(query_z, float)[st.q_ids])
    for name, array in (("A", answer_z), ("R0", r0), ("Rk", rk)):
        if array is not None:
            values[name] = np.asarray(array, float)
    return values


def oracle_values(st: Stages, doc_z: np.ndarray, doc_u: np.ndarray, pool_topic: np.ndarray) -> dict:
    """Ranking intent each answer would get from its own pool and ranking length under fixed orders:
    presented order, closest to the prompt on the axis first, highest topic similarity first."""
    length = np.diff(st.rank_offsets)
    z = doc_z[st.pool_docs]
    gap = np.abs(doc_u[st.pool_docs] - st.x[owner(st.pool_offsets)])
    return {"presented": top_weighted(st.pool_offsets, z, length=length),
            "intent_oracle": top_weighted(st.pool_offsets, z, sort_key=gap, length=length),
            "topic_oracle": top_weighted(st.pool_offsets, z, sort_key=-np.asarray(pool_topic, float), length=length)}


def term_values(values: dict, term: str) -> np.ndarray:
    """A stage ('P') or an increment between two stages ('K-P')."""
    if "-" in term:
        left, right = term.split("-")
        return values[left] - values[right]
    return values[term]


def available_terms(values: dict) -> list[str]:
    return [t for t in (*STAGES, *INCREMENTS) if all(s in values for s in t.split("-"))]


# ---------------------------------------------------------------- inference

def keyword_draws(keyword_count: int, replicates: int, seed: int) -> list[np.ndarray]:
    """Keyword resample counts (indexed by keyword code), shared by every stratum and term."""
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, keyword_count, keyword_count), minlength=keyword_count).astype(float)
            for _ in range(replicates)]


def prompt_table(st: Stages) -> tuple[np.ndarray, np.ndarray]:
    """Prompt position and keyword code, indexed by prompt code."""
    size = int(st.prompt.max()) + 1
    x, keyword = np.full(size, np.nan), np.full(size, -1, np.int64)
    x[st.prompt], keyword[st.prompt] = st.x, st.keyword
    return x, keyword


def shuffle_draws(prompt_x: np.ndarray, prompt_keyword: np.ndarray, replicates: int, seed: int) -> list[np.ndarray]:
    """Shuffles of the prompt positions among the prompts of each keyword (indexed by prompt code)."""
    rng = np.random.default_rng(seed)
    grouped = np.lexsort((np.arange(len(prompt_x)), prompt_keyword))
    draws = []
    for _ in range(replicates):
        shuffled = np.empty_like(prompt_x)
        shuffled[grouped] = prompt_x[np.lexsort((rng.random(len(prompt_x)), prompt_keyword))]
        draws.append(shuffled)
    return draws


def mediation(y, mediator, x, keyword, weight=None) -> tuple[float, float]:
    """Within-keyword regression of y on x and the mediator: (direct coefficient on x, coefficient on mediator)."""
    w = np.ones(len(x)) if weight is None else np.asarray(weight, float)
    size = int(keyword.max()) + 1
    w_sum = np.maximum(np.bincount(keyword, w, minlength=size), 1e-300)

    def centered(v):
        return v - (np.bincount(keyword, w * v, minlength=size) / w_sum)[keyword]

    xt, mt, yt = centered(x), centered(mediator), centered(y)
    a = np.array([[np.sum(w * xt * xt), np.sum(w * xt * mt)], [np.sum(w * xt * mt), np.sum(w * mt * mt)]])
    try:
        direct, via = np.linalg.solve(a, [np.sum(w * xt * yt), np.sum(w * mt * yt)])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    return float(direct), float(via)


def _ratio(a, b):
    return a / b if b != 0 else float("nan")


def analyse_stratum(st: Stages, mask: np.ndarray, values: dict, oracles: dict, *,
                    draws: list[np.ndarray], shuffles: list[np.ndarray]) -> dict:
    """Stage and increment slopes (keyword-bootstrap interval, within-keyword permutation p) and the
    chain quantities: shares, pool versus reordering, mediation, oracles and utilization, the
    replay split and the answer stage, each with keyword-bootstrap intervals."""
    keyword, prompt, x = st.keyword[mask], st.prompt[mask], st.x[mask]
    series = {t: term_values(values, t)[mask] for t in available_terms(values)}
    series.update({f"oracle_{name}": v[mask] for name, v in oracles.items()})
    ok = {name: np.isfinite(v) for name, v in series.items()}
    sources = {**{n: v[mask] for n, v in values.items()}, **{f"oracle_{n}": v[mask] for n, v in oracles.items()}}
    finite = lambda *names: np.logical_and.reduce([np.isfinite(sources[n]) for n in names])
    blocks = {"chain": finite(*CHAIN)}
    if "R0" in values:
        blocks["replay"] = blocks["chain"] & finite("R0")
    if "A" in values:
        blocks["answer"] = finite("A", "K")

    def slopes(xs, w):
        return {n: within_keyword_slope(xs[ok[n]], series[n][ok[n]], keyword[ok[n]], None if w is None else w[ok[n]])
                for n in series if ok[n].any()}

    def derived(xs, w):
        out = {}

        def b(name, block, regressor=None):
            m = blocks[block]
            return within_keyword_slope((xs if regressor is None else sources[regressor])[m], sources[name][m],
                                        keyword[m], None if w is None else w[m])

        if blocks["chain"].any():
            m = blocks["chain"]
            r, c, p, k = (b(s, "chain") for s in CHAIN)
            steps = {"retrieval": r, "deduplication_and_condition": c - r, "reranker": p - c, "reordering": k - p}
            out.update({f"increment:{s}": v for s, v in steps.items()})
            out.update({f"share_of_ranking:{s}": _ratio(v, k) for s, v in steps.items()})
            out.update({f"share_of_pool:{s}": _ratio(v, p) for s, v in steps.items() if s != "reordering"})
            out["share_of_ranking:pool"] = _ratio(p, k)
            out["pool_minus_reordering"] = p - (k - p)
            direct, _ = mediation(sources["K"][m], sources["P"][m], xs[m], keyword[m], None if w is None else w[m])
            out["mediated_by_pool"] = 1.0 - _ratio(direct, k)
            intent, topic, presented = (b(f"oracle_{n}", "chain") for n in ("intent_oracle", "topic_oracle", "presented"))
            out.update({"oracle_reordering:intent": intent - p, "oracle_reordering:topic": topic - p,
                        "oracle_reordering:presented_order": presented - p, "utilization": _ratio(k - p, intent - p)})
        if "replay" in blocks and blocks["replay"].any():
            r0, r, p = (b(s, "replay") for s in ("R0", "R", "P"))
            out.update({"query_rewriting": r - r0, "share_of_pool:prompt_words": _ratio(r0, p),
                        "share_of_pool:query_rewriting": _ratio(r - r0, p)})
        if "answer" in blocks and blocks["answer"].any():
            m = blocks["answer"]
            a = b("A", "answer")
            direct, _ = mediation(sources["A"][m], sources["K"][m], xs[m], keyword[m], None if w is None else w[m])
            out.update({"answer_on_ranking": b("A", "answer", regressor="K"),
                        "answer_mediated_by_ranking": 1.0 - _ratio(direct, a)})
        return out

    observed, observed_derived = slopes(x, None), derived(x, None)
    boot, boot_derived = defaultdict(list), defaultdict(list)
    for counts in draws:
        w = counts[keyword]
        for name, value in slopes(x, w).items():
            boot[name].append(value)
        for name, value in derived(x, w).items():
            boot_derived[name].append(value)
    null = defaultdict(list)
    for shuffled in shuffles:
        for name, value in slopes(shuffled[prompt], None).items():
            null[name].append(value)
    return {"answers": int(mask.sum()), "keywords": int(len(np.unique(keyword))),
            "block_answers": {name: int(m.sum()) for name, m in blocks.items()},
            "slopes": {name: {"slope": value, "ci95": _interval(boot[name]),
                              "permutation_p": _p_value(null[name], value) if shuffles else None,
                              "answers": int(ok[name].sum())} for name, value in observed.items()},
            "derived": {name: {"estimate": float(value), "ci95": _interval(boot_derived[name])}
                        for name, value in observed_derived.items()}}


def contrast(st: Stages, mask_a: np.ndarray, mask_b: np.ndarray, values: dict, terms, draws) -> dict:
    """Slope differences (a − b) under the shared keyword resamples."""
    out = {}
    for term in terms:
        series = term_values(values, term)
        parts = [(st.x[m], series[m], st.keyword[m]) for m in (mask_a & np.isfinite(series), mask_b & np.isfinite(series))]
        if not all(len(p[0]) for p in parts):
            continue

        def difference(counts):
            a, b = (within_keyword_slope(x, y, k, None if counts is None else counts[k]) for x, y, k in parts)
            return a - b

        out[term] = {"difference": difference(None), "ci95": _interval([difference(c) for c in draws])}
    return out


def dispersion(st: Stages, doc_z: np.ndarray, mask: np.ndarray) -> dict:
    """Variance of page intent over presented pages, split exactly into within pool, between pools
    of the same keyword, and between keywords (entries weighted equally)."""
    who = owner(st.pool_offsets)
    keep = mask[who]
    z = doc_z[st.pool_docs][keep]
    pool_mean = csr_mean(st.pool_offsets, doc_z[st.pool_docs])[who][keep]
    keyword = st.keyword[who][keep]
    keyword_mean = (np.bincount(keyword, z) / np.maximum(np.bincount(keyword), 1))[keyword]
    total = float(np.sum((z - z.mean()) ** 2))
    parts = {"within_pool": float(np.sum((z - pool_mean) ** 2)),
             "between_pools_same_keyword": float(np.sum((pool_mean - keyword_mean) ** 2)),
             "between_keywords": float(np.sum((keyword_mean - z.mean()) ** 2))}
    return {"entries": int(len(z)), "variance": total / max(len(z), 1),
            "shares": {k: v / total if total > 0 else None for k, v in parts.items()},
            "mean_within_pool_sd": float(np.sqrt(parts["within_pool"] / max(len(z), 1)))}


# ---------------------------------------------------------------- reranker

def driver_columns(z, u, x, on_keyword, topic) -> dict:
    """The four E2 drivers, relative to the prompt."""
    return {"on_keyword": np.asarray(on_keyword, float), "topic_similarity": np.asarray(topic, float),
            "page_intent": np.asarray(z, float), "intent_alignment": -np.abs(np.asarray(u, float) - np.asarray(x, float))}


def selection_rows(event, selected, generation, *, z, u, on_keyword, topic) -> DriverRows:
    """Top-k Plackett–Luce rows of the reranker's shortlist: in each event the kept candidates are
    picked in score order, each from the candidates not yet picked. No position effects."""
    _, event = np.unique(event, return_inverse=True)
    kept = np.bincount(event, selected >= 0).astype(np.int64)
    repeat = np.where(selected >= 0, selected + 1, kept[event])
    row = np.repeat(np.arange(len(event)), repeat)
    step = np.arange(len(row)) - np.repeat(np.cumsum(repeat) - repeat, repeat)
    choice = (np.cumsum(kept) - kept)[event[row]] + step
    order = np.argsort(choice, kind="stable")
    row, step, choice = row[order], step[order], choice[order]
    return DriverRows(np.asarray(z, float)[row], np.asarray(u, float)[row], np.asarray(on_keyword, float)[row],
                      np.asarray(topic, float)[row], np.zeros(len(row), np.int64), choice.astype(np.int64),
                      np.asarray(generation)[row], selected[row] == step, int(kept.sum()), 1)


def score_drivers(event, score, features: dict, keyword, draws) -> dict:
    """Within compaction event, least squares of the reranker score on standardized drivers. All rows
    of an event share a keyword, so keyword-resample weights leave the within-event centering
    unchanged and each replicate re-weights per-keyword cross products."""
    names = list(features)
    _, event = np.unique(event, return_inverse=True)
    count = np.bincount(event).astype(float)
    centered = lambda v: np.asarray(v, float) - (np.bincount(event, np.asarray(v, float)) / count)[event]
    columns, y = [centered(features[n]) for n in names], centered(score)
    codes, local = np.unique(keyword, return_inverse=True)
    gram, cross = np.empty((len(codes), len(names), len(names))), np.empty((len(codes), len(names)))
    for i, left in enumerate(columns):
        cross[:, i] = np.bincount(local, left * y, minlength=len(codes))
        for j in range(i, len(names)):
            gram[:, i, j] = gram[:, j, i] = np.bincount(local, left * columns[j], minlength=len(codes))
    yy = np.bincount(local, y * y, minlength=len(codes))

    def solve(w):
        a, c = np.tensordot(w, gram, 1), w @ cross
        beta = np.linalg.lstsq(a, c, rcond=None)[0]
        return beta, 1.0 - (w @ yy - beta @ c) / (w @ yy)

    beta, r2 = solve(np.ones(len(codes)))
    boot = np.asarray([solve(d[codes])[0] for d in draws]) if draws else np.empty((0, len(names)))
    sd = float(np.sqrt(np.mean(y * y)))
    return {"rows": int(len(y)), "events": int(len(count)), "within_event_r2": float(r2), "within_event_score_sd": sd,
            "drivers": {n: {"score_per_sd": float(beta[j]), "score_sd_per_sd": float(beta[j] / sd) if sd > 0 else None,
                            "ci95": _interval(boot[:, j]) if len(boot) else [None, None]} for j, n in enumerate(names)}}
