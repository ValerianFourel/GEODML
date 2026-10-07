"""Per-answer stage values, prompt metadata and shown-row scores from the existing funnel tables.

Inputs (all already on disk, read-only): the assembled exploration tables (``funnel_study.py assemble``),
the trace extract (``items.npz``, ``events.npz``, ``queries.jsonl.gz``), the prompt-text replay
(``replay.npz``) and the archived prompt snapshot. Page intent is ``u``, the page's percentile on the
prompt scale. Stage values per answer (PREREG.md): R0, R, C, P (means), I (shown order, first L shown,
rank weights) and K (cited order, rank weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import re

import numpy as np

CHAIN = ("R0", "R", "C", "P", "I", "K")
PARALLEL, REACTIVE = "Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"
TOKEN = re.compile(r"[a-z0-9]+")
HOME = Path.home() / "Hamburg"
DEFAULT_INPUTS = HOME / "geodml-inputs"
DEFAULT_PROMPTS = HOME / "GEODML_Unified/ARR_ACL_CycleOct2026/hf-min/snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz"


def rank_weights(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(n) + 2.0)


def tokens(text: str) -> set:
    return set(TOKEN.findall((text or "").casefold()))


def stage_values(offsets: np.ndarray, row: np.ndarray, scored: np.ndarray, presented: np.ndarray, ranked: np.ndarray,
                 u: np.ndarray) -> dict:
    """R, C, P, I, K and counts per answer from the trace items (CSR by ``offsets``).

    ``presented`` and ``ranked`` hold the 0-based shown slot and cited rank (−1 when absent). I is the
    rank-weighted mean of the first L shown rows in shown order, L the number cited."""
    n = len(offsets) - 1
    who = np.repeat(np.arange(n), np.diff(offsets))
    val = u[row]

    def mean(mask):
        c = np.bincount(who[mask], minlength=n).astype(float)
        s = np.bincount(who[mask], val[mask], minlength=n)
        return np.divide(s, c, out=np.full(n, np.nan), where=c > 0)

    shown, cited = presented >= 0, ranked >= 0
    L = np.bincount(who[cited], minlength=n)
    n_shown = np.bincount(who[shown], minlength=n)

    def weighted(position, mask):
        # rows with position < L of their answer, weight 1/log2(position+2)
        keep = mask & (position < L[who])
        w = np.where(keep, 1.0 / np.log2(np.maximum(position, 0) + 2.0), 0.0)
        tot = np.bincount(who, w, minlength=n)
        return np.divide(np.bincount(who, w * val, minlength=n), tot, out=np.full(n, np.nan), where=tot > 0)

    # positions are compacted to 0..m-1 within each answer (slots/ranks are already 0-based and dense)
    return {"R": mean(np.ones(len(row), bool)), "C": mean(scored == 1), "P": mean(shown),
            "I": weighted(presented, shown), "K": weighted(ranked, cited),
            "L": L.astype(float), "n_shown": n_shown.astype(float)}


def replay_values(prompt_ids, engines, replay_prompts: list[dict], replay_rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """R0 per answer: mean u over the 20 rows the frozen search returns for the prompt text itself."""
    index = {(p["prompt_id"], p["engine"]): i for i, p in enumerate(replay_prompts)}
    r0 = u[replay_rows].mean(axis=1)
    return np.asarray([r0[index[(p, e)]] if (p, e) in index else np.nan for p, e in zip(prompt_ids, engines)], float)


def prompt_metadata(path: Path) -> dict:
    """prompt id -> question, words, keyword_in_prompt, candidate_slot, target (normalized lattice target)."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            p = r["prompt"]
            q = p["question"]
            out[p["candidate_id"]] = {"question": q, "words": len(q.split()),
                                      "keyword_in_prompt": float(tokens(p["keyword"]) <= tokens(q)),
                                      "candidate_slot": int(p.get("candidate_slot", 0)),
                                      "target": float(p["target_normalized_axis_1"]),
                                      "target_index": int(p["target_index"])}
    return out


def shown_scores(cand: dict, event_search: np.ndarray, pres_answer: np.ndarray, pres_row: np.ndarray):
    """Reranker logit score and search block of each shown row: the first event (lowest index) of the answer
    that selected the row. Scores are sigmoid outputs; logits are clipped at 1e-6."""
    sel = cand["selected"] >= 0
    key = cand["answer"][sel].astype(np.int64) * (1 << 32) + cand["row"][sel]
    ev = cand["event"][sel]
    order = np.lexsort((ev, key))
    key, ev, score = key[order], ev[order], cand["score"][sel][order]
    first = np.r_[True, key[1:] != key[:-1]]
    key, ev, score = key[first], ev[first], score[first]
    want = pres_answer.astype(np.int64) * (1 << 32) + pres_row
    pos = np.searchsorted(key, want)
    pos = np.clip(pos, 0, len(key) - 1)
    found = key[pos] == want
    p = np.clip(score[pos], 1e-6, 1 - 1e-6)
    logit = np.where(found, np.log(p) - np.log1p(-p), np.nan)
    block = np.where(found, event_search[ev[pos]], -1)
    return logit, block


@dataclass
class Tables:
    """Per-answer arrays (natural and all conditions) plus shown-row arrays."""

    x: np.ndarray
    keyword: np.ndarray          # code
    keyword_text: np.ndarray
    prompt: np.ndarray           # code
    prompt_id: np.ndarray
    model: np.ndarray
    method: np.ndarray
    engine: np.ndarray
    condition: np.ndarray
    split: np.ndarray            # exploration keyword (addendum A1)
    values: dict                 # R0, R, C, P, I, K, L, n_shown
    meta: dict                   # words, keyword_in_prompt, candidate_slot, target, target_index (per answer)
    pres: dict = field(default_factory=dict)   # answer, row, slot, rank, topic, on_keyword, logit, block
    data: object = None          # funnel_study.load_assembled namespace (for stage models)
    ans: object = None           # funnel_study.answer_arrays namespace
    manifest: dict = field(default_factory=dict)

    def common(self, *names: str) -> np.ndarray:
        names = names or CHAIN
        return np.logical_and.reduce([np.isfinite(self.values[n]) for n in names])


def load(input_root: Path = DEFAULT_INPUTS, prompts: Path = DEFAULT_PROMPTS) -> Tables:
    from analysis.scripts import funnel_study as study
    root = Path(input_root)
    assembled, extract = root / "funnel-assembled-exploration-v1", root / "funnel-extract-exploration-v1"
    data = study.load_assembled(assembled)
    ans = study.answer_arrays(data)
    extracted = [json.loads(line) for line in gzip.open(extract / "answers.jsonl.gz", "rt")]
    if len(extracted) != len(data.answers) or any(a["fingerprint"] != b["fingerprint"] for a, b in zip(extracted, data.answers)):
        raise ValueError("extract and assembled answers are not aligned")
    items = np.load(extract / "items.npz")
    u = data.rf["u"]
    values = stage_values(items["offsets"], items["row"], items["scored"], items["presented"], items["ranked"], u)
    replay = root / "funnel-replay-v1"
    rp = [json.loads(line) for line in gzip.open(replay / "prompts.jsonl.gz", "rt")]
    prompt_ids = np.asarray([a["prompt_id"] for a in data.answers])
    values["R0"] = replay_values(prompt_ids, ans.engine, rp, np.load(replay / "replay.npz")["rows"], u)
    pm = prompt_metadata(prompts)
    meta = {k: np.asarray([pm[p][k] if p in pm else np.nan for p in prompt_ids], float)
            for k in ("words", "keyword_in_prompt", "candidate_slot", "target", "target_index")}
    pres = dict(data.pres)
    events = np.load(extract / "events.npz")
    pres["logit"], pres["block"] = shown_scores(data.cand, events["search"], pres["answer"], pres["row"])
    manifest = {"assembled": data.manifest.get("created_at"), "assembled_counts": data.manifest.get("counts"),
                "extract": json.loads((extract / "manifest.json").read_text()).get("counts"),
                "replay": json.loads((replay / "manifest.json").read_text()), "prompts": str(prompts)}
    return Tables(x=ans.x, keyword=ans.keyword, keyword_text=np.asarray([a["keyword_text"] or "" for a in data.answers]),
                  prompt=ans.prompt, prompt_id=prompt_ids, model=ans.model, method=ans.method, engine=ans.engine,
                  condition=ans.condition, split=study.split_mask(data, "exploration"), values=values, meta=meta,
                  pres=pres, data=data, ans=ans, manifest=manifest)


def strata(t: Tables, *, by_engine: bool = False) -> dict:
    """Natural-condition, exploration-split masks per model · method (· engine)."""
    base = (t.condition == "natural") & t.split
    out = {}
    for model in sorted(set(t.model)):
        for method in sorted(set(t.method)):
            m = base & (t.model == model) & (t.method == method)
            if not by_engine:
                out[f"{model} · {method.split('-')[0]}"] = m
                continue
            for engine in sorted(set(t.engine)):
                out[f"{model} · {method.split('-')[0]} · {engine}"] = m & (t.engine == engine)
    return out
