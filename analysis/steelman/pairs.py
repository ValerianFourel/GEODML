"""Score-matched adjacent shown links (PREREG.md, reported): two links in adjacent shown slots whose reranker
logit scores differ by less than ε are links the reranker found equally relevant. Within such pairs, does the
later slot win (cited above the earlier one, or kept while the earlier is dropped) more when it matches the
prompt's intent better? Pair-level logit of "later wins" on the later-minus-earlier differences in intent
alignment, page intent × (x − ½) and topic similarity (standardised); the intercept is the slot effect at equal
scores. Exact score ties are dropped; Reactive pairs must come from the same search block. Not a
discontinuity design: assignment is a deterministic function of the text.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from analysis.interpretability.pipeline import intent_stages as stages

from .chain import summarise
from .generator import shown_rows
from .tables import Tables, strata

EPSILONS = (0.1, 0.25)


def adjacent_pairs(answer, slot, logit, block, eps: float):
    """Index pairs (i, i+1) of rows sorted by (answer, slot) with adjacent slots, same block, 0 < |Δlogit| < eps."""
    i = np.arange(len(answer) - 1)
    ok = (answer[i] == answer[i + 1]) & (slot[i + 1] == slot[i] + 1) & (block[i] == block[i + 1])
    gap = np.abs(logit[i + 1] - logit[i])
    ok &= np.isfinite(gap) & (gap > 0) & (gap < eps)
    return i[ok], i[ok] + 1


def logit_fit(y, X, w=None) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(y)), X])
    w = np.ones(len(y)) if w is None else w

    def f(b):
        eta = X1 @ b
        ll = w * (y * eta - np.logaddexp(0, eta))
        p = 1 / (1 + np.exp(-eta))
        return -ll.sum() / w.sum(), -(X1.T @ (w * (y - p))) / w.sum()

    r = minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B")
    return r.x


def run(t: Tables, args) -> dict:
    rf = t.data.rf
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, args.bootstrap, args.seed)
    common = t.common()
    out = {"epsilons": list(EPSILONS), "strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        s = shown_rows(t, m & common)
        x = t.x[s.answer]
        feats = {"intent_alignment": -np.abs(rf["u"][s.row] - x), "intent_x_prompt": rf["page_intent_z"][s.row] * (x - 0.5),
                 "topic_similarity": s.topic}
        entry = {}
        for eps in EPSILONS:
            a, b = adjacent_pairs(s.answer, s.slot, s.logit, s.block, eps)
            res = {"pairs": int(len(a))}
            for outcome in ("order", "keep"):
                if outcome == "order":   # both cited: later cited above earlier
                    use = s.kept[a] & s.kept[b]
                    y = (s.rank[b] < s.rank[a]).astype(float)
                else:                    # exactly one cited: the later one
                    use = s.kept[a] ^ s.kept[b]
                    y = s.kept[b].astype(float)
                ia, ib, y = a[use], b[use], y[use]
                if len(y) < 100:
                    res[outcome] = {"pairs": int(len(y)), "skipped": True}
                    continue
                D = np.column_stack([feats[f][ib] - feats[f][ia] for f in feats])
                sd = D.std(axis=0)
                D = D / np.where(sd > 0, sd, 1.0)
                kw = t.keyword[s.answer[ia]]
                est = logit_fit(y, D)
                boot = np.asarray([logit_fit(y, D, d[kw]) for d in draws])
                res[outcome] = {"pairs": int(len(y)), "later_wins_share": float(y.mean()),
                                "intercept_slot_effect": summarise(est[0], boot[:, 0], None),
                                "difference_coefficients_per_sd": {f: summarise(est[j + 1], boot[:, j + 1], None)
                                                                   for j, f in enumerate(feats)}}
            entry[f"eps_{eps}"] = res
        out["strata"][name] = entry
    return out
