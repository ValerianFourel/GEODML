#!/usr/bin/env python3
"""What the generator keeps and how it orders it (Mac, CPU; exploratory).

Two decisions of the generator over the snippets it was shown; the reranker's shortlist is taken as given.

* keep: which shown snippets the answer cites at all (0/1). Conditional logit within each answer given how
  many it kept (answer fixed effects; answers that keep all or none carry no information). The shown slot
  enters as dummies.
* order: among the kept snippets, which comes first. Plackett–Luce over the kept snippets, shown-slot fixed
  effects.

Both use the main-specification features of funnel_study.py (natural condition, the analysis split) and
the same keyword draws and within-keyword shuffles as funnel_study.py analyze: odds ratios per SD with 95%
keyword-bootstrap intervals and permutation p for the x-dependent terms. How much each model explains:
McFadden's pseudo-R² = 1 − NLL/NLL₀ against a uniform choice (keep: any m of the n shown; order: any order
of the kept), and each feature's Pratt share β_j · Cov(x_j, η) / Var(η) of the within-answer (within-set)
variance of the fitted utility η; the shares sum to 1, a negative share marks a suppressor. Associations.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.scripts import funnel_study as study  # noqa: E402
from analysis.scripts.intent_stages_study import ResultCache  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

SLOT_CAP = 10
FIELDS = ["stratum", "decision", "feature", "block", "share", "odds_ratio_per_sd", "beta_ci95_lo", "beta_ci95_hi", "permutation_p",
          "pseudo_r2"]


def _set_mean(values: np.ndarray, choice_set: np.ndarray, sets: int) -> np.ndarray:
    return np.bincount(choice_set, values, minlength=sets) / np.bincount(choice_set, minlength=sets)


def null_nll(choice_set: np.ndarray, sets: int) -> float:
    """Mean negative log-likelihood of a uniform choice within every set."""
    return float(np.mean(np.log(np.bincount(choice_set, minlength=sets))))


def choice_nll(utility: np.ndarray, choice_set: np.ndarray, chosen: np.ndarray, sets: int) -> float:
    """Mean negative log-likelihood of a conditional logit (one chosen row per set), without any penalty."""
    peak = np.full(sets, -np.inf)
    np.maximum.at(peak, choice_set, utility)
    log_norm = peak + np.log(np.bincount(choice_set, np.exp(utility - peak[choice_set]), minlength=sets))
    return float(np.mean(log_norm - np.bincount(choice_set, utility * chosen, minlength=sets)))


def pratt_shares(X: np.ndarray, beta: np.ndarray, choice_set: np.ndarray, sets: int, extra: np.ndarray | None = None):
    """Shares of the within-set variance of η = Xβ (+ extra) carried by each column (and by ``extra``)."""
    eta = X @ beta + (0.0 if extra is None else extra)
    eta_c = eta - _set_mean(eta, choice_set, sets)[choice_set]
    variance = float(np.mean(eta_c * eta_c))
    shares = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        x_c = X[:, j] - _set_mean(X[:, j], choice_set, sets)[choice_set]
        shares[j] = beta[j] * float(np.mean(x_c * eta_c)) / variance
    rest = None
    if extra is not None:
        e_c = extra - _set_mean(extra, choice_set, sets)[choice_set]
        rest = float(np.mean(e_c * eta_c)) / variance
    return shares, rest, variance


def keep_null(group: np.ndarray, kept: np.ndarray, groups: int) -> float:
    """Mean negative log-likelihood of choosing the kept subset uniformly among all subsets of its size."""
    from math import comb, log
    n = np.bincount(group, minlength=groups)
    m = np.bincount(group, kept.astype(float), minlength=groups).astype(int)
    return float(np.mean([log(comb(int(a), int(b))) for a, b in zip(n, m)]))


def decisions(data, ans, stratum: str, stats: dict):
    """(keep stage, order stage, descriptive counts) of one stratum, built from the shown snippets."""
    from types import SimpleNamespace
    from analysis.interpretability.pipeline import funnel_models as fm
    p, rf = data.pres, data.rf
    mask = study.strata(ans)[stratum] & ans.split & (ans.condition == "natural")
    idx = np.flatnonzero(mask[p["answer"]])
    idx = idx[np.lexsort((p["slot"][idx], p["answer"][idx]))]
    answer, rank, slot = p["answer"][idx], p["rank"][idx], np.minimum(p["slot"][idx], SLOT_CAP)
    columns = {name: rf[name] for name, *_ in study.ROW_FEATURES}
    columns.update({k: rf[k] for k in study.MISSING_INDICATORS})
    columns["u"] = rf["u"]
    base = study._features("main", "K|P")

    def cols_for(rows):
        cols = {k: v[p["row"][idx[rows]]] for k, v in columns.items()}
        cols["topic_similarity"], cols["on_keyword"] = p["topic"][idx[rows]], p["on_keyword"][idx[rows]]
        return cols

    kept = rank >= 0
    adm, informative = fm.admission_data(answer, kept)
    keep_rows = np.flatnonzero(informative)
    keep_cols = cols_for(keep_rows)
    slots = sorted(set(slot[keep_rows].tolist()) - {0})
    slot_features = [fm.Feature(f"shown_slot_{s}", "slot") for s in slots]
    slot_stats = {}
    for s_, f in zip(slots, slot_features):
        keep_cols[f.name] = (slot[keep_rows] == s_).astype(float)
        sd = float(keep_cols[f.name].std())
        slot_stats[f.name] = (float(keep_cols[f.name].mean()), sd if sd > 0 else 1.0)
    keep_design = fm.Design(base + slot_features, keep_cols, {**stats, **slot_stats})
    keep_stage = fm.Stage("admission", keep_design, adm, answer[keep_rows])

    rows_out, sets, chosen, position, set_answer = [], [], [], [], []
    for a in np.unique(answer[kept]):
        members = np.flatnonzero((answer == a) & kept)
        if len(members) < 2:
            continue
        remaining = sorted(members.tolist(), key=lambda r: rank[r])
        while len(remaining) > 1:
            pick = remaining[0]
            for r in remaining:
                rows_out.append(r); sets.append(len(set_answer)); chosen.append(r == pick); position.append(int(slot[r]))
            set_answer.append(int(a))
            remaining = remaining[1:]
    rows_out = np.asarray(rows_out, np.int64)
    position = np.asarray(position, np.int64)
    order_data = SimpleNamespace(z=np.zeros(len(rows_out)), set=np.asarray(sets, np.int64), chosen=np.asarray(chosen, bool),
                                 position=position, sets=len(set_answer), levels=int(position.max()) + 1)
    order_design = fm.Design(base, cols_for(rows_out), dict(stats))
    order_stage = fm.Stage("choice", order_design, order_data, answer[rows_out], np.asarray(set_answer, np.int64))
    n = np.bincount(answer); k = np.bincount(answer, kept.astype(float)); has = n > 0
    counts = {"answers": int(has.sum()), "shown": int(n.sum()), "kept": int(k.sum()), "share_kept": float(k.sum() / n.sum()),
              "answers_keeping_all": int(np.sum(k[has] == n[has])), "answers_keeping_none": int(np.sum(k[has] == 0)),
              "keep_informative_answers": int(adm.groups), "order_choice_sets": int(order_data.sets)}
    return keep_stage, order_stage, counts


def explained(stage, result, x_row: np.ndarray) -> dict:
    """McFadden's pseudo-R² and Pratt shares of a fitted keep or order stage (x_row: x of each design row)."""
    from analysis.interpretability.pipeline import funnel_models as fm
    names = result["replicate_columns"]
    beta = np.asarray([result["features"][n]["beta_per_sd"] for n in names])
    X = stage.design.matrix(x_row)
    if stage.kind == "admission":
        d = stage.data
        nll0, nll = keep_null(d.group, d.admitted, d.groups), fm.admission_loss(beta, X, d)[0]
        shares, slot_share, variance = pratt_shares(X, beta, d.group, d.groups)
    else:
        d = stage.data
        slot = np.asarray(result.get("position_effects") or [0.0])[d.position]
        nll0, nll = null_nll(d.set, d.sets), choice_nll(X @ beta + slot, d.set, d.chosen, d.sets)
        shares, slot_share, variance = pratt_shares(X, beta, d.set, d.sets, slot)
    out = {"nll_uniform": nll0, "nll_model": float(nll), "pseudo_r2": 1 - float(nll) / nll0,
           "utility_sd_within": float(np.sqrt(variance)), "shares": {n: float(v) for n, v in zip(names, shares)}}
    if slot_share is not None:
        out["shares"]["shown_slot"] = slot_share
    blocks = {}
    for f in stage.design.features:
        blocks[f.block] = blocks.get(f.block, 0.0) + out["shares"][f.name]
    if slot_share is not None:
        blocks["slot"] = blocks.get("slot", 0.0) + slot_share
    out["block_shares"] = blocks
    return out


def build(args) -> int:
    from analysis.interpretability.pipeline import funnel_models as fm
    from analysis.interpretability.pipeline import intent_stages as stages
    fm.set_backend(args.backend)
    data = study.load_assembled(args.assembled)
    ans = study.answer_arrays(data)
    ans.split = study.split_mask(data, args.split)
    stats = study.standardisation(data, ans)
    # the same keyword draws and shuffles as funnel_study.py analyze
    draws = stages.keyword_draws(ans.keywords, args.bootstrap, args.seed)
    prompt_x, prompt_keyword = np.full(ans.prompt.max() + 1, np.nan), np.zeros(ans.prompt.max() + 1, np.int64)
    prompt_x[ans.prompt], prompt_keyword[ans.prompt] = ans.x, ans.keyword
    shuffles = [s_[ans.prompt] for s_ in stages.shuffle_draws(prompt_x, prompt_keyword, args.permutations, args.seed + 1)]
    out = {"split": args.split, "condition": "natural", "backend": args.backend, "bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
           "git_commit": readiness.git_commit(), "exploratory": args.split != "confirmation", "strata": {}}
    rows = []
    # finished decisions and checkpointed fits live in <output>.cache; a rerun continues from there (exit 4 = deadline)
    key = {"git_commit": readiness.git_commit(), "assembled": data.manifest.get("created_at"),
           "settings": [args.bootstrap, args.permutations, args.seed, args.min_units], "split": args.split}
    if args.backend != "cpu":
        key["backend"] = args.backend
    cache = ResultCache(Path(args.output), key)
    deadline = time.monotonic() + 60 * args.stop_after_minutes
    for stratum in study.strata(ans):
        if not (study.strata(ans)[stratum] & ans.split).any():
            continue
        keep_stage, order_stage, counts = decisions(data, ans, stratum, stats)
        entry = {"counts": counts}
        for name, stage in (("keep", keep_stage), ("order", order_stage)):
            units = stage.data.groups if stage.kind == "admission" else stage.data.sets
            if units < args.min_units:
                entry[name] = {"skipped": f"{units} informative answers or choice sets (minimum {args.min_units})"}
                continue
            task = f"{stratum}|{name}"
            digest = hashlib.sha256(task.encode()).hexdigest()[:16]
            if not (cache.directory / f"{digest}.json").exists() and time.monotonic() > deadline:
                print(json.dumps({"stopped_before": task, "reason": "deadline checkpoint"}), flush=True)
                return 4

            def compute(stage=stage, digest=digest):
                result = fm.estimate_blocks(stage, answer_x=ans.x, answer_keyword=ans.keyword, draws=draws, shuffles=shuffles,
                                            workers=args.workers, parts=cache.directory / f"{digest}.parts.jsonl", deadline=deadline)
                fit = explained(stage, result, ans.x[stage.row_answer])
                return {**fit, "features": result["features"], "failed_replicates": result["failed_replicates"],
                        "position_effects": result.get("position_effects")}

            try:
                entry[name] = cache.get(task, compute)
            except fm.DeadlineReached:
                print(json.dumps({"stopped_in": task, "reason": "deadline checkpoint"}), flush=True)
                return 4
            fit = entry[name]
            for f, share in fit["shares"].items():
                e = fit["features"].get(f, {})
                rows.append({"stratum": stratum, "decision": name, "feature": f, "block": e.get("block", "slot"), "share": share,
                             "odds_ratio_per_sd": e.get("odds_ratio_per_sd"), "beta_ci95_lo": (e.get("ci95") or [None, None])[0],
                             "beta_ci95_hi": (e.get("ci95") or [None, None])[1], "permutation_p": e.get("permutation_p"),
                             "pseudo_r2": fit["pseudo_r2"]})
            print(json.dumps({"stratum": stratum, "decision": name, "pseudo_r2": round(fit["pseudo_r2"], 4),
                              "blocks": {b: round(v, 3) for b, v in sorted(fit["block_shares"].items(), key=lambda t: -t[1])}}), flush=True)
        out["strata"][stratum] = entry
    target = Path(args.output)
    study.refuse_existing(target)
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "decisions.json", json.loads(json.dumps(out, default=float).replace("NaN", "null")))
    with open(partial / "decisions.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    partial.rename(target.resolve())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembled", type=Path, required=True)
    parser.add_argument("--split", choices=["confirmation", "exploration", "all"], default="exploration")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("cpu", "torch-cpu", "cuda"), default="cpu",
                        help="fits on numpy/scipy (cpu, the reference) or the PyTorch Newton backend")
    parser.add_argument("--min-units", type=int, default=30, help="fit a decision only with at least this many answers or sets")
    parser.add_argument("--stop-after-minutes", type=float, default=1e9,
                        help="after this many minutes start no new fit; finished fits are checkpointed and the command exits 4")
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
