#!/usr/bin/env python3
"""What explains the reranker's shortlist and the generator's ranking (Mac, CPU; exploratory).

For the shortlist (P|C) and the ranking (K|P) of each stratum, from the fitted main-specification stage
models stored by ``funnel_study.py report`` (no refitting):

* McFadden's pseudo-R² = 1 − NLL/NLL₀, with NLL₀ the uniform choice within every choice set; for the
  ranking also the R² of the presented order alone (one small fit without features).
* Each feature's share of the explained utility variance (Pratt measure): β_j · Cov(x_j, η) / Var(η),
  with x and the fitted utility η demeaned within choice sets (a conditional logit uses only within-set
  differences). The shares sum to 1 over the features and, for the ranking, the presented slot; a negative
  share marks a suppressor. Point estimates; associations only.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline.page_readiness_ordering import fit_choice_model  # noqa: E402
from analysis.scripts import funnel_study as study  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

STAGES = {"P|C": "shortlist", "K|P": "ranking"}


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


def build(args) -> int:
    results = json.loads(Path(args.results).read_text())
    data = study.load_assembled(args.assembled)
    ans = study.answer_arrays(data)
    ans.split = study.split_mask(data, results["settings"]["split"])
    stats = study.standardisation(data, ans)
    main = results["models"]["main"]
    out, rows = {"results": str(args.results), "split": results["settings"]["split"], "analysis_commit": results.get("analysis_commit"),
                 "measure": "McFadden pseudo-R2; Pratt shares of the within-set utility variance", "exploratory": True, "stages": {}}, []
    for stratum in [s for s in study.strata(ans) if s in main]:
        for stage, label in STAGES.items():
            fit = main[stratum].get(stage, {})
            task = study.stage_task(data, ans, stratum, stage, "main")
            if task is None or not fit.get("features"):
                continue
            task.design.stats = dict(stats)
            names = [f.name for f in task.design.features]
            X = task.design.matrix(ans.x[task.row_answer])
            beta = np.asarray([fit["features"][n]["beta_per_sd"] for n in names])
            d = task.data
            slot = np.asarray(fit.get("position_effects") or [0.0])[d.position] if d.levels > 1 else None
            nll0 = null_nll(d.set, d.sets)
            nll = choice_nll(X @ beta + (0.0 if slot is None else slot), d.set, d.chosen, d.sets)
            if abs(nll - fit["mean_negative_log_likelihood"]) > 2e-3:  # the stored value adds a tiny ridge term
                raise ValueError(f"{stratum} {stage}: stored fit does not reproduce ({nll} vs {fit['mean_negative_log_likelihood']})")
            entry = {"choice_sets": int(d.sets), "rows": int(len(d.set)), "nll_uniform": nll0, "nll_model": nll,
                     "pseudo_r2": 1 - nll / nll0}
            if slot is not None:
                order_only = fit_choice_model(np.zeros((len(d.set), 0)), d)
                entry["pseudo_r2_presented_order_only"] = 1 - choice_nll(np.r_[0.0, order_only.x][d.position], d.set, d.chosen, d.sets) / nll0
            shares, slot_share, variance = pratt_shares(X, beta, d.set, d.sets, slot)
            entry["utility_sd_within_sets"] = float(np.sqrt(variance))
            entry["shares"] = {n: float(v) for n, v in zip(names, shares)}
            if slot_share is not None:
                entry["shares"]["presented_slot"] = slot_share
            blocks = {}
            for n, f in zip(names, task.design.features):
                blocks[f.block] = blocks.get(f.block, 0.0) + entry["shares"][n]
            if slot_share is not None:
                blocks["slot"] = slot_share
            entry["block_shares"] = blocks
            out["stages"].setdefault(stratum, {})[label] = entry
            for n in entry["shares"]:
                e = fit["features"].get(n, {})
                rows.append({"stratum": stratum, "stage": label, "feature": n, "block": e.get("block", "slot"),
                             "share": entry["shares"][n], "odds_ratio_per_sd": e.get("odds_ratio_per_sd"),
                             "pseudo_r2": entry["pseudo_r2"]})
            print(json.dumps({"stratum": stratum, "stage": label, "pseudo_r2": round(entry["pseudo_r2"], 4),
                              "blocks": {b: round(v, 3) for b, v in sorted(blocks.items(), key=lambda t: -t[1])}}), flush=True)
    target = Path(args.output)
    study.refuse_existing(target)
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "importance.json", out)
    with open(partial / "importance.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.rename(target.resolve())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembled", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="results.json of funnel_study.py report")
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
