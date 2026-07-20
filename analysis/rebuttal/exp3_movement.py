#!/usr/bin/env python3
"""Experiment 3 — re-ranker movement statistics among admitted items,
overall and by engine x model, plus admission-rate cross-check."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, get_pool, get_admitted

t0 = time.time()
adm = get_admitted().dropna(subset=["rank_delta"]).copy()
adm["abs_delta"] = adm["rank_delta"].abs()


def stats(g):
    return pd.Series({
        "n": len(g),
        "median_abs_delta": g["abs_delta"].median(),
        "mean_abs_delta": g["abs_delta"].mean(),
        "share_moved_ge1": (g["abs_delta"] >= 1).mean(),
        "share_moved_ge3": (g["abs_delta"] >= 3).mean(),
    })


rows = [stats(adm).rename("OVERALL")]
for (e, m), g in adm.groupby(["search_engine", "llm_model"]):
    rows.append(stats(g).rename(f"{e} x {m}"))
tab = pd.DataFrame(rows)
tab.index.name = "cell"
tab.to_csv(OUT / "exp3_movement_stats.csv")
print(tab.to_string())

# admission-rate cross-check (paper: 58.4%)
pool = get_pool()
rate = pool["selected"].mean()
print(f"\nadmission rate (full pool frame, n={len(pool):,}): {rate*100:.2f}%")
by = pool.groupby(["search_engine", "llm_model"])["selected"].mean() * 100
print(by.round(2).to_string())
pd.DataFrame({"overall_admission_rate_pct": [rate * 100]}).to_csv(
    OUT / "exp3_admission_rate.csv", index=False)
print(f"[exp3] total {time.time()-t0:.0f}s")
