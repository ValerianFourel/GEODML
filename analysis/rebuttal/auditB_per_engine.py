#!/usr/bin/env python3
"""Audit B — per-engine DML at the headline spec (Spec B mutually controlled),
split by engine, all 6 treatments x 3 outcomes x 2 engines. The engine cell
dummy is dropped from X within each engine split (it is constant there)."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, TREATMENTS, CONFOUNDERS, OUTCOMES, get_pool,
                    get_admitted, specB_fit, ALPHA6, COL_TO_PAPER)

t_start = time.time()
pool, admitted = get_pool(), get_admitted()

rows = []
for engine in ("ddg", "searxng"):
    for outcome, is_clf in OUTCOMES:
        base = pool if outcome == "selected" else admitted
        df = base[base["search_engine"] == engine].copy()
        df = df.drop(columns=["cell_engine_searxng"], errors="ignore")
        for tr in TREATMENTS:
            r = specB_fit(df, tr, outcome, is_clf)
            r["engine"] = engine
            r["paper_label"] = COL_TO_PAPER.get(tr, tr)
            rows.append(r)
            print(f"  [{engine:7s}|{outcome:10s}] {tr:30s} n={r['n']:>6} "
                  f"coef={r['coef']:+.4f} se={r['se']:.4f} p={r['p_val']:.2e} "
                  f"({r['seconds']}s)", flush=True)

res = pd.DataFrame(rows)
res["ci_lo"] = res.coef - 1.96 * res.se
res["ci_hi"] = res.coef + 1.96 * res.se
res["sig_alpha6"] = res.p_val < ALPHA6
res.to_csv(OUT / "auditB_per_engine_specB.csv", index=False)

t2 = res[res.treatment == "treat_question_headings"]
print("\nT2 (question headings) per engine per outcome:")
print(t2[["engine", "outcome", "coef", "se", "p_val", "sig_alpha6"]]
      .to_string(index=False))
print(f"[auditB] total {time.time()-t_start:.0f}s")
