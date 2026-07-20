#!/usr/bin/env python3
"""Step 0 — reproduce the headline (Spec B POOLED, 18 treatment-outcome pairs)
and diff against the published Table 2 parquet."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, TREATMENTS, OUTCOMES, get_pool, get_admitted,
                    specB_fit, published_specB)

t_start = time.time()
pool = get_pool()
admitted = get_admitted()
print(f"[step0] pool={len(pool):,} rows, admitted={len(admitted):,} rows "
      f"({time.time()-t_start:.0f}s to build/load)", flush=True)

rows = []
for outcome, is_clf in OUTCOMES:
    df = pool if outcome == "selected" else admitted
    for tr in TREATMENTS:
        r = specB_fit(df, tr, outcome, is_clf)
        rows.append(r)
        print(f"  [{outcome:10s}] {tr:30s} n={r['n']:>6} "
              f"coef={r['coef']:+.4f} se={r['se']:.4f} p={r['p_val']:.2e} "
              f"({r['seconds']}s)", flush=True)

rep = pd.DataFrame(rows)
pub = published_specB().rename(columns={
    "coef": "coef_pub", "se": "se_pub", "p_val": "p_pub", "n": "n_pub"})
m = rep.merge(pub[["treatment", "outcome", "coef_pub", "se_pub", "p_pub", "n_pub"]],
              on=["treatment", "outcome"], how="left")
m["d_coef"] = (m.coef - m.coef_pub).abs()
m["d_se"] = (m.se - m.se_pub).abs()
m["dev_in_se"] = m.d_coef / m.se_pub
m["sign_flip"] = (m.coef * m.coef_pub) < 0

m.to_csv(OUT / "step0_reproduction_diff.csv", index=False)
print("\n=== Step 0 reproduction diff ===")
print(m[["outcome", "treatment", "n", "n_pub", "coef", "coef_pub",
         "d_coef", "se", "se_pub", "dev_in_se", "sign_flip"]].to_string(index=False))
print(f"\nmax |Δcoef| = {m.d_coef.max():.6f}")
print(f"max deviation in units of published SE = {m.dev_in_se.max():.4f}")
print(f"any sign flip: {bool(m.sign_flip.any())}")
print(f"GATE {'PASSED' if (m.dev_in_se.max() < 1.0 and not m.sign_flip.any()) else 'FAILED'}")
print(f"[step0] total {time.time()-t_start:.0f}s")
