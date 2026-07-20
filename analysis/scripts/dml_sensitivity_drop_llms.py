"""Fast path: build the no-llms.txt DML parquet by

  1. dropping every T7 = has_llms_txt row from the 2026-05-24 parquet
     (Spec A T7 + Spec B T7 = 36 rows)
  2. re-fitting Spec B (mutually controlled) for T1-T6 with the
     T7 control removed; 6 treatments × 3 outcomes = 18 models
  3. concatenating the result and writing
     dml_canonical_2026-05-25_no_llms.parquet

Why this is correct: Spec A only ever used the 28 confounders, never the
other treatments, so the 198 Spec A T1-T6 rows are bit-identical with or
without T7 in the analysis. Only the 18 Spec B T1-T6 rows actually
change, because they previously controlled for T7.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as REPO, DML  # noqa: E402
sys.path.insert(0, str(REPO))

# Re-use the canonical script's data-build + fit functions.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dml_canonical",
    Path(__file__).resolve().parent / "dml_canonical.py",
)
dml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dml)

# Sanity: the script we just imported must already have T7 dropped from
# TREATMENTS (we did this earlier today).
assert "has_llms_txt" not in dml.TREATMENTS, \
    "expected has_llms_txt to be already removed from dml.TREATMENTS"
T1_T6 = list(dml.TREATMENTS)
print(f"6-treatment set: {T1_T6}")

# ---------------------------------------------------------------------------
# 1) Load the OLD parquet, drop everything that changes or goes away.
# ---------------------------------------------------------------------------
OLD = DML / "dml_canonical_2026-05-24.parquet"
NEW = DML / "dml_canonical_2026-05-25_no_llms.parquet"

old = pd.read_parquet(OLD)
print(f"\nLoaded old parquet: {len(old)} rows")
print(old.groupby(["spec", "treatment"]).size().unstack(fill_value=0))

# Drop T7 entirely
keep = old[old.treatment != "has_llms_txt"].copy()
# Drop Spec B for T1-T6 (will be re-fit below)
keep = keep[~((keep.spec == "B") & (keep.treatment.isin(T1_T6)))].copy()
print(f"\nAfter dropping T7 + old Spec B: {len(keep)} rows  "
      f"(should be {len(old) - 33 - 3 - 18} = 198 Spec A T1-T6)")
print(keep.groupby(["spec", "outcome"]).size())

# ---------------------------------------------------------------------------
# 2) Re-build the pool + admitted datasets (the expensive merge step).
# ---------------------------------------------------------------------------
print("\nBuilding pool …")
t0 = time.time()
pool = dml.build_pool_admission()
pool, _ = dml.add_cell_dummies(pool)
print(f"  pool: {len(pool):,} rows  ({time.time() - t0:.1f}s)")

print("Building admitted …")
t0 = time.time()
admitted = dml.build_admitted_sample()
admitted, _ = dml.add_cell_dummies(admitted)
print(f"  admitted: {len(admitted):,} rows  ({time.time() - t0:.1f}s)")

# ---------------------------------------------------------------------------
# 3) Re-fit Spec B for T1-T6 across the 3 outcomes (18 models).
# ---------------------------------------------------------------------------
cell_cols_pool = [c for c in pool.columns if c.startswith("cell_")]
cell_cols_adm = [c for c in admitted.columns if c.startswith("cell_")]
X_pool = dml.CONFOUNDERS + cell_cols_pool
X_adm = dml.CONFOUNDERS + cell_cols_adm

new_rows = []
jobs = [
    ("selected",   pool,     X_pool, True),
    ("rank_delta", admitted, X_adm, False),
    ("post_rank",  admitted, X_adm, False),
]

print("\nRe-fitting Spec B (T1-T6 × 3 outcomes = 18 models)")
for outcome, data, X_base, is_clf in jobs:
    print(f"\n  Y = {outcome}  (is_clf={is_clf})")
    for treatment in T1_T6:
        other = [t for t in T1_T6 if t != treatment]
        t0 = time.time()
        r = dml.plr_estimate(data, treatment, other, X_base, outcome, is_clf=is_clf)
        r.update(spec="B", slice="POOLED", treatment=treatment,
                 outcome=outcome, seconds=round(time.time() - t0, 1))
        new_rows.append(r)
        stars = ("***" if r["p_val"] < 1e-3 else
                 "**"  if r["p_val"] < 1e-2 else
                 "*"   if r["p_val"] < 5e-2 else
                 "·"   if r["p_val"] < 1e-1 else "")
        print(f"    [B/POOLED] {treatment:30s} "
              f"n={r['n']:>6} coef={r['coef']:+.4f} se={r['se']:.4f} "
              f"p={r['p_val']:.4f}{stars}  ({r['seconds']}s)  "
              f"(controls: {len(other)} other T + {len(X_base)} X)")

new_b = pd.DataFrame(new_rows)
new_b["sig"] = new_b["p_val"].apply(
    lambda p: "***" if p < 1e-3 else
              "**"  if p < 1e-2 else
              "*"   if p < 5e-2 else
              "·"   if p < 1e-1 else ""
)

# ---------------------------------------------------------------------------
# 4) Merge with the kept Spec A rows, recompute Bonferroni, save.
# ---------------------------------------------------------------------------
combined = pd.concat([keep, new_b], ignore_index=True)
# Drop the stale Bonferroni columns and recompute on the new set.
for c in ("p_bonferroni", "bonferroni_sig"):
    if c in combined.columns:
        combined = combined.drop(columns=c)

for o in combined["outcome"].unique():
    mask = combined["outcome"] == o
    n_tests = int(mask.sum())
    thresh = 0.05 / n_tests
    combined.loc[mask, "p_bonferroni"] = (
        combined.loc[mask, "p_val"] * n_tests
    ).clip(upper=1.0)
    combined.loc[mask, "bonferroni_sig"] = combined.loc[mask, "p_val"] < thresh

combined.to_parquet(NEW)
print(f"\nSaved {NEW.name}  rows={len(combined)}")
print(combined.groupby(["spec", "outcome"]).size())
print(f"\nTimestamp: {datetime.now(timezone.utc).isoformat()}")
