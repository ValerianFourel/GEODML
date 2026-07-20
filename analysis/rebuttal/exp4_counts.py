#!/usr/bin/env python3
"""Experiment 4 — small counts: llms.txt domain-level prevalence and the
T5 (topical competence) effect-size conversion / scaling check."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, get_pool, published_specB

t0 = time.time()
MAIN = Path.home() / "geodml_data/data/main"
reg = pd.read_parquet(MAIN / "regression_dataset.parquet",
                      columns=["domain", "url", "has_llms_txt"])

# domain-level count (a domain counts as llms.txt-positive if any row says so)
dom = reg.groupby("domain")["has_llms_txt"].max()
n_domains = len(dom)
n_pos = int((dom > 0).sum())
print(f"unique domains in regression_dataset: {n_domains:,}")
print(f"domains with /llms.txt: {n_pos:,} ({n_pos/n_domains*100:.2f}%)")
# also strict: every row positive (should match .max for a domain-level flag)
dom_min = reg.groupby("domain")["has_llms_txt"].min()
inconsistent = int(((dom > 0) & (dom_min == 0)).sum())
print(f"domains with inconsistent flag across rows: {inconsistent}")

pd.DataFrame({
    "unique_domains": [n_domains],
    "domains_with_llms_txt": [n_pos],
    "pct": [n_pos / n_domains * 100],
    "inconsistent_flag_domains": [inconsistent],
}).to_csv(OUT / "exp4_llms_txt_domains.csv", index=False)

# ── T5 scaling + effect-size conversion ────────────────────────────────────
pool = get_pool()
t5 = pool["treat_topical_comp"].dropna()
print("\nT5 (treat_topical_comp) distribution in the admission frame:")
desc = t5.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print(desc.to_string())
print(f"SD = {t5.std():.4f}")

base_rate = pool["selected"].mean()
pub = published_specB()
th = pub[(pub.treatment == "treat_topical_comp") & (pub.outcome == "selected")]
theta = float(th["coef"].iloc[0]); se = float(th["se"].iloc[0])
rel_per_unit = theta / base_rate
rel_per_sd = theta * t5.std() / base_rate
print(f"\nbase admission rate = {base_rate*100:.2f}%")
print(f"theta(T5, admission) published = {theta:+.4f} ({se:.4f})")
print(f"relative admission change per +1.0 unit of T5: {rel_per_unit*100:+.2f}%")
print(f"relative admission change per +1 SD ({t5.std():.3f}): {rel_per_sd*100:+.2f}%")

pd.DataFrame({
    "theta_T5_admission": [theta], "se": [se],
    "base_admission_rate": [base_rate],
    "t5_min": [t5.min()], "t5_max": [t5.max()],
    "t5_mean": [t5.mean()], "t5_sd": [t5.std()],
    "rel_change_per_unit_pct": [rel_per_unit * 100],
    "rel_change_per_sd_pct": [rel_per_sd * 100],
}).to_csv(OUT / "exp4_t5_effect_size.csv", index=False)
print(f"[exp4] total {time.time()-t0:.0f}s")
