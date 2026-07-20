#!/usr/bin/env python3
"""verify.py — read every condensed table and print/assert each paper claim.

Run from the reviewer-pack root:

    python verify.py

Exits with code 0 if all assertions pass (within a 2e-3 tolerance on
coefficient values), prints any mismatches otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"

TOL = 2e-3  # tolerance on coef/AUC equality checks

def banner(s: str) -> None:
    print(f"\n{'═' * 78}\n  {s}\n{'═' * 78}")

def claim(desc: str, actual: float, expected: float, tol: float = TOL) -> bool:
    ok = abs(actual - expected) <= tol
    tick = "PASS" if ok else "FAIL"
    print(f"  [{tick}]  {desc}\n          actual={actual:.4f}   expected={expected:.4f}   diff={abs(actual-expected):.4f}")
    return ok


def main() -> int:
    fails = 0

    # ─── Table 2 (DML Spec B POOLED) ────────────────────────────────
    banner("Table 2 — DML Spec B headline (mutually-controlled, 6 treatments)")
    t2 = pd.read_csv(TABLES / "table2_dml_headline.csv")
    print(t2.to_string(index=False))

    # Sample assertions — the paper's headline numbers
    def lookup(outcome: str, t_pretty: str) -> float:
        sub = t2[(t2.outcome == outcome) & (t2.treatment == t_pretty)]
        return float(sub.iloc[0]["coef"])

    banner("DML claim checks")
    fails += not claim("T5 topical comp.  selected", lookup("selected","T5 topical competence"),     +0.037, tol=0.005)
    fails += not claim("T2a Q-headings    selected", lookup("selected","T2a Q-headings"),            +0.016, tol=0.005)
    fails += not claim("T3 schema         selected", lookup("selected","T3 schema (JSON-LD)"),       -0.014, tol=0.005)
    fails += not claim("T6 freshness      selected", lookup("selected","T6 freshness"),              -0.005, tol=0.005)
    fails += not claim("T5 topical comp.  rank_delta", lookup("rank_delta","T5 topical competence"), -0.530, tol=0.02)
    fails += not claim("T2a Q-headings    rank_delta", lookup("rank_delta","T2a Q-headings"),        +0.136, tol=0.02)
    fails += not claim("T3 schema         post_rank",  lookup("post_rank","T3 schema (JSON-LD)"),    +0.095, tol=0.01)

    # ─── Admission probe headline ───────────────────────────────────
    banner("Admission probe — pre-commitment headline (mean pooling)")
    adm = pd.read_csv(TABLES / "admission_probe_headline.csv")
    pooled = adm[adm.pooling == "mean"].mean(numeric_only=True)
    print(adm[adm.pooling == "mean"].round(4).to_string(index=False))

    banner("Admission probe claim checks (variant-averaged)")
    fails += not claim("Layer 0 ROC AUC",        pooled["layer_0"],    0.671, tol=0.02)
    fails += not claim("Peak    ROC AUC",        pooled["auc_peak"],   0.860, tol=0.02)
    fails += not claim("L0 → peak gain",         pooled["delta_L0_to_peak"], 0.190, tol=0.03)

    # ─── Saliency headline ──────────────────────────────────────────
    banner("Saliency — Llama vs Qwen on 4 treatments")
    sal = pd.read_csv(TABLES / "saliency_summary.csv")
    print(sal.to_string(index=False))

    banner("Saliency claim checks")
    def sal_ratio(model: str, t: str) -> float:
        return float(sal[(sal.model == model) & (sal.treatment == t)].iloc[0]["saliency_ratio"])
    fails += not claim("Qwen attends to T1b stats (>>1)",  sal_ratio("Qwen-2.5-72B", "T1b_stats_density"),    1.93, tol=0.05)
    fails += not claim("Llama ~baseline on T1b (<1)",      sal_ratio("Llama-3.3-70B","T1b_stats_density"),    0.89, tol=0.05)
    fails += not claim("Llama ignores T3 schema (<<1)",    sal_ratio("Llama-3.3-70B","T3_structured_data_new"),0.19, tol=0.05)
    fails += not claim("Qwen ignores T3 schema (<<1)",     sal_ratio("Qwen-2.5-72B", "T3_structured_data_new"),0.40, tol=0.05)

    # ─── Ablation headline ──────────────────────────────────────────
    banner("Ablation — mean Δrank per (treatment, prompt) on full frame")
    abl = pd.read_csv(TABLES / "ablation_summary.csv")
    full_abl = abl[abl.frame == "full"]
    print(full_abl.to_string(index=False))

    banner("Ablation claim checks")
    def abl_mean(treatment, prompt) -> float:
        sub = full_abl[(full_abl.treatment == treatment) & (full_abl.prompt == prompt)]
        return float(sub.iloc[0]["mean_delta_r"])
    fails += not claim("T5 sign flip — biased  (promotes URL)",   abl_mean("T5_topical_comp","biased"),  -0.167, tol=0.03)
    fails += not claim("T5 sign flip — neutral (demotes URL)",    abl_mean("T5_topical_comp","neutral"), +0.038, tol=0.03)

    print()
    print(f"{'═' * 78}")
    if fails:
        print(f"   {fails} claim(s) FAILED — please inspect the printed values.")
        return 1
    print("   All paper claims VERIFIED against the tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
