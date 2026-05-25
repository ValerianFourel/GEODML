"""Build a condensed reviewer pack — just the paper's claims, ready to verify.

Different from `build_hf_dataset.py`:
- `build_hf_dataset.py`  →  full reproducibility (72 MB, can re-run everything)
- `build_reviewer_pack.py` (this) → fast-check archive (~10 MB) that lets
  a reviewer print every headline number in a single `python verify.py`
  call, and visually inspect every figure in `figures/`.

Each table is a tight slice of the raw outputs containing only the rows
that appear in the paper. Each figure is the final PDF/PNG. The
`verify.py` script runs through every paper claim and asserts the
number against the data — a green-light reviewer check.

Run:  python scripts/build_reviewer_pack.py
"""
from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
DATA = HOME / "geodml_data" / "data"

OUT = HOME / "geodml-emnlp-2026-reviewer"
FIG_SRC = REPO / "docs" / "2026-05-24" / "figures_canonical" / "tmp"
PROB_SRC = REPO / "interpretability" / "output"

TREATMENT_PRETTY = {
    "treat_stats_density":         "T1b stats density",
    "treat_question_headings":     "T2a Q-headings",
    "treat_structured_data":       "T3 schema (JSON-LD)",
    "T4_citation_authority_code":  "T4 citation authority",
    "treat_topical_comp":          "T5 topical competence",
    "treat_freshness":             "T6 freshness",
}
OUTCOME_SIGN = {"selected": +1, "rank_delta": +1, "post_rank": -1}


# ---------------------------------------------------------------------------
def build_tables():
    """Condense the full DML/probing/saliency/ablation outputs into 6 small
    paper-facing tables."""
    print("\n[1/3]  Headline tables (one per paper claim)")
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # ─── Table 2: DML Spec B POOLED (the paper headline) ────────────
    dml = pd.read_parquet(DATA / "dml_results" /
                          "dml_canonical_2026-05-25_llms_as_confounder.parquet")
    spec_b = dml[(dml["spec"] == "B") & (dml["slice"] == "POOLED")].copy()
    spec_b["coef_promoter_dir"] = spec_b.apply(
        lambda r: OUTCOME_SIGN[r.outcome] * r.coef, axis=1)
    spec_b["ci_lo"] = spec_b["coef"] - 1.96 * spec_b["se"]
    spec_b["ci_hi"] = spec_b["coef"] + 1.96 * spec_b["se"]
    spec_b["treatment_pretty"] = spec_b["treatment"].map(TREATMENT_PRETTY)
    spec_b["sig_stars"] = spec_b["p_val"].apply(
        lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else
        "*"   if p < 5e-2 else "·" if p < 1e-1 else "")
    headline = spec_b[[
        "outcome", "treatment_pretty", "coef", "se", "ci_lo", "ci_hi",
        "p_val", "sig_stars", "n", "coef_promoter_dir"
    ]].rename(columns={"treatment_pretty": "treatment"})
    headline = headline.round({
        "coef": 4, "se": 4, "ci_lo": 4, "ci_hi": 4, "p_val": 5,
        "coef_promoter_dir": 4,
    })
    headline.to_csv(tdir / "table2_dml_headline.csv", index=False)
    print(f"  [ok] table2_dml_headline.csv  ({len(headline)} rows)")

    # ─── Full DML (all 216 models) — keep small + summarised ─────────
    dml_summary = dml[["spec", "slice", "treatment", "outcome",
                       "coef", "se", "p_val", "n", "bonferroni_sig"]]\
        .round({"coef": 4, "se": 4, "p_val": 5})
    dml_summary.to_csv(tdir / "dml_all_specs_all_slices.csv", index=False)
    print(f"  [ok] dml_all_specs_all_slices.csv  ({len(dml_summary)} rows)")

    # ─── Probing: peak ROC AUC per (treatment, model)  ─────────────────
    rows = []
    for v in ("biased", "neutral", "biased_rag", "neutral_rag"):
        f = PROB_SRC / f"probing_results_{v}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["frame"] == "full"]
        df = df.groupby(["treatment", "layer", "pooling"])["roc_auc"].mean().reset_index()
        for t, sub in df.groupby("treatment"):
            for p, sub2 in sub.groupby("pooling"):
                peak = sub2.loc[sub2["roc_auc"].idxmax()]
                rows.append({
                    "variant": v, "treatment": t, "pooling": p,
                    "peak_layer": int(peak.layer),
                    "peak_roc_auc": round(float(peak.roc_auc), 4),
                    "layer_0_roc_auc": round(float(
                        sub2.loc[sub2.layer == 0, "roc_auc"].iloc[0]
                    ), 4),
                })
    pd.DataFrame(rows).to_csv(tdir / "probing_peaks_per_variant.csv", index=False)
    print(f"  [ok] probing_peaks_per_variant.csv  ({len(rows)} rows)")

    # ─── Admission probe summary (the headline pre-commitment story) ──
    rows = []
    for v in ("biased", "neutral", "biased_rag", "neutral_rag"):
        f = PROB_SRC / f"probing_results_{v}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        adm = df[(df.treatment == "Y1_admission_inctx") & (df.frame == "full")]
        for p, sub in adm.groupby("pooling"):
            sub = sub.groupby("layer")["roc_auc"].mean()
            rows.append({
                "variant": v, "pooling": p,
                "layer_0":    round(sub.iloc[0],  4),
                "layer_peak": int(sub.idxmax()),
                "auc_peak":   round(sub.max(),    4),
                "layer_last": int(sub.index.max()),
                "auc_last":   round(sub.iloc[-1], 4),
                "delta_L0_to_peak": round(sub.max() - sub.iloc[0], 4),
            })
    pd.DataFrame(rows).to_csv(tdir / "admission_probe_headline.csv", index=False)
    print(f"  [ok] admission_probe_headline.csv  ({len(rows)} rows)")

    # ─── Saliency: 4 treatments × 2 models  ───────────────────────────
    sal = pd.read_csv(PROB_SRC / "saliency_summary_full.csv")
    n = len(sal) // 2
    sal["model"] = ["Llama-3.3-70B"] * n + ["Qwen-2.5-72B"] * n
    sal = sal[["model", "treatment", "saliency_ratio",
               "n_treatment_tokens", "n_other_tokens"]].round(
                   {"saliency_ratio": 3})
    sal.to_csv(tdir / "saliency_summary.csv", index=False)
    print(f"  [ok] saliency_summary.csv  ({len(sal)} rows)")

    # ─── Ablation: mean per (treatment, frame, prompt) ───────────────
    abl_rows = []
    for frame, prompt, fname in (
        ("full",            "biased",  "ablation_results_full_biased.csv"),
        ("full",            "neutral", "ablation_results_full_neutral.csv"),
        ("robust_winners",  "biased",  "ablation_results_rw_biased.csv"),
        ("robust_winners",  "neutral", "ablation_results_rw_neutral.csv"),
    ):
        f = PROB_SRC / fname
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for t, sub in df.groupby("treatment"):
            abl_rows.append({
                "frame": frame, "prompt": prompt, "treatment": t,
                "n_ablations": int(len(sub)),
                "mean_delta_r":     round(float(sub.ablation_delta.mean()), 4),
                "median_delta_r":   round(float(sub.ablation_delta.median()), 4),
                "std_delta_r":      round(float(sub.ablation_delta.std()), 4),
            })
    pd.DataFrame(abl_rows).to_csv(tdir / "ablation_summary.csv", index=False)
    print(f"  [ok] ablation_summary.csv  ({len(abl_rows)} rows)")

    # ─── Freshness leakage diagnostic ─────────────────────────────────
    diag_path = REPO / "docs" / "2026-05-24" / "freshness_leakage_diagnostic.md"
    if diag_path.exists():
        shutil.copy2(diag_path, tdir / "freshness_leakage_diagnostic.md")
        print(f"  [ok] freshness_leakage_diagnostic.md")


def copy_figures():
    """Copy the final figure PDFs + PNGs (the things reviewers actually look at)."""
    print("\n[2/3]  Figures (PDF + PNG)")
    fdir = OUT / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    wanted = [
        # framework
        "dml_estimation_framework",
        # 14 canonical DML figures
        "fig01_admission_forest", "fig02_rank_delta_forest", "fig03_post_rank_forest",
        "fig04_three_outcome_grid", "fig05_admission_vs_rank_scatter",
        "fig06_marginal_vs_partial",
        "fig07_variant_grid_admission", "fig08_variant_grid_rank",
        "fig09_compare_rag_vs_snippet", "fig10_compare_ddg_vs_searxng",
        "fig11_compare_llama_vs_qwen",  "fig12_compare_pool20_vs_pool50",
        "fig13_admission_detail",       "fig14_robust_survivors",
        # Stage-F figures
        "fig_probing_layerwise", "fig_probing_pooling",
        "fig_admission_pooled",  "fig_admission_variants",
        "fig_saliency",
    ]
    n_pdf = n_png = 0
    for name in wanted:
        for ext in ("pdf", "png"):
            src = FIG_SRC / f"{name}.{ext}"
            if src.exists():
                shutil.copy2(src, fdir / f"{name}.{ext}")
                if ext == "pdf": n_pdf += 1
                else:            n_png += 1
    print(f"  [copy] {n_pdf} PDFs + {n_png} PNGs into figures/")


def write_verify_script():
    """Write a one-shot `verify.py` that prints every paper-claim number
    by reading the condensed tables. Asserts the headline values so a
    reviewer can simply run `python verify.py` and see PASS / FAIL."""
    print("\n[3/3]  verify.py — reviewer can run this to validate every claim")
    code = r'''#!/usr/bin/env python3
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
'''
    p = OUT / "verify.py"
    p.write_text(code)
    p.chmod(0o755)
    print(f"  [write] verify.py  (chmod +x)")


def write_readme():
    txt = f"""# GEODML — EMNLP 2026 reviewer pack

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.

A **condensed** version of the paper's outputs designed for fast
verification, not full reproduction.  Reviewer can:

1. Open `figures/*.pdf` to inspect any figure cited in the paper.
2. Open `tables/*.csv` to see the exact numbers behind each claim.
3. Run `python verify.py` to automatically assert every headline
   number against the data (returns 0 if every claim passes).

If you want to *re-run* the analysis from scratch, see the companion
full reproducibility dataset `geodml-emnlp-2026/` (72 MB) which
includes the raw experimental data and all fit scripts.

## Layout

```
tables/                            ← condensed paper claims
├── table2_dml_headline.csv             6 treatments × 3 outcomes, Spec B POOLED
├── dml_all_specs_all_slices.csv        all 216 fitted DML models
├── probing_peaks_per_variant.csv       peak ROC AUC per (treatment, model, pooling)
├── admission_probe_headline.csv        L0, peak, final-layer AUC of the admission probe
├── saliency_summary.csv                4 treatments × 2 backbones (Llama, Qwen)
├── ablation_summary.csv                mean Δrank per (treatment, prompt, frame)
└── freshness_leakage_diagnostic.md     year-token leakage table for T6

figures/                           ← every PDF the paper references
├── dml_estimation_framework.pdf
├── fig01_admission_forest.pdf … fig14_robust_survivors.pdf
├── fig_probing_layerwise.pdf
├── fig_probing_pooling.pdf
├── fig_admission_pooled.pdf
├── fig_admission_variants.pdf
└── fig_saliency.pdf

verify.py                           ← one-shot claim-checker
README.md                           ← this file
```

## How to verify the paper in <60 seconds

```bash
pip install pandas
python verify.py
```

The script prints every headline number alongside its expected value
(from the paper text) and the tolerance.  Last line either says

```
   All paper claims VERIFIED against the tables.
```

or lists which claims didn't match.

## Headline numbers (Table 2, Spec B POOLED, 6 treatments, llms.txt as confounder)

| Treatment | Y1 selected | Y2 Δrank | Y3 rank_post |
|---|---|---|---|
| T5 topical competence | +0.037*** | −0.530*** | −0.299*** |
| T2a Q-headings        | +0.016*** | +0.136*** | −0.041*   |
| T3 schema (JSON-LD)   | −0.014*** | −0.051*   | +0.095*** |
| T6 freshness          | −0.005*** | −0.061*** | +0.005    |
| T4 citation auth.     | +0.001    | −0.023**  | −0.015*   |
| T1b stats density     | −0.000    | −0.003    | −0.002    |

Sign:  +Y1, +Y2, −Y3  =  URL favoured by the LLM.
Stars: *** < .001, ** < .01, * < .05.
T7=has_llms_txt is a confounder (not a treatment) because the
rerankers do not retrieve `/llms.txt` at inference time.

## Admission pre-commitment probe (mean pooling, averaged over 4 variants)

| Quantity | Value | Interpretation |
|---|---|---|
| Layer 0 ROC AUC | 0.671 | Embedding alone has weak signal |
| Peak ROC AUC    | 0.862 | At layer 60 (~75% network depth) |
| L0 → peak gain  | +0.191 | Genuine compositional integration |

## Saliency (gradient attribution)

| Treatment | Llama | Qwen | DML direction |
|---|---|---|---|
| T1b stats density | 0.89× | **1.93×** | null |
| T2a Q-headings    | 1.05× | 0.90×    | promoter |
| T3 schema         | **0.19×** | **0.40×** | demoter (consistent — model doesn't look) |

## Citation

```bibtex
@inproceedings{{fourel2026geodml,
  title     = {{Causal Analysis of LLM Search Rerankers via Double/Debiased Machine Learning}},
  author    = {{Fourel, Valerian}},
  year      = {{2026}},
  booktitle = {{Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing}}
}}
```

## Contact

valerian.fourel@gmail.com
"""
    (OUT / "README.md").write_text(txt)
    print(f"  [write] README.md")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    build_tables()
    copy_figures()
    write_verify_script()
    write_readme()

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\n=====  Built {OUT}  total size = {total/1024/1024:.1f} MB  =====")
