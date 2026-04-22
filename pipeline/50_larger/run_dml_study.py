#!/usr/bin/env python3
"""
DML Causal Inference Study — 50_larger variant

Same as root run_dml_study.py but reads from 50_larger/results/ and 50_larger/data/.
Uses the dataset with 50 SERP results and top-20 LLM re-ranking.

Model: Partially Linear Regression (PLR)
    Y = D*theta_0 + g_0(X) + zeta
    D = m_0(X) + V
where theta_0 is the causal parameter of interest.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DATA_DIR = SCRIPT_DIR / "data"

# Auto-detect input files
def _find_latest(pattern, directory):
    candidates = sorted(directory.glob(pattern), reverse=True)
    return candidates[0] if candidates else None

CSV_PATH = _find_latest("dml_dataset_*.csv", RESULTS_DIR)
JSON_PATH = _find_latest("duckduckgo_*_20*.json", RESULTS_DIR)
if not JSON_PATH:
    JSON_PATH = _find_latest("searxng_*_20*.json", RESULTS_DIR)

OUT_CSV = RESULTS_DIR / "dml_results.csv"
OUT_JSON = RESULTS_DIR / "dml_results.json"
OUT_PNG = RESULTS_DIR / "dml_coefficients.png"

# ── Treatment and confounder definitions ─────────────────────────────────────
TREATMENTS_CODE = {
    "T1_code": "T1_statistical_density",
    "T2_code": "T2_question_heading_match",
    "T3_code": "T3_structured_data",
    "T4_code": "T4_citation_authority",
}
TREATMENTS_LLM = {
    "T1_llm": "T1_llm_statistical_density",
    "T2_llm": "T2_llm_question_heading",
    "T3_llm": "T3_llm_structured_data",
    "T4_llm": "T4_llm_citation_authority",
}
TREATMENTS = {**TREATMENTS_CODE, **TREATMENTS_LLM}

CONFOUNDERS = [
    "X1_domain_authority",
    "X2_domain_age_years",
    "X3_word_count",
    "X6_readability",
    "X7_internal_links",
    "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt",
]

TREATMENT_LABELS = {
    "T1_code": "T1 Statistical Density (code)",
    "T2_code": "T2 Question Headings (code)",
    "T3_code": "T3 Structured Data (code)",
    "T4_code": "T4 Citation Authority (code)",
    "T1_llm": "T1 Statistical Density (LLM)",
    "T2_llm": "T2 Question Headings (LLM)",
    "T3_llm": "T3 Structured Data (LLM)",
    "T4_llm": "T4 Citation Authority (LLM)",
}


# ── 1. Load & merge data ────────────────────────────────────────────────────
def load_data():
    if not CSV_PATH or not CSV_PATH.exists():
        print(f"ERROR: No dml_dataset CSV found in {RESULTS_DIR}")
        print("Run the full pipeline first: run_ai_search.py → extract → scraper → build_clean_dataset.py")
        sys.exit(1)
    if not JSON_PATH or not JSON_PATH.exists():
        print(f"ERROR: No experiment JSON found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")

    df = pd.read_csv(CSV_PATH)

    with open(JSON_PATH) as f:
        experiment = json.load(f)

    rc_lookup = {}
    for kw_result in experiment["per_keyword_results"]:
        query = kw_result["query"]
        for rc in kw_result["rank_changes"]:
            rc_lookup[(query, rc["domain"])] = rc

    df["pre_rank"] = df.apply(
        lambda r: rc_lookup.get((r["keyword"], r["domain"]), {}).get("pre_rank"), axis=1
    )
    df["post_rank"] = df.apply(
        lambda r: rc_lookup.get((r["keyword"], r["domain"]), {}).get("post_rank"), axis=1
    )
    df["rank_delta"] = df.apply(
        lambda r: rc_lookup.get((r["keyword"], r["domain"]), {}).get("rank_delta"), axis=1
    )

    n_total = len(df)
    df = df.dropna(subset=["rank_delta"])
    n_valid = len(df)
    print(f"Loaded {n_total} rows, {n_valid} with valid rank_delta ({n_total - n_valid} dropped)")
    print(f"rank_delta: mean={df['rank_delta'].mean():.2f}, median={df['rank_delta'].median():.1f}, "
          f"std={df['rank_delta'].std():.2f}, range=[{df['rank_delta'].min():.0f}, {df['rank_delta'].max():.0f}]")
    return df


# ── 2. Preprocessing ────────────────────────────────────────────────────────
def preprocess(df, treatment_col):
    # Filter out confounders that are entirely null
    available_confounders = [c for c in CONFOUNDERS if df[c].notna().any()]
    dropped_confounders = [c for c in CONFOUNDERS if c not in available_confounders]

    cols_needed = available_confounders + [treatment_col, "rank_delta"]
    sub = df[cols_needed].copy()

    miss = sub.isna().sum()
    miss_pct = (miss / len(sub) * 100).round(1)
    missing_info = {c: f"{miss[c]}/{len(sub)} ({miss_pct[c]}%)" for c in cols_needed if miss[c] > 0}
    if dropped_confounders:
        missing_info["_dropped_confounders"] = ", ".join(dropped_confounders)

    sub = sub.dropna(subset=[treatment_col])
    n_after_treatment = len(sub)

    if n_after_treatment == 0:
        return None, None, None, 0, missing_info

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(sub[available_confounders]),
        columns=available_confounders,
        index=sub.index,
    )

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=available_confounders,
        index=sub.index,
    )

    Y = sub["rank_delta"].values
    D = sub[treatment_col].values

    return X_scaled, Y, D, n_after_treatment, missing_info


# ── 3. DML estimation ───────────────────────────────────────────────────────
def run_dml(X, Y, D, learner_type="lgbm", n_folds=5):
    if learner_type == "lgbm":
        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
    else:
        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)

    dml_data = dml.DoubleMLData.from_arrays(
        x=X.values,
        y=Y,
        d=D,
    )

    plr = dml.DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        score="partialling out",
    )
    plr.fit()

    coef = plr.coef[0]
    se = plr.se[0]
    t_stat = plr.t_stat[0]
    p_val = plr.pval[0]
    ci = plr.confint(level=0.95)
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]

    return {
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_val": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def interpret(treatment_name, coef, p_val):
    direction = "promotes" if coef > 0 else "demotes"
    magnitude = abs(coef)
    sig = "significantly " if p_val < 0.05 else "not significantly "
    label = TREATMENT_LABELS[treatment_name]
    return f"{label} {sig}{direction} results by {magnitude:.2f} rank positions (p={p_val:.3f})"


# ── 4. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DML Causal Inference Study: Page Features → LLM Re-Ranking (50_larger)")
    print("  50 SERP results, top-20 LLM re-ranking")
    print("=" * 70)
    print()

    df = load_data()
    print()

    results = []
    sensitivity_results = []

    for treatment_name, treatment_col in TREATMENTS.items():
        print(f"── {TREATMENT_LABELS[treatment_name]} ──")

        # Skip treatments with no data
        if treatment_col not in df.columns or df[treatment_col].notna().sum() == 0:
            print(f"   SKIPPED: {treatment_col} has no data (0% coverage)")
            print()
            continue

        X, Y, D, n_obs, missing_info = preprocess(df, treatment_col)
        print(f"   Observations: {n_obs} | Missing: {missing_info if missing_info else 'none'}")

        if X is None or n_obs < 10:
            print(f"   SKIPPED: insufficient observations ({n_obs})")
            print()
            continue

        res_lgbm = run_dml(X, Y, D, learner_type="lgbm")
        stars = significance_stars(res_lgbm["p_val"])
        print(f"   LGBM:  theta={res_lgbm['coef']:+.3f}  SE={res_lgbm['se']:.3f}  "
              f"p={res_lgbm['p_val']:.4f}{stars}  "
              f"CI=[{res_lgbm['ci_lower']:.3f}, {res_lgbm['ci_upper']:.3f}]")

        res_rf = run_dml(X, Y, D, learner_type="rf")
        stars_rf = significance_stars(res_rf["p_val"])
        print(f"   RF:    theta={res_rf['coef']:+.3f}  SE={res_rf['se']:.3f}  "
              f"p={res_rf['p_val']:.4f}{stars_rf}  "
              f"CI=[{res_rf['ci_lower']:.3f}, {res_rf['ci_upper']:.3f}]")
        print()

        results.append({
            "treatment": treatment_name,
            "label": TREATMENT_LABELS[treatment_name],
            "n_obs": n_obs,
            "coef": res_lgbm["coef"],
            "se": res_lgbm["se"],
            "t_stat": res_lgbm["t_stat"],
            "p_val": res_lgbm["p_val"],
            "ci_lower": res_lgbm["ci_lower"],
            "ci_upper": res_lgbm["ci_upper"],
            "significance": significance_stars(res_lgbm["p_val"]),
            "interpretation": interpret(treatment_name, res_lgbm["coef"], res_lgbm["p_val"]),
        })

        sensitivity_results.append({
            "treatment": treatment_name,
            "lgbm_coef": res_lgbm["coef"],
            "lgbm_se": res_lgbm["se"],
            "lgbm_pval": res_lgbm["p_val"],
            "rf_coef": res_rf["coef"],
            "rf_se": res_rf["se"],
            "rf_pval": res_rf["p_val"],
            "coef_diff": abs(res_lgbm["coef"] - res_rf["coef"]),
        })

    # ── Print summary table ──────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY: DML Causal Effect Estimates (LGBM learner)")
    print("=" * 70)
    header = f"{'Treatment':<35} {'theta':>7} {'SE':>7} {'p-val':>8} {'Sig':>4}  {'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for r in results:
        ci_str = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
        print(f"{r['label']:<35} {r['coef']:+7.3f} {r['se']:7.3f} {r['p_val']:8.4f} {r['significance']:>4}  {ci_str:>20}")
    print()

    # ── Sensitivity comparison ───────────────────────────────────────────
    print("SENSITIVITY: LGBM vs Random Forest")
    print("-" * 60)
    header2 = f"{'Treatment':<20} {'LGBM theta':>8} {'RF theta':>8} {'delta':>7} {'LGBM p':>8} {'RF p':>8}"
    print(header2)
    print("-" * len(header2))
    for s in sensitivity_results:
        print(f"{s['treatment']:<20} {s['lgbm_coef']:+8.3f} {s['rf_coef']:+8.3f} {s['coef_diff']:7.3f} "
              f"{s['lgbm_pval']:8.4f} {s['rf_pval']:8.4f}")
    print()

    # ── Code vs LLM comparison ───────────────────────────────────────────
    code_results = [r for r in results if r["treatment"].endswith("_code")]
    llm_results = [r for r in results if r["treatment"].endswith("_llm")]
    if code_results and llm_results:
        print("COMPARISON: Code-based vs LLM-based treatment measurement")
        print("-" * 60)
        for code_r in code_results:
            t_num = code_r["treatment"].replace("_code", "")
            llm_r = next((r for r in llm_results if r["treatment"].replace("_llm", "") == t_num), None)
            if llm_r:
                print(f"  {t_num}: Code theta={code_r['coef']:+.3f} (p={code_r['p_val']:.3f})  |  "
                      f"LLM theta={llm_r['coef']:+.3f} (p={llm_r['p_val']:.3f})")
        print()
    elif code_results:
        print("COMPARISON: LLM-based treatments not available (0% coverage)")
        print()

    # ── Interpretations ──────────────────────────────────────────────────
    print("INTERPRETATIONS:")
    for r in results:
        print(f"  - {r['interpretation']}")
    print()

    # ── Save outputs ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Saved results table → {OUT_CSV}")

    output_json = {
        "study": "DML Causal Inference: Page Features → LLM Re-Ranking (50_larger)",
        "variant": "50 SERP results, top-20 LLM re-ranking",
        "method": "DoubleML Partially Linear Regression (PLR)",
        "outcome": "rank_delta (pre_rank - post_rank, positive = LLM promoted)",
        "nuisance_learner": "LGBMRegressor (primary), RandomForestRegressor (sensitivity)",
        "n_folds": 5,
        "score": "partialling out",
        "confounders": CONFOUNDERS,
        "confounders_dropped": {
            "X10_https": "zero variance (all 1)",
            "X4_lcp_ms": "0% coverage",
        },
        "imputation": "median (sklearn SimpleImputer) for confounders",
        "scaling": "StandardScaler for confounders",
        "results": results,
        "sensitivity": sensitivity_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"Saved full results → {OUT_JSON}")

    plot_coefficients(results)
    print(f"Saved coefficient plot → {OUT_PNG}")


def plot_coefficients(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r["label"] for r in results]
    coefs = [r["coef"] for r in results]
    ci_lowers = [r["ci_lower"] for r in results]
    ci_uppers = [r["ci_upper"] for r in results]
    p_vals = [r["p_val"] for r in results]

    y_pos = np.arange(len(labels))
    colors = ["#2166ac" if p < 0.05 else "#b2182b" if p < 0.1 else "#999999" for p in p_vals]

    xerr_lower = [c - cl for c, cl in zip(coefs, ci_lowers)]
    xerr_upper = [cu - c for c, cu in zip(coefs, ci_uppers)]

    ax.barh(y_pos, coefs, xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, edgecolor="black", linewidth=0.5,
            capsize=4, ecolor="black")

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Causal Effect on rank_delta (theta)", fontsize=11)
    ax.set_title("DML Estimates: Effect of Page Features on LLM Re-Ranking (50_larger)\n"
                 "50 SERP results, top-20 re-ranking\n"
                 "(blue=p<0.05, red=p<0.1, grey=n.s.)", fontsize=12)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
