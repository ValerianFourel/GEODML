#!/usr/bin/env python3
"""
GEODML — Full Model Diagnostics — 50_larger variant

For each DML experiment, extract and report:
  1. Causal estimate (theta, SE, p-value, CI)
  2. Nuisance model performance (cross-validated R2 for ml_l and ml_m)
  3. Confounder feature importances (LGBM split counts, both nuisance models)
  4. Naive OLS comparison (treatment + confounders, no DML orthogonalization)
  5. Descriptive statistics for all variables

Runs on: Y = {rank_delta, pre_rank, post_rank}, all 4 treatments x 2 paths, PLR only.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_50 = SCRIPT_DIR.parent
DATA_CSV = PROJECT_50 / "data" / "geodml_dataset.csv"
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
TREATMENTS = {
    "T1": {"code": "T1_statistical_density_code", "llm": "T1_statistical_density_llm"},
    "T2": {"code": "T2_question_heading_code",    "llm": "T2_question_heading_llm"},
    "T3": {"code": "T3_structured_data_code",     "llm": "T3_structured_data_llm"},
    "T4": {"code": "T4_citation_authority_code",   "llm": "T4_citation_authority_llm"},
}

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

CONFOUNDER_LABELS = {
    "X1_domain_authority": "X1 Domain Authority",
    "X2_domain_age_years": "X2 Domain Age (years)",
    "X3_word_count":       "X3 Word Count",
    "X6_readability":      "X6 Readability (F-K)",
    "X7_internal_links":   "X7 Internal Links",
    "X7B_outbound_links":  "X7B Outbound Links",
    "X8_keyword_difficulty":"X8 Keyword Difficulty",
    "X9_images_with_alt":  "X9 Images with Alt",
}

TREATMENT_LABELS = {
    "T1": "Statistical Density",
    "T2": "Question Headings",
    "T3": "Structured Data",
    "T4": "Citation Authority",
}

OUTCOMES = {
    "rank_delta": "rank_delta",
    "pre_rank": "pre_rank",
    "post_rank": "post_rank",
}


def sig_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


def load_data():
    if not DATA_CSV.exists():
        print(f"ERROR: {DATA_CSV} not found.")
        sys.exit(1)
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows")
    return df


def get_active_confounders(df):
    """Return confounders that have at least some non-null data."""
    return [c for c in CONFOUNDERS if c in df.columns and df[c].notna().any()]


def prepare(df, y_col, d_col):
    # Skip if treatment column is all null
    if d_col not in df.columns or df[d_col].notna().sum() == 0:
        return None, None, None, None, None, None

    active_confounders = get_active_confounders(df)
    cols = active_confounders + [d_col, y_col]
    sub = df[cols].dropna(subset=[y_col, d_col]).copy()

    if len(sub) < 10:
        return None, None, None, None, None, None

    Y = sub[y_col].values
    D = sub[d_col].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(sub[active_confounders])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, Y, D, sub, scaler, active_confounders


def run_full_model(X, Y, D, t_name, path, y_name, active_confounders):
    result = {}

    # 1. DML PLR
    data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D)
    ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    plr = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=5,
                          score="partialling out")
    plr.fit()

    ci = plr.confint(level=0.95)
    result["dml"] = {
        "coef": float(plr.coef[0]),
        "se": float(plr.se[0]),
        "t_stat": float(plr.t_stat[0]),
        "p_val": float(plr.pval[0]),
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
    }

    # 2. Nuisance model R2
    y_pred_cv = plr.predictions["ml_l"].flatten()
    d_pred_cv = plr.predictions["ml_m"].flatten()
    result["nuisance"] = {
        "r2_ml_l": float(r2_score(Y, y_pred_cv)),
        "r2_ml_m": float(r2_score(D, d_pred_cv)),
        "rmse_ml_l": float(plr.nuisance_loss["ml_l"][0][0]),
        "rmse_ml_m": float(plr.nuisance_loss["ml_m"][0][0]),
    }

    # 3. Feature importances
    ml_l_full = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                              num_leaves=31, verbose=-1, random_state=42)
    ml_l_full.fit(X, Y)
    ml_m_full = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                              num_leaves=31, verbose=-1, random_state=42)
    ml_m_full.fit(X, D)

    imp_l = ml_l_full.feature_importances_.astype(float)
    imp_m = ml_m_full.feature_importances_.astype(float)
    imp_l_pct = imp_l / imp_l.sum() * 100 if imp_l.sum() > 0 else imp_l
    imp_m_pct = imp_m / imp_m.sum() * 100 if imp_m.sum() > 0 else imp_m

    result["importances"] = {
        "ml_l": {c: {"splits": int(imp_l[i]), "pct": round(float(imp_l_pct[i]), 1)}
                 for i, c in enumerate(active_confounders)},
        "ml_m": {c: {"splits": int(imp_m[i]), "pct": round(float(imp_m_pct[i]), 1)}
                 for i, c in enumerate(active_confounders)},
    }
    result["active_confounders"] = active_confounders

    # 4. Naive OLS
    XD = np.column_stack([D, X])
    ols = LinearRegression().fit(XD, Y)
    ols_coef_d = ols.coef_[0]
    n_boot = 1000
    rng = np.random.RandomState(42)
    boot_coefs = []
    for _ in range(n_boot):
        idx = rng.choice(len(Y), len(Y), replace=True)
        ols_b = LinearRegression().fit(XD[idx], Y[idx])
        boot_coefs.append(ols_b.coef_[0])
    ols_se = np.std(boot_coefs)
    ols_t = ols_coef_d / ols_se if ols_se > 0 else 0
    from scipy import stats as sp_stats
    ols_p = 2 * (1 - sp_stats.norm.cdf(abs(ols_t)))
    ols_r2 = ols.score(XD, Y)

    result["ols"] = {
        "coef_treatment": float(ols_coef_d),
        "se": float(ols_se),
        "p_val": float(ols_p),
        "r2": float(ols_r2),
        "coef_confounders": {c: float(ols.coef_[i + 1]) for i, c in enumerate(active_confounders)},
        "intercept": float(ols.intercept_),
    }

    return result


def main():
    df = load_data()

    all_results = {}
    all_rows = []

    for y_name, y_col in OUTCOMES.items():
        working_df = df.dropna(subset=[y_col]) if y_col == "rank_delta" else df.copy()

        for t_name in ["T1", "T2", "T3", "T4"]:
            for path in ["code", "llm"]:
                d_col = TREATMENTS[t_name][path]
                key = f"{y_name}__{t_name}_{path}"
                label = f"Y={y_name:<12} D={t_name}_{path}"

                prep_result = prepare(working_df, y_col, d_col)
                X, Y, D, sub, scaler, active_confounders = prep_result

                if X is None:
                    print(f"{label}  SKIPPED: {d_col} has no data or insufficient observations")
                    continue

                n = len(Y)

                result = run_full_model(X, Y, D, t_name, path, y_name, active_confounders)
                result["meta"] = {
                    "outcome": y_name, "treatment": t_name, "path": path,
                    "treatment_col": d_col, "n_obs": n,
                }

                d = result["dml"]
                nu = result["nuisance"]
                o = result["ols"]
                stars = sig_stars(d["p_val"])
                ols_stars = sig_stars(o["p_val"])

                print(f"{label}  n={n:<4}  "
                      f"DML: theta={d['coef']:+.3f} p={d['p_val']:.4f}{stars:<3}  "
                      f"OLS: beta={o['coef_treatment']:+.3f} p={o['p_val']:.4f}{ols_stars:<3}  "
                      f"R2(Y~X)={nu['r2_ml_l']:+.3f}  R2(D~X)={nu['r2_ml_m']:+.3f}")

                all_results[key] = result
                all_rows.append({
                    "outcome": y_name, "treatment": t_name, "path": path,
                    "n_obs": n,
                    "dml_coef": d["coef"], "dml_se": d["se"],
                    "dml_pval": d["p_val"], "dml_ci_lower": d["ci_lower"],
                    "dml_ci_upper": d["ci_upper"], "dml_sig": stars,
                    "ols_coef": o["coef_treatment"], "ols_se": o["se"],
                    "ols_pval": o["p_val"], "ols_r2": o["r2"], "ols_sig": ols_stars,
                    "r2_Y_X": nu["r2_ml_l"], "r2_D_X": nu["r2_ml_m"],
                    "rmse_Y_X": nu["rmse_ml_l"], "rmse_D_X": nu["rmse_ml_m"],
                })

    results_df = pd.DataFrame(all_rows)

    print_tables(results_df, all_results)

    results_df.to_csv(OUT_DIR / "full_diagnostics.csv", index=False)
    print(f"\nSaved -> {OUT_DIR / 'full_diagnostics.csv'}")

    with open(OUT_DIR / "full_diagnostics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved -> {OUT_DIR / 'full_diagnostics.json'}")

    plot_dml_vs_ols(results_df)
    plot_nuisance_r2(results_df)
    plot_feature_importances(all_results)
    plot_ols_confounder_coefs(all_results)

    save_descriptive_stats(df)

    print("\nDone.")


def print_tables(results_df, all_results):
    for y_name in OUTCOMES:
        sub = results_df[results_df["outcome"] == y_name]
        print(f"\n{'=' * 95}")
        print(f"Y = {y_name}")
        print(f"{'=' * 95}")
        header = (f"{'Treatment':<20} {'n':>4}  "
                  f"{'DML theta':>8} {'p':>7} {'Sig':>3}  "
                  f"{'OLS beta':>8} {'p':>7} {'Sig':>3}  "
                  f"{'R2(Y~X)':>8} {'R2(D~X)':>8}")
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            print(f"{r['treatment']}_{r['path']:<15} {r['n_obs']:4d}  "
                  f"{r['dml_coef']:+8.3f} {r['dml_pval']:7.4f} {r['dml_sig']:>3}  "
                  f"{r['ols_coef']:+8.3f} {r['ols_pval']:7.4f} {r['ols_sig']:>3}  "
                  f"{r['r2_Y_X']:+8.4f} {r['r2_D_X']:+8.4f}")

    # Determine which keys are available for rank_delta
    available_keys = []
    for t_name in ["T1", "T2", "T3", "T4"]:
        for path in ["code", "llm"]:
            key = f"rank_delta__{t_name}_{path}"
            if key in all_results:
                available_keys.append((t_name, path, key))

    if available_keys:
        # Get active confounders from any available result
        sample_key = available_keys[0][2]
        active_conf = all_results[sample_key].get("active_confounders", CONFOUNDERS)

        print(f"\n{'=' * 95}")
        print("OLS CONFOUNDER COEFFICIENTS (Y = rank_delta, standardized X)")
        print("=" * 95)
        header2 = f"{'Confounder':<25}"
        for t_name, path, key in available_keys:
            header2 += f"  {t_name}_{path:<4}"
        print(header2)
        print("-" * len(header2))
        for c in active_conf:
            label = CONFOUNDER_LABELS.get(c, c)
            row = f"{label:<25}"
            for t_name, path, key in available_keys:
                coef = all_results[key]["ols"]["coef_confounders"].get(c, 0)
                row += f"  {coef:+7.3f}"
            print(row)

        print(f"\n{'=' * 95}")
        print("LGBM FEATURE IMPORTANCES — ml_l (Y ~ X) — rank_delta models (% of total splits)")
        print("=" * 95)
        header3 = f"{'Confounder':<25}"
        for t_name, path, key in available_keys:
            header3 += f"  {t_name}_{path:<4}"
        print(header3)
        print("-" * len(header3))
        for c in active_conf:
            label = CONFOUNDER_LABELS.get(c, c)
            row = f"{label:<25}"
            for t_name, path, key in available_keys:
                pct = all_results[key]["importances"]["ml_l"].get(c, {}).get("pct", 0)
                row += f"  {pct:6.1f}%"
            print(row)


def plot_dml_vs_ols(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, y_name in enumerate(OUTCOMES):
        ax = axes[idx]
        sub = results_df[results_df["outcome"] == y_name]
        colors = ["#2166ac" if p < 0.05 else "#b2182b" if p < 0.1 else "#999999"
                  for p in sub["dml_pval"]]
        ax.scatter(sub["ols_coef"], sub["dml_coef"], c=colors, s=80, edgecolors="black",
                   linewidth=0.5, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['treatment']}_{r['path']}", (r["ols_coef"], r["dml_coef"]),
                        fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points")
        lim = max(abs(sub["ols_coef"]).max(), abs(sub["dml_coef"]).max()) * 1.3
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("OLS beta (naive)", fontsize=10)
        ax.set_ylabel("DML theta (causal)", fontsize=10)
        ax.set_title(f"Y = {y_name}", fontsize=11)

    fig.suptitle("DML vs OLS (50_larger)\n(blue=p<0.05, red=p<0.1, grey=n.s.)",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "dml_vs_ols.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'dml_vs_ols.png'}")


def plot_nuisance_r2(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, y_name in enumerate(OUTCOMES):
        ax = axes[idx]
        sub = results_df[results_df["outcome"] == y_name].copy()
        labels = [f"{r['treatment']}_{r['path']}" for _, r in sub.iterrows()]
        y_pos = np.arange(len(labels))
        w = 0.35

        ax.barh(y_pos - w/2, sub["r2_Y_X"], height=w, color="#4393c3", alpha=0.8,
                label="R2(Y~X) outcome model")
        ax.barh(y_pos + w/2, sub["r2_D_X"], height=w, color="#d6604d", alpha=0.8,
                label="R2(D~X) treatment model")
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Cross-validated R2", fontsize=9)
        ax.set_title(f"Y = {y_name}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.invert_yaxis()

    fig.suptitle("Nuisance Model Performance (50_larger)\n"
                 "Negative R2 = confounders predict worse than the mean",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "nuisance_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'nuisance_r2.png'}")


def plot_feature_importances(all_results):
    treatments = []
    imp_matrix = []
    active_conf = None
    for t_name in ["T1", "T2", "T3", "T4"]:
        for path in ["code", "llm"]:
            key = f"rank_delta__{t_name}_{path}"
            if key not in all_results:
                continue
            if active_conf is None:
                active_conf = all_results[key].get("active_confounders", CONFOUNDERS)
            treatments.append(f"{t_name}_{path}")
            row = [all_results[key]["importances"]["ml_l"].get(c, {}).get("pct", 0) for c in active_conf]
            imp_matrix.append(row)

    if not treatments or active_conf is None:
        return

    imp_matrix = np.array(imp_matrix)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(imp_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(active_conf)))
    ax.set_xticklabels([CONFOUNDER_LABELS.get(c, c) for c in active_conf], fontsize=8, rotation=35, ha="right")
    ax.set_yticks(range(len(treatments)))
    ax.set_yticklabels(treatments, fontsize=9)

    for i in range(len(treatments)):
        for j in range(len(active_conf)):
            val = imp_matrix[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if val > 18 else "black")

    ax.set_title("LGBM Feature Importances (50_larger) — ml_l (Y=rank_delta ~ X)\n"
                 "% of total splits per confounder", fontsize=11)
    plt.colorbar(im, ax=ax, label="% of splits", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'feature_importances.png'}")


def plot_ols_confounder_coefs(all_results):
    # Collect available experiments
    available = []
    for t_name in ["T1", "T2", "T3", "T4"]:
        for path in ["code", "llm"]:
            key = f"rank_delta__{t_name}_{path}"
            if key in all_results:
                available.append((t_name, path, key))

    if not available:
        return

    n_cols = len(set(t for t, _, _ in available))
    paths = list(set(p for _, p, _ in available))
    n_rows = len(paths)
    t_names = list(dict.fromkeys(t for t, _, _ in available))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=True,
                             squeeze=False)

    for t_name, path, key in available:
        ti = t_names.index(t_name)
        pi = paths.index(path)
        ax = axes[pi, ti]
        active_conf = all_results[key].get("active_confounders", CONFOUNDERS)
        coefs = all_results[key]["ols"]["coef_confounders"]
        r2 = all_results[key]["ols"]["r2"]

        labels = [CONFOUNDER_LABELS.get(c, c).replace(" ", "\n", 1) for c in active_conf]
        vals = [coefs.get(c, 0) for c in active_conf]
        colors = ["#4393c3" if v > 0 else "#d6604d" for v in vals]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{t_name} {TREATMENT_LABELS[t_name]}\n({path}) R2={r2:.3f}",
                     fontsize=9)
        ax.set_xlabel("OLS beta (standardized)", fontsize=7)
        ax.invert_yaxis()

    fig.suptitle("OLS Confounder Coefficients (50_larger, Y = rank_delta)\n"
                 "Blue = positive, Red = negative",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ols_confounder_coefs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'ols_confounder_coefs.png'}")


def save_descriptive_stats(df):
    working = df.dropna(subset=["rank_delta"]).copy()
    all_vars = (["rank_delta", "pre_rank", "post_rank"]
                + list(TREATMENTS["T1"].values()) + list(TREATMENTS["T2"].values())
                + list(TREATMENTS["T3"].values()) + list(TREATMENTS["T4"].values())
                + CONFOUNDERS)

    rows = []
    for col in all_vars:
        if col in working.columns:
            s = working[col]
            rows.append({
                "variable": col,
                "n": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 1),
                "mean": round(s.mean(), 3) if s.notna().any() else None,
                "std": round(s.std(), 3) if s.notna().any() else None,
                "min": round(s.min(), 3) if s.notna().any() else None,
                "p25": round(s.quantile(0.25), 3) if s.notna().any() else None,
                "median": round(s.median(), 3) if s.notna().any() else None,
                "p75": round(s.quantile(0.75), 3) if s.notna().any() else None,
                "max": round(s.max(), 3) if s.notna().any() else None,
            })

    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(OUT_DIR / "descriptive_stats.csv", index=False)

    print(f"\n{'=' * 95}")
    print("DESCRIPTIVE STATISTICS (rows with valid rank_delta)")
    print("=" * 95)
    print(f"{'Variable':<35} {'n':>4} {'miss%':>5} {'mean':>8} {'std':>8} "
          f"{'min':>8} {'p25':>8} {'med':>8} {'p75':>8} {'max':>8}")
    print("-" * 115)
    for _, r in desc_df.iterrows():
        print(f"{r['variable']:<35} {r['n']:4d} {r['missing_pct']:5.1f} "
              f"{r['mean']:8.3f} {r['std']:8.3f} {r['min']:8.3f} {r['p25']:8.3f} "
              f"{r['median']:8.3f} {r['p75']:8.3f} {r['max']:8.3f}")

    print(f"\nSaved -> {OUT_DIR / 'descriptive_stats.csv'}")


if __name__ == "__main__":
    main()
