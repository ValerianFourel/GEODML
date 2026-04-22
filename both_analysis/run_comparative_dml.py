#!/usr/bin/env python3
"""
GEODML — Comparative DML Analysis: 20-SERP/10-rerank vs 50-SERP/20-rerank

Runs the full DML analysis on BOTH datasets side-by-side:
  - Dataset A: 20 search results, top-10 LLM re-ranking  (data/geodml_dataset.csv)
  - Dataset B: 50 search results, top-20 LLM re-ranking  (50_larger/data/geodml_dataset.csv)

For each dataset:
  3 outcomes x 4 treatments x 2 paths x 2 methods = 48 experiments (max)

Then produces comparative visualizations and summary tables.

Output → both_analysis/results/ and both_analysis/figures/
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_20 = ROOT / "data" / "geodml_dataset.csv"
DATA_50 = ROOT / "50_larger" / "data" / "geodml_dataset.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Experiment dimensions ────────────────────────────────────────────────────
TREATMENTS = {
    "T1": {"code": "T1_statistical_density_code", "llm": "T1_statistical_density_llm"},
    "T2": {"code": "T2_question_heading_code",    "llm": "T2_question_heading_llm"},
    "T3": {"code": "T3_structured_data_code",     "llm": "T3_structured_data_llm"},
    "T4": {"code": "T4_citation_authority_code",   "llm": "T4_citation_authority_llm"},
}

OUTCOMES = ["rank_delta", "pre_rank", "post_rank"]

METHODS = ["PLR", "IRM"]

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

NEEDS_BINARIZE = {"T1", "T4"}

TREATMENT_LABELS = {
    "T1": "Stat. Density",
    "T2": "Question Hdgs",
    "T3": "Struct. Data",
    "T4": "Citation Auth.",
}

DATASET_LABELS = {
    "20serp": "20-SERP / 10-rerank",
    "50serp": "50-SERP / 20-rerank",
}


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "\u2020"
    return ""


def load_data(path, label):
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[{label}] Loaded {len(df)} rows from {path.name}")
    return df


def get_active_confounders(df):
    return [c for c in CONFOUNDERS if c in df.columns and df[c].notna().any()]


def prepare(df, y_col, d_col, method, t_name):
    if d_col not in df.columns or df[d_col].notna().sum() == 0:
        return None, None, None, 0, "no data", None

    active_conf = get_active_confounders(df)
    cols = active_conf + [d_col, y_col]
    sub = df[cols].dropna(subset=[y_col, d_col]).copy()

    if len(sub) < 10:
        return None, None, None, len(sub), f"only {len(sub)} obs", None

    Y = sub[y_col].values
    D = sub[d_col].values

    if method == "IRM" and t_name in NEEDS_BINARIZE:
        if t_name == "T1":
            threshold = np.median(D)
            D = (D > threshold).astype(float)
        elif t_name == "T4":
            D = (D > 0).astype(float)

    if method == "IRM":
        n1 = int(D.sum())
        n0 = len(D) - n1
        if n1 < 10 or n0 < 10:
            return None, None, None, len(sub), f"IRM split {n0}/{n1} imbalanced", None

    X_raw = sub[active_conf].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, Y, D, len(sub), None, active_conf


def fit_plr(X, Y, D):
    data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D)
    ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    model = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=5,
                            score="partialling out")
    model.fit()
    return extract_results(model), model


def fit_irm(X, Y, D):
    data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D)
    ml_g = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                          num_leaves=31, verbose=-1, random_state=42)
    model = dml.DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=5,
                            score="ATE")
    model.fit()
    return extract_results(model), model


def extract_results(model):
    ci = model.confint(level=0.95)
    return {
        "coef": float(model.coef[0]),
        "se": float(model.se[0]),
        "t_stat": float(model.t_stat[0]),
        "p_val": float(model.pval[0]),
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
    }


def run_ols(X, Y, D, active_conf):
    """Naive OLS: Y ~ D + X"""
    XD = np.column_stack([D, X])
    ols = LinearRegression().fit(XD, Y)
    from sklearn.utils import resample
    n = len(Y)
    y_pred = ols.predict(XD)
    residuals = Y - y_pred
    sse = np.sum(residuals**2)
    mse = sse / (n - XD.shape[1] - 1)
    XtX_inv = np.linalg.pinv(XD.T @ XD)
    se_all = np.sqrt(np.diag(mse * XtX_inv))
    from scipy import stats
    t_stat = ols.coef_[0] / se_all[0] if se_all[0] > 0 else 0
    p_val = 2 * stats.t.sf(abs(t_stat), df=n - XD.shape[1] - 1)
    return {
        "ols_coef": float(ols.coef_[0]),
        "ols_se": float(se_all[0]),
        "ols_pval": float(p_val),
        "ols_r2": float(r2_score(Y, y_pred)),
    }


def run_one_dataset(df, dataset_key):
    """Run all experiments for one dataset, return list of result dicts."""
    results = []
    exp_id = 0
    total = len(OUTCOMES) * 4 * 2 * 2
    ds_label = DATASET_LABELS[dataset_key]

    for y_col in OUTCOMES:
        df_y = df.dropna(subset=[y_col]).copy()
        for t_name in ["T1", "T2", "T3", "T4"]:
            for path in ["code", "llm"]:
                for method in METHODS:
                    exp_id += 1
                    d_col = TREATMENTS[t_name][path]
                    label = f"[{dataset_key}] [{exp_id:2d}/{total}] Y={y_col:<12} D={t_name}_{path:<4} {method}"

                    X, Y, D, n_obs, skip, active_conf = prepare(df_y, y_col, d_col, method, t_name)

                    if skip:
                        print(f"{label}  n={n_obs:<4}  SKIP: {skip}")
                        results.append({
                            "dataset": dataset_key, "exp_id": exp_id,
                            "outcome": y_col, "treatment": t_name,
                            "treatment_col": d_col, "path": path,
                            "method": method, "n_obs": n_obs,
                            "coef": None, "se": None, "t_stat": None,
                            "p_val": None, "ci_lower": None, "ci_upper": None,
                            "significance": "", "skipped": skip,
                            "ols_coef": None, "ols_se": None, "ols_pval": None, "ols_r2": None,
                            "n_confounders": 0,
                        })
                        continue

                    try:
                        if method == "PLR":
                            res, model = fit_plr(X, Y, D)
                        else:
                            res, model = fit_irm(X, Y, D)

                        stars = sig_stars(res["p_val"])
                        print(f"{label}  n={n_obs:<4}  theta={res['coef']:+7.3f}  "
                              f"SE={res['se']:.3f}  p={res['p_val']:.4f} {stars}")

                        ols_res = run_ols(X, Y, D, active_conf)

                        results.append({
                            "dataset": dataset_key, "exp_id": exp_id,
                            "outcome": y_col, "treatment": t_name,
                            "treatment_col": d_col, "path": path,
                            "method": method, "n_obs": n_obs,
                            "skipped": None, **res, "significance": stars,
                            **ols_res, "n_confounders": len(active_conf),
                        })
                    except Exception as e:
                        print(f"{label}  n={n_obs:<4}  ERROR: {e}")
                        results.append({
                            "dataset": dataset_key, "exp_id": exp_id,
                            "outcome": y_col, "treatment": t_name,
                            "treatment_col": d_col, "path": path,
                            "method": method, "n_obs": n_obs,
                            "coef": None, "se": None, "t_stat": None,
                            "p_val": None, "ci_lower": None, "ci_upper": None,
                            "significance": "", "skipped": f"error: {e}",
                            "ols_coef": None, "ols_se": None, "ols_pval": None, "ols_r2": None,
                            "n_confounders": 0,
                        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1_comparative_forest(all_df):
    """Side-by-side forest plots: 20-SERP vs 50-SERP, PLR on rank_delta."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for idx, ds in enumerate(["20serp", "50serp"]):
        ax = axes[idx]
        sub = all_df[(all_df["dataset"] == ds) & (all_df["outcome"] == "rank_delta")
                     & (all_df["method"] == "PLR") & (all_df["coef"].notna())].copy()
        sub = sub.sort_values(["treatment", "path"], ascending=[True, False])

        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14)
            ax.set_title(DATASET_LABELS[ds], fontsize=13, fontweight="bold")
            continue

        labels = [f"{TREATMENT_LABELS[r['treatment']]} ({r['path']})" for _, r in sub.iterrows()]
        y_pos = np.arange(len(labels))

        colors = []
        for _, r in sub.iterrows():
            p = r["p_val"]
            if p < 0.01:   colors.append("#2166ac")
            elif p < 0.05: colors.append("#b2182b")
            elif p < 0.1:  colors.append("#f4a582")
            else:           colors.append("#999999")

        for i, (_, r) in enumerate(sub.iterrows()):
            ax.plot([r["ci_lower"], r["ci_upper"]], [i, i], color=colors[i],
                    linewidth=2.5, solid_capstyle="round")
            ax.plot(r["coef"], i, "o", color=colors[i], markersize=8, zorder=5)
            if r["p_val"] < 0.1:
                ax.annotate(sig_stars(r["p_val"]),
                            (max(r["ci_upper"], r["coef"]) + 0.15, i),
                            fontsize=11, fontweight="bold",
                            color=colors[i], va="center")

        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Causal Effect on rank_delta (theta)\n<-- LLM demotes | LLM promotes -->",
                      fontsize=10)
        ax.set_title(DATASET_LABELS[ds], fontsize=13, fontweight="bold")
        ax.invert_yaxis()

    patches = [mpatches.Patch(color="#2166ac", label="p < 0.01"),
               mpatches.Patch(color="#b2182b", label="p < 0.05"),
               mpatches.Patch(color="#f4a582", label="p < 0.10"),
               mpatches.Patch(color="#999999", label="n.s.")]
    axes[1].legend(handles=patches, loc="lower right", fontsize=9)

    fig.suptitle("DML Causal Estimates: 20-SERP vs 50-SERP\n"
                 "Y = rank_delta, PLR method, LGBM nuisance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_comparative_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig1_comparative_forest.png'}")


def plot_fig2_coefficient_scatter(all_df):
    """Scatter: 20-SERP theta vs 50-SERP theta for matched experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, y_col in enumerate(OUTCOMES):
        ax = axes[idx]
        plr = all_df[(all_df["method"] == "PLR") & (all_df["outcome"] == y_col)
                     & (all_df["coef"].notna())].copy()

        d20 = plr[plr["dataset"] == "20serp"].set_index(["treatment", "path"])
        d50 = plr[plr["dataset"] == "50serp"].set_index(["treatment", "path"])
        common = d20.index.intersection(d50.index)

        if len(common) == 0:
            ax.text(0.5, 0.5, "No matched experiments", transform=ax.transAxes, ha="center")
            ax.set_title(f"Y = {y_col}", fontsize=12)
            continue

        for key in common:
            r20 = d20.loc[key]
            r50 = d50.loc[key]
            t_name, path = key

            # color by min p-value across both
            min_p = min(r20["p_val"], r50["p_val"])
            if min_p < 0.01:   c = "#2166ac"
            elif min_p < 0.05: c = "#b2182b"
            elif min_p < 0.1:  c = "#f4a582"
            else:               c = "#999999"

            ax.scatter(r20["coef"], r50["coef"], color=c, s=80, edgecolor="black",
                       linewidth=0.5, zorder=5)
            ax.annotate(f"{t_name}_{path}", (r20["coef"], r50["coef"]),
                        fontsize=7.5, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

        lims = ax.get_xlim() + ax.get_ylim()
        lo, hi = min(lims), max(lims)
        margin = (hi - lo) * 0.1
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlabel("20-SERP theta", fontsize=10)
        ax.set_ylabel("50-SERP theta", fontsize=10)
        ax.set_title(f"Y = {y_col}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle("Coefficient Comparison: 20-SERP vs 50-SERP (PLR)\n"
                 "Points on diagonal = identical estimates across designs",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_coefficient_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig2_coefficient_scatter.png'}")


def plot_fig3_pvalue_heatmap(all_df):
    """Heatmap: p-values for all experiments, both datasets."""
    plr = all_df[(all_df["method"] == "PLR")].copy()

    row_labels = []
    for t in ["T1", "T2", "T3", "T4"]:
        for p in ["code", "llm"]:
            row_labels.append(f"{TREATMENT_LABELS[t]}\n({p})")

    col_labels = []
    for y in OUTCOMES:
        for ds in ["20serp", "50serp"]:
            short = "20S" if ds == "20serp" else "50S"
            col_labels.append(f"{y}\n({short})")

    n_rows, n_cols = len(row_labels), len(col_labels)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    c_matrix = np.full((n_rows, n_cols), np.nan)

    for _, r in plr.iterrows():
        ri = ["T1", "T2", "T3", "T4"].index(r["treatment"]) * 2 + ["code", "llm"].index(r["path"])
        ci = OUTCOMES.index(r["outcome"]) * 2 + ["20serp", "50serp"].index(r["dataset"])
        if r["coef"] is not None and not pd.isna(r.get("coef")):
            p_matrix[ri, ci] = r["p_val"]
            c_matrix[ri, ci] = r["coef"]

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(p_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Add vertical lines between outcome groups
    for i in [2, 4]:
        ax.axvline(i - 0.5, color="black", linewidth=1.5)

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(p_matrix[i, j]):
                p = p_matrix[i, j]
                c = c_matrix[i, j]
                stars = sig_stars(p)
                txt = f"{c:+.2f}\n{stars}" if stars else f"{c:+.2f}"
                color = "white" if p < 0.15 or p > 0.85 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold" if stars else "normal", color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="gray")

    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    ax.set_title("DML P-value Heatmap: All Experiments (PLR)\n"
                 "20-SERP vs 50-SERP side-by-side; *** p<0.001, ** p<0.01, * p<0.05, \u2020 p<0.1",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_pvalue_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig3_pvalue_heatmap.png'}")


def plot_fig4_effect_size_comparison(all_df):
    """Grouped bar chart comparing effect sizes between the two designs."""
    plr_rd = all_df[(all_df["method"] == "PLR") & (all_df["outcome"] == "rank_delta")
                    & (all_df["coef"].notna())].copy()

    treatments_paths = []
    for t in ["T1", "T2", "T3", "T4"]:
        for p in ["code", "llm"]:
            treatments_paths.append((t, p))

    labels = [f"{TREATMENT_LABELS[t]} ({p})" for t, p in treatments_paths]
    x = np.arange(len(labels))
    width = 0.35

    vals_20, errs_20, colors_20 = [], [], []
    vals_50, errs_50, colors_50 = [], [], []

    for t, p in treatments_paths:
        for ds, vals, errs, cols in [("20serp", vals_20, errs_20, colors_20),
                                      ("50serp", vals_50, errs_50, colors_50)]:
            row = plr_rd[(plr_rd["dataset"] == ds) & (plr_rd["treatment"] == t) & (plr_rd["path"] == p)]
            if len(row) == 1:
                r = row.iloc[0]
                vals.append(r["coef"])
                errs.append(r["se"] * 1.96)
                pv = r["p_val"]
                if pv < 0.01:   cols.append("#2166ac")
                elif pv < 0.05: cols.append("#b2182b")
                elif pv < 0.1:  cols.append("#f4a582")
                else:            cols.append("#999999")
            else:
                vals.append(0)
                errs.append(0)
                cols.append("#dddddd")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, vals_20, width, yerr=errs_20, capsize=3,
                   color=colors_20, edgecolor="black", linewidth=0.5,
                   label="20-SERP / 10-rerank", alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_50, width, yerr=errs_50, capsize=3,
                   color=colors_50, edgecolor="black", linewidth=0.5,
                   label="50-SERP / 20-rerank", alpha=0.85, hatch="//")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Causal Effect (theta) with 95% CI", fontsize=11)
    ax.set_title("Effect Size Comparison: 20-SERP vs 50-SERP\n"
                 "Y = rank_delta, PLR | Solid = 20-SERP, Hatched = 50-SERP",
                 fontsize=13, fontweight="bold")

    # Add significance annotations
    for i, (t, p) in enumerate(treatments_paths):
        for ds, offset in [("20serp", -width/2), ("50serp", width/2)]:
            row = plr_rd[(plr_rd["dataset"] == ds) & (plr_rd["treatment"] == t) & (plr_rd["path"] == p)]
            if len(row) == 1:
                r = row.iloc[0]
                stars = sig_stars(r["p_val"])
                if stars:
                    ypos = r["coef"] + (r["se"] * 1.96 + 0.3) * np.sign(r["coef"])
                    ax.text(i + offset, ypos, stars, ha="center", va="bottom" if r["coef"] > 0 else "top",
                            fontsize=9, fontweight="bold", color="#333333")

    ax.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_effect_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig4_effect_comparison.png'}")


def plot_fig5_dml_vs_ols(all_df):
    """DML vs OLS scatter for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, ds in enumerate(["20serp", "50serp"]):
        ax = axes[idx]
        plr = all_df[(all_df["dataset"] == ds) & (all_df["method"] == "PLR")
                     & (all_df["outcome"] == "rank_delta") & (all_df["coef"].notna())].copy()

        if len(plr) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(DATASET_LABELS[ds])
            continue

        for _, r in plr.iterrows():
            p = r["p_val"]
            if p < 0.01:   c = "#2166ac"
            elif p < 0.05: c = "#b2182b"
            elif p < 0.1:  c = "#f4a582"
            else:           c = "#999999"
            size = 120 if p < 0.05 else 60
            ax.scatter(r["ols_coef"], r["coef"], color=c, s=size,
                       edgecolor="black", linewidth=0.5, zorder=5)
            ax.annotate(f"{r['treatment']}_{r['path']}", (r["ols_coef"], r["coef"]),
                        fontsize=7.5, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

        lims = list(ax.get_xlim()) + list(ax.get_ylim())
        lo, hi = min(lims), max(lims)
        m = (hi - lo) * 0.1
        ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", alpha=0.3, linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlabel("OLS beta (naive)", fontsize=10)
        ax.set_ylabel("DML theta (causal)", fontsize=10)
        ax.set_title(DATASET_LABELS[ds], fontsize=12, fontweight="bold")

    fig.suptitle("DML Causal vs Naive OLS: Both Designs\n"
                 "Y = rank_delta, PLR | Deviation from diagonal = confounding bias",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dml_vs_ols.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig5_dml_vs_ols.png'}")


def plot_fig6_plr_vs_irm(all_df):
    """PLR vs IRM comparison for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, ds in enumerate(["20serp", "50serp"]):
        ax = axes[idx]
        valid = all_df[(all_df["dataset"] == ds) & (all_df["outcome"] == "rank_delta")
                       & (all_df["coef"].notna())].copy()

        plr = valid[valid["method"] == "PLR"].set_index(["treatment", "path"])
        irm = valid[valid["method"] == "IRM"].set_index(["treatment", "path"])
        common = plr.index.intersection(irm.index)

        if len(common) == 0:
            ax.text(0.5, 0.5, "No matched experiments", transform=ax.transAxes, ha="center")
            ax.set_title(DATASET_LABELS[ds])
            continue

        for key in common:
            rp = plr.loc[key]
            ri = irm.loc[key]
            p = min(rp["p_val"], ri["p_val"])
            if p < 0.01:   c = "#2166ac"
            elif p < 0.05: c = "#b2182b"
            elif p < 0.1:  c = "#f4a582"
            else:           c = "#999999"
            size = 120 if p < 0.05 else 60
            ax.scatter(rp["coef"], ri["coef"], color=c, s=size,
                       edgecolor="black", linewidth=0.5, zorder=5)
            t_name, path = key
            ax.annotate(f"{t_name}_{path}", (rp["coef"], ri["coef"]),
                        fontsize=7.5, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

        lims = list(ax.get_xlim()) + list(ax.get_ylim())
        lo, hi = min(lims), max(lims)
        m = (hi - lo) * 0.1
        ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", alpha=0.3, linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlabel("PLR theta", fontsize=10)
        ax.set_ylabel("IRM theta", fontsize=10)
        ax.set_title(DATASET_LABELS[ds], fontsize=12, fontweight="bold")

    fig.suptitle("Method Sensitivity: PLR vs IRM (Both Designs)\n"
                 "Y = rank_delta | Points near diagonal = methods agree",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_plr_vs_irm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig6_plr_vs_irm.png'}")


def plot_fig7_multi_outcome_forest(all_df):
    """3x2 panel: forest plots for each outcome x each dataset."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharey="row")

    for row, y_col in enumerate(OUTCOMES):
        for col, ds in enumerate(["20serp", "50serp"]):
            ax = axes[row, col]
            sub = all_df[(all_df["dataset"] == ds) & (all_df["outcome"] == y_col)
                         & (all_df["method"] == "PLR") & (all_df["coef"].notna())].copy()
            sub = sub.sort_values(["treatment", "path"], ascending=[True, False])

            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=11)
                ax.set_title(f"{y_col} — {DATASET_LABELS[ds]}", fontsize=10)
                continue

            labels = [f"{TREATMENT_LABELS[r['treatment']]} ({r['path']})" for _, r in sub.iterrows()]
            y_pos = np.arange(len(labels))

            for i, (_, r) in enumerate(sub.iterrows()):
                p = r["p_val"]
                if p < 0.01:   c = "#2166ac"
                elif p < 0.05: c = "#b2182b"
                elif p < 0.1:  c = "#f4a582"
                else:           c = "#999999"
                ax.plot([r["ci_lower"], r["ci_upper"]], [i, i], color=c, linewidth=2, solid_capstyle="round")
                ax.plot(r["coef"], i, "o", color=c, markersize=6, zorder=5)
                if p < 0.1:
                    ax.annotate(sig_stars(p), (r["ci_upper"] + 0.1, i),
                                fontsize=9, fontweight="bold", color=c, va="center")

            ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()

            if row == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("theta", fontsize=9)

            # Add outcome label on left
            if col == 0:
                ax.set_ylabel(y_col, fontsize=11, fontweight="bold")

    fig.suptitle("DML Causal Estimates: All Outcomes x Both Designs (PLR)\n"
                 "95% CIs shown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_multi_outcome_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig7_multi_outcome_forest.png'}")


def plot_fig8_summary_table(all_df):
    """Publication-style summary table as a figure."""
    plr_rd = all_df[(all_df["method"] == "PLR") & (all_df["outcome"] == "rank_delta")].copy()

    rows = []
    for t in ["T1", "T2", "T3", "T4"]:
        for p in ["code", "llm"]:
            row = {"Treatment": TREATMENT_LABELS[t], "Path": p}
            for ds in ["20serp", "50serp"]:
                r = plr_rd[(plr_rd["dataset"] == ds) & (plr_rd["treatment"] == t)
                           & (plr_rd["path"] == p)]
                prefix = "20S" if ds == "20serp" else "50S"
                if len(r) == 1 and pd.notna(r.iloc[0]["coef"]):
                    r = r.iloc[0]
                    row[f"{prefix} N"] = int(r["n_obs"])
                    row[f"{prefix} theta"] = f"{r['coef']:+.3f}"
                    row[f"{prefix} SE"] = f"{r['se']:.3f}"
                    row[f"{prefix} p"] = f"{r['p_val']:.4f}"
                    row[f"{prefix} Sig"] = r["significance"]
                    row[f"{prefix} CI"] = f"[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]"
                else:
                    row[f"{prefix} N"] = "—"
                    row[f"{prefix} theta"] = "—"
                    row[f"{prefix} SE"] = "—"
                    row[f"{prefix} p"] = "—"
                    row[f"{prefix} Sig"] = ""
                    row[f"{prefix} CI"] = "—"
            rows.append(row)

    tbl_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(22, 5))
    ax.axis("off")

    col_labels = list(tbl_df.columns)
    cell_text = tbl_df.values.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1a3c5e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Highlight significant rows
    for i, row in enumerate(rows):
        for ds_prefix in ["20S", "50S"]:
            sig = row.get(f"{ds_prefix} Sig", "")
            if sig and sig.strip():
                # Determine which columns belong to this dataset
                start_col = 2 if ds_prefix == "20S" else 8
                n_cols_per = 6
                if "***" in sig or "**" in sig:
                    bg = "#d0e4f5"
                elif "*" in sig:
                    bg = "#e8f0f8"
                else:
                    bg = "#fff8dc"
                for j in range(start_col, start_col + n_cols_per):
                    if j < len(col_labels):
                        table[i + 1, j].set_facecolor(bg)

    ax.set_title("Comparative Summary: DML Causal Estimates on rank_delta (PLR)\n"
                 "20S = 20-SERP/10-rerank | 50S = 50-SERP/20-rerank\n"
                 "Blue highlight = significant",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig8_summary_table.png'}")


def plot_fig9_dataset_descriptives(df20, df50):
    """Compare descriptive statistics of key variables between datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    vars_to_compare = [
        ("rank_delta", "Rank Delta (pre - post)"),
        ("pre_rank", "Pre-Rank (SERP position)"),
        ("post_rank", "Post-Rank (LLM position)"),
        ("X3_word_count", "Word Count"),
        ("X7_internal_links", "Internal Links"),
        ("X1_domain_authority", "Domain Authority"),
    ]

    for idx, (var, title) in enumerate(vars_to_compare):
        ax = axes[idx // 3, idx % 3]

        v20 = df20[var].dropna() if var in df20.columns else pd.Series(dtype=float)
        v50 = df50[var].dropna() if var in df50.columns else pd.Series(dtype=float)

        data = []
        labels_list = []
        colors = []
        if len(v20) > 0:
            data.append(v20.values)
            labels_list.append(f"20-SERP\n(n={len(v20)})")
            colors.append("#4393c3")
        if len(v50) > 0:
            data.append(v50.values)
            labels_list.append(f"50-SERP\n(n={len(v50)})")
            colors.append("#d6604d")

        if len(data) > 0:
            bp = ax.boxplot(data, labels=labels_list, patch_artist=True, widths=0.5)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dataset Comparison: Variable Distributions\n"
                 "20-SERP/10-rerank vs 50-SERP/20-rerank",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_dataset_descriptives.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {FIG_DIR / 'fig9_dataset_descriptives.png'}")


def print_comparative_summary(all_df):
    """Print side-by-side comparison of key results."""
    plr_rd = all_df[(all_df["method"] == "PLR") & (all_df["outcome"] == "rank_delta")].copy()

    print("\n" + "=" * 100)
    print("COMPARATIVE SUMMARY: Y = rank_delta, PLR method")
    print("  20-SERP / 10-rerank  vs  50-SERP / 20-rerank")
    print("=" * 100)

    header = (f"{'Treatment':<20} {'Path':<5} | "
              f"{'20S theta':>10} {'20S p':>8} {'20S Sig':>4} {'20S n':>5} | "
              f"{'50S theta':>10} {'50S p':>8} {'50S Sig':>4} {'50S n':>5} | "
              f"{'Direction':>10}")
    print(header)
    print("-" * len(header))

    for t in ["T1", "T2", "T3", "T4"]:
        for p in ["code", "llm"]:
            r20 = plr_rd[(plr_rd["dataset"] == "20serp") & (plr_rd["treatment"] == t) & (plr_rd["path"] == p)]
            r50 = plr_rd[(plr_rd["dataset"] == "50serp") & (plr_rd["treatment"] == t) & (plr_rd["path"] == p)]

            t_label = f"{TREATMENT_LABELS[t]}"

            def fmt(r):
                if len(r) == 1 and pd.notna(r.iloc[0]["coef"]):
                    r = r.iloc[0]
                    return f"{r['coef']:+10.3f} {r['p_val']:8.4f} {r['significance']:>4} {int(r['n_obs']):5d}"
                return f"{'—':>10} {'—':>8} {'':>4} {'—':>5}"

            # Direction comparison
            if (len(r20) == 1 and pd.notna(r20.iloc[0]["coef"]) and
                len(r50) == 1 and pd.notna(r50.iloc[0]["coef"])):
                c20 = r20.iloc[0]["coef"]
                c50 = r50.iloc[0]["coef"]
                if (c20 > 0) == (c50 > 0):
                    direction = "AGREE"
                else:
                    direction = "DISAGREE"
            else:
                direction = "—"

            print(f"{t_label:<20} {p:<5} | {fmt(r20)} | {fmt(r50)} | {direction:>10}")

    # Print significant findings
    sig = all_df[(all_df["coef"].notna()) & (all_df["p_val"] < 0.1)].sort_values("p_val")
    print(f"\n{'=' * 100}")
    print(f"ALL SIGNIFICANT FINDINGS (p < 0.1): {len(sig)} experiments")
    print("=" * 100)
    for _, r in sig.iterrows():
        direction = "promotes" if r["coef"] > 0 else "demotes"
        ds_label = DATASET_LABELS[r["dataset"]]
        print(f"  [{ds_label}] Y={r['outcome']:<12} {r['treatment']} {TREATMENT_LABELS[r['treatment']]:<16} "
              f"({r['path']}, {r['method']}): theta={r['coef']:+.3f}, p={r['p_val']:.4f}{r['significance']}  "
              f"-> LLM {direction} by {abs(r['coef']):.2f} ranks")

    # Print dataset size comparison
    print(f"\n{'=' * 100}")
    print("DATASET COMPARISON")
    print("=" * 100)
    for ds in ["20serp", "50serp"]:
        ds_results = all_df[all_df["dataset"] == ds]
        valid = ds_results[ds_results["coef"].notna()]
        sig_count = len(valid[valid["p_val"] < 0.05])
        total = len(valid)
        print(f"  {DATASET_LABELS[ds]:>25}: {total} experiments run, "
              f"{sig_count} significant at p<0.05 ({100*sig_count/total:.1f}%)" if total > 0 else "")


def save_results(all_df, df20, df50):
    """Save all results to CSV and JSON."""
    csv_path = OUT_DIR / "all_experiments.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # Descriptive stats for both datasets
    for ds_key, df, name in [("20serp", df20, "20-SERP"), ("50serp", df50, "50-SERP")]:
        desc = df.describe().T
        desc["n_missing"] = len(df) - df.count()
        desc["pct_missing"] = (desc["n_missing"] / len(df) * 100).round(1)
        desc.to_csv(OUT_DIR / f"descriptive_stats_{ds_key}.csv")

    # JSON summary
    output = {
        "study": "GEODML Comparative DML Analysis",
        "design": "20-SERP/10-rerank vs 50-SERP/20-rerank",
        "experiments_per_dataset": "3 outcomes x 4 treatments x 2 paths x 2 methods = 48",
        "datasets": {
            "20serp": {"rows": len(df20), "label": DATASET_LABELS["20serp"],
                       "file": str(DATA_20)},
            "50serp": {"rows": len(df50), "label": DATASET_LABELS["50serp"],
                       "file": str(DATA_50)},
        },
        "confounders": CONFOUNDERS,
        "nuisance_learners": "LGBM",
        "n_folds": 5,
        "n_experiments_total": len(all_df),
        "n_significant_005": int(len(all_df[(all_df["coef"].notna()) & (all_df["p_val"] < 0.05)])),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {OUT_DIR / 'summary.json'}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("GEODML — Comparative DML Analysis")
    print("  Dataset A: 20 SERP results, top-10 LLM re-ranking")
    print("  Dataset B: 50 SERP results, top-20 LLM re-ranking")
    print("  3 outcomes (rank_delta, pre_rank, post_rank)")
    print("  4 treatments (T1-T4) x 2 paths (code, llm) x 2 methods (PLR, IRM)")
    print("=" * 100)
    print()

    # Load both datasets
    df20 = load_data(DATA_20, "20serp")
    df50 = load_data(DATA_50, "50serp")

    print(f"\n{'─' * 60}")
    print(f"Dataset A active confounders: {get_active_confounders(df20)}")
    print(f"Dataset B active confounders: {get_active_confounders(df50)}")
    print(f"{'─' * 60}\n")

    # Run all experiments on both
    print("=" * 100)
    print("RUNNING EXPERIMENTS ON 20-SERP DATASET")
    print("=" * 100)
    results_20 = run_one_dataset(df20, "20serp")

    print(f"\n{'=' * 100}")
    print("RUNNING EXPERIMENTS ON 50-SERP DATASET")
    print("=" * 100)
    results_50 = run_one_dataset(df50, "50serp")

    # Combine
    all_df = pd.DataFrame(results_20 + results_50)

    # Print summary
    print_comparative_summary(all_df)

    # Save results
    save_results(all_df, df20, df50)

    # Generate all visualizations
    print(f"\n{'=' * 100}")
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)
    plot_fig1_comparative_forest(all_df)
    plot_fig2_coefficient_scatter(all_df)
    plot_fig3_pvalue_heatmap(all_df)
    plot_fig4_effect_size_comparison(all_df)
    plot_fig5_dml_vs_ols(all_df)
    plot_fig6_plr_vs_irm(all_df)
    plot_fig7_multi_outcome_forest(all_df)
    plot_fig8_summary_table(all_df)
    plot_fig9_dataset_descriptives(df20, df50)

    print(f"\n{'=' * 100}")
    print("DONE — All results in both_analysis/results/ and both_analysis/figures/")
    print("=" * 100)


if __name__ == "__main__":
    main()
