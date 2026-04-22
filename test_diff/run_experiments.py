#!/usr/bin/env python3
"""
GEODML — DML Experiment Suite on rank_delta (16 models)

Y = rank_delta = pre_rank - post_rank
    Positive → LLM promoted the result
    Negative → LLM demoted the result

Factorial design:
  1 outcome       × 4 treatments × 2 measurement paths × 2 DML methods = 16
  (rank_delta)      (T1, T2,       (code, llm)           (PLR, IRM)
                     T3, T4)

IRM requires binary treatment:
  T1 → 1 if above median, 0 otherwise
  T4 → 1 if any citation present, 0 otherwise
  T2, T3 are already binary (0/1)

Confounders (X): X1_domain_authority, X2_domain_age_years, X3_word_count,
  X6_readability, X7_internal_links, X7B_outbound_links, X8_keyword_difficulty,
  X9_images_with_alt
  (Dropped: X4_lcp_ms = 0% coverage, X10_https = zero variance)

Output:
  test_diff/results/all_experiments.csv      — 16-row summary table
  test_diff/results/all_experiments.json     — full results + metadata
  test_diff/results/heatmap_pvalues.png      — p-value heatmap (treatment × method)
  test_diff/results/coef_comparison.png      — PLR vs IRM coefficient comparison
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT / "data" / "geodml_dataset.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Experiment dimensions ────────────────────────────────────────────────────
TREATMENTS = {
    "T1": {"code": "T1_statistical_density_code", "llm": "T1_statistical_density_llm"},
    "T2": {"code": "T2_question_heading_code",    "llm": "T2_question_heading_llm"},
    "T3": {"code": "T3_structured_data_code",     "llm": "T3_structured_data_llm"},
    "T4": {"code": "T4_citation_authority_code",   "llm": "T4_citation_authority_llm"},
}

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
    "T1": "Statistical Density",
    "T2": "Question Headings",
    "T3": "Structured Data",
    "T4": "Citation Authority",
}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows from {DATA_CSV.name}")

    # Compute rank_delta and drop rows where it's missing
    n_before = len(df)
    df = df.dropna(subset=["rank_delta"]).copy()
    print(f"After dropping missing rank_delta: {len(df)} rows ({n_before - len(df)} dropped)")
    print(f"rank_delta: mean={df['rank_delta'].mean():.2f}, median={df['rank_delta'].median():.1f}, "
          f"std={df['rank_delta'].std():.2f}, range=[{df['rank_delta'].min():.0f}, {df['rank_delta'].max():.0f}]")
    return df


def prepare(df, d_col, method, t_name):
    """Prepare X, Y, D arrays. Binarize treatment for IRM if needed."""
    cols = CONFOUNDERS + [d_col, "rank_delta"]
    sub = df[cols].dropna(subset=[d_col]).copy()

    Y = sub["rank_delta"].values
    D = sub[d_col].values

    # Binarize for IRM if treatment is continuous
    if method == "IRM" and t_name in NEEDS_BINARIZE:
        if t_name == "T1":
            threshold = np.median(D)
            D = (D > threshold).astype(float)
        elif t_name == "T4":
            D = (D > 0).astype(float)

    # Check binary balance for IRM
    if method == "IRM":
        n1 = int(D.sum())
        n0 = len(D) - n1
        if n1 < 10 or n0 < 10:
            return None, None, None, len(sub), f"skipped: binary split {n0}/{n1} too imbalanced"

    # Impute and scale confounders
    X_raw = sub[CONFOUNDERS].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, Y, D, len(sub), None


# ── Model fitting ────────────────────────────────────────────────────────────
def fit_plr(X, Y, D):
    data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D)
    ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    model = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=5,
                            score="partialling out")
    model.fit()
    return extract_results(model)


def fit_irm(X, Y, D):
    data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=D)
    ml_g = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                         num_leaves=31, verbose=-1, random_state=42)
    ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                          num_leaves=31, verbose=-1, random_state=42)
    model = dml.DoubleMLIRM(data, ml_g=ml_g, ml_m=ml_m, n_folds=5,
                            score="ATE")
    model.fit()
    return extract_results(model)


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


def sig_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return ""


# ── Main experiment loop ─────────────────────────────────────────────────────
def run_all():
    df = load_data()
    print(f"\nRunning 16 DML experiments (1 Y × 4 T × 2 paths × 2 methods)\n")

    all_results = []
    exp_id = 0

    for t_name in ["T1", "T2", "T3", "T4"]:
        for path in ["code", "llm"]:
            for method in METHODS:
                exp_id += 1
                d_col = TREATMENTS[t_name][path]
                label = f"[{exp_id:2d}/16] D={t_name}_{path:<4}  {method}"

                X, Y, D, n_obs, skip_reason = prepare(df, d_col, method, t_name)

                if skip_reason:
                    print(f"{label}  n={n_obs:<4}  {skip_reason}")
                    all_results.append({
                        "exp_id": exp_id, "outcome": "rank_delta",
                        "treatment": t_name, "treatment_col": d_col,
                        "path": path, "method": method, "n_obs": n_obs,
                        "coef": None, "se": None, "t_stat": None,
                        "p_val": None, "ci_lower": None, "ci_upper": None,
                        "significance": "", "skipped": skip_reason,
                    })
                    continue

                try:
                    res = fit_plr(X, Y, D) if method == "PLR" else fit_irm(X, Y, D)
                    stars = sig_stars(res["p_val"])
                    print(f"{label}  n={n_obs:<4}  θ̂={res['coef']:+7.3f}  "
                          f"SE={res['se']:.3f}  p={res['p_val']:.4f}{stars}")

                    all_results.append({
                        "exp_id": exp_id, "outcome": "rank_delta",
                        "treatment": t_name, "treatment_col": d_col,
                        "path": path, "method": method, "n_obs": n_obs,
                        "skipped": None, **res, "significance": stars,
                    })
                except Exception as e:
                    print(f"{label}  n={n_obs:<4}  ERROR: {e}")
                    all_results.append({
                        "exp_id": exp_id, "outcome": "rank_delta",
                        "treatment": t_name, "treatment_col": d_col,
                        "path": path, "method": method, "n_obs": n_obs,
                        "coef": None, "se": None, "t_stat": None,
                        "p_val": None, "ci_lower": None, "ci_upper": None,
                        "significance": "", "skipped": f"error: {e}",
                    })

    return pd.DataFrame(all_results)


# ── Output ───────────────────────────────────────────────────────────────────
def print_summary(results_df):
    valid = results_df[results_df["coef"].notna()].copy()
    skipped = results_df[results_df["coef"].isna()]

    print("\n" + "=" * 80)
    print("FULL RESULTS: Y = rank_delta (pre_rank - post_rank)")
    print("  Positive θ̂ → treatment causes LLM to promote results more")
    print("  Negative θ̂ → treatment causes LLM to demote results more")
    print("=" * 80)
    header = (f"{'#':>2}  {'Treatment':<30} {'Path':<5} {'Method':<4}  "
              f"{'n':>4}  {'θ̂':>8}  {'SE':>6}  {'p':>7} {'Sig':>3}  {'95% CI':>22}")
    print(header)
    print("-" * len(header))
    for _, r in results_df.iterrows():
        t_label = f"{r['treatment']} {TREATMENT_LABELS[r['treatment']]}"
        if r["coef"] is None or pd.isna(r["coef"]):
            print(f"{r['exp_id']:2d}  {t_label:<30} "
                  f"{r['path']:<5} {r['method']:<4}  {r['n_obs']:4d}  {'SKIPPED':>8}  "
                  f"{r['skipped']}")
        else:
            ci = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
            print(f"{r['exp_id']:2d}  {t_label:<30} "
                  f"{r['path']:<5} {r['method']:<4}  {r['n_obs']:4d}  {r['coef']:+8.3f}  "
                  f"{r['se']:6.3f}  {r['p_val']:7.4f} {r['significance']:>3}  {ci:>22}")

    # Significant findings
    sig = valid[valid["p_val"] < 0.1].sort_values("p_val")
    if len(sig) > 0:
        print(f"\n{'=' * 80}")
        print(f"SIGNIFICANT FINDINGS (p < 0.1): {len(sig)} of {len(valid)} experiments")
        print("=" * 80)
        for _, r in sig.iterrows():
            direction = "promotes" if r["coef"] > 0 else "demotes"
            t_label = f"{r['treatment']} {TREATMENT_LABELS[r['treatment']]}"
            print(f"  {t_label:<30} ({r['path']}, {r['method']}): "
                  f"θ̂={r['coef']:+.3f}, p={r['p_val']:.4f}{r['significance']}  "
                  f"→ LLM {direction} by {abs(r['coef']):.2f} ranks")
    else:
        print(f"\nNo significant findings at p < 0.1")

    if len(skipped) > 0:
        print(f"\nSkipped {len(skipped)} experiments (imbalanced binary treatment for IRM)")

    # PLR vs IRM comparison
    print(f"\n{'=' * 80}")
    print("PLR vs IRM COMPARISON")
    print("=" * 80)
    for t_name in ["T1", "T2", "T3", "T4"]:
        for path in ["code", "llm"]:
            plr = valid[(valid["treatment"] == t_name) & (valid["path"] == path)
                       & (valid["method"] == "PLR")]
            irm = valid[(valid["treatment"] == t_name) & (valid["path"] == path)
                       & (valid["method"] == "IRM")]
            if len(plr) == 1 and len(irm) == 1:
                p = plr.iloc[0]
                i = irm.iloc[0]
                agree = "agree" if (p["coef"] > 0) == (i["coef"] > 0) else "DISAGREE"
                print(f"  {t_name}_{path:<4}: "
                      f"PLR θ̂={p['coef']:+.3f} (p={p['p_val']:.3f}) | "
                      f"IRM θ̂={i['coef']:+.3f} (p={i['p_val']:.3f})  [{agree}]")
            elif len(plr) == 1:
                p = plr.iloc[0]
                print(f"  {t_name}_{path:<4}: "
                      f"PLR θ̂={p['coef']:+.3f} (p={p['p_val']:.3f}) | IRM skipped")

    # Code vs LLM comparison
    print(f"\n{'=' * 80}")
    print("CODE vs LLM MEASUREMENT COMPARISON (PLR only)")
    print("=" * 80)
    for t_name in ["T1", "T2", "T3", "T4"]:
        code = valid[(valid["treatment"] == t_name) & (valid["path"] == "code")
                    & (valid["method"] == "PLR")]
        llm = valid[(valid["treatment"] == t_name) & (valid["path"] == "llm")
                   & (valid["method"] == "PLR")]
        if len(code) == 1 and len(llm) == 1:
            c = code.iloc[0]
            l = llm.iloc[0]
            agree = "agree" if (c["coef"] > 0) == (l["coef"] > 0) else "DISAGREE"
            print(f"  {t_name} {TREATMENT_LABELS[t_name]:<25}: "
                  f"Code θ̂={c['coef']:+.3f} (p={c['p_val']:.3f}) | "
                  f"LLM θ̂={l['coef']:+.3f} (p={l['p_val']:.3f})  [{agree}]")


def save_outputs(results_df):
    # CSV
    csv_path = OUT_DIR / "all_experiments.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # JSON
    json_path = OUT_DIR / "all_experiments.json"
    output = {
        "study": "GEODML DML Experiment Suite — rank_delta",
        "design": "1 outcome (rank_delta) × 4 treatments × 2 paths × 2 methods = 16 experiments",
        "outcome": {
            "rank_delta": "pre_rank - post_rank (positive = LLM promoted the result)",
        },
        "treatments": {k: TREATMENT_LABELS[k] for k in TREATMENT_LABELS},
        "methods": {
            "PLR": "Partially Linear Regression (continuous D ok)",
            "IRM": "Interactive Regression Model (binary D, ATE)",
        },
        "confounders": CONFOUNDERS,
        "binarization_for_irm": {
            "T1": "above median → 1, else → 0",
            "T4": "> 0 → 1, else → 0",
            "T2": "already binary",
            "T3": "already binary",
        },
        "nuisance_learners": "LGBM (Regressor for PLR ml_l/ml_m and IRM ml_g; Classifier for IRM ml_m)",
        "n_folds": 5,
        "results": results_df.where(results_df.notna(), None).to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved → {json_path}")

    # Plots
    plot_heatmap(results_df)
    plot_coef_comparison(results_df)


def plot_heatmap(results_df):
    """P-value heatmap: rows = treatment×path, columns = method."""
    valid = results_df[results_df["coef"].notna()].copy()

    row_labels = []
    for t in ["T1", "T2", "T3", "T4"]:
        for p in ["code", "llm"]:
            row_labels.append(f"{t} {TREATMENT_LABELS[t]}\n({p})")
    col_labels = ["PLR", "IRM"]

    matrix = np.full((len(row_labels), len(col_labels)), np.nan)
    coef_matrix = np.full((len(row_labels), len(col_labels)), np.nan)

    for _, r in valid.iterrows():
        row_idx = (["T1", "T2", "T3", "T4"].index(r["treatment"]) * 2
                   + ["code", "llm"].index(r["path"]))
        col_idx = METHODS.index(r["method"])
        matrix[row_idx, col_idx] = r["p_val"]
        coef_matrix[row_idx, col_idx] = r["coef"]

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if not np.isnan(matrix[i, j]):
                p = matrix[i, j]
                c = coef_matrix[i, j]
                stars = sig_stars(p)
                text = f"θ̂={c:+.2f}\np={p:.3f}{stars}"
                color = "white" if p < 0.3 or p > 0.8 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)
            else:
                ax.text(j, i, "skipped", ha="center", va="center", fontsize=9, color="gray")

    ax.set_title("Y = rank_delta (pre_rank − post_rank)\n"
                 "Causal Effects & P-values\n"
                 "(green = n.s., red = significant)", fontsize=12)
    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_pvalues.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_DIR / 'heatmap_pvalues.png'}")


def plot_coef_comparison(results_df):
    """Side-by-side PLR vs IRM coefficient plot."""
    valid = results_df[results_df["coef"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        subset = valid[valid["method"] == method].sort_values(["treatment", "path"])

        if len(subset) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(method)
            continue

        labels = [f"{r['treatment']} {TREATMENT_LABELS[r['treatment']]}\n({r['path']})"
                  for _, r in subset.iterrows()]
        coefs = subset["coef"].values
        ci_lo = subset["ci_lower"].values
        ci_hi = subset["ci_upper"].values
        pvals = subset["p_val"].values

        y_pos = np.arange(len(labels))
        colors = ["#2166ac" if p < 0.05 else "#b2182b" if p < 0.1 else "#999999"
                  for p in pvals]

        xerr = [coefs - ci_lo, ci_hi - coefs]
        ax.barh(y_pos, coefs, xerr=xerr, color=colors, alpha=0.7,
                edgecolor="black", linewidth=0.5, capsize=3, ecolor="black", height=0.6)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Method: {method}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Causal effect on rank_delta (θ̂)\n← LLM demotes | LLM promotes →",
                      fontsize=9)
        ax.invert_yaxis()

    fig.suptitle("DML on rank_delta: Treatment Effects with 95% CIs\n"
                 "(blue = p<0.05, red = p<0.1, grey = n.s.)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "coef_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_DIR / 'coef_comparison.png'}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("GEODML — DML Experiment Suite: rank_delta (16 models)")
    print("  Y = rank_delta = pre_rank - post_rank")
    print("      (positive = LLM promoted, negative = LLM demoted)")
    print("  4 treatments (T1-T4) × 2 paths (code, llm) × 2 methods (PLR, IRM)")
    print("=" * 80)
    print()

    results_df = run_all()
    print_summary(results_df)
    save_outputs(results_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
