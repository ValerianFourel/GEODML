#!/usr/bin/env python3
"""DML causal inference analysis with configurable methods, learners, outcomes.

Self-contained script — imports nothing from ../src/.

Uses DoubleML to estimate causal effects of page treatments (T1-T4) on
LLM re-ranking outcomes. Supports PLR, IRM, multiple learners, and
multiple outcome variables.

Usage:
  python pipeline/analyze.py
  python pipeline/analyze.py --input output/small_pool/geodml_dataset.csv --method plr --learner lgbm
  python pipeline/analyze.py --outcome rank_delta --method all --learner all --measurement all
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml

warnings.filterwarnings("ignore", category=UserWarning)

# ── Treatment and confounder definitions ──────────────────────────────────────

# Legacy treatment columns (from gather_data.py code-based + LLM-based extraction)
TREATMENTS_CODE = {
    "T1_code": "T1_statistical_density_code",
    "T2_code": "T2_question_heading_code",
    "T3_code": "T3_structured_data_code",
    "T4_code": "T4_citation_authority_code",
}
TREATMENTS_LLM = {
    "T1_llm": "T1_statistical_density_llm",
    "T2_llm": "T2_question_heading_llm",
    "T3_llm": "T3_structured_data_llm",
    "T4_llm": "T4_citation_authority_llm",
}

# New treatment columns (from extract_features.py)
TREATMENTS_NEW = {
    "T1a_stats_present": "treat_stats_present",
    "T1b_stats_density": "treat_stats_density",
    "T2a_question_headings": "treat_question_headings",
    "T2b_structural_modularity": "treat_structural_modularity",
    "T3_structured_data_new": "treat_structured_data",
    "T4a_ext_citations": "treat_ext_citations_any",
    "T4b_auth_citations": "treat_auth_citations",
    "T5_topical_comp": "treat_topical_comp",
    "T6_freshness": "treat_freshness",
    "T7_source_earned": "treat_source_earned",
}

TREATMENT_LABELS = {
    # Legacy
    "T1_code": "T1 Statistical Density (code)",
    "T2_code": "T2 Question Headings (code)",
    "T3_code": "T3 Structured Data (code)",
    "T4_code": "T4 Citation Authority (code)",
    "T1_llm": "T1 Statistical Density (LLM)",
    "T2_llm": "T2 Question Headings (LLM)",
    "T3_llm": "T3 Structured Data (LLM)",
    "T4_llm": "T4 Citation Authority (LLM)",
    # New
    "T1a_stats_present": "T1a Stats Present (binary)",
    "T1b_stats_density": "T1b Stats Density (continuous)",
    "T2a_question_headings": "T2a Question Headings (binary)",
    "T2b_structural_modularity": "T2b Structural Modularity (count)",
    "T3_structured_data_new": "T3 Structured Data (expanded)",
    "T4a_ext_citations": "T4a External Citations (binary)",
    "T4b_auth_citations": "T4b Authority Citations (count)",
    "T5_topical_comp": "T5 Topical Competence (cosine)",
    "T6_freshness": "T6 Freshness (ordinal 0-4)",
    "T7_source_earned": "T7 Source: Earned",
}

# New confounder columns (from extract_features.py)
CONFOUNDERS_NEW = [
    "conf_title_kw_sim",
    "conf_snippet_kw_sim",
    "conf_title_len",
    "conf_snippet_len",
    "conf_brand_recog",
    "conf_title_has_kw",
    "conf_word_count",
    "conf_readability",
    "conf_internal_links",
    "conf_outbound_links",
    "conf_images_alt",
    "conf_bm25",
    "conf_https",
    "conf_domain_authority",
    "conf_backlinks",
    "conf_referring_domains",
    "conf_serp_position",
]

# Legacy confounder columns (from gather_data.py)
CONFOUNDERS_LEGACY = [
    "X1_domain_authority",
    "X2_domain_age_years",
    "X3_word_count",
    "X6_readability",
    "X7_internal_links",
    "X7B_outbound_links",
    "X8_keyword_difficulty",
    "X9_images_with_alt",
]

# Default: prefer new columns, fall back to legacy
CONFOUNDERS = CONFOUNDERS_NEW

OUTCOMES = ["rank_delta", "pre_rank", "post_rank"]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df, treatment_col, outcome_col, confounders):
    """Impute missing values, standardize confounders, return clean arrays."""
    cols_needed = confounders + [treatment_col, outcome_col]
    sub = df[cols_needed].copy()

    # Report missingness
    miss = sub.isna().sum()
    miss_pct = (miss / len(sub) * 100).round(1)
    missing_info = {c: f"{miss[c]}/{len(sub)} ({miss_pct[c]}%)" for c in cols_needed if miss[c] > 0}

    # Drop rows where treatment or outcome is missing
    sub = sub.dropna(subset=[treatment_col, outcome_col])
    n_after = len(sub)

    if n_after < 10:
        return None, None, None, n_after, missing_info

    # Median imputation for confounders
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(sub[confounders]),
        columns=confounders, index=sub.index,
    )

    # Standardize confounders
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=confounders, index=sub.index,
    )

    Y = sub[outcome_col].values
    D = sub[treatment_col].values

    return X_scaled, Y, D, n_after, missing_info


# ── DML Estimation ────────────────────────────────────────────────────────────

def _get_learners(learner_type, method):
    """Return (ml_l or ml_g, ml_m) learner pair."""
    if learner_type == "lgbm":
        from lightgbm import LGBMRegressor, LGBMClassifier
        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        if method == "irm":
            ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  num_leaves=31, verbose=-1, random_state=42)
    else:  # rf
        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        if method == "irm":
            ml_m = RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=42, n_jobs=-1)
    return ml_l, ml_m


def run_dml(X, Y, D, method="plr", learner_type="lgbm", n_folds=5):
    """Run DoubleML and return results dict."""
    ml_l, ml_m = _get_learners(learner_type, method)

    dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)

    if method == "plr":
        model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                n_folds=n_folds, score="partialling out")
    elif method == "irm":
        # IRM needs binary treatment; binarize at median if continuous
        if len(np.unique(D)) > 2:
            median_d = np.median(D)
            D_binary = (D > median_d).astype(float)
            dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D_binary)
        model = dml.DoubleMLIRM(dml_data, ml_g=ml_l, ml_m=ml_m,
                                n_folds=n_folds, score="ATE")
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit()

    coef = model.coef[0]
    se = model.se[0]
    t_stat = model.t_stat[0]
    p_val = model.pval[0]
    ci = model.confint(level=0.95)
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]

    return {
        "coef": coef, "se": se, "t_stat": t_stat, "p_val": p_val,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
    }


def run_ols(X, Y, D):
    """Run naive OLS for comparison (no causal correction)."""
    from numpy.linalg import lstsq
    # Y = alpha + beta*D + gamma*X + epsilon
    n = len(Y)
    design = np.column_stack([np.ones(n), D, X.values])
    coeffs, _, _, _ = lstsq(design, Y, rcond=None)
    beta = coeffs[1]

    # Compute SE
    residuals = Y - design @ coeffs
    sigma2 = np.sum(residuals ** 2) / (n - design.shape[1])
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(cov[1, 1])

    from scipy import stats
    t_stat = beta / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - design.shape[1]))
    ci_lower = beta - 1.96 * se
    ci_upper = beta + 1.96 * se

    return {
        "coef": beta, "se": se, "t_stat": t_stat, "p_val": p_val,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def significance_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


def interpret(treatment_name, coef, p_val, outcome="rank_delta"):
    """Interpret DML coefficient.

    Rank semantics: rank 1 is best, rank 10 is worst.
    - rank_delta = pre_rank - post_rank: positive = LLM promoted it (improved)
    - pre_rank / post_rank: negative coef = better rank (closer to 1)
    """
    magnitude = abs(coef)
    sig = "significantly " if p_val < 0.05 else "not significantly "
    label = TREATMENT_LABELS.get(treatment_name, treatment_name)

    if outcome == "rank_delta":
        # positive coef = larger delta = more improvement by LLM
        direction = "improves LLM re-ranking" if coef > 0 else "worsens LLM re-ranking"
        return f"{label} {sig}{direction} by {magnitude:.2f} positions (p={p_val:.3f})"
    else:
        # pre_rank / post_rank: negative coef = better (lower) rank
        direction = "improves" if coef < 0 else "worsens"
        return f"{label} {sig}{direction} {outcome} by {magnitude:.2f} positions (p={p_val:.3f})"


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_coefficients(results, output_path):
    """Generate a coefficient plot with 95% CIs."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.5)))

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
    ax.set_xlabel("Causal Effect (theta)", fontsize=11)
    ax.set_title("DML Causal Effect Estimates\n(blue=p<0.05, red=p<0.1, grey=n.s.)", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_heatmap(all_results, output_path):
    """Generate a p-value heatmap across treatments and methods."""
    if not all_results:
        return

    df = pd.DataFrame(all_results)
    if df.empty:
        return

    # Create pivot: treatment × (method_learner_outcome)
    df["config"] = df["method"] + "_" + df["learner"] + "_" + df["outcome"]
    pivot = df.pivot_table(index="treatment", columns="config", values="p_val")

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.2)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if val < 0.1 else "black")

    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_title("P-value Heatmap Across Specifications", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DML causal inference analysis")
    parser.add_argument("--input", type=str, default="output/geodml_dataset.csv",
                        help="Input CSV (default: output/geodml_dataset.csv)")
    parser.add_argument("--output-dir", type=str, default="pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/",
                        help="Output directory for results")
    parser.add_argument("--outcome", type=str, default="all",
                        choices=["rank_delta", "pre_rank", "post_rank", "all"],
                        help="Outcome variable (default: all)")
    parser.add_argument("--method", type=str, default="all",
                        choices=["plr", "irm", "all"],
                        help="DML method (default: all)")
    parser.add_argument("--learner", type=str, default="all",
                        choices=["lgbm", "rf", "all"],
                        help="Nuisance learner (default: all)")
    parser.add_argument("--measurement", type=str, default="all",
                        choices=["code", "llm", "new", "all"],
                        help="Treatment measurement method (default: all)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of cross-fitting folds (default: 5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Basic data summary
    for outcome in OUTCOMES:
        if outcome in df.columns:
            valid = df[outcome].dropna()
            print(f"  {outcome}: n={len(valid)}, mean={valid.mean():.2f}, "
                  f"std={valid.std():.2f}, range=[{valid.min():.0f}, {valid.max():.0f}]")

    # ── Determine experiment grid ─────────────────────────────────────────
    outcomes = OUTCOMES if args.outcome == "all" else [args.outcome]
    methods = ["plr", "irm"] if args.method == "all" else [args.method]
    learners = ["lgbm", "rf"] if args.learner == "all" else [args.learner]

    if args.measurement == "all":
        treatments = {**TREATMENTS_CODE, **TREATMENTS_LLM, **TREATMENTS_NEW}
    elif args.measurement == "code":
        treatments = TREATMENTS_CODE
    elif args.measurement == "new":
        treatments = TREATMENTS_NEW
    else:
        treatments = TREATMENTS_LLM

    # Determine which confounder set to use
    if args.measurement == "new":
        confounders_list = CONFOUNDERS_NEW
    elif args.measurement in ("code", "llm"):
        confounders_list = CONFOUNDERS_LEGACY
    else:
        # "all" — prefer new confounders if available, fall back to legacy
        confounders_list = CONFOUNDERS_NEW

    # Filter to treatments that actually exist in the data and have variance
    available_treatments = {}
    for name, col in treatments.items():
        if col not in df.columns or df[col].notna().sum() == 0:
            print(f"  Skipping {name} ({col}): not available in data")
        elif df[col].dropna().nunique() < 2:
            print(f"  Skipping {name} ({col}): zero variance (all values identical)")
        else:
            available_treatments[name] = col

    if not available_treatments:
        print("\nNo treatments available. Check your input data.")
        return

    # Filter confounders to those available in data
    available_confounders = [c for c in confounders_list if c in df.columns]
    # Drop all-NaN confounders
    available_confounders = [c for c in available_confounders
                            if df[c].notna().sum() > 0]
    # Drop zero-variance confounders
    available_confounders = [c for c in available_confounders
                            if df[c].dropna().nunique() > 1]
    if len(available_confounders) < len(confounders_list):
        dropped = set(confounders_list) - set(available_confounders)
        if dropped:
            print(f"  Auto-dropped confounders (missing/zero-var): {sorted(dropped)}")

    total_experiments = len(outcomes) * len(available_treatments) * len(methods) * len(learners)
    print(f"\nExperiment grid: {len(outcomes)} outcomes x {len(available_treatments)} treatments "
          f"x {len(methods)} methods x {len(learners)} learners = {total_experiments} experiments")
    print(f"Confounders ({len(available_confounders)}): {available_confounders}")
    print()

    # ── Run experiments ───────────────────────────────────────────────────
    all_results = []
    confounder_importances = []
    experiment_num = 0

    for outcome in outcomes:
        if outcome not in df.columns:
            print(f"Outcome '{outcome}' not in data, skipping.")
            continue

        for treatment_name, treatment_col in available_treatments.items():
            for method in methods:
                for learner in learners:
                    experiment_num += 1
                    label = TREATMENT_LABELS.get(treatment_name, treatment_name)
                    config = f"[{experiment_num}/{total_experiments}] {label} | {outcome} | {method} | {learner}"
                    print(f"{config}")

                    X, Y, D, n_obs, missing_info = preprocess(
                        df, treatment_col, outcome, available_confounders
                    )

                    if X is None:
                        print(f"  Skipped: only {n_obs} valid observations")
                        continue

                    print(f"  n={n_obs}", end="")
                    if missing_info:
                        print(f"  missing: {missing_info}", end="")

                    # DML
                    try:
                        res = run_dml(X, Y, D, method=method, learner_type=learner,
                                      n_folds=args.n_folds)
                        stars = significance_stars(res["p_val"])
                        print(f"  DML: theta={res['coef']:+.3f} SE={res['se']:.3f} "
                              f"p={res['p_val']:.4f}{stars} "
                              f"CI=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
                    except Exception as e:
                        print(f"  DML Error: {str(e)[:100]}")
                        res = {"coef": None, "se": None, "t_stat": None,
                               "p_val": None, "ci_lower": None, "ci_upper": None}

                    # OLS (only for PLR to avoid redundant OLS runs)
                    ols_res = {"coef": None, "se": None, "p_val": None}
                    if method == "plr":
                        try:
                            ols_res = run_ols(X, Y, D)
                            print(f"  OLS: beta={ols_res['coef']:+.3f} SE={ols_res['se']:.3f} "
                                  f"p={ols_res['p_val']:.4f}")
                        except Exception as e:
                            print(f"  OLS Error: {str(e)[:80]}")

                    # ── Confounder importance + significance ──────────────────
                    if learner == "lgbm" and res["coef"] is not None:
                        try:
                            from lightgbm import LGBMRegressor as _LGBMR
                            from scipy import stats as _stats

                            # LightGBM feature importances
                            lgb_y = _LGBMR(n_estimators=200, learning_rate=0.05,
                                           max_depth=5, verbose=-1, random_state=42)
                            lgb_y.fit(X, Y)
                            imp_y = lgb_y.feature_importances_

                            lgb_d = _LGBMR(n_estimators=200, learning_rate=0.05,
                                           max_depth=5, verbose=-1, random_state=42)
                            lgb_d.fit(X, D)
                            imp_d = lgb_d.feature_importances_

                            # OLS p-values for each confounder on Y and D
                            n_obs_x = X.shape[0]
                            X_ols = np.column_stack([np.ones(n_obs_x), X.values])
                            dof = n_obs_x - X_ols.shape[1]

                            def _ols_pvals(X_design, y_vec, dof_):
                                coeffs, _, _, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
                                resid = y_vec - X_design @ coeffs
                                sigma2 = np.sum(resid ** 2) / dof_
                                cov = sigma2 * np.linalg.inv(X_design.T @ X_design)
                                se = np.sqrt(np.diag(cov))
                                t_vals = coeffs / np.where(se > 0, se, 1.0)
                                pvals = 2 * (1 - _stats.t.cdf(np.abs(t_vals), df=dof_))
                                # skip intercept (index 0)
                                return coeffs[1:], se[1:], pvals[1:]

                            coefs_y, ses_y, pvals_y = _ols_pvals(X_ols, Y, dof)
                            coefs_d, ses_d, pvals_d = _ols_pvals(X_ols, D, dof)

                            for k, conf_name in enumerate(available_confounders):
                                confounder_importances.append({
                                    "treatment": treatment_name,
                                    "outcome": outcome,
                                    "learner": learner,
                                    "confounder": conf_name,
                                    "importance_outcome": float(imp_y[k]),
                                    "importance_treatment": float(imp_d[k]),
                                    "coef_outcome": float(coefs_y[k]),
                                    "se_outcome": float(ses_y[k]),
                                    "pval_outcome": float(pvals_y[k]),
                                    "coef_treatment": float(coefs_d[k]),
                                    "se_treatment": float(ses_d[k]),
                                    "pval_treatment": float(pvals_d[k]),
                                })
                        except Exception as e:
                            print(f"  Confounder importance error: {str(e)[:80]}")

                    all_results.append({
                        "treatment": treatment_name,
                        "treatment_col": treatment_col,
                        "label": label,
                        "outcome": outcome,
                        "method": method,
                        "learner": learner,
                        "n_obs": n_obs,
                        "coef": res["coef"],
                        "se": res["se"],
                        "t_stat": res["t_stat"],
                        "p_val": res["p_val"],
                        "ci_lower": res["ci_lower"],
                        "ci_upper": res["ci_upper"],
                        "significance": significance_stars(res["p_val"]) if res["p_val"] is not None else "",
                        "ols_coef": ols_res.get("coef"),
                        "ols_se": ols_res.get("se"),
                        "ols_pval": ols_res.get("p_val"),
                    })
                    print()

    if not all_results:
        print("No experiments completed.")
        return

    # ── Summary Table ─────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Treatment':<30} {'Outcome':<12} {'Method':<5} {'Learner':<6} {'theta':>7} {'SE':>7} {'p-val':>8} {'Sig':>4}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r["coef"] is not None:
            print(f"{r['label']:<30} {r['outcome']:<12} {r['method']:<5} {r['learner']:<6} "
                  f"{r['coef']:+7.3f} {r['se']:7.3f} {r['p_val']:8.4f} {r['significance']:>4}")
    print()

    # ── Code vs LLM comparison ────────────────────────────────────────────
    # Find matching code/llm pairs for default outcome + method + learner
    code_results = {r["treatment"]: r for r in all_results
                    if r["treatment"].endswith("_code") and r["coef"] is not None}
    llm_results = {r["treatment"]: r for r in all_results
                   if r["treatment"].endswith("_llm") and r["coef"] is not None}

    if code_results and llm_results:
        print("COMPARISON: Code-based vs LLM-based treatment measurement")
        print("-" * 60)
        for i in range(1, 5):
            code_key = f"T{i}_code"
            llm_key = f"T{i}_llm"
            if code_key in code_results and llm_key in llm_results:
                cr = code_results[code_key]
                lr = llm_results[llm_key]
                print(f"  T{i}: Code theta={cr['coef']:+.3f} (p={cr['p_val']:.3f})  |  "
                      f"LLM theta={lr['coef']:+.3f} (p={lr['p_val']:.3f})")
        print()

    # ── Interpretations ───────────────────────────────────────────────────
    print("INTERPRETATIONS (first spec per treatment):")
    seen_treatments = set()
    for r in all_results:
        if r["treatment"] not in seen_treatments and r["coef"] is not None:
            seen_treatments.add(r["treatment"])
            print(f"  - {interpret(r['treatment'], r['coef'], r['p_val'], r['outcome'])}")
    print()

    # ── Save outputs ──────────────────────────────────────────────────────
    # CSV
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "all_experiments.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results table -> {csv_path}")

    # JSON
    json_path = output_dir / "summary.json"
    output_json = {
        "study": "DML Causal Inference: Page Features -> LLM Re-Ranking",
        "methods": methods,
        "learners": learners,
        "outcomes": outcomes,
        "n_folds": args.n_folds,
        "confounders": available_confounders,
        "total_experiments": len(all_results),
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"Saved summary JSON -> {json_path}")

    # Coefficient plot (first method/learner combo, all treatments for rank_delta)
    primary_results = [r for r in all_results
                       if r["outcome"] == "rank_delta" and r["method"] == methods[0]
                       and r["learner"] == learners[0] and r["coef"] is not None]
    if primary_results:
        coef_plot_path = output_dir / "dml_coefficients.png"
        plot_coefficients(primary_results, coef_plot_path)
        print(f"Saved coefficient plot -> {coef_plot_path}")

    # P-value heatmap
    valid_results = [r for r in all_results if r["p_val"] is not None]
    if valid_results:
        heatmap_path = output_dir / "pvalue_heatmap.png"
        plot_pvalue_heatmap(valid_results, heatmap_path)
        print(f"Saved p-value heatmap -> {heatmap_path}")

    # Confounder importances CSV
    if confounder_importances:
        conf_df = pd.DataFrame(confounder_importances)
        conf_path = output_dir / "confounder_importances.csv"
        conf_df.to_csv(conf_path, index=False)
        print(f"Saved confounder importances -> {conf_path}")

    print(f"\nDone. {len(all_results)} experiments completed.")


if __name__ == "__main__":
    main()
