#!/usr/bin/env python3
"""Cross-model DML analysis for the paper-size experiment.

Analyzes the merged dataset across all (model, pool_size) combinations.
Produces:
  - Per-model DML results
  - Per-pool-size DML results
  - Cross-model robustness comparison
  - Pool-size effect analysis
  - Publication-quality plots

Usage:
  python paperSizeExperiment/analyze_cross_model.py
  python paperSizeExperiment/analyze_cross_model.py --input output/merged_all_runs.csv
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

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    ALL_TREATMENTS, TREATMENTS_CODE, TREATMENTS_LLM, TREATMENTS_NEW,
    TREATMENT_LABELS, CONFOUNDERS, CONFOUNDERS_LEGACY,
    DML_METHODS, DML_LEARNERS, DML_N_FOLDS, DML_OUTCOMES,
    OUTPUT_ROOT,
)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df, treatment_col, outcome_col, confounders):
    cols_needed = confounders + [treatment_col, outcome_col]
    sub = df[cols_needed].copy()

    miss = sub.isna().sum()
    miss_pct = (miss / len(sub) * 100).round(1)
    missing_info = {c: f"{miss[c]}/{len(sub)} ({miss_pct[c]}%)" for c in cols_needed if miss[c] > 0}

    sub = sub.dropna(subset=[treatment_col, outcome_col])
    n_after = len(sub)

    if n_after < 10:
        return None, None, None, n_after, missing_info

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(sub[confounders]),
        columns=confounders, index=sub.index,
    )

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=confounders, index=sub.index,
    )

    Y = sub[outcome_col].values
    D = sub[treatment_col].values

    return X_scaled, Y, D, n_after, missing_info


def _get_learners(learner_type, method):
    if learner_type == "lgbm":
        from lightgbm import LGBMRegressor, LGBMClassifier
        ml_l = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        ml_m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             num_leaves=31, verbose=-1, random_state=42)
        if method == "irm":
            ml_m = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  num_leaves=31, verbose=-1, random_state=42)
    else:
        ml_l = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        ml_m = RandomForestRegressor(n_estimators=200, max_depth=5,
                                     random_state=42, n_jobs=-1)
        if method == "irm":
            ml_m = RandomForestClassifier(n_estimators=200, max_depth=5,
                                          random_state=42, n_jobs=-1)
    return ml_l, ml_m


def run_dml(X, Y, D, method="plr", learner_type="lgbm", n_folds=5):
    ml_l, ml_m = _get_learners(learner_type, method)
    dml_data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)

    if method == "plr":
        model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                                n_folds=n_folds, score="partialling out")
    elif method == "irm":
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


def significance_stars(p):
    if p is None:
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


# ── Analysis Functions ────────────────────────────────────────────────────────

def analyze_subset(df, subset_label, treatments, confounders, outcomes, methods, learners, n_folds):
    """Run DML on a subset of data. Returns list of result dicts."""
    results = []

    # Filter to treatments available in data
    available_treatments = {}
    for name, col in treatments.items():
        if col in df.columns and df[col].notna().sum() > 0 and df[col].dropna().nunique() >= 2:
            available_treatments[name] = col

    available_confounders = [c for c in confounders if c in df.columns
                             and df[c].notna().sum() > 0 and df[c].dropna().nunique() > 1]

    if not available_treatments:
        print(f"  [{subset_label}] No treatments available, skipping.")
        return results

    if not available_confounders:
        # Try legacy confounders
        available_confounders = [c for c in CONFOUNDERS_LEGACY if c in df.columns
                                 and df[c].notna().sum() > 0 and df[c].dropna().nunique() > 1]

    print(f"  [{subset_label}] {len(available_treatments)} treatments, "
          f"{len(available_confounders)} confounders, {len(df)} rows")

    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for treatment_name, treatment_col in available_treatments.items():
            for method in methods:
                for learner in learners:
                    label = TREATMENT_LABELS.get(treatment_name, treatment_name)

                    X, Y, D, n_obs, missing_info = preprocess(
                        df, treatment_col, outcome, available_confounders
                    )

                    if X is None:
                        continue

                    try:
                        res = run_dml(X, Y, D, method=method, learner_type=learner,
                                      n_folds=n_folds)
                        stars = significance_stars(res["p_val"])
                        print(f"    {label:<40} {outcome:<12} {method} {learner} "
                              f"n={n_obs} theta={res['coef']:+.3f} p={res['p_val']:.4f}{stars}")
                    except Exception as e:
                        print(f"    {label:<40} ERROR: {str(e)[:60]}")
                        res = {"coef": None, "se": None, "t_stat": None,
                               "p_val": None, "ci_lower": None, "ci_upper": None}

                    results.append({
                        "subset": subset_label,
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
                        "significance": significance_stars(res["p_val"]),
                    })

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cross_model_coefficients(all_results_df, output_dir):
    """Plot treatment coefficients grouped by model and pool size."""
    if all_results_df.empty:
        return

    # Filter to rank_delta + plr + lgbm for clarity
    df = all_results_df[
        (all_results_df["outcome"] == "rank_delta") &
        (all_results_df["method"] == "plr") &
        (all_results_df["learner"] == "lgbm") &
        (all_results_df["coef"].notna())
    ].copy()

    if df.empty:
        return

    subsets = sorted(df["subset"].unique())
    treatments = sorted(df["treatment"].unique())

    if not treatments or not subsets:
        return

    fig, ax = plt.subplots(figsize=(14, max(5, len(treatments) * 0.6)))

    n_subsets = len(subsets)
    bar_height = 0.8 / n_subsets
    colors = plt.cm.Set2(np.linspace(0, 1, n_subsets))

    y_positions = np.arange(len(treatments))

    for j, subset in enumerate(subsets):
        sub_df = df[df["subset"] == subset]
        coefs = []
        ci_lowers = []
        ci_uppers = []
        for t in treatments:
            row = sub_df[sub_df["treatment"] == t]
            if len(row) > 0:
                row = row.iloc[0]
                coefs.append(row["coef"])
                ci_lowers.append(row["coef"] - row["ci_lower"])
                ci_uppers.append(row["ci_upper"] - row["coef"])
            else:
                coefs.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)

        offset = (j - n_subsets / 2 + 0.5) * bar_height
        ax.barh(y_positions + offset, coefs,
                xerr=[ci_lowers, ci_uppers],
                height=bar_height, color=colors[j], alpha=0.8,
                edgecolor="black", linewidth=0.3,
                capsize=2, ecolor="grey", label=subset)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions)
    labels = [TREATMENT_LABELS.get(t, t) for t in treatments]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Causal Effect (theta)", fontsize=11)
    ax.set_title("DML Coefficients Across Models & Pool Sizes\n(rank_delta, PLR, LGBM)", fontsize=12)
    ax.legend(loc="lower right", fontsize=7, title="Run", title_fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(output_dir / "cross_model_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cross-model coefficient plot")


def plot_robustness_heatmap(all_results_df, output_dir):
    """Heatmap: treatment significance across all runs."""
    if all_results_df.empty:
        return

    df = all_results_df[
        (all_results_df["outcome"] == "rank_delta") &
        (all_results_df["method"] == "plr") &
        (all_results_df["learner"] == "lgbm") &
        (all_results_df["p_val"].notna())
    ].copy()

    if df.empty:
        return

    pivot = df.pivot_table(index="treatment", columns="subset", values="p_val")
    if pivot.empty:
        return

    # Rename treatments for readability
    pivot.index = [TREATMENT_LABELS.get(t, t) for t in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                     max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.2)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if val < 0.1 else "black")

    plt.colorbar(im, ax=ax, label="p-value")
    ax.set_title("Treatment Robustness: P-values Across Specifications", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "robustness_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved robustness heatmap")


def plot_pool_size_comparison(all_results_df, output_dir):
    """Compare coefficients between pool sizes for each model."""
    if all_results_df.empty:
        return

    df = all_results_df[
        (all_results_df["outcome"] == "rank_delta") &
        (all_results_df["method"] == "plr") &
        (all_results_df["learner"] == "lgbm") &
        (all_results_df["coef"].notna())
    ].copy()

    if df.empty or "subset" not in df.columns:
        return

    # Extract pool size from subset label
    df["pool_info"] = df["subset"].str.extract(r"(serp\d+_top\d+)")

    pools = sorted(df["pool_info"].dropna().unique())
    if len(pools) < 2:
        return

    treatments = sorted(df["treatment"].unique())
    if not treatments:
        return

    fig, axes = plt.subplots(1, len(treatments), figsize=(max(12, len(treatments) * 2.5), 5),
                              sharey=False)
    if len(treatments) == 1:
        axes = [axes]

    for idx, treatment in enumerate(treatments):
        ax = axes[idx]
        t_df = df[df["treatment"] == treatment]

        for pool in pools:
            pool_df = t_df[t_df["pool_info"] == pool]
            if pool_df.empty:
                continue
            coefs = pool_df["coef"].values
            ci_lo = pool_df["ci_lower"].values
            ci_hi = pool_df["ci_upper"].values
            models = pool_df["subset"].str.replace(f"_{pool}", "").values

            y = np.arange(len(models))
            ax.errorbar(coefs, y, xerr=[coefs - ci_lo, ci_hi - coefs],
                        fmt="o", capsize=3, label=pool, markersize=4)

        ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_title(TREATMENT_LABELS.get(treatment, treatment), fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)

    plt.suptitle("Pool Size Effect on Treatment Coefficients", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "pool_size_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pool size comparison plot")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model DML analysis")
    parser.add_argument("--input", type=str, default=str(OUTPUT_ROOT / "merged_all_runs.csv"),
                        help="Merged dataset CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT / "cross_model_analysis"),
                        help="Output directory for results")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"  Keywords: {df['keyword'].nunique()}")
    print(f"  Domains:  {df['domain'].nunique()}")

    if "run_id" in df.columns:
        print(f"  Runs:     {df['run_id'].nunique()}")
        for run_id in sorted(df["run_id"].unique()):
            n = len(df[df["run_id"] == run_id])
            print(f"    {run_id}: {n} rows")

    # Determine confounders
    available_confounders = [c for c in CONFOUNDERS if c in df.columns
                             and df[c].notna().sum() > 0 and df[c].dropna().nunique() > 1]
    if not available_confounders:
        available_confounders = [c for c in CONFOUNDERS_LEGACY if c in df.columns
                                 and df[c].notna().sum() > 0 and df[c].dropna().nunique() > 1]

    outcomes = [o for o in DML_OUTCOMES if o in df.columns]
    methods = DML_METHODS
    learners = DML_LEARNERS

    all_results = []

    # ── 1. Per-run analysis ───────────────────────────────────────────────
    if "run_id" in df.columns:
        print(f"\n{'='*70}")
        print("PER-RUN DML ANALYSIS")
        print(f"{'='*70}")
        for run_id in sorted(df["run_id"].unique()):
            run_df = df[df["run_id"] == run_id]
            results = analyze_subset(
                run_df, run_id, ALL_TREATMENTS, available_confounders,
                outcomes, methods, learners, DML_N_FOLDS
            )
            all_results.extend(results)

    # ── 2. Pooled analysis (all runs combined) ────────────────────────────
    print(f"\n{'='*70}")
    print("POOLED ANALYSIS (all runs combined)")
    print(f"{'='*70}")
    results = analyze_subset(
        df, "POOLED", ALL_TREATMENTS, available_confounders,
        outcomes, methods, learners, DML_N_FOLDS
    )
    all_results.extend(results)

    # ── 3. Per-pool-size analysis ─────────────────────────────────────────
    if "serp_pool_size" in df.columns:
        print(f"\n{'='*70}")
        print("PER-POOL-SIZE ANALYSIS")
        print(f"{'='*70}")
        for pool_size in sorted(df["serp_pool_size"].dropna().unique()):
            pool_df = df[df["serp_pool_size"] == pool_size]
            results = analyze_subset(
                pool_df, f"pool_serp{int(pool_size)}", ALL_TREATMENTS, available_confounders,
                outcomes, methods, learners, DML_N_FOLDS
            )
            all_results.extend(results)

    # ── 4. Per-model analysis ─────────────────────────────────────────────
    if "llm_model" in df.columns:
        print(f"\n{'='*70}")
        print("PER-MODEL ANALYSIS")
        print(f"{'='*70}")
        for model in sorted(df["llm_model"].dropna().unique()):
            model_df = df[df["llm_model"] == model]
            model_short = model.split("/")[-1] if "/" in model else model
            results = analyze_subset(
                model_df, f"model_{model_short}", ALL_TREATMENTS, available_confounders,
                outcomes, methods, learners, DML_N_FOLDS
            )
            all_results.extend(results)

    # ── Save results ──────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results produced.")
        return

    results_df = pd.DataFrame(all_results)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (rank_delta, PLR, LGBM)")
    print(f"{'='*70}")
    summary = results_df[
        (results_df["outcome"] == "rank_delta") &
        (results_df["method"] == "plr") &
        (results_df["learner"] == "lgbm") &
        (results_df["coef"].notna())
    ]
    if not summary.empty:
        header = f"{'Subset':<45} {'Treatment':<30} {'n':>5} {'theta':>7} {'SE':>7} {'p-val':>8} {'Sig':>4}"
        print(header)
        print("-" * len(header))
        for _, r in summary.iterrows():
            print(f"{r['subset']:<45} {r['label']:<30} {r['n_obs']:>5} "
                  f"{r['coef']:+7.3f} {r['se']:7.3f} {r['p_val']:8.4f} {r['significance']:>4}")

    # Robustness report: treatments significant across all runs
    print(f"\n{'='*70}")
    print("ROBUSTNESS REPORT")
    print(f"{'='*70}")
    if "run_id" in df.columns:
        per_run = results_df[
            (results_df["subset"] != "POOLED") &
            (~results_df["subset"].str.startswith("pool_").fillna(False)) &
            (~results_df["subset"].str.startswith("model_").fillna(False)) &
            (results_df["outcome"] == "rank_delta") &
            (results_df["method"] == "plr") &
            (results_df["learner"] == "lgbm") &
            (results_df["p_val"].notna())
        ]
        if not per_run.empty:
            for treatment in sorted(per_run["treatment"].unique()):
                t_results = per_run[per_run["treatment"] == treatment]
                n_sig = (t_results["p_val"] < 0.05).sum()
                n_total = len(t_results)
                avg_coef = t_results["coef"].mean()
                sign_consistent = (t_results["coef"] > 0).all() or (t_results["coef"] < 0).all()
                label = TREATMENT_LABELS.get(treatment, treatment)
                status = "ROBUST" if n_sig == n_total and sign_consistent else "FRAGILE" if n_sig == 0 else "MIXED"
                print(f"  {label:<40} sig={n_sig}/{n_total} avg_theta={avg_coef:+.3f} sign_consistent={sign_consistent} [{status}]")

    # Save CSV
    csv_path = output_dir / "all_cross_model_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV -> {csv_path}")

    # Save JSON summary
    json_path = output_dir / "summary.json"
    summary_json = {
        "total_experiments": len(all_results),
        "subsets": sorted(results_df["subset"].unique().tolist(), key=str),
        "treatments": sorted(results_df["treatment"].unique().tolist(), key=str),
        "outcomes": outcomes,
        "confounders": available_confounders,
    }
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"Saved summary JSON -> {json_path}")

    # Plots
    print(f"\nGenerating plots...")
    plot_cross_model_coefficients(results_df, output_dir)
    plot_robustness_heatmap(results_df, output_dir)
    plot_pool_size_comparison(results_df, output_dir)

    print(f"\nDone. {len(all_results)} experiments across {results_df['subset'].nunique()} subsets.")


if __name__ == "__main__":
    main()
