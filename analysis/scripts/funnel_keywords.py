#!/usr/bin/env python3
"""By-keyword table of the exploratory review (Mac, CPU; addendum A1 of funnel_study.md).

One row per keyword: prompts and natural answers per model; the within-keyword slope of the cited
sources' intent on the prompt position x (prompt-level means, least squares) with its standard error,
an empirical-Bayes shrunken slope (DerSimonian-Laird between-keyword variance) and a 95% posterior
interval; DataForSEO intent class, difficulty, volume and CPC; off-topic, glued-title and Google
top-20 shares of the cited sources; mean log organic count of their domains; glued-row exposure of the
keyword's own snapshot rows; Gemma mean support (development judgments; empty until the HoreKa
extract is supplied); three example prompts (lowest, middle, highest x). Plus a heterogeneity test
(Q of the keyword slopes against a within-keyword permutation null), pooled slopes by intent class and
difficulty tercile, and the top and bottom keywords by shrunken slope. Associations only.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402


def keyword_slopes(prompt_x: np.ndarray, prompt_y: np.ndarray, prompt_kw: np.ndarray, n_kw: int):
    """Least-squares slope of y on x within each keyword (prompt-level means) and its standard error."""
    slope, se, n = np.full(n_kw, np.nan), np.full(n_kw, np.nan), np.zeros(n_kw, np.int64)
    order = np.argsort(prompt_kw, kind="stable")
    cuts = np.flatnonzero(np.diff(prompt_kw[order])) + 1
    for idx in np.split(order, cuts):
        if not len(idx):
            continue
        k = prompt_kw[idx[0]]
        x, y = prompt_x[idx], prompt_y[idx]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        n[k] = len(x)
        if len(x) < 5 or np.var(x) == 0:
            continue
        xc = x - x.mean()
        b = float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))
        resid = y - y.mean() - b * xc
        slope[k] = b
        se[k] = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2) / np.sum(xc * xc)))
    return slope, se, n


def shrink(slope: np.ndarray, se: np.ndarray) -> dict:
    """Empirical-Bayes normal-normal shrinkage. tau2 is the DerSimonian-Laird between-keyword variance
    (Q around the fixed-effect mean); the common centre mu is the random-effects mean, weights
    1/(se^2 + tau2), so that precise keywords do not dominate it when the slopes are heterogeneous."""
    ok = np.isfinite(slope) & np.isfinite(se) & (se > 0)
    w = 1 / se[ok] ** 2
    fixed = float(np.sum(w * slope[ok]) / np.sum(w))
    q = float(np.sum(w * (slope[ok] - fixed) ** 2))
    df = int(ok.sum()) - 1
    tau2 = max(0.0, (q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    wr = 1 / (se[ok] ** 2 + tau2)
    mu = float(np.sum(wr * slope[ok]) / np.sum(wr))
    b = np.full(len(slope), np.nan)
    post_mean, post_sd = np.full(len(slope), np.nan), np.full(len(slope), np.nan)
    b[ok] = tau2 / (tau2 + se[ok] ** 2)
    post_mean[ok] = mu + b[ok] * (slope[ok] - mu)
    post_sd[ok] = np.sqrt(b[ok] * se[ok] ** 2) if tau2 > 0 else 0.0
    return {"mu": mu, "mu_se": float(np.sqrt(1 / np.sum(wr))), "fixed_effect_mean": fixed,
            "tau2": tau2, "q": q, "df": df, "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
            "shrinkage": b, "mean": post_mean, "sd": post_sd}


def build(args) -> int:
    import pandas as pd
    review = pd.read_parquet(Path(args.review) / "answers.parquet")
    natural = review[review["condition"] == "natural"].copy()
    prompts = []
    with gzip.open(args.prompts, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            prompts.append((r["prompt"]["candidate_id"], r["prompt"]["keyword"], float(r["axis"]["axis_1_percentile_0_1"]), r["prompt"]["question"]))
    prompts = pd.DataFrame(prompts, columns=["prompt_id", "keyword", "x", "question"])
    keywords = sorted(prompts["keyword"].unique())
    code = {k: i for i, k in enumerate(keywords)}
    n_kw = len(keywords)

    # prompt-level cited intent (mean over the prompt's natural cells, all models)
    per_prompt = natural.groupby("prompt_id").agg(x=("x", "first"), keyword=("keyword", "first"), y=("cited_u", "mean")).reset_index()
    px, py = per_prompt["x"].to_numpy(float), per_prompt["y"].to_numpy(float)
    pk = per_prompt["keyword"].map(code).to_numpy(np.int64)
    slope, se, n_prompts = keyword_slopes(px, py, pk, n_kw)
    eb = shrink(slope, se)

    # heterogeneity: Q of the keyword slopes against shuffles of x among each keyword's prompts
    rng = np.random.default_rng(args.seed)
    null_q = []
    for _ in range(args.permutations):
        shuffled = px.copy()
        for k in np.unique(pk):
            idx = np.flatnonzero(pk == k)
            shuffled[idx] = px[idx][rng.permutation(len(idx))]
        s_null, se_null, _ = keyword_slopes(shuffled, py, pk, n_kw)
        null_q.append(shrink(s_null, se_null)["q"])
    p_het = float((1 + np.sum(np.asarray(null_q) >= eb["q"])) / (len(null_q) + 1))

    # per-keyword descriptors
    table = pd.DataFrame({"keyword": keywords})
    table["prompts"] = prompts.groupby("keyword").size().reindex(keywords).to_numpy()
    for model in ("llama4", "qwen38"):
        table[f"answers_{model}"] = natural[natural["model"] == model].groupby("keyword").size().reindex(keywords).fillna(0).astype(int).to_numpy()
    table["slope_cited_intent"] = slope
    table["slope_se"] = se
    table["slope_shrunk"] = eb["mean"]
    table["slope_shrunk_lo95"] = eb["mean"] - 1.96 * eb["sd"]
    table["slope_shrunk_hi95"] = eb["mean"] + 1.96 * eb["sd"]
    table["shrinkage_weight"] = eb["shrinkage"]
    agg = natural.groupby("keyword")
    for column, source in (("cited_offtopic_share", "cited_on_topic"), ("cited_glued_share", "cited_glued"),
                           ("cited_google_top20_url", "cited_google_url"), ("cited_google_top20_domain", "cited_google_domain"),
                           ("cited_log_organic_count", "cited_dfs_organic_count"), ("cited_intent_mean", "cited_u"),
                           ("alignment_gain", "alignment_gain"), ("ranking_len", "ranking_len")):
        values = agg[source].mean().reindex(keywords)
        table[column] = (1 - values).to_numpy() if column == "cited_offtopic_share" else values.to_numpy()
    feats = Path(args.features)
    kw = pd.read_parquet(feats / "keywords.parquet").set_index("keyword").reindex(keywords)
    for column in ("kw_main_intent", "kw_difficulty", "kw_search_volume", "kw_cpc", "kw_competition", "kw_difficulty_tercile"):
        table[column] = kw[column].to_numpy()
    rows = pd.read_parquet(feats / "rows.parquet", columns=["engine", "keyword", "doc_id"])
    corpus = pd.read_parquet(args.snippets, columns=["snippet_id", "glued_titles"])
    rows["glued"] = rows["doc_id"].map(dict(zip(corpus["snippet_id"], corpus["glued_titles"] > 0))).astype(float)
    table["snapshot_glued_share"] = rows.groupby("keyword")["glued"].mean().reindex(keywords).to_numpy()
    if args.explore and (Path(args.explore) / "answers.parquet").exists():
        ex = pd.read_parquet(Path(args.explore) / "answers.parquet")
        ex = ex[ex["condition"] == "natural"]
        table["retrieved_offtopic_share"] = ex.groupby("keyword_text")["R_offtopic"].mean().reindex(keywords).to_numpy()
    table["gemma_mean_support"] = np.nan  # development judgments: filled from the HoreKa gemma-extract when supplied
    if args.gemma and Path(args.gemma).exists():
        g = pd.read_parquet(args.gemma)
        table["gemma_mean_support"] = g.groupby("keyword")["mean_grade"].mean().reindex(keywords).to_numpy()
    examples = []
    for k in keywords:
        p = prompts[prompts["keyword"] == k].sort_values("x")
        picks = [p.iloc[0], p.iloc[len(p) // 2], p.iloc[-1]] if len(p) else []
        examples.append({f"example_{name}": f"[x={r['x']:.2f}] {r['question']}" for name, r in zip(("low", "mid", "high"), picks)})
    table = pd.concat([table, pd.DataFrame(examples)], axis=1)

    # pooled within-keyword slopes by DataForSEO intent class and difficulty tercile (keyword bootstrap)
    draws = stages.keyword_draws(n_kw, args.bootstrap, args.seed)
    nk = natural["keyword"].map(code).to_numpy(np.int64)
    nx, ny = natural["x"].to_numpy(float), natural["cited_u"].to_numpy(float)
    def pooled(mask):
        ok = mask & np.isfinite(ny)
        if ok.sum() < 50:
            return None
        point = geo.within_keyword_slope(nx[ok], ny[ok], nk[ok])
        boot = [geo.within_keyword_slope(nx[ok], ny[ok], nk[ok], d[nk[ok]]) for d in draws]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return {"keywords": int(len(np.unique(nk[ok]))), "answers": int(ok.sum()), "slope": float(point), "ci95": [float(lo), float(hi)]}
    intent_of = dict(zip(keywords, table["kw_main_intent"]))
    tercile_of = dict(zip(keywords, table["kw_difficulty_tercile"]))
    kw_names = natural["keyword"].to_numpy()
    by_class = {c: pooled(np.asarray([intent_of.get(k) == c for k in kw_names])) for c in ("commercial", "informational", "transactional", "navigational")}
    by_tercile = {str(t): pooled(np.asarray([tercile_of.get(k) == t for k in kw_names])) for t in (0, 1, 2)}

    ranked = table.dropna(subset=["slope_shrunk"]).sort_values("slope_shrunk")
    summary = {"keywords": n_kw, "keywords_with_slope": int(np.isfinite(slope).sum()),
               "pooled_mean_slope": eb["mu"], "pooled_mean_slope_ci95": [eb["mu"] - 1.96 * eb["mu_se"], eb["mu"] + 1.96 * eb["mu_se"]],
               "fixed_effect_mean_slope": eb["fixed_effect_mean"],
               "raw_slope_mean_by_se_quintile": [float(v) for v in pd.Series(slope[np.isfinite(slope) & (se > 0)]).groupby(
                   pd.qcut(se[np.isfinite(slope) & (se > 0)], 5, labels=False)).mean()],
               "between_keyword_variance_tau2": eb["tau2"], "i2": eb["i2"],
               "heterogeneity_q": eb["q"], "heterogeneity_df": eb["df"], "heterogeneity_permutation_p": p_het,
               "null_q_mean": float(np.mean(null_q)) if null_q else None, "null_q_max": float(np.max(null_q)) if null_q else None,
               "permutations": args.permutations, "share_keywords_shrunk_interval_above_0": float(np.mean(table["slope_shrunk_lo95"] > 0)),
               "share_keywords_shrunk_interval_below_0": float(np.mean(table["slope_shrunk_hi95"] < 0)),
               "by_intent_class": by_class, "by_difficulty_tercile": by_tercile,
               "top20": ranked.tail(20)[["keyword", "slope_shrunk", "slope_shrunk_lo95", "slope_shrunk_hi95", "kw_main_intent"]].iloc[::-1].to_dict("records"),
               "bottom20": ranked.head(20)[["keyword", "slope_shrunk", "slope_shrunk_lo95", "slope_shrunk_hi95", "kw_main_intent"]].to_dict("records"),
               "correlations_with_shrunk_slope": {c: float(table[["slope_shrunk", c]].corr(method="spearman").iloc[0, 1])
                                                  for c in ("kw_difficulty", "kw_search_volume", "kw_cpc", "cited_offtopic_share",
                                                            "cited_glued_share", "snapshot_glued_share", "cited_log_organic_count",
                                                            "cited_google_top20_url") if table[c].notna().sum() > 30}}
    target = Path(args.output)
    for candidate in (target, target.with_name(target.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")
    partial = readiness.new_directory(target)
    table.to_csv(partial / "keywords.csv", index=False)
    readiness.write_json(partial / "keywords-summary.json", json.loads(json.dumps(summary, default=float).replace("NaN", "null")))
    partial.rename(target.resolve())
    print(json.dumps({k: summary[k] for k in ("keywords_with_slope", "pooled_mean_slope", "between_keyword_variance_tau2",
                                              "heterogeneity_permutation_p")}), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "Hamburg"
    hf = home / "GEODML_Unified/ARR_ACL_CycleOct2026/hf-min"
    parser.add_argument("--review", type=Path, required=True, help="funnel_review.py output folder")
    parser.add_argument("--explore", type=Path, help="funnel_explore.py output folder (exploration keywords)")
    parser.add_argument("--features", type=Path, default=home / "geodml-inputs/funnel-features-v1")
    parser.add_argument("--snippets", type=Path, default=hf / "derived/snippet-embeddings/snippet-embeddings-corpus-v1/snippets.parquet")
    parser.add_argument("--prompts", type=Path, default=hf / "snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz")
    parser.add_argument("--gemma", type=Path, help="per-answer Gemma table (development) with keyword, mean_grade")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
