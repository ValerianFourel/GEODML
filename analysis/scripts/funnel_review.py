#!/usr/bin/env python3
"""Exploratory review of the published generation rows (Mac, CPU; addendum A1 of funnel_study.md).

For every answer: behaviour (ranking length, searches, shown evidence, answer length and style), the
cited sources (top-weighted over the ranked URLs, weights 1/log2(rank+1)): intent percentile, on-topic
share, glued/ad/user-content share, SEO and page features (DataForSEO domain data, Open PageRank,
llms.txt, Google top-20 for the prompt's keyword, JSON-LD, question headings, freshness, word count),
alignment with the prompt against the keyword's own rows, overlap with the frozen search on the
prompt text (R0) and that search's own intent; cross-model top-1 agreement; within-keyword ranking
change. Each metric: within-keyword slope on the prompt position x with a keyword bootstrap, per model x
method (engines pooled) and per model x engine, plus 10-bin curves. Associations only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import funnel_features as ff  # noqa: E402
from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

SHORT = {"meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4", "Qwen/Qwen3.8-27B": "qwen38"}
STEP = re.compile(r"(?:^|\s)(?:1[\.\)]|step\s*1\b|first,)", re.I)
IMPERATIVE = re.compile(r"\b(click|sign up|log in|go to|select|enter|download|install|book|order|buy|contact|call|visit|apply|register)\b", re.I)
CURRENCY = re.compile(r"[$€£]\s?\d")
URL_TEXT = re.compile(r"https?://|www\.")

URL_FEATURES = ("url_user_content", "url_wikipedia", "url_ad_redirect", "body_structured_data", "body_question_headings",
                "body_freshness", "body_word_count", "html_usable")
DOMAIN_FEATURES = ("dfs_organic_count", "dfs_organic_pos_1", "opr", "has_llms_txt", "brand_list", "earned_list")


def load_tables(features: Path, snippets: Path, replay: Path | None, prompts: dict):
    import pandas as pd
    rows = pd.read_parquet(features / "rows.parquet")
    urls = pd.read_parquet(features / "urls.parquet")
    domains = pd.read_parquet(features / "domains.parquet")
    google = pd.read_parquet(features / "google.parquet")
    corpus = pd.read_parquet(snippets, columns=["snippet_id", "prompt_scale_percentile_0_1", "glued_titles"])
    u_of_doc = dict(zip(corpus["snippet_id"], corpus["prompt_scale_percentile_0_1"]))
    glued_of_doc = dict(zip(corpus["snippet_id"], corpus["glued_titles"] > 0))
    rows["u"] = rows["doc_id"].map(u_of_doc)
    rows["glued"] = rows["doc_id"].map(glued_of_doc).astype(float)
    # URL-level values (a URL seen with several texts takes the mean over its rows)
    url_u = rows.groupby("url")["u"].mean()
    url_glued = rows.groupby("url")["glued"].max()
    u = urls.set_index("url")
    dom = domains.set_index("domain")
    table = pd.DataFrame({"u": url_u, "glued": url_glued})
    table["domain"] = u["domain"].reindex(table.index)
    table["url_normalized"] = u["url_normalized"].reindex(table.index)
    for f in URL_FEATURES:
        table[f] = pd.to_numeric(u[f].reindex(table.index), errors="coerce")
    table["body_word_count"] = np.log1p(table["body_word_count"])
    for f in DOMAIN_FEATURES:
        values = pd.to_numeric(dom[f].reindex(table["domain"]).to_numpy(), errors="coerce")
        table[f] = np.log1p(values) if f.startswith("dfs_organic") else values
    keyword_rows = defaultdict(set)
    for e, k, url in zip(rows["engine"], rows["keyword"], rows["url"]):
        keyword_rows[(e, k)].add(url)
    keyword_u = defaultdict(list)
    for e, k, val in zip(rows["engine"], rows["keyword"], rows["u"]):
        keyword_u[(e, k)].append(val)
    google_url = set(zip(google["keyword"], google["url_normalized"]))
    google_domain = set(zip(google["keyword"], google["domain"]))
    r0 = {}
    if replay:
        rp = readiness.read_jsonl(replay / "prompts.jsonl.gz")
        chosen = np.load(replay / "replay.npz")["rows"]
        row_url = rows["url"].to_numpy()
        row_u = rows["u"].to_numpy(float)
        for p, c in zip(rp, chosen):
            r0[(p["prompt_id"], p["engine"])] = (set(row_url[c]), float(np.nanmean(row_u[c])))
    return {"url": table.to_dict("index"), "keyword_rows": keyword_rows, "keyword_u": keyword_u, "google_url": google_url,
            "google_domain": google_domain, "r0": r0}


def answer_metrics(model, row, prompts, t) -> dict:
    keyword, x, _ = prompts[row["prompt_id"]]
    engine = row["engine"]
    ranking = row.get("ranking") or []
    answer = row.get("answer") or ""
    out = {"model": model, "engine": engine, "method": row["method"], "condition": row["condition"], "prompt_id": row["prompt_id"],
           "keyword": keyword, "x": x, "ranking_len": len(ranking), "search_count": row.get("search_count"),
           "shown": row.get("final_snippet_count"), "answer_chars": len(answer),
           "answer_steps": int(bool(STEP.search(answer))), "answer_imperatives": len(IMPERATIVE.findall(answer)),
           "answer_currency": int(bool(CURRENCY.search(answer))), "answer_urls": int(bool(URL_TEXT.search(answer))),
           "top1": ranking[0] if ranking else None, "top3": tuple(ranking[:3])}
    table = t["url"]
    own = t["keyword_rows"].get((engine, keyword), set())
    weights, values = [], defaultdict(list)
    for r, url in enumerate(ranking):
        rec = table.get(url)
        if rec is None:
            continue
        w = 1.0 / np.log2(r + 2.0)
        weights.append(w)
        values["cited_u"].append(rec["u"])
        values["cited_on_topic"].append(float(url in own))
        values["cited_gap"].append(abs(rec["u"] - x))
        values["cited_glued"].append(rec["glued"])
        for f in URL_FEATURES + DOMAIN_FEATURES:
            values[f"cited_{f}"].append(rec[f])
        values["cited_google_url"].append(float((keyword, rec["url_normalized"]) in t["google_url"]))
        values["cited_google_domain"].append(float((keyword, rec["domain"]) in t["google_domain"]))
    w = np.asarray(weights)
    for name, vals in values.items():
        v = np.asarray(vals, float)
        ok = np.isfinite(v)
        out[name] = float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else np.nan
    if ranking and ranking[0] in table:
        out["top1_gap"] = abs(table[ranking[0]]["u"] - x)
    own_u = np.asarray(t["keyword_u"].get((engine, keyword), []), float)
    if len(own_u):
        out["null_gap"] = float(np.nanmean(np.abs(own_u - x)))
        out["alignment_gain"] = out["null_gap"] - out.get("cited_gap", np.nan)
    r0 = t["r0"].get((row["prompt_id"], engine))
    if r0:
        out["r0_u"] = r0[1]
        out["r0_overlap"] = float(np.mean([u in r0[0] for u in ranking])) if ranking else np.nan
    return out


METRICS = ["ranking_len", "search_count", "shown", "answer_chars", "answer_steps", "answer_imperatives", "answer_currency",
           "answer_urls", "cited_u", "cited_on_topic", "cited_glued", "cited_gap", "top1_gap", "null_gap", "alignment_gain",
           "r0_u", "r0_overlap", "cited_google_url", "cited_google_domain"] + [f"cited_{f}" for f in URL_FEATURES + DOMAIN_FEATURES]


def slopes(frame, metric, draws, keyword_code):
    y = frame[metric].to_numpy(float)
    x = frame["x"].to_numpy(float)
    k = frame["keyword"].map(keyword_code).to_numpy(np.int64)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 50:
        return None
    x, y, k = x[ok], y[ok], k[ok]
    point = geo.within_keyword_slope(x, y, k)
    boot = [geo.within_keyword_slope(x, y, k, d[k]) for d in draws]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"n": int(ok.sum()), "mean": float(y.mean()), "slope": float(point), "ci95": [float(lo), float(hi)],
            "low_x_mean": float(y[x < 0.2].mean()) if (x < 0.2).any() else None,
            "high_x_mean": float(y[x >= 0.8].mean()) if (x >= 0.8).any() else None}


def review(args) -> int:
    import pandas as pd
    hf = Path(args.hf_root)
    prompts = {}
    with gzip.open(args.prompts, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            prompts[r["prompt"]["candidate_id"]] = (r["prompt"]["keyword"], float(r["axis"]["axis_1_percentile_0_1"]), r["prompt"]["question"])
    t = load_tables(Path(args.features), Path(args.snippets), Path(args.replay) if args.replay else None, prompts)
    mapping = json.loads((hf / "generation-objects.json").read_text())
    records, seen = [], set()
    for n, (path, model) in enumerate(mapping.items()):
        if args.limit_files and n >= args.limit_files:
            break
        if model not in SHORT or not (hf / path).exists():
            continue
        for line in open(hf / path, encoding="utf-8"):
            row = json.loads(line)["row"]
            key = (SHORT[model], row["prompt_id"], row["engine"], row["method"], row["condition"])
            if key in seen or row["prompt_id"] not in prompts:
                continue
            seen.add(key)
            records.append(answer_metrics(SHORT[model], row, prompts, t))
    frame = pd.DataFrame(records)
    keywords = sorted(frame["keyword"].unique())
    keyword_code = {k: i for i, k in enumerate(keywords)}
    draws = stages.keyword_draws(len(keywords), args.bootstrap, args.seed)
    natural = frame[frame["condition"] == "natural"]
    out = {"answers": int(len(frame)), "natural_answers": int(len(natural)),
           "answers_by_model": frame["model"].value_counts().to_dict(), "slopes": {}, "curves": {}}
    groups = {}
    for (model, method), g in natural.groupby(["model", "method"]):
        groups[f"{model} · {method.split('-')[0]}"] = g
    for (model, engine), g in natural.groupby(["model", "engine"]):
        groups[f"{model} · {engine}"] = g
    for name, g in groups.items():
        out["slopes"][name] = {m: slopes(g, m, draws, keyword_code) for m in METRICS if m in g}
        x = g["x"].to_numpy(float)
        k = g["keyword"].map(keyword_code).to_numpy(np.int64)
        curves = stages.stage_curves(x, k, {m: g[m].to_numpy(float) for m in METRICS if m in g},
                                     np.ones(len(g), bool), bins=10)
        out["curves"][name] = curves
    # cross-model agreement on identical cells
    pivot = natural.pivot_table(index=["prompt_id", "engine", "method"], columns="model", values="top1", aggfunc="first")
    if {"llama4", "qwen38"} <= set(pivot.columns):
        both = pivot.dropna()
        agree = (both["llama4"] == both["qwen38"]).astype(float)
        xs = np.asarray([prompts[p][1] for p in both.index.get_level_values(0)])
        bins = np.minimum((xs * 10).astype(int), 9)
        out["cross_model_top1_agreement"] = {"cells": int(len(both)), "agreement": float(agree.mean()),
                                             "by_decile": [float(agree[bins == b].mean()) for b in range(10)]}
    # within-keyword ranking change: top-3 Jaccard distance vs axis gap (sampled pairs)
    rng = np.random.default_rng(args.seed)
    change = {}
    for name, g in groups.items():
        pairs_gap, pairs_dist, pairs_order = [], [], []
        for _, h in g.groupby(["keyword", "engine", "method"] if "·" in name else ["keyword"]):
            h = h[h["top3"].map(len) == 3]
            if len(h) < 2:
                continue
            idx = rng.choice(len(h), size=(min(40, len(h) * (len(h) - 1) // 2), 2))
            xs, tops = h["x"].to_numpy(), h["top3"].to_list()
            for a, b in idx:
                if a == b:
                    continue
                sa, sb = set(tops[a]), set(tops[b])
                pairs_gap.append(abs(xs[a] - xs[b]))
                pairs_dist.append(1 - len(sa & sb) / len(sa | sb))
        if len(pairs_gap) > 100:
            from scipy.stats import spearmanr
            change[name] = {"pairs": len(pairs_gap), "spearman_gap_vs_top3_distance": float(spearmanr(pairs_gap, pairs_dist).statistic),
                            "mean_top3_distance": float(np.mean(pairs_dist))}
    out["ranking_change"] = change
    # keyword moderators of the cited-source slope
    kw = pd.read_parquet(Path(args.features) / "keywords.parquet").set_index("keyword")
    mods = {}
    for label, members in (("commercial_or_transactional", kw.index[kw["kw_commercial_or_transactional"] == 1]),
                           ("informational_or_navigational", kw.index[kw["kw_commercial_or_transactional"] == 0])):
        sub = natural[natural["keyword"].isin(set(members))]
        mods[label] = {m: slopes(sub, m, draws, keyword_code) for m in ("cited_u", "alignment_gain", "ranking_len", "cited_on_topic")}
    for tercile in (0, 1, 2):
        members = set(kw.index[kw["kw_difficulty_tercile"] == tercile])
        sub = natural[natural["keyword"].isin(members)]
        mods[f"difficulty_tercile_{tercile}"] = {m: slopes(sub, m, draws, keyword_code) for m in ("cited_u", "alignment_gain")}
    out["moderators"] = mods
    target = Path(args.output)
    for candidate in (target, target.with_name(target.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "review.json", _finite(out))
    frame.drop(columns=["top3"]).to_parquet(partial / "answers.parquet", index=False)
    partial.rename(target.resolve())
    print(json.dumps({"answers": len(frame), "output": str(target)}), flush=True)
    return 0


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "Hamburg"
    hf = home / "GEODML_Unified/ARR_ACL_CycleOct2026/hf-min"
    parser.add_argument("--hf-root", type=Path, default=hf)
    parser.add_argument("--prompts", type=Path, default=hf / "snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz")
    parser.add_argument("--features", type=Path, default=home / "geodml-inputs/funnel-features-v1")
    parser.add_argument("--snippets", type=Path, default=hf / "derived/snippet-embeddings/snippet-embeddings-corpus-v1/snippets.parquet")
    parser.add_argument("--replay", type=Path, default=home / "geodml-inputs/funnel-replay-v1")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--limit-files", type=int, help="development only: read only the first N generation files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.replay and not (Path(args.replay) / "manifest.json").exists():
        args.replay = None
    return review(args)


if __name__ == "__main__":
    raise SystemExit(main())
