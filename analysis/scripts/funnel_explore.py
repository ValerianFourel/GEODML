#!/usr/bin/env python3
"""Stage-level exploration on the exploration-keyword trace sample (Mac, CPU; addendum A1).

Per answer and stage S in {R retrieved, C scored, P presented, K ranked}: mean page intent
(prompt-scale percentile u; K top-weighted), share of rows from another keyword (off-topic), of
glued-title rows and of ad redirects; overlap of the AI's retrieval with the frozen search on the
prompt text (R0); lexical profile of the AI's own search queries (count, length, prompt-word reuse,
keyword inclusion, action and information vocabulary). Within-keyword slopes on x with a keyword
bootstrap, per model x method (engines pooled) and per model x engine; 10-bin curves. Exploratory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

TOKEN = re.compile(r"[\w-]+")
ACTION = {"buy", "price", "prices", "pricing", "cost", "costs", "cheap", "cheapest", "deal", "deals", "discount", "coupon", "order",
          "book", "booking", "near", "sign", "signup", "login", "log", "download", "install", "setup", "set", "subscribe",
          "trial", "free", "quote", "hire", "contact", "apply", "register", "steps", "step", "how", "start", "today", "now",
          "same-day", "instant", "plan", "plans", "purchase", "checkout", "account", "activate", "configure", "migrate", "transfer"}
INFORMATION = {"what", "why", "guide", "explained", "explain", "definition", "meaning", "history", "overview", "vs", "versus",
               "compare", "comparison", "difference", "differences", "review", "reviews", "pros", "cons", "benefits",
               "advantages", "types", "examples", "research", "study", "analysis", "impact", "trends", "statistics", "best"}


def tokens(text: str) -> set:
    return {t.casefold() for t in TOKEN.findall(text or "")}


def query_metrics(queries, prompt, keyword) -> dict:
    if not queries:
        return {}
    p, k = tokens(prompt), tokens(keyword)
    qt = [tokens(q) for q in queries]
    words = [w for q in qt for w in q]
    return {"q_count": len(queries), "q_words": float(np.mean([len(TOKEN.findall(q)) for q in queries])),
            "q_prompt_reuse": float(np.mean([len(q & p) / max(len(q), 1) for q in qt])),
            "q_keyword_inclusion": float(np.mean([k <= q for q in qt])) if k else np.nan,
            "q_action_share": float(np.mean([w in ACTION for w in words])) if words else np.nan,
            "q_information_share": float(np.mean([w in INFORMATION for w in words])) if words else np.nan,
            "q_distinct_share": len(set(queries)) / len(queries)}


def explore(args) -> int:
    import pandas as pd
    ext = Path(args.extract)
    answers = readiness.read_jsonl(ext / "answers.jsonl.gz")
    queries = {r["answer"]: r["queries"] for r in readiness.read_jsonl(ext / "queries.jsonl.gz")}
    items = np.load(ext / "items.npz")
    rows = pd.read_parquet(Path(args.features) / "rows.parquet", columns=["row_id", "engine", "keyword", "url", "doc_id"])
    urls = pd.read_parquet(Path(args.features) / "urls.parquet", columns=["url", "url_ad_redirect"])
    corpus = pd.read_parquet(args.snippets, columns=["snippet_id", "prompt_scale_percentile_0_1", "glued_titles"])
    u_of = dict(zip(corpus["snippet_id"], corpus["prompt_scale_percentile_0_1"]))
    glued_of = dict(zip(corpus["snippet_id"], corpus["glued_titles"] > 0))
    row_u = rows["doc_id"].map(u_of).to_numpy(float)
    row_glued = rows["doc_id"].map(glued_of).astype(float).to_numpy()
    row_ad = rows["url"].map(dict(zip(urls["url"], urls["url_ad_redirect"]))).astype(float).to_numpy()
    row_keyword = np.asarray([k.casefold() for k in rows["keyword"]], object)
    replay = {}
    if args.replay:
        rp = readiness.read_jsonl(Path(args.replay) / "prompts.jsonl.gz")
        chosen = np.load(Path(args.replay) / "replay.npz")["rows"]
        replay = {(p["prompt_id"], p["engine"]): set(map(int, c)) for p, c in zip(rp, chosen)}
    prompt_text = {}
    import gzip
    with gzip.open(args.prompts, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            prompt_text[r["prompt"]["candidate_id"]] = r["prompt"]["question"]
    records = []
    off = items["offsets"]
    for i, a in enumerate(answers):
        lo, hi = off[i], off[i + 1]
        row = items["row"][lo:hi]
        scored, presented, ranked = items["scored"][lo:hi] == 1, items["presented"][lo:hi] >= 0, items["ranked"][lo:hi] >= 0
        keyword = (a["keyword_text"] or "").casefold()
        rec = {k: a[k] for k in ("model", "engine", "method", "condition", "prompt_id", "keyword_text", "x")}
        for name, mask in (("R", np.ones(len(row), bool)), ("C", scored), ("P", presented)):
            r = row[mask]
            if len(r):
                rec[f"{name}_u"] = float(np.nanmean(row_u[r]))
                rec[f"{name}_offtopic"] = float(np.mean(row_keyword[r] != keyword))
                rec[f"{name}_glued"] = float(np.nanmean(row_glued[r]))
                rec[f"{name}_ad"] = float(np.nanmean(row_ad[r]))
        if ranked.any():
            order = np.argsort(items["ranked"][lo:hi][ranked])
            r = row[ranked][order]
            w = 1.0 / np.log2(np.arange(len(r)) + 2.0)
            u = row_u[r]
            ok = np.isfinite(u)
            rec["K_u"] = float(np.sum(w[ok] * u[ok]) / np.sum(w[ok])) if ok.any() else np.nan
            rec["K_offtopic"] = float(np.mean(row_keyword[r] != keyword))
            rec["K_glued"] = float(np.nanmean(row_glued[r]))
        r0 = replay.get((a["prompt_id"], a["engine"]))
        if r0 is not None:
            got = set(map(int, row))
            rec["R0_recovered"] = len(r0 & got) / len(r0) if r0 else np.nan
            rec["R_from_R0"] = len(r0 & got) / len(got) if got else np.nan
            rec["R0_u"] = float(np.nanmean(row_u[list(r0)])) if r0 else np.nan
        rec.update(query_metrics(queries.get(i, []), prompt_text.get(a["prompt_id"], ""), a["keyword_text"] or ""))
        records.append(rec)
    frame = pd.DataFrame(records)
    natural = frame[frame["condition"] == "natural"]
    keywords = sorted(frame["keyword_text"].unique())
    code = {k: i for i, k in enumerate(keywords)}
    draws = stages.keyword_draws(len(keywords), args.bootstrap, args.seed)
    metrics = [c for c in frame.columns if c[:2] in ("R_", "C_", "P_", "K_", "R0", "q_")]
    groups = {}
    for (m, meth), g in natural.groupby(["model", "method"]):
        groups[f"{m} · {meth.split('-')[0]}"] = g
    for (m, e), g in natural.groupby(["model", "engine"]):
        groups[f"{m} · {e}"] = g
    out = {"answers": int(len(frame)), "natural": int(len(natural)), "keywords": len(keywords), "slopes": {}, "curves": {}}
    for name, g in groups.items():
        x = g["x"].to_numpy(float)
        k = g["keyword_text"].map(code).to_numpy(np.int64)
        res = {}
        for m in metrics:
            y = g[m].to_numpy(float)
            ok = np.isfinite(y)
            if ok.sum() < 50:
                continue
            point = geo.within_keyword_slope(x[ok], y[ok], k[ok])
            boot = [geo.within_keyword_slope(x[ok], y[ok], k[ok], d[k[ok]]) for d in draws]
            lo_, hi_ = np.nanpercentile(boot, [2.5, 97.5])
            res[m] = {"n": int(ok.sum()), "mean": float(y[ok].mean()), "slope": float(point), "ci95": [float(lo_), float(hi_)],
                      "low_x_mean": float(np.mean(y[ok][x[ok] < 0.2])) if (x[ok] < 0.2).any() else None,
                      "high_x_mean": float(np.mean(y[ok][x[ok] >= 0.8])) if (x[ok] >= 0.8).any() else None}
        out["slopes"][name] = res
        out["curves"][name] = stages.stage_curves(x, k, {m: g[m].to_numpy(float) for m in metrics}, np.ones(len(g), bool), bins=10)
    target = Path(args.output)
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "explore.json", json.loads(json.dumps(out, default=float).replace("NaN", "null")))
    frame.to_parquet(partial / "answers.parquet", index=False)
    partial.rename(target.resolve())
    print(json.dumps({"answers": len(frame), "output": str(target)}), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "Hamburg"
    hf = home / "GEODML_Unified/ARR_ACL_CycleOct2026/hf-min"
    parser.add_argument("--extract", type=Path, default=home / "geodml-inputs/funnel-extract-exploration-v1")
    parser.add_argument("--features", type=Path, default=home / "geodml-inputs/funnel-features-v1")
    parser.add_argument("--snippets", type=Path, default=hf / "derived/snippet-embeddings/snippet-embeddings-corpus-v1/snippets.parquet")
    parser.add_argument("--replay", type=Path, default=home / "geodml-inputs/funnel-replay-v1")
    parser.add_argument("--prompts", type=Path, default=hf / "snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, required=True)
    return explore(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
