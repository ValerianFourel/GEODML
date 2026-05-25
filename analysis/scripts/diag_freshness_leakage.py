"""Diagnostic: does the T6 freshness probe rely on year-token leakage?

For 250 fresh + 250 stale pages (median split on treat_freshness),
count the rate at which each year string 2015-2026 appears in the body
digest. A ratio >> 1 means fresh pages mention recent years much more,
i.e. the layer-0 probe is doing lexical year-detection rather than
deep semantic reasoning about temporal recency.

Run:
    python scripts/diag_freshness_leakage.py

Prints a small table and (optionally) writes
    docs/2026-05-24/freshness_leakage_diagnostic.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
from interpretability.utils import load_main_table, page_digest, HTMLLoader, data_root


YEARS = list(range(2015, 2027))
SAMPLE_PER_CLASS = 250


def main():
    df = load_main_table()
    df = df[df["treat_freshness"].notna()].copy()
    med = df["treat_freshness"].median()
    df["fresh"] = (df["treat_freshness"] > med).astype(int)

    print(f"main rows with treat_freshness: {len(df):,}")
    print(f"freshness median: {med}")
    print(f"label balance: {df['fresh'].value_counts().to_dict()}")
    print(f"treat_freshness distribution: "
          f"{df['treat_freshness'].value_counts().sort_index().to_dict()}")

    rng = 0
    sample_f = df[df.fresh == 1].sample(min(SAMPLE_PER_CLASS, (df.fresh == 1).sum()), random_state=rng)
    sample_s = df[df.fresh == 0].sample(min(SAMPLE_PER_CLASS, (df.fresh == 0).sum()), random_state=rng)
    sample = pd.concat([sample_f, sample_s])
    print(f"\nsample: {len(sample_f)} fresh + {len(sample_s)} stale = {len(sample)} URLs")

    loaders: dict[str, HTMLLoader] = {}
    counts = {0: {y: 0 for y in YEARS}, 1: {y: 0 for y in YEARS}}
    n_seen = {0: 0, 1: 0}

    for _, row in sample.iterrows():
        rid = row["run_id"]
        if rid not in loaders:
            loaders[rid] = HTMLLoader(rid, root=data_root())
        try:
            html = loaders[rid].get_html(row["url"])
        except Exception:
            continue
        if not html:
            continue
        text = page_digest(html)
        label = int(row["fresh"])
        n_seen[label] += 1
        for y in YEARS:
            if str(y) in text:
                counts[label][y] += 1

    print(f"\nactually scanned: stale={n_seen[0]}  fresh={n_seen[1]}\n")
    print(f"{'year':<6} {'stale=0':>10} {'fresh=1':>10}  {'ratio f/s':>10}")
    print("-" * 42)
    md_rows = []
    for y in YEARS:
        p0 = counts[0][y] / max(1, n_seen[0]) * 100
        p1 = counts[1][y] / max(1, n_seen[1]) * 100
        ratio = p1 / max(0.1, p0)
        marker = " ←" if (ratio >= 2.0 or ratio <= 0.5) else ""
        print(f"{y:<6} {p0:>9.1f}% {p1:>9.1f}%   {ratio:>8.2f}x{marker}")
        md_rows.append((y, p0, p1, ratio))

    # write a short markdown report
    out_md = REPO / "docs" / "2026-05-24" / "freshness_leakage_diagnostic.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write("# T6 freshness probe — lexical-leakage diagnostic\n\n")
        f.write(f"Sample: {len(sample_f)} fresh + {len(sample_s)} stale pages "
                f"(median split on `treat_freshness`, median = {med}).\n\n")
        f.write(f"Pages actually loaded: stale = {n_seen[0]}, fresh = {n_seen[1]}.\n\n")
        f.write("## Year-token presence rate in page body text\n\n")
        f.write("| year | stale rate | fresh rate | ratio fresh/stale |\n")
        f.write("|---|---|---|---|\n")
        for y, p0, p1, ratio in md_rows:
            f.write(f"| {y} | {p0:.1f}% | {p1:.1f}% | {ratio:.2f}× |\n")
        f.write("\n")
        f.write("**Interpretation.** Ratios ≥ 2× indicate that the corresponding "
                "year token appears disproportionately in fresh-class pages. "
                "Such concentration provides linear separation in the layer-0 "
                "token-embedding space, explaining why the linear probe reaches "
                "ROC AUC ≥ 0.85 at layer 0 without any contextualisation.\n")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
