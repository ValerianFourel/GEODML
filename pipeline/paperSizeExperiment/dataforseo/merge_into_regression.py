"""Merge DataForSEO outputs into consolidated_results/regression_dataset.csv.

Adds per-keyword columns (search_volume, cpc, competition, keyword_difficulty,
main_intent) and per-(keyword, domain) SERP columns (google rank, results
count, top url) to the regression dataset. Also fills existing empty
confounder columns where DataForSEO gives a direct answer.

Backlinks-derived columns (domain_authority, backlinks, referring_domains)
are left untouched because the DataForSEO backlinks endpoints are still
blocked on subscription 40204 — the corresponding CSVs are empty. When the
subscription activates and the CSVs fill up, rerun this script.

Usage:
  python -m paperSizeExperiment.dataforseo.merge_into_regression
  python -m paperSizeExperiment.dataforseo.merge_into_regression --in-place
  python -m paperSizeExperiment.dataforseo.merge_into_regression \
      --input paperSizeExperiment/consolidated_results/regression_dataset.csv \
      --output paperSizeExperiment/consolidated_results/regression_dataset_with_dfs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

DEFAULT_INPUT = EXPERIMENT_DIR / "consolidated_results" / "regression_dataset.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "consolidated_results" / "regression_dataset_with_dfs.csv"


def _load_keyword_features() -> pd.DataFrame:
    """Return per-keyword features from DataForSEO (one row per unique keyword)."""
    bulk_kd_path = OUTPUT_DIR / "bulk_keyword_difficulty.csv"
    ko_path = OUTPUT_DIR / "keyword_overview.csv"

    frames = []
    if bulk_kd_path.exists() and bulk_kd_path.stat().st_size > 0:
        kd = pd.read_csv(bulk_kd_path, low_memory=False)
        kd = kd[["keyword", "keyword_difficulty"]].rename(
            columns={"keyword_difficulty": "dfs_keyword_difficulty"}
        )
        frames.append(kd)
    else:
        print("  [warn] bulk_keyword_difficulty.csv empty/missing — dfs_keyword_difficulty will be blank")

    if ko_path.exists() and ko_path.stat().st_size > 0:
        ko = pd.read_csv(ko_path, low_memory=False)
        ko_sub = ko[
            [
                "keyword",
                "ko.search_volume",
                "ko.cpc",
                "ko.competition",
                "ko.competition_level",
                "ko.keyword_difficulty",
                "ko.main_intent",
                "ko.foreign_intent",
            ]
        ].rename(
            columns={
                "ko.search_volume": "dfs_search_volume",
                "ko.cpc": "dfs_cpc",
                "ko.competition": "dfs_competition",
                "ko.competition_level": "dfs_competition_level",
                "ko.keyword_difficulty": "dfs_keyword_difficulty_ko",
                "ko.main_intent": "dfs_main_intent",
                "ko.foreign_intent": "dfs_foreign_intent",
            }
        )
        frames.append(ko_sub)
    else:
        print("  [warn] keyword_overview.csv empty/missing — search_volume/cpc/intent will be blank")

    if not frames:
        return pd.DataFrame(columns=["keyword"])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="keyword", how="outer")

    if "dfs_keyword_difficulty" in merged.columns and "dfs_keyword_difficulty_ko" in merged.columns:
        merged["dfs_keyword_difficulty"] = merged["dfs_keyword_difficulty"].fillna(
            merged["dfs_keyword_difficulty_ko"]
        )
        merged = merged.drop(columns=["dfs_keyword_difficulty_ko"])
    elif "dfs_keyword_difficulty_ko" in merged.columns:
        merged = merged.rename(columns={"dfs_keyword_difficulty_ko": "dfs_keyword_difficulty"})

    # Backfill from Google Ads search_volume for keywords missing from Labs ko.
    ga_path = OUTPUT_DIR / "google_ads_search_volume.csv"
    if ga_path.exists() and ga_path.stat().st_size > 0:
        ga = pd.read_csv(ga_path, low_memory=False)
        ga = ga.rename(
            columns={
                "ga_search_volume": "_ga_search_volume",
                "ga_cpc": "_ga_cpc",
                "ga_competition": "_ga_competition_level",
            }
        )[["keyword", "_ga_search_volume", "_ga_cpc", "_ga_competition_level"]]
        merged = merged.merge(ga, on="keyword", how="outer")
        # Labs numeric + GA ordinal for the competition_level string column
        for col, ga_col in [
            ("dfs_search_volume", "_ga_search_volume"),
            ("dfs_cpc", "_ga_cpc"),
            ("dfs_competition_level", "_ga_competition_level"),
        ]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[ga_col])
            else:
                merged[col] = merged[ga_col]
        merged = merged.drop(
            columns=["_ga_search_volume", "_ga_cpc", "_ga_competition_level"]
        )

    # Backfill main_intent from Labs search_intent for keywords missing from ko.
    si_path = OUTPUT_DIR / "search_intent.csv"
    if si_path.exists() and si_path.stat().st_size > 0:
        si = pd.read_csv(si_path, low_memory=False)[
            ["keyword", "si_main_intent", "si_secondary_intents"]
        ]
        merged = merged.merge(si, on="keyword", how="outer")
        if "dfs_main_intent" in merged.columns:
            merged["dfs_main_intent"] = merged["dfs_main_intent"].fillna(merged["si_main_intent"])
        else:
            merged["dfs_main_intent"] = merged["si_main_intent"]
        if "dfs_foreign_intent" in merged.columns:
            merged["dfs_foreign_intent"] = merged["dfs_foreign_intent"].fillna(
                merged["si_secondary_intents"]
            )
        else:
            merged["dfs_foreign_intent"] = merged["si_secondary_intents"]
        merged = merged.drop(columns=["si_main_intent", "si_secondary_intents"])

    return merged


def _load_serp_features() -> pd.DataFrame:
    """Return per-(keyword, domain) Google SERP features.

    A domain can appear multiple times for a keyword; we keep the best
    (lowest) rank_group and the url that achieved it.
    """
    path = OUTPUT_DIR / "serp_google_organic.csv"
    if not path.exists() or path.stat().st_size == 0:
        print("  [warn] serp_google_organic.csv empty/missing — dfs_google_rank will be blank")
        return pd.DataFrame(columns=["keyword", "domain"])

    serp = pd.read_csv(path, low_memory=False)
    serp = serp.sort_values(["keyword", "domain", "rank_group"])
    best = serp.drop_duplicates(subset=["keyword", "domain"], keep="first")
    best = best[
        [
            "keyword",
            "domain",
            "rank_group",
            "rank_absolute",
            "se_results_count",
            "url",
        ]
    ].rename(
        columns={
            "rank_group": "dfs_google_rank",
            "rank_absolute": "dfs_google_rank_absolute",
            "se_results_count": "dfs_se_results_count",
            "url": "dfs_google_top_url",
        }
    )
    return best


def _load_domain_features() -> pd.DataFrame:
    """Return per-domain backlinks features when available.

    Currently the 4 backlinks CSVs are empty (subscription 40204). This
    function returns an empty frame and logs the status, so downstream
    merge is a no-op until the subscription activates.
    """
    targets = {
        "bulk_ranks.csv": "rank",
        "bulk_backlinks.csv": "backlinks",
        "bulk_referring_domains.csv": "referring_domains",
        "bulk_spam_score.csv": "backlinks_spam_score",
    }
    frames = []
    for fname, _ in targets.items():
        p = OUTPUT_DIR / fname
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p, low_memory=False)
                if len(df) > 0:
                    frames.append(df)
                    print(f"  [info] {fname}: {len(df)} rows available for domain-level merge")
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        print("  [info] no backlinks CSVs populated yet — skipping domain-level merge")
        return pd.DataFrame(columns=["domain"])

    # When populated, these have a 'target' column (lowercased domain).
    # Normalise and inner-join.
    out = None
    for df in frames:
        key = "target" if "target" in df.columns else "domain"
        df = df.rename(columns={key: "domain"})
        sub_cols = ["domain"] + [c for c in df.columns if c != "domain"]
        df = df[sub_cols]
        out = df if out is None else out.merge(df, on="domain", how="outer")
    prefix = {
        "rank": "dfs_domain_rank",
        "backlinks": "dfs_backlinks",
        "referring_domains": "dfs_referring_domains",
        "backlinks_spam_score": "dfs_spam_score",
    }
    rename_map = {}
    for old, new in prefix.items():
        if old in out.columns:
            rename_map[old] = new
    return out.rename(columns=rename_map)


def _coverage(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"{col}: missing"
    n = int(df[col].notna().sum())
    return f"{col}: {n}/{len(df)} ({100*n/len(df):.1f}%)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input instead of writing to --output.",
    )
    args = parser.parse_args()

    print(f"Loading regression dataset: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    n_rows = len(df)
    print(f"  {n_rows:,} rows, {df.shape[1]} columns")
    print(f"  {df['keyword'].nunique()} unique keywords, {df['domain'].nunique()} unique domains")

    # Idempotency: strip any existing DataForSEO columns (including pandas
    # _x/_y merge artifacts) before re-merging.
    stale = [c for c in df.columns if c.startswith("dfs_") or c.endswith(("_x", "_y"))]
    if stale:
        df = df.drop(columns=stale)
        print(f"  dropped {len(stale)} stale dfs_*/_x/_y columns before re-merge")

    print("\nLoading DataForSEO features...")
    kw_feats = _load_keyword_features()
    serp_feats = _load_serp_features()
    domain_feats = _load_domain_features()
    print(
        f"  keyword features: {len(kw_feats):,} rows, cols={list(kw_feats.columns)}"
    )
    print(
        f"  serp features:    {len(serp_feats):,} rows, cols={list(serp_feats.columns)}"
    )
    print(
        f"  domain features:  {len(domain_feats):,} rows, cols={list(domain_feats.columns)}"
    )

    print("\nMerging...")
    dfs_cols_added: list[str] = []

    if len(kw_feats) > 0:
        df = df.merge(kw_feats, on="keyword", how="left")
        dfs_cols_added.extend(c for c in kw_feats.columns if c != "keyword")
    if len(serp_feats) > 0:
        df = df.merge(serp_feats, on=["keyword", "domain"], how="left")
        dfs_cols_added.extend(c for c in serp_feats.columns if c not in ("keyword", "domain"))
    if len(domain_feats) > 0:
        df = df.merge(domain_feats, on="domain", how="left")
        dfs_cols_added.extend(c for c in domain_feats.columns if c != "domain")

    if "dfs_main_intent" in df.columns:
        intents = ["commercial", "informational", "navigational", "transactional"]
        for lab in intents:
            col = f"dfs_intent_{lab}"
            df[col] = (df["dfs_main_intent"] == lab).astype(float)
            dfs_cols_added.append(col)

    print("\nFilling existing NaN confounders from DataForSEO where possible...")
    fill_rules = [
        ("X8_keyword_difficulty", "dfs_keyword_difficulty"),
    ]
    if "dfs_backlinks" in df.columns:
        fill_rules.append(("conf_backlinks", "dfs_backlinks"))
    if "dfs_referring_domains" in df.columns:
        fill_rules.append(("conf_referring_domains", "dfs_referring_domains"))
    if "dfs_domain_rank" in df.columns:
        fill_rules.append(("X1_global_rank", "dfs_domain_rank"))

    for target, source in fill_rules:
        if target in df.columns and source in df.columns:
            before = int(df[target].notna().sum())
            df[target] = df[target].fillna(df[source])
            after = int(df[target].notna().sum())
            print(f"  {target} <- {source}: {before} -> {after} non-null (+{after-before})")

    print("\nCoverage of DataForSEO columns in merged dataset:")
    for c in dfs_cols_added:
        print(f"  {_coverage(df, c)}")

    out_path = args.input if args.in_place else args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows, {df.shape[1]} columns -> {out_path}")
    print(f"Added {len(dfs_cols_added)} DataForSEO columns: {dfs_cols_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
