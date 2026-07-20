"""
Ready-to-run examples for every config in the dataset card.

Run any single example:
    python scripts/load_example.py main
    python scripts/load_example.py dml_results
    python scripts/load_example.py serp
    python scripts/load_example.py all

If the dataset is already on the Hub, pass --repo:
    python scripts/load_example.py main --repo valerianfourel/geodml-papersize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA = BUNDLE_ROOT / "data"


# ---- local (no HuggingFace Hub) ------------------------------------------

def local_main() -> None:
    import pandas as pd
    df = pd.read_parquet(LOCAL_DATA / "main" / "full_experiment_data.parquet")
    print(f"main: {len(df):,} rows × {len(df.columns)} cols")
    print(df[["run_id", "keyword", "domain", "pre_rank", "post_rank", "rank_delta"]].head())


def local_dml_results() -> None:
    import pandas as pd
    fits = pd.read_parquet(LOCAL_DATA / "dml_results" / "dml_results_long.parquet")
    pooled = fits[(fits["subset"] == "POOLED") & (fits["outcome"] == "rank_delta")]
    sig = pooled[pooled["p_val"] < 0.01].sort_values("coef")
    print(f"dml_results: {len(fits):,} fits total, {len(sig)} significant POOLED on rank_delta")
    print(sig[["treatment", "coef", "se", "p_val", "stars"]].to_string(index=False))


def local_serp() -> None:
    import pandas as pd
    df = pd.read_parquet(LOCAL_DATA / "serp" / "phase0_top50_searxng.parquet")
    print(f"serp.searxng_top50: {len(df):,} rows")
    print(df.head(3))


def local_dataforseo() -> None:
    import pandas as pd
    df = pd.read_parquet(LOCAL_DATA / "dataforseo" / "keyword_overview.parquet")
    print(f"dataforseo.keyword_overview: {len(df):,} rows × {len(df.columns)} cols")
    print(df.head(3))


def local_domains() -> None:
    import pandas as pd
    df = pd.read_parquet(LOCAL_DATA / "domains_llms_txt.parquet")
    print(f"domains_llms_txt: {len(df):,} rows; has_llms_txt rate:"
          f" {df['has_llms_txt'].mean():.3%}" if "has_llms_txt" in df.columns else "")
    print(df.head(3))


# ---- HuggingFace Hub loaders ----------------------------------------------

def hub_all(repo: str) -> None:
    from datasets import load_dataset
    for cfg in ["main", "main_pre_dfs", "dml_results", "dml_results_pre_dfs",
                "serp", "dataforseo", "domains"]:
        ds = load_dataset(repo, cfg)
        print(f"\n=== {cfg} ===")
        for split, d in ds.items():
            print(f"  {split}: {len(d):,} rows, {len(d.column_names)} cols")


CONFIGS = {
    "main": local_main,
    "dml_results": local_dml_results,
    "serp": local_serp,
    "dataforseo": local_dataforseo,
    "domains": local_domains,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(CONFIGS) + ["all"])
    ap.add_argument("--repo", help="HuggingFace repo id (e.g. user/geodml-papersize). "
                    "If given, loads via datasets.load_dataset; otherwise reads local parquet.")
    args = ap.parse_args()

    if args.repo:
        if args.config == "all":
            hub_all(args.repo)
            return 0
        from datasets import load_dataset
        ds = load_dataset(args.repo, args.config)
        print(f"loaded {args.repo} / {args.config}")
        for split, d in ds.items():
            print(f"  {split}: {len(d):,} rows, {len(d.column_names)} cols")
        return 0

    if args.config == "all":
        for name, fn in CONFIGS.items():
            print(f"\n=== {name} ===")
            fn()
        return 0

    CONFIGS[args.config]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
