#!/usr/bin/env python3
"""Merge DataForSEO Whois Overview signals into every main parquet, nuking
the (low-coverage) Moz columns and the (heuristic) conf_brand_recog.

Reads:  ~/geodml_data/data/dataforseo/domain_authority_dfs.parquet
        (produced by fetch_dfs_domain_authority.py — uses the
         /domain_analytics/whois/overview/live endpoint, ~99 % coverage)
Writes back the per-variant + regression parquets at:
        ~/geodml_data/data/main/full_experiment_data_{biased,neutral,biased_rag,neutral_rag}.parquet
        ~/geodml_data/data/main/regression_dataset.parquet

For every target parquet:
  1. Back up the original to *.bak-pre-dfs.parquet (one-shot, idempotent)
  2. Replace the 4 canonical confounder slots with DFS-derived values:
        conf_domain_authority   ← dfs_authority_log  (= log10(organic_count + 1))
        conf_backlinks          ← dfs_organic_count  (Google organic visibility)
        conf_referring_domains  ← dfs_organic_pos_1  (count of #1 organic positions)
        conf_brand_recog        ← dfs_brand_proxy    (binary, brand-scale visibility)
  3. Add three new columns (no prior analogues):
        conf_dfs_paid_count
        conf_dfs_etv             (estimated traffic value, USD)
        conf_dfs_domain_age_years
  4. Save back.

The four canonical slots stay so existing scripts and figures keep working
without changes, but the values inside them are now empirically grounded and
have ~99 % coverage rather than 22 %.

Run:
  python scripts/merge_dfs_domain_authority.py            # apply
  python scripts/merge_dfs_domain_authority.py --dry-run  # print plan only
  python scripts/merge_dfs_domain_authority.py --restore  # revert to *.bak-pre-dfs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "geodml_data"
MAIN = ROOT / "data" / "main"
DFS_PARQUET = ROOT / "data" / "dataforseo" / "domain_authority_dfs.parquet"

TARGETS = [
    MAIN / "regression_dataset.parquet",
    MAIN / "full_experiment_data_biased.parquet",
    MAIN / "full_experiment_data_neutral.parquet",
    MAIN / "full_experiment_data_biased_rag.parquet",
    MAIN / "full_experiment_data_neutral_rag.parquet",
]


def coverage(df, col):
    if col not in df.columns:
        return "(missing)"
    return f"{df[col].notna().mean()*100:5.1f}%"


def apply_merge(dfs_lookup: pd.DataFrame, target: Path, dry_run: bool = False):
    print(f"\n  → {target.name}")
    bak = target.with_name(target.stem + ".bak-pre-dfs.parquet")

    df = pd.read_parquet(target)
    n0 = len(df)

    if not bak.exists() and not dry_run:
        # one-shot backup so --restore can revert
        df.to_parquet(bak, index=False)
        print(f"    backed up original → {bak.name}")

    # Columns we'll overwrite (nuke Moz + heuristic brand)
    to_replace = ["conf_domain_authority", "conf_backlinks",
                  "conf_referring_domains", "conf_brand_recog"]
    new_cols = ["conf_dfs_paid_count", "conf_dfs_etv", "conf_dfs_domain_age_years"]
    pre_cov = {c: coverage(df, c) for c in to_replace + new_cols}

    if dry_run:
        print(f"    rows={n0:,}  current coverage:")
        for c in to_replace + new_cols:
            print(f"      {c:25s} {pre_cov[c]}")
        return

    # Drop existing instances so the merge fills them cleanly
    for c in to_replace + new_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Build replacement columns from the DFS lookup
    repl = dfs_lookup.rename(columns={
        "dfs_authority_log":     "conf_domain_authority",   # log10(organic_count + 1)
        "dfs_organic_count":     "conf_backlinks",          # external visibility proxy
        "dfs_organic_pos_1":     "conf_referring_domains",  # # of #1 organic positions
        "dfs_brand_proxy":       "conf_brand_recog",        # binary, brand-scale visibility
        "dfs_paid_count":        "conf_dfs_paid_count",
        "dfs_organic_etv":       "conf_dfs_etv",
        "dfs_domain_age_years":  "conf_dfs_domain_age_years",
    })

    merge_cols = ["domain", "conf_domain_authority", "conf_backlinks",
                  "conf_referring_domains", "conf_brand_recog",
                  "conf_dfs_paid_count", "conf_dfs_etv",
                  "conf_dfs_domain_age_years"]
    repl = repl[merge_cols]

    df = df.merge(repl, on="domain", how="left")
    df.to_parquet(target, index=False)
    print(f"    rows={n0:,}  new coverage:")
    for c in to_replace + new_cols:
        print(f"      {c:28s} was {pre_cov.get(c, '(new)')}  →  now {coverage(df, c)}")


def restore(target: Path):
    bak = target.with_name(target.stem + ".bak-pre-dfs.parquet")
    if not bak.exists():
        print(f"  → {target.name}: no backup at {bak.name}, skipping")
        return
    df = pd.read_parquet(bak)
    df.to_parquet(target, index=False)
    print(f"  → {target.name}: restored from {bak.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan, don't modify any files")
    ap.add_argument("--restore", action="store_true",
                    help="Revert each target to its *.bak-pre-dfs.parquet")
    args = ap.parse_args()

    if args.restore:
        print("Restoring originals from *.bak-pre-dfs.parquet …")
        for t in TARGETS:
            restore(t)
        return 0

    if not DFS_PARQUET.exists():
        print(f"ERROR: {DFS_PARQUET} not found.", file=sys.stderr)
        print("Run `python scripts/fetch_dfs_domain_authority.py` first.",
              file=sys.stderr)
        return 1

    dfs = pd.read_parquet(DFS_PARQUET)
    print(f"Loaded DFS authority parquet: {len(dfs):,} domains  "
          f"(organic-count coverage {dfs['dfs_organic_count'].notna().mean()*100:.1f}%)")
    print(f"  median organic_count = {int(dfs['dfs_organic_count'].median())}, "
          f"brand-proxy = 1 for {dfs['dfs_brand_proxy'].sum():,} domains")

    if args.dry_run:
        print("\n[DRY RUN] would update these files:")
    else:
        print("\nApplying merge (with one-shot backups → *.bak-pre-dfs.parquet) …")

    for t in TARGETS:
        if t.exists():
            apply_merge(dfs, t, dry_run=args.dry_run)
        else:
            print(f"\n  → {t.name}: missing on disk, skipping")

    if not args.dry_run:
        print("\nDone. Re-run any DML script or fig 13 to pick up new coverage.")


if __name__ == "__main__":
    sys.exit(main() or 0)
