"""
Walk the bundle's data/ tree and produce a .parquet alongside every .csv.

Parquet lets HuggingFace's load_dataset() stream rows without re-parsing CSV,
and unlocks the dataset Preview UI. CSVs are kept for human/grep-ability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BUNDLE_ROOT / "data"


def convert_one(csv_path: Path) -> tuple[str, int | None]:
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return ("skipped", None)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return (f"read-error: {e}", None)
    try:
        df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)
    except Exception as e:
        return (f"write-error: {e}", None)
    return ("ok", len(df))


def main() -> int:
    csvs = sorted(DATA_DIR.rglob("*.csv"))
    if not csvs:
        print("no CSVs found under", DATA_DIR)
        return 1
    errors = 0
    for csv_path in csvs:
        status, nrows = convert_one(csv_path)
        rel = csv_path.relative_to(BUNDLE_ROOT)
        if status == "ok":
            print(f"  {rel}  ({nrows:,} rows)")
        elif status == "skipped":
            print(f"  {rel}  (skipped, up-to-date)")
        else:
            print(f"  {rel}  FAILED: {status}", file=sys.stderr)
            errors += 1
    print(f"\nDone. {len(csvs)} CSVs, {errors} errors.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
