"""Annotate all geodml_dataset.csv + merged CSVs with has_llms_txt column
from the existing consolidated_results/domains_llms_txt.csv.

Domains not present in the audit CSV default to has_llms_txt=0.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent / "consolidated_results"
AUDIT = ROOT / "domains_llms_txt.csv"
RUNS = sorted((ROOT / "runs").glob("*/geodml_dataset.csv"))
MERGED = [
    ROOT / "merged" / "merged_all_runs.csv",
    ROOT / "merged" / "merged_all_8runs.csv",
]

audit = pd.read_csv(AUDIT, dtype=str)
audit["domain"] = audit["domain"].str.strip().str.lower()
audit["has_llms_txt"] = audit["has_llms_txt"].fillna("0").astype(int)
mapping = dict(zip(audit["domain"], audit["has_llms_txt"]))
print(f"audit: {len(mapping)} domains, {sum(mapping.values())} hits")


def annotate(path: Path) -> None:
    if not path.exists():
        print(f"  skip missing: {path}")
        return
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "domain" not in df.columns:
        print(f"  skip (no domain col): {path}")
        return
    keys = df["domain"].fillna("").astype(str).str.strip().str.lower()
    df["has_llms_txt"] = keys.map(mapping).fillna(0).astype(int)
    unchecked = keys[~keys.isin(mapping) & keys.ne("")].nunique()
    df.to_csv(path, index=False)
    print(
        f"  {path.relative_to(ROOT.parent)}  rows={len(df)}  "
        f"hits={int(df['has_llms_txt'].sum())}  unchecked-domains={unchecked}"
    )


for f in RUNS:
    annotate(f)
for f in MERGED:
    annotate(f)
