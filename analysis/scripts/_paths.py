"""Single source of truth for input/output directories.

Supports two layouts so the same script works for both audiences:

1. **Dataset layout** (reviewer running from the downloaded HF dataset).
   Activated when the env var GEODML_DATA_ROOT is set. Looks for
   `<root>/data/{main,serp,dml,probing,saliency,ablation,features,...}`
   and writes figures to `<root>/figures/`.

2. **Dev layout** (developer running from the source repo, env var unset).
   Reads bulk parquet from `~/geodml_data/data/`, reads analysis CSVs
   from `<repo>/interpretability/output/`, writes figures to
   `<repo>/docs/2026-05-24/figures_canonical/tmp/`.

Every other script imports the constants below — never hardcode a path.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_HF = os.environ.get("GEODML_DATA_ROOT")

if _HF:
    ROOT = Path(_HF).expanduser().resolve()
    DATA = ROOT / "data"
    MAIN = DATA / "main"
    SERP = DATA / "serp"
    DML = DATA / "dml"
    PROBING = DATA / "probing"
    SALIENCY = DATA / "saliency"
    ABLATION = DATA / "ablation"
    FEATURES = DATA / "features"
    FIGURES = ROOT / "figures"
    DOCS = ROOT / "docs"
else:
    ROOT = REPO_ROOT
    DATA = Path.home() / "geodml_data" / "data"
    MAIN = DATA / "main"
    SERP = REPO_ROOT / "geodml_data" / "data" / "serp"
    DML = DATA / "dml_results"
    PROBING = REPO_ROOT / "interpretability" / "output"
    SALIENCY = REPO_ROOT / "interpretability" / "output"
    ABLATION = REPO_ROOT / "interpretability" / "output"
    FEATURES = DATA / "features"
    FIGURES = REPO_ROOT / "docs" / "2026-05-24" / "figures_canonical" / "tmp"
    DOCS = REPO_ROOT / "docs"

FIGURES.mkdir(parents=True, exist_ok=True)
