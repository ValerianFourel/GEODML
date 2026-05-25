"""Build a clean, self-contained reproducibility dataset for HuggingFace.

Assembles every input table, every fitted-model output, and every
figure-generation script needed to reproduce the EMNLP 2026 paper's
plots and statistics, into a single directory:

    ~/geodml-emnlp-2026/

Run:  python scripts/build_hf_dataset.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
DATA = HOME / "geodml_data" / "data"
SERP_LOCAL = REPO / "geodml_data" / "data" / "serp"

OUT = HOME / "geodml-emnlp-2026"


def copy_one(src: Path, dst: Path, *, optional: bool = False) -> bool:
    if not src.exists():
        if optional:
            print(f"  [skip] {src.relative_to(HOME) if src.is_relative_to(HOME) else src}")
            return False
        raise FileNotFoundError(f"Required source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    sz = src.stat().st_size
    print(f"  [copy] {dst.relative_to(OUT)}  ({sz/1024:.1f} KB)")
    return True


def copy_tree_filtered(src_dir: Path, dst_dir: Path, *, patterns: list[str]) -> int:
    """Copy files whose name matches any of the patterns. Recursive."""
    if not src_dir.exists():
        return 0
    n = 0
    for p in src_dir.rglob("*"):
        if p.is_file() and any(p.match(pat) for pat in patterns):
            rel = p.relative_to(src_dir)
            (dst_dir / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst_dir / rel)
            n += 1
    return n


# ---------------------------------------------------------------------------
def build():
    if OUT.exists():
        print(f"WARNING: {OUT} already exists. Existing files will be overwritten.")
    OUT.mkdir(exist_ok=True)

    # ───────── 1. Main experimental data ─────────
    print("\n[1/8]  Main experimental tables")
    main_out = OUT / "data" / "main"
    main_out.mkdir(parents=True, exist_ok=True)
    for fname in (
        "full_experiment_data.parquet",
        "full_experiment_data_biased.parquet",
        "full_experiment_data_neutral.parquet",
        "full_experiment_data_biased_rag.parquet",
        "full_experiment_data_neutral_rag.parquet",
        "regression_dataset.parquet",
    ):
        copy_one(DATA / "main" / fname, main_out / fname, optional=True)

    # ───────── 2. SERP snapshots ─────────
    print("\n[2/8]  SERP snapshots (phase 0)")
    serp_out = OUT / "data" / "serp"
    serp_out.mkdir(parents=True, exist_ok=True)
    for backend in ("searxng", "ddg"):
        for pool in (20, 50):
            fname = f"phase0_top{pool}_{backend}.parquet"
            # check both possible local locations
            src = DATA / "serp" / fname
            if not src.exists():
                src = SERP_LOCAL / fname
            copy_one(src, serp_out / fname, optional=True)

    # ───────── 3. DML model outputs ─────────
    print("\n[3/8]  DML fitted-model outputs")
    dml_out = OUT / "data" / "dml"
    dml_out.mkdir(parents=True, exist_ok=True)
    for fname in (
        "dml_canonical_2026-05-25_llms_as_confounder.parquet",  # HEADLINE
        "dml_canonical_2026-05-25_no_llms.parquet",             # sensitivity
        "dml_canonical_2026-05-24.parquet",                     # legacy (with T7=llms.txt)
        "confounder_loo_r2.parquet",
        "confounder_ols_significance.parquet",
        "nuisance_r2.parquet",
    ):
        copy_one(DATA / "dml_results" / fname, dml_out / fname, optional=True)

    # ───────── 4. Probing results (treatment + admission) ─────────
    print("\n[4/8]  Probing results (treatment + admission)")
    probing_out = OUT / "data" / "probing"
    probing_out.mkdir(parents=True, exist_ok=True)
    interp_dir = REPO / "interpretability" / "output"
    for v in ("biased", "neutral", "biased_rag", "neutral_rag"):
        copy_one(interp_dir / f"probing_results_{v}.csv",
                 probing_out / f"probing_results_{v}.csv", optional=True)

    # ───────── 5. Saliency summaries (per-token scores omitted; too large) ─────────
    print("\n[5/8]  Saliency summaries")
    sal_out = OUT / "data" / "saliency"
    sal_out.mkdir(parents=True, exist_ok=True)
    for fname in ("saliency_summary_full.csv", "saliency_summary_rw.csv"):
        copy_one(interp_dir / fname, sal_out / fname, optional=True)
    # also include the smaller per-cell SUMMARY (not per-token scores)
    n = copy_tree_filtered(interp_dir, sal_out / "per_cell",
                           patterns=["saliency_*-Instruct_*/saliency_summary_*.csv"])
    print(f"  [copy-tree] {n} per-cell summary files under per_cell/")

    # ───────── 6. Ablation results ─────────
    print("\n[6/8]  Ablation results")
    abl_out = OUT / "data" / "ablation"
    abl_out.mkdir(parents=True, exist_ok=True)
    for fname in (
        "ablation_results_full_biased.csv",
        "ablation_results_full_neutral.csv",
        "ablation_results_rw_biased.csv",
        "ablation_results_rw_neutral.csv",
        "ablation_results_full.csv",
        "ablation_results_rw.csv",
    ):
        copy_one(interp_dir / fname, abl_out / fname, optional=True)

    # ───────── 7. Figure-generation + analysis scripts ─────────
    print("\n[7/8]  Scripts to reproduce all figures + DML")
    scr_out = OUT / "scripts"
    scr_out.mkdir(parents=True, exist_ok=True)
    paper_scripts = [
        # DML refit pipeline
        "dml_canonical_2026-05-24.py",
        "dml_drop_llms_fast.py",
        # Canonical figures (14 fig01-fig14)
        "make_figures_canonical_2026-05-24.py",
        # DML framework diagram
        "make_dml_framework.py",
        # Stage-F figures
        "make_fig_probing.py",
        "make_fig_admission_probe.py",
        "make_fig_saliency.py",
        # Diagnostics
        "diag_freshness_leakage.py",
        # Build script (this one)
        "build_hf_dataset.py",
    ]
    for s in paper_scripts:
        copy_one(REPO / "scripts" / s, scr_out / s, optional=True)
    # also include the interpretability/ package since probing.py is a module
    copy_tree_filtered(REPO / "interpretability", OUT / "interpretability",
                       patterns=["*.py"])
    print(f"  [copy-tree] interpretability/*.py module")

    # ───────── 8. Paper-text artifacts ─────────
    print("\n[8/8]  Paper-text artifacts")
    docs_out = OUT / "docs"
    docs_out.mkdir(parents=True, exist_ok=True)
    paper_docs = [
        "docs/2026-05-24/table2_updated_no_llms_treatment.md",
        "docs/2026-05-24/probing_section_final.md",
        "docs/2026-05-24/probing_caveat_paragraph.md",
        "docs/2026-05-24/freshness_leakage_diagnostic.md",
    ]
    for d in paper_docs:
        src = REPO / d
        copy_one(src, docs_out / Path(d).name, optional=True)

    # ───────── final: README + dataset card + reproducer ─────────
    write_readme()
    write_dataset_card()
    write_reproducer()

    # size summary
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\n=====  Built {OUT}  total size = {total/1024/1024:.1f} MB  =====")


# ---------------------------------------------------------------------------
def write_readme():
    txt = f"""# GEODML — EMNLP 2026 reproducibility dataset

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.

This dataset accompanies the EMNLP 2026 submission *Causal Analysis of
LLM Search Rerankers via Double/Debiased Machine Learning*. It contains
every input table, every fitted-model output, and every script needed
to reproduce the paper's tables, the 14 canonical figures, the three
Stage-F mechanism figures (probing, admission probe, saliency), and the
DML estimation framework diagram.

## Directory layout

```
data/
├── main/         Per-(keyword, url, variant) experimental table.
│                 65K rows × 73 columns: treatments T1-T6, has_llms_txt,
│                 28 confounders, ranks, admission, model, variant.
│   ├── full_experiment_data.parquet                  (unified)
│   ├── full_experiment_data_biased.parquet           (per-variant)
│   ├── full_experiment_data_neutral.parquet
│   ├── full_experiment_data_biased_rag.parquet
│   ├── full_experiment_data_neutral_rag.parquet
│   └── regression_dataset.parquet                    (+ T4_code, has_llms_txt)
│
├── serp/         Phase-0 SERP snapshots, used to build rerank prompts.
│   ├── phase0_top{{20,50}}_searxng.parquet
│   └── phase0_top{{20,50}}_ddg.parquet
│
├── dml/          DML fitted-model outputs (216 models for the headline,
│                 252 for the legacy spec that included T7=llms.txt).
│   ├── dml_canonical_2026-05-25_llms_as_confounder.parquet   ← HEADLINE
│   ├── dml_canonical_2026-05-25_no_llms.parquet              (sensitivity)
│   ├── dml_canonical_2026-05-24.parquet                      (legacy)
│   ├── confounder_loo_r2.parquet                             (LOO R² per confounder)
│   ├── confounder_ols_significance.parquet
│   └── nuisance_r2.parquet                                   (g, m model R²)
│
├── probing/      Linear-probe ROC AUC per (model, treatment, layer, pooling).
│                 Includes both content-treatment probes (T1-T6) and the
│                 admission pre-commitment probe (Y1_admission_inctx).
│   └── probing_results_{{biased,neutral,biased_rag,neutral_rag}}.csv
│
├── saliency/     Gradient-saliency aggregates per (model, treatment).
│                 Per-token scores omitted from this release (~5 GB);
│                 the per-cell summaries are sufficient to reproduce
│                 the saliency figure.
│   ├── saliency_summary_full.csv
│   ├── saliency_summary_rw.csv
│   └── per_cell/saliency_<MODEL>_<VARIANT>/saliency_summary_{{full,rw}}.csv
│
└── ablation/     Token-level ablation: rank change when treatment tokens
                  are stripped from a URL's snippet/title.
    ├── ablation_results_full_{{biased,neutral}}.csv
    └── ablation_results_rw_{{biased,neutral}}.csv

scripts/         All figure-generation and analysis code.
interpretability/  The interpretability package (probing, saliency, ablation).
docs/             Paper-text drop-ins (LaTeX tables, methodology paragraphs).
```

## Reproducing the paper's figures

Install:
```bash
pip install pandas numpy matplotlib lightgbm scikit-learn pyarrow scipy \
            sentence-transformers selectolax beautifulsoup4 lxml
```

Then from the dataset root:
```bash
export GEODML_DATA_ROOT="$PWD"     # the scripts read $GEODML_DATA_ROOT/data/...
bash reproduce_all.sh
```

This runs:
1. `scripts/dml_canonical_2026-05-24.py` — refits the 216 DML models
   (~25 min on a modern multi-core machine). Output:
   `data/dml/dml_canonical_2026-05-25_llms_as_confounder.parquet`.
   You can skip this and use the pre-computed parquet.
2. `scripts/make_figures_canonical_2026-05-24.py` — renders the 14
   canonical figures (fig01-fig14) into `figures/`.
3. `scripts/make_dml_framework.py` — renders the framework diagram.
4. `scripts/make_fig_probing.py` — renders the layer-wise probing
   figures.
5. `scripts/make_fig_admission_probe.py` — renders the admission
   pre-commitment figures.
6. `scripts/make_fig_saliency.py` — renders the saliency bar chart.

Each script accepts no required arguments; all paths are derived from
the dataset structure.

## Key conventions

- **6 canonical treatments**: T1b stats density, T2a question headings,
  T3 structured data (JSON-LD), T4 citation authority, T5 topical
  competence (keyword–page cosine), T6 freshness. T7=has_llms_txt is
  included in the confounder set X because the rerankers under study
  do not retrieve the `/llms.txt` file at inference time.
- **28 confounders**: page-HTML features (6), SERP-derived (4),
  semantic IR (3), DFS Whois (7), DFS keyword-level (8). See
  `docs/treatments_and_confounders.md` for the precise list.
- **3 outcomes**: Y1 = `selected_by_llm` (binary admission), Y2 =
  `rank_delta` = `rank_pre - rank_post`, Y3 = `rank_post`.
- **Sign convention**: positive coefficients on Y1 and Y2 favour the
  URL; negative coefficients on Y3 (lower post-rank = better
  position) favour the URL.

## Citation

```bibtex
@inproceedings{{fourel2026geodml,
  title     = {{Causal Analysis of LLM Search Rerankers via Double/Debiased Machine Learning}},
  author    = {{Fourel, Valerian}},
  booktitle = {{Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing}},
  year      = {{2026}}
}}
```

## License

CC BY 4.0.

## Contact

valerian.fourel@gmail.com
"""
    (OUT / "README.md").write_text(txt)
    print(f"  [write] README.md  ({len(txt)} chars)")


def write_dataset_card():
    txt = """---
language:
  - en
license: cc-by-4.0
pretty_name: GEODML — Causal Analysis of LLM Search Rerankers
size_categories:
  - 10K<n<100K
tags:
  - causal-inference
  - double-ml
  - dml
  - llm
  - reranker
  - search
  - geo
  - generative-engine-optimization
  - interpretability
  - probing
  - saliency
  - ablation
task_categories:
  - text-ranking
  - text-classification
---

# GEODML — EMNLP 2026 reproducibility dataset

See `README.md` in the dataset root for the full directory layout and
reproduction instructions.

**Summary.** This dataset contains every input table and every
fitted-model output needed to reproduce the figures and tables of the
EMNLP 2026 submission. It comprises:
- 65K-row experimental main table with treatments, confounders, ranks,
  and LLM admission decisions across 4 prompt variants
  ({biased, neutral} × {snippet-only, snippet+RAG}) and 2 reranker
  backbones (Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct).
- 216 Double/Debiased Machine Learning fitted models covering 6
  content treatments × 3 outcomes × 11 sample slices.
- ~10K probing rows (linear-probe ROC AUC per layer × pooling).
- 4 ablation outputs and the saliency summary aggregates.
- All 14 canonical figure-generation scripts plus the three Stage-F
  scripts (probing, admission probe, saliency).

**What's NOT included** (deliberately): the LLM weights (use the
official HF repos), the raw scraped HTML cache (~5 GB), and the
per-token saliency `.npz` chunks (~5 GB). The summaries in
`data/saliency/per_cell/` are sufficient to reproduce all reported
saliency results.
"""
    (OUT / "dataset_card.md").write_text(txt)
    print(f"  [write] dataset_card.md")


def write_reproducer():
    txt = """#!/usr/bin/env bash
# reproduce_all.sh — runs every analysis + figure script in order.

set -e

if [ -z "${GEODML_DATA_ROOT:-}" ]; then
  export GEODML_DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
  echo "[reproduce] GEODML_DATA_ROOT not set; defaulting to $GEODML_DATA_ROOT"
fi

cd "$(dirname "$0")"
mkdir -p figures

echo
echo "==========  Step 1/4  DML refit  =========="
echo "(Skip with --skip-dml if you want to use the precomputed parquet.)"
if [ "${1:-}" != "--skip-dml" ]; then
  python scripts/dml_canonical_2026-05-24.py
fi

echo
echo "==========  Step 2/4  Canonical figures (fig01-fig14)  =========="
python scripts/make_figures_canonical_2026-05-24.py

echo
echo "==========  Step 3/4  DML framework diagram  =========="
python scripts/make_dml_framework.py

echo
echo "==========  Step 4/4  Stage-F figures (probing, admission, saliency)  =========="
python scripts/make_fig_probing.py
python scripts/make_fig_admission_probe.py
python scripts/make_fig_saliency.py

echo
echo "Done. All figures landed in figures/."
"""
    p = OUT / "reproduce_all.sh"
    p.write_text(txt)
    p.chmod(0o755)
    print(f"  [write] reproduce_all.sh  (chmod +x)")


# ---------------------------------------------------------------------------
def write_treatments_doc():
    txt = """# Canonical treatment + confounder definitions

## 6 content treatments (T1-T6)

| Symbol | Column (main table)               | What it measures |
|--------|-----------------------------------|------------------|
| T1b    | `treat_stats_density`             | Numeric tokens density: count of \\d patterns per 1K words |
| T2a    | `treat_question_headings`         | Binary: page has at least one question-form heading |
| T3     | `treat_structured_data`           | Binary: page exposes JSON-LD or schema.org markup |
| T4     | `T4_citation_authority_code`      | Continuous citation-authority score (deterministic, see Phase-3 docs) |
| T5     | `treat_topical_comp`              | Cosine similarity between keyword and page-body sentence-transformer embeddings (all-MiniLM-L6-v2). Topical competence / relevance. |
| T6     | `treat_freshness`                 | 5-bin discretization of "years since publication" (0 = oldest, 4 = newest) |

`has_llms_txt` is included in the confounder set X rather than as a
treatment, because the rerankers do not retrieve the `/llms.txt` file
at inference.

## 28 confounders

Page-HTML (6):  `conf_word_count`, `conf_readability`,
`conf_internal_links`, `conf_outbound_links`, `conf_images_alt`,
`conf_https`.

SERP-derived (4):  `conf_title_has_kw`, `conf_title_len`,
`conf_snippet_len`, `conf_serp_position`.

Semantic IR (3):  `conf_title_kw_sim`, `conf_snippet_kw_sim`,
`conf_bm25`.

DFS Whois (7):  `conf_domain_authority`, `conf_backlinks`,
`conf_referring_domains`, `conf_brand_recog`, `conf_dfs_paid_count`,
`conf_dfs_etv`, `conf_dfs_domain_age_years`.

DFS keyword-level (8):  `dfs_keyword_difficulty`, `dfs_search_volume`,
`dfs_cpc`, `dfs_competition`, `dfs_intent_commercial`,
`dfs_intent_informational`, `dfs_intent_navigational`,
`dfs_intent_transactional`.

GEO-intent proxy (1):  `has_llms_txt`.

## 3 outcomes

| Symbol | Column                | Meaning |
|--------|-----------------------|---------|
| Y1     | `selected_by_llm`     | Binary admission: 1 if URL was kept by LLM in its top-K |
| Y2     | `rank_delta`          | `rank_pre - rank_post`. Positive = URL moved UP (promoted) |
| Y3     | `rank_post`           | Final LLM-assigned position. Lower = better (1 is top). |

Sign convention for "promoter direction":  +Y1, +Y2, -Y3.
"""
    (OUT / "docs" / "treatments_and_confounders.md").write_text(txt)
    print(f"  [write] docs/treatments_and_confounders.md")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build()
    write_treatments_doc()
    print(f"\nDataset built at {OUT}")
    print(f"Push to HuggingFace with:")
    print(f"  huggingface-cli upload <user>/geodml-emnlp-2026 {OUT}/ . --repo-type=dataset")
