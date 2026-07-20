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


def copy_tree(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    """Mirror src_dir → dst_dir verbatim. Returns (n_files, total_bytes)."""
    if not src_dir.exists():
        print(f"  [skip] {src_dir} (missing)")
        return 0, 0
    n_files = 0
    n_bytes = 0
    for p in src_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src_dir)
            (dst_dir / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst_dir / rel)
            n_files += 1
            n_bytes += p.stat().st_size
    return n_files, n_bytes


# ---------------------------------------------------------------------------
def build():
    if OUT.exists():
        print(f"WARNING: {OUT} already exists. Existing files will be overwritten.")
    OUT.mkdir(exist_ok=True)

    # ───────── 1. Main experimental data ─────────
    print("\n[1/13]  Main experimental tables")
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
    print("\n[2/13]  SERP snapshots (phase 0)")
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
    print("\n[3/13]  DML fitted-model outputs")
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
    print("\n[4/13]  Probing results (treatment + admission)")
    probing_out = OUT / "data" / "probing"
    probing_out.mkdir(parents=True, exist_ok=True)
    interp_dir = REPO / "interpretability" / "output"
    for v in ("biased", "neutral", "biased_rag", "neutral_rag"):
        copy_one(interp_dir / f"probing_results_{v}.csv",
                 probing_out / f"probing_results_{v}.csv", optional=True)

    # ───────── 5. Saliency summaries (per-token scores omitted; too large) ─────────
    print("\n[5/13]  Saliency summaries")
    sal_out = OUT / "data" / "saliency"
    sal_out.mkdir(parents=True, exist_ok=True)
    for fname in ("saliency_summary_full.csv", "saliency_summary_rw.csv"):
        copy_one(interp_dir / fname, sal_out / fname, optional=True)
    # also include the smaller per-cell SUMMARY (not per-token scores)
    n = copy_tree_filtered(interp_dir, sal_out / "per_cell",
                           patterns=["saliency_*-Instruct_*/saliency_summary_*.csv"])
    print(f"  [copy-tree] {n} per-cell summary files under per_cell/")

    # ───────── 6. Ablation results ─────────
    print("\n[6/13]  Ablation results")
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
    print("\n[7/13]  Scripts to reproduce all figures + DML")
    scr_out = OUT / "scripts"
    # Wipe scripts/ so renamed-away files don't linger between builds.
    if scr_out.exists():
        shutil.rmtree(scr_out)
    scr_out.mkdir(parents=True, exist_ok=True)
    paper_scripts = [
        # Shared path resolver — every other script imports it
        "_paths.py",
        # DML refit pipeline
        "dml_canonical.py",
        "dml_sensitivity_drop_llms.py",
        # Canonical figures (fig01-fig14)
        "make_canonical_figures.py",
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
    print("\n[8/13]  Paper-text artifacts")
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

    # ───────── 9. Page-level features (pre-aggregation) ─────────
    print("\n[9/13]  Page-level features (pre-aggregation)")
    feat_out = OUT / "data" / "features"
    n, b = copy_tree(DATA / "features", feat_out)
    print(f"  [copy-tree] {n} files, {b/1024/1024:.1f} MB → data/features/")

    # ───────── 10. DataForSEO raw confounder inputs ─────────
    print("\n[10/13]  DataForSEO raw confounder inputs")
    dfs_out = OUT / "data" / "dataforseo"
    n, b = copy_tree(DATA / "dataforseo", dfs_out)
    print(f"  [copy-tree] {n} files, {b/1024/1024:.1f} MB → data/dataforseo/")

    # ───────── 11. RAG coverage / dataset-gap bridge tables ─────────
    print("\n[11/13]  RAG coverage / dataset-gap bridge tables")
    cov_out = OUT / "data" / "coverage"
    n, b = copy_tree(DATA / "coverage", cov_out)
    print(f"  [copy-tree] {n} files, {b/1024/1024:.1f} MB → data/coverage/")

    # ───────── 12. Raw LLM rerank outputs (phase 2) ─────────
    print("\n[12/13]  Raw LLM rerank outputs (phase 2)")
    runs_out = OUT / "data" / "runs"
    n, b = copy_tree(DATA / "runs", runs_out)
    print(f"  [copy-tree] {n} files, {b/1024/1024:.1f} MB → data/runs/")

    # ───────── 13. Order-probe results (presentation-order sub-experiment) ─────────
    print("\n[13/13]  Order-probe results (presentation-order sub-experiment)")
    op_out = OUT / "data" / "order_probe"
    n, b = copy_tree(DATA / "order_probe", op_out)
    print(f"  [copy-tree] {n} files, {b/1024/1024:.1f} MB → data/order_probe/")

    # ───────── final: README + dataset card + reproducer ─────────
    write_readme()
    write_dataset_card()
    write_reproducer()

    # size summary
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"\n=====  Built {OUT}  total size = {total/1024/1024:.1f} MB  =====")


# ---------------------------------------------------------------------------
def write_readme():
    txt = """---
language:
  - en
license: cc-by-4.0
pretty_name: GEODML — Causal Analysis of LLM Search Rerankers (Full Reproducibility Dataset)
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
  - emnlp
  - emnlp-2026
task_categories:
  - text-ranking
  - text-classification
---

# GEODML — EMNLP 2026 full reproducibility dataset

Companion data + code for the EMNLP 2026 submission
**"Causal Analysis of LLM Search Rerankers via Double/Debiased Machine
Learning."** Contains *every* input table, every fitted-model output,
every figure-generation script, **and the raw LLM rerank outputs**
needed to reproduce the paper end-to-end.

- **Condensed reviewer pack** (5.6 MB, fast verification) →
  [`ValerianFourel/geodml-emnlp-2026-reviewer`](https://huggingface.co/datasets/ValerianFourel/geodml-emnlp-2026-reviewer)
- **This dataset** (1.8 GB, full pipeline) → re-runs everything from
  raw LLM outputs through DML estimation through final figures.

---

## TL;DR — what's inside

| Layer | Path | Size | What it is |
|---|---|---|---|
| Main experimental table | `data/main/` | 41 MB | 65K-row (keyword × url × variant) table — treatments T1–T6, has_llms_txt, 28 confounders, ranks, LLM admission. The DML input. |
| SERP snapshots (phase 0) | `data/serp/` | 6.4 MB | Frozen top-20 / top-50 search results from DuckDuckGo + SearXNG. Pre-rerank URL pools. |
| Page-level features | `data/features/` | 8.3 MB | Per-(keyword, url) page features before they're aggregated into the main table. Source of the HTML/SERP confounders. |
| DataForSEO confounders | `data/dataforseo/` | 4.3 MB | Raw DFS API responses (domain authority, keyword difficulty, intent shares). |
| RAG coverage | `data/coverage/` | 36 KB | Which keywords have RAG snippets vs not — supports the Spec-B sample-size claim. |
| **Raw LLM rerank outputs** | `data/runs/` | **813 MB** | One `phase2/keywords.jsonl` per (backend × model × pool × variant × seed). The source-of-truth LLM admission/order decisions the main table is derived from. |
| Order-probe experiment | `data/order_probe/` | 906 MB | Same prompts as `runs/` but with shuffled URL-list order (seeds 42 / 123). Source of the ordering-effect appendix. |
| DML fitted models | `data/dml/` | 80 KB | 216 fitted Double/Debiased ML models (6 treatments × 3 outcomes × 11 slices × 2 specs, plus sensitivity + legacy specs). |
| Linear probes | `data/probing/` | 1.3 MB | ROC AUC per (model, treatment, layer, pooling). Treatment probes (T1–T6) + admission pre-commitment probe. |
| Saliency | `data/saliency/` | 72 KB | Gradient-saliency aggregates per (model, treatment). Per-token .npz omitted (~5 GB). |
| Ablation | `data/ablation/` | 23 MB | Token-level ablation Δrank when treatment tokens are stripped from a URL's snippet/title. |
| **Total** | | **1.8 GB** | |

---

## What we did (study design)

**Question.** When an LLM is asked to rerank a list of search results,
which content features of a web page *cause* the LLM to keep that URL
in its top-K and to push it toward the top — as opposed to merely
correlating with admission because the LLM happens to like
high-authority sites?

**Method.** Six content features are declared a priori as treatments
and tested with Double/Debiased Machine Learning (Robinson partial
linear model with LightGBM nuisance):

| | Treatment | Column in `data/main/` | Measures |
|---|---|---|---|
| T1b | stats density | `treat_stats_density` | numeric-token density per 1K words |
| T2a | Q-headings | `treat_question_headings` | binary: page has any question-form heading |
| T3 | schema (JSON-LD) | `treat_structured_data` | binary: page exposes JSON-LD / schema.org |
| T4 | citation authority | `T4_citation_authority_code` | deterministic citation-authority score |
| T5 | topical competence | `treat_topical_comp` | keyword–page cosine (MiniLM-L6) |
| T6 | freshness | `treat_freshness` | 5-bin discretization of years-since-publication |

Two specifications per (treatment × outcome):

- **Spec A (marginal):** focal treatment T_k controlled for the 28
  confounders X only.
- **Spec B (mutually controlled):** focal treatment T_k controlled for
  the 28 confounders X **and** the other 5 treatments. This is what
  Table 2 reports — it isolates the effect of each treatment net of
  every other treatment's contribution.

**Design grid.** 4 prompt variants
({biased, neutral} × {snippet-only, +RAG}) × 2 reranker backbones
(Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct) × 2 SERP backends
(DuckDuckGo, SearXNG) × 2 candidate-pool sizes (top-20, top-50) →
65K (keyword × url × condition) rows feeding 216 fitted DML models
(6 treatments × 3 outcomes × 11 slices × 2 specs, plus a sensitivity
spec that drops `has_llms_txt`).

**Mechanism check.** DML alone cannot say *how* the LLM uses a
feature, only whether the feature has a net causal effect. We verify
mechanism with three orthogonal interpretability probes on the LLM's
hidden activations:

1. **Linear probes** (per layer × pooling) for each treatment's
   in-context recognizability.
2. **Gradient saliency** — does the LLM actually look at treatment
   tokens when forming its admission decision?
3. **Token ablation** — does Δrank change when treatment tokens are
   stripped from the URL's snippet/title?

A treatment passes the mechanism check only if at least two of the
three probes agree with the sign of its DML coefficient.

## Headline findings (Table 2 — Spec B POOLED)

Reading in **promoter direction** (`+Y1`, `+Y2`, `−Y3` = URL favoured
by the LLM):

| Treatment | Y1 selected | Y2 Δrank | Y3 rank_post | Verdict |
|---|---|---|---|---|
| **T5 topical competence** | **+0.037*** | **−0.530*** | **−0.299*** | Dominant promoter (every outcome, every slice). |
| **T2a Q-headings** | **+0.016*** | **+0.136*** | **−0.041*** | Small but robust promoter. |
| **T3 schema (JSON-LD)** | **−0.014*** | **−0.051*** | **+0.095*** | Small demoter — JSON-LD does NOT help. |
| T6 freshness | **−0.005*** | **−0.061*** | +0.005 | Null / marginal. |
| T4 citation authority | +0.001 | **−0.023*** | **−0.015*** | Null on admission, marginal on rank. |
| T1b stats density | −0.000 | −0.003 | −0.002 | **Null DML** — *but* Qwen saliency = 1.93×. |

Stars: `***` p < .001, `**` p < .01, `*` p < .05.

**Attention ≠ effect (the T1b finding).** Qwen's gradient saliency on
stats-density tokens is 1.93× the surrounding-context baseline — the
model clearly *looks* at numbers — yet the DML coefficient is
indistinguishable from zero. The interpretability story would have
been "Qwen is a stats-oriented reranker"; the DML story is "stats
density doesn't move ranks once you control for what stats density
co-varies with." We treat this as the paper's central methodological
contribution.

---

## Quick start (60 seconds)

```bash
# 1. Pull the dataset
huggingface-cli download ValerianFourel/geodml-emnlp-2026 \\
    --repo-type=dataset --local-dir ./geodml-emnlp-2026
cd geodml-emnlp-2026

# 2. Install
pip install pandas numpy matplotlib lightgbm scikit-learn \\
            pyarrow scipy sentence-transformers \\
            selectolax beautifulsoup4 lxml

# 3. Reproduce every figure + DML table
export GEODML_DATA_ROOT="$PWD"
bash reproduce_all.sh           # ~25 min on a modern CPU

# 4. Or use the pre-computed DML output to skip the 25-min refit
bash reproduce_all.sh --skip-dml
```

### Hardware

- **CPU-only path (DML + figures):** any 8+ core modern x86_64 / arm64
  machine with 16 GB RAM. ~25 min for the DML refit, ~1 min for all
  figures.
- **GPU path (probing + saliency, NOT in `reproduce_all.sh`):** the
  per-layer linear probes and gradient-saliency scoring require a
  single A100/H100-class GPU and the local LLM weights
  (Llama-3.3-70B, Qwen-2.5-72B). Pre-computed probe/saliency CSVs are
  included so you can skip this step.

### Python version

Tested on Python 3.11. Anything ≥ 3.10 should work.

---

## Directory layout (with file-by-file descriptions)

```
geodml-emnlp-2026/
│
├── README.md                          ← this file
├── dataset_card.md                    ← duplicate of frontmatter (HF metadata)
├── reproduce_all.sh                   ← runs the whole pipeline end-to-end
│
├── data/
│   │
│   ├── main/                          ── PRIMARY EXPERIMENTAL TABLE ──
│   │   ├── full_experiment_data.parquet            65K rows, all variants pooled
│   │   ├── full_experiment_data_biased.parquet     biased-prompt slice
│   │   ├── full_experiment_data_neutral.parquet    neutral-prompt slice
│   │   ├── full_experiment_data_biased_rag.parquet   biased + RAG snippets
│   │   ├── full_experiment_data_neutral_rag.parquet  neutral + RAG snippets
│   │   └── regression_dataset.parquet              + T4_code + has_llms_txt
│   │
│   │   Columns: keyword, url, variant, model, backend, pool_size,
│   │   ─────── seed, treat_stats_density, treat_question_headings,
│   │           treat_structured_data, T4_citation_authority_code,
│   │           treat_topical_comp, treat_freshness, has_llms_txt,
│   │           28 confounders (conf_*, dfs_*), rank_pre, rank_post,
│   │           rank_delta, selected_by_llm.
│   │
│   ├── serp/                          ── PHASE 0: pre-rerank URL pools ──
│   │   └── phase0_top{20,50}_{searxng,ddg}.parquet
│   │       Columns: keyword, rank_pre, url, title, snippet, backend.
│   │
│   ├── features/                      ── PAGE-LEVEL features pre-aggregation ──
│   │   ├── features_ddg_top{20,50}.parquet
│   │   ├── features_searxng_top{20,50}.parquet
│   │   └── dfs_keyword_confounders.parquet
│   │
│   ├── dataforseo/                    ── DFS API raw responses ──
│   │   └── domain_authority_dfs.parquet
│   │       (+ keyword-difficulty / intent-share parquets per keyword)
│   │
│   ├── coverage/                      ── RAG coverage / bridge tables ──
│   │   ├── rag_coverage.parquet               which keywords have RAG snippets
│   │   └── missing_rag_keywords.parquet       which don't (supports sample-size claim)
│   │
│   ├── runs/                          ── RAW LLM RERANK OUTPUTS ──
│   │   └── <backend>_<MODEL>_serp<POOL>_top10_<VARIANT>/phase2/keywords.jsonl
│   │
│   │   Each .jsonl row: {keyword, urls_pre, urls_post, selected, raw_response}
│   │   urls_pre     = pool of 20/50 URLs presented to the LLM
│   │   urls_post    = the LLM's reordered top-K subset
│   │   selected     = urls_post[:K] — the binary admission outcome
│   │   raw_response = the LLM's verbatim JSON/text output
│   │
│   ├── order_probe/                   ── ORDER-PROBE sub-experiment ──
│   │   └── <config>_seed{42,123}.jsonl
│   │
│   ├── dml/                           ── DML FITTED-MODEL OUTPUTS ──
│   │   ├── dml_canonical_2026-05-25_llms_as_confounder.parquet   ← HEADLINE (Table 2)
│   │   ├── dml_canonical_2026-05-25_no_llms.parquet              sensitivity
│   │   ├── dml_canonical_2026-05-24.parquet                      legacy
│   │   ├── confounder_loo_r2.parquet                             LOO R² per confounder
│   │   ├── confounder_ols_significance.parquet
│   │   └── nuisance_r2.parquet                                   nuisance-model R²
│   │
│   ├── probing/                       ── LINEAR-PROBE ROC AUC ──
│   │   └── probing_results_{biased,neutral,biased_rag,neutral_rag}.csv
│   │
│   ├── saliency/                      ── GRADIENT-SALIENCY SUMMARIES ──
│   │   ├── saliency_summary_full.csv
│   │   ├── saliency_summary_rw.csv
│   │   └── per_cell/saliency_<MODEL>_<VARIANT>/saliency_summary_{full,rw}.csv
│   │
│   └── ablation/                      ── TOKEN-ABLATION Δ RANK ──
│       └── ablation_results_{full,rw}_{biased,neutral}.csv
│
├── scripts/                           ── ALL ANALYSIS + FIGURE CODE ──
├── interpretability/                  ── INTERPRETABILITY PACKAGE ──
└── docs/                              ── PAPER-TEXT DROP-INS ──
```

## Reproducing the paper end-to-end

`reproduce_all.sh` is the single entry point:

```bash
export GEODML_DATA_ROOT="$PWD"
bash reproduce_all.sh                # full
bash reproduce_all.sh --skip-dml     # use pre-computed DML, just re-render figures
```

## Key conventions

- **6 canonical content treatments (T1–T6):** stats density,
  question-form headings, JSON-LD structured data, citation authority,
  topical competence (keyword–page cosine), freshness.
- **has_llms_txt** is a **confounder, not a treatment** — the
  rerankers studied don't retrieve the `/llms.txt` file at inference.
- **28 confounders** = 6 page-HTML + 4 SERP + 3 semantic-IR + 7 DFS
  Whois + 8 DFS keyword-level. See `docs/treatments_and_confounders.md`.
- **3 outcomes:**
  - Y1 = `selected_by_llm` (binary admission)
  - Y2 = `rank_delta = rank_pre − rank_post` (promotion magnitude)
  - Y3 = `rank_post` (final LLM-assigned position, lower = better)
- **Promoter direction:** +Y1, +Y2, −Y3 means the URL is favoured.

---

## How to analyze this dataset

Five copy-pasteable recipes. All assume you've set
`GEODML_DATA_ROOT="$PWD"` and you're in the dataset root.

### 1. Reproduce Table 2 (DML headline)

```python
import pandas as pd
dml = pd.read_parquet("data/dml/dml_canonical_2026-05-25_llms_as_confounder.parquet")
table2 = (dml[(dml.spec == "B") & (dml.slice == "POOLED")]
          [["outcome","treatment","coef","se","p_val","n","bonferroni_sig"]]
          .round({"coef":4,"se":4,"p_val":5})
          .sort_values(["outcome","coef"], ascending=[True, False]))
print(table2.to_string(index=False))
```

The `coef` column is the partial DML effect of the treatment on the
outcome, controlling for 28 confounders + the 5 other treatments.

### 2. Re-fit DML on any custom slice

```python
import sys; sys.path.insert(0, "scripts")
import pandas as pd
from _paths import MAIN
from dml_canonical import plr_estimate, CONFOUNDERS, TREATMENTS

df = pd.read_parquet(MAIN / "full_experiment_data_biased.parquet")
df = df[df.search_engine == "ddg"]            # any slice you like
focal = "treat_topical_comp"
other = [t for t in TREATMENTS if t != focal]  # Spec B (mutually controlled)

result = plr_estimate(df, focal_T=focal, ctrl_T=other,
                      X_cols=CONFOUNDERS,
                      outcome_col="selected_by_llm", is_clf=True)
print(result)   # → dict(coef, se, p_val, n)
```

Pass `ctrl_T=[]` for Spec A (marginal). Pass any other outcome column
(`rank_delta`, `rank_post`) with `is_clf=False` for the regression
outcomes.

### 3. Verify a single LLM admission against the raw model output

```python
import json, pandas as pd
main = pd.read_parquet("data/main/full_experiment_data_biased.parquet")

config = "ddg_Llama-3.3-70B-Instruct_serp20_top10_biased"
with open(f"data/runs/{config}/phase2/keywords.jsonl") as f:
    runs = {row["keyword"]: row for row in (json.loads(l) for l in f)}

kw = next(iter(runs))
admitted_raw = set(runs[kw]["selected"])
admitted_main = set(
    main[(main.keyword == kw)
         & (main.search_engine == "ddg")
         & (main.pool_size == 20)
         & main.selected_by_llm].url
)
print("match:", admitted_raw == admitted_main)
```

This is the integrity check a reviewer can run to confirm the binary
admission column in `data/main/` matches the actual LLM output.

### 4. Plot the layer-wise admission pre-commitment probe

```python
import pandas as pd, matplotlib.pyplot as plt
dfs = [pd.read_csv(f"data/probing/probing_results_{v}.csv")
       for v in ("biased","neutral","biased_rag","neutral_rag")]
adm = (pd.concat(dfs)
       .query("treatment == 'Y1_admission_inctx' and pooling == 'mean'")
       .groupby("layer")["roc_auc"].mean())
adm.plot(figsize=(7,3), title="Admission probe — mean pooling, 4-variant avg")
plt.axhline(0.5, ls="--", c="grey"); plt.xlabel("layer"); plt.ylabel("ROC AUC")
plt.tight_layout(); plt.show()
```

Expect L0 ≈ 0.67 (weak), peak ≈ 0.86 around layer 60 (~75% network
depth), L0→peak gain ≈ +0.19 — the model genuinely *composes* the
admission decision rather than reading it off the embedding.

### 5. Compute saliency × DML agreement

```python
import pandas as pd
sal = pd.read_csv("data/saliency/saliency_summary_full.csv")
sal["model"] = ["Llama-3.3-70B"] * (len(sal)//2) + ["Qwen-2.5-72B"] * (len(sal)//2)
dml = pd.read_parquet("data/dml/dml_canonical_2026-05-25_llms_as_confounder.parquet")
hdl = dml[(dml.spec == "B") & (dml.slice == "POOLED") & (dml.outcome == "selected")]
# saliency ratio > 1 ↔ model attends to the treatment
# DML coef > 0 ↔ treatment causally promotes admission
# A treatment is "real" only when both line up; T1b stats is the famous
# disagreement case (saliency 1.93× on Qwen, DML coef ≈ 0).
```

### What you do NOT need to re-run

The bundled CSVs in `data/probing/`, `data/saliency/`,
`data/ablation/` are already the outputs of the GPU-bound
interpretability code (a 70B-parameter forward pass per row). Reading
them is enough to reproduce every figure. You only need GPUs +
LLM weights if you want to re-extract from a *different* model or
prompt variant.

---

## What's deliberately NOT included

- **LLM weights.** Use the official HF repos:
  `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`.
- **Raw scraped HTML.** ~5 GB; we only release the extracted features.
- **Per-token saliency .npz.** ~5 GB; per-cell summary CSVs suffice.

## Citation

```bibtex
@inproceedings{fourel2026geodml,
  title     = {Causal Analysis of LLM Search Rerankers via Double/Debiased Machine Learning},
  author    = {Fourel, Valerian},
  year      = {2026},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing}
}
```

## License

CC BY 4.0 — free reuse with attribution.

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
# reproduce_all.sh — re-runs the DML fit + every figure in the paper.
#
# Usage:
#   bash reproduce_all.sh            # full re-run (~25 min for the DML refit)
#   bash reproduce_all.sh --skip-dml # reuse the precomputed parquet, just re-render figures
#
# Every script reads input from $GEODML_DATA_ROOT/data/... and writes figures
# to $GEODML_DATA_ROOT/figures/. The default root is this script's directory.

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
  python scripts/dml_canonical.py
fi

echo
echo "==========  Step 2/4  Canonical figures (fig01-fig14)  =========="
python scripts/make_canonical_figures.py

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
