# GEODML — What Drives LLM Re-Ranking?

Causal analysis of LLM search re-rankers with Double Machine Learning (DML).
Companion monorepo for the EMNLP 2026 submission
**"What Drives LLM Re-Ranking"** (ACL Rolling Review #9568,
[OpenReview `DjnTjAzZ8o`](https://openreview.net/pdf?id=DjnTjAzZ8o)).

The study asks which page- and domain-level properties *causally* move a
document up or down when a large language model re-ranks web search results —
separating genuine treatment effects (content statistics, freshness,
authority signals, llms.txt adoption, …) from confounding — across search
engines, models, prompt framings, and evidence conditions.

**Experimental grid.** 2 search engines (DuckDuckGo, SearXNG) × 2 LLMs
(Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct, bf16) × 2 SERP pool depths
(20, 50) × prompt variants (biased / neutral) × evidence conditions
(SERP snippet vs. RAG over fetched page content), ~1,000 commercial-intent
keywords, top-10 re-rank per query; plus a seeded order-randomization probe
and a passage-level pilot. DML fits use cross-fitted LightGBM/RF nuisances
with DataForSEO whois-based domain confounders.

---

## Repository layout

| path | was | contents |
|---|---|---|
| [`pipeline/`](pipeline/) | repo `GEODML` | Original local pipeline (Feb–Apr 2026): SERP collection, page scraping, LLM re-ranking, feature extraction, early DML studies (`paperSizeExperiment/`, `50_larger/`, `src/`, `both_analysis/`). See [`pipeline/EXPERIMENT_REGISTRY.md`](pipeline/EXPERIMENT_REGISTRY.md) and [`pipeline/FINDINGS.md`](pipeline/FINDINGS.md). |
| [`analysis/`](analysis/) | repo `GEODML_Analysis` | The paper-era pipeline (May 2026 →): ported HPC pipeline (`interpretability/pipeline/` — rerank, merge, features, DML, probing, saliency, ablation), SLURM tooling for JUPITER Booster (`scripts/slurm/`, repair loop, audits), analysis/figure scripts (`scripts/`), dated work logs (`docs/`), and the ARR rebuttal experiments (`rebuttal/`). |
| [`docs/`](docs/) | archive `GEODML_Unified` | Cross-repo archive extras: provenance of the (deleted) local dataset tree ([`docs/dataset_geodml-dataset/`](docs/dataset_geodml-dataset/)), curated early pipeline results ([`docs/pipeline_extra/`](docs/pipeline_extra/)), literature index, and [`docs/DATA_POINTERS.md`](docs/DATA_POINTERS.md) — where every artifact lives. |

The two source repos' full git histories are preserved under their subdirectories
(merged 2026-07-20; the standalone repos are retired).

## Data (Hugging Face)

| dataset | size | role |
|---|---|---|
| [`ValerianFourel/geodml-papersize`](https://huggingface.co/datasets/ValerianFourel/geodml-papersize) | 41,882 files / 37.6 GB | **Full raw archive** — per-run rerank outputs (`keywords.jsonl`, `rankings.csv`), HTML caches (tarballs), RAG index incl. chunk embeddings, order probe, SERP snapshots, DataForSEO pulls, consolidated features. The deleted 55 GB local tree's sole surviving copy. |
| [`ValerianFourel/geodml-emnlp-2026`](https://huggingface.co/datasets/ValerianFourel/geodml-emnlp-2026) | 1.8 GB | **Paper reproducibility bundle** — the exact May-24 Stage-C inputs, DML results, interpretability outputs, figures, `reproduce_all.sh`. Also the only holder of the whois `domain_authority_dfs.parquet` feeding the paper-final confounders. |
| [`ValerianFourel/geodml-emnlp-2026-reviewer`](https://huggingface.co/datasets/ValerianFourel/geodml-emnlp-2026-reviewer) | 10 MB | **Reviewer verification pack** — headline tables + figures + `verify.py`. |

**Reproducibility is verified end-to-end** (2026-07-20): downloading only
`keywords.jsonl` + features from `geodml-papersize` and running the committed
Stage-C merge (`analysis/scripts/build_main_table.py`) reproduces the archived
per-variant parquet **bit-exactly** (52/52 columns), and applying the whois
merge (`analysis/scripts/merge_dfs_domain_authority.py`) reproduces the
paper-final May-24 file (55/55 columns). Details:
[`analysis/docs/2026-07-20/hf_reconciliation_verify_2026-07-20.md`](analysis/docs/2026-07-20/hf_reconciliation_verify_2026-07-20.md).

## Pipeline at a glance

```
phase0  SERP collection            → data/serp/phase0_top{20,50}_{ddg,searxng}.*
phase2  page fetch + LLM re-rank   → data/runs/<engine>_<model>_serp<pool>_top10[_variant]/phase2/
phase3  page/domain features       → data/features/features_<engine>_top<pool>.parquet
RAG     chunk + embed + retrieve   → data/rag_index/<engine>_top<pool>/
StageC  merge → DML-ready parquet  → data/main/full_experiment_data_<variant>.parquet
  +     DataForSEO whois merge     → paper-final confounder columns
DML     cross-fitted PLR fits      → data/dml_results/          (analysis/interpretability/pipeline/dml.py)
StageF  probing / saliency / ablation → interpretability/output/
probe   order-randomization probe  → data/order_probe/
```

Compute: single-node runs locally (Stage C, DML, figures) and on
**JUPITER Booster** (SLURM) for GPU stages — see
[`analysis/docs/2026-05-18/runbook-jupiter-2026-05-18.md`](analysis/docs/2026-05-18/runbook-jupiter-2026-05-18.md)
(canonical end-to-end procedure) and `analysis/scripts/slurm/` (dispatch,
resumable repair loop, audits). External services used at collection time:
a local [SearXNG](https://github.com/searxng/searxng) instance
(config in [`pipeline/searxng-config/`](pipeline/searxng-config/)) and
[Perplexica](https://github.com/ItzCrazyKns/Perplexica) for early experiments
(both are upstream projects, not vendored here).

## Reproducing the paper numbers

1. Get the bundle: `huggingface-cli download ValerianFourel/geodml-emnlp-2026 --repo-type dataset` and run its `reproduce_all.sh`, **or** rebuild inputs from raw as verified above.
2. DML tables: `analysis/scripts/dml_canonical.py` (headline Spec B), `dml_selected.py`, `dml_code_vs_llm.py`, `rag_vs_nonrag.py`, `full_paper_analysis.py`.
3. Figures: `analysis/scripts/make_canonical_figures.py`, `make_fig_*.py`.
4. Rebuttal chain (ARR #9568): `analysis/rebuttal/` — `step0_reproduce.py` onward; outputs in `analysis/rebuttal/out/` incl. `REBUTTAL_NUMBERS.md`.

## Historical work logs

Day-by-day logs live in `analysis/docs/<date>/` (2026-04-27 → 2026-07-20);
`analysis/docs/2026-05-23/session_log_2026-05-23.md` and
`analysis/docs/2026-05-24/session_log_2026-05-24.md` document the final
pre-submission data fixes and findings; early-phase findings are in
`pipeline/FINDINGS.md` / `pipeline/results_findings.md` (curated copies under
`docs/pipeline_extra/`).

## License / citation

Data on Hugging Face is released CC-BY-4.0. If you use the code or data,
please cite ARR submission #9568 ("What Drives LLM Re-Ranking", under review,
EMNLP 2026). Contact: Valerian Fourel — valerian.fourel@gmail.com.
