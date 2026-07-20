# Heavy-data pointers (not copied into this archive)

Everything below stays in the source repos (~100 GB total). Sizes measured 2026-07-12.

## `~/Hamburg/GEODML` (42 GB) — original pipeline repo

| Path | Size | Contents |
|---|---|---|
| `paperSizeExperiment/` | 34 GB | Scaled 1,011-keyword cross-model study: own scripts, `output/` per-cell runs, `consolidated_results/` (incl. 17 GB `dml_study*/` + `merged/`; small summaries already copied here). |
| `huggingface_bundle/` | 3.4 GB | Published HF dataset "GEODML — Paper-Size Experiment" (65,203 rows, 570 DML fits). |
| `Perplexica/` | 1.4 GB | Third-party AI-search UI (infra dependency only). |
| `venv/`, `venv312/` | ~2 GB | Python venvs — regenerable. |
| `50_larger/` | 596 MB | Large-pool experiment data + `html_cache/` (figures copied here). |
| `results/`, `output/` | ~360 MB | Primary-experiment raw outputs; mostly `html_cache/` dirs. |
| `searxng-local/`, `searxng-config/` | 162 MB | Local SearXNG checkout + config. |
| `pipeline/` | 107 MB | 10-treat/16-conf pipeline; `intermediate/` (features_new.csv, moz_data.csv, embeddings.npz), `earned_media_results/earned_html_cache/` (73 MB). |

## `~/Hamburg/geodml-dataset` (55 GB) — HF-push tree (**DELETED locally ~2026-07**; sole surviving copy: HF `ValerianFourel/geodml-papersize`)

| Path | Size | Contents |
|---|---|---|
| `data/runs/` | 44 GB | Per-cell Stage-A outputs: extracted `html_cache/` (bulk of repo), `keywords.jsonl`, rerank checkpoints, `rankings.csv` — 44 run dirs (the "45" previously noted here was a miscount; verified against the HF tree 2026-07-20). |
| `data/rag_index/` | 5.7 GB | RAG retrieval index: 906k chunks, float32 chunk embeddings (~5.3 GB), retrieved_top3. |
| `interpretability/output/` | 3.6 GB | Stage-F ablation/saliency/weights/probing artifacts + plots. |
| `archives/` | 606 MB | Cluster + local result zips. |
| `data/order_probe/` | 177 MB | Order-sensitivity probe JSONLs (seeds 42/123) + summary parquet. |
| `data/main/` | 74 MB | `full_experiment_data_<variant>.parquet` ×4 + `full_experiment_unified.parquet`. |
| `data/dataforseo/`, `data/serp/`, `data/features/`, `data/dml_results/` | <80 MB | Confounders, input SERPs, Stage-B features, Stage-D DML parquets + comparison CSVs. |

## `~/Hamburg/GEODML_Analysis` (11 GB) — analysis repo (git)

| Path | Size | Contents |
|---|---|---|
| `geodml_data/` | 4.4 GB | Mirror of private HF dataset `ValerianFourel/geodml-emnlp-2026` (serp, runs, features, main, order_probe, dml_results, dataforseo, docs, manifests). |
| `interpretability/output/` | 3.6 GB | Ablation/saliency/probing compute outputs (gitignored). |
| `archives/` | 749 MB | Zips of prior local results + passage-run archive. |
| `papers/` | 81 MB | 46 reference PDFs (index copied to `03_analysis_GEODML_Analysis/papers_index/`). |
| `scripts/` | small | ~60 analysis/repair/figure scripts — live in git, not duplicated here. |

## Remote / published

- HF dataset (analysis input): `ValerianFourel/geodml-emnlp-2026` (private)
- HF dataset (full raw, 41,882 files / 37.6 GB logical): `ValerianFourel/geodml-papersize` — the name `geodml-papersize-full` in old scripts was a phantom; the push went to `geodml-papersize`.
- JUPITER Booster (`scifi` project): bf16-full reconciliation runs — see
  `03_analysis_GEODML_Analysis/docs/2026-05-18/runbook-jupiter-2026-05-18.md`.
