# Provenance

Every artifact in this tree, where it came from, and when.

Last updated: 2026-05-17T11:00Z (LLM-execution-regime columns added; pre-JUWELS-bf16-redo state — see CHANGELOG.md entry 2026-05-17).

## 2026-05-17 update — `llm_backend` + `llm_precision` columns

Every record in `data/runs/<cell>/phase2/keywords.jsonl`,
`data/order_probe/*.jsonl`, and `data/main/full_experiment_data_<variant>.parquet`
now carries two new fields capturing the LLM execution regime:

| Field | Values | Meaning |
|---|---|---|
| `llm_parameters.backend` (JSONL) → `llm_backend` (parquet) | `local`, `api`, `openai` | Python class that served the inference. |
| `llm_parameters.precision` (JSONL) → `llm_precision` (parquet) | `bf16-full`, `4bit-nf4`, `api-hf`, `api-openai`, `unknown` | Canonical regime label per `precision_label()` in `interpretability/pipeline/rerank.py`. |

**Backfill methodology.** Historical records (pre-2026-05-17) were patched
in-place by `scripts/backfill_precision.py` using path-based inference:

- Variant suffix `_biased`/`_neutral` → `backend=local, precision=4bit-nf4`
  (these cells were produced on JUWELS booster via `run_rerank.sbatch`
  with the older `LocalRanker(quantize=True)` default).
- Variant suffix `_biased_rag`/`_neutral_rag`/`_biased_passage`/`_neutral_passage`
  → `backend=api, precision=api-hf` (produced 2026-05-08 via
  `scripts/finish_via_api.sh` with `BACKEND=hf`, which calls HF Inference
  endpoint at full precision).

The backfill is idempotent: records with `llm_parameters.precision` already
set are skipped. Files with mtime after the switchover (2026-05-17 00:00 UTC)
are skipped because they were written by the new code path.

**Breakdown after backfill** (current state):

```
biased         {'4bit-nf4': 45,967}   ← snippet on cluster
neutral        {'4bit-nf4': 52,256}   ← snippet on cluster
biased_rag     {'api-hf':   33,384}   ← RAG via HF Inference
neutral_rag    {'api-hf':   31,525}   ← RAG via HF Inference
```

A planned bf16-full re-run of the snippet arm on the JUWELS `scifi`
project (see `docs/runbook-full-cycle-2026-05-17.md`) will replace the
`4bit-nf4` records with `bf16-full`, making the snippet vs RAG comparison
identifiable at one precision class (full precision both arms).

## Source-of-truth map

| Path in dataset | Source | Coverage / count | Notes |
|---|---|---|---|
| `data/serp/phase0_top<pool>_<engine>.parquet` | cluster snapshot (HF dataset `archives/geodml_data_.zip`) | 4 files, 9k–14k rows each | input SERPs from the original DataForSEO + DDG/SearXNG capture |
| `data/dataforseo/` | cluster snapshot | full | keyword-level API outputs (search_volume, cpc, competition, keyword_difficulty, search_intent) |
| `data/runs/<engine>_<Model>_serp<N>_top10/phase2/html_cache/` | **2 sources merged**:<br>① upstream Mac (`~/Hamburg/GEODML/paperSizeExperiment/output/<legacy_cell>/html_cache/`) — initial population; `duckduckgo_` renamed to `ddg_`<br>② local scrape (`scripts/scrape_missing_html.py` 2026-05-07) — gap-fills | 11k–17k html files per cell after scrape (was 5.8k–7.9k before) | sha256(url)[:16].html naming; verified vs upstream cache writers |
| `data/runs/<cell>_<variant>/phase2/keywords.jsonl` | cluster rerank for snippet variants (biased, neutral); local API rerank for passage variants (smoke MAX_KW=20 as of 17:00) | 5,336 records each for snippet; 20 keywords each for passage smoke | `prompt_variant` recorded inline; `prompt` field shows what the LLM actually saw |
| `data/runs/<cell>_<variant>/phase2/.rerank_ckpt.json` | same as above | per-cell | `--resume` consumes these |
| `data/order_probe/*_seed{42,123}.jsonl` | cluster (snippet) + local API (passage smoke) | 96k summary rows | shuffled-input rerank for sensitivity analysis |
| `data/order_probe/order_probe_summary.parquet` | regenerated locally on 2026-05-07 | 96,048 → 98,928 rows | pairwise Jaccard / OAK overlap aggregation |
| `data/features/features_<engine>_top<pool>.parquet` | **built locally 2026-05-07** by `scripts/build_features_from_legacy.py` from upstream `paperSizeExperiment/output/<cell>/phase3/features_new.parquet` (union of both models per engine×pool) | 13,603–15,737 rows × 32 cols per file | 91% treatment non-null, includes `treat_topical_comp` and Moz `domain_authority` |
| `data/features/dfs_keyword_confounders.parquet` | built locally from `dataforseo/keyword_overview.parquet` and `search_intent.parquet` | 850 keywords × 9 cols | `dfs_intent_*` is 0/850 because keyword sets in the two DfS bundles are disjoint |
| `data/main/full_experiment_data_<variant>.parquet` | merge.py output, locally produced 2026-05-07 with `--external-features-parquet dfs_keyword_confounders.parquet` | biased 45,967 / neutral 52,256 / biased_passage 1,390 / neutral_passage 1,548 | 50 cols each |
| `data/main/full_experiment_unified.parquet` | `scripts/build_unified_main.py` 2026-05-07 | 101,161 rows × 48 cols | concatenation of the 4 variant tables with `axis_*` columns |
| `data/dml_results/dml_results_long_<variant>.parquet` | `python -m interpretability.pipeline.dml` runs 2026-05-07 | biased: 280 fits, 0 errors; neutral: 280 fits, 0 errors; passage variants: 280 fits with ~12 errors each (small-data overfit on binary treatments) | full PLR × {LGBM, RF} × 7 subsets × 10 treatments × 2 outcomes grid |
| `data/passages/passages_<engine>_top<pool>_max800.parquet` | `_build_passage_map` cache (rerank.py) | **archived 2026-05-08** to `archives/passage_runs_20260508_143229/passages/` — superseded by `data/rag_index/` | trafilatura output capped at 800 chars; replaced by full chunked extraction in rag_index |
| `data/rag_index/<engine>_top<pool>/full_passages.parquet` | `interpretability.pipeline.build_rag_index` 2026-05-08 | 4 cells, ~9k–14k URLs each | full body text from trafilatura, capped at 50k chars (vs 800 before) |
| `data/rag_index/<engine>_top<pool>/chunks.parquet` | `interpretability.pipeline.build_rag_index` 2026-05-08 | 188k–282k chunks per cell, **906k total** | char-recursive chunker (size=800, overlap=200, min_size=100) |
| `data/rag_index/<engine>_top<pool>/chunk_embeddings.npy` | OpenAI `text-embedding-3-small` (1536-dim, L2-normalized) 2026-05-08 | float32 (n_chunks × 1536), aligned to chunks.parquet rows | ~1.1–1.7 GB per cell, ~5.3 GB total |
| `data/rag_index/<engine>_top<pool>/keywords.parquet` | unique keyword list per cell, built 2026-05-08 | 980–1,011 keywords per cell | aligned to keyword_embeddings.npy |
| `data/rag_index/<engine>_top<pool>/keyword_embeddings.npy` | OpenAI `text-embedding-3-small` 2026-05-08 | float32 (n_keywords × 1536), L2-normalized | ~6 MB per cell |
| `data/rag_index/<engine>_top<pool>/retrieved_top3.parquet` | `interpretability.pipeline.rerank._build_retrieved_map` 2026-05-08 | (keyword, url) pairs from cell SERP × top-3 chunk join, ~10k–15k rows per cell | precomputed retrieval cache to avoid re-doing cosine sim on rerank resume |
| `data/rag_index/<engine>_top<pool>/meta.json` | `build_rag_index` 2026-05-08 | per cell | embedding_model, dim, chunk config, n_urls / n_chunks / n_keywords, built_at |
| `data/runs/<cell>_<biased_rag\|neutral_rag>/phase2/keywords.jsonl` | local API rerank 2026-05-08 (HF Inference Llama-3.3-70B + Qwen2.5-72B) | 16 cells × 400+ records each, 6,919 records total, 0 fallbacks | `prompt_variant` ∈ {biased_rag, neutral_rag}; per-result `passage:` is the top-K=3 retrieved chunks for that (keyword, url) joined by ` --- ` |
| `data/order_probe/<cell>_<biased_rag\|neutral_rag>_seed{42,123}.jsonl` | local API order-probe 2026-05-08 — **PARTIAL** | 6 cells fully done (400 each) + 1 partial (23 of 400) | stopped at HF 402 Payment Required; 25 of 32 cells pending — see `docs/RESUME-RAG.md` |
| `data/main/full_experiment_data_<biased_rag\|neutral_rag>.parquet` | `merge.py` 2026-05-08 | biased_rag: 33,384 rows × 42 cols (744 keywords, 8 runs); neutral_rag: 31,525 rows × 42 cols (615 keywords, 8 runs) | rank_delta non-null: 57.5% biased_rag, 89.8% neutral_rag |
| `data/dml_results/dml_results_long_<biased_rag\|neutral_rag>.parquet` | `dml.py` 2026-05-08 | full PLR × {LGBM, RF} × 7 subsets × 10 treatments × 2 outcomes grid (~280 fits each) | **in progress at update time** — both running in parallel from `merge` outputs |
| `archives/passage_runs_20260508_143229/` | local archival 2026-05-08 (~71 MB) | 16 `_passage` rerank dirs + 32 order_probe jsonls + 4 leading-body parquets + 4 Stage C/D parquets | superseded by `_rag` variants; kept for reproducibility but no longer in active pipeline |
| `interpretability/output/ablation_*/` `saliency_*/` `weights_*/` `probing_*/` | cluster Stage F snapshot (`archives/interpretability_.zip`) | ablation 48/48, saliency 16/16, weights 8/8, probing 2/8 | probing 6/8 still pending — needs cluster GPU |
| `interpretability/output/plots/figure_*.png` | regenerated locally 2026-05-07 by `interpretability.make_figures` | 13 figures including new `figure_a_dml_{biased,neutral,delta}.png` | based on cluster Stage F + local DML |
| `archives/local_results_<date>.zip` | local CPU runs, `continue_pipeline.sh` packaging | one per run | upload-ready single-commit bundles |

## Naming conventions

- **Engine prefix in run dirs**: `searxng_*` and `ddg_*`. The legacy upstream uses `duckduckgo_*`; the renaming happens at link/copy time so the new pipeline finds files at the canonical paths.
- **HTML cache filename**: `sha256(url)[:16] + ".html"`. **Confirmed against 6 upstream files** that all use the same hash function:
  - `GEODML/pipeline/gather_data.py:_url_to_cache_key`
  - `GEODML/pipeline/extract_features.py:_url_to_cache_key`
  - `GEODML/pipeline/rebuild_features.py:_url_to_cache_key`
  - `GEODML/paperSizeExperiment/save_phase2_progress.py:_url_to_cache_key`
  - `GEODML/paperSizeExperiment/export_html_cache.py:url_to_hash`
  - `GEODML/paperSizeExperiment/fill_confounders.py:url_hash`
- **Run id**: `<engine>_<ModelTag>_serp<N>_top<K>[_<variant>]` where variant ∈ {biased, neutral, biased_passage, neutral_passage, biased_rag, neutral_rag} or absent for the un-suffixed cell-base dir. As of 2026-05-08, `_passage` runs are archived (see archives/) — only the {biased, neutral, biased_rag, neutral_rag} four variants are in the active pipeline.
- **Variant decomposition** (in unified parquet):
  - `axis_prompt` ∈ {biased, neutral} — which prompt header was used
  - `axis_passage_mode` ∈ {snippet, passage, rag} — what was injected per result:
    - `snippet` → SERP snippet only (default biased/neutral)
    - `passage` → first 800 chars of body text, keyword-agnostic (archived 2026-05-08)
    - `rag` → top-K=3 retrieved chunks for that (keyword, url), keyword-conditional (new 2026-05-08)

## LLM config (matches cluster)

- `LLM_TEMPERATURE = 0.1`
- `LLM_MAX_TOKENS = 500`
- Source: `interpretability/pipeline/config.py`
- Both `LocalRanker` (cluster), `InferenceRanker` (HF API), and
  `OpenAIRanker` (DeepInfra/Together/Fireworks) read from the same constants.
- `finish_via_api.sh` asserts these values before any API call.

## DataForSEO confounders

| Column | Coverage | Source field |
|---|---|---|
| `dfs_search_volume` | 795/850 (94%) | `keyword_overview.parquet:ko.search_volume` |
| `dfs_cpc` | 664/850 (78%) | `keyword_overview.parquet:ko.cpc` |
| `dfs_competition` | 788/850 (93%) | `keyword_overview.parquet:ko.competition` |
| `dfs_keyword_difficulty` | 786/850 (92%) | `keyword_overview.parquet:ko.keyword_difficulty` |
| `dfs_intent_*` | 0/850 (0%) | `search_intent.parquet:si_main_intent` — keywords disjoint with keyword_overview |

## Bug history

| Bug | Discovered | Fixed | Commit |
|---|---|---|---|
| `url_to_html_filename` used MD5; upstream uses SHA-256. Every "passage variant" run in the project's history had empty passages. | 2026-05-07 16:00 | same day | `be385c7` `fix(html_cache): SHA-256 not MD5 for url→filename hash` |
| `scripts/build_main_table.py` ModuleNotFoundError on `interpretability.*` | 2026-05-07 13:30 | same day | `474e9df` |
| Stage B torch/numpy mismatch (torch 2.2.2 vs numpy 2.4.4 in venv) | 2026-05-07 13:00 | worked around with `--no-embed`; permanent fix is `pip install --upgrade torch>=2.4 numpy<2` | n/a (workaround) |
| DML duplicate-process race for variant=neutral | 2026-05-07 14:30 | both processes used `--resume`, no data corruption; user killed one to recover speed | n/a |
| `build_dataset_mirror.sh` first attempt out of disk at 6.9 GB | 2026-05-07 12:00 | freed disk and re-ran with `FORCE=1` | n/a |

## Build-time audit (initial 2026-05-07 03:06)

```
Stage A   24/32   Stage B   0/4    Stage C   0/4    Stage D   0/4    Stage F  74/80   Order probe  48/64
```

## Current audit (snapshot in AUDIT.txt)

```
Stage A   32/32   Stage B   4/4    Stage C   4/4    Stage D   4/4    Stage F  74/80   Order probe  64/64
```

(Stage D is 4/4 by file presence but only biased + neutral are reliable;
biased_passage / neutral_passage are partial — full passage rerun pending.)

## Git provenance (latest)

Run `git -C ~/Hamburg/GEODML_Analysis log -10 --oneline` for the full list. Today's commits include:

```
591e2d5 docs(utils): clarify SHA-256 hash matches all 6 upstream cache writers
be385c7 fix(html_cache): SHA-256 not MD5 for url→filename hash
5eb9eb5 perf(rerank): cache passages on disk per (engine, pool, max_chars)
97d1e6e feat(api): add BACKEND=hf default to finish_via_api.sh
474e9df feat(analysis): build_features_from_legacy.py reuses upstream phase3 features
178fdb3 feat(analysis): build_unified_main.py — single DML-ready parquet over all 32 cells
c4d4724 feat(scrape): scrape_missing_html.py — fill html_cache gaps
a75559d docs: comprehensive work log for 2026-05-07
```
