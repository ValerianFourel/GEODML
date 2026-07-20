# CHANGELOG

Date-stamped record of changes to this dataset.

## 2026-05-17 — LLM-execution-regime columns (`llm_backend`, `llm_precision`)

Added per-row metadata so consumers can distinguish records produced under
different inference stacks. Critical for the upcoming snippet-vs-RAG
analysis: the two arms were originally created under different precision
regimes (snippet → 4-bit nf4 on cluster, RAG → full-precision via HF
Inference API). Without these columns the cross is confounded.

### Schema additions

- Every JSONL record (`data/runs/*/phase2/keywords.jsonl`, `data/order_probe/*.jsonl`)
  now has `llm_parameters.{backend, precision}`.
- Every Stage C parquet (`data/main/full_experiment_data_<variant>.parquet`)
  now has top-level columns `llm_backend` (3 values) + `llm_precision` (5 values).
- See `PROVENANCE.md` § "2026-05-17 update" for the full taxonomy.

### Backfill

- `scripts/backfill_precision.py` patched 41,383 historical records across
  71 JSONLs (32,016 records → `local/4bit-nf4`; 9,367 records → `api/api-hf`).
  Idempotent and dry-run-friendly.
- Stage C parquets re-built by `interpretability/pipeline/merge.py` so the
  new top-level columns propagate.

### Code changes (in `~/Hamburg/GEODML_Analysis`)

- `interpretability/utils.py:LocalRanker` default flipped to `quantize=False`
  (bf16). `make_ranker(backend, model, *, precision)` honors `LOCAL_PRECISION`.
- `interpretability/pipeline/rerank.py:precision_label` — canonical mapping.
- `interpretability/pipeline/rerank.py:rank_one_keyword` — writes
  `llm_parameters.{backend, precision}` into every record.
- `interpretability/pipeline/order_probe.py` — same pass-through.
- `interpretability/pipeline/merge.py` — Stage C parquet schema extended.
- `interpretability/weight_analysis.py` — variant-aware (was silently
  defaulting to `biased`).
- `scripts/slurm/_common.sh:skip_if_at_max` — sbatch pre-flight short-circuit.
- `scripts/slurm/dispatch_all.sh` — `LOCAL_PRECISION` injected into every
  child job; `ORDER_PROBE_VARIANTS` swapped passage→rag.
- `scripts/finish_on_gpu.sh` — GPU equivalent of `finish_via_api.sh`.

### Status snapshot (this push)

| Variant | Records | Backend | Precision |
|---|---|---|---|
| `biased`      | 45,967 | local | 4bit-nf4 |
| `neutral`     | 52,256 | local | 4bit-nf4 |
| `biased_rag`  | 33,384 | api   | api-hf   |
| `neutral_rag` | 31,525 | api   | api-hf   |

### Next dataset revision

The JUWELS `scifi` bf16 redo (see
`docs/runbook-full-cycle-2026-05-17.md`) will replace snippet records with
`bf16-full`, producing the final 4-cell × full-precision matrix. ETA ~3
days. After that, DML results + a generated `RESULTS_SUMMARY.md` are
pushed back to this repo as the final revision.

## 2026-05-08 — RAG variants

Added two new prompt variants — `biased_rag` and `neutral_rag` — that perform
real query-conditional retrieval over chunked page bodies, replacing the
keyword-agnostic `_passage` (leading-body) condition. Implementation log:
`docs/work-log-2026-05-08.md`. Resume runbook: `docs/RESUME-RAG.md`.

### Added

- `data/rag_index/<engine>_top<pool>/` — 4 cells, **5.3 GB total**:
  - `full_passages.parquet` — 4 × ~9k–14k URL bodies (trafilatura, max_chars=50,000, vs the 800 cap for the archived `_passage` flow)
  - `chunks.parquet` — **906,322 chunks total** across 4 cells, char-recursive splitter (size=800, overlap=200, min_size=100)
  - `chunk_embeddings.npy` — float32 (n_chunks, 1536), L2-normalized, OpenAI `text-embedding-3-small`
  - `keywords.parquet`, `keyword_embeddings.npy` — same model, 4,011 unique keywords across cells
  - `retrieved_top3.parquet` — precomputed top-3 chunks per (keyword, url) for fast rerank resume
  - `meta.json` — embedding model name, dim, chunk config, build timestamp
- `data/runs/<cell>_<biased_rag|neutral_rag>/phase2/keywords.jsonl` — **16 new run dirs**, 6,919 records total, 0 fallbacks
- `data/order_probe/<cell>_<biased_rag|neutral_rag>_seed{42,123}.jsonl` — **PARTIAL**: 6 cells fully done (400 each), cell 7 partial (23 records), 25 of 32 cells pending after HF 402
- `data/main/full_experiment_data_{biased_rag,neutral_rag}.parquet` — Stage C merged tables: 33,384 rows (biased_rag, 744 kw) + 31,525 rows (neutral_rag, 615 kw)
- `data/dml_results/dml_results_long_{biased_rag,neutral_rag}.parquet` — Stage D, full PLR × {LGBM, RF} × 7 subsets × 10 treatments × 2 outcomes grid (~280 fits each, **in progress** at update time)

### Archived → `archives/passage_runs_20260508_143229/` (71 MB)

The `_passage` variants used the leading 800 chars of body text — same content
for every keyword on a given page, no retrieval. Superseded by `_rag` and
removed from the active pipeline. Kept for reproducibility.

- 16 `_passage` rerank dirs from `data/runs/`
- 32 `_passage` order_probe jsonls + done markers from `data/order_probe/`
- `data/main/full_experiment_data_{biased,neutral}_passage.parquet` × 2
- `data/dml_results/dml_results_long_{biased,neutral}_passage.parquet` × 2
- `data/passages/passages_*_max800.parquet` × 4 (leading-body cache; superseded by `data/rag_index/<cell>/full_passages.parquet` + `chunks.parquet`)

### Schema changes

- **Active variant set**: `biased`, `neutral`, **`biased_rag`** (NEW, replaces `_passage`), **`neutral_rag`** (NEW, replaces `_passage`)
- `axis_passage_mode` enum: `{snippet, passage, rag}` — `passage` rows now exist only in `archives/`

### Bugs fixed today

- `interpretability/pipeline/rerank.py:_serp_to_results` now skips rows with NaN `position` (ddg_top20 had 6 such rows; ddg_top50 had 97). Pre-existing bug, surfaced at MAX_KW=400.
- 90s per-call timeout added to `InferenceRanker` to prevent indefinite hangs on HF routing-wedge.
- `scripts/finish_via_api.sh` skips cells where `keywords.jsonl` already has ≥ MAX_KW lines (both Phase 2 and Phase 3) — makes resume cheap.

### Cost

| Stage | Wall | Cost |
|---|---|---|
| Build RAG index (one-time, 4 cells) | ~2 hr | ~$2.72 OpenAI |
| Phase 2 RAG rerank (16 cells × 400+ kw) | ~3 hr | ~$8 HF Inference |
| Phase 3 partial (6.5 / 32 cells) | ~1 hr | ~$3 HF Inference |
| **Today's total** | **~6 hr** | **~$14** |
| To finish (Phase 3 remaining 25 cells + DML × 2) | ~9 hr | ~$15 HF |

### Known incomplete

- **Phase 3 order_probe**: 25 of 32 RAG cells pending. HF 402 — credits depleted. Top up at https://huggingface.co/settings/billing then `bash scripts/finish_via_api.sh` (skip-guard handles resume).
- **Stage F interpretability** for RAG: untouched. Needs cluster GPU.

## 2026-05-07 — full-day session

### Added
- `data/features/features_<engine>_top<pool>.parquet` × 4 (built from upstream phase3 features at 91% treatment coverage; includes `treat_topical_comp` and Moz `domain_authority`)
- `data/features/dfs_keyword_confounders.parquet` (DataForSEO keyword-level: search_volume, cpc, competition, keyword_difficulty for 78–94% of keywords)
- `data/main/full_experiment_data_{biased,neutral,biased_passage,neutral_passage}.parquet` × 4 (variant-suffixed merged tables)
- `data/main/full_experiment_unified.parquet` — single 101,161-row × 48-col DML-ready table with explicit `axis_*` columns covering all 32 (engine × model × pool × prompt × passage_mode) cells
- `data/dml_results/dml_results_long_{biased,neutral}.parquet` — 280 fits each, full PLR × {LGBM, RF} × 7 subsets × 10 treatments × 2 outcomes grid, 0 errors
- `data/dml_results/dml_results_long_{biased,neutral}_passage.parquet` — partial (small-n smoke data, ~12 binary-prediction errors per variant; will be re-built when full passage rerank lands)
- `data/order_probe/order_probe_summary.parquet` regenerated to include passage smoke data — 98,928 rows
- `data/passages/passages_<engine>_top<pool>_max800.parquet` × 4 (trafilatura output cache)
- ~11,107 new HTML pages added to `data/runs/<cell>/phase2/html_cache/` via `scripts/scrape_missing_html.py` (Phase B HTTP scrape)
- ~17,186 cross-cell html_cache copies (Phase A: existing files copied into model-dirs that were missing them — the same URL was already cached for a different model in the same engine×pool)
- `interpretability/output/plots/figure_a_dml_{biased,neutral,delta}.png` — new DML coefficient figures
- `AUDIT.txt` — snapshot of the latest `scripts/audit_pipeline.py` output, refreshed after each major change

### Changed
- `data/runs/<cell>_<variant>/phase2/keywords.jsonl` and `.rerank_ckpt.json` — 16 passage cells re-run with the SHA-256 fix (each currently has 20 keywords from the smoke; full ~1009-keyword rerun pending)
- `data/order_probe/<cell>_<variant>_seed{42,123}.jsonl` — 32 passage order-probe cells rebuilt (smoke; full pending)
- `interpretability/output/plots/figure_{a,b,c}_*.png` regenerated against the new DML/Stage F outputs

### Removed (via `scripts/clean_passage_results.sh FORCE=1`)
- 16 polluted passage-variant `keywords.jsonl` + ckpts that had been generated before the SHA-256 fix (every passage in those prompts was empty)
- 32 polluted passage-variant order-probe jsonls + done markers
- 4 polluted passage caches in `data/passages/`
- 1 polluted aggregated `order_probe_summary.parquet` (re-built afterwards)
- Total: 138 items deleted

### Bugs fixed
- `interpretability/utils.py:url_to_html_filename` was MD5; upstream uses SHA-256 → every passage variant produced empty passages until this fix
- `scripts/build_main_table.py` did not prepend repo root to `sys.path` → ModuleNotFoundError when called as a script
- `scripts/build_unified_main.py` had a duplicate-axis-column edge case that produced 50-col instead of 48-col output → fixed via dedup

### Headline DML result computed today
```
POOLED · plr · lgbm · rank_delta
T7_source_earned: biased -1.61***  →  neutral -0.42***   Δ = +1.19
```
n_obs = 19,652 (biased) / 22,117 (neutral). Reproduces the GEODML paper's main finding via full DML controls.

## 2026-05-07 — 12:00 (initial build)

- Created the dataset tree with `scripts/build_dataset_mirror.sh`
- Imported all 8 html_cache directories from `~/Hamburg/GEODML/paperSizeExperiment/output/` (`duckduckgo_*` renamed to `ddg_*`)
- Imported cluster snapshot from HF `archives/geodml_data_.zip` and `archives/interpretability_.zip`
- Wrote initial `PROVENANCE.md`, `README.md`, `refresh.sh`, `push_to_hf.sh`
- First `COPY_HTML=1 COPY_DATA=1` attempt ran out of disk at 6.9 GB; user freed space and re-ran with `FORCE=1` to complete

## Earlier — pre-2026-05-07

The dataset did not exist before today. The components came from:
- Cluster JUWELS pipeline runs (rerank, order_probe, ablation, saliency, weights, partial probing) — uploaded to HF as `geodml-papersize` archive zips
- Upstream Mac experiment (`paperSizeExperiment/`) — html_caches and phase3 features that pre-date the cluster work
