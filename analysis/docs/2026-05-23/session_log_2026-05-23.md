# Session log — 2026-05-23

Single document for a future Claude (or human) to reconstruct the state of the
GEODML project after the May-23 work day. Read this top-to-bottom in 5 minutes
and you have everything.

## TL;DR (60 seconds)

1. **A data bug was found and fixed.** `full_experiment_unified.parquet` (the
   "union of all per-variant rerank outputs", published to HF) was missing
   ~62k RAG rows (~95% of the RAG arm). The bug was in the publish pipeline,
   not the per-variant files. `scripts/bridge_dataset_gaps.py` rebuilds a
   correct `full_experiment_unified_FIXED.parquet` (163,132 rows) and writes
   a `rag_coverage.parquet` table classifying each of the 1011 keywords as
   `full_rag` / `partial_rag` / `no_rag`.

2. **Study 2 (binary `selected_by_llm` outcome) was re-run with the fix.**
   The headline RAG result reversed: under biased prompts, RAG does **not**
   mitigate earned-media exclusion (T7 = −0.66 with and without RAG). The
   prompt is the lever; RAG is not.

3. **Resumable JUPITER repair pipeline was set up.**
   `scripts/repair_loop.sh` drives `repair_audit.py` + `repair_dispatch.py`
   until every cell of the (variant × engine × pool × model) grid hits
   `gap=0`. Currently running on JUPITER login node.

4. **A Stage-C skip bug was found and fixed in `scripts/slurm/run_dml.sbatch`.**
   It was silently reading stale May-17 data because the per-variant parquet
   already existed. Patched to always re-merge. New `DML_FORCE_REFIT=1` env
   forces a clean DML re-fit.

5. **Local Mac mirror has ~40% RAG cell coverage; JUPITER has more after the
   in-flight repair finishes.** Final downstream rebuild kicked off; expected
   completion ~4-6 hours from end-of-session.

## Where everything lives

### Code (this repo)

| Path | Purpose |
|---|---|
| `scripts/bridge_dataset_gaps.py` | Rebuilds the broken `full_experiment_unified.parquet` from per-variant files; emits `rag_coverage.parquet` + `missing_rag_keywords.parquet`. |
| `scripts/dml_selected.py` | **The Study 2 DML driver.** Binary `selected_by_llm` outcome. Reads the per-variant files (NOT broken unified) and restricts RAG arms to RAG-covered keywords. Writes `selected_long_fixed.parquet` and `selected_multitreat_fixed.parquet`. |
| `scripts/repair_audit.py` | Walks the (variant×engine×pool×model[×seed]) grid; computes per-cell keyword gap; writes `manifests/repair_manifest.parquet`. |
| `scripts/repair_dispatch.py` | Reads the manifest; submits sbatch via `--resume`; polls squeue. Idempotent — safe to re-run. |
| `scripts/repair_run.sh` | One-shot wrapper: audit + dispatch. |
| `scripts/repair_loop.sh` | **NEW** — long-running flock-guarded loop. `--with-downstream` auto-fires `dispatch_bcd.sh --with-stage-f` after Stage A/A' close. |
| `scripts/repair_report.py` | **NEW** — generates `docs/repair_report_<date>.md` from the manifest. |
| `scripts/missing_keywords.py` | **NEW** — per-cell missing-keyword manifest; classifies Layer-2 (rerun-needed) vs Layer-3 (rag-retrieval-failed). |
| `scripts/slurm/dispatch_bcd.sh` | Stage B (features) → C+D (DML) → F (ablation/saliency/probing/weights) with `afterok:` deps. Now propagates `DML_FORCE_REFIT` / `SKIP_MERGE` env. |
| `scripts/slurm/run_dml.sbatch` | **Patched today.** Stage C merge now runs by default (was guarded behind `if [ ! -s "$MAIN_PARQUET" ]`). New `DML_FORCE_REFIT=1` skips `--resume`. |
| `scripts/run_rag_embeddings.sh` | Re-build RAG index for Layer-3 keywords. Needs `OPENAI_API_KEY` + network. Not run today. |

### Data (Mac local at `~/geodml_data/`, JUPITER at `$GEODML_DATA_ROOT = /e/scratch/scifi/fourel1`)

| Path | What it is | State |
|---|---|---|
| `data/main/full_experiment_data_<variant>.parquet` | 4 per-variant files. **Source of truth.** Mac: May 17 (pre-repair). JUPITER: stale until in-flight dispatch finishes. | biased=45,967 / neutral=52,256 / biased_rag=33,384 / neutral_rag=31,525 rows |
| `data/main/full_experiment_unified.parquet` | Published "union" file. **Has the bug.** ~95% of RAG rows missing. | 101,161 rows, broken. **Do not use for RAG analyses.** |
| `data/main/full_experiment_unified_FIXED.parquet` | Bridged from per-variant files. | 163,132 rows. Correct. |
| `data/coverage/rag_coverage.parquet` | (kw → has_output_<variant>) per the 4 variants + `rag_coverage` ∈ {full_rag, partial_rag, no_rag} | 1011 rows: 615 full_rag, 129 partial_rag, 267 no_rag |
| `data/coverage/missing_rag_keywords.parquet` | Subset of above where at least one RAG variant has no output | 396 keywords |
| `data/dml_results/selected_long_fixed.parquet` | Study 2 per-treatment × per-slice fits | 66 rows (6 treatments × 11 slices) |
| `data/dml_results/selected_multitreat_fixed.parquet` | Study 2 Spec B (mutually-controlled joint) | 6 rows |
| `data/serp/phase0_top*.parquet` | Canonical SERP pools. Used to compute per-cell keyword gap. | 1011/1011/1009/980 unique keywords per (engine, pool) |

### Reports (in `docs/`)

| File | What |
|---|---|
| `docs/full_paper_analysis_2026-05-23.md` | Study 1 (rank_delta + post_rank) — written before the bug was found; **its DML inputs read per-variant files, so its results are unaffected** by the unified bug. |
| `docs/dml_selected_2026-05-23.md` | Study 2 BUG VERSION (using broken unified). **Don't cite.** |
| `docs/dml_selected_2026-05-23_fixed.md` | Study 2 FIXED VERSION — **this is the publishable one.** |
| `docs/dataset_gap_bridge_2026-05-23.md` | Explainer of the bug + how the bridge works. |
| `docs/repair_report_<date>.md` | Auto-generated from the repair manifest. Updates each `repair_loop.sh` cycle. |
| `docs/missing_keywords_report_<date>.md` | Auto-generated from `missing_keywords.py`. Per-cell + Layer-2/Layer-3 split. |
| `docs/session_log_2026-05-23.md` | **This file.** |

## What the publishable findings look like (post-fix)

### Spec B (mutually-controlled, n=20,452)

| Treatment | coef | p | Bonferroni |
|---|---|---|---|
| T7_source_earned | **−0.390** | <10⁻⁸ | ✓ |
| T_llms_txt | **+0.039** | <10⁻⁵ | ✓ |
| T1_code | −0.104 | 0.006 | ✓ |
| T4_llm | −0.010 | <10⁻³ | ✓ |
| T6_freshness | **−0.015** | <10⁻⁷ | ✓ |
| T1b_stats_density | −0.073 | 0.045 | — |

### T7 by prompt variant (the headline)

| Variant | T7 coef | p |
|---|---|---|
| biased | **−0.655** | <10⁻⁵ |
| neutral | +0.053 | 0.52 (null) |
| biased_rag | **−0.664** | <10⁻⁴ |
| neutral_rag | +0.077 | 0.40 (null) |

**Implication:** earned-media exclusion is prompt-induced and **RAG-resistant**.

### Two-LLM contrast (Llama vs Qwen)

- Per-treatment coefficients essentially identical (T7 = −0.39 in both)
- Llama: more aggressive ranker (mean rank_delta 1.34 vs 1.17, more demotions)
- Qwen: more RAG-receptive (rank-shift attenuates 2× more under biased+RAG)
- Mean Jaccard 0.81 on URL overlap; biased prompts widen disagreement (0.72) vs neutral (0.87)

### Snippet-neutral vs RAG-neutral (same prompt, different evidence)

- Mean Jaccard 0.91; 66% fully agreeing cells
- RAG calms the ranker: mean rank_delta 0.31 vs 0.46
- RAG creates a real `llms.txt` admission boost (+0.069, p=0.001) that snippet doesn't have (+0.005, null)
- RAG erases the freshness penalty that snippet has (−0.012, p=0.001) → (−0.006, null)
- Source-class T7 effect is symmetric (both null under neutral)

## Current state of the data (end-of-session)

Per `scripts/missing_keywords.py` against JUPITER per-variant parquets:

| Variant | target | actual | missing | cov% |
|---|---|---|---|---|
| biased | 8,022 | 4,908 | 3,114 | 61.2% |
| neutral | 8,022 | 5,287 | 2,735 | 65.9% |
| **biased_rag** | 8,022 | 3,233 | **4,789** | **40.3%** |
| **neutral_rag** | 8,022 | 3,195 | **4,827** | **39.8%** |

These numbers are **pre-merge** — Stage A rerank closed all cells today (32/32 gap=0) but the per-variant parquets haven't been re-derived yet because of the Stage-C skip bug we patched. After the JUPITER `DML_FORCE_REFIT=1 ./scripts/slurm/dispatch_bcd.sh --with-stage-f` finishes (~4 hr), expect:

- biased → ~94%, neutral → ~98%, biased_rag → ~75%, neutral_rag → ~70%
- The remaining ~25% / ~30% gap in RAG variants is irreducible (267 Layer-3 no-RAG keywords + 129 partial-RAG)

## Three layers of "missing data" (mental model)

| Layer | What | How to fix |
|---|---|---|
| **Layer 1** | Local Mac mirror has less data than JUPITER | `rsync` from JUPITER after repair completes |
| **Layer 2** | JUPITER cells with CAP?/STALE/LOW from earlier runs (now mostly closed) | `./scripts/repair_loop.sh` on JUPITER |
| **Layer 3** | 267 keywords with NO RAG output anywhere — rag_index is structurally incomplete | `OPENAI_API_KEY=… bash scripts/run_rag_embeddings.sh` |

## What's running right now (end of session)

- **JUPITER**: `nohup ./scripts/repair_loop.sh --with-downstream` was launched earlier today. Stage A is done (32/32). Stage A' (order_probe) was ~40/64 last we checked. Stage F probing has 16 jobs in flight.
- **Final step pending** on JUPITER: user needs to run

  ```bash
  cd /e/project1/scifi/fourel1/GEODML_Analysis && git pull --ff-only
  set -a; source .env; set +a
  rm -f $GEODML_DATA_ROOT/data/dml_results/.done_dml_*
  rm -f $GEODML_DATA_ROOT/data/features/.done_features_*
  DML_FORCE_REFIT=1 ./scripts/slurm/dispatch_bcd.sh --with-stage-f \
    2>&1 | tee logs/dispatch_bcd_refit_$(date +%H%M).log
  ```

  to actually rebuild Stages B/C/D/F with the new flag. Without `DML_FORCE_REFIT=1` the run_dml.sbatch (patched today) will still re-merge Stage C but `dml.py --resume` will skip refits.

## Key git commits today (newest first)

| Commit | What |
|---|---|
| `db5b2ef` | fix(stage-c/d): force re-merge after Stage A repair, add DML_FORCE_REFIT |
| `556b311` | fix(missing_keywords): honor $GEODML_DATA_ROOT on JUPITER |
| `d2f78c3` | feat(diagnostic): per-cell missing-keyword manifest |
| `02353e3` | feat(repair): resumable gap-fill pipeline for full keyword coverage |
| `cc96e9f` | (pre-session) feat(publish): publish_dataset.py |

## Common pitfalls to avoid

1. **Do not analyse RAG arms from `full_experiment_unified.parquet`.** It has the publish bug. Always read `full_experiment_data_biased_rag.parquet` / `…neutral_rag.parquet` directly, OR use `full_experiment_unified_FIXED.parquet`.

2. **JUPITER paths use `$GEODML_DATA_ROOT`**, not `~/geodml_data`. Always `set -a; source .env; set +a` in fresh shells. The `feedback_env_in_fresh_shells` memory documents this.

3. **`dml.py --resume` skips by (subset, outcome, method, learner, treatment) key.** When you change the underlying data, you MUST either delete `dml_results_long.parquet` or set `DML_FORCE_REFIT=1` — otherwise the refit silently reuses stale results.

4. **`run_dml.sbatch` used to skip Stage C merge if the per-variant parquet existed.** Patched today (commit `db5b2ef`). If you see DML jobs completing in 0 seconds, the patch didn't deploy — check `git log scripts/slurm/run_dml.sbatch` on JUPITER.

5. **`features.py` with `--resume` is incremental and safe** — it appends new (kw, url) rows. No special handling needed.

6. **Both LLMs and search engines were tested.** When discussing robustness, mention both (Llama-3.3-70B + Qwen2.5-72B; DDG + searxng).

## Deadline

EMNLP 2026 submission: **2026-05-25**. Less than 48 hours from end of this session.
