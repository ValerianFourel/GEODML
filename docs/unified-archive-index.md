# GEODML — Unified Project Archive

One consolidated knowledge folder for the entire GEODML project (Feb–May 2026),
gathered from three sibling repos on 2026-07-12:

| Subfolder | Source repo | What it was |
|---|---|---|
| `01_pipeline_GEODML/` | `~/Hamburg/GEODML` (42 GB) | Original pipeline + first studies (Feb–Apr 2026): SearXNG SERP → LLM re-rank → HTML features → DML. |
| `02_dataset_geodml-dataset/` | `~/Hamburg/geodml-dataset` (55 GB) | Dataset-packaging repo: consolidates Mac + JUWELS + local sources into the HF dataset `ValerianFourel/geodml-papersize-full`. |
| `03_analysis_GEODML_Analysis/` | `~/Hamburg/GEODML_Analysis` (11 GB) | Analysis + interpretability repo for the EMNLP 2026 paper (deadline 2026-05-25): canonical DML, RAG-vs-snippet, ablation/saliency/probing, all dated work logs. |

This folder holds **all documentation, findings, key result tables, and figures**
(~18 MB). Heavy data (HTML caches, parquets, embeddings, venvs — ~100 GB) stays
in the source repos; see `DATA_POINTERS.md` for where everything lives.

---

## The research question

**Which on-page features cause an LLM to promote or demote a search result
relative to its original search-engine ranking?** — studied with Double Machine
Learning (Chernozhukov et al., PLR/IRM, 5-fold cross-fitting, LightGBM + RF
nuisances) over B2B SaaS keywords, re-ranked by Llama-3.3-70B-Instruct and
Qwen2.5-72B-Instruct (DeepSeek-R1 in early runs).

Outcomes: `pre_rank` (SERP), `post_rank` (LLM), `rank_delta = pre − post`
(positive = LLM promoted), and later binary `selected_by_llm` (admission into
the LLM's top-10).

---

## Project arc

### Phase 1 — Original studies (Feb–Mar 2026, `01_pipeline_GEODML/`)
- 50 keywords, 492 obs (small pool 20/10) + 996 obs (large pool 50/20).
- Flagship early finding: **T2 question headings** promoted ~+1.2 ranks
  (p=0.009) in the small pool — but the effect **reverses in the large pool**
  (−2.9, p=0.002): the search engine already rewards FAQ pages, the LLM adds no
  uplift (`both_analysis/COMPARATIVE_FINDINGS.md` — pool size reverses conclusions).
- **T7 source_earned** emerges as the largest effect: earned-media pages
  (G2, Capterra, TechCrunch) demoted ~5 positions by Llama (p<0.001).
- Key docs: `FINDINGS.md`, `results_findings.md` (most complete),
  `EXPERIMENT_REGISTRY.md` (all 6 experiments), `RECOMMENDATIONS.md`,
  `slides_findings.html`.

### Phase 2 — Paper-size scale-up (Apr 2026, `01_pipeline_GEODML/paperSizeExperiment/`)
- 1,011 keywords, 32-cell design over 5 binary axes: engine (searxng/ddg) ×
  model (Llama-3.3-70B/Qwen2.5-72B) × pool (20/50) × prompt (biased/neutral) ×
  passage-mode (snippet/rag); constant top-10 re-rank. DataForSEO enrichment.
- Published as HF dataset (65,203-row main table, 570 DML fits) —
  `huggingface_bundle_README.md`.

### Phase 3 — Dataset consolidation (May 2026, `02_dataset_geodml-dataset/`)
- Three sources of truth (Mac upstream, JUWELS cluster snapshot, local May-07/08
  runs) merged into one push-ready HF tree; `PROVENANCE.md` maps every artifact.
- Discovered + fixed the SHA-256/MD5 filename bug that made every historical
  "passage" run functionally snippet-only; replaced passage arms with true
  query-conditional **RAG** arms (2026-05-08, `CHANGELOG.md`).
- Every record carries `llm_backend`/`llm_precision` (snippet arms = cluster
  4-bit, RAG arms = HF API full precision) — precision-confounding tracked
  explicitly; bf16-full reconciliation re-run on JUPITER Booster.

### Phase 4 — EMNLP analysis (Apr 27 – May 27 2026, `03_analysis_GEODML_Analysis/`)
Dated work logs in `docs/2026-*/` trace the whole run-up; the canonical final
state is **`docs/2026-05-24/`** (see below). Includes the JUWELS→JUPITER
runbooks, the resumable SLURM repair pipeline, the unified-parquet publish-bug
fix (missing ~62k RAG rows, 2026-05-23), and three mechanistic-interpretability
validations (input ablation, gradient×input saliency, per-layer probing).

---

## Headline results (canonical, 2026-05-24/25)

Data complete: 4 variants (biased, neutral, biased_rag, neutral_rag) × 1,011
keywords; unified table 431,856 rows.

**Study 1 — rank_delta (POOLED · PLR · LightGBM), T7_source_earned:**
biased **−2.242***, neutral −0.591***, biased_rag −1.931***, neutral_rag
−0.515*** (all p<1e-8). SEO-biased prompts demote earned-media domains ~2.2
ranks — 4× the neutral effect. RAG attenuates rank-level bias only ~14–17%.

**Study 2 — binary admission (`selected_by_llm`), Spec B joint (n=24,224):**
Bonferroni survivors: T7_source_earned −0.660, T1b_stats_density −0.078,
T6_freshness −0.013. Per-variant T7: biased −1.205***, biased_rag −1.087***,
neutral ≈ 0 (null).

**RAG-vs-snippet:** the earlier "RAG is resistant" claim (May-23) was a
sparse-data artifact. Revised: RAG attenuates rank-level bias modestly but does
**not** significantly attenuate binary-selection bias — *"the dominant lever is
the prompt, not the evidence."*

**Paper restructure:** T7 reclassified from treatment to **descriptive control**
(list-membership flag, AUC 0.92); excluded from the DML treatment set and all
figures. Canonical set: 7 treatments + 28 confounders (sensitivity variant:
llms.txt moved to confounders → 6 + 29). Romano–Wolf survivors: T7,
T6_freshness, T5_topical_comp, T_llms_txt, T2a_question_headings.

Start here:
- `03_analysis_GEODML_Analysis/docs/2026-05-24/session_log_2026-05-24.md` — master current-state doc
- `03_analysis_GEODML_Analysis/docs/2026-05-24/full_paper_analysis_2026-05-24.md` — full Study 1 + audits
- `03_analysis_GEODML_Analysis/docs/2026-05-24/canonical_treatments_and_confounders_2026-05-24.md` — locked variable set
- `03_analysis_GEODML_Analysis/docs/presentation_2026-05-27/slides.html` — final 30-slide deck
- `03_analysis_GEODML_Analysis/docs/2026-05-24/figures_canonical/` — EMNLP figures

---

## Map of this folder

```
01_pipeline_GEODML/
  CLAUDE.md FINDINGS.md results_findings.md EXPERIMENT_REGISTRY.md
  RECOMMENDATIONS.md EXPANSION.md PROPOSITION.md slides_findings.html keywords.txt
  data/                      geodml_dataset.csv (492 rows, canonical clean set) + data dictionary
  both_analysis/             small-vs-large pool comparison (COMPARATIVE_FINDINGS.md + figs + results)
  pipeline_results/          final 10-treat/16-conf DML runs (deepseek-r1, llama3.3-70b) + validation report
  earned_media_results/      dedicated T7 DML (both models)
  experiment_diagnostics/    test/ test_diff/ test_full/ test_full_rf/ result tables + diagnostic plots
  50_larger_figures/         large-pool publication figures (fig1–fig9)
  paperSizeExperiment/       README, audit, presentation, dml_robust_winners + manifests
  huggingface_bundle_README.md   HF dataset card (published)

02_dataset_geodml-dataset/
  README.md PROVENANCE.md CHANGELOG.md RESULTS_SUMMARY.md AUDIT.txt refresh.sh push_to_hf.sh

03_analysis_GEODML_Analysis/
  README.md
  docs/2026-04-27 … 2026-05-24/   dated work logs, runbooks, analyses (May-24 = canonical)
  docs/presentation_2026-05-27/   final slide deck
  docs/archive_*/                 superseded May-23 / May-24-am versions (provenance only)
  figures/                        legacy fig1–fig4 (PDF+PNG)
  manifests/                      repair_manifest.parquet
  papers_index/                   filename map + URLs of the 46 reference PDFs

DATA_POINTERS.md                  where all heavy data lives (do-not-copy index)
```

**Superseded material** (kept for provenance, do not cite): `03_…/docs/archive_2026-05-23/`,
`docs/archive_2026-05-24-am/`, `docs/2026-05-23/` analysis files (pre-bridge-fix numbers),
and `docs/2026-05-12/ALL-*.md` concatenated dumps. `02_…/AUDIT.txt` (May-08) is
older than `RESULTS_SUMMARY.md` (May-17).
