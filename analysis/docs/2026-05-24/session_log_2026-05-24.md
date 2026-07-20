# Session log — 2026-05-24

Single document for the EMNLP-eve reconciliation. Read top-to-bottom for the new picture
after re-running every Mac-side analysis against the freshly-pulled HF snapshot
`ValerianFourel/geodml-emnlp-2026` (5.4 GB, downloaded 12:27 local).

The May-23 reports (`docs/archive_2026-05-23/`) are preserved as the previous picture.

## TL;DR (60 seconds)

1. **Data is now complete.** All 4 variants × all 1011 keywords. The RAG coverage gap that
   dominated the May-23 reports is closed: bridge says `full_rag: 1011, partial_rag: 0, no_rag: 0`.
   Unified table is 431,856 rows (vs 163,132 May-23). RAG arms now ~103k rows each (vs ~31k yesterday).

2. **Headline coefficients got STRONGER**, not weaker, with more data.
   Study 1 T7 (Spec A, single-treatment + 25 confounders, rank_delta):
   biased −2.24 / neutral −0.59 / biased_rag −1.93 / neutral_rag −0.52, all p < 10⁻⁸.
   Compare May-23 reference: biased −1.61 / neutral −0.42 / biased_rag −1.27 / neutral_rag −0.50.
   The bf16-full reconciliation plus full RAG coverage moved the magnitude up ~40 % on biased
   and ~50 % on biased_rag; signs and significance unchanged.

3. **The "RAG-resistant" claim from May-23 was an artifact of sparse RAG data.** With full
   coverage, T7 attenuates 14 % on rank_delta (p = 0.09 ·) and 17 % on post_rank (p = 0.02 *)
   under biased prompts. Binary `selected_by_llm` (Spec B): T7 attenuates 9.8 % under biased+RAG
   but the difference is NOT significant (p = 0.62). New framing for the paper: *"earned-media
   exclusion is prompt-induced and only marginally attenuated by retrieval augmentation
   (~10–17 %); the dominant lever is the prompt, not the evidence."*

4. **Study 2 (binary selection) got tighter.** Joint Spec B Bonferroni survivors went from 6
   to 3: `T7_source_earned`, `T1b_stats_density`, `T6_freshness`. The shrinkage is a feature —
   the headline is now defensible.

5. **Reanalysis confirms the paper restructure**: T7 as control (Spec B framing), content
   PROMOTERS = T1a_stats_present, T5_topical_comp, T2a_question_headings, T_llms_txt;
   DEMOTERS = T6_freshness, T3_structured_data_new, T2_llm. Romano–Wolf survivors:
   T7, T6_freshness, T5_topical_comp, T_llms_txt, T2a_question_headings.

6. **JUPITER queue still draining**: 24 order_probe + 8 probing jobs running. These are
   nice-to-have for the paper (probing for §7 mechanism, order_probe for ddg_serp50 top-off).
   Neither is on the critical path — current data is sufficient to submit.

## What was re-run today (all from `~/geodml_data/` post-HF-pull)

| Script | Output | Size | Time |
|---|---|---|---|
| `bridge_dataset_gaps.py` | `data/main/full_experiment_unified.parquet` (rebuilt, 431,856 rows), `data/coverage/{rag_coverage,missing_rag_keywords}.parquet`, `docs/dataset_gap_bridge_2026-05-24.md` | 2.4 KB report | ~30 s |
| `dml_selected.py` | `data/dml_results/selected_long_fixed.parquet`, `selected_multitreat_fixed.parquet`, `docs/dml_selected_2026-05-24_fixed.md` | 10 KB report | 14.2 min |
| `rag_vs_nonrag.py` | `docs/rag_vs_nonrag_2026-05-24.md` | 22 KB | ~5 s |
| `reanalysis_v2.py` | `docs/reanalysis_v2_2026-05-24.md` | 28 KB | ~30 s |
| `full_paper_analysis.py` | `data/dml_results/{category_switch_audit,rag_cell_heterogeneity}.parquet`, `docs/full_paper_analysis_2026-05-24.md` | 64 KB report | ~12 min |

May-23 versions of the same reports are at `docs/archive_2026-05-23/`.

## Per-variant coverage (post-bridge, end-of-day)

| variant | rows | unique kw | unique URLs | RAG coverage |
|---|---|---|---|---|
| biased | 96,778 | 1,011 | 14,619 | (non-RAG) |
| neutral | 125,613 | 1,011 | 17,641 | (non-RAG) |
| biased_rag | 103,073 | 1,011 | 13,506 | **full** |
| neutral_rag | 106,392 | 1,011 | 17,496 | **full** |

`rag_coverage` parquet now classifies all 1011 keywords as `full_rag`; the 267 Layer-3
"no_rag" keywords from May-23 are gone (rag_index re-embed completed on JUPITER).

## Study 1 — rank_delta headline (Spec A, POOLED+plr+lgbm, T7_source_earned)

| Variant | coef (rank_delta) | se | n | coef (post_rank) | se |
|---|---|---|---|---|---|
| biased | **−2.242*** | 0.117 | 62,102 | **+1.437*** | 0.066 |
| neutral | −0.591*** | 0.037 | 101,671 | −0.324*** | 0.034 |
| biased_rag | **−1.931*** | 0.141 | 53,600 | **+1.192*** | 0.084 |
| neutral_rag | −0.515*** | 0.036 | 78,106 | −0.356*** | 0.039 |

Read: under biased prompts the LLM **demotes earned-media domains by ~2.24 ranks** relative
to their SERP position; absolute `post_rank` ends up ~1.44 ranks worse. Effects are 4× smaller
under neutral prompts. RAG attenuates ~14 % on rank_delta and ~17 % on post_rank but the bias
remains massive.

Full per-treatment headline in `docs/full_paper_analysis_2026-05-24.md` §3 and in the
auto-generated `docs/dml_headline.md`.

## Study 2 — binary `selected_by_llm` (Spec B joint, mutually-controlled)

Joint fit on n = 24,224 (6 treatments simultaneously controlled for one another + 25 confounders):

| Treatment | coef | p | Bonferroni at 0.05 (6 tests) |
|---|---|---|---|
| T7_source_earned | **−0.660** | <10⁻⁸ | ✓ |
| T_llms_txt | +0.010 | 0.14 | — |
| T1b_stats_density | −0.078 | 0.006 | ✓ |
| T1_code | (NS, joint) | n/a | — |
| T4_llm | (significant per-variant but not joint) | n/a | — |
| T6_freshness | **−0.014** | <10⁻⁷ | ✓ |

**Per-variant binary `selected_by_llm` T7 coefficients** (each variant fit alone with mutual controls):

| Variant | coef | p |
|---|---|---|
| biased | **−1.205*** | <10⁻⁸ |
| biased_rag | **−1.087*** | <10⁻⁸ |
| neutral | +0.016 | 0.73 (null) |
| neutral_rag | +0.005 | 0.90 (null) |

→ binary admission exhibits the same prompt-induced asymmetry as rank-position.

**RAG attenuation on binary selection** (revised vs May-23):

| Pair | Treatment | Δ | SE | p | Attenuation % |
|---|---|---|---|---|---|
| biased_rag − biased | T7_source_earned | +0.118 | 0.235 | 0.62 | 9.8 % (NS) |
| neutral_rag − neutral | T7_source_earned | −0.011 | 0.060 | 0.86 | — |

The selection-level bias is even MORE RAG-resistant than the rank-level bias (9.8 % vs 14–17 %),
and the difference is not statistically significant. Worth flagging in the paper:
**RAG attenuates how high the LLM places earned-media docs more than whether it includes them at all.**

## Reanalysis_v2 — paper-restructure conclusions (`reanalysis_v2_2026-05-24.md`)

Confirmed and unchanged from May-23:
- T7 is best treated as a CONTROL, not a treatment (AUC = 0.92 for T7 ≈ a coarse domain-quality marker).
- Content-treatment Spec B coefficients (rank_delta) with T7 in the X-set:
  - **Promoters**: T1a_stats_present +1.02 **, T5_topical_comp +0.46 ***, T2a_question_headings +0.13 **, T_llms_txt +0.13 ***
  - **Demoters**: T6_freshness −0.056 ***, T3_structured_data_new −0.13 ***, T2_llm −0.11 ***
- **Romano–Wolf survivors** (joint inference): T7, T6_freshness, T5_topical_comp, T_llms_txt, T2a_question_headings
- T7 itself remains the largest single coefficient (−1.77 ***), reframed as a DESCRIPTIVE finding.

## Snippet vs RAG — `rag_vs_nonrag_2026-05-24.md` §7

Per-domain RAG effect now has rich data. Some highlights:
- 3836 domains qualify for `biased vs biased_rag` (≥5 rows each); 54 are earned media.
- Mean Δ_post_rank: non-earned +0.18 (RAG-hurts), earned −0.10 (RAG-helps slightly).
- Top RAG-helped domains: scalable.capital, illumina-interactive.com, circlecare.app (all non-earned).
- Top RAG-hurt domains: sageintacct.com, easygenerator.com, invsify.com (all non-earned).

The earned-media class on average gains modestly from RAG (Δ_post_rank ≈ −0.10), but the
heterogeneity is huge. Worth a paragraph noting that RAG is a noisy intervention at the
domain level even when it shifts the headline coefficient by ~15 %.

## Cell heterogeneity (RAG attenuation) — `full_paper_analysis_2026-05-24.md` §6

Sharpest RAG attenuation cell: **searxng × Qwen2.5-72B × serp50**. Suggested for the
heterogeneity paragraph in §6. The `data/dml_results/rag_cell_heterogeneity.parquet` parquet
has the full per-cell deltas (8 cells × 4 variants × 2 outcomes × N treatments).

## What did NOT change vs May-23

- Two-LLM agreement (Llama + Qwen): per-treatment coefficients essentially identical, T7 = same direction & p < 0.001 in both. Llama still slightly more aggressive ranker; Qwen still slightly more RAG-receptive.
- Snippet-neutral vs RAG-neutral mean Jaccard ~ 0.91 on URL overlap; 66 % fully-agreeing cells. RAG calms ranker (rank_delta mean 0.31 vs 0.46).
- Stage F ablation / saliency / weights still complete on all 4 variants × 2 models.
- Stage F probing still missing — JUPITER jobs in flight.
- Order_probe Stage A' still showing biased prompts ~13 pts less order-stable than neutral (mean_jacc ≈ 0.69 vs 0.82). Caveat: 24 partial cells (~411/1011 kw on ddg_serp50) still topping off on JUPITER.

## Where everything lives (Mac)

| Path | What |
|---|---|
| `~/geodml_data/` | Fresh HF snapshot + restored Mac-only artifacts (Spec B parquets, coverage tables, FIXED unified) |
| `~/geodml_data.bak-20260524-1226/` | Pre-pull backup (May-23 state, 3.6 GB) |
| `docs/session_log_2026-05-24.md` | **This file.** |
| `docs/dataset_gap_bridge_2026-05-24.md` | Updated bridge explainer — now reports 0 gaps |
| `docs/dml_selected_2026-05-24_fixed.md` | Study 2 Spec B with full RAG coverage |
| `docs/rag_vs_nonrag_2026-05-24.md` | RAG vs non-RAG comparison (revised) |
| `docs/reanalysis_v2_2026-05-24.md` | Paper-restructure recommendation |
| `docs/full_paper_analysis_2026-05-24.md` | Comprehensive Study 1 + category-switch audit + RAG-cell heterogeneity |
| `docs/archive_2026-05-23/` | Previous-day versions for diff |

## What to write the paper with (recommended ordering)

1. **Headline (§3)**: Spec B content-treatment effects under prompt × RAG. Use the
   reanalysis_v2 PROMOTERS / DEMOTERS table.
2. **Earned-media demotion (§4)**: T7 as descriptive finding. Joint Spec B coefficient
   = −0.66 *** (binary) and −1.77 *** (rank_delta). Frame as systematic LLM-vs-curated-list bias.
3. **RAG mitigation (§5)**: REVISED — RAG modestly attenuates the rank-level bias (~14–17 %, p ≤ 0.09)
   but does NOT significantly attenuate the binary-selection bias (~10 %, p = 0.62). The
   dominant lever is the prompt, not the evidence.
4. **Heterogeneity (§6)**: Per-cell pivots from `rag_cell_heterogeneity.parquet`. Sharpest
   RAG attenuation: searxng × Qwen2.5 × serp50.
5. **Mechanism (§7)**: Stage F ablation + saliency + weights for biased vs neutral.
   Probing TBD pending JUPITER queue.
6. **Robustness (§8)**: Order-probe Jaccard table (biased 0.69 / neutral 0.82) + two-LLM
   per-treatment agreement (Llama vs Qwen identical signs and significance).

## JUPITER state (last known, 03:19 UTC)

- Stage A rerank: 32/32 ✓
- Stage A' order_probe: 40/64 done + 24 in flight (R)
- Stage B/C/D: complete with bf16-full headline
- Stage F: ablation 72/72, saliency 16/16, weights 8/8 ✓; probing 0/8 (8 R)
- Total queue: 32 R, 0 PD (the 80 DependencyNeverSatisfied jobs were cancelled)
- Re-push to HF will happen when probing lands and order_probe tops off; new
  `pipeline_status.py --no-refresh` is the watch command.

## Deadline

EMNLP 2026 ARR submission: **2026-05-25** (tomorrow). Current data is publishable;
remaining JUPITER work is bonus.
