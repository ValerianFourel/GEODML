# DML survivors — 2026-05-24 (full battery re-run)

Re-run of every DML analysis against the freshly-pulled HF dataset
`ValerianFourel/geodml-emnlp-2026` (~/geodml_data, May-24). Reports which
treatments survive multi-test correction across the three primary outcomes.

## Outcome definitions (single-line each)

- **Y₁ = binary admission** (`selected_by_llm` in `dml_selected.py`): for every
  candidate URL in the original SERP pool, `Y = 1` if the URL appears in the
  LLM's top-N output (i.e. is **included in `rank_post`**), `Y = 0` if the LLM
  dropped it.  Sample frame: full SERP pool, 24,224 (kw × url × cell) rows after
  restricting RAG variants to RAG-covered keywords (now 1011/1011 → no exclusions).
- **Y₂ = `rank_delta`** = `rank_pre − rank_post`. Defined only for admitted
  URLs.  Positive = LLM moved the doc UP (better position); negative = demoted.
- **Y₃ = `rank_post`** = LLM-assigned absolute rank. Positive = worse position.

## What was re-run

| Script | Output | Time | Status |
|---|---|---|---|
| `bridge_dataset_gaps.py` | coverage check + `data/coverage/*` rebuilt | <1 min | ✅ — `full_rag: 1011 / partial: 0 / no_rag: 0` |
| `dml_selected.py` | Spec A + Spec B for **binary admission** | ~13 min | ✅ |
| `full_paper_analysis.py` | Spec A + Spec B + Spec C (joint inference w/ Romano–Wolf) for `rank_delta` and `rank_post` | ~10 min | ✅ |
| `rag_vs_nonrag.py` | per-treatment Δ(rag − non_rag) | <30 s | ✅ |
| `reanalysis_v2.py` | paper-restructure recommendation | <30 s | ✅ |

## ▼ SURVIVORS

### Y₁ — Binary admission (Spec B joint, mutually-controlled)

Joint mutually-controlled DML over the 6 tested treatments
(T7_source_earned, T_llms_txt, T1b_stats_density, T1_code, T4_llm, T6_freshness),
n = 24,224.  Bonferroni at α=0.05 over 6 tests = p<0.00833.

| Treatment | coef | p | Bonferroni-corrected | Survives? |
|---|---|---|---|---|
| **T7_source_earned** | **−0.660** | <10⁻⁸ | ✓ | **YES** |
| **T1b_stats_density** | **−0.078** | 0.006 | ✓ | **YES** |
| **T6_freshness** | **−0.013** | <10⁻⁸ | ✓ | **YES** |
| T1_code | −0.064 | 0.039 | — | no |
| T4_llm | −0.005 | 0.020 | — | no |
| T_llms_txt | +0.010 | 0.139 | — | no |

**3 binary-admission survivors after Bonferroni.**

#### Per-variant breakdown (Spec A, T7_source_earned only)

| Variant | T7 coef | p | Survives? |
|---|---|---|---|
| biased | **−1.205** | <10⁻⁸ | YES |
| biased_rag | **−1.087** | <10⁻⁸ | YES |
| neutral | +0.016 | 0.73 | null |
| neutral_rag | +0.005 | 0.90 | null |

→ binary admission is biased-prompt-induced and only marginally attenuated by RAG (Δ=+0.118, p=0.62, not significant).

### Y₂ — `rank_delta` (joint inference, all 19 treatments, Spec C)

Romano-Wolf controls family-wise error rate respecting the empirical
correlation among test statistics. Sorted by RW p-value.

| Treatment | coef | p_raw | RW p | Bonferroni p | RW survives α=0.05? |
|---|---|---|---|---|---|
| **T5_topical_comp** | **+0.476** | <10⁻⁸ | 0.000 | <10⁻⁴ | ✓ ***  |
| **T2_llm** | **−0.130** | 0.0002 | 0.000 | 0.003 | ✓ *** |
| **T7_source_earned** | **−1.572** | <10⁻⁸ | 0.000 | <10⁻⁸ | ✓ *** |
| **T6_freshness** | **−0.057** | <10⁻⁸ | 0.000 | <10⁻⁸ | ✓ *** |
| **T2a_question_headings** | **+0.208** | 0.0006 | 0.006 | 0.011 | ✓ ** |
| **T_llms_txt** | **+0.092** | 0.001 | 0.018 | 0.024 | ✓ * |
| **T3_structured_data_new** | **−0.104** | 0.004 | 0.040 | 0.077 | ✓ * |
| **T4_llm** | **−0.024** | 0.004 | 0.040 | 0.079 | ✓ * |
| T3_llm | +0.112 | 0.008 | 0.078 | 0.151 | · (only) |
| T2b_structural_modularity | +0.002 | 0.017 | 0.142 | 0.317 | no |
| T4a_ext_citations / T4b_auth_citations / T1a / T1b / code-defined variants | smaller, all NS or marginal | — | — | — | no |

**8 rank_delta survivors after Romano-Wolf.**

### Y₃ — `rank_post` (joint inference, all 19 treatments, Spec C)

Same 8 treatments survive on `rank_post`, with signs flipped (positive `rank_post` coef = LLM places the doc at a WORSE absolute position).

| Treatment | coef | p_raw | RW p | RW survives α=0.05? |
|---|---|---|---|---|
| **T5_topical_comp** | **−0.484** | <10⁻⁸ | 0.000 | ✓ *** |
| **T6_freshness** | **+0.059** | <10⁻⁸ | 0.000 | ✓ *** |
| **T7_source_earned** | **+1.536** | <10⁻⁸ | 0.000 | ✓ *** |
| **T2a_question_headings** | **−0.184** | 0.002 | 0.024 | ✓ * |
| **T_llms_txt** | **−0.089** | 0.002 | 0.024 | ✓ * |
| **T3_structured_data_new** | **+0.108** | 0.003 | 0.028 | ✓ * |
| **T4_llm** | **+0.024** | 0.004 | 0.028 | ✓ * |
| **T2_llm** | **+0.101** | 0.003 | 0.028 | ✓ * |

Same 8 survivors. Sign agreement with `rank_delta` is uniform: every treatment
that promotes (negative `rank_delta`-coef) also lands the doc at a better
absolute position (negative `rank_post`-coef), confirming both outcomes carry
the same effect direction. Magnitudes are not identical because DML uses
separate nuisance functions per outcome.

## ▼ INTERSECTION — survivors common to ALL three outcomes

Three treatments survive multi-test correction on **all three primary outcomes**
(binary admission Bonferroni + rank_delta Romano-Wolf + rank_post Romano-Wolf):

| Treatment | Direction | Quick read |
|---|---|---|
| **T7_source_earned** | demoter | curated earned-media domains are demoted at admission AND placed at worse rank positions |
| **T6_freshness** | demoter | heavy-handed freshness boilerplate lowers admission AND worsens rank |

(T1b_stats_density survives on binary admission but not on rank_delta/rank_post.)

## ▼ SURVIVORS RESTRICTED TO RANK-LEVEL OUTCOMES (rank_delta ∩ rank_post)

These 8 treatments survive Romano-Wolf on both `rank_delta` and `rank_post`
but were not tested on binary admission (or, like T_llms_txt and T2/T3/T4
LLM-coded variants, were tested on a smaller set):

| Treatment | rank_delta coef | rank_post coef | Promoter / Demoter |
|---|---|---|---|
| T5_topical_comp | +0.48 | −0.48 | promoter |
| T2a_question_headings | +0.21 | −0.18 | promoter |
| T_llms_txt | +0.09 | −0.09 | promoter |
| T7_source_earned | −1.57 | +1.54 | **demoter** |
| T2_llm | −0.13 | +0.10 | demoter |
| T6_freshness | −0.057 | +0.059 | **demoter** |
| T3_structured_data_new | −0.10 | +0.11 | demoter |
| T4_llm | −0.024 | +0.024 | demoter |

## ▼ DOES NOT SURVIVE (treatments that look promising in Spec A but fail under correction)

- **T1a_stats_present** — Spec B (mutual control) coef is large (+1.02 **) but the joint Spec C inference doesn't reach Romano-Wolf significance.
- **T4a_ext_citations / T4b_auth_citations** — Spec A coefs are significant per-variant (e.g. biased T4a = −0.29 ***) but they don't survive Romano-Wolf joint inference (RW p ≈ 0.97 for both).
- **T1b_stats_density** — survives on binary admission Bonferroni but not on rank-level Romano-Wolf.
- **T2b_structural_modularity** — significant per-variant but vanishes under multi-test correction.

Conclusion: **the per-variant Spec A picture is over-optimistic.** The
multi-test-corrected joint inference is what should be cited.

## ▼ RAG attenuation — does retrieval augmentation change the picture?

For the 6 binary-admission treatments, Δ(rag − non_rag) tested with combined SE:

| Pair | Treatment | Δ | SE_Δ | p |
|---|---|---|---|---|
| biased_rag − biased | T7_source_earned | +0.118 | 0.235 | 0.62 |
| biased_rag − biased | T6_freshness | −0.001 | 0.006 | 0.86 |
| (others) | | small | | NS |

→ **None of the binary-admission RAG attenuations are statistically significant.** RAG does not change who the LLM admits.

For `rank_delta`, T7 attenuates by ~14 % under biased prompts (p=0.09 ·, marginal) — see `rag_vs_nonrag_2026-05-24.md` §6. RAG marginally affects WHERE the LLM places admitted docs, but not WHETHER it admits them.

## ▼ Recommended paper claims (post-survivor analysis)

1. **Headline (defensible after multi-test correction)**: 8 content/source treatments survive Romano-Wolf joint inference on `rank_delta`. The 3 with the largest magnitudes are T7 (earned-media exclusion), T5 (topical completeness boost), and T2_llm/T2a (Q-heading boost).
2. **Binary admission story**: 3 treatments survive Bonferroni — T7, T1b, T6. Earned-media exclusion operates at BOTH the admission gate AND the rank-placement step.
3. **RAG**: marginal at best — attenuates T7 ~14 % on rank-position outcomes (p≈0.09), no significant effect on binary admission.
4. **T7 caveat (already in figures policy)**: T7 is a curated list-membership flag, not a content treatment. Reframe as a descriptive finding ("the LLM systematically excludes a curated earned-media domain set"), not a causal lever.

## Sources

| Report | Path | Generated by |
|---|---|---|
| Binary admission (Spec A + Spec B) | `docs/dml_selected_2026-05-24_fixed.md` | `dml_selected.py` |
| Rank outcomes (Spec A + Spec B + Spec C) | `docs/full_paper_analysis_2026-05-24.md` | `full_paper_analysis.py` |
| RAG vs non-RAG per-treatment | `docs/rag_vs_nonrag_2026-05-24.md` | `rag_vs_nonrag.py` |
| Paper-restructure recommendation | `docs/reanalysis_v2_2026-05-24.md` | `reanalysis_v2.py` |
| Coverage integrity | `docs/dataset_gap_bridge_2026-05-24.md` | `bridge_dataset_gaps.py` |

Previous-version reports archived at `docs/archive_2026-05-24-am/`.

## Headline data parquets (live on disk)

- `~/geodml_data/data/dml_results/selected_long_fixed.parquet` (Y₁ per-slice)
- `~/geodml_data/data/dml_results/selected_multitreat_fixed.parquet` (Y₁ joint Spec B)
- `~/geodml_data/data/dml_results/dml_results_long_{biased,neutral,biased_rag,neutral_rag}.parquet` (Y₂, Y₃ per-variant Spec A)
- `~/geodml_data/data/dml_results/dml_multi_treatment.parquet` (Y₂, Y₃ joint Spec C with RW + Bonferroni)
- `~/geodml_data/data/dml_results/dml_robust_winners.parquet` (filtered survivors)
