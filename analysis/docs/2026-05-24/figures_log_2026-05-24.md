# Figures + late-day work log — 2026-05-24

Follow-up to `docs/session_log_2026-05-24.md`, covering the figure-generation
work that came after the analysis re-runs. Read together if you want the full
end-of-day picture.

## TL;DR (30 seconds)

- Wrote `scripts/make_figures.py` — eight EMNLP-ready figures, each as `.pdf` (vector, Type-42 fonts) and `.png` (300 DPI), all in `docs/figures/`.
- All figures draw from the fresh `~/geodml_data/` snapshot (full 1011-kw coverage; see `session_log_2026-05-24.md` for the data story).
- **T7_source_earned is excluded from every figure.** It is significant but a curated list-membership flag — descriptive, not a clean causal estimand. The paper headline is the content treatments under mutual control.
- Δrank is the display name for the `rank_delta` outcome everywhere on plots; the parquet column is unchanged.

## Eight figures

| # | File | What it shows |
|---|---|---|
| 1 | `fig1_content_headline.{pdf,png}` | Forest plot of 10 content treatments under mutually-controlled DML. Promoters (blue): T1a stats +1.022**, T5 topical +0.458***, T_llms_txt +0.130***, T2a Q-headings +0.128**. Demoters (red): T3 schema −0.132***, T6 freshness −0.056***. The headline result. |
| 2 | `fig2_coef_grid.{pdf,png}` | 9 content treatments × 4 variants annotated heatmap of single-treatment DML coefficients (Δrank). Diverging RdBu palette. Lets the reader scan every (treatment, variant) cell at once. |
| 3 | `fig3_rag_attenuation.{pdf,png}` | Per-treatment Δrank shift = (RAG − non-RAG) coefficient, two panels (biased / neutral). Confidence intervals from combined SE. RAG barely moves anything for content treatments — supports "RAG is not the lever" claim. |
| 4 | `fig4_marginal_vs_partial.{pdf,png}` | Paired horizontal bars: single-treatment DML (teal) vs mutually-controlled DML (navy) per content treatment. Shows the sign-flip in T1a stats (≈0 → +1.02) and T5 topical (−0.3 → +0.45) when other content treatments are properly controlled. Motivates the mutual-control framing. |
| 5 | `fig5_robust_survivors.{pdf,png}` | Joint-inference DML coefficient per content treatment with Romano–Wolf p-value annotations. Blue bars: 5 RW survivors at α=0.05 (T5 topical, T2a Q-headings, T_llms_txt, T6 freshness, T3 schema). Grey bars: not robust. The defensible "tight" finding. |
| 6 | `fig6_two_llm_agreement.{pdf,png}` | Llama vs Qwen coefficient scatter, content treatments only, Δrank. Pearson r = 0.72 over 36 (treatment × variant) cells. Markers by variant. Two-LLM corroboration without T7 dominating. |
| 7 | `fig7_jaccard_agreement.{pdf,png}` | URL-set Jaccard, two panels. (a) Two-LLM URL agreement per (keyword × engine × pool) by variant: μ = 0.712 / 0.866 / 0.723 / 0.869. (b) Snippet-vs-RAG agreement per (LLM × prompt): μ = 0.746 / 0.729 / 0.880 / 0.903. Biased prompts ⇒ ~15 pts less agreement, both cuts. |
| 8 | `fig8_pool_size_sensitivity.{pdf,png}` | Paired horizontal bars on Study-2 binary admission: pool=20 vs pool=50 for 5 content treatments. T4 LLM-coded citations strengthen with larger pool (−0.006* → −0.011**); T6 freshness weakens (−0.016*** → −0.012***); others pool-stable. |

## Suggested paper figure budget

A typical EMNLP paper carries 4–6 figures. From the eight above:

| Main paper | Use | Why |
|---|---|---|
| Fig 1 | §3 headline | The single most defensible chart in the set — content treatments under mutual control. |
| Fig 4 | §3 methodology | Justifies *why* we use mutually-controlled DML rather than naive single-treatment estimates. Sign-flips are visually striking. |
| Fig 5 | §4 robustness | Romano–Wolf survivors makes the robustness pitch in one image. |
| Fig 7 | §6 robustness | Independent of the DML pipeline, so reviewers who don't trust DML still see a real signal. |

| Appendix | Use | Why |
|---|---|---|
| Fig 2 | Appendix A: full per-variant breakdown | Lets reviewers verify every cell of the headline. |
| Fig 3 | Appendix B: RAG attenuation details | Supports the §5 RAG-mitigation claim with per-treatment evidence. |
| Fig 6 | Appendix C: two-LLM agreement | Backs up §8 robustness statement. |
| Fig 8 | Appendix D: pool-size sensitivity | Side discussion that doesn't fit the §3 headline. |

## Style choices

- `scripts/make_figures.py` sets a single `setup_style()` with Helvetica/Arial, 11pt body, 12pt titles, 10pt ticks, no top/right spines, no grid, 14pt `axes.titlepad` for breathing room.
- `pdf.fonttype = 42` produces editable Type-42 fonts; PDFs open in Illustrator if camera-ready tweaks are needed.
- `savefig.dpi = 300, savefig.bbox = "tight", savefig.pad_inches = 0.15` for paper-ready PNG previews.
- Palette: warm `#d6604d` (biased) / cool `#4393c3` (neutral) for prompt; blue `#2c7fb8` for promoters / red `#e34a33` for demoters in fig 1.
- Δrank used throughout instead of `rank_delta` (the data column name is unchanged; only display labels are renamed).
- "single-treatment DML" and "mutually-controlled DML" replace the internal "Spec A / Spec B" jargon.

## Iteration log

1. **Initial 6 figures** (with T7 as the headline). Worked structurally but two issues flagged:
   - T7 is descriptive, not a causal estimand — too easy for reviewers to dismiss.
   - Layout was crammed (small fonts, tight margins).
2. **Revision 1: T7 removed, larger sizes, fixed sign convention.** Dropped T7 from all displays. Bumped font sizes 9→11, axes.titlepad 8→14, added padding. Replaced the T7-only "cell heterogeneity" and "select-vs-rank" plots with the more paper-relevant `marginal_vs_partial` and `robust_survivors`. Fixed the colorbar/legend label that originally inverted the sign convention (POSITIVE Δrank = LLM promotes, NEGATIVE = LLM demotes — opposite of what the first draft said).
3. **Fig 7 added**: Jaccard URL agreement. Two panels (two-LLM, snippet-vs-RAG) computed directly from per-variant main parquets.
4. **Greek-letter rename**: `rank_delta` → `Δrank` in every visible label. Filter strings unchanged.
5. **"Spec A / Spec B" rename**: replaced with descriptive "single-treatment DML" and "mutually-controlled DML" in fig 4 title, legend, and filename (`fig4_marginal_vs_partial`).
6. **Fig 8 added**: pool-size sensitivity using Study-2 binary admission slices (POOL:20, POOL:50).

The pre-revision draft is preserved at `scripts/make_figures.py.bak-2026-05-24` (the May-12 CSV-driven version).

## Files produced

```
docs/figures/
├── fig1_content_headline.pdf            fig1_content_headline.png
├── fig2_coef_grid.pdf                   fig2_coef_grid.png
├── fig3_rag_attenuation.pdf             fig3_rag_attenuation.png
├── fig4_marginal_vs_partial.pdf         fig4_marginal_vs_partial.png
├── fig5_robust_survivors.pdf            fig5_robust_survivors.png
├── fig6_two_llm_agreement.pdf           fig6_two_llm_agreement.png
├── fig7_jaccard_agreement.pdf           fig7_jaccard_agreement.png
└── fig8_pool_size_sensitivity.pdf       fig8_pool_size_sensitivity.png

scripts/
├── make_figures.py                      (current — 8 figures, no T7, Δrank)
└── make_figures.py.bak-2026-05-24       (pre-revision May-12 CSV version)

docs/
├── session_log_2026-05-24.md            (data/analysis re-run; read first)
├── figures_log_2026-05-24.md            (THIS file)
├── dataset_gap_bridge_2026-05-24.md     (rag_coverage: 1011 full, 0 partial, 0 no_rag)
├── dml_selected_2026-05-24_fixed.md     (Study 2 binary Spec B, T7 −1.21*** biased)
├── rag_vs_nonrag_2026-05-24.md          (snippet vs RAG, per-domain analysis)
├── reanalysis_v2_2026-05-24.md          (paper-restructure recommendation)
├── full_paper_analysis_2026-05-24.md    (comprehensive Study 1 report)
└── archive_2026-05-23/                  (previous-day versions, for diff)
```

## How to re-run

```bash
cd ~/Hamburg/GEODML_Analysis

# All eight figures
python scripts/make_figures.py

# Subset
python scripts/make_figures.py --only fig1,fig4,fig7

# List figure keys
python scripts/make_figures.py --list

# Open all PDFs
open docs/figures/*.pdf
```

The script reads from `~/geodml_data/data/{dml_results, main}/*.parquet`. If you re-pull from HF or run a new analysis, just re-execute — figures pick up the latest data automatically.

## Numbers worth quoting in the paper

**Mutually-controlled DML on Δrank** (`fig1_content_headline`):

| Treatment | coef | p |
|---|---|---|
| T1a stats present | +1.022 | <0.01 |
| T5 topical comp. | +0.458 | <0.001 |
| T_llms_txt | +0.130 | <0.001 |
| T2a Q-headings | +0.128 | <0.01 |
| T6 freshness | −0.056 | <0.001 |
| T3 schema | −0.132 | <0.001 |

**Romano–Wolf survivors at α=0.05** (`fig5_robust_survivors`):
T5 (RW p=0.000), T2a (RW p=0.006), T_llms_txt (RW p=0.018), T6 (RW p=0.000), T3 (RW p=0.040).

**Two-LLM URL Jaccard by variant** (`fig7_jaccard_agreement`):
biased 0.712, neutral 0.866, biased_rag 0.723, neutral_rag 0.869. ≈15-point gap.

**Pool-size sensitivity** (`fig8_pool_size_sensitivity`):
T4_llm strengthens (−0.006* → −0.011**), T6_freshness weakens (−0.016*** → −0.012***) as pool grows from 20 to 50.

## Narrative shifts vs the May-23 picture

1. T7 dropped from the headline. Reframed as a brief descriptive note in §4 or appendix; not the lead.
2. Headline coefficients are now under mutual control (Spec B), which gives sharper, less-confounded estimates. T1a stats and T5 topical move from ≈0 to large positive when other treatments are controlled — a genuine multicollinearity correction.
3. RAG attenuation revised down: ~10–17% reduction in T7 magnitudes, marginally significant on rank-position outcomes, not significant on binary admission. Phrased as "marginal attenuation" rather than "RAG-resistant" or "RAG mitigates".
4. Pool size matters in a small but systematic way: larger pools shift the LLM toward citation signals and away from surface freshness cues.

## What's NOT in any figure (intentional)

- Stage F probing — JUPITER jobs were still running at end of session.
- Order-probe stability (Jaccard between original-vs-shuffled rerank). The summary parquet is only on JUPITER; would need an rsync. The two-LLM and snippet-vs-RAG Jaccards in fig 7 cover a similar robustness claim with on-disk data.
- T7 — see above.
- Stage F ablation / saliency / weights — these are mechanism evidence for §7, computed but rendered separately if needed.

## Deadline

EMNLP 2026 ARR submission: **2026-05-25** (tomorrow). Figures are paper-ready and stable; numbers in the table above are the recommended quotes.
