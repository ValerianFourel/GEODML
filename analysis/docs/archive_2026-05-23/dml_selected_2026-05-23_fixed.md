# DML Study — selected_by_llm (build + fit)

  [build] loading 4 SERP pool files …
  [build] base pool rows = 49,364  unique (kw,url) = 28,331
  [build] expanded pool×model×variant rows = 394,912
  [build] loading 4 per-variant LLM-output files (true source of selection) …
  [build] biased_rag: dropping 25,708 pool rows from keywords with no RAG output (only 744 of 1011 keywords have RAG coverage)
  [build] neutral_rag: dropping 39,560 pool rows from keywords with no RAG output (only 615 of 1011 keywords have RAG coverage)

  [build] overall selection rate after fix = 38.92%
  [build] selection rate per variant (corrected):
```
    variant    rate  n_selected  n_pool
     biased 32.8200       32404   98728
 biased_rag 26.1900       19123   73020
    neutral 49.1600       48535   98728
neutral_rag 47.6900       28219   59168
```
  [build] attaching features …
  [build] rich features file: rows=65,203 cols=73
  [build] domain LUT covers 13,436 domains; NaN T7 in pool: 22.5%
  [build] URL LUT covers 25,481 unique URLs

  [build] feature coverage in expanded table (% non-null):

# DML Study — Binary outcome `selected_by_llm` (2026-05-23, FIXED RAG)

*New study triggered by the May-23 category-switch findings.* Re-runs DML with a different outcome: **was this candidate URL selected into the LLM's top-10 output (1) or rejected (0)**.

**This is the FIXED version.** The earlier run sourced the `selected` indicator from `full_experiment_unified.parquet`, which has a publish-pipeline bug — it contains only 2,938 of the actual ~64,909 RAG-output rows. This rerun reads the `selected` flag from the per-variant `full_experiment_data_{variant}.parquet` files (164 k rows total, full coverage). RAG variants are also restricted to the keywords that actually have RAG output (744 of 1011 for biased_rag, 615 of 1011 for neutral_rag), so the unselected rows represent real LLM rejections, not RAG-retrieval failures.

**Sample frame**: the SERP candidate pool. Each row is (keyword × URL × search-engine × pool-size × LLM × prompt-variant × passage-mode). Total rows ≈ 395 k.

**Treatments (audit-clean only)**: T7_source_earned, T_llms_txt, T1b_stats_density, T1_code, T4_llm, T6_freshness. *Note: T5_topical_comp is dropped — it needs per-(kw,url) similarity recomputation that wasn't available for unselected pool rows.*

**X-set**: 12 demoted treatments (per the audit) + 18 reconstructable confounders (conf_*, dfs_*) + 5 cell dummies (engine, model, pool, prompt variant, RAG).

**Method**: Robinson-style partially-linear DML — cross-fitted (K=3) GradientBoosting nuisance models for both E[Y|X] and E[D|X], heteroskedastic-robust SE. Same identifying assumption as DoubleML PLR.


## 0. Sample composition
```
    variant  selection_rate  n_selected  n_pool
     biased          0.3282       32404   98728
 biased_rag          0.2619       19123   73020
    neutral          0.4916       48535   98728
neutral_rag          0.4769       28219   59168
```
```
 engine  pool_size                  model   mean  count
    ddg         20 Llama-3.3-70B-Instruct 0.2092  35116
    ddg         20   Qwen2.5-72B-Instruct 0.1818  35116
    ddg         50 Llama-3.3-70B-Instruct 0.2439  51169
    ddg         50   Qwen2.5-72B-Instruct 0.2456  51169
searxng         20 Llama-3.3-70B-Instruct 0.5610  44283
searxng         20   Qwen2.5-72B-Instruct 0.5038  44283
searxng         50 Llama-3.3-70B-Instruct 0.6099  34254
searxng         50   Qwen2.5-72B-Instruct 0.6264  34254
```
  [dml] running PLR per (treatment, slice) …

  [dml] saved → data/dml_results/selected_long_fixed.parquet

  [joint] mutually-controlled multi-treatment fit …

  [joint] saved → data/dml_results/selected_multitreat_fixed.parquet

## 1. Headline — pooled single-treatment DML (Spec A)
Each row: PLR estimate of treatment → `selected`, with the 12 demoted treatments + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig   r2_g   r2_m
 T7_source_earned 20452 -0.3977 0.0636 [-0.5224, -0.2731] 0.0000 *** 0.3392 0.9778
     T6_freshness 20452 -0.0133 0.0027 [-0.0185, -0.0081] 0.0000 *** 0.3392 0.6653
           T4_llm 20452 -0.0116 0.0027 [-0.0169, -0.0064] 0.0000 *** 0.3392 0.6065
          T1_code 20452  0.0001 0.0014 [-0.0028, +0.0029] 0.9684     0.3392 0.9259
T1b_stats_density 20452  0.0003 0.0014 [-0.0025, +0.0031] 0.8239     0.3392 0.9255
       T_llms_txt 20452  0.0379 0.0086 [+0.0210, +0.0548] 0.0000 *** 0.3392 0.5459
```

## 2. Headline — mutually-controlled multi-treatment DML (Spec B)
Each treatment estimated with the OTHER 5 treatments + 12 demoted + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig  p_val_bonferroni BF_sig   r2_g   r2_m
 T7_source_earned 20452 -0.3900 0.0651 [-0.5176, -0.2625] 0.0000 ***            0.0000    *** 0.3412 0.9788
          T1_code 20452 -0.1038 0.0377 [-0.1777, -0.0300] 0.0059  **            0.0351      * 0.3495 0.9999
T1b_stats_density 20452 -0.0727 0.0363 [-0.1437, -0.0016] 0.0451   *            0.2705        0.3495 0.9999
     T6_freshness 20452 -0.0148 0.0026 [-0.0199, -0.0096] 0.0000 ***            0.0000    *** 0.3487 0.6650
           T4_llm 20452 -0.0097 0.0028 [-0.0151, -0.0043] 0.0004 ***            0.0027     ** 0.3493 0.6286
       T_llms_txt 20452  0.0389 0.0086 [+0.0220, +0.0558] 0.0000 ***            0.0000    *** 0.3489 0.5459
```

## 3. Per-variant breakdown
**Coefficient by variant** (rows: treatment, cols: variant):
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned -0.6551     -0.6637   0.0533       0.0768
       T_llms_txt  0.0262      0.0486   0.0045       0.0692
T1b_stats_density  0.0032      0.0025  -0.0008      -0.0003
          T1_code  0.0032      0.0025  -0.0007       0.0001
           T4_llm -0.0199     -0.0191  -0.0063      -0.0059
     T6_freshness -0.0235     -0.0035  -0.0116      -0.0063
```

**p-value by variant**:
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned  0.0000      0.0000   0.5221       0.4048
       T_llms_txt  0.0801      0.0138   0.7041       0.0012
T1b_stats_density  0.1907      0.4740   0.6801       0.9314
          T1_code  0.1978      0.4810   0.7003       0.9773
           T4_llm  0.0000      0.0043   0.1222       0.3681
     T6_freshness  0.0000      0.5790   0.0012       0.3334
```

### 3a. RAG attenuation per treatment (Δ = coef_rag − coef_nonrag)
```
                 pair         treatment  coef_nonrag  coef_rag       Δ   SE_Δ       z  p_val sig
  biased_rag − biased  T7_source_earned      -0.6551   -0.6637 -0.0085 0.2077 -0.0400 0.9672    
  biased_rag − biased        T_llms_txt       0.0262    0.0486  0.0224 0.0248  0.9000 0.3657    
  biased_rag − biased T1b_stats_density       0.0032    0.0025 -0.0007 0.0043 -0.1600 0.8702    
  biased_rag − biased           T1_code       0.0032    0.0025 -0.0007 0.0043 -0.1600 0.8692    
  biased_rag − biased            T4_llm      -0.0199   -0.0191  0.0008 0.0082  0.1000 0.9181    
  biased_rag − biased      T6_freshness      -0.0235   -0.0035  0.0200 0.0077  2.6100 0.0092  **
neutral_rag − neutral  T7_source_earned       0.0533    0.0768  0.0236 0.1242  0.1900 0.8496    
neutral_rag − neutral        T_llms_txt       0.0045    0.0692  0.0647 0.0245  2.6400 0.0083  **
neutral_rag − neutral T1b_stats_density      -0.0008   -0.0003  0.0005 0.0040  0.1100 0.9109    
neutral_rag − neutral           T1_code      -0.0007    0.0001  0.0008 0.0040  0.2000 0.8408    
neutral_rag − neutral            T4_llm      -0.0063   -0.0059  0.0004 0.0077  0.0500 0.9638    
neutral_rag − neutral      T6_freshness      -0.0116   -0.0063  0.0053 0.0074  0.7100 0.4759
```

## 4. Heterogeneity — marginal slices (engine, model, pool size)

**Coefficient by single-dimension slice:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned  -0.2187      -0.6721    -0.3910      -0.3888  -0.3348  -0.1542
       T_llms_txt   0.0068       0.0648     0.0371       0.0364   0.0430   0.0292
T1b_stats_density  -0.0020       0.0014    -0.0006       0.0011  -0.0018   0.0010
          T1_code  -0.0018       0.0016    -0.0004       0.0015  -0.0017   0.0012
           T4_llm  -0.0123      -0.0117    -0.0075      -0.0176  -0.0087  -0.0186
     T6_freshness  -0.0101      -0.0168    -0.0138      -0.0138  -0.0084  -0.0188
```

**p-value:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned   0.0068       0.0000     0.0000       0.0000   0.0000   0.1045
       T_llms_txt   0.6078       0.0000     0.0019       0.0026   0.0002   0.0189
T1b_stats_density   0.4089       0.4413     0.7517       0.5664   0.3467   0.6319
          T1_code   0.4405       0.3815     0.8278       0.4664   0.3616   0.5691
           T4_llm   0.0020       0.0030     0.0487       0.0000   0.0132   0.0000
     T6_freshness   0.0211       0.0000     0.0002       0.0002   0.0173   0.0000
```

## 5. Narrative — what this study tells us

Bonferroni-survivors at 0.05 (treatments × 6 tests): T7_source_earned, T1_code, T6_freshness, T4_llm, T_llms_txt.


The headline question — *does the LLM disproportionately admit or reject candidate URLs based on
domain class, content density, or freshness?* — is now answered on a clean binary outcome with
the entire SERP pool as the sample frame, so the estimands aren't contaminated by rank-conditional
selection.

**Read this together with the rank_delta study from earlier today:**

- If a treatment carries a *negative* effect on `rank_delta` (LLM promotes the doc upward) AND a
  *positive* effect on `selected` (LLM is more likely to admit it at all), then the LLM is
  unambiguously favouring that signal. Mention both numbers in the paper.

- If `T7_source_earned` carries a *negative* effect on `selected` here too, that strengthens the
  LLM-bias-against-earned-media story — the bias isn't only "rank lower if admitted", it's also
  "don't admit at all".

- A treatment that flips sign between the two outcomes is candidate for a Section 5 caveat:
  the LLM might bring documents *in* but then rank them at the bottom (or the other way around).

Use the per-variant table (§3) for RAG mitigation claims, and the cell table (§4) for the
heterogeneity paragraph.

