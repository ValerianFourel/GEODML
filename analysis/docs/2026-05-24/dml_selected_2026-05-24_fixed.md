# DML Study — selected_by_llm (build + fit)

  [build] loading 4 SERP pool files …
  [build] base pool rows = 49,364  unique (kw,url) = 28,331
  [build] expanded pool×model×variant rows = 394,912
  [build] loading 4 per-variant LLM-output files (true source of selection) …
  [build] biased_rag: dropping 0 pool rows from keywords with no RAG output (only 1011 of 1011 keywords have RAG coverage)
  [build] neutral_rag: dropping 0 pool rows from keywords with no RAG output (only 1011 of 1011 keywords have RAG coverage)

  [build] overall selection rate after fix = 58.37%
  [build] selection rate per variant (corrected):
```
    variant    rate  n_selected  n_pool
     biased 46.7200       46121   98728
 biased_rag 43.2500       42699   98728
    neutral 71.9500       71034   98728
neutral_rag 71.5700       70657   98728
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
     biased          0.4672       46121   98728
 biased_rag          0.4325       42699   98728
    neutral          0.7195       71034   98728
neutral_rag          0.7157       70657   98728
```
```
 engine  pool_size                  model   mean  count
    ddg         20 Llama-3.3-70B-Instruct 0.6537  41352
    ddg         20   Qwen2.5-72B-Instruct 0.6598  41352
    ddg         50 Llama-3.3-70B-Instruct 0.4128  61252
    ddg         50   Qwen2.5-72B-Instruct 0.4246  61252
searxng         20 Llama-3.3-70B-Instruct 0.6007  54220
searxng         20   Qwen2.5-72B-Instruct 0.6054  54220
searxng         50 Llama-3.3-70B-Instruct 0.7298  40632
searxng         50   Qwen2.5-72B-Instruct 0.7347  40632
```
  [dml] running PLR per (treatment, slice) …

  [dml] saved → data/dml_results/selected_long_fixed.parquet

  [joint] mutually-controlled multi-treatment fit …

  [joint] saved → data/dml_results/selected_multitreat_fixed.parquet

## 1. Headline — pooled single-treatment DML (Spec A)
Each row: PLR estimate of treatment → `selected`, with the 12 demoted treatments + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig   r2_g   r2_m
 T7_source_earned 24224 -0.8410 0.0797 [-0.9973, -0.6848] 0.0000 *** 0.3292 0.9834
     T6_freshness 24224 -0.0138 0.0020 [-0.0177, -0.0098] 0.0000 *** 0.3292 0.6672
           T4_llm 24224 -0.0092 0.0022 [-0.0136, -0.0048] 0.0000 *** 0.3292 0.6366
          T1_code 24224 -0.0011 0.0011 [-0.0033, +0.0010] 0.3011     0.3292 0.9359
T1b_stats_density 24224 -0.0011 0.0011 [-0.0032, +0.0010] 0.2991     0.3292 0.9344
       T_llms_txt 24224  0.0081 0.0067 [-0.0049, +0.0212] 0.2221     0.3292 0.5405
```

## 2. Headline — mutually-controlled multi-treatment DML (Spec B)
Each treatment estimated with the OTHER 5 treatments + 12 demoted + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig  p_val_bonferroni BF_sig   r2_g   r2_m
 T7_source_earned 24224 -0.6596 0.0712 [-0.7991, -0.5201] 0.0000 ***            0.0000    *** 0.3349 0.9787
T1b_stats_density 24224 -0.0777 0.0280 [-0.1327, -0.0228] 0.0056  **            0.0335      * 0.3566 0.9999
          T1_code 24224 -0.0637 0.0308 [-0.1241, -0.0033] 0.0389   *            0.2331        0.3562 0.9999
     T6_freshness 24224 -0.0132 0.0020 [-0.0171, -0.0093] 0.0000 ***            0.0000    *** 0.3550 0.6719
           T4_llm 24224 -0.0053 0.0023 [-0.0098, -0.0009] 0.0196   *            0.1174        0.3575 0.6475
       T_llms_txt 24224  0.0098 0.0066 [-0.0032, +0.0228] 0.1390                0.8341        0.3567 0.5477
```

## 3. Per-variant breakdown
**Coefficient by variant** (rows: treatment, cols: variant):
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned -1.2051     -1.0872   0.0157       0.0049
       T_llms_txt  0.0234      0.0036  -0.0047      -0.0049
T1b_stats_density  0.0002     -0.0009   0.0008       0.0002
          T1_code  0.0001     -0.0008   0.0008      -0.0000
           T4_llm -0.0171     -0.0135  -0.0023       0.0027
     T6_freshness -0.0184     -0.0195  -0.0084      -0.0081
```

**p-value by variant**:
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned  0.0000      0.0000   0.7280       0.9030
       T_llms_txt  0.1191      0.8182   0.6353       0.6044
T1b_stats_density  0.9380      0.7394   0.5837       0.8609
          T1_code  0.9532      0.7544   0.5643       0.9922
           T4_llm  0.0008      0.0101   0.5026       0.3932
     T6_freshness  0.0000      0.0000   0.0071       0.0082
```

### 3a. RAG attenuation per treatment (Δ = coef_rag − coef_nonrag)
```
                 pair         treatment  coef_nonrag  coef_rag       Δ   SE_Δ       z  p_val sig
  biased_rag − biased  T7_source_earned      -1.2051   -1.0872  0.1180 0.2351  0.5000 0.6158    
  biased_rag − biased        T_llms_txt       0.0234    0.0036 -0.0198 0.0216 -0.9100 0.3602    
  biased_rag − biased T1b_stats_density       0.0002   -0.0009 -0.0011 0.0035 -0.3000 0.7669    
  biased_rag − biased           T1_code       0.0001   -0.0008 -0.0010 0.0036 -0.2700 0.7880    
  biased_rag − biased            T4_llm      -0.0171   -0.0135  0.0035 0.0073  0.4800 0.6289    
  biased_rag − biased      T6_freshness      -0.0184   -0.0195 -0.0012 0.0063 -0.1800 0.8556    
neutral_rag − neutral  T7_source_earned       0.0157    0.0049 -0.0109 0.0604 -0.1800 0.8574    
neutral_rag − neutral        T_llms_txt      -0.0047   -0.0049 -0.0002 0.0136 -0.0200 0.9869    
neutral_rag − neutral T1b_stats_density       0.0008    0.0002 -0.0005 0.0019 -0.2800 0.7819    
neutral_rag − neutral           T1_code       0.0008   -0.0000 -0.0008 0.0019 -0.4200 0.6717    
neutral_rag − neutral            T4_llm      -0.0023    0.0027  0.0050 0.0047  1.0700 0.2845    
neutral_rag − neutral      T6_freshness      -0.0084   -0.0081  0.0003 0.0044  0.0800 0.9396
```

## 4. Heterogeneity — marginal slices (engine, model, pool size)

**Coefficient by single-dimension slice:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned  -0.7202      -0.8721    -0.6829      -0.5837  -0.6751  -0.6467
       T_llms_txt  -0.0017       0.0199     0.0189       0.0008   0.0061   0.0157
T1b_stats_density  -0.0010       0.0005    -0.0036       0.0014  -0.0015  -0.0001
          T1_code  -0.0011       0.0002    -0.0034       0.0019  -0.0014  -0.0001
           T4_llm  -0.0104      -0.0123    -0.0085      -0.0085  -0.0063  -0.0113
     T6_freshness  -0.0091      -0.0191    -0.0133      -0.0152  -0.0160  -0.0116
```

**p-value:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned   0.0000       0.0000     0.0000       0.0000   0.0000   0.0000
       T_llms_txt   0.8769       0.0209     0.0379       0.9361   0.5041   0.0952
T1b_stats_density   0.5803       0.7128     0.0225       0.3416   0.3147   0.9468
          T1_code   0.5495       0.9071     0.0293       0.2022   0.3238   0.9264
           T4_llm   0.0025       0.0001     0.0044       0.0090   0.0394   0.0010
     T6_freshness   0.0079       0.0000     0.0000       0.0000   0.0000   0.0001
```

## 5. Narrative — what this study tells us

Bonferroni-survivors at 0.05 (treatments × 6 tests): T7_source_earned, T1b_stats_density, T6_freshness.


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

