# DML Study — selected_by_llm (build + fit)

  [build] loading 4 SERP pool files …
  [build] base pool rows = 49,364  unique (kw,url) = 28,331
  [build] expanded pool×model×variant rows = 394,912
  [build] loading unified file & marking selected rows …
  [build] overall selection rate = 21.14%
  [build] selection rate per variant:
```
    variant    rate  n_selected  n_pool
     biased 32.8200       32404   98728
 biased_rag  1.1300        1114   98728
    neutral 49.1600       48535   98728
neutral_rag  1.4700        1449   98728
```
  [build] attaching features …
  [build] rich features file: rows=65,203 cols=73
  [build] domain LUT covers 13,436 domains; NaN T7 in pool: 22.5%
  [build] URL LUT covers 25,481 unique URLs

  [build] feature coverage in expanded table (% non-null):

# DML Study — Binary outcome `selected_by_llm` (2026-05-23)

*New study triggered by the May-23 category-switch findings.* Re-runs DML with a different outcome: **was this candidate URL selected into the LLM's top-10 output (1) or rejected (0)**.

**Sample frame**: the SERP candidate pool. Each row is (keyword × URL × search-engine × pool-size × LLM × prompt-variant × passage-mode). Total rows ≈ 395 k.

**Treatments (audit-clean only)**: T7_source_earned, T_llms_txt, T1b_stats_density, T1_code, T4_llm, T6_freshness. *Note: T5_topical_comp is dropped — it needs per-(kw,url) similarity recomputation that wasn't available for unselected pool rows.*

**X-set**: 12 demoted treatments (per the audit) + 18 reconstructable confounders (conf_*, dfs_*) + 5 cell dummies (engine, model, pool, prompt variant, RAG).

**Method**: Robinson-style partially-linear DML — cross-fitted (K=3) GradientBoosting nuisance models for both E[Y|X] and E[D|X], heteroskedastic-robust SE. Same identifying assumption as DoubleML PLR.


## 0. Sample composition
```
    variant  selection_rate  n_selected  n_pool
     biased          0.3282       32404   98728
 biased_rag          0.0113        1114   98728
    neutral          0.4916       48535   98728
neutral_rag          0.0147        1449   98728
```
```
 engine  pool_size                  model   mean  count
    ddg         20 Llama-3.3-70B-Instruct 0.0367  41352
    ddg         20   Qwen2.5-72B-Instruct 0.0365  41352
    ddg         50 Llama-3.3-70B-Instruct 0.1362  61252
    ddg         50   Qwen2.5-72B-Instruct 0.1375  61252
searxng         20 Llama-3.3-70B-Instruct 0.3050  54220
searxng         20   Qwen2.5-72B-Instruct 0.3075  54220
searxng         50 Llama-3.3-70B-Instruct 0.3725  40632
searxng         50   Qwen2.5-72B-Instruct 0.3780  40632
```
  [dml] running PLR per (treatment, slice) …

  [dml] saved → data/dml_results/selected_long.parquet

  [joint] mutually-controlled multi-treatment fit …

  [joint] saved → data/dml_results/selected_multitreat.parquet

## 1. Headline — pooled single-treatment DML (Spec A)
Each row: PLR estimate of treatment → `selected`, with the 12 demoted treatments + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig   r2_g   r2_m
 T7_source_earned 24224 -0.3205 0.0438 [-0.4062, -0.2347] 0.0000 *** 0.6456 0.9834
     T6_freshness 24224 -0.0110 0.0016 [-0.0142, -0.0078] 0.0000 *** 0.6456 0.6672
           T4_llm 24224 -0.0055 0.0018 [-0.0089, -0.0020] 0.0018  ** 0.6456 0.6366
T1b_stats_density 24224 -0.0012 0.0009 [-0.0028, +0.0005] 0.1737     0.6456 0.9344
          T1_code 24224 -0.0010 0.0009 [-0.0027, +0.0006] 0.2244     0.6456 0.9359
       T_llms_txt 24224  0.0042 0.0055 [-0.0066, +0.0149] 0.4472     0.6456 0.5405
```

## 2. Headline — mutually-controlled multi-treatment DML (Spec B)
Each treatment estimated with the OTHER 5 treatments + 12 demoted + 18 confounders + 5 cell dummies in X.

```
        treatment     n    coef     se             95% CI  p_val sig  p_val_bonferroni BF_sig   r2_g   r2_m
 T7_source_earned 24224 -0.2339 0.0417 [-0.3155, -0.1522] 0.0000 ***            0.0000    *** 0.6447 0.9787
T1b_stats_density 24224 -0.0724 0.0224 [-0.1163, -0.0286] 0.0012  **            0.0073     ** 0.6512 0.9999
          T1_code 24224 -0.0462 0.0241 [-0.0934, +0.0010] 0.0550   ·            0.3298        0.6512 0.9999
     T6_freshness 24224 -0.0113 0.0016 [-0.0145, -0.0081] 0.0000 ***            0.0000    *** 0.6511 0.6719
           T4_llm 24224 -0.0038 0.0018 [-0.0073, -0.0003] 0.0323   *            0.1938        0.6509 0.6475
       T_llms_txt 24224  0.0051 0.0055 [-0.0057, +0.0159] 0.3542                1.0000        0.6508 0.5477
```

## 3. Per-variant breakdown
**Coefficient by variant** (rows: treatment, cols: variant):
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned -0.6551     -0.0227   0.0533      -0.0172
       T_llms_txt  0.0262     -0.0044   0.0045      -0.0047
T1b_stats_density  0.0032     -0.0015  -0.0008      -0.0011
          T1_code  0.0032     -0.0015  -0.0007      -0.0010
           T4_llm -0.0199      0.0009  -0.0063       0.0005
     T6_freshness -0.0235     -0.0018  -0.0116      -0.0007
```

**p-value by variant**:
```
        treatment  biased  biased_rag  neutral  neutral_rag
 T7_source_earned  0.0000      0.0000   0.5221       0.0000
       T_llms_txt  0.0801      0.3769   0.7041       0.2034
T1b_stats_density  0.1907      0.0001   0.6801       0.0016
          T1_code  0.1978      0.0002   0.7003       0.0032
           T4_llm  0.0000      0.4495   0.1222       0.5877
     T6_freshness  0.0000      0.0891   0.0012       0.5075
```

### 3a. RAG attenuation per treatment (Δ = coef_rag − coef_nonrag)
```
                 pair         treatment  coef_nonrag  coef_rag       Δ   SE_Δ       z  p_val sig
  biased_rag − biased  T7_source_earned      -0.6551   -0.0227  0.6324 0.1416  4.4700 0.0000 ***
  biased_rag − biased        T_llms_txt       0.0262   -0.0044 -0.0306 0.0158 -1.9400 0.0525   ·
  biased_rag − biased T1b_stats_density       0.0032   -0.0015 -0.0047 0.0025 -1.9000 0.0572   ·
  biased_rag − biased           T1_code       0.0032   -0.0015 -0.0047 0.0025 -1.8800 0.0595   ·
  biased_rag − biased            T4_llm      -0.0199    0.0009  0.0208 0.0049  4.2500 0.0000 ***
  biased_rag − biased      T6_freshness      -0.0235   -0.0018  0.0217 0.0046  4.6900 0.0000 ***
neutral_rag − neutral  T7_source_earned       0.0533   -0.0172 -0.0705 0.0833 -0.8500 0.3977    
neutral_rag − neutral        T_llms_txt       0.0045   -0.0047 -0.0093 0.0125 -0.7400 0.4581    
neutral_rag − neutral T1b_stats_density      -0.0008   -0.0011 -0.0003 0.0019 -0.1700 0.8644    
neutral_rag − neutral           T1_code      -0.0007   -0.0010 -0.0003 0.0019 -0.1800 0.8561    
neutral_rag − neutral            T4_llm      -0.0063    0.0005  0.0068 0.0042  1.6300 0.1030    
neutral_rag − neutral      T6_freshness      -0.0116   -0.0007  0.0109 0.0037  2.9300 0.0034  **
```

## 4. Heterogeneity — marginal slices (engine, model, pool size)

**Coefficient by single-dimension slice:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned  -0.0913      -0.4957    -0.2063      -0.2364  -0.2466  -0.2285
       T_llms_txt  -0.0118       0.0186     0.0093       0.0026   0.0059   0.0029
T1b_stats_density  -0.0014      -0.0002    -0.0019      -0.0001  -0.0020  -0.0001
          T1_code  -0.0017      -0.0004    -0.0019      -0.0001  -0.0021  -0.0002
           T4_llm  -0.0087      -0.0029    -0.0010      -0.0095  -0.0047  -0.0070
     T6_freshness  -0.0081      -0.0114    -0.0101      -0.0117  -0.0073  -0.0122
```

**p-value:**
```
        treatment  ENG:ddg  ENG:searxng  MOD:Llama  MOD:Qwen2.5  POOL:20  POOL:50
 T7_source_earned   0.0515       0.0000     0.0000       0.0000   0.0000   0.0005
       T_llms_txt   0.1734       0.0081     0.2232       0.7347   0.4020   0.7226
T1b_stats_density   0.3202       0.8261     0.1236       0.9236   0.0666   0.9242
          T1_code   0.2383       0.7067     0.1307       0.9378   0.0550   0.9043
           T4_llm   0.0002       0.2819     0.6845       0.0002   0.0351   0.0196
     T6_freshness   0.0037       0.0000     0.0000       0.0000   0.0005   0.0000
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

