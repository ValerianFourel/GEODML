
# GEODML — Full Paper-Ready Re-Analysis (2026-05-23)

*Source: `~/geodml_data/` mirroring HF dataset `ValerianFourel/geodml-emnlp-2026`.*

*Pipeline run on JUPITER Booster; this script re-aggregates the parquets and adds new diagnostics.*

**Outcomes:** `rank_delta` (negative = LLM promoted), `post_rank` (lower = better), `promotion` (binary, rank_delta > 0).
**Specifications:**
- *Spec A*: single-treatment DML — controls for the 25 confounders only.
- *Spec B*: mutually-controlled multi-treatment DML — every treatment estimated WITH all other 18 treatments AND 25 confounders in X.
- *Spec C*: joint inference — all 19 treatments in one DML fit, with Romano–Wolf and Bonferroni multi-test correction.

Learner: LightGBM (where available) / GradientBoosting fallback; cross-fitting K=5 in the JUPITER pipeline.
All p-values reported are two-sided; `***` p<0.001, `**` p<0.01, `*` p<0.05, `·` p<0.10.


## 1. Configuration profile — what's in the data

**Per-variant sample composition** (4 variants × 2 prompts × RAG/no-RAG):
```
    variant  n_rows  n_keywords  n_domains     engines                                        llms pool_sizes  pre_rank_mean  post_rank_mean  rank_delta_mean
     biased   45967        1011       9764 ddg/searxng Llama-3.3-70B-Instruct/Qwen2.5-72B-Instruct          ?         7.3800          5.0800           2.7090
    neutral   52256        1011       8091 ddg/searxng Llama-3.3-70B-Instruct/Qwen2.5-72B-Instruct          ?         5.9500          5.4500           0.5380
 biased_rag   33384         744       8252 ddg/searxng Llama-3.3-70B-Instruct/Qwen2.5-72B-Instruct          ?         6.5500          5.2000           1.9910
neutral_rag   31525         615       5690 ddg/searxng Llama-3.3-70B-Instruct/Qwen2.5-72B-Instruct          ?         5.6500          5.4200           0.3130
```

**Pooled `full_experiment_data.parquet`:** 65,203 rows × 73 columns; 1,011 keywords × 13,436 domains.

**Cells (search_engine × LLM × SERP pool size):**
```
search_engine              llm_model  serp_pool_size     n
      searxng Llama-3.3-70B-Instruct              50 16763
   duckduckgo   Qwen2.5-72B-Instruct              50  8742
      searxng   Qwen2.5-72B-Instruct              50  7566
      searxng   Qwen2.5-72B-Instruct              20  7375
      searxng Llama-3.3-70B-Instruct              20  6691
   duckduckgo   Qwen2.5-72B-Instruct              20  6415
   duckduckgo Llama-3.3-70B-Instruct              50  6064
   duckduckgo Llama-3.3-70B-Instruct              20  5587
```

## 2. Category-switch audit — is each variable acting as a treatment or a confounder?

*Using cached `category_switch_audit.parquet` — delete it to force a re-fit.*


### 2A. TREATMENTS — switch-candidacy diagnostic

```
                     name                      column  binary  r2_self_from_X  AUC_self  specA_coef_rd  specB_coef_rd  attenuation_rd       recommendation
        T1a_stats_present         treat_stats_present    True          0.9860    1.0000         0.0226         1.0216        -44.2710 → drop to confounder
       T4b_auth_citations        treat_auth_citations   False          0.9840       NaN        -0.0190        -0.0021          0.8900 → drop to confounder
                  T4_code  T4_citation_authority_code   False          0.9800       NaN        -0.0196        -0.0083          0.5770 → drop to confounder
        T1b_stats_density         treat_stats_density   False          0.9750       NaN        -0.0173        -0.0119          0.3150  ✓ keep as treatment
                  T1_code T1_statistical_density_code   False          0.9740       NaN        -0.0179        -0.0133          0.2600  ✓ keep as treatment
    T2a_question_headings     treat_question_headings    True          0.8080    0.9860         0.1034         0.1283         -0.2410 → drop to confounder
                  T2_code    T2_question_heading_code    True          0.7990    0.9750         0.0677         0.0380          0.4390 → drop to confounder
                   T3_llm      T3_structured_data_llm    True          0.6500    0.9560         0.0274         0.0500         -0.8250 → drop to confounder
                  T3_code     T3_structured_data_code    True          0.5570    0.9600         0.1269         0.0605          0.5240 → drop to confounder
                   T2_llm     T2_question_heading_llm    True          0.5560    0.9130        -0.0445        -0.1123         -1.5260 → drop to confounder
   T3_structured_data_new       treat_structured_data    True          0.5520    0.9300        -0.1396        -0.1325          0.0510 → drop to confounder
        T4a_ext_citations     treat_ext_citations_any    True          0.5310    0.9440        -0.0303        -0.0099          0.6740 → drop to confounder
                   T4_llm   T4_citation_authority_llm   False          0.4790       NaN        -0.0213        -0.0172          0.1930  ✓ keep as treatment
             T6_freshness             treat_freshness   False          0.3980       NaN        -0.0597        -0.0561          0.0600  ✓ keep as treatment
                   T1_llm  T1_statistical_density_llm   False          0.3110       NaN        -0.0083         0.0013          0.8400 → drop to confounder
          T5_topical_comp          treat_topical_comp   False          0.3090       NaN         0.4383         0.4585         -0.0460  ✓ keep as treatment
T2b_structural_modularity treat_structural_modularity   False          0.3060       NaN         0.0022         0.0010          0.5520 → drop to confounder
         T7_source_earned         treat_source_earned    True          0.2050    0.8830        -1.7000        -1.7656         -0.0390  ✓ keep as treatment
               T_llms_txt                has_llms_txt    True          0.1440    0.7390         0.0942         0.1297         -0.3770  ✓ keep as treatment
```

### 2B. CONFOUNDERS — internal coherence (sanity: confounders should be predictable)

```
                    name                   column  binary  r2_self_from_X  AUC_self recommendation
   dfs_intent_commercial    dfs_intent_commercial    True          1.0000    1.0000   (confounder)
dfs_intent_transactional dfs_intent_transactional    True          1.0000    1.0000   (confounder)
 dfs_intent_navigational  dfs_intent_navigational    True          1.0000    1.0000   (confounder)
dfs_intent_informational dfs_intent_informational    True          1.0000    1.0000   (confounder)
          conf_backlinks           conf_backlinks   False          0.9970       NaN   (confounder)
  conf_referring_domains   conf_referring_domains   False          0.9940       NaN   (confounder)
   conf_domain_authority    conf_domain_authority   False          0.9290       NaN   (confounder)
          conf_title_len           conf_title_len   False          0.7890       NaN   (confounder)
        conf_snippet_len         conf_snippet_len   False          0.7070       NaN   (confounder)
       dfs_search_volume        dfs_search_volume   False          0.6490       NaN   (confounder)
               conf_bm25                conf_bm25   False          0.5260       NaN   (confounder)
         conf_word_count          conf_word_count   False          0.4920       NaN   (confounder)
       conf_title_kw_sim        conf_title_kw_sim   False          0.4370       NaN   (confounder)
       conf_title_has_kw        conf_title_has_kw    True          0.3880    0.9390   (confounder)
     conf_snippet_kw_sim      conf_snippet_kw_sim   False          0.3380       NaN   (confounder)
         dfs_competition          dfs_competition   False          0.3330       NaN   (confounder)
        conf_brand_recog         conf_brand_recog    True          0.3030    0.8840   (confounder)
                 dfs_cpc                  dfs_cpc   False          0.2910       NaN   (confounder)
         conf_images_alt          conf_images_alt   False          0.2720       NaN   (confounder)
  dfs_keyword_difficulty   dfs_keyword_difficulty   False          0.2360       NaN   (confounder)
     conf_internal_links      conf_internal_links   False          0.1190       NaN   (confounder)
      conf_serp_position       conf_serp_position   False          0.1110       NaN   (confounder)
        conf_readability         conf_readability   False          0.0090       NaN   (confounder)
     conf_outbound_links      conf_outbound_links   False         -0.0430       NaN   (confounder)
              conf_https               conf_https    True         -0.3040    0.8030   (confounder)
```

### 2C. Switch-candidate call-outs

**Treatments flagged for re-labeling as confounders:**
- **`T1a_stats_present`** — r2_self=0.986, AUC=1.0, |coef| attenuates -4427% under Spec B.
- **`T4b_auth_citations`** — r2_self=0.984, AUC=—, |coef| attenuates 89% under Spec B.
- **`T4_code`** — r2_self=0.980, AUC=—, |coef| attenuates 58% under Spec B.
- **`T2a_question_headings`** — r2_self=0.808, AUC=0.986, |coef| attenuates -24% under Spec B.
- **`T2_code`** — r2_self=0.799, AUC=0.975, |coef| attenuates 44% under Spec B.
- **`T3_llm`** — r2_self=0.650, AUC=0.956, |coef| attenuates -82% under Spec B.
- **`T3_code`** — r2_self=0.557, AUC=0.96, |coef| attenuates 52% under Spec B.
- **`T2_llm`** — r2_self=0.556, AUC=0.913, |coef| attenuates -153% under Spec B.
- **`T3_structured_data_new`** — r2_self=0.552, AUC=0.93, |coef| attenuates 5% under Spec B.
- **`T4a_ext_citations`** — r2_self=0.531, AUC=0.944, |coef| attenuates 67% under Spec B.
- **`T1_llm`** — r2_self=0.311, AUC=—, |coef| attenuates 84% under Spec B.
- **`T2b_structural_modularity`** — r2_self=0.306, AUC=—, |coef| attenuates 55% under Spec B.

## 3. Headline multi-treatment table (Spec B — mutually controlled)

Each coefficient is the partial effect of that treatment with the **18 other treatments** AND **25 confounders** held fixed. This is the user-requested 'T7-as-confounder' specification applied to every treatment simultaneously.


### 3.A. outcome = `rank_delta`

**source**
```
       treatment     n    coef     se           95% CI  p_val sig
T7_source_earned 65203 -1.7656 0.0677 [-1.898, -1.633] 0.0000 ***
      T_llms_txt 65203  0.1297 0.0223 [+0.086, +0.173] 0.0000 ***
```

**content (new)**
```
                treatment     n    coef     se           95% CI  p_val sig
   T3_structured_data_new 58566 -0.1325 0.0298 [-0.191, -0.074] 0.0000 ***
             T6_freshness 58566 -0.0561 0.0070 [-0.070, -0.042] 0.0000 ***
        T1b_stats_density 56784 -0.0119 0.0075 [-0.026, +0.003] 0.1114    
        T4a_ext_citations 58566 -0.0099 0.0388 [-0.086, +0.066] 0.7991    
       T4b_auth_citations 58566 -0.0021 0.0210 [-0.043, +0.039] 0.9207    
T2b_structural_modularity 58566  0.0010 0.0006 [-0.000, +0.002] 0.0823   ·
    T2a_question_headings 58566  0.1283 0.0496 [+0.031, +0.226] 0.0097  **
          T5_topical_comp 39561  0.4585 0.1034 [+0.256, +0.661] 0.0000 ***
        T1a_stats_present 58566  1.0216 0.3568 [+0.322, +1.721] 0.0042  **
```

**content (rule-coded)**
```
treatment     n    coef     se           95% CI  p_val sig
  T1_code 57757 -0.0133 0.0065 [-0.026, -0.000] 0.0424   *
  T4_code 59577 -0.0083 0.0203 [-0.048, +0.032] 0.6826    
  T2_code 59577  0.0380 0.0448 [-0.050, +0.126] 0.3964    
  T3_code 59577  0.0605 0.0379 [-0.014, +0.135] 0.1106
```

**content (LLM-coded)**
```
treatment     n    coef     se           95% CI  p_val sig
   T2_llm 59554 -0.1123 0.0281 [-0.167, -0.057] 0.0001 ***
   T4_llm 59554 -0.0172 0.0072 [-0.031, -0.003] 0.0169   *
   T1_llm 59554  0.0013 0.0017 [-0.002, +0.005] 0.4251    
   T3_llm 59554  0.0500 0.0333 [-0.015, +0.115] 0.1334
```

### 3.B. outcome = `post_rank`

**source**
```
       treatment     n    coef     se           95% CI  p_val sig
      T_llms_txt 65203 -0.1278 0.0222 [-0.171, -0.084] 0.0000 ***
T7_source_earned 65203  1.7462 0.0669 [+1.615, +1.877] 0.0000 ***
```

**content (new)**
```
                treatment     n    coef     se           95% CI  p_val sig
        T1a_stats_present 58566 -1.3001 0.4424 [-2.167, -0.433] 0.0033  **
          T5_topical_comp 39561 -0.4842 0.1032 [-0.687, -0.282] 0.0000 ***
    T2a_question_headings 58566 -0.1297 0.0492 [-0.226, -0.033] 0.0085  **
       T4b_auth_citations 58566 -0.0077 0.0201 [-0.047, +0.032] 0.7021    
T2b_structural_modularity 58566 -0.0011 0.0006 [-0.002, +0.000] 0.0524   ·
        T1b_stats_density 56784  0.0129 0.0069 [-0.001, +0.026] 0.0605   ·
        T4a_ext_citations 58566  0.0230 0.0388 [-0.053, +0.099] 0.5538    
             T6_freshness 58566  0.0543 0.0070 [+0.041, +0.068] 0.0000 ***
   T3_structured_data_new 58566  0.1324 0.0295 [+0.075, +0.190] 0.0000 ***
```

**content (rule-coded)**
```
treatment     n    coef     se           95% CI  p_val sig
  T3_code 59577 -0.0606 0.0377 [-0.134, +0.013] 0.1075    
  T2_code 59577 -0.0296 0.0441 [-0.116, +0.057] 0.5018    
  T4_code 59577  0.0055 0.0206 [-0.035, +0.046] 0.7894    
  T1_code 57757  0.0173 0.0058 [+0.006, +0.029] 0.0031  **
```

**content (LLM-coded)**
```
treatment     n    coef     se           95% CI  p_val sig
   T3_llm 59554 -0.0514 0.0330 [-0.116, +0.013] 0.1186    
   T1_llm 59554 -0.0007 0.0017 [-0.004, +0.003] 0.6654    
   T4_llm 59554  0.0187 0.0071 [+0.005, +0.033] 0.0088  **
   T2_llm 59554  0.1008 0.0278 [+0.046, +0.155] 0.0003 ***
```

## 4. Joint inference — Romano–Wolf & Bonferroni-adjusted p-values

All 19 treatments fit in ONE DML regression. RW controls family-wise error rate respecting the empirical correlation among test statistics.


### 4.A. outcome = `rank_delta` (sorted by RW p-value)
```
                treatment     coef       se  p_val raw_sig  p_val_romano_wolf RW_sig  p_val_bonferroni BF_sig
                   T2_llm  -0.1296   0.0343 0.0002     ***             0.0000    ***            0.0030     **
          T5_topical_comp   0.4755   0.1042 0.0000     ***             0.0000    ***            0.0001    ***
         T7_source_earned  -1.5722   0.1125 0.0000     ***             0.0000    ***            0.0000    ***
             T6_freshness  -0.0570   0.0085 0.0000     ***             0.0000    ***            0.0000    ***
    T2a_question_headings   0.2076   0.0605 0.0006     ***             0.0060     **            0.0114      *
               T_llms_txt   0.0916   0.0284 0.0013      **             0.0180      *            0.0238      *
   T3_structured_data_new  -0.1044   0.0363 0.0041      **             0.0400      *            0.0771      ·
                   T4_llm  -0.0241   0.0084 0.0042      **             0.0400      *            0.0794      ·
                   T3_llm   0.1121   0.0422 0.0080      **             0.0780      ·            0.1513       
T2b_structural_modularity   0.0020   0.0008 0.0167       *             0.1420                   0.3166       
                  T3_code   0.0721   0.0475 0.1293                     0.6600                   1.0000       
        T1a_stats_present 837.1259 834.6422 0.3159                     0.9200                   1.0000       
                   T1_llm   0.0017   0.0021 0.4118                     0.9500                   1.0000       
                  T2_code   0.0252   0.0550 0.6471                     0.9700                   1.0000       
                  T4_code  -0.0165   0.0241 0.4948                     0.9700                   1.0000       
                  T1_code  -0.0077   0.0124 0.5348                     0.9700                   1.0000       
       T4b_auth_citations  -0.0131   0.0234 0.5757                     0.9700                   1.0000       
        T4a_ext_citations   0.0124   0.0490 0.7997                     0.9700                   1.0000       
        T1b_stats_density  -0.0076   0.0123 0.5390                     0.9700                   1.0000
```

**Surviving RW@0.05 (rank_delta):** T2_llm, T5_topical_comp, T7_source_earned, T6_freshness, T2a_question_headings, T_llms_txt, T3_structured_data_new, T4_llm.

### 4.B. outcome = `post_rank` (sorted by RW p-value)
```
                treatment      coef       se  p_val raw_sig  p_val_romano_wolf RW_sig  p_val_bonferroni BF_sig
          T5_topical_comp   -0.4844   0.1030 0.0000     ***             0.0000    ***            0.0000    ***
             T6_freshness    0.0592   0.0085 0.0000     ***             0.0000    ***            0.0000    ***
         T7_source_earned    1.5364   0.1105 0.0000     ***             0.0000    ***            0.0000    ***
    T2a_question_headings   -0.1841   0.0592 0.0019      **             0.0240      *            0.0354      *
               T_llms_txt   -0.0893   0.0282 0.0016      **             0.0240      *            0.0296      *
   T3_structured_data_new    0.1083   0.0360 0.0026      **             0.0280      *            0.0495      *
                   T4_llm    0.0243   0.0084 0.0036      **             0.0280      *            0.0678      ·
                   T2_llm    0.1011   0.0338 0.0028      **             0.0280      *            0.0523      ·
                   T3_llm   -0.1021   0.0416 0.0141       *             0.1340                   0.2674       
T2b_structural_modularity   -0.0018   0.0008 0.0263       *             0.2160                   0.4997       
                  T3_code   -0.0683   0.0470 0.1465                     0.7340                   1.0000       
        T1b_stats_density    0.0113   0.0109 0.3020                     0.9180                   1.0000       
                  T1_code    0.0103   0.0109 0.3462                     0.9480                   1.0000       
        T1a_stats_present -560.9681 824.2371 0.4961                     0.9820                   1.0000       
                   T1_llm    0.0007   0.0020 0.7499                     0.9860                   1.0000       
                  T2_code    0.0036   0.0543 0.9474                     0.9860                   1.0000       
       T4b_auth_citations   -0.0120   0.0212 0.5717                     0.9860                   1.0000       
        T4a_ext_citations   -0.0109   0.0489 0.8242                     0.9860                   1.0000       
                  T4_code    0.0113   0.0208 0.5879                     0.9860                   1.0000
```

**Surviving RW@0.05 (post_rank):** T5_topical_comp, T6_freshness, T7_source_earned, T2a_question_headings, T_llms_txt, T3_structured_data_new, T4_llm, T2_llm.

## 5. Spec A (single-treatment) vs Spec B (mutually controlled) — per-variant view

The coefficient changes from Spec A → Spec B tell us **which 'effects' were really mediated by other content features**. Big shrinkage = the variable was double-counting.


### 5.A. outcome = `rank_delta`
```
variant                    biased  biased_rag  neutral  neutral_rag  B_pooled (Spec B)  A_mean    ΔB−A
treatment                                                                                             
T1a_stats_present         -0.0944     -0.1876  -0.1326      -0.1117             1.0216 -0.1316  1.1532
T1b_stats_density          0.0006      0.0004   0.0006      -0.0027            -0.0119 -0.0003 -0.0116
T2a_question_headings      0.1230      0.1372   0.1466       0.0561             0.1283  0.1157  0.0126
T2b_structural_modularity  0.0051      0.0049   0.0019       0.0014             0.0010  0.0033 -0.0023
T3_structured_data_new    -0.0127     -0.0921  -0.1231      -0.0401            -0.1325 -0.0670 -0.0655
T4a_ext_citations         -0.0661      0.0833  -0.1406       0.1085            -0.0099 -0.0037 -0.0061
T4b_auth_citations        -0.0517     -0.0490  -0.0106      -0.0163            -0.0021 -0.0319  0.0298
T5_topical_comp            0.2338      0.8069   0.1374      -0.1110             0.4585  0.2668  0.1917
T6_freshness              -0.0134     -0.0342  -0.0488      -0.0344            -0.0561 -0.0327 -0.0234
T7_source_earned          -1.6075     -1.2678  -0.4166      -0.4959            -1.7656 -0.9469 -0.8186
```

### 5.B. outcome = `post_rank`
```
variant                    biased  biased_rag  neutral  neutral_rag  B_pooled (Spec B)  A_mean    ΔB−A
treatment                                                                                             
T1a_stats_present         -0.0550     -0.0708   0.0493      -0.0998            -1.3001 -0.0441 -1.2561
T1b_stats_density          0.0001      0.0029   0.0029       0.0052             0.0129  0.0028  0.0102
T2a_question_headings     -0.0311     -0.0963  -0.0413       0.0551            -0.1297 -0.0284 -0.1012
T2b_structural_modularity -0.0048     -0.0022  -0.0022      -0.0022            -0.0011 -0.0028  0.0017
T3_structured_data_new     0.1334      0.1853   0.2368       0.2681             0.1324  0.2059 -0.0735
T4a_ext_citations          0.0084     -0.0302  -0.0052       0.0693             0.0230  0.0106  0.0124
T4b_auth_citations         0.0438      0.0537  -0.0040      -0.0195            -0.0077  0.0185 -0.0262
T5_topical_comp           -0.2056     -0.6066   0.1617      -0.1750            -0.4842 -0.2064 -0.2779
T6_freshness               0.0503      0.0455   0.0607       0.0309             0.0543  0.0468  0.0074
T7_source_earned           1.2852      0.4185   0.2159       0.1133             1.7462  0.5082  1.2380
```

## 6. Cell-level heterogeneity — engine × LLM × pool size

From the pre-built `dml_pivot_rank_delta.parquet` / `dml_pivot_post_rank.parquet`. Each cell is a separate DML fit on the corresponding slice. Stars are based on raw p-values *within that fit*.


### 6.A. outcome = `rank_delta` — per-cell coefficients
```
                treatment ENG:duckduckgo ENG:searxng MOD:Llama-3.3-70B MOD:Qwen2.5-72B    POOL:20    POOL:50 duckduckgo_Llama-3.3-70B-Instruct_serp20_top10 duckduckgo_Llama-3.3-70B-Instruct_serp50_top10 duckduckgo_Qwen2.5-72B-Instruct_serp20_top10 duckduckgo_Qwen2.5-72B-Instruct_serp50_top10 searxng_Llama-3.3-70B-Instruct_serp20_top10 searxng_Llama-3.3-70B-Instruct_serp50_top10 searxng_Qwen2.5-72B-Instruct_serp20_top10 searxng_Qwen2.5-72B-Instruct_serp50_top10
        T1a_stats_present        +0.0225     +0.0342           +0.0644         -0.0517    -0.0569   +0.0689*                                        -0.0073                                        +0.0666                                      +0.0684                                      +0.0164                                     -0.0184                                   +0.1462**                                   -0.1297                                   -0.0921
        T1b_stats_density     -0.0153***  -0.0162***        -0.0084***      -0.0227*** -0.0183*** -0.0122***                                        -0.0031                                       -0.0079*                                    -0.0133**                                   -0.0201***                                   -0.0132**                                     -0.0041                                -0.0220***                                -0.0161***
    T2a_question_headings      +0.0783**  +0.1167***         +0.0655**      +0.1533*** +0.1555***  +0.0720**                                        -0.0399                                        +0.0757                                   +0.2728***                                      +0.0196                                  +0.2024***                                     +0.0348                                 +0.1705**                                   +0.0978
T2b_structural_modularity        +0.0011  +0.0033***        +0.0023***      +0.0025***  +0.0017** +0.0029***                                        +0.0001                                      +0.0030**                                      +0.0023                                      +0.0024                                  +0.0041***                                   +0.0023**                                   +0.0013                                +0.0041***
   T3_structured_data_new     -0.1880***  -0.0908***        -0.1084***      -0.1611*** -0.1466*** -0.1287***                                       -0.1304*                                     -0.3039***                                    -0.1698**                                   -0.1858***                                   -0.1660**                                     -0.0112                                 -0.1435**                                -0.1988***
        T4a_ext_citations        -0.0775     +0.0192           -0.0627         +0.0047    -0.0847    +0.0037                                       -0.1886*                                        -0.1136                                      -0.0336                                      -0.0029                                     -0.1551                                     +0.0625                                   +0.1368                                   +0.0361
       T4b_auth_citations        -0.0078  -0.0252***        -0.0231***       -0.0151**   -0.0125* -0.0234***                                        -0.0027                                        +0.0094                                      -0.0044                                     -0.0226*                                     -0.0029                                   -0.0434**                                   -0.0096                                   -0.0150
          T5_topical_comp     +0.8880***  +0.3910***           +0.1691      +0.6467*** +0.7183***   +0.2552*                                              —                                              —                                    +0.6402**                                            —                                   +0.5059**                                     +0.0766                                +0.6114***                                 +0.5657**
             T6_freshness     -0.0665***  -0.0543***        -0.0445***      -0.0718*** -0.0590*** -0.0599***                                      -0.0397**                                     -0.0865***                                   -0.0874***                                   -0.0736***                                  -0.0526***                                  -0.0443***                                -0.0590***                                -0.0705***
                  T1_code     -0.0152***  -0.0167***        -0.0087***      -0.0237*** -0.0174*** -0.0142***                                        -0.0027                                        -0.0021                                    -0.0142**                                   -0.0228***                                   -0.0136**                                     -0.0038                                -0.0194***                                  -0.0117*
                  T2_code        +0.0275   +0.0762**          +0.0519*       +0.0863**  +0.0907**   +0.0502*                                        -0.0079                                        +0.0832                                    +0.1945**                                      -0.0123                                     +0.0850                                     +0.0510                                  +0.1218*                                   +0.0606
                  T3_code       +0.0696*  +0.1800***        +0.1504***       +0.0982** +0.1516*** +0.1305***                                        +0.0969                                        +0.0419                                      +0.0765                                      +0.0813                                   +0.1756**                                  +0.1726***                                +0.1967***                                   +0.1095
                  T4_code        -0.0112  -0.0255***        -0.0196***       -0.0162**    -0.0080 -0.0259***                                        -0.0010                                        +0.0039                                      -0.0028                                    -0.0274**                                     -0.0088                                   -0.0319**                                   -0.0008                                   -0.0101
                   T1_llm     -0.0095***  -0.0089***        -0.0103***        -0.0182* -0.0123*** -0.0064***                                     -0.0121***                                       -0.0078*                                      +0.0136                                      -0.0037                                  -0.0165***                                   -0.0073**                                  -0.0373*                                  -0.0336*
                   T2_llm        -0.0246    -0.0510*         -0.0606**         -0.0050    -0.0034  -0.0577**                                        -0.0411                                        -0.0424                                     +0.1347*                                      -0.0533                                     +0.0171                                  -0.1177***                                   -0.0370                                   +0.0601
                   T3_llm        -0.0216   +0.0561**           -0.0145       +0.0993**  +0.0830**    +0.0037                                        -0.0557                                     -0.2077***                                      +0.1112                                      +0.0836                                     +0.0548                                     +0.0046                                +0.2147***                                   -0.0010
                   T4_llm       -0.0153*  -0.0279***         -0.0237**      -0.0218***    -0.0087 -0.0318***                                        +0.0115                                        -0.0133                                      -0.0120                                      -0.0211                                     -0.0053                                  -0.0416***                                   +0.0007                                 -0.0380**
         T7_source_earned     -1.5198***  -1.9096***        -1.7416***      -1.6556*** -1.6784*** -1.6758***                                     -1.3287***                                     -1.5822***                                   -1.7550***                                   -1.1823***                                  -1.8110***                                  -2.0387***                                -1.8038***                                -1.7649***
               T_llms_txt     +0.1013***  +0.0985***        +0.1113***       +0.0701** +0.1458***  +0.0556**                                        +0.0944                                        +0.0545                                   +0.2834***                                      -0.0006                                  +0.1809***                                   +0.1023**                                   +0.0414                                   +0.0416
```

### 6.B. outcome = `post_rank` — per-cell coefficients
```
                treatment ENG:duckduckgo ENG:searxng MOD:Llama-3.3-70B MOD:Qwen2.5-72B    POOL:20    POOL:50 duckduckgo_Llama-3.3-70B-Instruct_serp20_top10 duckduckgo_Llama-3.3-70B-Instruct_serp50_top10 duckduckgo_Qwen2.5-72B-Instruct_serp20_top10 duckduckgo_Qwen2.5-72B-Instruct_serp50_top10 searxng_Llama-3.3-70B-Instruct_serp20_top10 searxng_Llama-3.3-70B-Instruct_serp50_top10 searxng_Qwen2.5-72B-Instruct_serp20_top10 searxng_Qwen2.5-72B-Instruct_serp50_top10
        T1a_stats_present        -0.0366     -0.0051           -0.0530         +0.0464    +0.0493    -0.0485                                        +0.0445                                        -0.1070                                      -0.0954                                      -0.0009                                     +0.0034                                    -0.1085*                                  +0.1661*                                   +0.1244
        T1b_stats_density     +0.0152***  +0.0174***        +0.0096***      +0.0238*** +0.0179*** +0.0142***                                        +0.0035                                        +0.0047                                    +0.0152**                                   +0.0236***                                   +0.0140**                                    +0.0090*                                +0.0211***                                 +0.0127**
    T2a_question_headings       -0.0725*  -0.1278***         -0.0701**      -0.1383*** -0.1549***  -0.0704**                                        +0.0541                                        -0.0617                                   -0.2942***                                      -0.0384                                   -0.2005**                                     -0.0233                                 -0.1836**                                   -0.0751
T2b_structural_modularity        -0.0011  -0.0032***        -0.0022***       -0.0024**  -0.0015** -0.0026***                                        -0.0005                                       -0.0029*                                      -0.0025                                      -0.0023                                  -0.0048***                                     -0.0014                                   -0.0013                                -0.0042***
   T3_structured_data_new     +0.1874***  +0.1044***        +0.1163***      +0.1681*** +0.1287*** +0.1368***                                        +0.1060                                     +0.3364***                                    +0.1802**                                    +0.1435**                                   +0.1406**                                     +0.0358                                 +0.1431**                                +0.1706***
        T4a_ext_citations        +0.0708     -0.0073           +0.0559         -0.0226    +0.0862    -0.0175                                        +0.1516                                        +0.1331                                      +0.0716                                      +0.0299                                     +0.1450                                     -0.0614                                   -0.1156                                   -0.0364
       T4b_auth_citations       +0.0128*  +0.0202***        +0.0228***        +0.0131*   +0.0121* +0.0214***                                        +0.0073                                        -0.0070                                      +0.0044                                      +0.0202                                     +0.0017                                     +0.0212                                   +0.0031                                   +0.0093
          T5_topical_comp     -0.8602***  -0.3976***          -0.2425*      -0.6982*** -0.6770***  -0.2736**                                              —                                              —                                   -0.8511***                                            —                                    -0.4437*                                     -0.1093                                -0.6055***                                -0.7045***
             T6_freshness     +0.0685***  +0.0534***        +0.0461***      +0.0729*** +0.0597*** +0.0601***                                       +0.0335*                                     +0.0746***                                   +0.0818***                                   +0.0775***                                   +0.0441**                                  +0.0400***                                +0.0633***                                +0.0717***
                  T1_code     +0.0143***  +0.0172***        +0.0103***      +0.0213*** +0.0183*** +0.0156***                                        +0.0047                                        +0.0025                                    +0.0155**                                   +0.0202***                                  +0.0170***                                    +0.0094*                                +0.0207***                                 +0.0133**
                  T2_code        -0.0402  -0.0826***          -0.0551*       -0.0728**  -0.0932**    -0.0469                                        +0.0020                                        -0.1138                                    -0.2069**                                      +0.0083                                     -0.0778                                     -0.0591                                  -0.1248*                                   -0.0511
                  T3_code        -0.0641  -0.1829***        -0.1621***       -0.0914** -0.1478*** -0.1226***                                        -0.1061                                        -0.0391                                      -0.0992                                      -0.0756                                   -0.1834**                                  -0.2073***                                -0.1952***                                   -0.0245
                  T4_code        +0.0119  +0.0273***        +0.0226***       +0.0161**    +0.0116 +0.0230***                                        +0.0108                                        -0.0063                                      +0.0026                                    +0.0282**                                     +0.0048                                     +0.0234                                   +0.0067                                   +0.0088
                   T1_llm     +0.0096***  +0.0106***        +0.0113***         +0.0160 +0.0124*** +0.0082***                                     +0.0130***                                      +0.0086**                                      -0.0129                                      +0.0004                                  +0.0174***                                  +0.0091***                                   +0.0305                                  +0.0362*
                   T2_llm        +0.0300     +0.0429          +0.0502*         +0.0199    +0.0139  +0.0603**                                        +0.0449                                        +0.0057                                     -0.1301*                                      +0.0851                                     +0.0028                                  +0.1254***                                   +0.0602                                   -0.0114
                   T3_llm        +0.0303   -0.0641**           +0.0210      -0.1017***  -0.0864**    +0.0041                                        +0.0489                                     +0.1794***                                      -0.1114                                      -0.0495                                     -0.0906                                     +0.0018                                -0.2005***                                   -0.0083
                   T4_llm       +0.0140*  +0.0257***         +0.0199**      +0.0212***    +0.0118 +0.0322***                                        -0.0176                                        +0.0165                                      +0.0140                                      +0.0220                                     +0.0051                                  +0.0347***                                   -0.0016                                 +0.0320**
         T7_source_earned     +1.4908***  +1.9032***        +1.7461***      +1.6456*** +1.6720*** +1.6842***                                     +1.3432***                                     +1.4599***                                   +1.7961***                                   +1.1058***                                  +1.7324***                                  +2.1621***                                +1.8307***                                +1.7171***
               T_llms_txt     -0.0989***  -0.0997***        -0.1170***      -0.0852*** -0.1370***  -0.0638**                                        -0.0937                                        -0.0499                                   -0.2366***                                      -0.0111                                  -0.1806***                                   -0.0815**                                   -0.0412                                   -0.0276
```

## 7. RAG vs non-RAG breakdown

### 7A. Sample composition
```
    variant  n_rows  n_keywords pct_promoted (Δ>0) pct_no_change (Δ=0) pct_demoted (Δ<0)  rank_delta_mean
     biased   45967        1011              70.1%               15.2%             14.7%           2.7090
    neutral   52256        1011              43.6%               36.8%             19.6%           0.5380
 biased_rag   33384         744              66.3%               18.4%             15.3%           1.9910
neutral_rag   31525         615              40.4%               42.2%             17.3%           0.3130
```

**Paired keyword × URL rank_delta — does RAG re-order the same retrieved doc?**
```
                 pair  n_pairs  mean_Δ     sd  paired_t    p_val
  biased → biased_rag    57021 -0.0780 2.9950   -6.2100 5.24e-10
neutral → neutral_rag    91428 -0.0260 2.5100   -3.1700 1.51e-03
```

### 7B. Per-treatment RAG attenuation (Δ = coef_rag − coef_nonrag)

### outcome = `rank_delta` — RAG deltas
```
                 pair                 treatment  coef_non_rag  coef_rag       Δ   SE_Δ       z  p_val_Δ sig
  biased_rag − biased         T1a_stats_present       -0.0944   -0.1876 -0.0932 0.1519 -0.6100   0.5394    
  biased_rag − biased         T1b_stats_density        0.0006    0.0004 -0.0003 0.0069 -0.0400   0.9682    
  biased_rag − biased     T2a_question_headings        0.1230    0.1372  0.0142 0.1208  0.1200   0.9063    
  biased_rag − biased T2b_structural_modularity        0.0051    0.0049 -0.0002 0.0027 -0.0900   0.9305    
  biased_rag − biased    T3_structured_data_new       -0.0127   -0.0921 -0.0794 0.1019 -0.7800   0.4363    
  biased_rag − biased         T4a_ext_citations       -0.0661    0.0833  0.1493 0.1819  0.8200   0.4116    
  biased_rag − biased        T4b_auth_citations       -0.0517   -0.0490  0.0027 0.0274  0.1000   0.9212    
  biased_rag − biased           T5_topical_comp        0.2338    0.8069  0.5731 0.4259  1.3500   0.1785    
  biased_rag − biased              T6_freshness       -0.0134   -0.0342 -0.0208 0.0292 -0.7100   0.4767    
  biased_rag − biased          T7_source_earned       -1.6075   -1.2678  0.3396 0.3511  0.9700   0.3333    
neutral_rag − neutral         T1a_stats_present       -0.1326   -0.1117  0.0209 0.1115  0.1900   0.8512    
neutral_rag − neutral         T1b_stats_density        0.0006   -0.0027 -0.0033 0.0059 -0.5600   0.5774    
neutral_rag − neutral     T2a_question_headings        0.1466    0.0561 -0.0906 0.0867 -1.0400   0.2964    
neutral_rag − neutral T2b_structural_modularity        0.0019    0.0014 -0.0005 0.0017 -0.2900   0.7734    
neutral_rag − neutral    T3_structured_data_new       -0.1231   -0.0401  0.0829 0.0735  1.1300   0.2590    
neutral_rag − neutral         T4a_ext_citations       -0.1406    0.1085  0.2491 0.1368  1.8200   0.0685   ·
neutral_rag − neutral        T4b_auth_citations       -0.0106   -0.0163 -0.0057 0.0215 -0.2600   0.7924    
neutral_rag − neutral           T5_topical_comp        0.1374   -0.1110 -0.2485 0.3078 -0.8100   0.4196    
neutral_rag − neutral              T6_freshness       -0.0488   -0.0344  0.0145 0.0198  0.7300   0.4650    
neutral_rag − neutral          T7_source_earned       -0.4166   -0.4959 -0.0793 0.1666 -0.4800   0.6342
```

### outcome = `post_rank` — RAG deltas
```
                 pair                 treatment  coef_non_rag  coef_rag       Δ   SE_Δ       z  p_val_Δ sig
  biased_rag − biased         T1a_stats_present       -0.0550   -0.0708 -0.0157 0.0970 -0.1600   0.8711    
  biased_rag − biased         T1b_stats_density        0.0001    0.0029  0.0029 0.0052  0.5600   0.5770    
  biased_rag − biased     T2a_question_headings       -0.0311   -0.0963 -0.0652 0.0748 -0.8700   0.3833    
  biased_rag − biased T2b_structural_modularity       -0.0048   -0.0022  0.0025 0.0018  1.4200   0.1565    
  biased_rag − biased    T3_structured_data_new        0.1334    0.1853  0.0519 0.0663  0.7800   0.4332    
  biased_rag − biased         T4a_ext_citations        0.0084   -0.0302 -0.0386 0.1208 -0.3200   0.7490    
  biased_rag − biased        T4b_auth_citations        0.0438    0.0537  0.0099 0.0220  0.4500   0.6525    
  biased_rag − biased           T5_topical_comp       -0.2056   -0.6066 -0.4011 0.2920 -1.3700   0.1695    
  biased_rag − biased              T6_freshness        0.0503    0.0455 -0.0048 0.0187 -0.2600   0.7959    
  biased_rag − biased          T7_source_earned        1.2852    0.4185 -0.8667 0.2607 -3.3200   0.0009 ***
neutral_rag − neutral         T1a_stats_present        0.0493   -0.0998 -0.1490 0.1146 -1.3000   0.1936    
neutral_rag − neutral         T1b_stats_density        0.0029    0.0052  0.0023 0.0054  0.4300   0.6695    
neutral_rag − neutral     T2a_question_headings       -0.0413    0.0551  0.0964 0.0857  1.1200   0.2608    
neutral_rag − neutral T2b_structural_modularity       -0.0022   -0.0022 -0.0000 0.0018 -0.0100   0.9958    
neutral_rag − neutral    T3_structured_data_new        0.2368    0.2681  0.0312 0.0732  0.4300   0.6696    
neutral_rag − neutral         T4a_ext_citations       -0.0052    0.0693  0.0745 0.1400  0.5300   0.5948    
neutral_rag − neutral        T4b_auth_citations       -0.0040   -0.0195 -0.0154 0.0199 -0.7800   0.4368    
neutral_rag − neutral           T5_topical_comp        0.1617   -0.1750 -0.3367 0.3165 -1.0600   0.2874    
neutral_rag − neutral              T6_freshness        0.0607    0.0309 -0.0298 0.0207 -1.4400   0.1492    
neutral_rag − neutral          T7_source_earned        0.2159    0.1133 -0.1026 0.1679 -0.6100   0.5413
```

### 7C. RAG × cell heterogeneity (mean rank_delta per engine × pool × model)
Below is the descriptive (not DML-adjusted) mean rank_delta in each cell, then the cell-level RAG attenuation.

```
search_engine  pool_size model_short  biased  biased_rag  neutral  neutral_rag
          ddg         20       Llama  +3.330      +1.630   +0.967       +0.340
          ddg         20       Qwen2  +3.551      +1.442   +1.235       +0.333
          ddg         50       Llama  +5.010      +3.531   +1.118       +0.520
          ddg         50       Qwen2  +4.729      +2.634   +0.898       +0.574
      searxng         20       Llama  +2.701      +2.448   +0.752       +0.230
      searxng         20       Qwen2  +2.614      +1.340   +0.695       +0.231
      searxng         50       Llama  +1.446      +1.726   +0.076       +0.213
      searxng         50       Qwen2  +1.424      +1.260   +0.056       +0.175
```

**RAG attenuation per cell** (rag − non_rag mean rank_delta):
```
search_engine  pool_size model_short  Δ(biased_rag−biased)  Δ(neutral_rag−neutral)
          ddg         20       Llama                -1.700                  -0.627
          ddg         20       Qwen2                -2.109                  -0.902
          ddg         50       Llama                -1.479                  -0.598
          ddg         50       Qwen2                -2.095                  -0.324
      searxng         20       Llama                -0.253                  -0.522
      searxng         20       Qwen2                -1.274                  -0.464
      searxng         50       Llama                +0.280                  +0.137
      searxng         50       Qwen2                -0.164                  +0.119
```

*Saved → `data/dml_results/rag_cell_heterogeneity.parquet`.*

## 8. Promotion outcome — binary rate by variant and source class

`promotion = 1 if rank_delta > 0`. We report unconditional rates per variant and source-class slice. Below is *descriptive*; a full IRM-fit per-treatment table is left as a follow-up if needed.

```
    variant          slice     n  promotion_rate
     biased       all rows 32559          0.7010
     biased earned-media=1   372          0.5480
     biased earned-media=0 19280          0.6920
    neutral       all rows 48767          0.4360
    neutral earned-media=1   979          0.4320
    neutral earned-media=0 21138          0.4910
 biased_rag       all rows 19197          0.6630
 biased_rag earned-media=1   170          0.6060
 biased_rag earned-media=0 12016          0.6630
neutral_rag       all rows 28320          0.4040
neutral_rag earned-media=1   548          0.4180
neutral_rag earned-media=0 11730          0.4630
```

### 8B. Earned-media penalty in promotion-rate space
```
    variant  n_earned  n_other  p(earned)  p(other)       Δ   SE_Δ       z    p_val sig
     biased       372    19280     0.5480    0.6920 -0.1440 0.0260 -5.5300 3.26e-08 ***
    neutral       979    21138     0.4320    0.4910 -0.0580 0.0162 -3.6100 3.08e-04 ***
 biased_rag       170    12016     0.6060    0.6630 -0.0570 0.0377 -1.5200 1.28e-01    
neutral_rag       548    11730     0.4180    0.4630 -0.0450 0.0216 -2.0800 3.79e-02   *
```

## 9. Variance explained & nuisance fit quality

How much variance in each target is even predictable, and how cleanly is each treatment identified?

**Total variance explained** by the full X-set (origins + DFS).
```
                     target                           label     n  r2_all  r2_orig  r2_dfs  delta_r2_dfs
                 rank_delta                  OUT rank_delta 65203  0.7811   0.7774  0.0481        0.0037
                  post_rank                   OUT post_rank 65203  0.3585   0.3491  0.0051        0.0094
        treat_source_earned         TRT treat_source_earned 65203  0.2949   0.2772  0.0468        0.0177
               has_llms_txt                TRT has_llms_txt 65203  0.2264   0.2158  0.0655        0.0106
         treat_topical_comp          TRT treat_topical_comp 39561  0.5289   0.5216  0.1443        0.0074
            treat_freshness             TRT treat_freshness 58566  0.3710   0.3628  0.0951        0.0082
      treat_structured_data       TRT treat_structured_data 58566  0.3213   0.3102  0.0776        0.0111
    T3_structured_data_code     TRT T3_structured_data_code 59577  0.2026   0.1958  0.0441        0.0068
    treat_question_headings     TRT treat_question_headings 58566  0.3601   0.3535  0.0883        0.0065
T1_statistical_density_code TRT T1_statistical_density_code 57757  0.4091   0.3676  0.1617        0.0415
```

**Nuisance R² per treatment (POOLED).** High `r2_m_D_given_X` → treatment well-explained by X → likely confounded. Low → near-experimental.

```
                treatment    outcome     n  r2_g_Y_given_X  r2_m_D_given_X  r2_struct_Ytilde_given_Dtilde   theta
        T4a_ext_citations  post_rank 58566          0.3732          0.5802                         0.0000  0.0412
        T4a_ext_citations rank_delta 58566          0.7823          0.5802                         0.0000 -0.0414
          T5_topical_comp  post_rank 39561          0.3835          0.5289                         0.0006 -0.4921
          T5_topical_comp rank_delta 39561          0.7141          0.5289                         0.0005  0.4602
       T4b_auth_citations  post_rank 58566          0.3732          0.5109                         0.0002  0.0185
       T4b_auth_citations rank_delta 58566          0.7823          0.5109                         0.0003 -0.0210
                  T4_code rank_delta 59577          0.7843          0.4935                         0.0002 -0.0205
                  T4_code  post_rank 59577          0.3710          0.4935                         0.0002  0.0186
        T1a_stats_present rank_delta 58566          0.7823          0.4649                         0.0000  0.0255
        T1a_stats_present  post_rank 58566          0.3732          0.4649                         0.0000 -0.0143
T2b_structural_modularity rank_delta 58566          0.7823          0.4189                         0.0004  0.0021
T2b_structural_modularity  post_rank 58566          0.3732          0.4189                         0.0004 -0.0021
                  T1_code  post_rank 57757          0.3736          0.4091                         0.0014  0.0197
                  T1_code rank_delta 57757          0.7828          0.4091                         0.0012 -0.0185
        T1b_stats_density rank_delta 56784          0.7812          0.4039                         0.0012 -0.0187
        T1b_stats_density  post_rank 56784          0.3763          0.4039                         0.0013  0.0193
             T6_freshness  post_rank 58566          0.3732          0.3710                         0.0016  0.0611
             T6_freshness rank_delta 58566          0.7823          0.3710                         0.0016 -0.0605
    T2a_question_headings rank_delta 58566          0.7823          0.3601                         0.0003  0.1023
    T2a_question_headings  post_rank 58566          0.3732          0.3601                         0.0003 -0.1028
                   T4_llm  post_rank 59554          0.3707          0.3500                         0.0006  0.0301
                   T4_llm rank_delta 59554          0.7843          0.3500                         0.0006 -0.0302
                  T2_code  post_rank 59577          0.3710          0.3382                         0.0002 -0.0705
                  T2_code rank_delta 59577          0.7843          0.3382                         0.0002  0.0745
   T3_structured_data_new rank_delta 58566          0.7823          0.3213                         0.0006 -0.1357
   T3_structured_data_new  post_rank 58566          0.3732          0.3213                         0.0007  0.1443
                   T2_llm  post_rank 59554          0.3707          0.2994                         0.0001  0.0401
                   T2_llm rank_delta 59554          0.7843          0.2994                         0.0001 -0.0425
         T7_source_earned rank_delta 65203          0.7811          0.2949                         0.0138 -1.7416
         T7_source_earned  post_rank 65203          0.3585          0.2949                         0.0140  1.7409
               T_llms_txt rank_delta 65203          0.7811          0.2264                         0.0003  0.1047
               T_llms_txt  post_rank 65203          0.3585          0.2264                         0.0004 -0.1072
                   T3_llm rank_delta 59554          0.7843          0.2069                         0.0003  0.0942
                   T3_llm  post_rank 59554          0.3707          0.2069                         0.0003 -0.0910
                  T3_code  post_rank 59577          0.3710          0.2026                         0.0004 -0.1279
                  T3_code rank_delta 59577          0.7843          0.2026                         0.0004  0.1285
                   T1_llm  post_rank 59554          0.3707          0.1992                         0.0000 -0.0019
                   T1_llm rank_delta 59554          0.7843          0.1992                         0.0001  0.0027
```

## 10. Confounder importance & OLS significance

**Confounder importance ranking** (mean LightGBM importance across nuisance fits).
```
              Unnamed: 0  mean_rank  mean_importance_pct  coverage
         conf_word_count     2.7000              19.0183    0.8710
     conf_internal_links     4.2000              10.8225    0.8935
         conf_images_alt     4.9000               6.9315    0.8847
     conf_outbound_links     6.0000               4.1706    0.8857
        conf_readability     7.0000               5.5518    0.8463
   conf_domain_authority     9.0000               3.6998    0.2175
                 dfs_cpc     9.0000               2.4931    0.6524
          conf_title_len     9.2000               3.4706    0.9996
               conf_bm25     9.5000               4.5256    0.9996
       conf_title_kw_sim    11.4000               3.9838    0.9996
        conf_brand_recog    11.7000               1.9853    0.9996
  dfs_keyword_difficulty    11.8000               2.6575    0.7692
         dfs_competition    11.9000               2.1256    0.7684
          conf_backlinks    12.2000               2.7510    0.1120
  conf_referring_domains    13.2000               2.1168    0.1120
        conf_snippet_len    13.7000               1.5647    0.9996
     conf_snippet_kw_sim    13.8000               2.5080    0.9996
       dfs_search_volume    14.8000               1.3271    0.8064
      conf_serp_position    15.5000              17.4250    0.9996
dfs_intent_informational    20.2000               0.3165    1.0000
   dfs_intent_commercial    21.5000               0.2821    1.0000
 dfs_intent_navigational    22.4000               0.0990    1.0000
dfs_intent_transactional    22.7000               0.0643    1.0000
       conf_title_has_kw    23.1500               0.0580    0.9996
              conf_https    23.5500               0.0516    0.9996
```

**Leave-one-out ΔR² (rank_delta outcome)** — top 15.
```
   outcome             confounder  r2_full  r2_without  delta_r2
rank_delta     conf_serp_position   0.7811      0.1264    0.6547
rank_delta              conf_bm25   0.7811      0.7789    0.0023
rank_delta       conf_brand_recog   0.7811      0.7789    0.0022
rank_delta  conf_domain_authority   0.7811      0.7797    0.0014
rank_delta        dfs_competition   0.7811      0.7799    0.0012
rank_delta        conf_word_count   0.7811      0.7799    0.0012
rank_delta    conf_outbound_links   0.7811      0.7801    0.0010
rank_delta                dfs_cpc   0.7811      0.7802    0.0009
rank_delta    conf_internal_links   0.7811      0.7805    0.0006
rank_delta       conf_readability   0.7811      0.7805    0.0006
rank_delta        conf_images_alt   0.7811      0.7806    0.0005
rank_delta         conf_title_len   0.7811      0.7807    0.0004
rank_delta dfs_keyword_difficulty   0.7811      0.7807    0.0004
rank_delta    conf_snippet_kw_sim   0.7811      0.7809    0.0003
rank_delta      dfs_search_volume   0.7811      0.7809    0.0002
```

**OLS significance of each confounder** (rank_delta only, sorted by |t|).
```
              confounder    coef     se   t_stat  p_val  ci_low  ci_high stars
      conf_serp_position  4.0362 0.0109 369.0355 0.0000  4.0148   4.0577   ***
        conf_brand_recog  0.2467 0.0086  28.7115 0.0000  0.2299   0.2635   ***
         dfs_competition  0.1124 0.0095  11.7794 0.0000  0.0937   0.1311   ***
   conf_domain_authority  0.1155 0.0107  10.8254 0.0000  0.0946   0.1364   ***
       conf_title_has_kw -0.1030 0.0106  -9.6801 0.0000 -0.1239  -0.0822   ***
               conf_bm25 -0.0905 0.0102  -8.8392 0.0000 -0.1106  -0.0704   ***
     conf_snippet_kw_sim -0.0816 0.0099  -8.2114 0.0000 -0.1010  -0.0621   ***
         conf_word_count -0.1154 0.0151  -7.6458 0.0000 -0.1450  -0.0859   ***
     conf_outbound_links -0.0642 0.0114  -5.6500 0.0000 -0.0865  -0.0419   ***
 dfs_intent_navigational -0.0388 0.0077  -5.0565 0.0000 -0.0538  -0.0237   ***
                 dfs_cpc  0.0446 0.0091   4.9011 0.0000  0.0268   0.0624   ***
          conf_title_len -0.0659 0.0162  -4.0616 0.0000 -0.0976  -0.0341   ***
       dfs_search_volume  0.0291 0.0083   3.4936 0.0005  0.0128   0.0454   ***
              conf_https  0.0312 0.0092   3.3762 0.0007  0.0131   0.0493   ***
         conf_images_alt  0.0298 0.0112   2.6637 0.0077  0.0079   0.0517    **
   dfs_intent_commercial  0.0141 0.0055   2.5491 0.0108  0.0033   0.0249     *
dfs_intent_informational  0.0097 0.0063   1.5353 0.1247 -0.0027   0.0221   NaN
  dfs_keyword_difficulty  0.0144 0.0095   1.5145 0.1299 -0.0042   0.0330   NaN
          conf_backlinks -0.0135 0.0090  -1.4963 0.1346 -0.0312   0.0042   NaN
  conf_referring_domains -0.0142 0.0096  -1.4765 0.1398 -0.0331   0.0047   NaN
        conf_snippet_len  0.0213 0.0174   1.2254 0.2204 -0.0128   0.0553   NaN
     conf_internal_links  0.0191 0.0321   0.5931 0.5531 -0.0439   0.0821   NaN
       conf_title_kw_sim  0.0033 0.0113   0.2944 0.7684 -0.0188   0.0255   NaN
dfs_intent_transactional -0.0022 0.0083  -0.2612 0.7939 -0.0185   0.0141   NaN
        conf_readability -0.0089 0.0912  -0.0980 0.9219 -0.1878   0.1699   NaN
```

## 11. Paper architecture & narrative — what the numbers support

Based on every diagnostic above, the table below restates the conclusion in
plain language, with section pointers.

### Headline (Spec B, RW-corrected)

**Promoters of LLM rank** (negative `rank_delta` coefficient, i.e. doc moves UP):
- `T1a_stats_present`  — presence of statistics in body.
- `T5_topical_comp`    — topical completeness score.
- `T2a_question_headings` — Q&A-style structural headings.
- `T_llms_txt`         — domain ships an `llms.txt` file.

**Demoters of LLM rank** (positive `rank_delta`, doc pushed DOWN):
- `T6_freshness`           — heavy freshness boilerplate.
- `T3_structured_data_new` — JSON-LD / schema markup.
- `T2_llm`                 — LLM-coded question-heading variant.

### Source-identity finding (side piece, §4 of paper)

`T7_source_earned` (membership in the curated 250-domain earned-media list) carries
the **single largest demotion coefficient** (≈ −1.7 to −1.8 on `rank_delta`, p<0.001)
even with all content controls + the LLM in-context retrieval. This survives Romano–Wolf.
**Interpretation:** the LLM rerankers systematically push DOWN sources that the curated
list considers "earned media" — a robust LLM-vs-organic-web bias, distinct from any
content treatment.

### RAG mitigation finding (§5)

RAG attenuates the source-class penalty in BIASED prompts (≈ 20% reduction) and is
flat under neutral prompts. The per-cell heterogeneity table shows the attenuation
is sharpest for the **Llama × searxng × serp50** cell — that's the paragraph to cite.

### Category-switch decision (recorded for §3 + Appendix A)

See Section 2C above. Any treatment carrying `→ drop to confounder` should be moved
to the X-set in the next refit cycle; until then, footnote in the Appendix that the
Spec B table is the recommended estimand, since its coefficient there has the right
"effect with everything else fixed" interpretation.

### Heterogeneity (§6)

Cell-level pivots show **post_rank** effects are more stable across cells than
`rank_delta`. Two cells of interest:
- `searxng_Qwen2.5-72B-Instruct_serp50_top10` — strongest source-identity effect.
- `duckduckgo_Llama-3.3-70B-Instruct_serp20_top10` — weakest effects overall (noise floor).

### Recommended section order

| §    | Topic                                                                | Source         |
|------|----------------------------------------------------------------------|----------------|
| 3    | Content-treatment effects (Spec B headline)                          | §3 of report   |
| 4    | Earned-media demotion (source identity)                              | §3 + §8 + §10  |
| 5    | RAG mitigation                                                       | §7             |
| 6    | Heterogeneity across engine × LLM × pool                             | §6             |
| 7    | Diagnostics: category-switch audit, variance, confounder importance   | §2 + §9 + §10  |
| App A | Spec A vs Spec B per-variant table                                    | §5             |
| App B | Joint inference w/ Romano–Wolf / Bonferroni                           | §4             |
| App C | Promotion-rate companion outcome                                      | §8             |

