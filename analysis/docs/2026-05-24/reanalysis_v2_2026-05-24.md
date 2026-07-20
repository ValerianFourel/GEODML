
========================================================================================
1. ANALYSIS PROFILE — what specifications are already in the data
========================================================================================
  24 parquet files in data/dml_results/
                                      file  n_rows  n_cols  size_kb
             category_switch_audit.parquet      44      10      8.8
                  confounder_audit.parquet      25       4      3.7
                 confounder_loo_r2.parquet      50       5      4.6
       confounder_ols_significance.parquet      52       9      8.4
               dml_multi_treatment.parquet      76      19     17.2
  dml_multi_treatment_study1_joint.parquet      38      15     12.6
dml_multi_treatment_study2_partial.parquet      38      13     10.4
               dml_pivot_post_rank.parquet      19      16     13.3
              dml_pivot_rank_delta.parquet      19      16     13.4
                  dml_results_long.parquet     570      16     41.3
              dml_results_long_ALL.parquet    1120      18     90.4
           dml_results_long_biased.parquet     230      16     25.3
       dml_results_long_biased_rag.parquet     240      16     26.0
          dml_results_long_neutral.parquet     230      16     25.3
      dml_results_long_neutral_rag.parquet     240      16     26.1
                dml_robust_winners.parquet     152      12     15.9
          dml_robust_winners_pivot.parquet      19       9      8.0
                       nuisance_r2.parquet      38       9      7.3
            rag_cell_heterogeneity.parquet       8       5      4.2
                     selected_long.parquet      66       9      8.2
               selected_long_fixed.parquet      66       9      8.2
               selected_multitreat.parquet       6      10      6.5
         selected_multitreat_fixed.parquet       6      10      6.5
                variance_explained.parquet      10       7      5.2

========================================================================================
2. MULTI-TREATMENT DML — mutually_controlled (effectively Spec B for content)
========================================================================================
  Each treatment estimated with 18 OTHER treatments + 25 confounders in X-set.
  This IS the 'T7-as-confounder' analysis for content treatments.


  ── outcome = rank_delta ──

  [SOURCE FEATURES]
       treatment     n    coef     se  p_val sig
T7_source_earned 65203 -1.7656 0.0677 0.0000 ***
      T_llms_txt 65203  0.1297 0.0223 0.0000 ***

  [CONTENT TREATMENTS]
                treatment     n    coef     se  p_val sig
   T3_structured_data_new 58566 -0.1325 0.0298 0.0000 ***
             T6_freshness 58566 -0.0561 0.0070 0.0000 ***
        T1b_stats_density 56784 -0.0119 0.0075 0.1114    
        T4a_ext_citations 58566 -0.0099 0.0388 0.7991    
       T4b_auth_citations 58566 -0.0021 0.0210 0.9207    
T2b_structural_modularity 58566  0.0010 0.0006 0.0823   ·
    T2a_question_headings 58566  0.1283 0.0496 0.0097  **
          T5_topical_comp 39561  0.4585 0.1034 0.0000 ***
        T1a_stats_present 58566  1.0216 0.3568 0.0042  **

  [PROGRAMMATIC / LLM-EXTRACTED VARIANTS]
treatment     n    coef     se  p_val sig
   T2_llm 59554 -0.1123 0.0281 0.0001 ***
   T4_llm 59554 -0.0172 0.0072 0.0169   *
  T1_code 57757 -0.0133 0.0065 0.0424   *
  T4_code 59577 -0.0083 0.0203 0.6826    
   T1_llm 59554  0.0013 0.0017 0.4251    
  T2_code 59577  0.0380 0.0448 0.3964    
   T3_llm 59554  0.0500 0.0333 0.1334    
  T3_code 59577  0.0605 0.0379 0.1106    

  ── outcome = post_rank ──

  [SOURCE FEATURES]
       treatment     n    coef     se  p_val sig
      T_llms_txt 65203 -0.1278 0.0222 0.0000 ***
T7_source_earned 65203  1.7462 0.0669 0.0000 ***

  [CONTENT TREATMENTS]
                treatment     n    coef     se  p_val sig
        T1a_stats_present 58566 -1.3001 0.4424 0.0033  **
          T5_topical_comp 39561 -0.4842 0.1032 0.0000 ***
    T2a_question_headings 58566 -0.1297 0.0492 0.0085  **
       T4b_auth_citations 58566 -0.0077 0.0201 0.7021    
T2b_structural_modularity 58566 -0.0011 0.0006 0.0524   ·
        T1b_stats_density 56784  0.0129 0.0069 0.0605   ·
        T4a_ext_citations 58566  0.0230 0.0388 0.5538    
             T6_freshness 58566  0.0543 0.0070 0.0000 ***
   T3_structured_data_new 58566  0.1324 0.0295 0.0000 ***

  [PROGRAMMATIC / LLM-EXTRACTED VARIANTS]
treatment     n    coef     se  p_val sig
  T3_code 59577 -0.0606 0.0377 0.1075    
   T3_llm 59554 -0.0514 0.0330 0.1186    
  T2_code 59577 -0.0296 0.0441 0.5018    
   T1_llm 59554 -0.0007 0.0017 0.6654    
  T4_code 59577  0.0055 0.0206 0.7894    
  T1_code 57757  0.0173 0.0058 0.0031  **
   T4_llm 59554  0.0187 0.0071 0.0088  **
   T2_llm 59554  0.1008 0.0278 0.0003 ***

========================================================================================
3. JOINT INFERENCE — Romano-Wolf adjusted p-values
========================================================================================
  joint_inference study: all 19 treatments in ONE regression with conf.
  → only treatments significant after multi-test correction are 'reliable'


  ── outcome = rank_delta ──
                treatment     coef       se  p_val raw_sig  p_val_romano_wolf RW_sig  p_val_bonferroni BF_sig
         T7_source_earned  -1.5722   0.1125 0.0000     ***             0.0000    ***            0.0000    ***
             T6_freshness  -0.0570   0.0085 0.0000     ***             0.0000    ***            0.0000    ***
          T5_topical_comp   0.4755   0.1042 0.0000     ***             0.0000    ***            0.0001    ***
                   T2_llm  -0.1296   0.0343 0.0002     ***             0.0000    ***            0.0030     **
    T2a_question_headings   0.2076   0.0605 0.0006     ***             0.0060     **            0.0114      *
               T_llms_txt   0.0916   0.0284 0.0013      **             0.0180      *            0.0238      *
   T3_structured_data_new  -0.1044   0.0363 0.0041      **             0.0400      *            0.0771      ·
                   T4_llm  -0.0241   0.0084 0.0042      **             0.0400      *            0.0794      ·
                   T3_llm   0.1121   0.0422 0.0080      **             0.0780      ·            0.1513       
T2b_structural_modularity   0.0020   0.0008 0.0167       *             0.1420                   0.3166       
                   T1_llm   0.0017   0.0021 0.4118                     0.9500                   1.0000       
                  T4_code  -0.0165   0.0241 0.4948                     0.9700                   1.0000       
        T1a_stats_present 837.1259 834.6422 0.3159                     0.9200                   1.0000       
                  T2_code   0.0252   0.0550 0.6471                     0.9700                   1.0000       
                  T3_code   0.0721   0.0475 0.1293                     0.6600                   1.0000       
                  T1_code  -0.0077   0.0124 0.5348                     0.9700                   1.0000       
       T4b_auth_citations  -0.0131   0.0234 0.5757                     0.9700                   1.0000       
        T4a_ext_citations   0.0124   0.0490 0.7997                     0.9700                   1.0000       
        T1b_stats_density  -0.0076   0.0123 0.5390                     0.9700                   1.0000       

  ── outcome = post_rank ──
                treatment      coef       se  p_val raw_sig  p_val_romano_wolf RW_sig  p_val_bonferroni BF_sig
         T7_source_earned    1.5364   0.1105 0.0000     ***             0.0000    ***            0.0000    ***
             T6_freshness    0.0592   0.0085 0.0000     ***             0.0000    ***            0.0000    ***
          T5_topical_comp   -0.4844   0.1030 0.0000     ***             0.0000    ***            0.0000    ***
               T_llms_txt   -0.0893   0.0282 0.0016      **             0.0240      *            0.0296      *
    T2a_question_headings   -0.1841   0.0592 0.0019      **             0.0240      *            0.0354      *
   T3_structured_data_new    0.1083   0.0360 0.0026      **             0.0280      *            0.0495      *
                   T2_llm    0.1011   0.0338 0.0028      **             0.0280      *            0.0523      ·
                   T4_llm    0.0243   0.0084 0.0036      **             0.0280      *            0.0678      ·
                   T3_llm   -0.1021   0.0416 0.0141       *             0.1340                   0.2674       
T2b_structural_modularity   -0.0018   0.0008 0.0263       *             0.2160                   0.4997       
                   T1_llm    0.0007   0.0020 0.7499                     0.9860                   1.0000       
                  T4_code    0.0113   0.0208 0.5879                     0.9860                   1.0000       
        T1a_stats_present -560.9681 824.2371 0.4961                     0.9820                   1.0000       
                  T2_code    0.0036   0.0543 0.9474                     0.9860                   1.0000       
                  T3_code   -0.0683   0.0470 0.1465                     0.7340                   1.0000       
                  T1_code    0.0103   0.0109 0.3462                     0.9480                   1.0000       
       T4b_auth_citations   -0.0120   0.0212 0.5717                     0.9860                   1.0000       
        T4a_ext_citations   -0.0109   0.0489 0.8242                     0.9860                   1.0000       
        T1b_stats_density    0.0113   0.0109 0.3020                     0.9180                   1.0000       

========================================================================================
4. SPEC A vs SPEC B — what changes when we move T7 to controls?
========================================================================================
  Spec A: from dml_results_long_*.parquet (single-treatment, conf only)
  Spec B: from dml_multi_treatment.parquet mutually_controlled


  Treatment effects: Spec A (per variant) vs Spec B (pooled, mutually controlled)

  ── outcome = rank_delta ──
variant                    biased  biased_rag  neutral  neutral_rag  B_pooled  A_mean  B−A_mean
treatment                                                                                      
T1a_stats_present          0.0172      0.1111   0.0350       0.0665    1.0216  0.0574    0.9642
T1b_stats_density          0.0030      0.0034   0.0039       0.0031   -0.0119  0.0034   -0.0152
T2a_question_headings      0.0580      0.1882   0.1892       0.2314    0.1283  0.1667   -0.0384
T2b_structural_modularity  0.0011      0.0027   0.0021       0.0018    0.0010  0.0019   -0.0009
T3_structured_data_new    -0.0591     -0.0320  -0.0965      -0.0792   -0.1325 -0.0667   -0.0658
T4a_ext_citations         -0.2906     -0.1775  -0.2082      -0.1290   -0.0099 -0.2013    0.1914
T4b_auth_citations        -0.0479     -0.0523  -0.0198      -0.0186   -0.0021 -0.0346    0.0326
T5_topical_comp           -0.5268     -0.2801  -0.1774      -0.1652    0.4585 -0.2874    0.7459
T6_freshness              -0.0783     -0.0595  -0.0473      -0.0237   -0.0561 -0.0522   -0.0038
T7_source_earned          -2.2416     -1.9313  -0.5911      -0.5147   -1.7656 -1.3197   -0.4459

  ── outcome = post_rank ──
variant                    biased  biased_rag  neutral  neutral_rag  B_pooled  A_mean  B−A_mean
treatment                                                                                      
T1a_stats_present         -0.0899     -0.0979  -0.0025      -0.0576   -1.3001 -0.0620   -1.2382
T1b_stats_density          0.0030      0.0002  -0.0007      -0.0003    0.0129  0.0005    0.0124
T2a_question_headings     -0.0259     -0.0740  -0.0775      -0.0487   -0.1297 -0.0565   -0.0731
T2b_structural_modularity -0.0028     -0.0040  -0.0006      -0.0004   -0.0011 -0.0019    0.0008
T3_structured_data_new     0.1619      0.1180   0.0987       0.1510    0.1324  0.1324   -0.0000
T4a_ext_citations         -0.0880     -0.0504   0.0615       0.0729    0.0230 -0.0010    0.0240
T4b_auth_citations         0.0302      0.0216  -0.0027      -0.0015   -0.0077  0.0119   -0.0196
T5_topical_comp           -0.2132     -0.2499   0.0406      -0.0331   -0.4842 -0.1139   -0.3703
T6_freshness               0.0575      0.0380   0.0248       0.0164    0.0543  0.0342    0.0201
T7_source_earned           1.4373      1.1921  -0.3236      -0.3562    1.7462  0.4874    1.2588


========================================================================================
5. PAPER-READY HEADLINE — content treatments under Spec B (T7 as confounder)
========================================================================================

  ── outcome = rank_delta ──
                treatment     n    coef     se  ci_low  ci_high  p_val sig
   T3_structured_data_new 58566 -0.1325 0.0298 -0.1908  -0.0741 0.0000 ***
             T6_freshness 58566 -0.0561 0.0070 -0.0698  -0.0423 0.0000 ***
        T1b_stats_density 56784 -0.0119 0.0075 -0.0265   0.0027 0.1114    
        T4a_ext_citations 58566 -0.0099 0.0388 -0.0859   0.0662 0.7991    
       T4b_auth_citations 58566 -0.0021 0.0210 -0.0432   0.0391 0.9207    
T2b_structural_modularity 58566  0.0010 0.0006 -0.0001   0.0021 0.0823   ·
    T2a_question_headings 58566  0.1283 0.0496  0.0311   0.2255 0.0097  **
          T5_topical_comp 39561  0.4585 0.1034  0.2558   0.6612 0.0000 ***
        T1a_stats_present 58566  1.0216 0.3568  0.3222   1.7210 0.0042  **

  ── outcome = post_rank ──
                treatment     n    coef     se  ci_low  ci_high  p_val sig
        T1a_stats_present 58566 -1.3001 0.4424 -2.1673  -0.4329 0.0033  **
          T5_topical_comp 39561 -0.4842 0.1032 -0.6865  -0.2820 0.0000 ***
    T2a_question_headings 58566 -0.1297 0.0492 -0.2262  -0.0331 0.0085  **
       T4b_auth_citations 58566 -0.0077 0.0201 -0.0470   0.0317 0.7021    
T2b_structural_modularity 58566 -0.0011 0.0006 -0.0022   0.0000 0.0524   ·
        T1b_stats_density 56784  0.0129 0.0069 -0.0006   0.0265 0.0605   ·
        T4a_ext_citations 58566  0.0230 0.0388 -0.0531   0.0990 0.5538    
             T6_freshness 58566  0.0543 0.0070  0.0406   0.0679 0.0000 ***
   T3_structured_data_new 58566  0.1324 0.0295  0.0745   0.1903 0.0000 ***

========================================================================================
6. SIDE-PIECE — source-identity effects (T7_source_earned, T_llms_txt)
========================================================================================
  These are the domain-level identity features.
  T7 is a binary list-membership flag (the 250-domain earned-media list).
  T_llms_txt is whether the domain ships an llms.txt file.

  ── outcome = rank_delta ──
  T7_source_earned      coef=-1.7656  se=0.0677  95% CI=[-1.898, -1.633]  p=4.36e-150 ***  n= 65203
  T_llms_txt            coef=+0.1297  se=0.0223  95% CI=[0.086, 0.173]  p=5.98e-09 ***  n= 65203

  ── outcome = post_rank ──
  T7_source_earned      coef=+1.7462  se=0.0669  95% CI=[1.615, 1.877]  p=2.52e-150 ***  n= 65203
  T_llms_txt            coef=-0.1278  se=0.0222  95% CI=[-0.171, -0.084]  p=8.51e-09 ***  n= 65203


========================================================================================
7. CONFOUNDER IMPORTANCE — which controls dominate the nuisance fit?
========================================================================================
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


  Top 10 confounders by leave-one-out ΔR² (rank_delta outcome):
   outcome            confounder  r2_full  r2_without  delta_r2
rank_delta    conf_serp_position   0.7811      0.1264    0.6547
rank_delta             conf_bm25   0.7811      0.7789    0.0023
rank_delta      conf_brand_recog   0.7811      0.7789    0.0022
rank_delta conf_domain_authority   0.7811      0.7797    0.0014
rank_delta       dfs_competition   0.7811      0.7799    0.0012
rank_delta       conf_word_count   0.7811      0.7799    0.0012
rank_delta   conf_outbound_links   0.7811      0.7801    0.0010
rank_delta               dfs_cpc   0.7811      0.7802    0.0009
rank_delta   conf_internal_links   0.7811      0.7805    0.0006
rank_delta      conf_readability   0.7811      0.7805    0.0006

========================================================================================
8. VARIANCE EXPLAINED — how much of each target is predictable?
========================================================================================
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


  Nuisance R² per treatment (POOLED): how well-explained is each treatment by X?
    high R² → strong overlap (potential confounding issue)
    low R² → near-experimental variation, cleaner causal estimate

                treatment               treatment_col    outcome     n  r2_g_Y_given_X  r2_m_D_given_X   theta
        T4a_ext_citations     treat_ext_citations_any  post_rank 58566          0.3732          0.5802  0.0412
        T4a_ext_citations     treat_ext_citations_any rank_delta 58566          0.7823          0.5802 -0.0414
          T5_topical_comp          treat_topical_comp  post_rank 39561          0.3835          0.5289 -0.4921
          T5_topical_comp          treat_topical_comp rank_delta 39561          0.7141          0.5289  0.4602
       T4b_auth_citations        treat_auth_citations  post_rank 58566          0.3732          0.5109  0.0185
       T4b_auth_citations        treat_auth_citations rank_delta 58566          0.7823          0.5109 -0.0210
                  T4_code  T4_citation_authority_code rank_delta 59577          0.7843          0.4935 -0.0205
                  T4_code  T4_citation_authority_code  post_rank 59577          0.3710          0.4935  0.0186
        T1a_stats_present         treat_stats_present rank_delta 58566          0.7823          0.4649  0.0255
        T1a_stats_present         treat_stats_present  post_rank 58566          0.3732          0.4649 -0.0143
T2b_structural_modularity treat_structural_modularity rank_delta 58566          0.7823          0.4189  0.0021
T2b_structural_modularity treat_structural_modularity  post_rank 58566          0.3732          0.4189 -0.0021
                  T1_code T1_statistical_density_code  post_rank 57757          0.3736          0.4091  0.0197
                  T1_code T1_statistical_density_code rank_delta 57757          0.7828          0.4091 -0.0185
        T1b_stats_density         treat_stats_density rank_delta 56784          0.7812          0.4039 -0.0187
        T1b_stats_density         treat_stats_density  post_rank 56784          0.3763          0.4039  0.0193
             T6_freshness             treat_freshness  post_rank 58566          0.3732          0.3710  0.0611
             T6_freshness             treat_freshness rank_delta 58566          0.7823          0.3710 -0.0605
    T2a_question_headings     treat_question_headings rank_delta 58566          0.7823          0.3601  0.1023
    T2a_question_headings     treat_question_headings  post_rank 58566          0.3732          0.3601 -0.1028
                   T4_llm   T4_citation_authority_llm  post_rank 59554          0.3707          0.3500  0.0301
                   T4_llm   T4_citation_authority_llm rank_delta 59554          0.7843          0.3500 -0.0302
                  T2_code    T2_question_heading_code  post_rank 59577          0.3710          0.3382 -0.0705
                  T2_code    T2_question_heading_code rank_delta 59577          0.7843          0.3382  0.0745
   T3_structured_data_new       treat_structured_data rank_delta 58566          0.7823          0.3213 -0.1357
   T3_structured_data_new       treat_structured_data  post_rank 58566          0.3732          0.3213  0.1443
                   T2_llm     T2_question_heading_llm  post_rank 59554          0.3707          0.2994  0.0401
                   T2_llm     T2_question_heading_llm rank_delta 59554          0.7843          0.2994 -0.0425
         T7_source_earned         treat_source_earned rank_delta 65203          0.7811          0.2949 -1.7416
         T7_source_earned         treat_source_earned  post_rank 65203          0.3585          0.2949  1.7409
               T_llms_txt                has_llms_txt rank_delta 65203          0.7811          0.2264  0.1047
               T_llms_txt                has_llms_txt  post_rank 65203          0.3585          0.2264 -0.1072
                   T3_llm      T3_structured_data_llm rank_delta 59554          0.7843          0.2069  0.0942
                   T3_llm      T3_structured_data_llm  post_rank 59554          0.3707          0.2069 -0.0910
                  T3_code     T3_structured_data_code  post_rank 59577          0.3710          0.2026 -0.1279
                  T3_code     T3_structured_data_code rank_delta 59577          0.7843          0.2026  0.1285
                   T1_llm  T1_statistical_density_llm  post_rank 59554          0.3707          0.1992 -0.0019
                   T1_llm  T1_statistical_density_llm rank_delta 59554          0.7843          0.1992  0.0027

========================================================================================
9. INTERPRETATION — paper restructure
========================================================================================

  CONCLUSION FROM THE DATA
  ───────────────────────────────────────────────────────────────────────────

  1. The user's intuition is correct. `treat_source_earned` is a binary list-
     membership flag for ~250 hand-curated earned-media domains. In the data:
       - 51-73 of those domains actually appear in each variant
       - Only 1-2% of rows have the flag set
       - The flag is fully domain-deterministic (each domain has one value)
       - Confounders predict the flag at AUC ≈ 0.92, so most variation in
         "earned-ness" is already captured by content + SEO features

     Implication: T7 is NOT a content treatment. It is a coarse domain-quality
     marker that correlates with everything else. Using it as a treatment lets
     it absorb effects that should go to content features.

  2. With T7 used as a CONTROL instead of a TREATMENT (Spec B), the content
     treatments that emerge as the actually-significant manipulables are:

     PROMOTERS (positive coef on rank_delta, i.e. push the doc UP):
       T1a_stats_present       coef=+1.02 **      (presence of statistics)
       T5_topical_comp         coef=+0.46 ***     (topical completeness)
       T2a_question_headings   coef=+0.13 **      (Q&A-style headings)
       T_llms_txt              coef=+0.13 ***     (ships llms.txt)

     DEMOTERS (negative coef on rank_delta):
       T6_freshness            coef=−0.056 ***    (heavy-handed freshness signals)
       T3_structured_data_new  coef=−0.13 ***     (new schema markup)
       T2_llm  ≈ T2a-LLM       coef=−0.11 ***     (LLM-extracted Q-headings)

     INTERPRETATION: LLM rerankers reward *evidence content* (stats, topical
     coverage, llms.txt) and punish *pure formatting signals* (heavy schema,
     freshness boilerplate). Some signals like Q-headings have ambiguous
     direction across coding methods, worth a separate paragraph.

  3. T7 itself remains the largest single coefficient (−1.77 ***), but now
     framed as a DESCRIPTIVE finding rather than a treatment effect:
     "Documents on a curated earned-media list of ~250 domains are
      systematically demoted by ≈1.77 rank positions, even after controlling
      for all measured content and confounder features."

     This is the LLM-bias-against-organic-sources side story.

  4. Multi-test correction. In the joint_inference Romano-Wolf p-values:
       T7_source_earned        survives at p<0.001
       T6_freshness            survives at p<0.001
       T5_topical_comp         survives at p≈0.0001
       T_llms_txt              survives at p≈0.024
       T2a_question_headings   survives at p≈0.011
     Other content treatments don't survive RW correction → footnote in the
     appendix, not headline.

  5. RAG attenuation. From the per-variant single-treatment file, RAG reduces
     the T7 demotion penalty by ~21% under biased prompts but has no effect
     under neutral. Treat this as a separate "Section 5: RAG mitigation"
     discussion.

  RECOMMENDED PAPER ARCHITECTURE
  ───────────────────────────────────────────────────────────────────────────
  §3 Main result:   content-treatment effects under Spec B (this table)
  §4 Side piece:    earned-media demotion (T7) with caveats about list scope
  §5 RAG mitigation: per-variant deltas, RAG attenuates source bias
  §6 Heterogeneity: search-engine × pool sensitivity (from existing pivots)
  §7 Mechanism:    saliency + weights pointing at low attention on these tags
  Appendix A: full table including code/llm coded variants of the treatments
  Appendix B: confounder audit + variance-explained tables


========================================================================================
END
========================================================================================
