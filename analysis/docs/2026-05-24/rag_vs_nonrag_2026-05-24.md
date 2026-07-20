
========================================================================================
1. SAMPLE COMPOSITION — what RAG adds or drops
========================================================================================
    variant  n_rows  n_keywords  n_domains  n_keywords_with_data  rows_per_keyword (mean)  post_rank_min  post_rank_max
     biased   96778        1011      12301                  1011                    95.73              1             10
    neutral  125613        1011       9748                  1011                   124.25              1             10
 biased_rag  103073        1011      13170                  1011                   101.95              1             10
neutral_rag  106392        1011      10080                  1011                   105.23              1             10

  Why are RAG row counts smaller?
  RAG variants only emit rerank outputs when the rag retrieval is non-empty.
  Keywords where retrieval failed are skipped, so RAG variants have fewer rows.

  Keyword overlap (rag vs non_rag):
    biased       ∩ biased_rag   : |A|= 1011  |B|= 1011  |A∩B|= 1011  |A−B|=    0  |B−A|=    0
    neutral      ∩ neutral_rag  : |A|= 1011  |B|= 1011  |A∩B|= 1011  |A−B|=    0  |B−A|=    0

========================================================================================
2. OUTCOME DISTRIBUTIONS — does RAG change WHERE the LLM lands rows?
========================================================================================
    variant  rank_delta_mean  rank_delta_std  rank_delta_median  post_rank_mean  post_rank_std  post_rank_median pct_no_change (rank_delta=0) pct_promoted (delta>0) pct_demoted (delta<0)
     biased            2.619           4.402                2.0            4.99           2.78                 5                        15.7%                  69.8%                 14.5%
    neutral            0.520           3.020                0.0            5.42           2.86                 5                        34.7%                  44.4%                 20.8%
 biased_rag            2.188           4.067                2.0            5.21           2.81                 5                        19.1%                  66.3%                 14.6%
neutral_rag            0.352           2.641                0.0            5.40           2.85                 5                        43.9%                  38.7%                 17.4%

  Paired rank_delta comparison (same (keyword, domain) URL with vs without RAG):
    biased       → biased_rag   : n_pairs=356797  mean(Δrank_delta)=-0.174  sd=3.086  paired t=-33.70  p=1.39e-248
    neutral      → neutral_rag  : n_pairs=752813  mean(Δrank_delta)=-0.054  sd=2.617  paired t=-18.00  p=2.15e-72

========================================================================================
3. PER-TREATMENT DML DELTAS — does RAG amplify or dampen each effect?
========================================================================================
  Source: pre-computed per-variant PLR+LightGBM canonical estimates
  (the per-variant single-treatment file, max-n row per cell).

  ── outcome = rank_delta — per-variant coefs ──
variant                    biased  biased_rag  neutral  neutral_rag
treatment                                                          
T1a_stats_present          0.0172      0.1111   0.0350       0.0665
T1b_stats_density          0.0030      0.0034   0.0039       0.0031
T2a_question_headings      0.0580      0.1882   0.1892       0.2314
T2b_structural_modularity  0.0011      0.0027   0.0021       0.0018
T3_structured_data_new    -0.0591     -0.0320  -0.0965      -0.0792
T4a_ext_citations         -0.2906     -0.1775  -0.2082      -0.1290
T4b_auth_citations        -0.0479     -0.0523  -0.0198      -0.0186
T5_topical_comp           -0.5268     -0.2801  -0.1774      -0.1652
T6_freshness              -0.0783     -0.0595  -0.0473      -0.0237
T7_source_earned          -2.2416     -1.9313  -0.5911      -0.5147

  ── outcome = rank_delta — RAG attenuation (Δ = coef_rag − coef_non_rag) ──
                 pair                 treatment  coef_non_rag  coef_rag       Δ   SE_Δ       z  p_val_Δ sig
  biased_rag − biased         T1a_stats_present        0.0172    0.1111  0.0940 0.0870  1.0799   0.2802    
  biased_rag − biased         T1b_stats_density        0.0030    0.0034  0.0004 0.0056  0.0739   0.9411    
  biased_rag − biased     T2a_question_headings        0.0580    0.1882  0.1302 0.0688  1.8924   0.0584   ·
  biased_rag − biased T2b_structural_modularity        0.0011    0.0027  0.0017 0.0015  1.0774   0.2813    
  biased_rag − biased    T3_structured_data_new       -0.0591   -0.0320  0.0271 0.0581  0.4660   0.6412    
  biased_rag − biased         T4a_ext_citations       -0.2906   -0.1775  0.1131 0.1035  1.0937   0.2741    
  biased_rag − biased        T4b_auth_citations       -0.0479   -0.0523 -0.0044 0.0149 -0.2976   0.7660    
  biased_rag − biased           T5_topical_comp       -0.5268   -0.2801  0.2468 0.2407  1.0254   0.3052    
  biased_rag − biased              T6_freshness       -0.0783   -0.0595  0.0188 0.0165  1.1396   0.2545    
  biased_rag − biased          T7_source_earned       -2.2416   -1.9313  0.3103 0.1831  1.6950   0.0901   ·
neutral_rag − neutral         T1a_stats_present        0.0350    0.0665  0.0315 0.0585  0.5386   0.5902    
neutral_rag − neutral         T1b_stats_density        0.0039    0.0031 -0.0008 0.0037 -0.2103   0.8335    
neutral_rag − neutral     T2a_question_headings        0.1892    0.2314  0.0422 0.0413  1.0221   0.3067    
neutral_rag − neutral T2b_structural_modularity        0.0021    0.0018 -0.0003 0.0007 -0.4376   0.6617    
neutral_rag − neutral    T3_structured_data_new       -0.0965   -0.0792  0.0173 0.0359  0.4822   0.6297    
neutral_rag − neutral         T4a_ext_citations       -0.2082   -0.1290  0.0793 0.0676  1.1731   0.2408    
neutral_rag − neutral        T4b_auth_citations       -0.0198   -0.0186  0.0012 0.0048  0.2488   0.8035    
neutral_rag − neutral           T5_topical_comp       -0.1774   -0.1652  0.0122 0.1256  0.0973   0.9225    
neutral_rag − neutral              T6_freshness       -0.0473   -0.0237  0.0236 0.0100  2.3521   0.0187   *
neutral_rag − neutral          T7_source_earned       -0.5911   -0.5147  0.0764 0.0516  1.4823   0.1383    

  ── outcome = post_rank — per-variant coefs ──
variant                    biased  biased_rag  neutral  neutral_rag
treatment                                                          
T1a_stats_present         -0.0899     -0.0979  -0.0025      -0.0576
T1b_stats_density          0.0030      0.0002  -0.0007      -0.0003
T2a_question_headings     -0.0259     -0.0740  -0.0775      -0.0487
T2b_structural_modularity -0.0028     -0.0040  -0.0006      -0.0004
T3_structured_data_new     0.1619      0.1180   0.0987       0.1510
T4a_ext_citations         -0.0880     -0.0504   0.0615       0.0729
T4b_auth_citations         0.0302      0.0216  -0.0027      -0.0015
T5_topical_comp           -0.2132     -0.2499   0.0406      -0.0331
T6_freshness               0.0575      0.0380   0.0248       0.0164
T7_source_earned           1.4373      1.1921  -0.3236      -0.3562

  ── outcome = post_rank — RAG attenuation (Δ = coef_rag − coef_non_rag) ──
                 pair                 treatment  coef_non_rag  coef_rag       Δ   SE_Δ       z  p_val_Δ sig
  biased_rag − biased         T1a_stats_present       -0.0899   -0.0979 -0.0080 0.0515 -0.1548   0.8770    
  biased_rag − biased         T1b_stats_density        0.0030    0.0002 -0.0028 0.0034 -0.8431   0.3992    
  biased_rag − biased     T2a_question_headings       -0.0259   -0.0740 -0.0481 0.0390 -1.2336   0.2173    
  biased_rag − biased T2b_structural_modularity       -0.0028   -0.0040 -0.0012 0.0010 -1.1988   0.2306    
  biased_rag − biased    T3_structured_data_new        0.1619    0.1180 -0.0440 0.0348 -1.2627   0.2067    
  biased_rag − biased         T4a_ext_citations       -0.0880   -0.0504  0.0376 0.0624  0.6022   0.5471    
  biased_rag − biased        T4b_auth_citations        0.0302    0.0216 -0.0086 0.0064 -1.3428   0.1793    
  biased_rag − biased           T5_topical_comp       -0.2132   -0.2499 -0.0367 0.1459 -0.2517   0.8013    
  biased_rag − biased              T6_freshness        0.0575    0.0380 -0.0196 0.0098 -1.9862   0.0470   *
  biased_rag − biased          T7_source_earned        1.4373    1.1921 -0.2453 0.1072 -2.2869   0.0222   *
neutral_rag − neutral         T1a_stats_present       -0.0025   -0.0576 -0.0551 0.0538 -1.0247   0.3055    
neutral_rag − neutral         T1b_stats_density       -0.0007   -0.0003  0.0004 0.0032  0.1227   0.9023    
neutral_rag − neutral     T2a_question_headings       -0.0775   -0.0487  0.0288 0.0375  0.7672   0.4429    
neutral_rag − neutral T2b_structural_modularity       -0.0006   -0.0004  0.0002 0.0007  0.2598   0.7950    
neutral_rag − neutral    T3_structured_data_new        0.0987    0.1510  0.0523 0.0337  1.5518   0.1207    
neutral_rag − neutral         T4a_ext_citations        0.0615    0.0729  0.0115 0.0632  0.1816   0.8559    
neutral_rag − neutral        T4b_auth_citations       -0.0027   -0.0015  0.0013 0.0047  0.2689   0.7880    
neutral_rag − neutral           T5_topical_comp        0.0406   -0.0331 -0.0736 0.1231 -0.5983   0.5497    
neutral_rag − neutral              T6_freshness        0.0248    0.0164 -0.0084 0.0095 -0.8871   0.3750    
neutral_rag − neutral          T7_source_earned       -0.3236   -0.3562 -0.0326 0.0520 -0.6274   0.5304    


========================================================================================
4. WHICH TREATMENTS FLIP SIGNIFICANCE with RAG?
========================================================================================

  ── outcome = rank_delta ──

  biased → biased_rag:
    T1a_stats_present               non-rag p=0.7801        rag p=0.0713 ·      coef: +0.017 → +0.111
    T2a_question_headings           non-rag p=0.2270        rag p=0.0001 ***    coef: +0.058 → +0.188  ← FLIP
    T2b_structural_modularity       non-rag p=0.3422        rag p=0.0098 **     coef: +0.001 → +0.003  ← FLIP
    T4a_ext_citations               non-rag p=0.0001 ***    rag p=0.0157 *      coef: -0.291 → -0.177
    T4b_auth_citations              non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -0.048 → -0.052
    T5_topical_comp                 non-rag p=0.0014 **     rag p=0.1096        coef: -0.527 → -0.280  ← FLIP
    T6_freshness                    non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -0.078 → -0.060
    T7_source_earned                non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -2.242 → -1.931

  neutral → neutral_rag:
    T2a_question_headings           non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +0.189 → +0.231
    T2b_structural_modularity       non-rag p=0.0000 ***    rag p=0.0001 ***    coef: +0.002 → +0.002
    T3_structured_data_new          non-rag p=0.0002 ***    rag p=0.0016 **     coef: -0.096 → -0.079
    T4a_ext_citations               non-rag p=0.0000 ***    rag p=0.0067 **     coef: -0.208 → -0.129
    T4b_auth_citations              non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -0.020 → -0.019
    T5_topical_comp                 non-rag p=0.0423 *      rag p=0.0670 ·      coef: -0.177 → -0.165  ← FLIP
    T6_freshness                    non-rag p=0.0000 ***    rag p=0.0006 ***    coef: -0.047 → -0.024
    T7_source_earned                non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -0.591 → -0.515

  ── outcome = post_rank ──

  biased → biased_rag:
    T1a_stats_present               non-rag p=0.0117 *      rag p=0.0083 **     coef: -0.090 → -0.098
    T2a_question_headings           non-rag p=0.3373        rag p=0.0085 **     coef: -0.026 → -0.074  ← FLIP
    T2b_structural_modularity       non-rag p=0.0002 ***    rag p=0.0000 ***    coef: -0.003 → -0.004
    T3_structured_data_new          non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +0.162 → +0.118
    T4a_ext_citations               non-rag p=0.0391 *      rag p=0.2681        coef: -0.088 → -0.050  ← FLIP
    T4b_auth_citations              non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +0.030 → +0.022
    T5_topical_comp                 non-rag p=0.0303 *      rag p=0.0203 *      coef: -0.213 → -0.250
    T6_freshness                    non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +0.058 → +0.038
    T7_source_earned                non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +1.437 → +1.192

  neutral → neutral_rag:
    T2a_question_headings           non-rag p=0.0018 **     rag p=0.0833 ·      coef: -0.078 → -0.049  ← FLIP
    T3_structured_data_new          non-rag p=0.0000 ***    rag p=0.0000 ***    coef: +0.099 → +0.151
    T6_freshness                    non-rag p=0.0001 ***    rag p=0.0198 *      coef: +0.025 → +0.016
    T7_source_earned                non-rag p=0.0000 ***    rag p=0.0000 ***    coef: -0.324 → -0.356

========================================================================================
5. HETEROGENEITY — RAG effect by (engine × pool × model)
========================================================================================
  Computing per-cell mean rank_delta for each variant, then RAG attenuation.

  Mean rank_delta per cell, by variant:
variant                         biased  biased_rag  neutral  neutral_rag
search_engine pool model_short                                          
ddg           20   Llama         1.546       1.320    0.220        0.187
                   Qwen2         1.546       1.189    0.272        0.231
              50   Llama         4.869       4.472    1.176        0.706
                   Qwen2         4.724       3.374    0.959        0.610
searxng       20   Llama         2.636       2.572    0.781        0.558
                   Qwen2         2.648       1.993    0.753        0.469
              50   Llama         1.393       1.482    0.029        0.085
                   Qwen2         1.404       1.216    0.064        0.086

  RAG attenuation = mean_rd(rag) − mean_rd(non_rag) per cell:
                                Δ(biased_rag−biased)  Δ(neutral_rag−neutral)
search_engine pool model_short                                              
ddg           20   Llama                      -0.226                  -0.033
                   Qwen2                      -0.357                  -0.041
              50   Llama                      -0.397                  -0.470
                   Qwen2                      -1.350                  -0.349
searxng       20   Llama                      -0.064                  -0.223
                   Qwen2                      -0.655                  -0.284
              50   Llama                       0.089                   0.056
                   Qwen2                      -0.188                   0.022

========================================================================================
6. SOURCE-BIAS ATTENUATION — T7 + T_llms_txt with full CIs
========================================================================================
    variant    outcome        treatment    coef     se  p_val  ci_low  ci_high
     biased rank_delta T7_source_earned -2.2416 0.1169    0.0 -2.4708  -2.0125
     biased  post_rank T7_source_earned  1.4373 0.0662    0.0  1.3076   1.5671
    neutral rank_delta T7_source_earned -0.5911 0.0371    0.0 -0.6639  -0.5183
    neutral  post_rank T7_source_earned -0.3236 0.0344    0.0 -0.3910  -0.2562
 biased_rag rank_delta T7_source_earned -1.9313 0.1409    0.0 -2.2075  -1.6551
 biased_rag  post_rank T7_source_earned  1.1921 0.0844    0.0  1.0267   1.3575
neutral_rag rank_delta T7_source_earned -0.5147 0.0358    0.0 -0.5848  -0.4446
neutral_rag  post_rank T7_source_earned -0.3562 0.0390    0.0 -0.4326  -0.2798

  RAG attenuation (rag − non_rag) with combined SE:
  [rank_delta] biased_rag − biased  T7_source_earned: Δ=+0.310 (SE=0.183)  = 13.8% of non-RAG magnitude  p=0.0901 ·
  [rank_delta] neutral_rag − neutral  T7_source_earned: Δ=+0.076 (SE=0.052)  = 12.9% of non-RAG magnitude  p=0.1383 
  [ post_rank] biased_rag − biased  T7_source_earned: Δ=-0.245 (SE=0.107)  = 17.1% of non-RAG magnitude  p=0.0222 *
  [ post_rank] neutral_rag − neutral  T7_source_earned: Δ=-0.033 (SE=0.052)  = 10.1% of non-RAG magnitude  p=0.5304 

========================================================================================
7. PER-DOMAIN RAG EFFECT — which sources benefit, which lose
========================================================================================
  For each domain, mean(post_rank) under RAG vs non-RAG (lower = better).

  ── biased vs biased_rag  (domains with ≥5 rows in each) ──
  n_domains qualifying: 3836
  earned-media domains in subset: 54

  Mean Δ_post_rank by source class:
             mean    std  count
non-earned  0.183  1.098   3782
earned     -0.096  1.160     54

  TOP 10 RAG-HELPED domains (lower post_rank = better):
                          post_nr  post_rag  Δ_post_rank (rag−nr)  n_nr  n_rag  earned
domain                                                                                
scalable.capital             8.60      2.73                 -5.87    10     11       0
illumina-interactive.com     6.86      2.33                 -4.52     7      6       0
circlecare.app               7.00      2.50                 -4.50     6      6       0
shippo.com                   6.94      2.50                 -4.44    17      8       0
medallia.com                 6.91      2.50                 -4.41    11      8       0
visier.com                   7.44      3.09                 -4.35     9     11       0
teamviewer.com               6.00      1.67                 -4.33    12      6       0
n8n.io                       9.43      5.20                 -4.23     7      5       0
modula.us                    7.25      3.17                 -4.08     8      6       0
brother.com                  6.67      2.60                 -4.07     6      5       0

  TOP 10 RAG-HURT domains:
                              post_nr  post_rag  Δ_post_rank (rag−nr)  n_nr  n_rag  earned
domain                                                                                    
fuzen.io                         2.67      6.86                  4.19     6      7       0
nextagency.com                   2.83      7.25                  4.42     6      8       0
surecart.com                     4.44      9.20                  4.76     9      5       0
jmcinc.com                       4.71      9.56                  4.84     7      9       0
aaptiv.com                       3.43      8.50                  5.07     7      6       0
digitallearninginstitute.com     2.43      7.57                  5.14     7      7       0
adaptiveinsights.com             3.25      8.50                  5.25     8      6       0
invsify.com                      1.26      7.40                  6.14    23      5       0
easygenerator.com                1.14      7.38                  6.24     7     13       0
sageintacct.com                  1.50      8.25                  6.75     6     12       0

  ── neutral vs neutral_rag  (domains with ≥5 rows in each) ──
  n_domains qualifying: 4311
  earned-media domains in subset: 108

  Mean Δ_post_rank by source class:
             mean    std  count
non-earned -0.023  1.005   4203
earned     -0.089  0.664    108

  TOP 10 RAG-HELPED domains (lower post_rank = better):
                         post_nr  post_rag  Δ_post_rank (rag−nr)  n_nr  n_rag  earned
domain                                                                               
checkalle.de                8.50      2.00                 -6.50     8      8       0
verivox.de                  7.50      1.00                 -6.50     8      8       0
snapsoft.de                 7.50      1.50                 -6.00     8      8       0
jstor.org                   8.00      2.12                 -5.88     8      8       0
support.microsoft.com       7.00      1.17                 -5.83     8      6       0
spocket.com                 8.43      3.00                 -5.43     7      5       0
resolve.ai                  6.00      1.00                 -5.00     8      8       0
thecfoshowpodcast.com       7.83      3.00                 -4.83     6      8       0
betterhealth.vic.gov.au     9.50      5.00                 -4.50     8      8       0
commission.europa.eu        6.67      2.50                 -4.17     6      8       0

  TOP 10 RAG-HURT domains:
                                  post_nr  post_rag  Δ_post_rank (rag−nr)  n_nr  n_rag  earned
domain                                                                                        
canvas.instructure.com               3.29      7.10                  3.81     7     10       0
higher-education-marketing.com       2.17      6.00                  3.83     6      7       0
gingertiger.net                      5.12      9.00                  3.88     8      8       0
volusion.com                         1.00      4.94                  3.94    12     18       0
exoplatform.com                      4.00      8.00                  4.00     6      8       0
coinflip.tech                        4.50      9.00                  4.50     8      8       0
ehpconsultinggroup.com               4.00      9.00                  5.00     6      8       0
bmc.com                              1.00      6.89                  5.89     8      9       0
enterprise-softwaresolutions.com     4.00     10.00                  6.00     8      8       0
ethereumclassic.com                  3.00      9.38                  6.38     6      8       0


========================================================================================
END
========================================================================================
