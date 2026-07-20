# Code-only vs LLM-only DML comparison

_Generated: 2026-05-24 16:59:19_


========================================================================================
1. Load
========================================================================================
  loaded regression_dataset.parquet  rows=65,203  cols=73

========================================================================================
2. Run CODE-only DML
========================================================================================

  [CODE] treatments = ['T1_code', 'T2_code', 'T3_code', 'T4_code', 'T5_topical_comp', 'T6_freshness', 'T_llms_txt']  (7 cols)
  [CODE] confounders = 17 cols
    [CODE/rank_delta] T1_code                   n=65203 coef=-0.0110 se=0.0021 p=0.0000  (108.6s)
    [CODE/rank_delta] T2_code                   n=65203 coef=+0.0563 se=0.0225 p=0.0125  (112.2s)
    [CODE/rank_delta] T3_code                   n=65203 coef=+0.1200 se=0.0255 p=0.0000  (112.5s)
    [CODE/rank_delta] T4_code                   n=65203 coef=-0.0156 se=0.0057 p=0.0057  (128.3s)
    [CODE/rank_delta] T5_topical_comp           n=65203 coef=+0.5790 se=0.0859 p=0.0000  (110.1s)
    [CODE/rank_delta] T6_freshness              n=65203 coef=-0.0707 se=0.0060 p=0.0000  (117.7s)
    [CODE/rank_delta] T_llms_txt                n=65203 coef=+0.1026 se=0.0213 p=0.0000  (132.7s)
    [CODE/post_rank] T1_code                   n=65203 coef=+0.0120 se=0.0021 p=0.0000  (109.5s)
    [CODE/post_rank] T2_code                   n=65203 coef=-0.0579 se=0.0224 p=0.0097  (144.2s)
    [CODE/post_rank] T3_code                   n=65203 coef=-0.1213 se=0.0254 p=0.0000  (117.9s)
    [CODE/post_rank] T4_code                   n=65203 coef=+0.0136 se=0.0056 p=0.0150  (108.7s)
    [CODE/post_rank] T5_topical_comp           n=65203 coef=-0.5810 se=0.0857 p=0.0000  (103.5s)
    [CODE/post_rank] T6_freshness              n=65203 coef=+0.0707 se=0.0060 p=0.0000  (113.3s)
    [CODE/post_rank] T_llms_txt                n=65203 coef=-0.1003 se=0.0211 p=0.0000  (116.3s)

========================================================================================
3. Run LLM-only DML
========================================================================================

  [LLM] treatments = ['T1_llm', 'T2_llm', 'T3_llm', 'T4_llm', 'T5_topical_comp', 'T6_freshness', 'T_llms_txt']  (7 cols)
  [LLM] confounders = 17 cols
    [LLM/rank_delta] T1_llm                    n=65203 coef=+0.0001 se=0.0015 p=0.9248  (106.2s)
    [LLM/rank_delta] T2_llm                    n=65203 coef=-0.0555 se=0.0213 p=0.0090  (112.8s)
    [LLM/rank_delta] T3_llm                    n=65203 coef=+0.1034 se=0.0221 p=0.0000  (116.5s)
    [LLM/rank_delta] T4_llm                    n=65203 coef=-0.0210 se=0.0055 p=0.0001  (121.8s)
    [LLM/rank_delta] T5_topical_comp           n=65203 coef=+0.6208 se=0.0858 p=0.0000  (108.1s)
    [LLM/rank_delta] T6_freshness              n=65203 coef=-0.0696 se=0.0061 p=0.0000  (115.6s)
    [LLM/rank_delta] T_llms_txt                n=65203 coef=+0.1148 se=0.0213 p=0.0000  (112.9s)
    [LLM/post_rank] T1_llm                    n=65203 coef=+0.0004 se=0.0015 p=0.7824  (112.5s)
    [LLM/post_rank] T2_llm                    n=65203 coef=+0.0449 se=0.0211 p=0.0332  (118.4s)
    [LLM/post_rank] T3_llm                    n=65203 coef=-0.1007 se=0.0219 p=0.0000  (127.1s)
    [LLM/post_rank] T4_llm                    n=65203 coef=+0.0215 se=0.0055 p=0.0001  (123.6s)
    [LLM/post_rank] T5_topical_comp           n=65203 coef=-0.6267 se=0.0856 p=0.0000  (126.7s)
    [LLM/post_rank] T6_freshness              n=65203 coef=+0.0699 se=0.0060 p=0.0000  (112.8s)
    [LLM/post_rank] T_llms_txt                n=65203 coef=-0.1107 se=0.0211 p=0.0000  (120.8s)

  → saved geodml_data/data/dml_results/dml_code_vs_llm.parquet  rows=28

========================================================================================
4. Side-by-side comparison (rank_delta)
========================================================================================

  T1–T4 code vs LLM head-to-head on rank_delta:
family  code_coef  code_se       code_p code_sig  llm_coef   llm_se    llm_p llm_sig  delta_coef agree_sign
    T1  -0.011047 0.002094 1.324765e-07      ***  0.000146 0.001549 0.924752            0.011193         no
    T2   0.056340 0.022550 1.247358e-02        * -0.055504 0.021254 0.009015      **   -0.111843         no
    T3   0.120023 0.025477 2.463531e-06      ***  0.103420 0.022096 0.000003     ***   -0.016604        yes
    T4  -0.015645 0.005656 5.673353e-03       ** -0.021050 0.005526 0.000139     ***   -0.005404        yes

========================================================================================
5. Shared treatments — should give same number across specs
========================================================================================
spec                            CODE       LLM
treatment       outcome                       
T5_topical_comp post_rank  -0.580969 -0.626748
                rank_delta  0.578996  0.620768
T6_freshness    post_rank   0.070750  0.069914
                rank_delta -0.070746 -0.069569
T_llms_txt      post_rank  -0.100345 -0.110664
                rank_delta  0.102633  0.114782
