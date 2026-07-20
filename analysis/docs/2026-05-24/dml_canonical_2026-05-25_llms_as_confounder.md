# Canonical DML re-run — 2026-05-25T03:47:13.499538+00:00

Canonical treatments (6): ['treat_stats_density', 'treat_question_headings', 'treat_structured_data', 'T4_citation_authority_code', 'treat_topical_comp', 'treat_freshness']
Canonical confounders (29): ['conf_word_count', 'conf_readability', 'conf_internal_links', 'conf_outbound_links', 'conf_images_alt', 'conf_https', 'conf_title_has_kw', 'conf_title_len', 'conf_snippet_len', 'conf_serp_position', 'conf_title_kw_sim', 'conf_snippet_kw_sim', 'conf_bm25', 'conf_domain_authority', 'conf_backlinks', 'conf_referring_domains', 'conf_brand_recog', 'conf_dfs_paid_count', 'conf_dfs_etv', 'conf_dfs_domain_age_years', 'dfs_keyword_difficulty', 'dfs_search_volume', 'dfs_cpc', 'dfs_competition', 'dfs_intent_commercial', 'dfs_intent_informational', 'dfs_intent_navigational', 'dfs_intent_transactional', 'has_llms_txt']

========================================================================================
Y_1 = selected_by_llm  (binary admission — full SERP pool sample frame)
========================================================================================
  [build] loading 4 SERP pool files …
  [build] base pool rows = 49,364  unique (kw,url) = 28,331
  [build] expanded pool×model×variant rows = 394,912
  [build] marking `selected` from per-variant LLM outputs …
  [build] overall selection rate = 58.37%
  [build] biased_rag: dropping 0 rows (no RAG output)
  [build] neutral_rag: dropping 0 rows (no RAG output)
  [build] post-RAG-filter rows = 394,912  selection rate = 58.37%
  [build] joining treatments + confounders from per-variant parquets …
  [build] pool with features = 394,912 rows × 35 cols
  pool ready: 394,912 rows  selection rate = 58.37%

  [selected] Spec A — single treatment + confounders only
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0005 se=0.0003 p=0.1130  (25.3s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=+0.0118 se=0.0029 p=0.0000***  (29.0s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=-0.0260 se=0.0027 p=0.0000***  (36.0s)
    [A/POOLED        ] T4_citation_authority_code     n=185176 coef=+0.0009 se=0.0007 p=0.2155  (22.5s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=+0.0221 se=0.0094 p=0.0184*  (27.7s)
    [A/POOLED        ] treat_freshness                n=200000 coef=-0.0061 se=0.0007 p=0.0000***  (26.8s)
    [A/VAR:biased    ] treat_stats_density            n= 55430 coef=-0.0011 se=0.0005 p=0.0267*  (11.4s)
    [A/VAR:biased    ] treat_question_headings        n= 57964 coef=+0.0154 se=0.0053 p=0.0038**  (11.7s)
    [A/VAR:biased    ] treat_structured_data          n= 57964 coef=-0.0370 se=0.0051 p=0.0000***  (13.1s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 46294 coef=-0.0017 se=0.0011 p=0.1357  (11.7s)
    [A/VAR:biased    ] treat_topical_comp             n= 51098 coef=+0.0117 se=0.0186 p=0.5306  (13.0s)
    [A/VAR:biased    ] treat_freshness                n= 57964 coef=-0.0121 se=0.0014 p=0.0000***  (12.0s)
    [A/VAR:neutral   ] treat_stats_density            n= 55430 coef=+0.0004 se=0.0005 p=0.4382  (10.9s)
    [A/VAR:neutral   ] treat_question_headings        n= 57964 coef=+0.0024 se=0.0046 p=0.6039  (12.5s)
    [A/VAR:neutral   ] treat_structured_data          n= 57964 coef=-0.0084 se=0.0044 p=0.0561·  (13.8s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 46294 coef=+0.0002 se=0.0009 p=0.7929  (11.0s)
    [A/VAR:neutral   ] treat_topical_comp             n= 51098 coef=+0.0340 se=0.0157 p=0.0301*  (10.5s)
    [A/VAR:neutral   ] treat_freshness                n= 57964 coef=+0.0013 se=0.0012 p=0.2981  (12.0s)
    [A/VAR:biased_rag] treat_stats_density            n= 55430 coef=-0.0014 se=0.0005 p=0.0067**  (13.8s)
    [A/VAR:biased_rag] treat_question_headings        n= 57964 coef=+0.0172 se=0.0054 p=0.0016**  (14.3s)
    [A/VAR:biased_rag] treat_structured_data          n= 57964 coef=-0.0460 se=0.0052 p=0.0000***  (12.2s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 46294 coef=+0.0028 se=0.0012 p=0.0146*  (10.8s)
    [A/VAR:biased_rag] treat_topical_comp             n= 51098 coef=+0.0033 se=0.0187 p=0.8614  (10.8s)
    [A/VAR:biased_rag] treat_freshness                n= 57964 coef=-0.0156 se=0.0014 p=0.0000***  (14.2s)
    [A/VAR:neutral_rag] treat_stats_density            n= 55430 coef=+0.0006 se=0.0004 p=0.0924·  (14.3s)
    [A/VAR:neutral_rag] treat_question_headings        n= 57964 coef=+0.0091 se=0.0046 p=0.0465*  (13.5s)
    [A/VAR:neutral_rag] treat_structured_data          n= 57964 coef=+0.0030 se=0.0044 p=0.4959  (13.9s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 46294 coef=+0.0005 se=0.0008 p=0.5623  (11.2s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 51098 coef=+0.0389 se=0.0154 p=0.0117*  (13.6s)
    [A/VAR:neutral_rag] treat_freshness                n= 57964 coef=+0.0032 se=0.0012 p=0.0071**  (15.9s)
    [A/ENG:ddg       ] treat_stats_density            n= 87008 coef=-0.0002 se=0.0005 p=0.7174  (17.8s)
    [A/ENG:ddg       ] treat_question_headings        n= 89160 coef=+0.0121 se=0.0050 p=0.0164*  (18.1s)
    [A/ENG:ddg       ] treat_structured_data          n= 89160 coef=-0.0264 se=0.0049 p=0.0000***  (14.3s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 87984 coef=+0.0017 se=0.0008 p=0.0362*  (19.3s)
    [A/ENG:ddg       ] treat_topical_comp             n= 52328 coef=+0.0355 se=0.0240 p=0.1395  (12.6s)
    [A/ENG:ddg       ] treat_freshness                n= 89160 coef=-0.0026 se=0.0013 p=0.0488*  (16.6s)
    [A/ENG:searxng   ] treat_stats_density            n=134712 coef=-0.0006 se=0.0004 p=0.1283  (27.2s)
    [A/ENG:searxng   ] treat_question_headings        n=142696 coef=+0.0084 se=0.0034 p=0.0148*  (29.3s)
    [A/ENG:searxng   ] treat_structured_data          n=142696 coef=-0.0265 se=0.0032 p=0.0000***  (20.9s)
    [A/ENG:searxng   ] T4_citation_authority_code     n= 97192 coef=-0.0013 se=0.0012 p=0.2639  (20.8s)
    [A/ENG:searxng   ] treat_topical_comp             n=152064 coef=+0.0096 se=0.0109 p=0.3789  (29.0s)
    [A/ENG:searxng   ] treat_freshness                n=142696 coef=-0.0098 se=0.0009 p=0.0000***  (21.6s)
    [A/MOD:Llama     ] treat_stats_density            n=110860 coef=-0.0009 se=0.0004 p=0.0157*  (23.0s)
    [A/MOD:Llama     ] treat_question_headings        n=115928 coef=+0.0143 se=0.0037 p=0.0001***  (19.6s)
    [A/MOD:Llama     ] treat_structured_data          n=115928 coef=-0.0244 se=0.0035 p=0.0000***  (25.6s)
    [A/MOD:Llama     ] T4_citation_authority_code     n= 92588 coef=+0.0007 se=0.0009 p=0.3891  (16.7s)
    [A/MOD:Llama     ] treat_topical_comp             n=102196 coef=+0.0345 se=0.0128 p=0.0069**  (23.4s)
    [A/MOD:Llama     ] treat_freshness                n=115928 coef=-0.0056 se=0.0010 p=0.0000***  (23.9s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=110860 coef=+0.0001 se=0.0004 p=0.7099  (24.9s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=115928 coef=+0.0089 se=0.0038 p=0.0178*  (24.9s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=115928 coef=-0.0259 se=0.0036 p=0.0000***  (23.9s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n= 92588 coef=+0.0012 se=0.0008 p=0.1531  (18.4s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=102196 coef=+0.0132 se=0.0130 p=0.3078  (20.5s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=115928 coef=-0.0069 se=0.0010 p=0.0000***  (34.8s)
    [A/POOL:20       ] treat_stats_density            n=102896 coef=-0.0010 se=0.0005 p=0.0284*  (16.6s)
    [A/POOL:20       ] treat_question_headings        n=107552 coef=+0.0101 se=0.0041 p=0.0137*  (26.0s)
    [A/POOL:20       ] treat_structured_data          n=107552 coef=-0.0352 se=0.0038 p=0.0000***  (19.5s)
    [A/POOL:20       ] T4_citation_authority_code     n= 95016 coef=-0.0001 se=0.0009 p=0.8783  (21.1s)
    [A/POOL:20       ] treat_topical_comp             n= 93456 coef=+0.0230 se=0.0150 p=0.1244  (18.1s)
    [A/POOL:20       ] treat_freshness                n=107552 coef=-0.0091 se=0.0010 p=0.0000***  (23.6s)
    [A/POOL:50       ] treat_stats_density            n=118824 coef=-0.0003 se=0.0004 p=0.5000  (22.8s)
    [A/POOL:50       ] treat_question_headings        n=124304 coef=+0.0108 se=0.0037 p=0.0035**  (21.2s)
    [A/POOL:50       ] treat_structured_data          n=124304 coef=-0.0162 se=0.0035 p=0.0000***  (25.8s)
    [A/POOL:50       ] T4_citation_authority_code     n= 90160 coef=+0.0020 se=0.0009 p=0.0306*  (18.3s)
    [A/POOL:50       ] treat_topical_comp             n=110936 coef=+0.0247 se=0.0123 p=0.0446*  (34.8s)
    [A/POOL:50       ] treat_freshness                n=124304 coef=-0.0046 se=0.0010 p=0.0000***  (19.7s)

  [selected] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0005 se=0.0003 p=0.1231  (38.6s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=+0.0156 se=0.0029 p=0.0000***  (35.3s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=-0.0141 se=0.0031 p=0.0000***  (34.3s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] T4_citation_authority_code     n=185176 coef=+0.0009 se=0.0007 p=0.1861  (35.9s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=+0.0366 se=0.0095 p=0.0001***  (49.1s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=-0.0047 se=0.0008 p=0.0000***  (37.0s)  (controls: 6 other T + 34 X)

========================================================================================
Y_2 = rank_delta & Y_3 = rank_post  (admitted-URL sample frame)
========================================================================================
  admitted sample = 431,856 rows × 58 cols
  admitted ready: 431,856 rows

  [rank_delta] Spec A — single treatment + confounders only
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0042 se=0.0026 p=0.1106  (31.8s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=+0.1173 se=0.0240 p=0.0000***  (33.1s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=-0.1220 se=0.0217 p=0.0000***  (28.1s)
    [A/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0233 se=0.0084 p=0.0054**  (34.3s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=-0.4886 se=0.0738 p=0.0000***  (36.0s)
    [A/POOLED        ] treat_freshness                n=200000 coef=-0.0542 se=0.0060 p=0.0000***  (36.9s)
    [A/VAR:biased    ] treat_stats_density            n= 54683 coef=+0.0030 se=0.0042 p=0.4742  (13.8s)
    [A/VAR:biased    ] treat_question_headings        n= 56183 coef=+0.0679 se=0.0511 p=0.1838  (13.2s)
    [A/VAR:biased    ] treat_structured_data          n= 56183 coef=-0.0953 se=0.0468 p=0.0418*  (13.6s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 47616 coef=-0.0464 se=0.0115 p=0.0001***  (11.3s)
    [A/VAR:biased    ] treat_topical_comp             n= 46770 coef=-0.6114 se=0.1736 p=0.0004***  (11.3s)
    [A/VAR:biased    ] treat_freshness                n= 56183 coef=-0.0703 se=0.0129 p=0.0000***  (18.0s)
    [A/VAR:neutral   ] treat_stats_density            n= 83750 coef=-0.0014 se=0.0032 p=0.6554  (17.7s)
    [A/VAR:neutral   ] treat_question_headings        n= 87771 coef=+0.1098 se=0.0322 p=0.0006***  (16.3s)
    [A/VAR:neutral   ] treat_structured_data          n= 87771 coef=-0.1413 se=0.0298 p=0.0000***  (17.4s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 65155 coef=-0.0046 se=0.0093 p=0.6178  (12.5s)
    [A/VAR:neutral   ] treat_topical_comp             n= 81896 coef=-0.3708 se=0.1007 p=0.0002***  (14.9s)
    [A/VAR:neutral   ] treat_freshness                n= 87771 coef=-0.0661 se=0.0082 p=0.0000***  (19.8s)
    [A/VAR:biased_rag] treat_stats_density            n= 47640 coef=+0.0008 se=0.0045 p=0.8598  (13.0s)
    [A/VAR:biased_rag] treat_question_headings        n= 48952 coef=+0.1572 se=0.0552 p=0.0044**  (10.0s)
    [A/VAR:biased_rag] treat_structured_data          n= 48952 coef=-0.1510 se=0.0486 p=0.0019**  (18.1s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 43443 coef=-0.0288 se=0.0117 p=0.0142*  (19.9s)
    [A/VAR:biased_rag] treat_topical_comp             n= 39092 coef=-0.3282 se=0.1908 p=0.0854·  (9.5s)
    [A/VAR:biased_rag] treat_freshness                n= 48952 coef=-0.0573 se=0.0137 p=0.0000***  (10.2s)
    [A/VAR:neutral_rag] treat_stats_density            n= 64804 coef=-0.0009 se=0.0027 p=0.7357  (11.9s)
    [A/VAR:neutral_rag] treat_question_headings        n= 67909 coef=+0.1699 se=0.0322 p=0.0000***  (18.3s)
    [A/VAR:neutral_rag] treat_structured_data          n= 67909 coef=-0.1003 se=0.0294 p=0.0006***  (22.0s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 53281 coef=-0.0008 se=0.0076 p=0.9203  (13.6s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 59333 coef=-0.3665 se=0.1028 p=0.0004***  (13.9s)
    [A/VAR:neutral_rag] treat_freshness                n= 67909 coef=-0.0440 se=0.0080 p=0.0000***  (13.7s)
    [A/ENG:ddg       ] treat_stats_density            n= 86258 coef=+0.0012 se=0.0048 p=0.8074  (26.2s)
    [A/ENG:ddg       ] treat_question_headings        n= 88321 coef=-0.0219 se=0.0460 p=0.6337  (14.7s)
    [A/ENG:ddg       ] treat_structured_data          n= 88321 coef=-0.2546 se=0.0425 p=0.0000***  (13.4s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 82765 coef=-0.0024 se=0.0110 p=0.8284  (22.6s)
    [A/ENG:ddg       ] treat_topical_comp             n= 47884 coef=-1.4154 se=0.2259 p=0.0000***  (16.3s)
    [A/ENG:ddg       ] treat_freshness                n= 88321 coef=-0.1099 se=0.0112 p=0.0000***  (13.4s)
    [A/ENG:searxng   ] treat_stats_density            n=164619 coef=-0.0013 se=0.0028 p=0.6506  (29.1s)
    [A/ENG:searxng   ] treat_question_headings        n=172494 coef=+0.2293 se=0.0239 p=0.0000***  (23.3s)
    [A/ENG:searxng   ] treat_structured_data          n=172494 coef=-0.0457 se=0.0216 p=0.0342*  (27.8s)
    [A/ENG:searxng   ] T4_citation_authority_code     n=126730 coef=-0.0530 se=0.0105 p=0.0000***  (20.2s)
    [A/ENG:searxng   ] treat_topical_comp             n=179207 coef=-0.1701 se=0.0739 p=0.0214*  (30.1s)
    [A/ENG:searxng   ] treat_freshness                n=172494 coef=-0.0302 se=0.0061 p=0.0000***  (27.5s)
    [A/MOD:Llama     ] treat_stats_density            n=124403 coef=+0.0036 se=0.0030 p=0.2334  (25.7s)
    [A/MOD:Llama     ] treat_question_headings        n=129382 coef=+0.0918 se=0.0312 p=0.0032**  (22.8s)
    [A/MOD:Llama     ] treat_structured_data          n=129382 coef=-0.1425 se=0.0282 p=0.0000***  (20.6s)
    [A/MOD:Llama     ] T4_citation_authority_code     n=104381 coef=-0.0168 se=0.0107 p=0.1156  (20.9s)
    [A/MOD:Llama     ] treat_topical_comp             n=111772 coef=-0.6311 se=0.1005 p=0.0000***  (20.4s)
    [A/MOD:Llama     ] treat_freshness                n=129382 coef=-0.0651 se=0.0077 p=0.0000***  (25.6s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=126474 coef=-0.0057 se=0.0030 p=0.0539·  (18.0s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=131433 coef=+0.1348 se=0.0277 p=0.0000***  (30.7s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=131433 coef=-0.1245 se=0.0254 p=0.0000***  (27.8s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n=105114 coef=-0.0310 se=0.0075 p=0.0000***  (35.7s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=115319 coef=-0.2897 se=0.0903 p=0.0013**  (16.9s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=131433 coef=-0.0546 se=0.0070 p=0.0000***  (30.3s)
    [A/POOL:20       ] treat_stats_density            n=112637 coef=-0.0073 se=0.0032 p=0.0229*  (18.4s)
    [A/POOL:20       ] treat_question_headings        n=116981 coef=+0.2197 se=0.0296 p=0.0000***  (21.0s)
    [A/POOL:20       ] treat_structured_data          n=116981 coef=-0.0045 se=0.0262 p=0.8625  (22.0s)
    [A/POOL:20       ] T4_citation_authority_code     n=105269 coef=-0.0258 se=0.0076 p=0.0006***  (17.6s)
    [A/POOL:20       ] treat_topical_comp             n= 99674 coef=+0.0114 se=0.1036 p=0.9121  (22.6s)
    [A/POOL:20       ] treat_freshness                n=116981 coef=-0.0214 se=0.0074 p=0.0036**  (17.0s)
    [A/POOL:50       ] treat_stats_density            n=138240 coef=+0.0046 se=0.0035 p=0.1904  (22.3s)
    [A/POOL:50       ] treat_question_headings        n=143834 coef=+0.0581 se=0.0299 p=0.0522·  (21.3s)
    [A/POOL:50       ] treat_structured_data          n=143834 coef=-0.1975 se=0.0273 p=0.0000***  (22.5s)
    [A/POOL:50       ] T4_citation_authority_code     n=104226 coef=-0.0202 se=0.0156 p=0.1936  (22.1s)
    [A/POOL:50       ] treat_topical_comp             n=127417 coef=-0.7318 se=0.0957 p=0.0000***  (17.9s)
    [A/POOL:50       ] treat_freshness                n=143834 coef=-0.0853 se=0.0074 p=0.0000***  (32.1s)

  [rank_delta] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0027 se=0.0027 p=0.3152  (39.2s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=+0.1359 se=0.0245 p=0.0000***  (41.7s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=-0.0506 se=0.0245 p=0.0392*  (46.2s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0234 se=0.0087 p=0.0074**  (44.0s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=-0.5303 se=0.0747 p=0.0000***  (37.7s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=-0.0608 se=0.0066 p=0.0000***  (35.5s)  (controls: 6 other T + 34 X)

  [post_rank] Spec A — single treatment + confounders only
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0024 se=0.0019 p=0.1918  (38.8s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=-0.0480 se=0.0184 p=0.0089**  (36.2s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=+0.1225 se=0.0173 p=0.0000***  (33.6s)
    [A/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0184 se=0.0064 p=0.0041**  (71.4s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=-0.2141 se=0.0594 p=0.0003***  (44.2s)
    [A/POOLED        ] treat_freshness                n=200000 coef=+0.0150 se=0.0048 p=0.0017**  (39.8s)
    [A/VAR:biased    ] treat_stats_density            n= 64977 coef=-0.0008 se=0.0026 p=0.7425  (15.3s)
    [A/VAR:biased    ] treat_question_headings        n= 67034 coef=+0.0225 se=0.0300 p=0.4539  (18.6s)
    [A/VAR:biased    ] treat_structured_data          n= 67034 coef=+0.1258 se=0.0282 p=0.0000***  (15.9s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 47616 coef=+0.0050 se=0.0083 p=0.5520  (13.6s)
    [A/VAR:biased    ] treat_topical_comp             n= 53449 coef=-0.1205 se=0.1073 p=0.2615  (18.1s)
    [A/VAR:biased    ] treat_freshness                n= 67034 coef=+0.0381 se=0.0078 p=0.0000***  (24.8s)
    [A/VAR:neutral   ] treat_stats_density            n= 85958 coef=-0.0016 se=0.0026 p=0.5227  (38.0s)
    [A/VAR:neutral   ] treat_question_headings        n= 90099 coef=-0.0401 se=0.0274 p=0.1438  (23.9s)
    [A/VAR:neutral   ] treat_structured_data          n= 90099 coef=+0.1145 se=0.0263 p=0.0000***  (28.4s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 65155 coef=-0.0118 se=0.0077 p=0.1236  (32.7s)
    [A/VAR:neutral   ] treat_topical_comp             n= 83113 coef=-0.1332 se=0.0903 p=0.1400  (47.6s)
    [A/VAR:neutral   ] treat_freshness                n= 90099 coef=+0.0159 se=0.0072 p=0.0272*  (24.3s)
    [A/VAR:biased_rag] treat_stats_density            n= 60283 coef=-0.0057 se=0.0029 p=0.0481*  (30.4s)
    [A/VAR:biased_rag] treat_question_headings        n= 62162 coef=-0.0707 se=0.0319 p=0.0268*  (20.4s)
    [A/VAR:biased_rag] treat_structured_data          n= 62162 coef=+0.1112 se=0.0301 p=0.0002***  (19.3s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 43443 coef=-0.0010 se=0.0087 p=0.9058  (32.3s)
    [A/VAR:biased_rag] treat_topical_comp             n= 46890 coef=-0.3582 se=0.1169 p=0.0022**  (14.4s)
    [A/VAR:biased_rag] treat_freshness                n= 62162 coef=+0.0207 se=0.0083 p=0.0128*  (17.3s)
    [A/VAR:neutral_rag] treat_stats_density            n= 66908 coef=-0.0004 se=0.0026 p=0.8873  (25.0s)
    [A/VAR:neutral_rag] treat_question_headings        n= 70121 coef=-0.0538 se=0.0308 p=0.0809·  (21.1s)
    [A/VAR:neutral_rag] treat_structured_data          n= 70121 coef=+0.1429 se=0.0291 p=0.0000***  (17.0s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 53281 coef=-0.0144 se=0.0079 p=0.0668·  (30.3s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 60241 coef=-0.3077 se=0.1044 p=0.0032**  (19.6s)
    [A/VAR:neutral_rag] treat_freshness                n= 70121 coef=+0.0026 se=0.0080 p=0.7423  (22.0s)
    [A/ENG:ddg       ] treat_stats_density            n= 99103 coef=+0.0048 se=0.0028 p=0.0874·  (19.6s)
    [A/ENG:ddg       ] treat_question_headings        n=101894 coef=-0.0021 se=0.0288 p=0.9424  (22.5s)
    [A/ENG:ddg       ] treat_structured_data          n=101894 coef=+0.0401 se=0.0274 p=0.1434  (32.1s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 82765 coef=-0.0052 se=0.0076 p=0.4967  (18.6s)
    [A/ENG:ddg       ] treat_topical_comp             n= 50082 coef=-0.5142 se=0.1426 p=0.0003***  (21.4s)
    [A/ENG:ddg       ] treat_freshness                n=101894 coef=-0.0211 se=0.0073 p=0.0040**  (25.9s)
    [A/ENG:searxng   ] treat_stats_density            n=179023 coef=-0.0091 se=0.0024 p=0.0001***  (47.2s)
    [A/ENG:searxng   ] treat_question_headings        n=187522 coef=-0.0198 se=0.0195 p=0.3109  (36.2s)
    [A/ENG:searxng   ] treat_structured_data          n=187522 coef=+0.1487 se=0.0181 p=0.0000***  (45.3s)
    [A/ENG:searxng   ] T4_citation_authority_code     n=126730 coef=-0.0370 se=0.0114 p=0.0012**  (29.0s)
    [A/ENG:searxng   ] treat_topical_comp             n=193611 coef=-0.1502 se=0.0613 p=0.0142*  (50.2s)
    [A/ENG:searxng   ] treat_freshness                n=187522 coef=+0.0452 se=0.0050 p=0.0000***  (40.2s)
    [A/MOD:Llama     ] treat_stats_density            n=138621 coef=-0.0051 se=0.0021 p=0.0133*  (32.9s)
    [A/MOD:Llama     ] treat_question_headings        n=144317 coef=-0.0278 se=0.0215 p=0.1968  (28.6s)
    [A/MOD:Llama     ] treat_structured_data          n=144317 coef=+0.1354 se=0.0202 p=0.0000***  (43.8s)
    [A/MOD:Llama     ] T4_citation_authority_code     n=104381 coef=-0.0064 se=0.0073 p=0.3811  (31.3s)
    [A/MOD:Llama     ] treat_topical_comp             n=120201 coef=-0.1981 se=0.0755 p=0.0087**  (39.8s)
    [A/MOD:Llama     ] treat_freshness                n=144317 coef=+0.0201 se=0.0056 p=0.0003***  (34.0s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=139505 coef=-0.0006 se=0.0021 p=0.7690  (27.7s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=145099 coef=-0.0436 se=0.0213 p=0.0403*  (28.0s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=145099 coef=+0.0968 se=0.0200 p=0.0000***  (30.7s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n=105114 coef=-0.0137 se=0.0078 p=0.0777·  (25.6s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=123492 coef=-0.2278 se=0.0738 p=0.0020**  (21.3s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=145099 coef=+0.0133 se=0.0055 p=0.0163*  (30.6s)
    [A/POOL:20       ] treat_stats_density            n=125051 coef=-0.0017 se=0.0026 p=0.5003  (60.5s)
    [A/POOL:20       ] treat_question_headings        n=129913 coef=-0.0255 se=0.0232 p=0.2722  (25.9s)
    [A/POOL:20       ] treat_structured_data          n=129913 coef=+0.1714 se=0.0217 p=0.0000***  (34.0s)
    [A/POOL:20       ] T4_citation_authority_code     n=105269 coef=-0.0043 se=0.0082 p=0.5982  (26.9s)
    [A/POOL:20       ] treat_topical_comp             n=108269 coef=-0.0559 se=0.0830 p=0.5004  (32.6s)
    [A/POOL:20       ] treat_freshness                n=129913 coef=+0.0366 se=0.0060 p=0.0000***  (34.2s)
    [A/POOL:50       ] treat_stats_density            n=153075 coef=-0.0005 se=0.0022 p=0.8251  (42.6s)
    [A/POOL:50       ] treat_question_headings        n=159503 coef=-0.0182 se=0.0210 p=0.3875  (36.2s)
    [A/POOL:50       ] treat_structured_data          n=159503 coef=+0.0822 se=0.0197 p=0.0000***  (33.1s)
    [A/POOL:50       ] T4_citation_authority_code     n=104226 coef=-0.0160 se=0.0092 p=0.0822·  (45.5s)
    [A/POOL:50       ] treat_topical_comp             n=135424 coef=-0.3308 se=0.0737 p=0.0000***  (26.7s)
    [A/POOL:50       ] treat_freshness                n=159503 coef=+0.0037 se=0.0054 p=0.4970  (27.9s)

  [post_rank] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0015 se=0.0019 p=0.4054  (37.0s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=-0.0408 se=0.0187 p=0.0291*  (60.7s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=+0.0949 se=0.0196 p=0.0000***  (53.9s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0154 se=0.0066 p=0.0191*  (38.5s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=-0.2995 se=0.0601 p=0.0000***  (37.4s)  (controls: 6 other T + 34 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=+0.0051 se=0.0053 p=0.3334  (41.4s)  (controls: 6 other T + 34 X)

  → saved geodml_data/data/dml_results/dml_canonical_2026-05-25_llms_as_confounder.parquet  rows=216

========================================================================================
HEADLINE — Spec B POOLED (mutually-controlled) per outcome
========================================================================================

  Y = selected
                 treatment      n      coef       se        p_val sig  p_bonferroni bonferroni_sig
     treat_structured_data 200000 -0.014102 0.003051 3.801113e-06 ***  2.736801e-04           True
           treat_freshness 200000 -0.004741 0.000822 8.017344e-09 ***  5.772487e-07           True
       treat_stats_density 200000 -0.000477 0.000310 1.231064e-01      1.000000e+00          False
T4_citation_authority_code 185176  0.000924 0.000699 1.861280e-01      1.000000e+00          False
   treat_question_headings 200000  0.015647 0.002932 9.517709e-08 ***  6.852750e-06           True
        treat_topical_comp 200000  0.036639 0.009502 1.152896e-04 ***  8.300849e-03           True

  Y = rank_delta
                 treatment      n      coef       se        p_val sig  p_bonferroni bonferroni_sig
        treat_topical_comp 200000 -0.530291 0.074720 1.274536e-12 ***  9.176659e-11           True
           treat_freshness 200000 -0.060767 0.006599 0.000000e+00 ***  0.000000e+00           True
     treat_structured_data 200000 -0.050583 0.024530 3.919997e-02   *  1.000000e+00          False
T4_citation_authority_code 200000 -0.023416 0.008740 7.380191e-03  **  5.313738e-01          False
       treat_stats_density 200000 -0.002665 0.002653 3.151653e-01      1.000000e+00          False
   treat_question_headings 200000  0.135898 0.024477 2.821690e-08 ***  2.031617e-06           True

  Y = post_rank
                 treatment      n      coef       se        p_val sig  p_bonferroni bonferroni_sig
        treat_topical_comp 200000 -0.299483 0.060052 6.131572e-07 ***      0.000044           True
   treat_question_headings 200000 -0.040754 0.018673 2.907194e-02   *      1.000000          False
T4_citation_authority_code 200000 -0.015407 0.006572 1.905421e-02   *      1.000000          False
       treat_stats_density 200000 -0.001550 0.001863 4.054199e-01          1.000000          False
           treat_freshness 200000  0.005101 0.005274 3.333981e-01          1.000000          False
     treat_structured_data 200000  0.094864 0.019552 1.223086e-06 ***      0.000088           True
