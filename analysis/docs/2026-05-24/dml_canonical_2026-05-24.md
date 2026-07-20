# Canonical DML re-run — 2026-05-24T09:50:00.669738+00:00

Canonical treatments (7): ['treat_stats_density', 'treat_question_headings', 'treat_structured_data', 'T4_citation_authority_code', 'treat_topical_comp', 'treat_freshness', 'has_llms_txt']
Canonical confounders (28): ['conf_word_count', 'conf_readability', 'conf_internal_links', 'conf_outbound_links', 'conf_images_alt', 'conf_https', 'conf_title_has_kw', 'conf_title_len', 'conf_snippet_len', 'conf_serp_position', 'conf_title_kw_sim', 'conf_snippet_kw_sim', 'conf_bm25', 'conf_domain_authority', 'conf_backlinks', 'conf_referring_domains', 'conf_brand_recog', 'conf_dfs_paid_count', 'conf_dfs_etv', 'conf_dfs_domain_age_years', 'dfs_keyword_difficulty', 'dfs_search_volume', 'dfs_cpc', 'dfs_competition', 'dfs_intent_commercial', 'dfs_intent_informational', 'dfs_intent_navigational', 'dfs_intent_transactional']

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
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0005 se=0.0003 p=0.1411  (32.3s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=+0.0132 se=0.0029 p=0.0000***  (31.2s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=-0.0228 se=0.0027 p=0.0000***  (34.9s)
    [A/POOLED        ] T4_citation_authority_code     n=185176 coef=+0.0009 se=0.0007 p=0.2143  (31.9s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=+0.0249 se=0.0094 p=0.0081**  (32.5s)
    [A/POOLED        ] treat_freshness                n=200000 coef=-0.0053 se=0.0007 p=0.0000***  (46.1s)
    [A/POOLED        ] has_llms_txt                   n=200000 coef=+0.0057 se=0.0028 p=0.0389*  (36.9s)
    [A/VAR:biased    ] treat_stats_density            n= 55430 coef=-0.0010 se=0.0005 p=0.0354*  (14.6s)
    [A/VAR:biased    ] treat_question_headings        n= 57964 coef=+0.0179 se=0.0053 p=0.0008***  (22.4s)
    [A/VAR:biased    ] treat_structured_data          n= 57964 coef=-0.0330 se=0.0051 p=0.0000***  (16.4s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 46294 coef=-0.0015 se=0.0011 p=0.1899  (35.1s)
    [A/VAR:biased    ] treat_topical_comp             n= 51098 coef=+0.0150 se=0.0187 p=0.4226  (17.8s)
    [A/VAR:biased    ] treat_freshness                n= 57964 coef=-0.0114 se=0.0014 p=0.0000***  (16.0s)
    [A/VAR:biased    ] has_llms_txt                   n= 52524 coef=+0.0154 se=0.0055 p=0.0051**  (15.2s)
    [A/VAR:neutral   ] treat_stats_density            n= 55430 coef=+0.0006 se=0.0005 p=0.2188  (14.6s)
    [A/VAR:neutral   ] treat_question_headings        n= 57964 coef=+0.0020 se=0.0046 p=0.6553  (14.6s)
    [A/VAR:neutral   ] treat_structured_data          n= 57964 coef=-0.0080 se=0.0044 p=0.0680·  (18.6s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 46294 coef=+0.0005 se=0.0009 p=0.5909  (16.5s)
    [A/VAR:neutral   ] treat_topical_comp             n= 51098 coef=+0.0341 se=0.0157 p=0.0296*  (14.0s)
    [A/VAR:neutral   ] treat_freshness                n= 57964 coef=+0.0017 se=0.0012 p=0.1697  (15.9s)
    [A/VAR:neutral   ] has_llms_txt                   n= 52524 coef=+0.0042 se=0.0045 p=0.3420  (14.5s)
    [A/VAR:biased_rag] treat_stats_density            n= 55430 coef=-0.0015 se=0.0005 p=0.0046**  (15.8s)
    [A/VAR:biased_rag] treat_question_headings        n= 57964 coef=+0.0185 se=0.0055 p=0.0007***  (15.8s)
    [A/VAR:biased_rag] treat_structured_data          n= 57964 coef=-0.0427 se=0.0052 p=0.0000***  (14.6s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 46294 coef=+0.0029 se=0.0012 p=0.0123*  (15.1s)
    [A/VAR:biased_rag] treat_topical_comp             n= 51098 coef=+0.0064 se=0.0188 p=0.7335  (15.3s)
    [A/VAR:biased_rag] treat_freshness                n= 57964 coef=-0.0154 se=0.0014 p=0.0000***  (15.4s)
    [A/VAR:biased_rag] has_llms_txt                   n= 52524 coef=-0.0006 se=0.0058 p=0.9195  (15.3s)
    [A/VAR:neutral_rag] treat_stats_density            n= 55430 coef=+0.0007 se=0.0004 p=0.0597·  (14.1s)
    [A/VAR:neutral_rag] treat_question_headings        n= 57964 coef=+0.0094 se=0.0046 p=0.0394*  (15.6s)
    [A/VAR:neutral_rag] treat_structured_data          n= 57964 coef=+0.0040 se=0.0044 p=0.3615  (15.0s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 46294 coef=+0.0007 se=0.0008 p=0.4232  (14.6s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 51098 coef=+0.0407 se=0.0155 p=0.0085**  (17.0s)
    [A/VAR:neutral_rag] treat_freshness                n= 57964 coef=+0.0035 se=0.0012 p=0.0033**  (14.5s)
    [A/VAR:neutral_rag] has_llms_txt                   n= 52524 coef=-0.0019 se=0.0046 p=0.6758  (13.4s)
    [A/ENG:ddg       ] treat_stats_density            n= 87008 coef=-0.0000 se=0.0005 p=0.9240  (17.4s)
    [A/ENG:ddg       ] treat_question_headings        n= 89160 coef=+0.0131 se=0.0050 p=0.0092**  (17.2s)
    [A/ENG:ddg       ] treat_structured_data          n= 89160 coef=-0.0237 se=0.0049 p=0.0000***  (18.1s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 87984 coef=+0.0014 se=0.0008 p=0.0794·  (17.1s)
    [A/ENG:ddg       ] treat_topical_comp             n= 52328 coef=+0.0494 se=0.0241 p=0.0401*  (13.8s)
    [A/ENG:ddg       ] treat_freshness                n= 89160 coef=-0.0018 se=0.0013 p=0.1736  (17.8s)
    [A/ENG:ddg       ] has_llms_txt                   n=100912 coef=-0.0028 se=0.0042 p=0.4989  (19.6s)
    [A/ENG:searxng   ] treat_stats_density            n=134712 coef=-0.0006 se=0.0004 p=0.1540  (24.0s)
    [A/ENG:searxng   ] treat_question_headings        n=142696 coef=+0.0110 se=0.0034 p=0.0014**  (22.9s)
    [A/ENG:searxng   ] treat_structured_data          n=142696 coef=-0.0245 se=0.0032 p=0.0000***  (22.3s)
    [A/ENG:searxng   ] T4_citation_authority_code     n= 97192 coef=-0.0019 se=0.0012 p=0.1193  (18.7s)
    [A/ENG:searxng   ] treat_topical_comp             n=152064 coef=+0.0116 se=0.0108 p=0.2831  (28.2s)
    [A/ENG:searxng   ] treat_freshness                n=142696 coef=-0.0092 se=0.0009 p=0.0000***  (23.6s)
    [A/ENG:searxng   ] has_llms_txt                   n=109184 coef=+0.0106 se=0.0038 p=0.0059**  (47.6s)
    [A/MOD:Llama     ] treat_stats_density            n=110860 coef=-0.0009 se=0.0004 p=0.0180*  (25.9s)
    [A/MOD:Llama     ] treat_question_headings        n=115928 coef=+0.0153 se=0.0037 p=0.0000***  (24.8s)
    [A/MOD:Llama     ] treat_structured_data          n=115928 coef=-0.0209 se=0.0035 p=0.0000***  (24.9s)
    [A/MOD:Llama     ] T4_citation_authority_code     n= 92588 coef=+0.0005 se=0.0009 p=0.5886  (22.4s)
    [A/MOD:Llama     ] treat_topical_comp             n=102196 coef=+0.0367 se=0.0128 p=0.0042**  (22.4s)
    [A/MOD:Llama     ] treat_freshness                n=115928 coef=-0.0048 se=0.0010 p=0.0000***  (26.3s)
    [A/MOD:Llama     ] has_llms_txt                   n=105048 coef=+0.0069 se=0.0037 p=0.0658·  (22.5s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=110860 coef=+0.0003 se=0.0004 p=0.4866  (24.2s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=115928 coef=+0.0103 se=0.0038 p=0.0060**  (25.5s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=115928 coef=-0.0228 se=0.0036 p=0.0000***  (24.5s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n= 92588 coef=+0.0009 se=0.0008 p=0.2804  (24.2s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=102196 coef=+0.0189 se=0.0130 p=0.1481  (26.3s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=115928 coef=-0.0060 se=0.0010 p=0.0000***  (28.6s)
    [A/MOD:Qwen2.5   ] has_llms_txt                   n=105048 coef=+0.0017 se=0.0038 p=0.6627  (25.4s)
    [A/POOL:20       ] treat_stats_density            n=102896 coef=-0.0007 se=0.0005 p=0.1223  (25.4s)
    [A/POOL:20       ] treat_question_headings        n=107552 coef=+0.0111 se=0.0041 p=0.0064**  (26.4s)
    [A/POOL:20       ] treat_structured_data          n=107552 coef=-0.0317 se=0.0038 p=0.0000***  (32.6s)
    [A/POOL:20       ] T4_citation_authority_code     n= 95016 coef=+0.0001 se=0.0009 p=0.9411  (31.4s)
    [A/POOL:20       ] treat_topical_comp             n= 93456 coef=+0.0280 se=0.0150 p=0.0613·  (18.2s)
    [A/POOL:20       ] treat_freshness                n=107552 coef=-0.0082 se=0.0010 p=0.0000***  (19.6s)
    [A/POOL:20       ] has_llms_txt                   n=108112 coef=+0.0007 se=0.0037 p=0.8490  (20.6s)
    [A/POOL:50       ] treat_stats_density            n=118824 coef=-0.0002 se=0.0004 p=0.6973  (20.6s)
    [A/POOL:50       ] treat_question_headings        n=124304 coef=+0.0112 se=0.0037 p=0.0025**  (21.1s)
    [A/POOL:50       ] treat_structured_data          n=124304 coef=-0.0134 se=0.0035 p=0.0001***  (21.7s)
    [A/POOL:50       ] T4_citation_authority_code     n= 90160 coef=+0.0021 se=0.0009 p=0.0254*  (18.4s)
    [A/POOL:50       ] treat_topical_comp             n=110936 coef=+0.0259 se=0.0123 p=0.0358*  (21.4s)
    [A/POOL:50       ] treat_freshness                n=124304 coef=-0.0041 se=0.0010 p=0.0000***  (20.7s)
    [A/POOL:50       ] has_llms_txt                   n=101984 coef=+0.0058 se=0.0040 p=0.1427  (19.0s)

  [selected] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0004 se=0.0003 p=0.1522  (31.1s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=+0.0162 se=0.0029 p=0.0000***  (30.0s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=-0.0147 se=0.0031 p=0.0000***  (31.7s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] T4_citation_authority_code     n=185176 coef=+0.0009 se=0.0007 p=0.1901  (30.1s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=+0.0358 se=0.0095 p=0.0002***  (37.3s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=-0.0044 se=0.0008 p=0.0000***  (38.0s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] has_llms_txt                   n=200000 coef=+0.0073 se=0.0028 p=0.0093**  (34.5s)  (controls: 6 other T + 33 X)

========================================================================================
Y_2 = rank_delta & Y_3 = rank_post  (admitted-URL sample frame)
========================================================================================
  admitted sample = 431,856 rows × 58 cols
  admitted ready: 431,856 rows

  [rank_delta] Spec A — single treatment + confounders only
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0040 se=0.0026 p=0.1253  (32.9s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=+0.1022 se=0.0240 p=0.0000***  (31.6s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=-0.1215 se=0.0216 p=0.0000***  (29.5s)
    [A/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0240 se=0.0085 p=0.0048**  (30.3s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=-0.4926 se=0.0740 p=0.0000***  (32.4s)
    [A/POOLED        ] treat_freshness                n=200000 coef=-0.0548 se=0.0060 p=0.0000***  (33.0s)
    [A/POOLED        ] has_llms_txt                   n=200000 coef=+0.0445 se=0.0240 p=0.0632·  (31.9s)
    [A/VAR:biased    ] treat_stats_density            n= 54683 coef=+0.0037 se=0.0041 p=0.3657  (15.6s)
    [A/VAR:biased    ] treat_question_headings        n= 56183 coef=+0.0453 se=0.0513 p=0.3765  (15.5s)
    [A/VAR:biased    ] treat_structured_data          n= 56183 coef=-0.1215 se=0.0465 p=0.0090**  (16.1s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 47616 coef=-0.0465 se=0.0115 p=0.0001***  (14.5s)
    [A/VAR:biased    ] treat_topical_comp             n= 46770 coef=-0.6631 se=0.1742 p=0.0001***  (14.6s)
    [A/VAR:biased    ] treat_freshness                n= 56183 coef=-0.0725 se=0.0128 p=0.0000***  (18.1s)
    [A/VAR:biased    ] has_llms_txt                   n= 52054 coef=+0.0238 se=0.0534 p=0.6553  (15.7s)
    [A/VAR:neutral   ] treat_stats_density            n= 83750 coef=-0.0014 se=0.0032 p=0.6581  (19.2s)
    [A/VAR:neutral   ] treat_question_headings        n= 87771 coef=+0.1031 se=0.0321 p=0.0013**  (19.2s)
    [A/VAR:neutral   ] treat_structured_data          n= 87771 coef=-0.1459 se=0.0295 p=0.0000***  (21.4s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 65155 coef=-0.0068 se=0.0092 p=0.4602  (17.1s)
    [A/VAR:neutral   ] treat_topical_comp             n= 81896 coef=-0.3722 se=0.0997 p=0.0002***  (19.1s)
    [A/VAR:neutral   ] treat_freshness                n= 87771 coef=-0.0651 se=0.0081 p=0.0000***  (19.6s)
    [A/VAR:neutral   ] has_llms_txt                   n= 73990 coef=+0.0558 se=0.0333 p=0.0933·  (17.4s)
    [A/VAR:biased_rag] treat_stats_density            n= 47640 coef=+0.0010 se=0.0045 p=0.8313  (12.1s)
    [A/VAR:biased_rag] treat_question_headings        n= 48952 coef=+0.1645 se=0.0551 p=0.0028**  (15.0s)
    [A/VAR:biased_rag] treat_structured_data          n= 48952 coef=-0.1492 se=0.0483 p=0.0020**  (24.5s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 43443 coef=-0.0309 se=0.0118 p=0.0088**  (26.5s)
    [A/VAR:biased_rag] treat_topical_comp             n= 39092 coef=-0.3378 se=0.1917 p=0.0781·  (13.6s)
    [A/VAR:biased_rag] treat_freshness                n= 48952 coef=-0.0600 se=0.0136 p=0.0000***  (19.8s)
    [A/VAR:biased_rag] has_llms_txt                   n= 47190 coef=+0.0305 se=0.0544 p=0.5751  (19.2s)
    [A/VAR:neutral_rag] treat_stats_density            n= 64804 coef=-0.0016 se=0.0028 p=0.5668  (28.7s)
    [A/VAR:neutral_rag] treat_question_headings        n= 67909 coef=+0.1654 se=0.0320 p=0.0000***  (49.0s)
    [A/VAR:neutral_rag] treat_structured_data          n= 67909 coef=-0.0922 se=0.0294 p=0.0017**  (51.4s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 53281 coef=-0.0003 se=0.0076 p=0.9664  (53.3s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 59333 coef=-0.3325 se=0.1030 p=0.0012**  (56.9s)
    [A/VAR:neutral_rag] treat_freshness                n= 67909 coef=-0.0416 se=0.0079 p=0.0000***  (52.8s)
    [A/VAR:neutral_rag] has_llms_txt                   n= 60303 coef=+0.0833 se=0.0316 p=0.0084**  (52.8s)
    [A/ENG:ddg       ] treat_stats_density            n= 86258 coef=+0.0007 se=0.0048 p=0.8900  (58.2s)
    [A/ENG:ddg       ] treat_question_headings        n= 88321 coef=-0.0554 se=0.0461 p=0.2300  (57.9s)
    [A/ENG:ddg       ] treat_structured_data          n= 88321 coef=-0.2587 se=0.0426 p=0.0000***  (61.3s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 82765 coef=-0.0035 se=0.0110 p=0.7501  (58.0s)
    [A/ENG:ddg       ] treat_topical_comp             n= 47884 coef=-1.3878 se=0.2269 p=0.0000***  (52.2s)
    [A/ENG:ddg       ] treat_freshness                n= 88321 coef=-0.1107 se=0.0112 p=0.0000***  (57.7s)
    [A/ENG:ddg       ] has_llms_txt                   n= 93526 coef=-0.0185 se=0.0396 p=0.6408  (57.7s)
    [A/ENG:searxng   ] treat_stats_density            n=164619 coef=-0.0010 se=0.0029 p=0.7286  (61.7s)
    [A/ENG:searxng   ] treat_question_headings        n=172494 coef=+0.2299 se=0.0241 p=0.0000***  (72.6s)
    [A/ENG:searxng   ] treat_structured_data          n=172494 coef=-0.0336 se=0.0215 p=0.1181  (71.4s)
    [A/ENG:searxng   ] T4_citation_authority_code     n=126730 coef=-0.0562 se=0.0104 p=0.0000***  (62.1s)
    [A/ENG:searxng   ] treat_topical_comp             n=179207 coef=-0.1648 se=0.0737 p=0.0253*  (73.8s)
    [A/ENG:searxng   ] treat_freshness                n=172494 coef=-0.0279 se=0.0060 p=0.0000***  (76.7s)
    [A/ENG:searxng   ] has_llms_txt                   n=140011 coef=+0.1018 se=0.0283 p=0.0003***  (65.3s)
    [A/MOD:Llama     ] treat_stats_density            n=124403 coef=+0.0023 se=0.0030 p=0.4371  (62.0s)
    [A/MOD:Llama     ] treat_question_headings        n=129382 coef=+0.0928 se=0.0312 p=0.0029**  (65.0s)
    [A/MOD:Llama     ] treat_structured_data          n=129382 coef=-0.1380 se=0.0282 p=0.0000***  (26.0s)
    [A/MOD:Llama     ] T4_citation_authority_code     n=104381 coef=-0.0181 se=0.0106 p=0.0871·  (21.3s)
    [A/MOD:Llama     ] treat_topical_comp             n=111772 coef=-0.6136 se=0.1005 p=0.0000***  (23.8s)
    [A/MOD:Llama     ] treat_freshness                n=129382 coef=-0.0669 se=0.0077 p=0.0000***  (22.2s)
    [A/MOD:Llama     ] has_llms_txt                   n=116085 coef=+0.0583 se=0.0326 p=0.0743·  (18.7s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=126474 coef=-0.0064 se=0.0030 p=0.0317*  (21.2s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=131433 coef=+0.1212 se=0.0277 p=0.0000***  (20.6s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=131433 coef=-0.1266 se=0.0253 p=0.0000***  (20.9s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n=105114 coef=-0.0311 se=0.0075 p=0.0000***  (17.5s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=115319 coef=-0.2941 se=0.0901 p=0.0011**  (18.9s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=131433 coef=-0.0541 se=0.0070 p=0.0000***  (21.0s)
    [A/MOD:Qwen2.5   ] has_llms_txt                   n=117452 coef=+0.0294 se=0.0293 p=0.3154  (19.0s)
    [A/POOL:20       ] treat_stats_density            n=112637 coef=-0.0061 se=0.0032 p=0.0598·  (18.6s)
    [A/POOL:20       ] treat_question_headings        n=116981 coef=+0.2281 se=0.0297 p=0.0000***  (18.8s)
    [A/POOL:20       ] treat_structured_data          n=116981 coef=-0.0022 se=0.0262 p=0.9318  (20.2s)
    [A/POOL:20       ] T4_citation_authority_code     n=105269 coef=-0.0269 se=0.0076 p=0.0004***  (17.9s)
    [A/POOL:20       ] treat_topical_comp             n= 99674 coef=+0.0616 se=0.1030 p=0.5500  (17.0s)
    [A/POOL:20       ] treat_freshness                n=116981 coef=-0.0185 se=0.0073 p=0.0114*  (21.2s)
    [A/POOL:20       ] has_llms_txt                   n=117413 coef=+0.1214 se=0.0285 p=0.0000***  (18.0s)
    [A/POOL:50       ] treat_stats_density            n=138240 coef=+0.0041 se=0.0035 p=0.2446  (24.7s)
    [A/POOL:50       ] treat_question_headings        n=143834 coef=+0.0540 se=0.0299 p=0.0712·  (27.0s)
    [A/POOL:50       ] treat_structured_data          n=143834 coef=-0.2072 se=0.0272 p=0.0000***  (27.5s)
    [A/POOL:50       ] T4_citation_authority_code     n=104226 coef=-0.0232 se=0.0156 p=0.1384  (22.4s)
    [A/POOL:50       ] treat_topical_comp             n=127417 coef=-0.7270 se=0.0963 p=0.0000***  (23.0s)
    [A/POOL:50       ] treat_freshness                n=143834 coef=-0.0829 se=0.0073 p=0.0000***  (26.9s)
    [A/POOL:50       ] has_llms_txt                   n=116124 coef=+0.0211 se=0.0341 p=0.5368  (22.9s)

  [rank_delta] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0019 se=0.0027 p=0.4773  (44.5s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=+0.1426 se=0.0244 p=0.0000***  (46.9s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=-0.0480 se=0.0245 p=0.0502·  (56.8s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0252 se=0.0087 p=0.0038**  (82.2s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=-0.5078 se=0.0745 p=0.0000***  (77.2s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=-0.0596 se=0.0066 p=0.0000***  (81.8s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] has_llms_txt                   n=200000 coef=+0.0762 se=0.0241 p=0.0016**  (83.2s)  (controls: 6 other T + 33 X)

  [post_rank] Spec A — single treatment + confounders only
    [A/POOLED        ] treat_stats_density            n=200000 coef=-0.0027 se=0.0018 p=0.1401  (77.3s)
    [A/POOLED        ] treat_question_headings        n=200000 coef=-0.0534 se=0.0184 p=0.0037**  (76.0s)
    [A/POOLED        ] treat_structured_data          n=200000 coef=+0.1071 se=0.0172 p=0.0000***  (41.8s)
    [A/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0196 se=0.0065 p=0.0024**  (31.4s)
    [A/POOLED        ] treat_topical_comp             n=200000 coef=-0.2142 se=0.0595 p=0.0003***  (29.1s)
    [A/POOLED        ] treat_freshness                n=200000 coef=+0.0122 se=0.0047 p=0.0102*  (27.9s)
    [A/POOLED        ] has_llms_txt                   n=200000 coef=-0.0188 se=0.0196 p=0.3368  (26.7s)
    [A/VAR:biased    ] treat_stats_density            n= 64977 coef=-0.0004 se=0.0026 p=0.8806  (15.6s)
    [A/VAR:biased    ] treat_question_headings        n= 67034 coef=+0.0135 se=0.0300 p=0.6521  (16.1s)
    [A/VAR:biased    ] treat_structured_data          n= 67034 coef=+0.1149 se=0.0280 p=0.0000***  (14.5s)
    [A/VAR:biased    ] T4_citation_authority_code     n= 47616 coef=+0.0061 se=0.0083 p=0.4664  (13.5s)
    [A/VAR:biased    ] treat_topical_comp             n= 53449 coef=-0.1064 se=0.1069 p=0.3196  (14.6s)
    [A/VAR:biased    ] treat_freshness                n= 67034 coef=+0.0382 se=0.0078 p=0.0000***  (14.2s)
    [A/VAR:biased    ] has_llms_txt                   n= 52054 coef=-0.0311 se=0.0353 p=0.3785  (13.3s)
    [A/VAR:neutral   ] treat_stats_density            n= 85958 coef=-0.0025 se=0.0026 p=0.3203  (15.9s)
    [A/VAR:neutral   ] treat_question_headings        n= 90099 coef=-0.0392 se=0.0274 p=0.1527  (17.3s)
    [A/VAR:neutral   ] treat_structured_data          n= 90099 coef=+0.0991 se=0.0263 p=0.0002***  (16.7s)
    [A/VAR:neutral   ] T4_citation_authority_code     n= 65155 coef=-0.0169 se=0.0078 p=0.0307*  (14.8s)
    [A/VAR:neutral   ] treat_topical_comp             n= 83113 coef=-0.1081 se=0.0903 p=0.2314  (16.4s)
    [A/VAR:neutral   ] treat_freshness                n= 90099 coef=+0.0109 se=0.0072 p=0.1289  (17.3s)
    [A/VAR:neutral   ] has_llms_txt                   n= 73990 coef=+0.0346 se=0.0320 p=0.2807  (16.5s)
    [A/VAR:biased_rag] treat_stats_density            n= 60283 coef=-0.0047 se=0.0028 p=0.0957·  (14.1s)
    [A/VAR:biased_rag] treat_question_headings        n= 62162 coef=-0.0674 se=0.0320 p=0.0351*  (14.6s)
    [A/VAR:biased_rag] treat_structured_data          n= 62162 coef=+0.0807 se=0.0299 p=0.0071**  (14.8s)
    [A/VAR:biased_rag] T4_citation_authority_code     n= 43443 coef=+0.0011 se=0.0087 p=0.8963  (12.6s)
    [A/VAR:biased_rag] treat_topical_comp             n= 46890 coef=-0.4085 se=0.1169 p=0.0005***  (13.4s)
    [A/VAR:biased_rag] treat_freshness                n= 62162 coef=+0.0147 se=0.0083 p=0.0755·  (13.8s)
    [A/VAR:biased_rag] has_llms_txt                   n= 47190 coef=-0.0574 se=0.0373 p=0.1240  (12.8s)
    [A/VAR:neutral_rag] treat_stats_density            n= 66908 coef=-0.0009 se=0.0026 p=0.7461  (15.2s)
    [A/VAR:neutral_rag] treat_question_headings        n= 70121 coef=-0.0708 se=0.0309 p=0.0221*  (15.4s)
    [A/VAR:neutral_rag] treat_structured_data          n= 70121 coef=+0.1223 se=0.0290 p=0.0000***  (15.3s)
    [A/VAR:neutral_rag] T4_citation_authority_code     n= 53281 coef=-0.0141 se=0.0077 p=0.0697·  (12.9s)
    [A/VAR:neutral_rag] treat_topical_comp             n= 60241 coef=-0.2957 se=0.1041 p=0.0045**  (14.5s)
    [A/VAR:neutral_rag] treat_freshness                n= 70121 coef=+0.0023 se=0.0080 p=0.7751  (17.5s)
    [A/VAR:neutral_rag] has_llms_txt                   n= 60303 coef=-0.0252 se=0.0343 p=0.4637  (21.2s)
    [A/ENG:ddg       ] treat_stats_density            n= 99103 coef=+0.0046 se=0.0028 p=0.1035  (25.7s)
    [A/ENG:ddg       ] treat_question_headings        n=101894 coef=-0.0100 se=0.0288 p=0.7285  (25.3s)
    [A/ENG:ddg       ] treat_structured_data          n=101894 coef=+0.0173 se=0.0274 p=0.5264  (25.9s)
    [A/ENG:ddg       ] T4_citation_authority_code     n= 82765 coef=-0.0061 se=0.0076 p=0.4269  (39.3s)
    [A/ENG:ddg       ] treat_topical_comp             n= 50082 coef=-0.4490 se=0.1430 p=0.0017**  (52.2s)
    [A/ENG:ddg       ] treat_freshness                n=101894 coef=-0.0210 se=0.0073 p=0.0043**  (62.1s)
    [A/ENG:ddg       ] has_llms_txt                   n= 93526 coef=-0.0099 se=0.0308 p=0.7471  (57.8s)
    [A/ENG:searxng   ] treat_stats_density            n=179023 coef=-0.0088 se=0.0024 p=0.0002***  (69.5s)
    [A/ENG:searxng   ] treat_question_headings        n=187522 coef=-0.0302 se=0.0195 p=0.1220  (73.2s)
    [A/ENG:searxng   ] treat_structured_data          n=187522 coef=+0.1407 se=0.0180 p=0.0000***  (71.8s)
    [A/ENG:searxng   ] T4_citation_authority_code     n=126730 coef=-0.0331 se=0.0114 p=0.0037**  (65.6s)
    [A/ENG:searxng   ] treat_topical_comp             n=193611 coef=-0.1582 se=0.0611 p=0.0097**  (71.9s)
    [A/ENG:searxng   ] treat_freshness                n=187522 coef=+0.0412 se=0.0050 p=0.0000***  (75.9s)
    [A/ENG:searxng   ] has_llms_txt                   n=140011 coef=+0.0807 se=0.0248 p=0.0011**  (63.8s)
    [A/MOD:Llama     ] treat_stats_density            n=138621 coef=-0.0054 se=0.0021 p=0.0103*  (64.6s)
    [A/MOD:Llama     ] treat_question_headings        n=144317 coef=-0.0355 se=0.0215 p=0.0981·  (64.0s)
    [A/MOD:Llama     ] treat_structured_data          n=144317 coef=+0.1167 se=0.0201 p=0.0000***  (66.8s)
    [A/MOD:Llama     ] T4_citation_authority_code     n=104381 coef=-0.0062 se=0.0073 p=0.3948  (61.5s)
    [A/MOD:Llama     ] treat_topical_comp             n=120201 coef=-0.2057 se=0.0757 p=0.0066**  (61.1s)
    [A/MOD:Llama     ] treat_freshness                n=144317 coef=+0.0168 se=0.0056 p=0.0025**  (64.6s)
    [A/MOD:Llama     ] has_llms_txt                   n=116085 coef=-0.0329 se=0.0251 p=0.1900  (60.7s)
    [A/MOD:Qwen2.5   ] treat_stats_density            n=139505 coef=-0.0006 se=0.0021 p=0.7877  (65.6s)
    [A/MOD:Qwen2.5   ] treat_question_headings        n=145099 coef=-0.0496 se=0.0213 p=0.0201*  (64.0s)
    [A/MOD:Qwen2.5   ] treat_structured_data          n=145099 coef=+0.0853 se=0.0200 p=0.0000***  (69.8s)
    [A/MOD:Qwen2.5   ] T4_citation_authority_code     n=105114 coef=-0.0146 se=0.0077 p=0.0589·  (59.1s)
    [A/MOD:Qwen2.5   ] treat_topical_comp             n=123492 coef=-0.1956 se=0.0736 p=0.0079**  (63.4s)
    [A/MOD:Qwen2.5   ] treat_freshness                n=145099 coef=+0.0115 se=0.0055 p=0.0366*  (69.6s)
    [A/MOD:Qwen2.5   ] has_llms_txt                   n=117452 coef=+0.0293 se=0.0254 p=0.2474  (60.4s)
    [A/POOL:20       ] treat_stats_density            n=125051 coef=-0.0030 se=0.0026 p=0.2430  (63.9s)
    [A/POOL:20       ] treat_question_headings        n=129913 coef=-0.0333 se=0.0233 p=0.1524  (63.1s)
    [A/POOL:20       ] treat_structured_data          n=129913 coef=+0.1554 se=0.0217 p=0.0000***  (63.9s)
    [A/POOL:20       ] T4_citation_authority_code     n=105269 coef=-0.0086 se=0.0082 p=0.2955  (63.9s)
    [A/POOL:20       ] treat_topical_comp             n=108269 coef=-0.0765 se=0.0829 p=0.3563  (61.8s)
    [A/POOL:20       ] treat_freshness                n=129913 coef=+0.0369 se=0.0059 p=0.0000***  (64.5s)
    [A/POOL:20       ] has_llms_txt                   n=117413 coef=+0.0130 se=0.0251 p=0.6059  (58.6s)
    [A/POOL:50       ] treat_stats_density            n=153075 coef=-0.0012 se=0.0022 p=0.6031  (64.7s)
    [A/POOL:50       ] treat_question_headings        n=159503 coef=-0.0259 se=0.0211 p=0.2188  (62.6s)
    [A/POOL:50       ] treat_structured_data          n=159503 coef=+0.0567 se=0.0196 p=0.0039**  (65.4s)
    [A/POOL:50       ] T4_citation_authority_code     n=104226 coef=-0.0138 se=0.0092 p=0.1329  (53.7s)
    [A/POOL:50       ] treat_topical_comp             n=135424 coef=-0.3310 se=0.0741 p=0.0000***  (60.3s)
    [A/POOL:50       ] treat_freshness                n=159503 coef=+0.0009 se=0.0054 p=0.8637  (64.8s)
    [A/POOL:50       ] has_llms_txt                   n=116124 coef=-0.0030 se=0.0267 p=0.9097  (54.7s)

  [post_rank] Spec B — single treatment + other 6 treatments + confounders (POOLED)
    [B/POOLED        ] treat_stats_density            n=200000 coef=-0.0015 se=0.0019 p=0.4226  (73.9s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_question_headings        n=200000 coef=-0.0380 se=0.0186 p=0.0410*  (71.1s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_structured_data          n=200000 coef=+0.0937 se=0.0196 p=0.0000***  (71.5s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] T4_citation_authority_code     n=200000 coef=-0.0166 se=0.0066 p=0.0117*  (75.5s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_topical_comp             n=200000 coef=-0.2918 se=0.0600 p=0.0000***  (71.5s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] treat_freshness                n=200000 coef=+0.0043 se=0.0053 p=0.4139  (71.0s)  (controls: 6 other T + 33 X)
    [B/POOLED        ] has_llms_txt                   n=200000 coef=-0.0185 se=0.0197 p=0.3478  (70.5s)  (controls: 6 other T + 33 X)

  → saved geodml_data/data/dml_results/dml_canonical_2026-05-24.parquet  rows=252

========================================================================================
HEADLINE — Spec B POOLED (mutually-controlled) per outcome
========================================================================================

  Y = selected
                 treatment      n      coef       se        p_val sig  p_bonferroni bonferroni_sig
     treat_structured_data 200000 -0.014730 0.003050 1.369032e-06 ***      0.000115           True
           treat_freshness 200000 -0.004438 0.000820 6.245511e-08 ***      0.000005           True
       treat_stats_density 200000 -0.000442 0.000309 1.521940e-01          1.000000          False
T4_citation_authority_code 185176  0.000917 0.000700 1.900963e-01          1.000000          False
              has_llms_txt 200000  0.007275 0.002799 9.348171e-03  **      0.785246          False
   treat_question_headings 200000  0.016179 0.002929 3.303527e-08 ***      0.000003           True
        treat_topical_comp 200000  0.035807 0.009490 1.612556e-04 ***      0.013545           True

  Y = rank_delta
                 treatment      n      coef       se        p_val sig  p_bonferroni bonferroni_sig
        treat_topical_comp 200000 -0.507785 0.074462 9.141576e-12 ***  7.678924e-10           True
           treat_freshness 200000 -0.059611 0.006607 0.000000e+00 ***  0.000000e+00           True
     treat_structured_data 200000 -0.047995 0.024510 5.021175e-02   ·  1.000000e+00          False
T4_citation_authority_code 200000 -0.025240 0.008721 3.800630e-03  **  3.192529e-01          False
       treat_stats_density 200000 -0.001892 0.002663 4.772996e-01      1.000000e+00          False
              has_llms_txt 200000  0.076168 0.024092 1.569298e-03  **  1.318210e-01          False
   treat_question_headings 200000  0.142616 0.024354 4.746214e-09 ***  3.986820e-07           True

  Y = post_rank
                 treatment      n      coef       se    p_val sig  p_bonferroni bonferroni_sig
        treat_topical_comp 200000 -0.291833 0.059981 0.000001 ***      0.000096           True
   treat_question_headings 200000 -0.038043 0.018617 0.041010   *      1.000000          False
              has_llms_txt 200000 -0.018500 0.019704 0.347782          1.000000          False
T4_citation_authority_code 200000 -0.016583 0.006580 0.011727   *      0.985059          False
       treat_stats_density 200000 -0.001494 0.001864 0.422590          1.000000          False
           treat_freshness 200000  0.004308 0.005273 0.413913          1.000000          False
     treat_structured_data 200000  0.093719 0.019570 0.000002 ***      0.000141           True
