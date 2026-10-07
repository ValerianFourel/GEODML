## funnel qwen38 · Reactive|P|C|main (torch-cpu) — FAIL (2026-10-07 18:17)

Rows 1,116,687 × 57 features; choice sets 58,773; seconds {"full": 14.5, "bootstrap-0": 10.0, "bootstrap-1": 9.2, "bootstrap-2": 10.1}; Newton iterations (full) 5.

| quantity | CPU | torch | abs diff | threshold | ok |
|---|---|---|---|---|---|
| full page_intent_z | 0.2686082365 | 0.2686274273 | 1.92e-05 | 0.0001 | yes |
| full intent_x_prompt | -0.3653919478 | -0.3652456637 | 0.000146 | 0.0001 | NO |
| full intent_alignment | 0.1307547743 | 0.130626482 | 0.000128 | 0.0001 | NO |
| full topic_similarity | 1.685808398 | 1.685825591 | 1.72e-05 | 0.0001 | yes |
| full on_keyword | 0.4136697263 | 0.4136830606 | 1.33e-05 | 0.0001 | yes |
| full snip_title_chars | 0.1069600835 | 0.1069576124 | 2.47e-06 | 0.0001 | yes |
| full snip_text_words | 0.2926969881 | 0.2926931197 | 3.87e-06 | 0.0001 | yes |
| full snip_digits | 0.1316424204 | 0.1316701306 | 2.77e-05 | 0.0001 | yes |
| full snip_percent | 0.01747371933 | 0.01747392176 | 2.02e-07 | 0.0001 | yes |
| full snip_year | -0.2011710841 | -0.2011850076 | 1.39e-05 | 0.0001 | yes |
| full snip_currency | 0.04174810763 | 0.04173592603 | 1.22e-05 | 0.0001 | yes |
| full snip_title_question | -0.1029238955 | -0.102912522 | 1.14e-05 | 0.0001 | yes |
| full snip_title_listicle | -0.1248050747 | -0.1247880967 | 1.7e-05 | 0.0001 | yes |
| full snip_names_domain | 0.07921613199 | 0.07920724739 | 8.88e-06 | 0.0001 | yes |
| full snip_glued | 0.02983269999 | 0.0298355918 | 2.89e-06 | 0.0001 | yes |
| full url_https | -0.03375225772 | -0.03375255179 | 2.94e-07 | 0.0001 | yes |
| full url_path_depth | 0.05232814126 | 0.05237995242 | 5.18e-05 | 0.0001 | yes |
| full url_length | -0.01503387129 | -0.01513084803 | 9.7e-05 | 0.0001 | yes |
| full url_has_query | -0.0158732058 | -0.01591445978 | 4.13e-05 | 0.0001 | yes |
| full url_subdomain | 0.04693805107 | 0.04696736162 | 2.93e-05 | 0.0001 | yes |
| full url_tld_com | -0.01245992457 | -0.01240247056 | 5.75e-05 | 0.0001 | yes |
| full url_tld_org | -0.06602479825 | -0.06598132503 | 4.35e-05 | 0.0001 | yes |
| full url_tld_edu_gov | -0.08201399692 | -0.08199231949 | 2.17e-05 | 0.0001 | yes |
| full url_user_content | -0.02010267519 | -0.02001492195 | 8.78e-05 | 0.0001 | yes |
| full url_wikipedia | 0.07374875574 | 0.07374384576 | 4.91e-06 | 0.0001 | yes |
| full url_ad_redirect | 0 | 0 | 0 | 0.0001 | yes |
| full stored_position | -0.02544836367 | -0.02547774818 | 2.94e-05 | 0.0001 | yes |
| full searxng_score | -0.04860222344 | -0.04863239143 | 3.02e-05 | 0.0001 | yes |
| full searxng_engine_count | -0.07087661095 | -0.07085894333 | 1.77e-05 | 0.0001 | yes |
| full dfs_organic_count | 0.04572131382 | 0.04472052337 | 0.001 | 0.0001 | NO |
| full dfs_organic_top1 | 0.0466346478 | 0.04611099293 | 0.000524 | 0.0001 | NO |
| full dfs_traffic_value | -0.080545046 | -0.07913025615 | 0.00141 | 0.0001 | NO |
| full dfs_paid_count | -0.05170990505 | -0.05168720247 | 2.27e-05 | 0.0001 | yes |
| full dfs_domain_age | -0.05178306359 | -0.05179855672 | 1.55e-05 | 0.0001 | yes |
| full open_pagerank | 0.003131499096 | 0.003149588375 | 1.81e-05 | 0.0001 | yes |
| full has_llms_txt | -0.0004166089454 | -0.0004235851698 | 6.98e-06 | 0.0001 | yes |
| full brand_list | -0.008221748751 | -0.008207406206 | 1.43e-05 | 0.0001 | yes |
| full earned_list | -0.09796076102 | -0.09790029977 | 6.05e-05 | 0.0001 | yes |
| full google_top20_url | 0.06041555495 | 0.06048427211 | 6.87e-05 | 0.0001 | yes |
| full google_top20_domain | -0.03023595957 | -0.03034494687 | 0.000109 | 0.0001 | NO |
| full body_stats_density | -0.02941314398 | -0.02939324944 | 1.99e-05 | 0.0001 | yes |
| full body_question_headings | -0.07203458969 | -0.07205846694 | 2.39e-05 | 0.0001 | yes |
| full body_modularity | 0.02872915152 | 0.02874329522 | 1.41e-05 | 0.0001 | yes |
| full body_structured_data | 0.004377079248 | 0.004388359417 | 1.13e-05 | 0.0001 | yes |
| full body_ext_citations | 0.0291986184 | 0.02919369308 | 4.93e-06 | 0.0001 | yes |
| full body_auth_citations | 0.07400407309 | 0.07400725944 | 3.19e-06 | 0.0001 | yes |
| full body_word_count | -0.06176967143 | -0.06174228288 | 2.74e-05 | 0.0001 | yes |
| full body_readability | 0.0316124383 | 0.03162164878 | 9.21e-06 | 0.0001 | yes |
| full body_internal_links | 0.02187785928 | 0.02186814867 | 9.71e-06 | 0.0001 | yes |
| full body_outbound_links | -0.03181126955 | -0.03178747592 | 2.38e-05 | 0.0001 | yes |
| full body_images_alt | 0.001617262133 | 0.001589501951 | 2.78e-05 | 0.0001 | yes |
| full body_freshness | 0.01223327753 | 0.01222116374 | 1.21e-05 | 0.0001 | yes |
| full c1_dfs_missing | -0.01015347644 | -0.01013924741 | 1.42e-05 | 0.0001 | yes |
| full c1_opr_missing | 0.02198997997 | 0.02198449367 | 5.49e-06 | 0.0001 | yes |
| full c1_llms_missing | 0.01514620325 | 0.01514474143 | 1.46e-06 | 0.0001 | yes |
| full c2_html_missing | 0.1147091393 | 0.1144247968 | 0.000284 | 0.0001 | NO |
| full c2_readability_missing | -0.1010860075 | -0.1007875585 | 0.000298 | 0.0001 | NO |
| bootstrap-0 page_intent_z | 0.326519723 | 0.3265823611 | 6.26e-05 | 0.0001 | yes |
| bootstrap-0 intent_x_prompt | -0.4133360929 | -0.4132203024 | 0.000116 | 0.0001 | NO |
| bootstrap-0 intent_alignment | 0.1551821056 | 0.1550175996 | 0.000165 | 0.0001 | NO |
| bootstrap-0 topic_similarity | 1.694309361 | 1.694361788 | 5.24e-05 | 0.0001 | yes |
| bootstrap-0 on_keyword | 0.3669171116 | 0.366914708 | 2.4e-06 | 0.0001 | yes |
| bootstrap-0 snip_title_chars | 0.1408236947 | 0.1408330942 | 9.4e-06 | 0.0001 | yes |
| bootstrap-0 snip_text_words | 0.2814885313 | 0.2814899342 | 1.4e-06 | 0.0001 | yes |
| bootstrap-0 snip_digits | 0.1252453877 | 0.1252852242 | 3.98e-05 | 0.0001 | yes |
| bootstrap-0 snip_percent | 0.01179362607 | 0.01179682423 | 3.2e-06 | 0.0001 | yes |
| bootstrap-0 snip_year | -0.2454486696 | -0.245502022 | 5.34e-05 | 0.0001 | yes |
| bootstrap-0 snip_currency | 0.04771274238 | 0.04771273193 | 1.05e-08 | 0.0001 | yes |
| bootstrap-0 snip_title_question | -0.09556751115 | -0.09557014465 | 2.63e-06 | 0.0001 | yes |
| bootstrap-0 snip_title_listicle | -0.1187432212 | -0.1188014982 | 5.83e-05 | 0.0001 | yes |
| bootstrap-0 snip_names_domain | 0.06835492719 | 0.06839230413 | 3.74e-05 | 0.0001 | yes |
| bootstrap-0 snip_glued | 0.03058583524 | 0.03057392314 | 1.19e-05 | 0.0001 | yes |
| bootstrap-0 url_https | -0.0418761839 | -0.04188676953 | 1.06e-05 | 0.0001 | yes |
| bootstrap-0 url_path_depth | 0.02613065974 | 0.02599203108 | 0.000139 | 0.0001 | NO |
| bootstrap-0 url_length | 0.0134749401 | 0.01371151045 | 0.000237 | 0.0001 | NO |
| bootstrap-0 url_has_query | -0.1175341867 | -0.1175895943 | 5.54e-05 | 0.0001 | yes |
| bootstrap-0 url_subdomain | 0.08596835872 | 0.08596316062 | 5.2e-06 | 0.0001 | yes |
| bootstrap-0 url_tld_com | 0.009977076336 | 0.009908744239 | 6.83e-05 | 0.0001 | yes |
| bootstrap-0 url_tld_org | -0.05704708445 | -0.05705200255 | 4.92e-06 | 0.0001 | yes |
| bootstrap-0 url_tld_edu_gov | -0.08417749695 | -0.08421370382 | 3.62e-05 | 0.0001 | yes |
| bootstrap-0 url_user_content | -0.03028899712 | -0.03042064095 | 0.000132 | 0.0001 | NO |
| bootstrap-0 url_wikipedia | 0.06661562715 | 0.06658934626 | 2.63e-05 | 0.0001 | yes |
| bootstrap-0 url_ad_redirect | 0 | 0 | 0 | 0.0001 | yes |
| bootstrap-0 stored_position | -0.03092743235 | -0.03090422286 | 2.32e-05 | 0.0001 | yes |
| bootstrap-0 searxng_score | -0.06221369638 | -0.06213904584 | 7.47e-05 | 0.0001 | yes |
| bootstrap-0 searxng_engine_count | -0.04454709519 | -0.04462226006 | 7.52e-05 | 0.0001 | yes |
| bootstrap-0 dfs_organic_count | -0.06131791671 | -0.0597246228 | 0.00159 | 0.0001 | NO |
| bootstrap-0 dfs_organic_top1 | 0.1382810237 | 0.1389462904 | 0.000665 | 0.0001 | NO |
| bootstrap-0 dfs_traffic_value | -0.04832230178 | -0.05044004181 | 0.00212 | 0.0001 | NO |
| bootstrap-0 dfs_paid_count | -0.0697737085 | -0.06979718988 | 2.35e-05 | 0.0001 | yes |
| bootstrap-0 dfs_domain_age | -0.06607992153 | -0.06604518455 | 3.47e-05 | 0.0001 | yes |
| bootstrap-0 open_pagerank | -0.01595694587 | -0.01597874538 | 2.18e-05 | 0.0001 | yes |
| bootstrap-0 has_llms_txt | 0.01741291797 | 0.01742968138 | 1.68e-05 | 0.0001 | yes |
| bootstrap-0 brand_list | -0.002525531533 | -0.002524742014 | 7.9e-07 | 0.0001 | yes |
| bootstrap-0 earned_list | -0.05663608369 | -0.0566751124 | 3.9e-05 | 0.0001 | yes |
| bootstrap-0 google_top20_url | 0.0527296935 | 0.05268280302 | 4.69e-05 | 0.0001 | yes |
| bootstrap-0 google_top20_domain | -0.03531598189 | -0.03521180286 | 0.000104 | 0.0001 | NO |
| bootstrap-0 body_stats_density | -0.02179655369 | -0.02178925143 | 7.3e-06 | 0.0001 | yes |
| bootstrap-0 body_question_headings | -0.04362925808 | -0.04365572047 | 2.65e-05 | 0.0001 | yes |
| bootstrap-0 body_modularity | 0.04514987801 | 0.04522585504 | 7.6e-05 | 0.0001 | yes |
| bootstrap-0 body_structured_data | 0.01348428362 | 0.01349695386 | 1.27e-05 | 0.0001 | yes |
| bootstrap-0 body_ext_citations | 0.01248556076 | 0.01249350784 | 7.95e-06 | 0.0001 | yes |
| bootstrap-0 body_auth_citations | 0.07047739495 | 0.07048443938 | 7.04e-06 | 0.0001 | yes |
| bootstrap-0 body_word_count | -0.08855853466 | -0.08863676129 | 7.82e-05 | 0.0001 | yes |
| bootstrap-0 body_readability | 0.007686106015 | 0.007667337857 | 1.88e-05 | 0.0001 | yes |
| bootstrap-0 body_internal_links | 0.06950874036 | 0.06949270762 | 1.6e-05 | 0.0001 | yes |
| bootstrap-0 body_outbound_links | -0.03493279559 | -0.03496566914 | 3.29e-05 | 0.0001 | yes |
| bootstrap-0 body_images_alt | 0.03234753688 | 0.03239095747 | 4.34e-05 | 0.0001 | yes |
| bootstrap-0 body_freshness | 0.01542510509 | 0.01542941145 | 4.31e-06 | 0.0001 | yes |
| bootstrap-0 c1_dfs_missing | -0.0112723248 | -0.01129575887 | 2.34e-05 | 0.0001 | yes |
| bootstrap-0 c1_opr_missing | 0.02515925663 | 0.02519084076 | 3.16e-05 | 0.0001 | yes |
| bootstrap-0 c1_llms_missing | 0.03719430342 | 0.03719159314 | 2.71e-06 | 0.0001 | yes |
| bootstrap-0 c2_html_missing | 0.210051247 | 0.2103762297 | 0.000325 | 0.0001 | NO |
| bootstrap-0 c2_readability_missing | -0.1692035318 | -0.1695488123 | 0.000345 | 0.0001 | NO |
| bootstrap-1 page_intent_z | 0.2600490835 | 0.2600704241 | 2.13e-05 | 0.0001 | yes |
| bootstrap-1 intent_x_prompt | -0.3729872464 | -0.3735263282 | 0.000539 | 0.0001 | NO |
| bootstrap-1 intent_alignment | 0.1634548749 | 0.163747354 | 0.000292 | 0.0001 | NO |
| bootstrap-1 topic_similarity | 1.726537889 | 1.726549768 | 1.19e-05 | 0.0001 | yes |
| bootstrap-1 on_keyword | 0.4342734886 | 0.4342466358 | 2.69e-05 | 0.0001 | yes |
| bootstrap-1 snip_title_chars | 0.1115698415 | 0.1115602544 | 9.59e-06 | 0.0001 | yes |
| bootstrap-1 snip_text_words | 0.2762304708 | 0.2762395243 | 9.05e-06 | 0.0001 | yes |
| bootstrap-1 snip_digits | 0.1343806455 | 0.1344231595 | 4.25e-05 | 0.0001 | yes |
| bootstrap-1 snip_percent | 0.008772511621 | 0.008767762709 | 4.75e-06 | 0.0001 | yes |
| bootstrap-1 snip_year | -0.2261788543 | -0.2262168892 | 3.8e-05 | 0.0001 | yes |
| bootstrap-1 snip_currency | 0.06789375339 | 0.06788405218 | 9.7e-06 | 0.0001 | yes |
| bootstrap-1 snip_title_question | -0.05823576269 | -0.05820192292 | 3.38e-05 | 0.0001 | yes |
| bootstrap-1 snip_title_listicle | -0.1299770939 | -0.12998033 | 3.24e-06 | 0.0001 | yes |
| bootstrap-1 snip_names_domain | 0.1054470934 | 0.1054638668 | 1.68e-05 | 0.0001 | yes |
| bootstrap-1 snip_glued | 0.04501269267 | 0.04500140734 | 1.13e-05 | 0.0001 | yes |
| bootstrap-1 url_https | -0.03788162207 | -0.03788457574 | 2.95e-06 | 0.0001 | yes |
| bootstrap-1 url_path_depth | 0.03033694134 | 0.03033252789 | 4.41e-06 | 0.0001 | yes |
| bootstrap-1 url_length | 0.008166709464 | 0.008194864781 | 2.82e-05 | 0.0001 | yes |
| bootstrap-1 url_has_query | -0.02646587309 | -0.02643892568 | 2.69e-05 | 0.0001 | yes |
| bootstrap-1 url_subdomain | 0.02784281548 | 0.02784932021 | 6.5e-06 | 0.0001 | yes |
| bootstrap-1 url_tld_com | 0.004780153648 | 0.004763220474 | 1.69e-05 | 0.0001 | yes |
| bootstrap-1 url_tld_org | -0.07253202445 | -0.07251729469 | 1.47e-05 | 0.0001 | yes |
| bootstrap-1 url_tld_edu_gov | -0.08362119855 | -0.0836006188 | 2.06e-05 | 0.0001 | yes |
| bootstrap-1 url_user_content | -0.0484953814 | -0.04851475895 | 1.94e-05 | 0.0001 | yes |
| bootstrap-1 url_wikipedia | 0.08593692904 | 0.08597372739 | 3.68e-05 | 0.0001 | yes |
| bootstrap-1 url_ad_redirect | 0 | 0 | 0 | 0.0001 | yes |
| bootstrap-1 stored_position | -0.03725657562 | -0.03722338296 | 3.32e-05 | 0.0001 | yes |
| bootstrap-1 searxng_score | -0.1081593576 | -0.1081284665 | 3.09e-05 | 0.0001 | yes |
| bootstrap-1 searxng_engine_count | -0.04071730465 | -0.04075317564 | 3.59e-05 | 0.0001 | yes |
| bootstrap-1 dfs_organic_count | 0.1655482946 | 0.1659441628 | 0.000396 | 0.0001 | NO |
| bootstrap-1 dfs_organic_top1 | 0.08217965269 | 0.08191241032 | 0.000267 | 0.0001 | NO |
| bootstrap-1 dfs_traffic_value | -0.2403321575 | -0.2404916409 | 0.000159 | 0.0001 | NO |
| bootstrap-1 dfs_paid_count | -0.05251797764 | -0.05246051574 | 5.75e-05 | 0.0001 | yes |
| bootstrap-1 dfs_domain_age | -0.09218558057 | -0.09220172835 | 1.61e-05 | 0.0001 | yes |
| bootstrap-1 open_pagerank | -0.02460722587 | -0.02460476467 | 2.46e-06 | 0.0001 | yes |
| bootstrap-1 has_llms_txt | -0.002121536344 | -0.002097388695 | 2.41e-05 | 0.0001 | yes |
| bootstrap-1 brand_list | 0.01909153261 | 0.01907477691 | 1.68e-05 | 0.0001 | yes |
| bootstrap-1 earned_list | -0.02921856282 | -0.02924870378 | 3.01e-05 | 0.0001 | yes |
| bootstrap-1 google_top20_url | 0.06688148076 | 0.06701155296 | 0.00013 | 0.0001 | NO |
| bootstrap-1 google_top20_domain | -0.04359449084 | -0.04369185877 | 9.74e-05 | 0.0001 | yes |
| bootstrap-1 body_stats_density | -0.0359411416 | -0.03596246895 | 2.13e-05 | 0.0001 | yes |
| bootstrap-1 body_question_headings | -0.0009580220697 | -0.0009454654377 | 1.26e-05 | 0.0001 | yes |
| bootstrap-1 body_modularity | -0.02347849401 | -0.02344619797 | 3.23e-05 | 0.0001 | yes |
| bootstrap-1 body_structured_data | -0.01624505882 | -0.01624633031 | 1.27e-06 | 0.0001 | yes |
| bootstrap-1 body_ext_citations | 0.02549843373 | 0.02546840856 | 3e-05 | 0.0001 | yes |
| bootstrap-1 body_auth_citations | 0.08839696818 | 0.08838116964 | 1.58e-05 | 0.0001 | yes |
| bootstrap-1 body_word_count | -0.07661156523 | -0.07670535589 | 9.38e-05 | 0.0001 | yes |
| bootstrap-1 body_readability | 0.03188140086 | 0.03187341874 | 7.98e-06 | 0.0001 | yes |
| bootstrap-1 body_internal_links | 0.03239249869 | 0.03237311683 | 1.94e-05 | 0.0001 | yes |
| bootstrap-1 body_outbound_links | -0.04814733913 | -0.04811704342 | 3.03e-05 | 0.0001 | yes |
| bootstrap-1 body_images_alt | -0.001801731824 | -0.001797929286 | 3.8e-06 | 0.0001 | yes |
| bootstrap-1 body_freshness | 0.01235893836 | 0.01235312035 | 5.82e-06 | 0.0001 | yes |
| bootstrap-1 c1_dfs_missing | 0.002838738359 | 0.002823482938 | 1.53e-05 | 0.0001 | yes |
| bootstrap-1 c1_opr_missing | 0.03144767479 | 0.03143165813 | 1.6e-05 | 0.0001 | yes |
| bootstrap-1 c1_llms_missing | 0.01745699151 | 0.01746244696 | 5.46e-06 | 0.0001 | yes |
| bootstrap-1 c2_html_missing | 0.2065397583 | 0.2074043415 | 0.000865 | 0.0001 | NO |
| bootstrap-1 c2_readability_missing | -0.1882724036 | -0.1891162743 | 0.000844 | 0.0001 | NO |
| bootstrap-2 page_intent_z | 0.2753690763 | 0.2753391694 | 2.99e-05 | 0.0001 | yes |
| bootstrap-2 intent_x_prompt | -0.3494003667 | -0.3492936531 | 0.000107 | 0.0001 | NO |
| bootstrap-2 intent_alignment | 0.1279642075 | 0.1278869693 | 7.72e-05 | 0.0001 | yes |
| bootstrap-2 topic_similarity | 1.680555123 | 1.6805853 | 3.02e-05 | 0.0001 | yes |
| bootstrap-2 on_keyword | 0.4367998193 | 0.4368021472 | 2.33e-06 | 0.0001 | yes |
| bootstrap-2 snip_title_chars | 0.1148164979 | 0.114799576 | 1.69e-05 | 0.0001 | yes |
| bootstrap-2 snip_text_words | 0.2750897332 | 0.2750829434 | 6.79e-06 | 0.0001 | yes |
| bootstrap-2 snip_digits | 0.186292976 | 0.1863091554 | 1.62e-05 | 0.0001 | yes |
| bootstrap-2 snip_percent | 0.008407487209 | 0.008410407345 | 2.92e-06 | 0.0001 | yes |
| bootstrap-2 snip_year | -0.2265574654 | -0.2265436168 | 1.38e-05 | 0.0001 | yes |
| bootstrap-2 snip_currency | 0.04494673398 | 0.04492605036 | 2.07e-05 | 0.0001 | yes |
| bootstrap-2 snip_title_question | -0.08743245269 | -0.08745150932 | 1.91e-05 | 0.0001 | yes |
| bootstrap-2 snip_title_listicle | -0.09927219568 | -0.09926502613 | 7.17e-06 | 0.0001 | yes |
| bootstrap-2 snip_names_domain | 0.08137403907 | 0.081366666 | 7.37e-06 | 0.0001 | yes |
| bootstrap-2 snip_glued | 0.01137798779 | 0.01136573753 | 1.23e-05 | 0.0001 | yes |
| bootstrap-2 url_https | -0.03806470514 | -0.03806518285 | 4.78e-07 | 0.0001 | yes |
| bootstrap-2 url_path_depth | 0.06410109893 | 0.06411733398 | 1.62e-05 | 0.0001 | yes |
| bootstrap-2 url_length | -0.04163999933 | -0.04168581473 | 4.58e-05 | 0.0001 | yes |
| bootstrap-2 url_has_query | 0.03025610599 | 0.03015144793 | 0.000105 | 0.0001 | NO |
| bootstrap-2 url_subdomain | 0.02261309454 | 0.02262503755 | 1.19e-05 | 0.0001 | yes |
| bootstrap-2 url_tld_com | -0.05106197355 | -0.05102073058 | 4.12e-05 | 0.0001 | yes |
| bootstrap-2 url_tld_org | -0.09475068873 | -0.09473543015 | 1.53e-05 | 0.0001 | yes |
| bootstrap-2 url_tld_edu_gov | -0.05958990094 | -0.05960498817 | 1.51e-05 | 0.0001 | yes |
| bootstrap-2 url_user_content | -0.07069052085 | -0.07054378614 | 0.000147 | 0.0001 | NO |
| bootstrap-2 url_wikipedia | 0.09326883024 | 0.09325444435 | 1.44e-05 | 0.0001 | yes |
| bootstrap-2 url_ad_redirect | 0 | 0 | 0 | 0.0001 | yes |
| bootstrap-2 stored_position | -0.02633126229 | -0.02635051089 | 1.92e-05 | 0.0001 | yes |
| bootstrap-2 searxng_score | -0.06281231395 | -0.06276386476 | 4.84e-05 | 0.0001 | yes |
| bootstrap-2 searxng_engine_count | -0.02221111907 | -0.02229059885 | 7.95e-05 | 0.0001 | yes |
| bootstrap-2 dfs_organic_count | 0.1632405132 | 0.1618544069 | 0.00139 | 0.0001 | NO |
| bootstrap-2 dfs_organic_top1 | 0.05248201169 | 0.05189337398 | 0.000589 | 0.0001 | NO |
| bootstrap-2 dfs_traffic_value | -0.1462039266 | -0.1443523536 | 0.00185 | 0.0001 | NO |
| bootstrap-2 dfs_paid_count | -0.0615241229 | -0.06148129887 | 4.28e-05 | 0.0001 | yes |
| bootstrap-2 dfs_domain_age | -0.04755017236 | -0.04757186142 | 2.17e-05 | 0.0001 | yes |
| bootstrap-2 open_pagerank | 0.009184233416 | 0.009214487307 | 3.03e-05 | 0.0001 | yes |
| bootstrap-2 has_llms_txt | 0.01869926058 | 0.01867099593 | 2.83e-05 | 0.0001 | yes |
| bootstrap-2 brand_list | -0.02452287192 | -0.02450971403 | 1.32e-05 | 0.0001 | yes |
| bootstrap-2 earned_list | -0.1362939367 | -0.1362563231 | 3.76e-05 | 0.0001 | yes |
| bootstrap-2 google_top20_url | 0.05720010923 | 0.05721216271 | 1.21e-05 | 0.0001 | yes |
| bootstrap-2 google_top20_domain | -0.03167317311 | -0.03171432079 | 4.11e-05 | 0.0001 | yes |
| bootstrap-2 body_stats_density | -0.03824307979 | -0.03822955195 | 1.35e-05 | 0.0001 | yes |
| bootstrap-2 body_question_headings | -0.09787057514 | -0.09785799235 | 1.26e-05 | 0.0001 | yes |
| bootstrap-2 body_modularity | 0.03010521552 | 0.03007732382 | 2.79e-05 | 0.0001 | yes |
| bootstrap-2 body_structured_data | -0.01950258491 | -0.01951396255 | 1.14e-05 | 0.0001 | yes |
| bootstrap-2 body_ext_citations | 0.05139246955 | 0.05138374898 | 8.72e-06 | 0.0001 | yes |
| bootstrap-2 body_auth_citations | 0.06949161279 | 0.06949516894 | 3.56e-06 | 0.0001 | yes |
| bootstrap-2 body_word_count | -0.04862981933 | -0.0485828569 | 4.7e-05 | 0.0001 | yes |
| bootstrap-2 body_readability | 0.05606309987 | 0.05609026896 | 2.72e-05 | 0.0001 | yes |
| bootstrap-2 body_internal_links | 0.01069074934 | 0.0106974846 | 6.74e-06 | 0.0001 | yes |
| bootstrap-2 body_outbound_links | -0.07569494354 | -0.07567415589 | 2.08e-05 | 0.0001 | yes |
| bootstrap-2 body_images_alt | 0.002015922606 | 0.00200890465 | 7.02e-06 | 0.0001 | yes |
| bootstrap-2 body_freshness | 0.02479716889 | 0.0248046632 | 7.49e-06 | 0.0001 | yes |
| bootstrap-2 c1_dfs_missing | -0.05703808918 | -0.05702652349 | 1.16e-05 | 0.0001 | yes |
| bootstrap-2 c1_opr_missing | 0.01687688561 | 0.01687979969 | 2.91e-06 | 0.0001 | yes |
| bootstrap-2 c1_llms_missing | 0.03143245137 | 0.03142147104 | 1.1e-05 | 0.0001 | yes |
| bootstrap-2 c2_html_missing | 0.04749370348 | 0.04713039071 | 0.000363 | 0.0001 | NO |
| bootstrap-2 c2_readability_missing | -0.03534934076 | -0.03495945032 | 0.00039 | 0.0001 | NO |
| full objective (pass needs torch <= cpu + 1e-12) | 2.234822639 | 2.234822618 | 2.15e-08 | 1e-06 | yes |

### Diagnosis of the funnel gate failure (2026-10-08)

The gate as pre-stated (PREREG computational note C1: coefficients within 1e-4 of the **cached CPU fits**) fails on several
coefficients (largest 1.4e-3 on the full fit's nuisance controls, up to 5.4e-4 on intent terms in a bootstrap unit).
A tightly converged CPU fit of the same full problem (scipy L-BFGS-B, gtol 1e-11, ftol 1e-15; 63 iterations, 19.5 s,
max |gradient| 1.2e-8) settles which side is off:

| | objective | max abs coefficient difference to the tight CPU fit |
|---|---|---|
| cached CPU fit (default tolerances, exploration report v2) | 2.2348226394673 | 1.4e-3 (dfs_traffic_value) |
| tight CPU fit | 2.2348226179799 | 0 |
| PyTorch Newton (torch-cpu) | 2.2348226179799 | 4.3e-8 |

The backend reproduces the exact optimum; the cached CPU fits stop early (default `ftol` of L-BFGS-B) and carry
optimisation error of up to about 1e-3 per coefficient, small against their keyword-bootstrap intervals but larger than
the gate. Per note C1 the funnel and decisions estimators stay on the CPU path unless Valerian approves re-stating the
reference of this gate as the tightly converged CPU optimum (which they then pass at 4e-8).
## steelman generator qwen38 · Reactive (torch-cpu) — FAIL (2026-10-07 19:15)

100 keyword draws, 200 Monte-Carlo draws per answer; 3246.8 s on this machine; TOST verdict CPU True / torch True.

| quantity | CPU | torch | abs diff | threshold | ok |
|---|---|---|---|---|---|
| main delta_gen | -7.849773105e-05 | -7.634720721e-05 | 2.15e-06 | 0.0001 | yes |
| main delta_gen ci90[0] | -0.002067446869 | -0.002072094082 | 4.65e-06 | 0.0002 | yes |
| main delta_gen ci90[1] | 0.001452745195 | 0.001299859288 | 0.000153 | 0.0002 | yes |
| main delta_gen ci95[0] | -0.002208755589 | -0.002216780556 | 8.02e-06 | 0.0002 | yes |
| main delta_gen ci95[1] | 0.001679937665 | 0.001644413355 | 3.55e-05 | 0.0002 | yes |
| main model_check | -0.00098134391 | -0.0009779880904 | 3.36e-06 | 0.0001 | yes |
| no_slot delta_gen | 0.0002439900693 | 0.000247571858 | 3.58e-06 | 0.0001 | yes |
| no_slot delta_gen ci90[0] | -0.001743615872 | -0.001770861626 | 2.72e-05 | 0.0002 | yes |
| no_slot delta_gen ci90[1] | 0.002035474135 | 0.001987699742 | 4.78e-05 | 0.0002 | yes |
| no_slot delta_gen ci95[0] | -0.002391755787 | -0.002410536717 | 1.88e-05 | 0.0002 | yes |
| no_slot delta_gen ci95[1] | 0.002596607523 | 0.002478531735 | 0.000118 | 0.0002 | yes |
| no_slot model_check | -0.0009814720256 | -0.0009819347017 | 4.63e-07 | 0.0001 | yes |
| score delta_gen | 0.0002852023634 | 0.0002892726349 | 4.07e-06 | 0.0001 | yes |
| score delta_gen ci90[0] | -0.001698250956 | -0.001699956845 | 1.71e-06 | 0.0002 | yes |
| score delta_gen ci90[1] | 0.001784342739 | 0.001704081894 | 8.03e-05 | 0.0002 | yes |
| score delta_gen ci95[0] | -0.001890611433 | -0.001902465762 | 1.19e-05 | 0.0002 | yes |
| score delta_gen ci95[1] | 0.002137698927 | 0.002117170577 | 2.05e-05 | 0.0002 | yes |
| score model_check | -0.0009378236855 | -0.0009341852297 | 3.64e-06 | 0.0001 | yes |
| main|keep intent_alignment | 0.3201761456 | 0.3199970049 | 0.000179 | 0.0001 | NO |
| main|keep intent_x_prompt | -0.5501617488 | -0.5499079078 | 0.000254 | 0.0001 | NO |
| main|keep on_keyword | 0.3966111046 | 0.3965967439 | 1.44e-05 | 0.0001 | yes |
| main|keep page_intent_z | 0.02158451557 | 0.0216634407 | 7.89e-05 | 0.0001 | yes |
| main|keep topic_similarity | 1.929769831 | 1.929808423 | 3.86e-05 | 0.0001 | yes |
| main|order intent_alignment | 0.0958625586 | 0.0954212751 | 0.000441 | 0.0001 | NO |
| main|order intent_x_prompt | -0.0390120547 | -0.03809842345 | 0.000914 | 0.0001 | NO |
| main|order on_keyword | 0.00275721095 | 0.002816458341 | 5.92e-05 | 0.0001 | yes |
| main|order page_intent_z | 0.05490892969 | 0.05491129445 | 2.36e-06 | 0.0001 | yes |
| main|order topic_similarity | 0.6741088758 | 0.6740067115 | 0.000102 | 0.0001 | NO |
| no_slot|keep intent_alignment | 0.3365206242 | 0.3360805072 | 0.00044 | 0.0001 | NO |
| no_slot|keep intent_x_prompt | -0.5558294251 | -0.5550628362 | 0.000767 | 0.0001 | NO |
| no_slot|keep on_keyword | 0.3965666884 | 0.3965724044 | 5.72e-06 | 0.0001 | yes |
| no_slot|keep page_intent_z | 0.04938435386 | 0.04938818917 | 3.84e-06 | 0.0001 | yes |
| no_slot|keep topic_similarity | 1.982127359 | 1.982048396 | 7.9e-05 | 0.0001 | yes |
| no_slot|order intent_alignment | 0.02642057787 | 0.02666719187 | 0.000247 | 0.0001 | NO |
| no_slot|order intent_x_prompt | 0.007619540027 | 0.007181077536 | 0.000438 | 0.0001 | NO |
| no_slot|order on_keyword | 0.2441149518 | 0.2441207315 | 5.78e-06 | 0.0001 | yes |
| no_slot|order page_intent_z | 0.08744211414 | 0.08743211327 | 1e-05 | 0.0001 | yes |
| no_slot|order topic_similarity | 0.7593585173 | 0.7592576996 | 0.000101 | 0.0001 | NO |
| score|keep intent_alignment | 0.2909536758 | 0.2908129354 | 0.000141 | 0.0001 | NO |
| score|keep intent_x_prompt | -0.474808943 | -0.4745602951 | 0.000249 | 0.0001 | NO |
| score|keep on_keyword | 0.3157324422 | 0.3157318813 | 5.61e-07 | 0.0001 | yes |
| score|keep page_intent_z | 0.005827538202 | 0.005882921842 | 5.54e-05 | 0.0001 | yes |
| score|keep reranker_logit | 0.8292502613 | 0.8291377979 | 0.000112 | 0.0001 | NO |
| score|keep topic_similarity | 1.78668949 | 1.786712125 | 2.26e-05 | 0.0001 | yes |
| score|order intent_alignment | 0.0811946658 | 0.08103358935 | 0.000161 | 0.0001 | NO |
| score|order intent_x_prompt | -0.006516922256 | -0.006241815551 | 0.000275 | 0.0001 | NO |
| score|order on_keyword | -0.02287219288 | -0.02285398391 | 1.82e-05 | 0.0001 | yes |
| score|order page_intent_z | 0.04108108111 | 0.04113372194 | 5.26e-05 | 0.0001 | yes |
| score|order reranker_logit | 0.8420040793 | 0.8422192089 | 0.000215 | 0.0001 | NO |
| score|order topic_similarity | 0.6173043588 | 0.6173261852 | 2.18e-05 | 0.0001 | yes |

### Reading of the steelman generator gate (2026-10-08)

The section above was labelled FAIL by a first version of the script that also gated the keep/order coefficients, which
PREREG computational note C1 does not gate. Under C1 (Δ_gen within 1e-4, interval endpoints within 2e-4, same TOST verdict,
no failed replicates) the gate **passes** for Qwen · Reactive: largest Δ_gen difference 4.1e-6, largest endpoint
difference 1.53e-4 (main ci90 upper), TOST verdict True on both backends, 0 failed replicates. The keep/order coefficients
differ by up to 9.1e-4 (main|order intent_x_prompt), the same early stopping of the CPU L-BFGS-B fits diagnosed for the
funnel. Timing on this Mac (torch-cpu, 5 threads, one stratum, 101 refits + Monte Carlo): 3,247 s.
