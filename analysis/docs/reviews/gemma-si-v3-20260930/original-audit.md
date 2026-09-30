Valerian, the complete original-sample audit finds substantial under-credit and several citation pairs that lack the claimed support. This is a GPT-6 Astra reference-model assessment, not human gold.

Coverage: 20 cells, 106 whole sources, 212 source/pass judgments, 141 selected-pair occurrences, and 71 distinct selected pairs. Every source has a frozen acceptable range, rationale, whole-source check and both-pass judgment records. No network or cluster inference was used.

Blind ranges were saved before opening Gemma results. Frozen file SHA256: `9e39c9fcee356c550709d83dce92ef4b6acfa1cb5c0593cf03c5399a48b5f54e`.

| Metric | Pass 1 | Pass 2 | Combined |
|---|---:|---:|---:|
| Range agreement | 61/106 (57.5%) | 62/106 (58.5%) | 123/212 (58.0%) |
| Valid pair occurrences | 57/71 (80.3%) | 57/70 (81.4%) | 114/141 (80.9%) |
| Invalid pair occurrences | 4 | 3 | 7 |
| Ambiguous pair occurrences | 10 | 10 | 20 |

The 71 distinct pairs comprise 57 valid, 4 invalid and 10 ambiguous pairs. Repeated pairs in the second pass are not independent semantic evidence. Counting every ambiguous occurrence as correct gives an optimistic upper bound of 134/141 (95.0%); this is not an established correctness result. Both user screening targets fail under the strict assessment: 95% pair correctness and 90% range agreement. These targets are screening choices, not validated benchmarks.

All 89 grades outside the frozen ranges are below them. Seven source records, repeated in both passes, meet the stated severe-under-credit flag: at least two grades below a blind minimum of at least 3. Some one-grade disagreements are subjective centrality boundaries; the larger gaps below are more informative.

Important cases:

- Loyalty, cell `326525bc3e7b07f069b7`, Popupsmart source `55d79ac7ce6250528c6639a3`: grade 1 in both passes versus blind 4–5. The source explicitly recommends points-based rewards, tiered memberships and referral incentives. These supply the program types in both sentences. The selected a2 witness is valid, but whole-source review also supports central a1 points-based configuration.
- POS, cell `224b306298e4a68d7f74`, KiwiCommerce source `4ebcc4d66a625dc6e8461f23`: grade 0 twice versus 3–4. The title says "Connect Online and In-Store Operations", matching the answer’s main definition. It does not prove real-time synchronization. Startups source `44713bd403371a7cf6a4b89d` likewise supports combining online/in-store sales but gets 1 versus 3–4.
- Trello, cell `2dd070ba00e1c41c31f5`: Kinsta and DigitalProjectManager sources get 2 versus 4–5, and AlternativeTo gets 2 versus 4. The written answer centrally recommends Productive, Breeze and Wekan. Query failure to implement Trello automation does not make those recommendations secondary within this answer.
- Proctoring, cell `2e4098d7ee3d2547ee74`, WiFiTalents source `fdf7ffdfe58ecaa7bf682576`: grade 2 twice versus 4–5. The whole source supplies reliability, user-friendliness, exam-specific proctoring, analytics and the evaluation rationale, much of the answer’s stated factors. Unsupported explanations about accessibility or anomaly detection must remain separate.
- Product research, cell `9d3e114a47ba2f5b05a4`, WinningHunter source `240e0c94afbc91ceb5e76b40`: a7/text1 is invalid in pass 1; a10/text1 is invalid in both. Listing Google Trends/Helium10 does not support stable-trend screening, market-share/review-count functions or margin calculations. The a4 tool-choice subset is supported, while the Pipiads title-only a4 match is ambiguous.
- Fulfillment, cell `ccf34042bf62e1e6fea0`, EcommerceNews source `1aa43fc185984e6f5ac75058`: a2/text1 is invalid twice. "Very time consuming" does not establish a reduction caused by outsourcing. The frozen range’s allowance for an implicit time-burden premise is retained transparently; it does not validate this selected causal-benefit pair.
- POS Fieldstack, source `ce1d4c6dc1470b0011124d07`: a2/text2 is invalid twice. A system touching inventory/customer/loyalty domains does not establish that their data synchronize in real time. Other pairs support more-than-checkout and warehouse/forecasting content, so this pair defect does not imply the whole source lacks central support.
- Loyalty Reddit, source `d5424e3c86688a5ae92b8772`: the grade-4 pair is valid only for "$20 after five points". The source requires each purchase to be at least $20; "one point for every order" drops that condition. Partial-unit matching must not validate the full configuration.
- Forex/Braintree, cell `2d3df8f3d3f469070d97`: the settings location and login are real partial facts, but all three distinct selected pairs are ambiguous because merchant setup context does not establish a forex depositor’s access or failure-recovery procedure. Similar context uncertainty applies to selected supplier-directory a1 matches framed as compliance automation.

Eligibility and style:

Cell `f9c5c4e1940622aa3acb` and cell `212aa2c7a6c698d700d7` are evidence-lacks answers. A single source cannot establish absence across the entire evidence set. Their local positive descriptions are assessed separately; this is an eligibility issue, not evidence that every source should receive a high score. The original sample has 16 direct answers, 2 mixed answers and 2 evidence-lacks answers. Cell0 is narration-dominant within its mixed label. Manual query proxies are 15 action, 3 informational and 2 mixed, and are not measured semantic-axis strata.

Subgroup counts:

| Group | Cells | Sources | Range agreement | Valid / all pair occurrences |
|---|---:|---:|---:|---:|
| model: qwen38 | 10 | 53 | 53/106 (50.0%) | 82/95 |
| model: llama4 | 10 | 53 | 70/106 (66.0%) | 32/46 |
| method: Parallel-Expansion-v1 | 10 | 70 | 67/140 (47.9%) | 70/85 |
| method: Reactive-Snippet-Loop-v1 | 10 | 36 | 56/72 (77.8%) | 44/56 |
| style: mixed | 2 | 14 | 15/28 (53.6%) | 18/22 |
| style: evidence_lacks | 2 | 9 | 2/18 (11.1%) | 6/6 |
| style: direct | 16 | 83 | 106/166 (63.9%) | 90/113 |
| query_style_proxy: action | 15 | 79 | 101/158 (63.9%) | 68/93 |
| query_style_proxy: mixed | 2 | 11 | 14/22 (63.6%) | 16/16 |
| query_style_proxy: informational | 3 | 16 | 8/32 (25.0%) | 30/32 |

The selected sample and single-model reference cannot establish population validity, human agreement or semantic-axis effects. Ambiguous pairs remain explicit. All 106 source-specific rationales, exact supported content, selected quotations and limitations are in `original-audit.json`; the unmodified independent judgments are in `original-blind-ranges.json`.

Post-audit reviewer disagreement: the primary reviewer considers Popupsmart loyalty 2–3, possibly 3, because the exact five-order/$20 configuration is central and missing. Astra agrees 3 is plausible under that weighting. The 4–5 blind range and comparison counts remain unchanged; the severe flag is relative to that reference and is not an adjudicated error. For KiwiCommerce, missing positive title support is firmer than the precise 3–4 range. Trello remains the clearer central-recommendation under-credit example.
