# Fresh SI-v3 semantic audit

Independent GPT-6 Astra model-reference review; not human gold standard. All 20 cells, 108 sources, 216 judgments and every selected pair inspected. Blind ranges were written before opening grades. Full provided source text, not retrieved pages, determines support.

## Counts

- Frozen-range agreement: 100/216 = 46.3%; under-credit 114, over-credit 2.
- Pair occurrences: 220; {'valid': 200, 'ambiguous': 12, 'invalid': 8}.
- Unique source/answer-unit/evidence-unit pairs across passes: 110; {'valid': 100, 'ambiguous': 6, 'invalid': 4}.
- Strict valid-pair fraction: 90.9%; counting ambiguities favorably: 96.4%.
- All 133 grade-0/1 judgments audited against full source; 57 below frozen range. Low-grade capability boundary cases are flagged separately.
- Severe source findings: 9 (distance at least two grades or cross-entity support); missed central support: 30 sources.

User preliminary targets were 95% pair correctness and 90% range agreement. They are screening targets, not universal validity criteria. Repeat judgments and pairs are not independent observations. No population inference, confidence interval or scientific axis effect is estimated.

## Critical cases

- **Alibaba → AliExpress:** cell ab28bea4028bc50c3380, task b45ace7d463115a25e9814b9, grade 3 twice vs blind 0. Both a1/text1 and a3/text2 transfer Alibaba supplier conditions to AliExpress. Invalid in both passes. Another Alibaba source in the same cell correctly receives zero.
- **EMR demo centrality:** cell c9bb101b6dda08d9a391, task daf828a6b80a04316d7db35c, grade 2 twice vs 4–5. Explicit invitation to contact the vendor for an EMR/PMS demo supports the answer’s main recommendation, not merely peripheral quote text. H.R. task a663f56d5c1a2fc49c931619 is grade 1 twice vs 3–4. Its a4/title1 witness matches only “Similarly, H.R.” and is invalid; a5/text1 remains valid substantive demo support.
- **MetaMask centrality:** cell 0a04c9d1ba5852f0171a, task 67cefbcb7717d0c3e28d4b84, grade 2 twice vs 4–5. Setup invitation, supported assets, control and user count cover much of the actual answer; failure to answer pending-transaction verification is a separate J1 issue.
- **Prime delivery centrality:** cell 7d70633d57702bcb6b24, task fce0525562a09d5c847a519c, grade 2 twice vs 4–5. Free prescription delivery for Prime members is a main distinction repeated in the conclusion.
- **Unsupported specific views:** cell fe8032d957c9c74e4b8a, task bc74be285d016091e2223d79, a11/text1 is invalid. Generic customization/reporting/complexity does not establish specific Kanban/Gantt/List/Calendar views or automation. Full source still supports substantial selection criteria; pair error does not force a zero source grade.
- **Narration is supportable:** cells ebd04d863dcd705a9689 and 12303fd373d3dc7e65ee describe source contents. These descriptions remain substantive answer content even when unrelated to the requested technical procedure. No single source supports the source-set-wide absence conclusion.

## Uncertainty and zero/one review

- ac94acd69d4b1ddb5c80 employee-training sources 0/2/3 describe course creation/assignment/tracking or TMS administration/reporting. Frozen ranges 1–2 credit only the answer’s general capability premises. UI paths, button labels, deadlines, prerequisites, completion settings, real-time dashboard and export procedure are unsupported. Strict procedure-only interpretation can justify zero. These are not severe missed-central-support findings. Personal-trainer source 1 is correctly zero (different domain).
- c9975fd2bfc1c6d3e42d PMS billing versus payment processing is uncertain; frozen 1–2 is permissive to “may offer.” Zero is defensible. Cash App 2.75% pair is ambiguous because incoming-business-payment condition is omitted in the answer.
- 9d03b66f49d5d3e4943f bank-account requirement does not explicitly say prepare bank information; pair marked ambiguous.
- Generic collaboration/customization criteria and aggregated tutor attribution create remaining pair ambiguities. Exact reasons are in JSON; they are not silently treated as valid.
- No pure evidence_lacks cell: evidence-lacks claims occur alongside substantive source narration or advice, labeled mixed or narration. Query-style labels are manual informational/action/mixed proxies, never the measured semantic axis.
- Narrow grade anchors remain subjective: ranges preserve initial independent judgment, with post-unblind uncertainty explicitly recorded rather than changing the reference.

## Subgroups

|Group|Sources|In range/judgments|Valid/ambiguous/invalid pair occurrences|
|---|---:|---:|---|
|model: llama4|52|60/104|62/6/4|
|model: qwen38|56|40/112|138/6/4|
|method: Parallel-Expansion-v1|70|59/140|142/10/4|
|method: Reactive-Snippet-Loop-v1|38|41/76|58/2/4|
|style: direct|85|87/170|164/12/8|
|style: mixed|16|11/32|18/0/0|
|style: narration|7|2/14|18/0/0|
|query_style_proxy: action|75|76/150|112/4/2|
|query_style_proxy: informational|21|15/42|42/2/4|
|query_style_proxy: mixed|12|9/24|46/6/2|

Machine-readable details: `fresh-audit.json`. Frozen pre-grade reference: `fresh-blind-ranges.json`. Exact annotations: `fresh_pair_reviews.tsv`.
