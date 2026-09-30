# SI-v3 diagnosis, retaining ordinal grades

## Scope and evidence

Valerian requested careful reading of Claude's latest handoff and conversation,
then clarification of poor source grading. Explicit preference: retain a
Likert-style scale. No judge replacement, inference or protocol edit requested.

Read the preceding three indexed handoffs, the SI plan, answer-style codebook and
summary, Astra steering brief, and the closing exchanges in Claude session
`4f0f51e8-7095-49e7-af32-66cf45ed5641`. Its final substantive response was
2026-09-30 04:27 UTC, corresponding to handoff commit `d8bc2e2`.
Inspected SI-v3 instructions, passage preparation, validator, citation mask,
metrics and trace evidence extraction in the active checkout.

No live cluster access occurred. Cluster status remains historical. Shell A,
Qwen progress and the HoreKa Llama refresh still need fresh evidence before action.

## Findings and corrections to earlier interpretations

- The pasted pilot contains 212 valid SI outputs across two passes over 106
  source-answer pairs, plus 40 valid J1 outputs. Passage-ID validity establishes
  neither entailment nor grade accuracy. Calling the binary support filter or
  J1 validated is premature without an independent reference assessment.
- First-pass grades 2 and 3 account for 80/106 judgments, about 75.5%.
  Concentration alone is not an error: sources can genuinely deserve ties.
- The Trello example chose `a1` with `title1`, a concatenation of many titles,
  and grade 2. `text13`/`text22` more directly support the second answer sentence
  about powerful features and flexible workflows. This is evidence of a weak
  selected match, not proof that all essential answer content was supported or
  that 5 was the correct grade. The answer recommends alternatives instead of
  implementing the requested automation. Fulfilment and support must stay separate.
- The visible source record combines multiple publishers' titles, promotional
  material and snippets. SI segmentation preserves that field; it does not create
  the concatenation. Trace extraction reads stored selected snippets. The upstream
  cause and whether each URL faithfully identifies its associated text remain
  unverified. This matters for claims about a particular link. Audit mapping before
  interpreting source grades as page attribution; preserve frozen evidence.
- Qwen repeated 47/53 grades exactly; Llama 53/53. In cell `9407a920b4e1`, grades
  `[1,5,1,1,1,0,1]` became `[1,1,5,1,1,0,1]`. Serving/batching nondeterminism is
  plausible, not established. The printed top-source-alignment value remained
  false in both passes for this cell, despite the change in the top source.
- The mask handles bracketed/parenthesized S IDs but leaves `Snippet S1` exposed.
- A source can substantively support a narrated account of itself. Automatically
  lowering such grades because the answer is unhelpful would change the construct.
  Analyse answer style and J1 separately; any exclusion needs an explicit estimand.
- `cell_metrics` uses the highest positive grade for top-source alignment; a 4/5
  is not required. All-positive ties can make alignment trivially true and tau-b
  undefined. Absence of 4/5 makes the important-source-omission metric undefined,
  not necessarily top alignment.

## Next proposed milestone, not executed

Keep 0-5 and passage provenance. Clarify the question as strength/centrality of
textual support for the actual answer, not proof of internal causal reliance.
Distinguish support verification from importance assessment, sharpen adjacent
anchors with worked examples, explicitly separate answer fulfilment, and preserve
legitimate ties. Test prompt-only candidates on the same development cases before
changing presentation/masking in separate versioned comparisons. Check against
independent reference models, then a fresh held-out set. Keep the earlier
no-human-annotator preference. Do not optimize merely for a wider grade histogram.

Only this handoff and its index were changed. No production code, data, tests,
cluster jobs or API calls changed. Documentation verified with `git diff --check`.
