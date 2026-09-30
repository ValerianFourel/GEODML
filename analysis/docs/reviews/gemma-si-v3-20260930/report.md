# Gemma SI-v3 audit, 30 September 2026

**Decision: do not approve SI-v3 with Gemma for production attribution yet.** The saved runs completed successfully and repeated grades are stable. The semantic review finds missed support, under-credit for central answer content, and a clear entity mismatch. Resolve the input-provenance and citation-masking gaps, then test a narrowly revised protocol on new cases. A shared answer outline is a candidate for testing, not an established remedy for entity errors.

The batch contains **40 distinct cells and 40 distinct prompts: 20 original development cases and 20 fresh cases**. The fresh freeze excludes every original prompt; there is no cell, prompt or task overlap. Each split has ten Qwen and ten Llama answers, with five cells per generator/method combination. There are 214 source tasks, each judged twice. All 40 cells and all their sources were audited. These examples now constitute development evidence for any subsequent prompt changes.

This is a preliminary **model-reference audit**, not human gold-standard accuracy or a scientific result. Two GPT-6 Astra reviewers independently assessed the original and fresh splits, with a third GPT-6 Astra reviewer checking integrity and metrics. Each semantic reviewer saved acceptable ranges before opening the result packet. Reviewers saw the same rubric, and each source had one semantic reviewer, not three independent votes. The primary reviewer subsequently checked selected decisive and disputed examples without blinding. No new Gemma inference or Slurm allocation was run.

## Evidence and reproducibility

The private [Hugging Face archive](https://huggingface.co/datasets/ValerianFourel/geodml-experiment-v2-paper-private/blob/098d3f467b4a0f3fec6da2789043421428390770/reviews/gemma-si-v3/exports/31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f.zip) was downloaded at revision `098d3f467b4a0f3fec6da2789043421428390770`. Its 1,253,914 bytes match SHA256 `31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f`. All 57 member hashes and sizes match the manifest, and the inventory matches the ZIP.

Detailed evidence is retained in [original audit](original-audit.json), [fresh audit](fresh-audit.json), [integrity audit](integrity-audit.json), [runtime audit](runtime-audit.json) and [aggregate counts](aggregate-audit.json). The [original](original-blind-ranges.json) and [fresh](fresh-blind-ranges.json) acceptable ranges remain unchanged. Reviewer narratives are also retained separately. Raw corpus files, server logs and credentials are not copied into this report directory. Exact result/task IDs and excerpt-level justifications are in the audit JSON files.

| Run | Original | Fresh |
|---|---|---|
| Directory | `gemma4-si-v3-ddc93fe` | `gemma4-si-v3-fresh20-2026093011` |
| Inference commit | `ddc93fe5342152d7c33f2abfc0658a48714d7dcc` | `135d4685f232776d2e8157e93e41657f3b062ee9` |
| Slurm job | 5171305 | 5171481 |
| Source tasks per pass | 106 | 108 |
| J1 tasks per pass | 20 | 20 |
| Repetitions saved | 2 | 2 |

Both runs used `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, SI-v3, explicit request temperature 0 and thinking disabled. The configuration records four A100-SXM4-40GB GPUs, tensor parallelism 4, BF16, concurrency 4, eager execution, context 73,728 and GPU memory fraction 0.85. SI output cap was 640 tokens and J1 cap 64. Recorded versions were torch 2.13.0, transformers 5.17.0, vLLM 0.28.0 and xgrammar 0.2.7. A server warning about model-default temperature 1 does not override the explicit temperature-0 requests. Prompt hashes, task seeds and preprocessing are recorded and checked against the pinned implementations.

## The ten requested checks

| Check | Finding | Screening status |
|---|---|---|
| 1. Correct material | Frozen answers, masks, offsets, source mappings, task IDs and result hashes reproduce internally. Original sealed generator shards/traces are absent. Five cells retain prose `Snippet S1`-style references after masking. | Internal consistency passes; mandatory external provenance not fully established. |
| 2. Mechanical validity | All 428 SI outputs and 80 J1 outputs validate. SI grades are integers 0–5; zero has no pairs; positive grades have 1–3 existing, distinct pairs. No failed output or silent zero substitution observed. | Pass for retained outputs. |
| 3. Actual support | Of 181 distinct selected pairs, reviewers mark 157 supported, 8 unsupported and 16 ambiguous. Partial-unit support is explicitly allowed. | 95% correctness not demonstrated. |
| 4. Missed support | Every zero/one source was read in full. In pass 1, 57/141 low grades fall below reference ranges, including 28 zeros with positive reference ranges. These include uncertain capability/absence cases, not 57 established central omissions. | Material concerns; separate clear misses from boundaries. |
| 5. Importance anchors | 110/214 first-pass grades and 223/428 grades across both passes fall within frozen reference ranges. Most differences are under-credit. | Far below the proposed 90% reference-agreement target. |
| 6. Separate support and quality | Useful-looking unsupported instructions often receive zero appropriately. Conversely, supported recommendations and narration are often under-credited in off-target answers. | Mixed; criterion confusion is plausible, not causally proved. |
| 7. Stability | 209/214 grades exactly repeat; all are within one point. Top-source membership changes in 3/40 cells; one source crosses grade 4. | Promising two-pass screen; requested third pass missing. |
| 8. Controlled changes | No unrelated-evidence, deletion, boilerplate, ID-renaming, equivalent-source or decisive-condition interventions were run. | Untested. |
| 9. Study subgroups | Model, method, style and source-length reference error counts are below. Measured semantic-axis coordinates are absent. | Too small and selected to establish subgroup validity. |
| 10. Honest rankings | Ties and undefined metrics reproduce. Constant grades do not manufacture an ordering. Top alignment must be read with group size. | Pass internally; alignment is not a validity target. |

The integrity reviewer performed 24,286 checks, including archived Nemotron outputs. This is an assertion count, not 24,286 independent examples. Gemma's retained-output denominator remains 508. With no failed Gemma judgments in this archive, missing-as-missing behavior is supported by the join logic, not a naturally occurring failed case.

Twelve cells, six per split, carry `trace_full` recovery metadata. Their frozen complete-answer hashes and reversible masks reproduce. This does not independently prove that trace recovery captured the original final answer, because the underlying generator records were not exported. The six `truncated` flags in each summary concern the upstream stored generator answer and recovery path; no Gemma response ended through a token-limit cutoff. All 508 finished with `stop`.

Citation masking leaves prose S IDs in five cells, affecting 34 source tasks and 68 SI outputs. This limits any broad claim of citation-blind judging. It is not an observed task-ID mismatch, and its effect on grades has not been experimentally measured.

## Semantic findings and uncertainty

| Evidence unit | Original | Fresh | Combined |
|---|---:|---:|---:|
| Distinct selected pairs | 71 | 110 | 181 |
| Supported / unsupported / ambiguous | 57 / 4 / 10 | 100 / 4 / 6 | 157 / 8 / 16 |
| Pair occurrences across both passes | 141 | 220 | 361 |
| Supported / unsupported / ambiguous occurrences | 114 / 7 / 20 | 200 / 8 / 12 | 314 / 15 / 32 |
| First-pass grades in acceptable range | 61/106 | 49/108 | 110/214 |
| Both-pass grades in acceptable range | 123/212 | 100/216 | 223/428 |

The confirmed-supported fraction is 157/181, **86.7%**. If every ambiguous pair were accepted, it would become 173/181, **95.6%**. Neither treating all ambiguities as errors nor accepting them all establishes a gold-standard accuracy. Eight distinct pairs are judged unsupported. Both-pass range agreement is **52.1%**, with 203 grades below and two above the frozen ranges. Narrow reference ranges and subjective centrality decisions can inflate disagreement. They cannot explain away the explicit entity mismatch or all the large centrality gaps.

For scale only, task-independent Wilson 95% intervals are 94.6–99.0% for exact repeat agreement and 98.2–100% for within-one agreement. The analogous confirmed-support interval is 81.0–90.9%, and first-pass range-agreement interval 44.7–58.0%. Sources and pairs share cells, the sample was selected, and reference judgments are uncertain. These descriptive intervals are optimistic for population inference; the counts do not establish the proposed population thresholds.

Decisive and disputed cases follow. Cell IDs are abbreviated; complete task IDs and exact quotations are in the JSON reviews.

- **Wrong entity, fresh `ab28bea4028b`.** Gemma assigns grade 3 twice to Alibaba supplier evidence for an answer about AliExpress wholesale pricing. The answer says “AliExpress wholesale pricing is structured around bulk orders”; the evidence says prices are offered “by suppliers on Alibaba”. Both selected pairs make the same entity transfer. Another Alibaba source in the cell receives zero. The primary review confirms this substantive mismatch.
- **Central recommendation under-credit, fresh `c9bb101b6dda`.** The EMR source explicitly invites the reader to contact the vendor for a live demo of EMR and practice-management tools. That is the answer's main action. Grade 2 twice contrasts with the blind 4–5 range; the primary review agrees that calling this secondary is difficult to justify. A neighboring H.R. demo offer receives grade 1. One H.R. witness selects only the sentence fragment “Similarly, H.R.”; the other selects substantive demo content. The empty fragment is a witness defect even though the source has real support elsewhere.
- **Supported off-target answer, fresh `0a04c9d1ba58`.** The request asks about setting up and testing a wallet for transaction verification. The actual answer mainly recommends MetaMask and Trust Wallet. MetaMask's text supports its assets, control, setup invitation and user count, yet receives grade 2 twice against 4–5. Failure to give the requested verification procedure belongs to J1; it does not make the facts central to the actual answer secondary. The primary review confirms substantial support, while the precise 4-versus-5 boundary remains subjective.
- **Missed positive support, original `224b306298e4`.** A POS title explicitly says “Connect Online and In-Store Operations”, matching part of the answer's definition. It gets zero twice. This does not support the additional real-time synchronization claim. The primary review agrees that there is missed partial support but does not treat the reviewer's exact 3–4 centrality range as settled.
- **Recommendation centrality, original `2dd070ba00e1`.** The answer recommends Wekan, Productive and Breeze; supporting sources receive grade 2, against independent ranges of 4 or 4–5. The answer's failure to implement the requested Trello automation should not by itself lower support for those recommendations.
- **Dropped condition, original `326525bc3e7b`.** A source supports “$20 after five points” but requires each qualifying purchase to be at least $20. The answer says one point for every order. Partial support for the reward rule is valid; the whole configuration is not established. In this same cell the Popupsmart source receives grade 1 against a blind 4–5 range. The primary reviewer challenges that range because generic points, tier and referral ideas omit the central numerical mechanism; 2–3 is also defensible. The original range is preserved, with the challenge recorded, rather than silently corrected after seeing Gemma's score.
- **Uncertain capability boundary, fresh `ac94acd69d4b`.** Employee-training sources say software creates, assigns and tracks courses. This supports a general premise but not specific UI paths, buttons, deadlines, widgets or real-time completion behavior. Blind 1–2 versus Gemma zero is a disputed partial-support boundary, not an established missed central procedure. The personal-training source is correctly distinguished from employee training.

All long source records were read, including the text outside Gemma's chosen witnesses. The selected 1–3 pairs are representative evidence; their number does not determine importance. Several records aggregate titles and passages from multiple pages. The audit judges the frozen material, not independently visited web pages or whether their claims are true in the world.

Original `212aa2c7a6c6` says the snippets lack API re-authentication instructions, then says they describe educational software and student information systems. One source cannot establish absence across the complete evidence set, but a local source can support the second sentence. Mixed narration/absence answers require that distinction. A purely global-absence answer should receive a separate eligibility policy before production; do not silently turn this policy choice into ordinary zero support or exclude all descriptive narration.

## Error rates by subgroup

The table uses **pass 1 only**, so repeated judgments are not counted twice. “Grade disagreements” means outside the frozen model-reference range, not verified human error. Unsupported and ambiguous pair counts are separate. Source length is the provided text length, excluding title, split descriptively at 1,000 characters. Style and query-orientation labels are reviewer annotations; they were not preregistered. The original reviewer labels its narration-dominant cell mixed, so narration is also represented there.

| Group | Grade disagreements / sources | Unsupported / selected pairs | Ambiguous / selected pairs |
|---|---:|---:|---:|
| model: qwen38 | 64/109 (58.7%) | 6/122 (4.9%) | 6/122 (4.9%) |
| model: llama4 | 40/105 (38.1%) | 2/59 (3.4%) | 10/59 (16.9%) |
| method: Parallel-Expansion-v1 | 78/140 (55.7%) | 6/121 (5.0%) | 9/121 (7.4%) |
| method: Reactive-Snippet-Loop-v1 | 26/74 (35.1%) | 2/60 (3.3%) | 7/60 (11.7%) |
| style: mixed | 18/30 (60.0%) | 1/20 (5.0%) | 1/20 (5.0%) |
| style: evidence_lacks | 8/9 (88.9%) | 0/3 (0.0%) | 0/3 (0.0%) |
| style: direct | 72/168 (42.9%) | 7/149 (4.7%) | 15/149 (10.1%) |
| style: narration | 6/7 (85.7%) | 0/9 (0.0%) | 0/9 (0.0%) |
| query_proxy: action | 66/154 (42.9%) | 4/106 (3.8%) | 12/106 (11.3%) |
| query_proxy: mixed | 12/23 (52.2%) | 1/35 (2.9%) | 3/35 (8.6%) |
| query_proxy: informational | 26/37 (70.3%) | 3/40 (7.5%) | 1/40 (2.5%) |
| source_text_length: short_lt1000 | 91/186 (48.9%) | 8/145 (5.5%) | 11/145 (7.6%) |
| source_text_length: long_ge1000 | 13/28 (46.4%) | 0/36 (0.0%) | 5/36 (13.9%) |

Qwen and Parallel have more reference-grade disagreements in this diagnostic sample. This is a concern for validation, not evidence of systematic population bias: source content, answer style and reviewers differ across these small groups. The long-record stratum has only 28 sources. It does not show a concentrated unsupported-pair problem, but that small count does not establish robustness. Manual informational/action labels are not the measured 0–1 semantic axis, and no axis association or causal claim follows from this audit.

## Repeat stability and ranking behavior

Exact source-grade agreement is 104/106 for original and 105/108 for fresh, **209/214 = 97.7%** combined. All 214 comparisons are within one grade; none changes by two or more. Complete cell grade vectors agree in 35/40 cells. All 40 J1 comparisons are unchanged. Distinct HTTP audit events establish that the saved second passes issued requests, rather than merely copying cached judgments. They reuse fixed task seeds and temperature zero, so they measure operational repeatability at those settings.

| Changed source | Pass 1 → pass 2 | Consequence |
|---|---|---|
| Original WinningHunter | 2 → 1 | Top-group membership changes |
| Original UPS | 1 → 2 | Top-group membership changes |
| Fresh PeopleManagingPeople OKR | 3 → 2 | Top-group membership changes |
| Fresh DPM Microsoft alternatives | 1 → 2 | Top group unchanged |
| Fresh Beckers pharmacy | 3 → 4 | Grade-4 set changes; highest-source membership unchanged |

Highest-source membership agrees in 37/40 cells, or 30/33 cells with a nonempty top group. Seven all-zero cells trivially retain an empty group and are not evidence of a useful ranking. No `[1,5] → [5,1]` switch occurs. Different valid witness choices need not match exactly.

Four original and three fresh cells have all-zero grades. A further one original and two fresh cells have constant positive grades and thus provide no ordering information. Kendall's tau-b is undefined in 7/20 original and 9/20 fresh cells because listed-source grades are constant or fewer than two are listed. These undefined values are preserved.

Omission is undefined without grade-4/5 sources: 16/20 original cells in both passes and 18/20 then 17/20 fresh cells. The fresh change is the pharmacy 3→4 boundary crossing. Original has one defined omission of 1.0; fresh defined omissions are all 0.0. These values remain scientifically provisional because importance grading itself is under review.

Generator top alignment is 11/16 defined original cells and 13/17 fresh cells in both passes. Groups can be as broad as 6/7 original sources and 7/7 fresh sources, so alignment alone would overstate selectivity. Per-cell top-group size and fraction accompany alignment in the integrity JSON. High generator agreement was not used as a validation target.

## Timing and operational errors

| Quantity | Original | Fresh |
|---|---:|---:|
| Pass 1 judging | 47.4 s | 67.1 s |
| Pass 2 judging | 44.9 s | 61.6 s |
| Both judging passes | 92.3 s | 128.7 s |
| Completed requests across both passes | 252 | 256 |
| Requests/s during judging | 2.73 | 1.99 |
| Median SI request latency, pass 1 / 2 | 0.716 / 0.704 s | 1.884 / 1.860 s |
| 95th-percentile SI latency, pass 1 / 2 | 4.328 / 4.256 s | 6.568 / 6.269 s |
| Stage-reported elapsed time | 464.1 s | 566.1 s |
| Recorded wrapper start to server finish | 533.2 s | 654.5 s |
| Server start to first judgment | 276.1 s | 331.2 s |
| Weight loading within startup | 38.90 s | 60.61 s |
| Last judgment to server finish | 30.18 s | 30.11 s |
| Parent allocation elapsed | 3,607 s | 3,601 s |

The 508 requests comprise **428 source-importance and 80 fulfilment judgments**. All succeeded on the first validation and transport attempt, with no truncated output. Judging took **221.0 seconds total**, while recorded wrapper spans total about **19.8 minutes**. Startup, validation and teardown dominate these small pilots. Neither number is total Slurm occupancy. Four GPUs over both parent allocations account for approximately **8.009 GPU-hours**. Exclusive-node accounting records 152 allocated CPUs, despite 32 requested task CPUs, and `mem=497500M`.

Original requests ran 06:29:45–06:31:17 UTC; fresh requests ran 08:56:22–08:58:30 UTC. Servers finished 06:31:47 and 08:59:00 UTC. Slurm's displayed local end times were 09:19:36 and 11:30:07, respectively, two hours ahead of UTC in these records. Both allocations later reached `TIMEOUT` and interactive steps were cancelled. This happened after successful judging and server shutdown; it did not truncate the saved outputs. Both server/controller exit codes are zero.

There were operational warnings and earlier setup errors, so “no errors anywhere” would be false:

- No server ERROR lines appear before shutdown. The original log has an AsyncLLM/EngineDeadError traceback after shutdown/SIGTERM and worker teardown. Fresh has no ERROR lines. Both have semaphore/shared-memory cleanup warnings. No CUDA OOM is present.
- Four HTTP 401 responses per run are intentional anonymous/wrong-key authentication probes, not failed judge requests. Eager-mode, unsupported SymmMem-on-A100 and first-request JIT latency warnings accompany successful execution.
- Generator metadata records six repaired/recovered cells per split with three upstream final-validation failures each. These are generator-history events, not Gemma request retries or new failures in these runs.
- The supplied shell history shows stale job-ID checks, a missing unprepared output directory, and an earlier record-verification failure before successful fresh preparation. The successful frozen bundle reports no rejected candidates; that does not erase the earlier failed freeze.
- The first Hub upload returned 403 because its token lacked write permission. The hidden write-token retry produced the verified archive used here. No token was requested or stored in this report.

## Smallest defensible next step

Keep these artifacts as the SI-v3 baseline and adjudicate the cited entity and centrality cases with a human. Obtain the original sealed generation/trace records to close the provenance check, and decide the eligibility rule for global-absence answers. Version any broader citation mask separately; do not silently rewrite these inputs.

Then prepare an isolated v4 diagnostic that separates supported propositions from their importance within a single shared view of the full answer. This tests the main centrality hypothesis. It must also include entity and condition checks, because an outline alone cannot fix Alibaba/AliExpress substitution. Preserve ordinal grades, genuine ties and missingness. No production corpus changes or v4 inference were performed here.

Before choosing v3, v4 or a different judge, run a third unchanged pass on a prespecified 20–30-cell subset and confirm any revision on genuinely new cases. Cases in this report cannot serve as an untouched confirmation set. The following controlled probes remain **unexecuted** and must stay outside the frozen corpus:

| Probe | Expected behavior and control |
|---|---|
| Replace title and text by unrelated material | Grade 0; replace both fields so the title does not preserve support |
| Remove the only central-support passage | Grade decreases; first verify no redundant title/text support remains |
| Add irrelevant boilerplate | Grade generally stable; retain all original evidence |
| Rename passage IDs bijectively | Same grade and equivalent supporting text; preserve all content |
| Provide two equivalent-support sources | Similar grades; ties allowed |
| Flip a decisive entity, condition or negation | Support for that specific claim disappears; unchanged background may remain supported |

A repeated tendency to invent support after these controls would justify reconsidering Gemma as judge. The current audit establishes an explicit entity error and broader grading concerns, but does not demonstrate that a particular v4 design or replacement model solves them. A subsequent inference allocation needs its own runtime estimate and approval.
