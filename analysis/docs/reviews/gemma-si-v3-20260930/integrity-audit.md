Valerian, the saved Gemma judgments pass the structural and internal-integrity audit. This does not establish semantic validity. Five answers still expose prose source IDs, the original generator traces are absent, and the requested third repeat was not run.

## Scope and reproducibility

Read-only local audit of the archive at private HF revision `098d3f467b4a0f3fec6da2789043421428390770`, archive SHA256 `31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f`. All 57 files listed by the archive manifest reproduce their stored size and SHA256. The parent audit verified the content-addressed archive download. No remote connection, model inference, allocation, or repository edit was performed here.

Run the audit with Python 3.10 or later:

```sh
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 /private/tmp/gemma-v3-audit/integrity_audit.py
```

The script writes `integrity-audit.json`. It combines the existing SI-v3 task reconstruction and validation owner with independently computed schema, ID, substring, join, tie-group and metric checks. The three imported/relevant source files reproduce the bytes at both recorded run commits. There are 24,286 successful mechanical assertions and no recorded inconsistencies. This count includes repeated checks across final outputs, saved attempts and audit copies; it is not a count of independent observations or a software test suite.

## Saved observations

| Quantity | Original 20 | Fresh 20 |
|---|---:|---:|
| Distinct cells / prompts | 20 / 20 | 20 / 20 |
| Unique SI answer-source tasks | 106 | 108 |
| Gemma passes | 2 | 2 |
| Gemma SI judgments | 212 | 216 |
| Gemma J1 judgments | 40 | 40 |
| Recorded full-trace answer recoveries | 6 | 6 |
| Other answers labeled stored | 14 | 14 |
| Source text characters, median | 162 | 194.5 |
| Source text characters, min to max | 94 to 4,379 | 106 to 7,286 |

There are 40 distinct prompts, 40 distinct cells and 254 distinct tasks across both runs, with no overlap in any of those identities. The fresh selection's excluded-input SHA and excluded-prompt list reproduce the original bundle. Each run contains five cells per generator/method combination: Qwen38 and Llama4, each with Parallel-Expansion-v1 and Reactive-Snippet-Loop-v1. Conditions and search engines are not balanced. Original conditions are 9 ablated, 10 natural and 1 shuffled; fresh conditions are 6, 7 and 7. Each run is a diagnostic selection, not a representative population sample.

Source text length is strongly variable. Per generator/method distributions are in the JSON report. Neither a short title nor a short snippet is automatically a measurement failure, but the unit being assessed is the saved title/snippet, not a fetched whole page.

No measured semantic-axis coordinate is present in the exported cell/task records. The `readiness-question` prefix is a prompt identifier, not a measured position on the 0–1 axis. This export cannot validate axis strata, trends, or causal claims.

## Mechanical output and provenance checks

All 508 Gemma result records are present exactly once, with one start/end audit pair each. All return `ok=true`, finish reason `stop`, one validation attempt and no rejected outputs. Each request audit explicitly sets temperature 0 and the frozen task seed. The server's advertised sampling defaults do not override this evidence of explicit request settings.

All 428 Gemma SI outputs satisfy integer grades 0–5, empty matches for grade 0, one to three matches for positive grades, distinct ID pairs, and membership in that particular task's answer/evidence IDs. All 80 J1 values satisfy their separate integer 1–5 contract. Every resolved quote equals the selected stored passage. Evidence offsets index the original title or text field; answer offsets are relative to the selected answer unit, while unit offsets index the masked answer. These two coordinate systems are consistent in every record.

The embedded historical Nemotron results in the original inputs bundle were also checked: 252 final outputs and their saved SI validation attempts satisfy the same structural contract. They are comparison data, not additional Gemma observations.

Task IDs, prompt hashes, schema hashes, deterministic seeds, retry contracts, request hashes and complete task-record round trips reproduce. Pass inputs equal frozen inputs. J1 answers reproduce the declared full-answer hashes/lengths. Recomputed SI masking and span maps agree, and unmasking restores the J1 answer exactly. Source task associations use the same request/answer and the correct exported presented URL/position. Exported generator S-ID rankings resolve to the exported URL rankings.

These are internal checks. The archive omits the original generator result records, raw generator JSON, full trace event streams and original sealed-record manifest. Therefore it cannot independently prove the full-trace recovery, original stored-prefix match, complete evidence inventory, original S-ID-to-URL mapping, condition audit or generator fingerprint. Those remain metadata assertions supported by preparation code, not independently recovered source evidence. All 12 answers labeled `trace_full` are present in full within the judge tasks, but their claimed origin cannot be rechecked from this archive.

## Citation masking limitation

Five cells retain unbracketed prose IDs such as `Snippet S1`. This affects 34 unique SI tasks and 68 Gemma SI judgments across the two passes.

| Run | Cell | SI tasks affected |
|---|---|---:|
| Original | `9407a920b4e139ba05b2` | 7 |
| Original | `f223ac7238c28c2f044b` | 6 |
| Fresh | `ebd04d863dcd705a9689` | 7 |
| Fresh | `12303fd373d3dc7e65ee` | 7 |
| Fresh | `c9975fd2bfc1c6d3e42d` | 7 |

The implemented regex masks square-bracket/parenthesized S-ID markers and links matching observed source hosts. It does not cover prose S IDs. The saved masks faithfully reproduce that implementation; this is an uncovered citation form, not a broken offset or recovery map. A claim that all citation cues were removed would be too broad. Whether the remaining cues affected grades is unmeasured and needs a separately versioned diagnostic.

## Ranking and repeatability

All 80 saved cell metric records reproduce both the implementation and independent formulas. Positive grades remain tied groups; zero-grade sources remain separately unranked. Lexical sorting inside a group is display ordering and does not break a tie. Missing grades are joined as `None`, not zero. No naturally missing Gemma judgment occurs in these saved data, so the failure branch is not empirically exercised here.

| Metric | Original pass 1 / 2 | Fresh pass 1 / 2 |
|---|---:|---:|
| Top-source alignment, true / defined | 11/16 and 11/16 | 13/17 and 13/17 |
| Undefined ordering tau-b | 7 and 7 | 9 and 9 |
| Undefined important-source omission | 16 and 16 | 18 and 17 |
| All-zero cells | 4 and 4 | 3 and 3 |
| Cells with all sources tied at a positive grade | 1 and 1 | 2 and 2 |

Ordering tau-b is undefined for five original cells with constant grades over the generator list and two with fewer than two listed sources. Fresh counts are eight and one. No source of grade 4–5 means undefined omission, correctly saved as null. Among defined original omissions one cell has 1.0 and three have 0.0 in each pass; fresh defined omissions are all 0.0. Nulls must not be averaged as zeros.

Top alignment must be read with top-group size. Original groups reach six of seven sources; fresh groups reach seven of seven. For fresh cell `12303fd373d3dc7e65ee`, any first-ranked source would align because all seven are tied. Mean top-group fraction among defined cells is 34.5% original and 42.6% / 45.0% fresh. These are descriptive tie sizes, not a random-ranking performance baseline. Full per-cell group size and source count are saved in the JSON.

| Repeatability measure | Original | Fresh | Combined |
|---|---:|---:|---:|
| Identical SI grades | 104/106 | 105/108 | 209/214 |
| Identical complete cell grade vectors | 18/20 | 17/20 | 35/40 |
| Identical selected match lists | 105/106 | 107/108 | 212/214 |
| Identical J1 grades | 20/20 | 20/20 | 40/40 |

All five SI grade changes are one point. One original cell and two fresh cells change their ordering tau-b, while top-alignment booleans remain unchanged. Top-group membership changes in two original cells and one fresh cell despite stable top-alignment booleans. The fresh pharmacy maximum rises from 3 to 4 while its top-source membership stays unchanged. Stable grades or repeated errors do not establish entailment or centrality validity.

Only two passes exist for each 20-cell batch. The requested three repeats on 20–30 cells are incomplete. The saved passes reuse fixed task seeds and temperature zero, so they measure operational repeatability under those settings rather than uncertainty across seeds or models.

## Unrun diagnostic plan

Freeze a 20–30-cell subset before further inspection. After explicit allocation and wall-time approval, obtain a third unchanged pass with the same model revision, task/schema hashes, seeds and request settings. Report grade, match-list, complete-vector, top-group and tau-b changes, retaining undefined values.

Run controlled variants separately from the frozen SI-v3 baseline: remove residual prose source IDs; bijectively rename answer/evidence IDs; mask source titles; swap in same-topic unsupported snippets; and introduce a contradiction or a negation/qualification flip. Content-preserving renaming should preserve judgments; removing the only supported proposition should reduce support. Diagnose each variant against adjudicated support expectations rather than assuming every title removal must reduce a grade. No mutation, inference or new allocation was run in this audit.

Semantic support, missed support and centrality errors are handled by the independent blind-first source reviews. Structural validity alone cannot answer those questions.
