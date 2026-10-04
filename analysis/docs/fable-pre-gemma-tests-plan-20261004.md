# Pre-Gemma test plan for the Qwen and Llama generator corpus

Planning only, 2026-10-04. Written by Fable 5.1 in checkout
`.worktrees/generator-output-report` at commit `9e23a3d`
(`codex/generator-output-report-20261004`). No cluster command, job or code
edit was made. This document answers `analysis/docs/fable-pre-gemma-tests-prompt.md`.

Evidence labels used throughout:

- **[F]** verified by reading the named file in this checkout;
- **[P]** pasted or returned cluster evidence, with its date, recorded in a
  handoff or in the brief; not live state;
- **[A]** arithmetic on labelled numbers;
- **[H]** hypothesis, not a result.

Terms: *generator ranking* (the ordered URL subset the generator emitted),
*request relevance* (a judge's ordering of supplied sources by usefulness for
the request), *answer support* (a judge's grade of what each source supports
in the completed answer), *axis position* (`axis_1_percentile_0_1`, a measured
property of the prompt text; every axis result is an association). Embeddings
describe text; they are never treatments or confounders.

## 0. Decision summary

1. **Hold the auto-submitting Gemma sender now.** Yes, please pause it with the
   method that cancels no allocation and keeps the prepared plan. The judge
   fails its own frozen acceptance thresholds [F `analysis/docs/si-v4-r2-semantic-review-20261004.md`,
   `analysis/config/si_v4_evaluation.json` has `production_launch_authorized: false`],
   its documented failure mode (request fulfilment leaking into support grades)
   would bias exactly the axis association RQ2 asks about, and a 12,700–16,000
   GPU-hour pass [F `si-v4-gemma-five-hour-cost-20261004.json`] is 13–18× the
   cost of a keyword-clustered sample [A, §5] that answers the same question
   with proper uncertainty. Keep the preparation job's frozen inputs: they are
   the eligibility census this plan needs (§4, G8.1).
2. **Gates before any large Gemma pass:** (i) the SI-v4 validity gate passes on
   fresh, axis-stratified cells with the frozen thresholds and no error trend
   along the axis; (ii) the eligibility census and frozen analysis population
   exist (G8.1, G1.1, G1.8); (iii) the precision table from generator-level
   variance fixes the sample size (G8.3); (iv) the Qwen missingness check
   decides Llama-only, matched-prompt, or both (G1.2).
3. **Run first, on existing data, CPU only:** milestone 1 (generation rows +
   ledger + axis: coverage, Qwen missingness, axis-by-keyword variance,
   precision table, frozen population) and milestone 2 (trace features:
   truncation, repairs, queries, pools, generator-versus-compactor agreement,
   eligibility, pool-match census). The implementation prompt in §7 covers both.
4. **Valerian's current focus** (answer text along the axis) is served by
   reviewing the already-run text report, then rerunning it with trace-recovered
   full answers and a cap-sensitivity split (G4.1, G4.6), and only then spending
   GPU on answer embeddings (G4.3) after the relocation check passes (G2.1).

## 1. Questions for Valerian, and the assumptions used meanwhile

Blocking facts not available in this checkout:

1. **Gemma preparation freeze.** Does the CPU preparation (job 5179554 [P
   brief]) have a finished `cells.jsonl.gz` + `manifest.json` under its run
   root? If yes, give the root: its `counts` (`cells_ok`, `answer_stored`,
   `answer_trace_full`, `answer_trace_prefix`, `answer_trace_answer_mismatch`,
   `sources_unassessable`, `unique_source_importance_tasks`, unique map tasks)
   are the eligibility census G8.1 for free [F `analysis/scripts/prepare_source_importance_tasks.py`].
   *Assumption:* not yet available; M2 recomputes it.
2. **Axis map and registration on HoreKa.** Paths of `final-axis-map.jsonl`
   (sha256 `43189f68…`) and of `population-selection-records.jsonl` +
   `manifest.json` (`population-registration-v1`), which
   `report_axis_ranking_change.py` needs [F that script, `--registration`].
   *Assumption:* both are on `$W`; placeholders `$AXIS_MAP`, `$REGISTRATION`.
3. **Qwen dataset contract.** Does `$W/shared-hours/dataset/contract.json`
   carry `geodml-incremental-dataset-v1`? `report_latent_ranking_relationship.py`
   refuses otherwise [F]. Is the Qwen root still being written? *Assumption:*
   yes, live; every analysis records the ledger `event_count` and freezes a
   fingerprint list (G1.8).
4. **Relocation outcome.** Jobs page-extract 5179545 and page-relocate 5179548
   were live on 2026-10-04 [P handoff `2026-10-04_generator-output-report_handoff.md`].
   Did `relocation.json` pass (exit 0) and what did extract count
   (`answers`, `unique_pages`)? *Assumption:* unknown; all embedding-based
   tests (G2.1 fresh, G4.3, G5.3) wait on it.
5. **Archived prompt embeddings.** `relocate --qwen-embeddings final-audit/projections/qwen`
   replays archived prompt embeddings through the frozen map [F
   `analysis/docs/page_readiness_ordering.md`]. Are those the full LLM2Vec
   vectors for all 26,009 prompts, and in what format? If yes, RQ1's
   embedding-distance neighbourhood test (G3.3) is CPU-only.
6. **The answer-text report.** Paste `$W/reviews/answer-readiness-20261004-text-0941/report.md`.
   Until it is reviewed, G4.1's status is "run, unreviewed"; what has been seen
   decides which text outcomes can still be frozen as confirmatory.
7. **Original-500 Nemotron judgments.** `agentic_paper_dataset_observed_status.json`
   records 4,594 of 12,000 recorded-conversation judgments (user-supplied
   JUPITER output, capture time null) [F]; EXPERIEMENTV2.md records 1,298
   validated claims on 2026-09-22 [F]. Are those 500 prompts a subset of the
   26,009 with the same `prompt_id`s, and where do the outputs live? They carry
   `ideal_relevance_ranking` and `realized_support_ranking` under a protocol
   that is not ranking-blinded; they could pilot the RQ2 join design (G8.2),
   never substitute for it.
8. **Keyword text.** Query–keyword overlap (A) needs the keyword string per
   `primary_keyword_id`. Which file maps keyword IDs to text?

## 2. Where files contradict the brief or AGENTS.md

- The repository-root `AGENTS.md` given to this session says "Nemotron judges
  which sources the answer rested on". The files say otherwise: the Nemotron
  Nano v4 diagnostic returned 0/107 numeric grades and 18/20 unusable maps and
  is "not a viable replacement for Gemma" [F `si-v4-nemotron-semantic-review-20261004.md`];
  Gemma is the planned SI-v4 judge and is itself unaccepted [F r2 review]. The
  checkout's own `AGENTS.md` does not name a judge. Trust the files: no accepted
  answer-support judge exists today.
- The paper's design table reports a Llama registry of 310,380 cells, 1,728
  short of 312,108 [F `sections/methods.tex`, 2026-10-01 JUPITER audit]. The
  2026-10-04 HoreKa report counts 312,090 verified cells on 26,008 prompts with
  6 failures [P]. 26,008 × 12 − 6 = 312,090 [A], so the HoreKa Llama registry
  has 26,008 prompts: one of the 26,009 is absent, not 144. Different snapshot,
  not a contradiction, but the paper's number is stale. M1 names the missing prompt.
- `EXPERIEMENTV2.md` documents the 500-prompt pilot (6,000 cells per model, the
  Nemotron recorded-conversation judge) [F], not the 26,009-prompt corpus. The
  corpus design is in `agentic_search_retrieval_protocol.md` and the paper.
- The brief lists `report_axis_permutation_study.py` for reuse. Its fitter
  `fit_axis_rankings` raises unless every row shares one candidate pool [F
  `axis_permutation_metrics.py`], and its config needs plan manifests. For the
  agentic corpus, reuse `fit_axis_rankings` directly on pool-matched strata
  (G3.4) and `ranking_agreement` for everything else; do not drive the study
  report.
- The brief says the Reactive method keeps "3 per search". Correct, but the
  protocol also allows the generator to finish without searching further and
  forces a fourth call after a third search [F protocol]; `search_count` 1–3 is
  the generator's choice, which G5.1 treats as an outcome.

Everything else in the brief's §2 matches the files or is pasted evidence that
this checkout cannot verify (coverage counts, descriptive findings, job IDs).

## 3. A. Measurement inventory

All quantities below come from the sealed generator datasets (`data/<table>/part-*.jsonl`
with manifests), the task ledger (`control/task-ledger`), and the archived
axis artifacts. Readers: `agentic_cells.completed_generator_refs` / `iter_cells`
(generation + trace + prompt + membership), `report_generator_outputs.load_cells`
(generation rows only, with ledger states), `agentic_judging._trace_evidence`
(final presented evidence), `prepare_source_importance_tasks.judged_answer` /
`generator_output_record` / `provenance` [F]. **CPU** unless marked **GPU**.

### Record schema actually present [F `agentic_dataset.py`, `agentic_search.py`, smoke writer]

| Table / artifact | Fields used |
|---|---|
| `generations` | `cell_id`, `prompt_id`, `method`, `engine`, `condition`, `ranking` (URLs), `answer` (stored, cut at `FINAL_ANSWER_MAX_CHARACTERS` = 1,200), `final_snippet_count`, `search_count`, `trace_sha256`, `condition_audit{target_url, calls[], target_observed, target_removed_count}` |
| `traces` | `method_id`, `condition`, `user_prompt_sha256`, `search_engine`, `compactor_model_id/_revision`, `bounds`, `events[{event_index, event_type, payload}]` |
| trace `llm_call` | `purpose` ∈ {parallel_query_expansion, parallel_final, reactive_action, reactive_forced_finish}, `attempt`, `request{prompt, response_schema, force_finish}`, `raw_output`, `validation_error`, `transport_error`, `parsed_output` |
| trace `controller_repair` / `_rejected` | `trigger_validation_error`, `dropped_ranking_references`, `answer_truncated`, `forced_finish`, `malformed_json_recovered`, `repaired_output` |
| trace `search` / `search_error` | `engine`, `query`, `limit` (20), `raw_payload`, `snippets[{url,title,text}]` / `error` |
| trace `deduplication` (Parallel) | `input_count`, `output_count`, `output_urls` |
| trace `condition` | `condition`, `input_snippets`, `output_snippets`, `output_urls` |
| trace `compaction` | `query`, `top_k`, `scorer{model_id, model_revision}`, `scored_snippets[{url,title,text,score,source_index}]`, `selected_snippets[…]` |
| trace `observation` (Reactive) | `iteration`, `query`, `snippets` |
| `prompts` | `prompt_id`, `prompt_text`, `prompt_sha256`, `source` |
| `keyword_memberships` | `prompt_id`, `keyword_ids`, `primary_keyword_id`, `primary_priority_rank` |
| `task_definitions` + ledger | `task_id`, `model`, `method`, `engine`, `condition`, `claim_identity{model_id, model_revision, protocol, request_sha256}`; latest state (`completed`, `checkpointed`, `running`, never claimed, terminal failed) with `record_references` |
| `diagnostics`, `failed_attempts` | `cell_id`, `status`, `llm_calls`, `elapsed_seconds`; failed traces with `error` |
| `final-axis-map.jsonl` | `candidate_id`, `consensus_axis_1_z`, `axis_1_percentile_0_1` |
| `population-selection-records.jsonl` | `candidate_id`, `assigned_axis_1_0_1`, `observed_axis_1_percentile_0_1`, `axis_bin` |
| final-audit `merged/{qwen,mistral}`, battery, maps, `projections/` | per-view z, z-scaling, Procrustes rotation, frozen ridge maps, archived prompt embeddings |

### Quantities by level

| Level | Quantity | Source | Compute |
|---|---|---|---|
| Cell | ranking length; empty ranking; share of presented evidence ranked | `generations.ranking`, `final_snippet_count` | CPU |
| Cell | rank and reciprocal rank of the frozen target when observed; target observed; target removed count | `condition_audit` | CPU |
| Cell | search count; finish-before-budget (Reactive 1 or 2 searches); forced finish | `search_count`; `llm_call.purpose == reactive_forced_finish` | CPU |
| Cell | stored answer length; at-cap flag (= 1,200); truncated; recovered full-answer length; answer source ∈ {stored, trace_full, trace_prefix, trace_answer_mismatch} | `answer`; `controller_repair.answer_truncated`; `judged_answer` | CPU |
| Cell | raw emitted ranking vs stored; dropped references; final attempts; validation failures; transport errors; malformed JSON recovered | `generator_output_record` | CPU |
| Cell | provenance status (final visible evidence == trace evidence; final raw answer == stored) | `provenance` | CPU |
| Cell | presented evidence in order; presented page ids (sha256 of `title\ntext`); position of each ranked item; top-1 presented position | `_trace_evidence`, `page_readiness_ordering.page_text/page_id` | CPU |
| Cell | generator ranking vs cross-encoder score order (Kendall on ranked items); top score; score margin between kept and first dropped | `compaction.scored_snippets/selected_snippets` | CPU |
| Cell | runtime: elapsed seconds, LLM calls | `diagnostics` | CPU |
| Query | count; text; token length; query–prompt Jaccard; pairwise Jaccard among the three parallel queries; successive Reactive drift; query–keyword overlap (needs keyword text, Q8) | `search.query`, `llm_call.parsed_output`, `prompts.prompt_text` | CPU |
| Query | snippets returned (≤ 20); zero-result queries; search errors | `search.snippets`, `search_error` | CPU |
| Query | embedding distance query↔prompt | LLM2Vec | **GPU** |
| Pool | retrieved pool (union of `search.snippets` URLs); dedup input/output; condition removals; compacted pool (7 or ≤ 9); unique hosts at each stage (`source_importance.source_host`) | trace | CPU |
| Pool | which parallel query supplied each kept snippet; score concentration | `search` ∩ `compaction` | CPU |
| Pool | between-prompt overlap within keyword: identical presented page-id sets; identical URL sets; Jaccard | per-cell presented sets | CPU |
| Pool | cross-model same-cell pool identity | both models' presented sets | CPU |
| Answer | words, sentences, list lines, questions, digits, currency, URL mentions, second person, action verbs, immediacy, hedges, explanatory words (per 100 words) | `answer_readiness.text_features` | CPU |
| Answer | cites evidence IDs (`[S1]`-style markers); names a source URL | `source_importance.mask_answer_citations` | CPU |
| Answer | lexical overlap with the prompt; with presented snippets overall and per source (content-token share). *Lexical overlap, not answer support.* | answer + `_trace_evidence` | CPU |
| Answer | position on the prompt axis (`answer_axis_percentile`, `answer_minus_prompt_axis`, outside-range flag) | `answer_readiness analyze` after embed/merge | **GPU** then CPU |
| Source text | title/text length; host, TLD, path depth, http/https, query string; digits/price markers; occurrences across cells; ranked-vs-unranked indicator; compaction score | pages + presented sets | CPU |
| Source text | page readiness z and percentile on the prompt scale | `page_readiness_ordering embed/merge` | **GPU** then CPU |
| Prompt | axis percentile and z; keyword id; priority rank; prompt text and length; `source` field (prompt generator?, Q8-adjacent); completed cells per prompt per model; per-prompt means of all cell quantities | axis map, memberships, prompts, ledger | CPU |
| Prompt | pairwise embedding cosine distance within keyword (RQ1 neighbourhood) | archived embeddings (Q5) | CPU if archived, else **GPU** |
| Keyword | prompts per keyword (18–30); completed prompts per model; within-keyword axis range and variance; priority rank | memberships, axis | CPU |
| Reliability | ledger state counts; anomalies (`completed_without_exactly_one_generation`, `completed_with_unverified_generation`); failed attempts; repairs; validation failures; forced finishes; search/transport errors | ledger, `load_cells`, traces | CPU |

## 4. B. Test catalogue

Columns: **Q** question; **Unit/dep** unit of analysis and dependence; **Est.**
estimator, uncertainty, multiplicity; **Control** comparison or control;
**Code** existing script or gap; **Cost** CPU/GPU and rough cost (runtimes are
unmeasured unless stated); **Read** what +/0/− means; **Claim** strongest
allowed claim; **Pri** priority (P0 gate, P1 next, P2 later).

Uncertainty conventions for every test: keyword-cluster bootstrap for intervals
(`cluster_bootstrap.py`, keywords as clusters; cells of a prompt and prompts of
a keyword are never treated as independent) and within-keyword permutation of
axis positions for null distributions (`_association`), with ≥ 2,000 draws for
anything confirmatory (200 draws floor p at 1/201 ≈ 0.005, which is why every
reported p sat at 0.00498 [P]). Report effect sizes with intervals first;
Holm-adjust within each pre-registered confirmatory family; everything else is
exploratory and unadjusted.

### Group 1. Data integrity and validity gates

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G1.1 | Do all verified cells join to a prompt, keyword, axis position and task definition, and what is coverage by model × method × engine × condition? | cell; counts | counts; no uncertainty (census) | design 26,009 × 12 per generator | `report_generator_outputs` inventory (ran: 312,090 / 287,919 [P]); gap: frozen population manifest with fingerprint list, ledger `event_count`, shard manifest hashes | CPU, minutes | any `prompts_without_axis` > 0 or anomaly > 0 blocks | census | P0 |
| G1.2 | Are the missing Qwen prompts (keyword-priority order) different in axis position or keyword from the completed ones? | prompt; clustered in keyword | axis-decile and priority-decile distribution of completed vs never-claimed prompts; max CDF gap; per-keyword completion fraction | Llama (complete) as reference | gap (small): uses `load_cells` states + memberships + axis | CPU, minutes | large gap ⇒ Qwen-only associations describe a selected subset; use the matched-prompt set (both models complete) for cross-model work | census | P0 |
| G1.3 | How many rankings are empty, outside the evidence, or duplicated, by stratum? | cell | shares with keyword-bootstrap CI | by engine (Llama PE searxng 25% empty [P]) | `describe` (`empty_ranking_share`), `check_axis_ranking_change.describe` (length counts, URL shapes), `prepare_source_importance_tasks` statuses | CPU | decide pre-registration: empty rankings excluded from order tests, kept as outcome in length tests | census | P0 |
| G1.4 | How often is the stored answer cut at 1,200 characters, and can the full answer be recovered from the trace? | cell | shares of `stored` / `trace_full` / `trace_prefix` / `trace_answer_mismatch` by model × method; recovered length distribution | Qwen PE median 1,200 [P] | `judged_answer` (already in the Gemma preparation freeze, Q1); gap: census without judge tasks | CPU trace pass | high `trace_prefix` share ⇒ text measures must use rates per 100 words only | census | P0 |
| G1.5 | Do repairs, validation failures, forced finishes and search errors vary by stratum or axis? | cell → prompt means | shares; Spearman with axis; keyword permutation | by model, method | `generator_output_record`; gap: aggregate | CPU trace pass | an axis trend in failures confounds text and ranking outcomes; stratify or exclude | association | P1 |
| G1.6 | What are the 163 Qwen unverified-generation completions and 6 Llama failures? | task | list by writer/shard; exclude and record | — | `load_cells` anomalies, `failed_attempts` | CPU | record only | census | P1 |
| G1.7 | Which prompt of 26,009 is absent from the Llama registry; which Qwen tasks were never registered? | task | set difference against the design | — | gap (tiny) | CPU | — | census | P1 |
| G1.8 | Is the analysis population frozen? | dataset | fingerprint list + ledger snapshot + shard hashes written once; later runs verify | — | gap (part of M1) | CPU | any later analysis on a different list is a new population | — | P0 |

### Group 2. Measurement checks

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G2.1 | Do the archived artifacts reproduce the axis map, and do the encoders still place text where they did? | prompt | exact recompute (1e-9), map replay (1e-4), fresh re-embedding Spearman ≥ 0.999 on 512 prompts | — | `page_readiness_ordering relocate` (job 5179548 was live [P]; outcome unknown) | CPU for the first two; **GPU** for fresh | fail ⇒ no embedding-based test (G3.3, G4.3, G5.3) is interpretable | measurement check | P0 for embedding tests |
| G2.2 | Is the axis mostly topic? Between- vs within-keyword variance of `axis_1_percentile_0_1`; within-keyword range distribution | prompt in keyword | variance share (ICC), range quantiles, share of keywords with range ≥ 0.5 | — | gap (tiny; axis map + memberships) | CPU, seconds | high ICC ⇒ within-keyword contrasts have little leverage and pooled axis associations are mostly between-topic; report both always | description | P0 |
| G2.3 | How much of the axis is predictable from prompt surface features (length, imperative, question mark, second person)? | prompt | R² of a small linear model; keyword-bootstrap | — | gap | CPU | high R² says the axis is partly surface; not a validity failure, but worth reporting | description | P2 |
| G2.4 | Does axis distribution differ by prompt generator (Qwen3-32B vs Gemma-4-31B wrote 13,515 / 12,494 prompts [F methods.tex]) and should it be a stratum? | prompt | distribution by `prompts.source` if that field carries it | — | gap; unknown whether `source` encodes it | CPU | if it differs, add as stratum in confirmatory models | description | P1 |

### Group 3. RQ1 on generator orderings

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G3.1 | How many within-keyword prompt pairs share an identical presented pool (by page id; by URL), per model × method × engine × condition, and how many strata meet the fitter's 6 train + 2 test distinct questions? | prompt pair in keyword | counts; Jaccard bins | by method (PE vs Reactive) | `page_readiness_ordering extract` observations; gap: census script | CPU trace pass (or extract output) | few exact matches ⇒ G3.4 is infeasible at scale and RQ1 must be answered with membership-vs-order decomposition (G3.2, G3.6) | census | P0 |
| G3.2 | Within keyword, does ranking distance grow with axis gap, separately for membership (URL-set Jaccard) and order on shared sources (Kendall)? | prompt pair; pairs share prompts | per-group ρ(axis gap, distance), median within-keyword ρ, gap bins; gap: keyword-cluster bootstrap on binned means | each design cell separately; PE vs Reactive | `report_axis_ranking_change.py` (needs `$REGISTRATION`); `check_axis_ranking_change.py` validates | CPU | membership distance rising faster than order distance ⇒ retrieval, not preference, drives change | association | P0 |
| G3.3 | Same as G3.2 with full prompt-embedding cosine distance instead of axis gap (the paper's neighbourhood definition) | prompt pair in keyword | as G3.2 | axis-gap version | gap: distance loader for `compare_prompts` (Q5) | CPU if archived vectors exist | local stability high at small distance ⇒ nearby prompts keep leading sources | association | P1 |
| G3.4 | On exact-pool strata, does an axis-dependent Bradley–Terry model predict held-out pair order better than intercept-only? | task in stratum; held-out by question hash | `fit_axis_rankings`; pooled held-out log-loss / Brier reduction with keyword-cluster bootstrap over strata | intercept-only baseline | `axis_permutation_metrics.fit_axis_rankings`; gap: thin driver over extract observations | CPU | depends on G3.1 feasibility | directional association within matched evidence | P1 |
| G3.5 | Relaxed-pool directional model (observed pairs only, pool not required identical) | pair | BT on observed pairs with source fixed effects | — | new protocol; design decision for Valerian | CPU | — | would need preregistration | P2 |
| G3.6 | Does membership change exceed order change along the axis (directional contrast)? | prompt pair | difference of G3.2 slopes with keyword bootstrap | — | from G3.2 outputs | CPU | — | association | P1 |
| G3.7 | Does the generator ranking just echo the cross-encoder score order, and does the deviation change with axis? | cell → prompt | Kendall(ranking, score order); share top-1 = top-scored; Spearman with axis; keyword permutation | PE (presented order = score order) vs Reactive (presented = observation order) | gap: trace feature | CPU trace pass | high agreement ⇒ "generator ranking" effects are compaction effects; report before any RQ1 claim | association | P1 |

### Group 4. Answer text along the axis (Valerian's focus)

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G4.1 | How do heuristic text measures change with axis, and are the changes robust to the 1,200-character cap? | answer (natural) → prompt means | Spearman + keyword permutation; within-keyword high−low; three populations: all stored, untruncated only, trace-recovered full | by model, method × engine | `answer_readiness.py text` (ran, unreviewed [P]); gap: `export --recover-full-answers` via `judged_answer` | CPU | rates per 100 words stable across populations ⇒ cap-robust; totals not | association | P0 |
| G4.2 | Which words distinguish answers to higher- vs lower-axis prompts of the same keyword, with uncertainty? | word; keyword-blocked | informative-Dirichlet log-odds (exists); gap: keyword-bootstrap stability of top lists | per model | `distinctive_words` | CPU | — | description | P1 |
| G4.3 | Do answers move along the axis as much as their prompts (slope of answer percentile on prompt percentile)? | answer → prompt means | slope, Spearman, within-keyword contrast; keyword permutation | by model, method | `answer_readiness.py analyze` after `page_readiness_ordering embed/merge` (prepared, not run) | **GPU** embed ≈ 200k unique natural answers × 2 views (unmeasured; one shard first) | slope near 1 ⇒ answers track prompt readiness; near 0 ⇒ generators normalise | association (out-of-domain coordinates) | P1 after G2.1 |
| G4.4 | Do answers copy the prompt more at high axis? | answer | token-overlap share; keyword permutation | — | gap | CPU | — | description | P2 |
| G4.5 | Lexical overlap of the answer with each presented source: does the top-ranked source have the highest overlap, and does that change with axis? *Not answer support.* | cell → prompt | share top-1 = max-overlap; Spearman with axis; keyword bootstrap | by model, method | gap: trace feature | CPU trace pass | low agreement ⇒ RQ2 divergence is plausible even before judging; informs sampling | description, judge-free | P1 |
| G4.6 | Is truncation itself associated with axis? | cell → prompt | share truncated by decile; Spearman | — | from G1.4 | CPU | yes ⇒ length results need the cap-sensitivity split before interpretation | association | P0 |

### Group 5. Search and evidence behaviour along the axis

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G5.1 | Do query count, query length, query–prompt overlap, inter-query diversity, Reactive drift and finish-before-budget change with axis? | cell → prompt | Spearman + keyword permutation; deciles with keyword-bootstrap CI | PE (fixed 3 queries) vs Reactive | gap: trace feature | CPU trace pass | Reactive searching more at high axis (known [P]) with lower query–prompt overlap ⇒ reformulation, not repetition | association | P1 |
| G5.2 | Do retrieved pools differ: snippets per query, zero-result queries, dedup rate, unique hosts, score concentration, which query supplied kept snippets? | cell → prompt | as G5.1 | ddg vs searxng (explains Llama PE searxng empties?) | gap: trace feature | CPU trace pass | — | association | P1 |
| G5.3 | Do generators rank action-ready pages higher, more so for action-ready prompts? | choice set in answer; keyword clusters | top-k Plackett–Luce with page z × (axis − 0.5) and position effects; keyword bootstrap; within-keyword permutation | PE vs Reactive; Qwen-only / Mistral-only page views | `page_readiness_ordering analyze` (needs GPU `embed`) | **GPU** embed of unique presented pages (count unknown, Q4) | positive interaction ⇒ preference for action-ready pages grows with prompt readiness | association (out-of-domain page coordinates) | P1 after G2.1 |
| G5.4 | Which cheap source features (host, TLD, depth, length, digits) predict being ranked, and does it vary with axis? | presented source in cell | within-cell conditional logit or ranked-vs-unranked contrasts; keyword bootstrap | — | gap | CPU | — | description | P2 |
| G5.5 | Is the frozen target retrieved less often for action-ready prompts (retrieval effect behind the −0.07/−0.04 selection association [P])? | cell → prompt | `target_observed` rate by decile; Spearman | by engine | gap (small; `condition_audit.target_observed`) | CPU | yes ⇒ target-selection decline is retrieval, not ranking | association | P1 |

### Group 6. Generator, strategy and engine as conditions

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G6.1 | Direct paired differences: Llama − Qwen on the same cell; PE − Reactive on the same prompt/engine/condition; ddg − searxng; and differences in axis slopes (interactions) | matched pair → keyword clusters | `cluster_bootstrap.paired_difference`; slope-difference bootstrap | — | `compare` (ranking agreement) exists; gap: paired outcome differences and interaction CIs | CPU | report the difference with CI; never "significant here, not there" | descriptive comparison of conditions | P1 |
| G6.2 | Is cross-model top-1 disagreement (42→55% [P]) membership or order? | same-cell pair | disagreement conditional on identical presented pool vs different pool; by decile | — | gap (small; needs presented sets from M2) | CPU | mostly different pools ⇒ different queries, not different preferences | description | P1 |

### Group 7. Ablation as a design check (shuffle out of scope)

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G7.1 | Was the target shown / retrieved-not-shown / not retrieved on the natural path, and is it absent from the ablated evidence? | (prompt, method, engine) group | class counts by stratum | — | `audit_condition_manipulation.py` (exists; also classifies shuffle; report, do not interpret shuffle) | CPU trace pass; run `--prompt-ids` 1,000-prompt sample first | `target_in_ablated_evidence` must be 0 | design check | P1 |
| G7.2 | When the target was shown, how does the ranking change under ablation and what replaces it? | group | top-1 change, replacement URLs | — | `ablation_exposure` fields | CPU | — | design check | P2 |

### Group 8. De-risking and shrinking judge inference

| ID | Q | Unit/dep | Est. | Control | Code | Cost | Read | Claim | Pri |
|---|---|---|---|---|---|---|---|---|---|
| G8.1 | Eligibility census: cells by SI status (`ok`, `no_observed_sources`, `ranking_outside_evidence`, `evidence_count_mismatch`, `trace_answer_mismatch`, `missing_request_or_answer`), answer source, assessable sources per cell, unique maps U and source tasks P | cell | counts by stratum and decile | — | `prepare_source_importance_tasks.py --protocol si-v4` produces it; the Gemma preparation freeze already did (Q1) | CPU trace pass | U and P fix the exact judge cost (U map + P source calls [F source-importance-v4.md]) | census | P0 |
| G8.2 | Which cells carry information for RQ2: ranking length ≥ 2, ≥ 2 assessable sources, non-abstention answer, verified provenance, within a keyword with axis spread; which cells share a pool across models? | cell | counts; eligibility score | — | gap, from M2 features | CPU | defines the judge sampling frame | census | P0 |
| G8.3 | Precision from generator-level variance: keyword-cluster bootstrap SE of the axis slope / Spearman for proxy outcomes (ranking length, empty ranking, target selected, at-cap) at 50 / 100 / 200 / 400 / all keywords; within-keyword and within-prompt ICCs | prompt means in keyword | subsampled cluster bootstrap (seeded) | natural condition only; per model | gap (small; `cluster_bootstrap`, `_association`) | CPU | SE curve flattening by ~200 keywords [H] ⇒ sample size for the judge design | design calculation | P0 |
| G8.4 | Judge-validity gate (not generator data): SI-v4 on 120 fresh cells (96 core from 48 paired prompts + 24 stress from 12), 24 cells repeated 3× in fixed-map and end-to-end modes, references frozen before grades, thresholds in `si_v4_evaluation.json`; **add** axis-stratified selection and a check that grade error does not trend with axis | cell/source; keyword clusters | thresholds [F]; error-vs-axis Spearman | development cells excluded | existing SI-v4 tooling | **GPU** ≈ 264 cell-equivalents ≈ 1.3–1.5 warm node-hours [A from cost file] ⇒ 2–3 one-hour bouts | fail ⇒ no large pass with this judge | acceptance | P0 gate |
| G8.5 | Sampled judge design (§5): stratified by keyword-level axis spread, keywords as clusters, natural condition, both models on matched prompts, all method × engine | keyword | size from G8.3 | — | plan freeze | **GPU** see §5 | — | — | P0 |

## 5. C. Gemma decision

**Recommendation: hold the sender until the gates in §0 are in. Valerian decides.**

Why, in order of weight:

1. **Judge validity.** R2 misses every repeat and completion threshold (eligible
   completion 95.65% vs ≥ 99%; fixed-map exact repeat 88.71% vs ≥ 95%;
   positive top-group agreement 84.3% vs ≥ 95%; warm cost 7.94× vs ≤ 3×) and
   fails semantically on real cells (omitted substantive content, zero grades
   for directly supporting sources, wrong-source full support) [F r2 review].
   `production_launch_authorized` is `false` [F `si_v4_evaluation.json`]. A
   full pass would spend 12,700–14,960 GPU-hours (800-bout ceiling 16,000) [F
   cost file] on grades the project has already decided not to trust.
2. **The failure mode is axis-correlated.** The r2 review itself notes that
   penalising a source because the answer fails the request "could distort
   comparisons along the observational readiness axis when answers differ in
   their ability to fulfil increasingly specific requests" [F]. Action-ready
   prompts are exactly where answers are shorter and more specific [P]. A judge
   whose error moves with the axis biases the RQ2 association in the direction
   of interest; more cells make that bias more precise, not smaller.
3. **Uncertainty is governed by keywords, not cells.** With keyword-cluster
   bootstrap as the inference unit, the design effect is roughly
   1 + (m − 1)·ICC for m cells per keyword [standard; the ICC is what G8.3
   measures]. The corpus has 1,011 keywords [F]; the two non-natural conditions
   add near-duplicate cells for RQ2 (same prompt, nearly the same evidence) and
   are 2/3 of all cells [A]. Judging all 600k cells buys little precision over
   judging the natural cells of a few hundred keywords.
4. **The Qwen population is not frozen.** 287,919 of 312,108 Qwen design cells
   are verified, in keyword-priority order [P]. A full pass now freezes a
   non-random Qwen subset and needs a second pass later; a keyword-clustered
   sample on matched prompts does not.
5. **The census comes first anyway.** The preparation freeze is G8.1; the frozen
   plan should be built from it after G8.2–G8.3, not before.

Risk of holding: lost queue position and calendar time. Mitigation: the
validity gate (G8.4) and the sampled design need only a handful of bouts and
can start as soon as the r3 prompt revision exists.

**Sampled judge design (default proposal, to be resized by G8.3):**

- Frame: cells with SI status `ok`, natural condition, both generators on the
  same prompt (matched-prompt set), all four method × engine strata; keywords
  as sampling units, stratified by within-keyword axis range so every stratum
  has leverage along the axis.
- Size options, cells per filled five-hour bout 835–983 and 176.48–206.31
  complete cells per warm node-hour [F cost file]; one-hour bout ≈ 129–158
  cells [A: 60 − 8.9…10.9 − 5 warm minutes]:

| Option | Keywords | Cells [A] | Five-hour bouts [A] | Node-hours ceiling | GPU-hours ceiling | Fraction of the 624,192-proxy pass |
|---|---:|---:|---:|---:|---:|---:|
| Design pilot | ~10 | ~2,000 | 3 (or 13–16 one-hour) | 15 | 60 | — |
| **Recommended** | ~200 | ~40,000 (200 × ~24 prompts × 8 natural cells) | 41–48 | 205–240 | 820–960 | 1/13–1/18 |
| Larger | ~400 | ~80,000 | 82–96 | 410–480 | 1,640–1,920 | 1/7–1/9 |

- These are arithmetic on the cost file's rates, not a throughput promise;
  shards, startup and partial bouts change occupancy as the cost file warns.
  Qwen coverage per keyword is below Llama's, so the matched-prompt set per
  keyword is smaller than 24 for some keywords; G1.2 gives the real count.
- Five-hour bouts are the Gemma-specific authorisation [F five-hour-queue
  handoff]; the project default is one hour [F AGENTS.md]. Both are tabulated.

## 6. D. Ordered plan

Smallest first. Mac = development, synthetic tests, small samples. HoreKa CPU =
inside one interactive allocation, batch-style script, one hour, resumable.
GPU = one-hour four-A100 allocations, resumable shards. Committed code only;
every output in a new directory with `scientific_result: false`.

| # | Milestone | Where | Tests covered | Status |
|---|---|---|---|---|
| M0 | Freeze analysis plan v1: strata, confirmatory outcomes, permutation count, population rule. Write it before looking at new numbers. | Mac | — | confirmatory families below |
| **M1** | **Population census from generation rows + ledger + axis** (no traces): coverage, Qwen missingness, empty/at-cap shares, target observed, axis variance by keyword, matched-prompt set, precision table, frozen population manifest | Mac dev → HoreKa CPU (minutes) | G1.1–G1.3, G1.6–G1.8, G2.2, G4.6 (at-cap proxy), G5.5, G8.3 | §7 |
| **M2** | **Trace feature extraction** (resumable; 1,000-prompt stratified sample first, then full) + trace census report: truncation and recovery, repairs/failures, queries, pools, generator-vs-compactor agreement, presented page ids, eligibility, pool-match census, cross-model pool identity, lexical overlap | Mac dev → HoreKa CPU (hours; measure on the sample) | G1.4, G1.5, G3.1, G3.7, G4.5, G5.1, G5.2, G6.2, G8.1, G8.2 | §7 |
| M3 | Review the text report; add `export --recover-full-answers`; rerun `text` on three populations | Mac → HoreKa CPU | G4.1, G4.2 | after Q6 |
| M4 | `report_axis_ranking_change.py` with `$REGISTRATION` on the frozen population; add keyword-bootstrap to binned means; run `check_axis_ranking_change.py` | HoreKa CPU | G3.2, G3.6 | needs Q2 |
| M5 | Relocation: read `relocation.json`; if absent, rerun `relocate` (CPU parts) then fresh re-embedding (GPU, 512 prompts) | HoreKa CPU + 1 GPU hour | G2.1 | needs Q4 |
| M6 | Paired condition contrasts and interaction CIs | HoreKa CPU | G6.1 | — |
| M7 | Ablation audit on the sample, then full | HoreKa CPU | G7.1, G7.2 | — |
| M8 | Embedding distance neighbourhood (if archived vectors exist) | HoreKa CPU | G3.3 | needs Q5 |
| M9 | GPU embeddings: unique presented pages and unique natural answers, both views, 20,000-item shards; `merge`; `analyze` | HoreKa GPU, several one-hour allocations | G4.3, G5.3 | after M5 |
| M10 | Pool-matched BT fits if G3.1 shows enough strata | HoreKa CPU | G3.4 | after M2 |
| M11 | Judge-validity gate r3 (prompt revision is a separate task) and frozen sampled judge plan from G8.1–G8.3 | HoreKa GPU, 2–3 one-hour bouts | G8.4, G8.5 | gates Gemma |

**Already seen (descriptive only, cannot become confirmatory):** ranking
length, stored answer length, search count, cross-model top-1 disagreement and
target selection along the axis; natural-vs-shuffled agreement [P report
2026-10-04]. The text report has run but is unreviewed: treat as exploratory.

**Confirmatory candidates (freeze outcome, stratum, estimator and permutation
count before computing):** G3.2 order-vs-membership slopes; G3.7 generator
versus compactor agreement along the axis; G4.3 answer-position slope; G4.5
top-1 lexical-overlap agreement along the axis; G5.1 Reactive query–prompt
overlap along the axis; G5.3 page z × axis interaction; G5.5 target-observed
along the axis. Everything in Groups 1, 2, 6, 7, 8 is a census, check or
design calculation and is reported without hypothesis tests.

## 7. E. Implementation prompt (milestones 1 and 2)

Paste the block below into Claude Code or Codex opened in
`.worktrees/generator-output-report` (or a new worktree branched from it).

```text
You are implementing milestones 1 and 2 of analysis/docs/fable-pre-gemma-tests-plan-20261004.md
in the GEODML repository. Read AGENTS.md first and follow it: inspect, state a brief plan,
implement the smallest complete milestone, run focused tests, commit, report. CPU only.
No judge data, no inference, no cluster-only edits, no fabricated outputs. Every manifest
you write carries "scientific_result": false. Never overwrite an existing output directory.
The axis position is a measured prompt property: write "association", never "effect".

## Read before coding
- analysis/docs/fable-pre-gemma-tests-plan-20261004.md (sections 3, 4, 6, 7).
- analysis/scripts/report_generator_outputs.py (load_cells, compact_cell, deciles,
  associations, table, write_csv, _git_commit) and analysis/tests/test_report_generator_outputs.py.
- analysis/interpretability/pipeline/agentic_cells.py (completed_generator_refs, iter_cells,
  read_references) and analysis/tests/test_source_importance_pipeline.py (the `dataset`
  fixture that builds a synthetic sealed dataset; `trace` builder).
- analysis/scripts/prepare_source_importance_tasks.py (judged_answer, generator_output_record,
  provenance, FINAL_PURPOSES) and analysis/interpretability/pipeline/agentic_judging.py
  (_trace_evidence).
- analysis/interpretability/pipeline/cluster_bootstrap.py, condition_manipulation.py
  (trace_view), page_readiness_ordering.py (page_text, page_id),
  source_importance.py (source_host, is_assessable, mask_answer_citations).
- analysis/scripts/report_latent_ranking_relationship.py (_association).
Reuse these; do not reimplement ranking agreement, associations, bootstrap, trace
evidence, answer recovery or page ids.

## Milestone 1: population census (generation rows + ledger + axis; no traces)
New script analysis/scripts/census_generator_population.py. CLI:
  --dataset qwen38=ROOT --dataset llama4=ROOT (repeatable, same contract as
  report_generator_outputs.py) --axis-map PATH --output-dir NEW_DIR
  [--bootstrap 200] [--keyword-subsamples 50,100,200,400] [--seed 20261004] [--stripes 256]
Extend report_generator_outputs.load_cells minimally and compatibly: include the task
fingerprint in each compacted cell, and return in the inventory dict a `tasks` list with one
row per registered task {fingerprint, task_id, model, prompt_id, method, engine, condition,
latest_state}. Existing callers and tests must keep passing unchanged.
Outputs (all under NEW_DIR; write to NEW_DIR.partial and rename at the end):
  population/cells.jsonl.gz   one row per verified completed cell: fingerprint, cell_id, model,
      prompt_id, keyword_id, primary_priority_rank, method, engine, condition,
      axis_1_percentile_0_1, ranking_length, empty_ranking, answer_chars, answer_at_cap
      (answer_chars == 1200), search_count, final_snippet_count, target_url_present,
      target_observed, target_selected_if_seen, target_reciprocal_rank_if_seen.
  population/tasks.jsonl.gz   every registered generation task with latest_state and the same
      identity/axis fields (never-claimed tasks included).
  population/manifest.json    dataset roots, contract.json sha256 if present, ledger
      event_count per dataset, latest_states, anomalies, axis map path+sha256+prompts,
      sha256 of the sorted fingerprint list, git commit, created_at, scientific_result false.
  report.md plus CSVs, sections:
   1 Coverage by model x method x engine x condition: registered, verified, prompts, keywords;
     design 26,009 x 12 per generator from the axis map; list prompt_ids in the axis map but
     absent from each registry (Llama is expected to miss exactly one).
   2 Qwen missingness: for prompts with any Qwen task, completed-vs-not shares by axis decile
     and by primary_priority_rank decile; maximum CDF gap of axis between the two groups;
     per-keyword completion fraction histogram; Llama as reference. Descriptive, no p-values.
   3 Empty rankings and answer_at_cap shares by stratum with keyword-cluster bootstrap 95% CIs
     (cluster_bootstrap, mean_of).
   4 target_observed rate by stratum and by axis decile with keyword-bootstrap CIs; Spearman of
     prompt-mean target_observed with axis via _association, permutations as given.
   5 Axis variance by keyword: between/within variance share of axis_1_percentile_0_1 over all
     26,009 prompts in the axis map joined to memberships; within-keyword range quantiles;
     count of keywords with range >= 0.5.
   6 Matched-prompt set: prompts with all 12 verified cells for both generators; counts by
     keyword; write population/matched-prompts.txt (sorted prompt_ids) and
     population/sample-1000-prompts.txt: a seeded sample of 1,000 matched prompts stratified
     by axis decile (100 per decile; fewer if a decile is short, say so in the report).
   7 Precision table (natural condition, per model, prompt means): for outcomes ranking_length,
     empty_ranking, target_selected_if_seen, answer_at_cap compute Spearman with axis and its
     keyword-cluster bootstrap SE when keywords are subsampled to each --keyword-subsamples size
     and to all keywords (seeded; each subsample drawn once per replicate). Also report the
     within-keyword ICC of prompt means (one-way ANOVA estimator). Label the table "design
     calculation for judge sampling; descriptive".
Tests (analysis/tests/test_census_generator_population.py), synthetic only, reusing
test_source_importance_pipeline.dataset for both models plus an added never-claimed task (as
test_unfinished_tasks_are_counted_but_not_analysed does): coverage counts and states; the
missing-prompt list; matched set and stratified sample reproducible under the seed; precision
table has one row per outcome x subsample and finite SEs; manifest hashes verify; rerun into an
existing directory raises. Run also analysis/tests/test_report_generator_outputs.py and
test_answer_readiness.py to confirm compatibility.

## Milestone 2: trace features and trace census (resumable)
New module analysis/interpretability/pipeline/cell_trace_features.py with pure functions over
one cell dict as yielded by iter_cells (generation, trace, prompt_text, keyword_memberships):
  answer_recovery: answer_source and full_answer_chars via judged_answer; answer_truncated,
      malformed_json_recovered, dropped_ranking_reference_count, final_attempts,
      final_validation_failures via generator_output_record; forced_finish (any llm_call purpose
      reactive_forced_finish); transport_errors; search_errors; provenance_status via provenance.
  evidence: presented_count, presented_page_ids (page_id(page_text(title,text)) in presented
      order), ranked_presented_positions, top1_presented_position, share_presented_ranked,
      presented_unique_hosts, ranked_unique_hosts (source_host).
  compaction: for Parallel-Expansion-v1 the last compaction event: top_score, margin between the
      last kept and first dropped score, kendall_ranking_vs_score (ranking_agreement on the
      ranked URLs vs the kept URLs sorted by score), top1_is_top_scored; for Reactive, the same
      against the concatenated observation order (positions) and None for score margin unless a
      single compaction exists.
  queries: queries list, query_count, mean_query_tokens, mean_query_prompt_jaccard (lowercase
      word tokens), mean_interquery_jaccard (Parallel: pairwise; Reactive: successive),
      snippets_per_search, zero_result_searches, dedup_input/output (Parallel),
      retrieved_unique_urls, retrieved_unique_hosts.
  answer_lexical: answer_prompt_token_overlap (share of answer content tokens, stopwords removed
      with a small fixed list, present in the prompt), answer_evidence_token_overlap (same against
      all presented snippets), per_source_overlap (list aligned with presented order),
      top1_has_max_overlap, answer_cites_evidence_ids (source_importance._GENERATOR_ID_MARKER).
  eligibility: si_status following build_cell's statuses without creating judge tasks
      (missing_request_or_answer, trace_answer_mismatch, evidence_count_mismatch,
      ranking_outside_evidence, no_observed_sources, ok), assessable_sources
      (source_importance.is_assessable), rq2_informative = status ok and ranking_length >= 2 and
      assessable_sources >= 2 and answer_source in (stored, trace_full) and provenance verified.
New script analysis/scripts/extract_cell_trace_features.py:
  --dataset MODEL=ROOT (repeatable) --population DIR (milestone 1 output; restricts to its
  fingerprints) [--prompt-ids FILE] --output-dir DIR [--batch 2000] [--stripes 256]
  Resumable: one part file per (model, batch index) written atomically (temp + rename), skipped
  when present with a matching sidecar hash; a final `manifest.json` records parts, counts,
  skipped reasons, population manifest sha256, git commit; `--output-dir` may already exist
  only if it holds this script's own partial state. Log one JSON line per batch to stderr
  (phase, model, read, total, seconds) so throughput can be measured on the sample.
New script analysis/scripts/report_cell_trace_census.py: --features DIR --population DIR
  --output-dir NEW_DIR [--bootstrap 200] [--permutations 200] [--seed 20261004]; report.md + CSVs:
   1 Truncation and recovery by model x method (answer_source shares; full_answer_chars
     quantiles); truncation share by axis decile with keyword-bootstrap CI and Spearman of
     prompt-mean truncation with axis (_association).
   2 Repairs, validation failures, forced finishes, search/transport errors by stratum; Spearman
     with axis per model.
   3 Eligibility census: si_status counts by stratum and decile; assessable sources per cell
     distribution; rq2_informative counts by stratum, decile and keyword; unique presented page
     count and unique (prompt, answer) count (these are the SI-v4 map and source task
     proxies U and P; say they are proxies).
   4 Pool-match census: within (model, method, engine, condition, keyword), count prompt pairs
     with identical presented page-id sets, identical URL sets, and Jaccard in bins
     [0, .25, .5, .75, 1]; count groups with >= 8 distinct prompts sharing one identical pool.
     Cross-model: same-cell pairs with identical presented page-id sets, by stratum and decile.
   5 Generator vs compactor: kendall_ranking_vs_score and top1_is_top_scored by stratum and
     decile with keyword-bootstrap CIs; Spearman with axis per model x method.
   6 Query features (G5.1/G5.2) by decile with keyword-bootstrap CIs; Spearman with axis per
     model x method; Reactive finish-before-budget share.
   7 Answer lexical overlap: top1_has_max_overlap by stratum and decile; Spearman with axis.
     Label clearly: lexical overlap, not answer support.
Tests (analysis/tests/test_cell_trace_features.py): extend the fixture with a
Reactive-Snippet-Loop-v1 cell whose trace has two search/condition/compaction/observation
rounds, reactive_action llm_calls and a reactive_forced_finish, and with a Parallel cell whose
final llm_call raw_output has a >1,200-character answer plus a controller_repair event
(answer_truncated true, dropped_ranking_references ["S9"]); assert every feature on both; assert
judged_answer recovery gives trace_full; test resumability (run extract twice: second run skips
all parts and the manifests are byte-identical except created_at); test --prompt-ids selection;
test the census report on the synthetic features (pool-match counts, eligibility counts).

## Commit, push, record
Run: python3 -m pytest -q analysis/tests/test_census_generator_population.py
analysis/tests/test_cell_trace_features.py analysis/tests/test_report_generator_outputs.py
analysis/tests/test_answer_readiness.py analysis/tests/test_source_importance_pipeline.py
Then git diff --check; commit on a new branch codex/pre-gemma-census-<date>; push; record the
full SHA in your report and in a new dated handoff under analysis/docs/handoff/ (append it to
README.md there).

## Paste-ready HoreKa commands (already-open login shell; do not prefix ssh)
Use the existing environment script and runtime exactly as previous handoffs do; substitute
nothing else. $CODE is the HoreKa checkout, $PY the project Python, $AXIS_MAP the axis map.
Block A, setup and checks (login shell):
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  CODE=<horeka checkout path>; PY="$RT/bin/python"; AXIS_MAP=<path to final-axis-map.jsonl>
  git -C "$CODE" fetch origin && git -C "$CODE" checkout <FULL_SHA> && test -z "$(git -C "$CODE" status --porcelain)"
  sha256sum "$AXIS_MAP" | grep -q '^43189f68' && echo AXIS_OK
  QWEN="$W/shared-hours/dataset"; LLAMA="$W/llama-hf/dataset"; STAMP=$(date +%Y%m%d-%H%M)
Block B, milestone 1 inside the open interactive allocation (CPU; expected minutes):
  "$PY" "$CODE/analysis/scripts/census_generator_population.py" \
    --dataset qwen38="$QWEN" --dataset llama4="$LLAMA" --axis-map "$AXIS_MAP" \
    --output-dir "$W/reviews/population-census-$STAMP" 2> "$W/reviews/population-census-$STAMP.log"
Block C, milestone 2 sample then full (same allocation; resumable across allocations):
  POP="$W/reviews/population-census-$STAMP"
  "$PY" "$CODE/analysis/scripts/extract_cell_trace_features.py" --dataset qwen38="$QWEN" \
    --dataset llama4="$LLAMA" --population "$POP" --prompt-ids "$POP/population/sample-1000-prompts.txt" \
    --output-dir "$W/reviews/trace-features-sample1000-$STAMP" 2> "$W/reviews/trace-features-sample1000-$STAMP.log"
  "$PY" "$CODE/analysis/scripts/report_cell_trace_census.py" --features "$W/reviews/trace-features-sample1000-$STAMP" \
    --population "$POP" --output-dir "$W/reviews/trace-census-sample1000-$STAMP"
  # Read the per-batch seconds in the log, estimate the full pass, then (if it fits the
  # remaining allocation or in later one-hour allocations) rerun extract without --prompt-ids
  # into "$W/reviews/trace-features-full-$STAMP"; rerunning the same command resumes.
Block D, return: the two report.md files, both manifest.json files, and the extract log.

## Report
Changed files; behaviour implemented; exact test output; the SHA; the paste-ready blocks with
placeholders filled where you can; assumptions (axis map path, registration path, runtime);
blockers; smallest next step (milestone 3: answer_readiness export --recover-full-answers).
```
