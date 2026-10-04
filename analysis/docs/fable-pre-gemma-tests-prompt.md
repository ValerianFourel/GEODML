# Prompt for Fable: which tests can we run on the Qwen and Llama inference corpus?

Paste everything below the line into a Fable session (Claude Code) opened in
`GEODML_Unified/.worktrees/generator-output-report`, branch
`codex/generator-output-report-20261004`.

---

You are joining GEODML, a research project on how LLM search-and-answer systems
choose and use sources. Your task is **planning only**: determine which tests and
analyses we can run on the **existing Qwen and Llama generator corpus** (no new
LLM inference), decide which of them must come **before we spend compute on Gemma
judge inference**, and end with a ready-to-paste implementation prompt for a
coding agent. Do not run cluster commands, launch jobs or edit code. Do not
present any number as a result unless it comes from a file you read; mark
everything else as unknown or as a hypothesis.

## 1. Read first (in this order)

1. `AGENTS.md` (repo root): mission, scientific terms, cluster rules and the
   "descriptive, not causal" boundary. Follow its rules in everything you propose.
2. The last three entries in `analysis/docs/handoff/README.md`, then
   `analysis/docs/handoff/2026-10-04_generator-output-report_handoff.md`.
3. `analysis/docs/EXPERIEMENTV2.md` and `analysis/docs/agentic_search_retrieval_protocol.md`:
   how a generator cell is produced.
4. Paper draft: `analysis/paper/arr_oct2026_first_draft/sections/{introduction,methods,results}.tex`
   and `missing-evidence.md`. Its research questions are RQ1 (do nearby or
   axis-shifted prompts yield similar source orderings?) and RQ2 (request relevance
   vs answer support).
5. Judge status: `analysis/docs/source-importance-v4.md`,
   `analysis/docs/si-v4-r2-semantic-review-20261004.md`,
   `analysis/docs/si-v4-nemotron-semantic-review-20261004.md`,
   `analysis/docs/si-v4-gemma-five-hour-cost-20261004.md`.
6. Existing analysis code (reuse before inventing):
   `analysis/scripts/report_generator_outputs.py`, `analysis/scripts/answer_readiness.py`,
   `analysis/scripts/report_axis_ranking_change.py`, `analysis/scripts/check_axis_ranking_change.py`,
   `analysis/scripts/report_latent_ranking_relationship.py`, `analysis/scripts/report_axis_permutation_study.py`
   with `analysis/docs/axis_permutation_report.md`,
   `analysis/scripts/page_readiness_ordering.py` with `analysis/docs/page_readiness_ordering.md`,
   `analysis/scripts/audit_condition_manipulation.py`,
   `analysis/interpretability/pipeline/{agentic_cells,axis_permutation_metrics,cluster_bootstrap,condition_manipulation,agentic_judging}.py`,
   and `analysis/scripts/prepare_source_importance_tasks.py` (full-answer recovery from traces).

If something below contradicts a file, trust the file and say so.

## 2. What the corpus is (facts from saved evidence, 2026-10-04)

- **Prompts.** 26,009 synthetic prompts on 1,011 keyword topics (18–30 per topic),
  each with a measured position on a 0–1 axis from information seeking to action
  readiness (`axis_1_percentile_0_1`, plus `consensus_axis_1_z`). The position comes
  from two frozen LLM2Vec encoder views (Qwen3-8B and Mistral-7B) and frozen maps. It is a
  **property of the prompt text, not a randomized treatment**: axis results are
  associations. Embeddings describe prompts; they are never treatments or confounders.
  Axis map: HF `ValerianFourel/geoaxis-prompts-generation-26k`,
  `final-audit/final-axis-map.jsonl`, sha256 `43189f68…`.
- **Design per generator.** Prompt × 2 search strategies (Parallel-Expansion-v1:
  3 queries, keep 7 snippets scored against the prompt; Reactive-Snippet-Loop-v1:
  1–3 searches, keep 3 per search scored against the query) × 2 frozen engine
  snapshots (duckduckgo, searxng) × 3 evidence conditions (natural, ablated = one
  frozen target URL removed, shuffled) = 12 cells per prompt. Generators:
  Llama-4-Scout-17B-16E-Instruct and Qwen3.8 (as named in the paper draft).
  Evidence is search snippets only; no page fetching.
- **What one cell stores.** Generation row: `ranking` (ordered subset of the
  supplied evidence URLs), `answer` (stored, capped at about 1,200 characters),
  `final_snippet_count`, `search_count`, `condition_audit` (target URL, whether
  observed, removals). Trace: every search query, every returned snippet,
  cross-encoder scores, compaction selections, raw LLM outputs, validation
  failures, repairs and truncation. Plus the prompt record and keyword membership.
- **Coverage** (report run `$W/reviews/generator-output-report-20261004-0918`,
  commit 6fc1390):
  - Llama: 312,090 verified completed cells, all 26,008 registered prompts, 6 terminal failures.
  - Qwen: 287,919 verified cells, 24,153 prompts, 984 keywords. Another 23,753 tasks
    were never claimed, 250 are checkpointed, 11 running, and 163 completed cells
    have unverified generation records. Qwen is still being generated, and in
    keyword priority order, so missing Qwen prompts are not random.
  - Data locations on HoreKa: Qwen `$W/shared-hours/dataset` (live, being written);
    Llama `$W/llama-hf/dataset`; private HF `ValerianFourel/geodml-experiment-v2-paper-private`.
- **Descriptive findings so far** (generator outputs only, no judges):
  - Toward action readiness, both generators rank fewer sources (Llama 2.97→2.17,
    ρ −0.30; Qwen 5.15→4.79, ρ −0.22) and write shorter stored answers (Qwen ρ −0.41).
    The Qwen figure is affected by the 1,200-character cap.
  - They search more (Llama ρ +0.37, Qwen +0.19). Parallel Expansion has a fixed
    search count, so that rise comes from the Reactive method.
  - On the identical cell, Llama and Qwen pick different top sources 50% of the
    time, rising from 42% to 55% along the axis.
  - The target URL is selected less often along the axis (ρ −0.07 / −0.04), which
    may simply reflect shorter rankings.
  - Every p-value sits at the 200-permutation floor (0.00498); there are about 160 tests.
- **Answer-text analysis** (`answer_readiness.py text`, output
  `$W/reviews/answer-readiness-20261004-text-0941`): it has run, but its results
  have not been reviewed. Answer embeddings onto the same axis
  (`answer_readiness.py export` → `page_readiness_ordering.py embed` → `analyze`)
  are prepared but not run. The fresh re-embedding check (`relocate`), which
  confirms the encoders still reproduce prompt positions, has not been reported.
- **Known data issues.**
  - The shuffle hook runs **before** cross-encoder compaction, so compaction can
    undo the shuffle (natural vs shuffled top-1 changes: Qwen 0.08%, Llama 3.2%).
    Valerian has asked to **ignore the shuffling question**.
  - Candidate pools differ between prompts because queries differ, so order
    comparisons need pool matching, or separate membership and shared-source order.
  - Llama Parallel-Expansion on searxng has 25% empty rankings (11% on duckduckgo).
  - Qwen Parallel-Expansion answers hit the 1,200-character cap (median 1,200).
  - The generator's ranking is a third observable. It is neither request
    relevance nor answer support.
- **Judge status.**
  - SI-v4 r2 with Gemma fails its own semantic standard on the 40-cell development
    set: omitted answer content, zero grades for supporting sources, role changes
    under reordering. It also misses the repeat thresholds and costs 7.9× the v3
    bridge. Nemotron-v4 produced zero numeric grades on 20 cells.
  - A full-corpus Gemma SI-v4 pass is being prepared anyway (CPU preparation job
    5179554). Its sender will **automatically submit up to 200 five-hour 4-A100
    bouts** when preparation ends; the estimate is about 2,000–4,000 node-hours.
  - The J1 fulfilment judge and the request-relevance judge
    (`ideal_relevance_ranking`) exist; check what outputs they have, if any.

## 3. What to produce

### A. Measurement inventory

List every quantity we can compute **from generator data alone**, at cell,
source, query and prompt level, with the exact field or trace event each comes
from. Cover at least:

- search behaviour: query count, query text, query drift from the prompt, query–keyword overlap;
- evidence pools: retrieved vs compacted vs ranked, pool overlap between prompts;
- ranking: length, empty rankings, rank of the target, position effects;
- answer: length, structure, lexical markers, truncation and trace recovery,
  embedding position on the axis;
- source text features of ranked vs unranked snippets;
- reliability: repairs, validation failures, failures.

Say which require GPU (answer or snippet embeddings) and which are CPU-only.

### B. Test catalogue

Group the tests; for each give:

- the question in one sentence;
- data and unit of analysis (cell, prompt, keyword) and the dependence structure;
- estimator, uncertainty method (keyword-cluster bootstrap and/or within-keyword
  permutation) and multiplicity handling;
- the comparison or control: within-keyword contrasts, pool-matched strata,
  Llama vs Qwen on the same cell, strategy/engine;
- which existing script covers it (or the gap);
- CPU/GPU and a rough cost;
- what a positive, null or negative result would mean;
- the strongest claim it allows, in the project's terms (association, descriptive);
- priority.

Groups:

1. **Data integrity and validity gates**: verified joins, coverage by stratum,
   missing-not-at-random Qwen prompts, empty and invalid rankings, cap, truncation,
   repairs, unverified records, Llama design/registry differences.
2. **Measurement checks**: the axis map reproduces (`relocate`), and the axis is
   not just keyword or topic. Report within- vs between-keyword variance.
3. **RQ1 on generator orderings**:
   - local stability vs prompt embedding distance;
   - directional change along the axis (Bradley–Terry / Plackett–Luce fitters),
     separating membership changes from order changes on shared sources;
   - pool-matched subsets, and how many pool-matched pairs actually exist.
4. **How the axis changes the answer text**, Valerian's current focus: text
   measures, distinctive words within keyword, answer position vs prompt position
   after embedding. Check sensitivity to the 1,200-character cap and to recovered
   full answers.
5. **Search and evidence behaviour along the axis**: queries, retrieved pools, what
   kinds of pages get ranked. Use page readiness from `page_readiness_ordering.py`,
   including whether action-ready pages rise for action-ready prompts.
6. **Generator, strategy and engine comparisons as conditions**: direct
   comparisons, not "significant here, not there".
7. **Ablation as a design check only**: target removal behaves as intended.
   Shuffle is out of scope.
8. **Tests that de-risk or shrink judge inference**:
   - eligibility census (judgeable answers, sources per cell, truncation);
   - which cells carry the most information for RQ2 and the support-gap question;
   - a power/precision calculation from generator-level variance, so a sampled
     judge design can replace the full corpus;
   - an independent judge-validity gate that must pass before any large Gemma
     pass, given the failed SI-v4 r2 review.

### C. Gemma decision

From the tests above, state which results should **gate** Gemma inference. Say
whether Valerian should hold the auto-submitting Gemma sender until they are in;
he decides. If you recommend a smaller judge design (a stratified sample along
the axis, keywords as clusters), give its size, the reasoning and the expected
compute, using the cost file's throughput.

### D. Ordered plan

Smallest-first milestones: what runs on the Mac (synthetic tests, small samples),
what runs on HoreKa CPU, and what needs GPU (the answer and snippet embeddings).
Respect AGENTS.md: one-hour allocations, interactive allocation first,
committed code only, resumable jobs, no repeated work, no fabricated outputs.
Mark which analyses are confirmatory and need an analysis plan frozen before
looking, and which are exploratory. Some descriptive results above have already
been seen.

### E. Ready-to-paste implementation prompt

End with a self-contained prompt for the coding agent (Claude Code or Codex) that
implements milestone 1 and 2 of your plan. It must follow AGENTS.md's development
loop:

1. inspect, plan, implement the smallest complete milestone;
2. reuse the existing scripts listed above; add synthetic tests;
3. commit and push, recording the SHA;
4. produce paste-ready HoreKa commands for an already-open shell;
5. report changed files, test results and blockers.

Include the exact paths, inputs and outputs it should use.

## 4. Rules for your answer

- Distinguish verified facts (cite the file), pasted evidence (dated) and your
  hypotheses.
- Use the project's terms: generator ranking, request relevance, answer support,
  axis position (a prompt property). Never call embeddings confounders or the axis
  a treatment.
- If a blocking fact is missing, list your questions for Valerian at the top, then
  continue with clearly stated assumptions.
- Keep it decision-oriented: a table for the test catalogue, short prose elsewhere.
