# Prompt for Fable: rewrite the ACL ARR draft around the closed "mini internet"

You are rewriting the ACL ARR (October 2026 cycle) working draft
`analysis/paper/arr_oct2026_first_draft/` ("From Prompt Semantics to Source Rankings in
Generative Search"). Work in a checkout of branch `codex/page-readiness-ordering-20261004`
(it contains the draft and the analysis code cited below). Before anything else, read
`AGENTS.md`, the last three handoffs indexed in `analysis/docs/handoff/README.md`, the
draft's `README.md`, `missing-evidence.md`, `reviewer-response-map.md` and
`evidence/claim-source-map.md`, and `analysis/docs/page_readiness_ordering.md`.

Put any **blocking questions for Valerian at the top of your reply**, then continue under
stated assumptions. Label every number you write into the paper or your notes as
[F] read from a file (cite it), [P] pasted cluster evidence (with its date), [A] arithmetic
on labelled numbers, or [H] hypothesis. Never present a number as a result unless it was
read from a file or pasted evidence. Missing results stay as explicit `\todo{...}` markers.

## 1. The new framing (decided by Valerian)

Experiment V2 ran generative search inside a **closed, frozen "mini internet"**, not the
open web, which cannot be frozen or reproduced:

- 1,011 keyword topics [F: draft table]; frozen result snapshots from two engines,
  DuckDuckGo and SearXNG, up to 20 results per search action [F:
  `analysis/interpretability/pipeline/agentic_search.py`, `SEARCH_RESULT_LIMIT = 20`].
- The generator writes its own search queries (Parallel Expansion: exactly three per request;
  Reactive Loop: up to three search actions) [F: `agentic_search.py`].
- A deterministic lexical search engine scores every query against **every row of the
  whole snapshot, across all topics**: exact keyword match first, then
  4 × (query words shared with a row's keyword) + 1 × (query words shared with its title and
  snippet), then stored position, then a hash [F: `FrozenSnapshotSearchAdapter._select_rows`
  in `analysis/scripts/run_agentic_search_integration_smoke.py`, used by
  `submit_agentic_generator_backlog.py`]. The prompt's keyword is **not** passed to search.
  Tokens are not stopword-filtered.
- A cross-encoder (BAAI/bge-reranker-v2-m3) then compacts to at most 7 (Parallel) or 3 per
  action (Reactive) snippets shown to the model [F: `agentic_search.py`].

Present this as a deliberate **closed-world testbed**: reproducible, fully logged, every
retrieved row and its topic recorded in the traces, with an open-query retriever whose
results, as on the real web, can come from outside the request's topic. That framing is
legitimate **only if the paper states the mechanism exactly** (the current methods section
already does, line ~103; sharpen it, do not soften it) and discloses its known artifact:

- Some DuckDuckGo snapshot rows hold a dozen result titles glued into one snippet. Because
  they contain many common words, lexical scoring serves them to unrelated topics. The most
  shared row (a TurboTax tax-filing page, one URL) was shown 3,763 times under 530 of 1,011
  keywords; a medical-billing comparison row reached 469 keywords, including "best shipping
  companies" [P 2026-10-04, probe on `extract-llama-v2`].
- Llama answers: 312,090; snippet displays 1,654,235; distinct snippet texts 19,834; median
  7 snippets per answer; median distinct snippets per keyword 90 (both engines; 48 DuckDuckGo,
  45 SearXNG, median 3 shared); median keywords per snippet 3; 541 snippets under ≥ 20
  keywords; the 15 most shared account for ≈ 34,000 displays (≈ 2% [A]) [P 2026-10-04].
- How often an answer's evidence came from **another** topic, and whether that share changes
  along the prompt axis, is **not yet measured** (an audit is planned). Action-style queries use
  more generic words, so the share may rise with the axis [H]. Write it as a stated threat with
  a planned sensitivity analysis, not as a resolved issue. Do not describe the retrieval as
  keyword-scoped anywhere.

A one-page diagram of the mechanism exists (private artifact,
https://claude.ai/artifact/1N5JkPMM2493hsbeYYUbhN); adapt its two-panel "expected vs
implemented" idea into a paper figure, redrawn for print.

## 2. The thesis Valerian wants to test

**Intent alignment is a lever in generative engine optimization (GEO):** beyond topic, a
source is ranked higher when its position on the information-seeking → action-readiness axis
matches the request's position. Evidence so far (llama only; observational):

- **Axis reproducibility.** From archived artifacts alone, all 26,009 prompt coordinates were
  rebuilt with max consensus-z and percentile difference 0.0; archived prompt embeddings
  replayed through the frozen Qwen and Mistral LLM2Vec maps reproduce archived raw axes to
  1.39e-16 [P 2026-10-04 09:15 UTC, `relocation.json`]. Fresh GPU re-embedding of 512 archived
  prompts on HoreKa: Spearman 0.99997 (Qwen view), 0.99998 (Mistral view) [P 2026-10-04 08:30 UTC].
- **Snippets placed on the same axis.** Each distinct snippet (title + snippet text exactly
  as shown) was embedded with both frozen views and mapped through the same z-scaling and
  Procrustes alignment onto the prompt percentile scale. The axis was fitted on prompts, so
  snippet positions are out-of-domain descriptions; the report flags snippets outside the prompt
  range [F: `page_readiness_ordering.py`].
- **Ranking model.** Top-k Plackett–Luce over each answer's ranked prefix among its presented
  snippets, utility = b_page·z + b_int·z·(prompt_axis − 0.5) + presented-position effects.
  Llama, 282,000 answers with a non-empty ranking: b_page = +0.173, b_int = +1.252 (consensus
  view) [P 2026-10-04 08:57 UTC]. Implied effect of +1 z of page action-readiness: −0.45 at prompt
  axis 0, +0.17 at 0.5, +0.80 at 1, i.e. odds × 0.64 / 1.2 / 2.2 [A]. Keyword-cluster bootstrap
  intervals, the within-keyword permutation p-value, the per-view and per-method fits are in
  `analysis-llama/results.json` but **have not been read yet**: leave them as `\todo`.
- **Prompt axis → permutations** (`report_axis_ranking_change.py`, llama, within-keyword prompt
  pairs) [P 2026-10-04, `summary.txt`]. Top-3 set-distance Spearman with measured axis gap, natural
  condition: +0.144 (Parallel/DuckDuckGo), +0.176 (Parallel/SearXNG), +0.120 (Reactive/DuckDuckGo),
  +0.128 (Reactive/SearXNG); within-keyword medians nearly identical; assigned-coordinate
  correlations only +0.02 to +0.06; order among shared URLs +0.04 to +0.06. Top-3 membership differs
  for 86–94% of pairs regardless of gap. Pairs reuse prompts: no valid intervals for these ρ.
  Retrieval differs across prompts, so these mix retrieval and ranking effects.

Caution on evidence conditions: the shuffle hook runs **before** cross-encoder compaction,
which re-sorts snippets, so "shuffled" does not randomize the final presented order [H, from
project notes; verify in `agentic_search.py` before stating]. Treat natural/shuffled/ablated as
retained strata, not as an identification strategy. Presented-position effects in the ranking
model therefore partly absorb compaction relevance.

## 3. What the paper may and may not claim

- Associations only. The prompt axis is a measured text property, not a randomized treatment.
  Embeddings describe text; they never define treatments and are never called confounders
  (AGENTS.md). The GEO-lever claim stays observational unless a content intervention is run.
- Scope every finding to this closed testbed, llama (Qwen pending), these two snapshots.
- Do not reuse the earlier manuscript's page-feature or hidden-state estimates.
- No invented numbers, examples or citations. Keep the existing verified citation metadata.

## 4. Structure to propose (confirm with Valerian before deleting material)

- RQ1: How do source membership and order change with the request's position on the axis?
- RQ2: Does a source's intent position, relative to the request's, predict its rank among the
  sources presented (intent alignment)?
- The current judge-based relevance/support comparison (RQ2 today) becomes secondary or
  future work unless validated SI-v4 results arrive; ask Valerian.
- Methods: a clear "closed mini-internet testbed" subsection (corpus, retriever, snapshot
  artifact, exposure statistics), the snippet-axis instrument, the Plackett–Luce model, inference
  (keyword bootstrap, permutation null).
- Threats/robustness section with pre-specified checks, each a `\todo` until run: evidence
  audit (on-topic share by axis decile, engine, method); refits on on-topic-only, SearXNG-only
  and hub-free answers; per engine and per model × engine fits; Qwen replication; topical
  relevance control (prompt–snippet similarity outside the axis direction); instrument validation
  of snippet positions by an independent judge; optional content-rewrite intervention that would
  make intent alignment causal.

## 5. Deliverables and rules

- Edit `main.tex` and `sections/*.tex`; update `evidence/claim-source-map.md` (every new claim
  with its source), `missing-evidence.md`, `reviewer-response-map.md`, and the draft README.
- Add the mechanism figure (vector, print-legible, both panels labelled).
- Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` and fix errors.
- Do not run cluster jobs, inference or analyses; do not change pipeline code, datasets or
  outputs. Commit on a new branch from `codex/page-readiness-ordering-20261004`, record the SHA,
  and add a dated handoff to `analysis/docs/handoff/` per AGENTS.md.
- In your reply: blocking questions first, then a section-by-section change list, then the list
  of `\todo` items with the analysis that resolves each.
