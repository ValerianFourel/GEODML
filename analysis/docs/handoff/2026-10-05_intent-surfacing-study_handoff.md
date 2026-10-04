# Intent surfacing study: where prompt intent enters the pipeline

Valerian asked for one script with every test of the prompt axis → permutation
question, the subspace math, and the math showing intent is carried by the pool
rather than necessarily by the ranking. Mid-turn he asked that the math also answer:
how much does intent match raise a page's chance of being shortlisted, how intent
compares with topic and keyword at that step, and how the pool's intent splits between
the AI's queries and the reranker. Decisions: all stages including query embedding (GPU);
search replay only, no reranker replay; HoreKa-local data; answers included.

Protocol: `analysis/docs/intent_surfacing_study.md` (fixed before any result; addendum
of 2026-10-05, also before any result, adds the shortlisting model and the pool split;
reranker analyses use the natural condition).

Code (commit on `codex/page-readiness-ordering-20261004`, see the index line):
- `analysis/interpretability/pipeline/intent_stages.py`: trace parsing with checks
  (reranker kept the top k by score, shortlist equals presented evidence, candidates came
  from the searches); stage values Q, R0, Rk, R, C, P, K, A; within-keyword slopes with
  shared keyword resamples and shuffles; exact chain β_K = β_R + β_(C−R) + β_(P−C) + β_(K−P);
  pool-versus-reordering verdict; mediation; intent/topic/presented-order oracles and
  utilization; three-way variance split; top-k Plackett–Luce rows for the shortlist;
  within-event reranker score regression with exact keyword reweighting.
- `analysis/scripts/intent_stages_study.py`: `trace-extract`, `replay` (adapter's own
  `_select_rows`, stops unless 1/256 of recorded searches replay identically), `analyze`
  (results.json, final-report.html; finished reranker strata cached in `<output>.cache`).
- Runners: `analysis/docs/horeka-intent-stages.sbatch` (CPU; trace-extract, replay,
  answer-view merge, analyze once queries are embedded) and
  `analysis/docs/horeka-intent-queries-interactive.sh` (both LLM2Vec views of the queries
  in a four-A100 allocation).
- `geo_drivers.estimate_e2` takes optional standardization stats; `replication` reads any
  estimate field. GEO report fix: driver intervals now print as odds ratios (the earlier
  report printed log-odds intervals under odds ratios; results.json was correct).

Tests: `analysis/tests/test_intent_stages.py` (15, synthetic; real Parallel/Reactive traces
from the production classes over a frozen snapshot; planted pool- and ranking-driven
worlds; exact chain identity; shortlisting recovery; replay gate pass and refusal; end to
end with cache reuse). 81 pass with the GEO, page-readiness, snapshot-audit and SI suites,
twice. No scientific result; nothing has run on a cluster.

Estimates [H]: job 1 trace-extract 15–35 min + replay 5–15 min (32 CPUs, 1 h); query
embedding 20–40 min in one four-A100 hour (count printed by job 1); job 2 analyze 35–60
min (local benchmark: one shortlisting fit over 11.3M choice rows 5 s; 16 replicates on
8 cores 57 s); a second submission finishes from the cache if needed.

Open: answer export and embed folders of the other session (`answer_readiness.py`, 199,329
unique natural answers, both views embedded, not merged) must be located by probe and passed
as ANSWER_EXPORT / ANSWER_EMBED_QWEN / ANSWER_EMBED_MISTRAL. Still awaiting Valerian: Qwen
bout fixes (attempt number in the failure record ID; CPU phases under the all-at-once
override); the Qwen union missing count; Gemma 200-bout submission status.
