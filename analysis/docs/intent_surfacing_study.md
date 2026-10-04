# Where does intent surface? Prompt axis → queries → pool → ranking → answer

Protocol fixed on 2026-10-05, before any result below was computed, except the ones listed
as already seen. Companion to `geo_drivers_study.md`; same units, data and inference.

**Already seen (exploratory for this study):** E1 ranking, pool and reordering slopes per
model × engine; E2 ranking drivers; the pool slope by search method (Reactive ≥ Parallel).
**Not yet seen:** query intent, retrieved and candidate intent, the search replays, the chain
shares within method, mediation, oracle utilization, within-pool dispersion, reranker score
drivers, answer intent, and every model or method contrast below.

Observational throughout. The prompt's axis position `x` is a measured text property, not a
randomized treatment. Positions of queries, pages and answers are out-of-domain descriptions
on a subspace fitted on prompts. The search replays are exact reruns of a deterministic
component of the closed testbed; they isolate that component's contribution within the testbed.

## Stages

For each answer (HoreKa-local data: llama 312,090, Qwen 287,944; about 21,000 Qwen cells exist
only on Hugging Face and are a stated coverage limit), the intent value at each point of the
pipeline is a consensus axis-1 z (same frozen maps, z-scaling and rotation as the prompts):

| Stage | Value per answer |
| --- | --- |
| Q, queries | mean z of the queries the AI wrote (embedded with both views) |
| R₀, replay with the prompt text | mean z of the frozen search's top 20 for the prompt text itself |
| R_k, replay with the keyword | mean z of the top 20 for the bare keyword (identical within keyword: null anchor) |
| R, retrieved | mean z over the distinct pages all of the AI's searches returned |
| C, reranker candidates | mean z over the distinct pages the reranker scored |
| P, shortlist | mean z over the presented pages |
| K, ranking | top-weighted mean z of the ranked pages, weights `1/log2(r+1)` |
| A, answer | z of the answer text (natural condition only) |

Estimand per stage: the within-keyword slope of the stage value on `x` (keyword-demeaned least
squares), i.e. the change from the most informational to the most action-ready prompt of a keyword.

## Pre-specified analyses

1. **Instrument validity:** exact relocation and fresh re-embedding (recorded reports); view
   agreement (Qwen vs aligned Mistral z) for snippets, queries and answers; share outside the
   prompt range per text type; correlation of page z with intent-free topic similarity.
2. **Prompt axis → permutation:** within-keyword pairwise ranking-change correlations for both
   models (`report_axis_ranking_change.py`).
3. **Stage slopes** β_Q, β_R₀, β_R_k, β_R, β_C, β_P, β_K, β_A with keyword-bootstrap 95%
   intervals (200) and within-keyword permutation p (200).
4. **Chain identity** on answers with R, C, P, K: β_K = β_R + (β_C − β_R) + (β_P − β_C) + (β_K − β_P),
   exact by linearity; each increment and its share of β_K with intervals.
   Query-rewriting effect: β_R − β_R₀. Reranker effect: β_P − β_C.
5. **Mediation:** within-keyword regression of K on `x` and P; share of β_K mediated by the pool
   = 1 − (direct coefficient on `x`) / β_K.
6. **Oracle bounds** on each answer's pool with its own ranking length: presented order; random
   order (expectation = pool mean); intent oracle (closest to the prompt on the axis first);
   topic oracle (highest intent-free topic similarity first). **Utilization**
   U = (β_K − β_P) / (β_intent-oracle − β_P): the share of the available within-pool intent
   leverage the model uses. Topic oracle: whether ordering by topic alone reproduces β_K − β_P.
7. **Within-pool dispersion:** between- and within-pool variance of page z (intraclass share).
8. **Reranker score drivers:** within each compaction event, least squares of the reranker's
   score on the four standardized drivers (on-keyword, topic, page intent, intent alignment),
   by method (Parallel scores against the prompt, Reactive against the AI's query).
9. **Answer stage:** β_A, and the within-keyword slope of A on K (does the answer follow its ranking).
10. **Contrasts:** llama − Qwen and Reactive − Parallel for every stage slope and increment, with
    the same keyword resamples so differences have valid intervals.

## Replication and verdicts

Primary strata: model × engine (four). A stage slope or increment **replicates** if, in all four
strata, it has the same sign, its 95% interval excludes 0 and its permutation p < 0.05 (rule
implemented once, in `geo_drivers.replication`). Methods are reported per stratum and through the
pre-specified contrast; they are not a fifth replication requirement. The claim "intent is
carried by the pool more than by the ranking" holds for a stratum if the pool increment
β_P (equivalently β_R + (β_C − β_R) + (β_P − β_C)) exceeds the reordering increment β_K − β_P
and their difference has a 95% interval excluding 0. Thresholds are not changed after results
are seen.
