# Steelman results: does prompt intent reach the cited sources at the shortlist?

Exploratory (funnel study addendum A1). Models: qwen38; natural condition; 237 keywords of the exploration split. Observational: x is a measured property of the prompt text. Rules fixed in `PREREG.md` (sha256 `ff9e04839a29…`) before any part ran. Code commits: chain `8f3ef31`, generator `e46850b`, fe `e45260d`, lexical `eb2f2a2`, pairs `e45260d`. Rebuild: `python -m analysis.steelman report`.

Intervals are 95% keyword-bootstrap percentile intervals (200 draws; refitted generator models use the first 100); p values are within-keyword shuffles of x (200). Page intent is the page's percentile on the prompt scale (u).

## Verdicts

| Stratum | C1 admission | C2 selection (per model, both methods jointly) |
|---|---|---|
| qwen38 · Reactive | **supported** | **supported** |
| qwen38 · Parallel | **supported** | **supported** |


## Table 1. Share of the cited-intent slope added at each step

Shares of β_K, the within-keyword slope of the cited sources' rank-weighted intent on x. Increments sum to 1 down the first six rows in every column. The deck column is the earlier hand arithmetic (each slope on its own subset, no intervals, C and I not separated). Identity baseline = row K − I; random baseline = row K − P.

**qwen38 · Reactive** — common sample 9,757 answers; β_K common +0.068 [+0.059, +0.079], with controls +0.068 [+0.058, +0.078], given lattice target +0.078 [+0.062, +0.097].

| Step | Deck | Common sample | + prompt controls | Given lattice target | Only answers that dropped a link |
|---|---|---|---|---|---|
| R0 · prompt words (lexical replay, mechanical) | 18% | 17% [7%, 26%] | 36% [28%, 43%] | 27% [15%, 39%] | 15% [4%, 25%] |
| R − R0 · query rewriting | 33% | 32% [23%, 42%] | 6% [-3%, 18%] | 32% [21%, 44%] | 34% [22%, 47%] |
| C − R · deduplication | — | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] | 0% [0%, 0%] |
| P − C · reranker shortlist | 37% | 35% [28%, 42%] | 35% [27%, 44%] | 30% [23%, 38%] | 25% [17%, 34%] |
| I − P · shown-order weighting | — | 5% [0%, 10%] | 7% [1%, 12%] | 2% [-5%, 8%] | 7% [-3%, 16%] |
| K − I · generator's own choices | — | 10% [5%, 17%] | 16% [11%, 23%] | 9% [2%, 18%] | 18% [9%, 30%] |
| K − P · generator vs random selector | 13% | 16% [10%, 21%] | 23% [17%, 30%] | 11% [3%, 18%] | 26% [17%, 34%] |
| P · admission (everything before the generator) | 87% | 84% [79%, 90%] | 77% [70%, 83%] | 89% [82%, 97%] | 74% [66%, 83%] |

Dropping answers: 42% of the common sample dropped at least one shown link.

**qwen38 · Parallel** — common sample 9,950 answers; β_K common +0.060 [+0.052, +0.069], with controls +0.057 [+0.050, +0.065], given lattice target +0.072 [+0.061, +0.084].

| Step | Deck | Common sample | + prompt controls | Given lattice target | Only answers that dropped a link |
|---|---|---|---|---|---|
| R0 · prompt words (lexical replay, mechanical) | 20% | 20% [8%, 30%] | 45% [35%, 55%] | 30% [18%, 41%] | -22% [-463%, 385%] |
| R − R0 · query rewriting | 19% | 18% [10%, 27%] | -3% [-14%, 8%] | 12% [1%, 26%] | 49% [-424%, 327%] |
| C − R · deduplication | — | -0% [-1%, 0%] | -1% [-1%, 0%] | 0% [-1%, 1%] | -11% [-67%, 108%] |
| P − C · reranker shortlist | 47% | 47% [39%, 54%] | 40% [31%, 50%] | 45% [36%, 52%] | -32% [-640%, 805%] |
| I − P · shown-order weighting | — | 7% [4%, 9%] | 9% [6%, 12%] | 6% [3%, 9%] | -0% [-358%, 394%] |
| K − I · generator's own choices | — | 8% [5%, 12%] | 11% [7%, 15%] | 7% [3%, 11%] | 117% [-835%, 636%] |
| K − P · generator vs random selector | 14% | 15% [12%, 18%] | 19% [16%, 23%] | 13% [9%, 17%] | 117% [-604%, 517%] |
| P · admission (everything before the generator) | 86% | 85% [82%, 88%] | 81% [77%, 84%] | 87% [83%, 91%] | -17% [-417%, 704%] |

Dropping answers: 1% of the common sample dropped at least one shown link.

## O1. Admission is built into the pipeline

- **qwen38 · Reactive.** Query rewriting adds +0.022 [+0.016, +0.028] (p = 0.005); the reranker adds +0.024 [+0.017, +0.031] (p = 0.005); the prompt-text replay R0 +0.012 [+0.005, +0.018].
- **qwen38 · Parallel.** Query rewriting adds +0.011 [+0.006, +0.015] (p = 0.005); the reranker adds +0.028 [+0.022, +0.034] (p = 0.005); the prompt-text replay R0 +0.012 [+0.005, +0.018].

- **qwen38 · Reactive, frozen-search word-overlap selector against the reranker text.** Shortlist increment +0.012 [+0.007, +0.018] versus the reranker's +0.024 [+0.017, +0.031]; share of the reranker step 52% [33%, 72%].
- **qwen38 · Reactive, frozen-search word-overlap selector against the user prompt.** Shortlist increment +0.001 [-0.006, +0.007] versus the reranker's +0.024 [+0.017, +0.031]; share of the reranker step 4% [-28%, 29%].
- **qwen38 · Reactive, pre-registered BM25 selector (record only; not used in V2 or the paper) against the reranker text.** Shortlist increment +0.022 [+0.016, +0.028] versus the reranker's +0.024 [+0.017, +0.031]; share of the reranker step 92% [72%, 113%].
- **qwen38 · Reactive, pre-registered BM25 selector (record only; not used in V2 or the paper) against the user prompt.** Shortlist increment +0.032 [+0.026, +0.039] versus the reranker's +0.024 [+0.017, +0.031]; share of the reranker step 135% [115%, 168%].
- **qwen38 · Reactive, shortlisting model with word-overlap controls (30 keyword draws).** intent_x_prompt -0.364 → -0.376 per SD (change -0.012 [-0.028, +0.009]); intent_alignment +0.126 → +0.119 per SD (change -0.006 [-0.018, +0.004]). Overlap block share of fit 9%.
- **qwen38 · Parallel, frozen-search word-overlap selector against the reranker text.** Shortlist increment -0.007 [-0.014, -0.001] versus the reranker's +0.028 [+0.022, +0.034]; share of the reranker step -27% [-57%, -2%].
- **qwen38 · Parallel, pre-registered BM25 selector (record only; not used in V2 or the paper) against the reranker text.** Shortlist increment +0.033 [+0.027, +0.040] versus the reranker's +0.028 [+0.022, +0.034]; share of the reranker step 119% [101%, 142%].
- **qwen38 · Parallel, shortlisting model with word-overlap controls (30 keyword draws).** intent_x_prompt -0.269 → -0.284 per SD (change -0.015 [-0.020, -0.007]); intent_alignment +0.062 → +0.063 per SD (change +0.002 [-0.002, +0.004]). Overlap block share of fit 0%.

Action-word lexicon (50 words, from the confirmation-keyword prompts only): access, account, activate, active, and, api, app, automated, by, check, command, configuration, configure, confirm, confirmation, connection, deploy, error, exact, execute, ….

## O2. Shown order absorbs intent; C2 equivalence

| Quantity | qwen38 · Reactive | qwen38 · Parallel |
|---|---|---|
| Δ_gen, drop-intent refit (primary) | -0.0001 [-0.0022, +0.0017] | -0.0017 [-0.0028, -0.0003] |
| Δ_gen, intent coefficients zeroed | -0.0000 [-0.0022, +0.0020] | -0.0017 [-0.0028, -0.0003] |
| Δ_gen, models without shown slot | +0.0002 [-0.0024, +0.0026] | -0.0019 [-0.0032, -0.0003] |
| Δ_gen, reranker score as control | +0.0003 [-0.0019, +0.0021] | -0.0016 [-0.0027, -0.0002] |
| Model check: β_x(E[K]) − β_K | -0.0010 [-0.0021, +0.0000] | -0.0033 [-0.0042, -0.0024] |
| Δ_gen, 90% interval (TOST against ±0.015) | [-0.0021, +0.0015] | [-0.0026, -0.0004] |
| Δ_gen as share of β_K | -0% [-3%, 2%] | -3% [-5%, -0%] |
| Keep decision | modelled (4,057 informative answers) | fixed at the observed set (— informative) |

Keep and order coefficients (per SD, main variant):

- qwen38 · Reactive, keep: intent_alignment +0.320 [+0.189, +0.455]; intent_x_prompt -0.550 [-0.808, -0.351]; on_keyword +0.397 [+0.318, +0.485]; page_intent_z +0.022 [-0.063, +0.117]; topic_similarity +1.930 [+1.834, +2.082]
- qwen38 · Reactive, order: intent_alignment +0.096 [-0.003, +0.163]; intent_x_prompt -0.039 [-0.167, +0.171]; on_keyword +0.003 [-0.057, +0.054]; page_intent_z +0.055 [+0.003, +0.109]; topic_similarity +0.674 [+0.619, +0.745]
- qwen38 · Parallel, order: intent_alignment +0.102 [+0.039, +0.172]; intent_x_prompt -0.244 [-0.367, -0.121]; on_keyword +0.114 [+0.077, +0.155]; page_intent_z +0.059 [+0.018, +0.105]; topic_similarity +1.144 [+1.104, +1.213]

Within-answer correlation of the shown slot with:

| | qwen38 · Reactive | qwen38 · Parallel |
|---|---|---|
| u | +0.001 [-0.021, +0.020] | -0.006 [-0.023, +0.012] |
| intent_alignment | -0.017 [-0.030, -0.003] | -0.031 [-0.040, -0.020] |
| intent_x_prompt | -0.017 [-0.031, -0.001] | -0.026 [-0.036, -0.017] |
| topic_similarity | -0.223 [-0.245, -0.204] | -0.280 [-0.299, -0.260] |
| reranker_logit | -0.736 [-0.750, -0.723] | -0.936 [-0.938, -0.934] |

Two-way fixed effects (answer and snapshot row; the same snippet shown under different prompts), linear probability per SD of each regressor:

- qwen38 · Reactive, credit (mean 0.249; 37,440 shown rows, 5,144 snippets, within-snippet SD of x 0.243): intent_alignment +0.0029 [-0.0050, +0.0144]; intent_x_prompt -0.0099 [-0.0275, +0.0090]; topic_similarity +0.1336 [+0.1166, +0.1376]
- qwen38 · Reactive, keep (mean 0.576; 16,341 shown rows, 3,336 snippets, within-snippet SD of x 0.222): intent_alignment +0.0171 [-0.0108, +0.0508]; intent_x_prompt -0.0346 [-0.0786, +0.0202]; topic_similarity +0.3533 [+0.3019, +0.3634]
- qwen38 · Parallel, credit (mean 0.143; 67,091 shown rows, 6,746 snippets, within-snippet SD of x 0.255): intent_alignment +0.0018 [-0.0008, +0.0039]; intent_x_prompt -0.0019 [-0.0050, +0.0018]; topic_similarity +0.0424 [+0.0388, +0.0441]
- qwen38 · Parallel, keep (mean 0.414; 527 shown rows, 147 snippets, within-snippet SD of x 0.206): intent_alignment +0.0486 [-0.0198, +0.1825]; intent_x_prompt -0.1456 [-0.3712, -0.0476]; topic_similarity +0.2401 [+0.0845, +0.3046]

Score-matched adjacent shown links (|Δ logit score| < ε; pair logit of the later slot winning):

- qwen38 · Reactive, eps_0.1, order: 2,062 pairs, later wins 12%; slot intercept -2.071 [-2.255, -1.909]; Δintent_alignment -0.229 [-0.553, +0.033]; Δintent_x_prompt +0.143 [-0.081, +0.397]; Δtopic_similarity +0.489 [+0.333, +0.663]
- qwen38 · Reactive, eps_0.1, keep: 587 pairs, later wins 44%; slot intercept -0.245 [-0.491, -0.016]; Δintent_alignment -0.229 [-0.638, +0.205]; Δintent_x_prompt +0.037 [-0.412, +0.457]; Δtopic_similarity +2.146 [+1.869, +2.564]
- qwen38 · Reactive, eps_0.25, order: 4,692 pairs, later wins 12%; slot intercept -2.102 [-2.234, -1.995]; Δintent_alignment +0.011 [-0.174, +0.236]; Δintent_x_prompt +0.053 [-0.125, +0.218]; Δtopic_similarity +0.510 [+0.399, +0.623]
- qwen38 · Reactive, eps_0.25, keep: 1,357 pairs, later wins 42%; slot intercept -0.327 [-0.456, -0.226]; Δintent_alignment +0.004 [-0.322, +0.333]; Δintent_x_prompt -0.011 [-0.301, +0.349]; Δtopic_similarity +1.987 [+1.761, +2.228]
- qwen38 · Parallel, eps_0.1, order: 12,786 pairs, later wins 37%; slot intercept -0.609 [-0.650, -0.565]; Δintent_alignment +0.162 [+0.066, +0.272]; Δintent_x_prompt -0.167 [-0.270, -0.076]; Δtopic_similarity +0.943 [+0.888, +1.000]
- qwen38 · Parallel, eps_0.1, keep: 87 pairs, not fitted.
- qwen38 · Parallel, eps_0.25, order: 26,317 pairs, later wins 37%; slot intercept -0.607 [-0.638, -0.573]; Δintent_alignment +0.133 [+0.070, +0.199]; Δintent_x_prompt -0.141 [-0.213, -0.077]; Δtopic_similarity +0.948 [+0.898, +1.004]
- qwen38 · Parallel, eps_0.25, keep: 157 pairs, later wins 47%; slot intercept -0.464 [-1.868, -0.015]; Δintent_alignment +1.597 [+0.189, +5.868]; Δintent_x_prompt -2.360 [-8.027, -1.167]; Δtopic_similarity +6.041 [+4.530, +14.655]

## O3. Funnel arithmetic

- **qwen38 · Reactive.** Identity selector (first L shown, shown order) leaves the generator 10% [5%, 17%] of β_K; against a random selector (E[K] = P) 16% [10%, 21%]. Cited count on x -0.733 [-0.851, -0.612] links (p = 0.005); share of shown links cited -0.231 [-0.259, -0.199]; shown count +0.384 [+0.286, +0.499].
- **qwen38 · Parallel.** Identity selector (first L shown, shown order) leaves the generator 8% [5%, 12%] of β_K; against a random selector (E[K] = P) 15% [12%, 18%]. Cited count on x -0.052 [-0.101, -0.006] links (p = 0.020); share of shown links cited -0.007 [-0.014, -0.001]; shown count +0.000 [+0.000, +0.000].

## O4. Wording and length

- **qwen38 · Reactive.** Within (keyword, lattice target, engine) cells (2,519 cells with ≥ 2 prompts, 6,879 answers): slope of K on x +0.070 [+0.046, +0.100], of P +0.061 [+0.042, +0.088], of R +0.048 [+0.035, +0.063]; within-cell SD of K 0.051 versus within-keyword 0.072.
- **qwen38 · Parallel.** Within (keyword, lattice target, engine) cells (2,553 cells with ≥ 2 prompts, 7,064 answers): slope of K on x +0.065 [+0.049, +0.085], of P +0.056 [+0.040, +0.075], of R +0.024 [+0.017, +0.034]; within-cell SD of K 0.036 versus within-keyword 0.055.

## O5. Qwen only

Llama traces are not in this run; R0 and K for both models from the published answers (natural condition):

| Stratum | R0 slope | K slope | R0 / K | Cited count on x | Answers keeping every shown link |
|---|---|---|---|---|---|
| llama4 · Parallel · all | +0.016 [+0.013, +0.019] | +0.091 [+0.087, +0.098] | 18% | -1.93 [-1.99, -1.86] | 8% |
| llama4 · Parallel · exploration | +0.014 [+0.007, +0.019] | +0.090 [+0.080, +0.100] | 15% | -1.85 [-2.02, -1.68] | 7% |
| llama4 · Reactive · all | +0.016 [+0.013, +0.019] | +0.074 [+0.070, +0.079] | 22% | +0.08 [+0.03, +0.13] | 4% |
| llama4 · Reactive · exploration | +0.014 [+0.007, +0.019] | +0.075 [+0.066, +0.084] | 18% | +0.14 [+0.05, +0.23] | 4% |
| qwen38 · Parallel · all | +0.016 [+0.013, +0.019] | +0.057 [+0.054, +0.061] | 28% | -0.11 [-0.14, -0.09] | 98% |
| qwen38 · Parallel · exploration | +0.014 [+0.007, +0.020] | +0.058 [+0.051, +0.064] | 24% | -0.07 [-0.12, -0.03] | 98% |
| qwen38 · Reactive · all | +0.016 [+0.013, +0.019] | +0.068 [+0.064, +0.072] | 24% | -0.77 [-0.83, -0.72] | 8% |
| qwen38 · Reactive · exploration | +0.014 [+0.007, +0.020] | +0.064 [+0.056, +0.072] | 21% | -0.73 [-0.82, -0.61] | 8% |

## O6. Replication

No held-out (confirmation-keyword) result exists anywhere on disk; everything here is exploratory. Engine strata and leave-one-keyword-out:

| Stratum | Generator share K − I | Query rewriting | Reranker |
|---|---|---|---|
| qwen38 · Parallel · duckduckgo | 15% [10%, 22%] | +0.012 [+0.006, +0.018] | +0.026 [+0.019, +0.032] |
| qwen38 · Parallel · searxng | 3% [0%, 7%] | +0.010 [+0.005, +0.015] | +0.030 [+0.021, +0.038] |
| qwen38 · Reactive · duckduckgo | 16% [9%, 25%] | +0.026 [+0.020, +0.032] | +0.015 [+0.008, +0.022] |
| qwen38 · Reactive · searxng | 6% [-0%, 14%] | +0.018 [+0.012, +0.025] | +0.033 [+0.024, +0.042] |

qwen38 · Reactive, leave one keyword out: generator share 9.6%–11.3%, query rewriting 30.4%–33.8%, reranker 34.2%–35.8%.

qwen38 · Parallel, leave one keyword out: generator share 8.0%–8.7%, query rewriting 17.1%–19.8%, reranker 45.6%–47.5%.

## Exploratory follow-up (added after the first run, not pre-registered): whether the prompt names its keyword

Adding the prompt controls moved the query-rewriting share. One control at a time, shares of β_K:

| Stratum | Control | Prompt words R0 | Query rewriting | Reranker | Generator K − I |
|---|---|---|---|---|---|
| qwen38 · Reactive | candidate_slot | 17% [7%, 26%] | 32% [23%, 42%] | 35% [28%, 42%] | 10% [5%, 17%] |
| qwen38 · Reactive | keyword_in_prompt | 26% [18%, 33%] | 15% [7%, 26%] | 36% [27%, 45%] | 17% [12%, 23%] |
| qwen38 · Reactive | log_prompt_words | 20% [10%, 30%] | 30% [21%, 39%] | 35% [28%, 42%] | 10% [4%, 17%] |
| qwen38 · Reactive | subset: prompt_names_keyword | 24% [17%, 32%] | 15% [6%, 26%] | 38% [29%, 49%] | 17% [11%, 24%] |
| qwen38 · Reactive | subset: prompt_omits_keyword | 40% [16%, 67%] | 25% [6%, 60%] | 15% [-21%, 33%] | 7% [-18%, 29%] |
| qwen38 · Parallel | candidate_slot | 20% [9%, 30%] | 18% [10%, 27%] | 47% [40%, 54%] | 8% [5%, 12%] |
| qwen38 · Parallel | keyword_in_prompt | 33% [25%, 42%] | 6% [-4%, 17%] | 41% [31%, 51%] | 12% [8%, 16%] |
| qwen38 · Parallel | log_prompt_words | 23% [13%, 33%] | 16% [8%, 25%] | 47% [39%, 54%] | 8% [5%, 11%] |
| qwen38 · Parallel | subset: prompt_names_keyword | 30% [20%, 39%] | 10% [-0%, 21%] | 40% [30%, 51%] | 13% [9%, 18%] |
| qwen38 · Parallel | subset: prompt_omits_keyword | 67% [31%, 127%] | -15% [-64%, 26%] | 31% [-14%, 50%] | 6% [-8%, 17%] |

qwen38 · Reactive: within keyword, the share of prompts naming their keyword changes by -0.731 [-0.788, -0.676] from x = 0 to 1.

qwen38 · Parallel: within keyword, the share of prompts naming their keyword changes by -0.753 [-0.814, -0.694] from x = 0 to 1.

## Draft paper text

### qwen38

**Claim sentence (as the evidence supports it).**

> In agentic LLM search, the prompt's intent reaches the cited sources by changing which documents are shortlisted, through the queries the agent writes and the reranker that filters their results, while the generator's choice among shortlisted documents follows topical match and presentation order; removing the generator's sensitivity to intent changes the cited-intent slope by -0.1% [-3.4%, 2.3%] (Reactive Loop) and -2.8% [-4.8%, -0.5%] (Parallel Expansion).

Supporting numbers (Qwen, exploration keywords). Reactive Loop: of the cited-intent slope, the prompt-text replay accounts for 17%, query rewriting 32%, the reranker 35%, shown-order weighting 5% and the generator's own choices 10%. Parallel Expansion: 20%, 18%, 47%, 7%, 8%. Do not write "chiefly through the queries": the reranker step is at least as large in both methods, and the query share depends on whether the prompt names its keyword (follow-up section).

Applying the frozen search's own word-overlap rule to the reranker's candidates, against the same text the reranker scored, reproduces 52% [33%, 72%] of the reranker's intent increment under the Reactive Loop (the agent's own queries: word overlap accounts for part of it) and -27% [-57%, -2%] under Parallel Expansion (the user prompt: most of it goes beyond word overlap). Under the Reactive Loop the reranker never sees the prompt, so the intent it adds travels through the agent's queries.

**Limitations paragraph.**

> These results are exploratory and observational: the prompt's position on the intent axis is a measured property of its text, and the stage decomposition attributes an association, not an effect. The generator analysis rests on one model (Qwen); Llama, which drops most shown links, could not be analysed at the trace level here. The decisive test, which we did not run, holds the shortlist fixed: showing the same documents in the same order under prompts of the same keyword at different intent positions would measure the generator's intent sensitivity by design rather than through a fitted choice model. Our counterfactual also conditions on how many sources an answer cites, and that count itself falls with intent under the Reactive Loop (-0.73 [-0.85, -0.61] links from x = 0 to 1). Finally, part of what we call query rewriting travels with whether the prompt names its keyword, so it reflects the prompt's wording as much as the agent's reformulation.

**Positioning.**

> Tannenbaum (2026) and the survey of Martinez (2026) infer from live engines that exposure matters more than selection; in a closed testbed where every stage is observed, we decompose a prompt-intent effect stage by stage and find that the generator's own intent sensitivity leaves the cited-intent slope unchanged within a margin fixed before the analysis once the shortlist is fixed.

