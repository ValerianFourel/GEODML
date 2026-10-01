# GEODML / AxisGEO literature-search prompts

Prepared for Valerian on 1 October 2026. These ten prompts form one literature-search plan. Give each search agent the shared instructions and one numbered prompt. The final task combines their evidence. This document preserves the substance of the preceding plan in a standalone format.

## Shared instructions for every search

Build an evidence-based literature review through 1 October 2026. Search foundational work and recent research using scholarly search engines, ACL Anthology, ACM Digital Library, OpenReview, arXiv and publisher pages. Follow references backward and citations forward. Include competing explanations, negative findings and methodological criticism. Do not claim to have found every relevant paper.

The project studies how prompt intent and page properties relate to source selection and answer support in LLM search. Experiment V2 contains about 26,000 selected prompts across about 1,000 topic keywords. Each prompt has a measured position on a semantic axis from information seeking to action readiness. That score describes the prompt text; it is not a randomized treatment. Associations along this axis must not be described as causal effects.

The axis uses frozen Qwen3-8B and Mistral-7B LLM2Vec representations, supervised directions and independent validation. Multiple readiness rubrics and one- versus two-dimensional representations are examined. Prompt generation targets semantic coverage, with naturalness and search-validity checks, re-embedding in both views, and template-diversity constraints.

Current experiments compare parallel query expansion and reactive snippet search across engines and natural, ablated and shuffled evidence conditions. Historical experiments also include fixed-pool reranking and page-content RAG. A separate historical design uses P = G(B, S): randomized B changes explicit preference strength for first-party product sources; S changes wording, syntax, order and tone only. Embeddings describe P and do not define B. Keep these experimental generations separate.

Source request relevance, textual answer support, citation correctness, causal reliance and unique contribution are distinct constructs. The SI-v4 development judge first maps a complete citation-masked answer without sources, then grades each unchanged source against that map and the answer. It does not by itself establish causal reliance. Page-feature DML analyses remain observational unless page content was experimentally manipulated. Do not treat implemented protocols as validated scientific results.

For each theme, return:

- An annotated evidence table with verified title, authors, year, venue or preprint status, DOI or stable primary link, research question, method, supported finding, limitations and exact project relevance.
- A classification for each paper: direct competitor, methodological foundation, supporting context or critical/contrary evidence. Papers may have more than one role.
- Five papers to read first. Aim for 15–25 strong papers per theme, allowing overlap between themes. Do not pad the list with weak matches.
- A section/page locator or short quotation for important findings. State whether you inspected full text, an abstract or metadata only. Never infer findings from the title alone.
- Implications for the project, defensible novelty claims, unresolved questions and a search log with queries, dates, sources and gaps.

Verify bibliographic metadata. Merge preprint and published versions where their identity is established. Do not invent papers, author lists, findings or citations. Record inaccessible sources explicitly.

## 1. What is information seeking to action readiness?

Search information retrieval, information science, psychology and consumer research for theories and measurements of the transition from learning about a topic to evaluating options, committing to a choice and implementing it. Cover search-intent taxonomies, exploratory versus lookup search, search as learning, task stages, deliberative and implemental mindsets, implementation intentions and the intention–action gap.

Start with searches such as `"search intent" "task stage"`, `"exploratory search" decision making`, `"information seeking" implementation`, `"deliberative implemental mindset"` and `"transactional intent" taxonomy`. Expand using terms from the strongest papers.

Assess whether the literature supports a single continuum or several dimensions. Distinguish action readiness from urgency, purchase intent, commerciality, expertise, specificity and actual behavior. Find language-based operationalizations and validated instruments relevant to four rubrics: information seeking reversed, evaluation, selection/commitment and action/implementation. Explain where transferring a psychological construct to a short search prompt is defensible and where it requires new validation.

Return a construct-comparison table, recommended terminology and papers challenging a simple scalar interpretation. Identify what a score of 0 or 1 could mean operationally without claiming equal psychological distances or predicting action without behavioral evidence.

## 2. Locating semantic concepts as directions in embedding space

Find methods for estimating a semantic concept from frozen text representations. Prioritize LLM2Vec, MNTP, SimCSE, supervised linear probes, ridge regression, ordinal proportional-odds models, TCAV, contrastive concept directions, representation engineering, concept erasure, activation steering and the linear representation hypothesis.

The project fits readiness directions in two frozen embedding views and examines singular vectors of a matrix of rubric coefficients. Compare this with PCA, supervised dimensionality reduction, low-dimensional subspaces and nonlinear manifolds. Examine centering, normalization, pooling, regularization, label coding and scale calibration. Explain when a line or plane is justified and when it hides distinct constructs.

Separate decoding a property from causally controlling a model or identifying a unique concept representation. Evidence about internal activation interventions does not automatically validate a direction in pooled sentence embeddings. Look for alternative probe solutions, confounding lexical features, selectivity controls and critiques of interpretability claims.

Return an estimator/assumption/control/failure-mode comparison and a shortlist specifically applicable to frozen LLM2Vec embeddings. State what can and cannot be inferred from high held-out prediction accuracy.

## 3. Validating the axis across models, datasets and linguistic forms

Search construct validity, measurement invariance, multitrait–multimethod analysis, ordinal factor models, item-response theory, differential item functioning, cross-model representation alignment, Procrustes methods, domain-general probes and out-of-distribution evaluation.

The project uses two embedding views, development-only alignment, source-separated development and confirmation data, held-out tests, bootstrap direction stability and permutation controls. Find evidence for testing one versus two dimensions, evaluator disagreement and abstention, source transfer and score calibration. Assess whether two models offer independent evidence when they share training data, objectives or correlated errors.

Design negative controls for prompt length, question versus imperative form, politeness, topic, specificity, brand names and lexical action cues. Distinguish stability of rank order from stability of metric distances on a 0–1 scale. Examine alignment leakage, test-set reuse and selection-on-agreement when generated prompts must pass both views.

Return a claim-to-required-evidence validation matrix. Separate checks already supported by this design from additional human or external validation that would be needed. Include literature that warns against mistaking inter-model agreement for construct validity.

## 4. Generating a diverse prompt population with controlled semantic coverage

Search controllable text generation, synthetic query and instruction generation, semantic paraphrasing, contrast sets, minimal pairs, rejection sampling, coverage optimization, distribution matching, embedding inversion and latent-space feedback.

The project generates prompts for topic/readiness targets, validates them independently, re-embeds them in two models and assigns candidates to targets one-to-one. It uses support-aware targets, bounded refinement and delexicalized-template uniqueness. One contract uses standalone questions of 8–60 words, a literal keyword and dual-view tolerance 0.017. Another allows questions, imperatives and search phrases of 4–60 words, keyword metadata and tolerance 0.035, with topic/search/web-answerability/naturalness checks. Treat these as distinct contracts.

Find methods to separate topic, intent and style; obtain natural prompts in sparse high-readiness regions; and measure meaningful diversity beyond surface paraphrases. Examine independent validators, generator-held-out evaluation, embedding-score optimization, Goodhart effects and filtering-induced selection bias. Contrast true embedding inversion with candidate proposals followed by re-embedding. A decoder trained in another representation space is not automatically the inverse of the project's pooled embedding map.

Return alternative construction methods and a checklist of evidence needed for coverage, naturalness, topic fidelity and independence. Identify whether synthetic coverage supports population-general claims or only claims about the selected prompt set.

## 5. Generative engine optimization and which sources become visible

Find the closest work on generative engine optimization, answer engine optimization, LLM reranking, AI-search source visibility and citation selection. Search `generative engine optimization`, `LLM reranking source bias`, `AI search citation visibility`, `answer engine optimization` and `RAG source selection`.

Cover source ownership, reviews, comprehensiveness, statistics, quotations, citations, headings, structured data, freshness, readability and reputation. Separate page interventions from observational correlations. Distinguish retrieval inclusion, rank, brand mention, explicit citation and substantive answer support. Papers about one outcome should not be treated as proving effects on another.

Compare across model families, user intent, engines, candidate-pool sizes and snippet versus full-page evidence. Search replication failures, benchmark artifacts, adversarial optimization, prompt injection and quality tradeoffs. Identify whether sources gain visibility by better supporting an answer or by exploiting model behavior.

Return a closest-competitor matrix with datasets, interventions, outcomes, search architectures, identification assumptions and limitations. State which novelty claims are already occupied, which combinations may remain distinctive and what additional search is needed before making a priority claim.

## 6. Comparing direct reranking, RAG, parallel search and reactive agents

Map fixed-candidate reranking, retrieval-augmented generation, query expansion, decomposition, multiquery search, iterative/adaptive retrieval, ReAct-style agents and cross-encoder compaction.

The project's parallel method proposes three queries before seeing evidence, retrieves up to 20 snippets each, deduplicates and selects a target of seven with a cross-encoder. Its reactive method permits up to three searches and three snippets per step, with a possible final forced-finish call; it can also finish without searching. BGE cross-encoder relevance is a selection score, not a factuality verifier.

Find comparisons that match search calls, candidate evidence, retained evidence, tokens, latency and monetary or compute cost. Examine reranking against the original request versus generated subqueries. Ask whether apparent gains come from extra evidence, larger budgets, better selection or adaptive reasoning. Distinguish live search from frozen replay and snippets from passages or full-page browsing.

Return an architecture taxonomy, relevant benchmarks and an interpretation checklist. Explain where proprietary search products differ enough that this experiment should not be presented as a reproduction of them.

## 7. Evidence order, source ablation and experimental identification

Search lost-in-the-middle effects, document ordering, listwise ranking permutations, context-position bias, document ablation, counterfactual retrieval and adaptive treatment trajectories.

Examine where the intervention occurs relative to deduplication, cross-encoder scoring and top-k selection. An order shuffle before reranking may disappear; a retained-evidence order experiment should preserve membership. Removing a source can also change context length, replacement sources and later agent queries. Explain how these consequences alter the estimand.

Distinguish fixed-evidence ablation from intervention on an adaptive search trajectory. Cover paired randomness, repeated runs, redundant substitutes, target selection and absent targets. If the target never appeared, its ablation contrast may be undefined rather than a zero effect.

Return a contrast-to-estimand table with assumptions for causal language and controls for alternative explanations. Identify which interventions isolate position, information availability or total trajectory effects and which mix them.

## 8. Source relevance, answer support, citations and causal reliance

Search attributable generation, citation correctness/completeness, grounded generation, claim verification, source contribution, counterfactual attribution, leave-one-out methods and Shapley-style attribution.

The project distinguishes ideal request relevance from realized answer support. SI-v4 first creates a source-blind map of the complete citation-masked answer, then judges an unchanged source against the map and full answer using exact spans and source-passage witnesses. Support states include partial, full, contradicted and uncertain; importance is ordinal from 0 to 5. This protocol is under development, not an established validated measurement instrument.

Find methods that distinguish central recommendations from peripheral facts, supported from unsupported content, and shared support from unique contribution. Examine redundant sources, global-absence claims, abstention and answer length/structure effects. State why textual support or a citation is insufficient to establish that generation causally relied on that source.

Find ways to compare ideal and realized source rankings with ties, partial rankings and omitted sources. Return a terminology table and a method-to-evidence comparison describing what each attribution method measures and what intervention would be needed for stronger claims.

## 9. Reliable LLM judges for readiness and source importance

Search LLM-as-judge reliability, human agreement, ordinal scoring, rubric calibration, claim decomposition, self-preference and position, verbosity, style and model-family biases. Consider Nemotron bulk judging and a Gemma development comparison without assuming either is a human-validated gold standard.

Compare map-first and holistic judging. Examine error propagation from an incorrect answer map, partial support, uncertain grades, abstentions and repeated judging. Ideal-relevance judgments must not see generated answers if that would leak realized behavior into the ideal target. Citation markers, generator identities, search methods and evidence conditions can reveal information to judges; specify appropriate blinding.

Find reliability statistics appropriate to ordinal grades, ranked sources and clustered repeated judgments. Distinguish inter-judge agreement from validity and shared model errors from independent corroboration. Include adversarially constructed diagnostic cases and held-out human or expert calibration designs.

Return a practical validation design and failure taxonomy. Label human-validation recommendations as proposed work, not work already performed in this project. Explain how quarantined maps, N/A scores and failed judgments should differ from genuine zeros.

## 10. Statistical analysis: DML, repeated outcomes and measured readiness

Search double/debiased machine learning, orthogonal scores, clustered cross-fitting, continuous exposures, varying coefficients, hierarchical ranking models, measurement error and causal inference with text.

Keep three estimands separate: observed prompt-readiness associations, historical randomized preference-strength effects and observational page-feature associations. Define the target outcome before choosing an estimator. Examine nonlinear readiness trends, moderation versus mediation, keyword/domain dependence, repeated source outcomes and competition within ranked candidate sets.

Find guidance on cross-fitting without topic/domain leakage, top-k censoring, missing failed runs, unequal support at high readiness, selection caused by prompt filters and noisy readiness labels. Assess bootstrap units, multiple testing and sensitivity to unmeasured confounding. Explain why DML cannot create randomization, fix an invalid construct or remove unobserved confounding by itself.

Return an estimand-first analysis map listing outcomes, assumptions, estimators, dependence structures and permissible interpretations. Separate descriptive associations from identified effects, and explain what additional manipulation or assumptions would be required for causal claims.

## Combine and check the ten searches

Deduplicate a master bibliography by DOI or stable identifier, retaining all theme tags and distinguishing verified version matches from possible duplicates. Build an evidence matrix for construct definition, axis estimation, prompt generation, retrieval, interventions, attribution, judge evaluation and statistical analysis.

Identify closest competitors, justified method choices and validation gaps. Separate what is implemented, what is proposed and what is demonstrated with real results. End with a prioritized reading list and conservative novelty assessment. Preserve queries, retrieval dates, inspection levels and unresolved coverage gaps so another researcher can extend the search.
