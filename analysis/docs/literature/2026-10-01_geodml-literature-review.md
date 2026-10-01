# GEODML / AxisGEO literature review

Search date and inclusion cutoff: 1 October 2026. Prepared for Valerian.

The ten-theme search produced a deduplicated bibliography of 93 works. The most important finding for positioning is that recent GEO work already separates citation selection from contribution to the answer and examines page structure. The project's potential contribution is the joint study of measured prompt readiness, controlled semantic coverage, search architecture, and the gap between request relevance and realized answer support. Establishing that contribution still requires validation and results from this project's experiments.

This is a broad, documented literature map, not an exhaustive systematic review. Twelve core papers received targeted full-text inspection. Other entries have abstract or metadata inspection, marked individually. Publication details were checked against public scholarly records; an indexed record is not proof of peer review or sound methods. In particular, the recent 2025–2026 papers below are leads for detailed methodological comparison, not independently replicated findings.

## Files and scope

- [Ten detailed search prompts](2026-10-01_geodml-literature-search-plan.md)
- [Annotated bibliography and theme tables](2026-10-01_geodml-annotated-bibliography.md)
- [Structured bibliography](2026-10-01_geodml-bibliography.json)
- [Queries, sources, failures and exclusions](2026-10-01_geodml-search-log.md)

References such as [P024](2026-10-01_geodml-annotated-bibliography.md#p024) link to the complete citation, primary or DOI link, inspection level and project-specific annotation. The bibliography includes overlapping theme tags; counts across themes should not be added to obtain the number of unique papers.

The project interpretation follows its current instructions and the local documents on the readiness population, subspace robustness, agentic dataset contract and SI-v4. Older planning documents sometimes describe a larger planned model panel. They do not establish that those experiments ran. This review does not verify cluster state or infer scientific results from job completion counts.

## 1. Defining information seeking and action readiness

Read first: [Broder, P006](2026-10-01_geodml-annotated-bibliography.md#p006), [Marchionini, P010](2026-10-01_geodml-annotated-bibliography.md#p010), [Kuhlthau, P002](2026-10-01_geodml-annotated-bibliography.md#p002), [Gollwitzer and Brandstätter, P003](2026-10-01_geodml-annotated-bibliography.md#p003), and [Gollwitzer and Sheeran, P012](2026-10-01_geodml-annotated-bibliography.md#p012). Kuhlthau and the meta-analysis currently have metadata-only verification in this search.

The search supports several related constructs, not an established universal scalar running from information to action. Broder's abstract distinguishes informational, navigational and transactional needs. Transactional intent is broader than buying and is a category rather than an equally spaced numerical scale. Marchionini motivates exploration beyond fact lookup. The implementation-intention literature distinguishes an intention to achieve a goal from a plan linking an opportunity to an action. Those are useful anchors for the rubrics, but evidence about people carrying out plans cannot be transferred directly to short prompt strings.

| Construct | What it contributes | What the project must not infer automatically |
|---|---|---|
| Informational/transactional intent | A vocabulary for what a search request asks the system to provide | A single latent continuum or equal distances between scores |
| Exploratory search | Learning and understanding as legitimate outcomes | That broad requests are necessarily less competent or less valuable |
| Deliberation/implementation | A distinction between considering a goal and planning its execution | That a text score measures the user's actual motivational state |
| Implementation intention | Concrete conditions and actions that operationalize a plan | That an action-oriented question predicts subsequent behavior |

The mindset studies also complicate a simple “more implementation means more action” account. The abstract of [P007](2026-10-01_geodml-annotated-bibliography.md#p007) reports that persistence changes with the functional demands of the task. Use **prompt action-readiness score** or **expressed action orientation**. Treat the score as a measured text property and test whether evaluation/selection and implementation need separate dimensions.

## 2. Estimating a direction in embedding space

Read first: [LLM2Vec, P024](2026-10-01_geodml-annotated-bibliography.md#p024), [SimCSE, P019](2026-10-01_geodml-annotated-bibliography.md#p019), [linear representation hypothesis, P023](2026-10-01_geodml-annotated-bibliography.md#p023), [probe control tasks, P016](2026-10-01_geodml-annotated-bibliography.md#p016), and [TCAV, P015](2026-10-01_geodml-annotated-bibliography.md#p015).

LLM2Vec §2.1 directly supports the representation recipe: bidirectional attention, masked next-token prediction and unsupervised contrastive learning. Its benchmark results validate the encoder approach on the paper's tasks, not this project's readiness construct. SimCSE explains the contrastive objective and representation geometry underlying part of that recipe.

The linear-representation paper distinguishes measurement and intervention in §§2.2–2.3 and introduces an explicit geometry in §3. Its counterfactual concept assumptions do not automatically hold for pooled LLM2Vec readiness embeddings. A fitted ridge direction can predict a label without being unique, causally operative inside the generator, or psychologically fundamental. Probe control tasks and the probing review [P020](2026-10-01_geodml-annotated-bibliography.md#p020) are therefore essential qualifications.

| Estimator or diagnostic | Appropriate use here | Main limit |
|---|---|---|
| Ridge projection | A reproducible supervised text score | Depends on rubric, label coding, regularization and preprocessing |
| Ordinal model | Respect the order of rubric labels | Proportional-odds and calibration assumptions require checking |
| SVD of rubric coefficients | Examine shared versus distinct supervised directions | Singular values do not by themselves establish latent psychological factors |
| Procrustes/representation similarity | Compare views after development-only alignment | Agreement can reflect shared shortcuts or training data |
| Concept erasure, INLP/LEACE | Check whether linear information persists after removal | Linear removal does not remove every nonlinear encoding or establish construct validity |

Describe the result as a **supervised semantic direction** until held-out construct tests justify stronger language.

## 3. Validating the measurement

Read first: [construct validity, P001](2026-10-01_geodml-annotated-bibliography.md#p001), [multitrait–multimethod validation, P026](2026-10-01_geodml-annotated-bibliography.md#p026), [measurement invariance, P029](2026-10-01_geodml-annotated-bibliography.md#p029), [Procrustes, P028](2026-10-01_geodml-annotated-bibliography.md#p028), and [CheckList, P032](2026-10-01_geodml-annotated-bibliography.md#p032). The first two are foundational reading leads with metadata-only inspection here.

Two embedding models offer useful corroboration, but not independent human validation. Shared labels, corpora, generator styles and acceptance rules can make both views agree on the same shortcut. CKA [P017](2026-10-01_geodml-annotated-bibliography.md#p017) measures aspects of representational similarity; it is not a validity certificate for the score.

| Intended claim | Required evidence or diagnostic |
|---|---|
| The score orders expressed readiness | Held-out, blinded rubric judgments and ordinal agreement |
| The score is not mainly length or grammatical form | Matched question/imperative, length, politeness and lexical-cue controls |
| Meaning transfers across topics and sources | Source/topic holdouts with calibration and error analysis |
| One dimension is enough | Compare held-out performance and residual structure with two-dimensional alternatives |
| A numerical difference has stable meaning | Calibration evidence; ranking agreement alone is insufficient |
| Cross-view agreement is meaningful | Fit alignment on development data only; evaluate all confirmation cases before agreement filtering |
| The selected population represents a broader prompt population | Report rejected regions and compare accepted/rejected distributions |

The repository's development/confirmation split, permutation checks and bootstrap stability checks fit this literature. Their presence in code is not a result. Preserve the distinction between an exploratory robustness battery and a preregistered confirmation study.

## 4. Generating a prompt population

Read first: [Self-Instruct, P036](2026-10-01_geodml-annotated-bibliography.md#p036), [contrast sets, P034](2026-10-01_geodml-annotated-bibliography.md#p034), [FUDGE, P035](2026-10-01_geodml-annotated-bibliography.md#p035), [embedding inversion, P037](2026-10-01_geodml-annotated-bibliography.md#p037), and [LLM2Vec-Gen, P025](2026-10-01_geodml-annotated-bibliography.md#p025).

Self-Instruct generates and filters instructions for training. That is a useful construction precedent, but the project's objective is coverage of an evaluation population. WizardLM varies complexity; complexity should not silently become a substitute for readiness. PPLM and FUDGE control generation using attribute signals, whereas the current pipeline proposes text and tests its realized coordinates after generation.

The March 2026 LLM2Vec-Gen abstract explicitly places its embeddings in the LLM's output space and describes trainable special tokens and a reconstruction objective. This supports the repository's warning: it is not automatically the inverse of the frozen pooled input-embedding map. Re-embedding every decoded candidate in both readiness views remains necessary.

The most useful validation package would report target coverage, realized coordinates, acceptance rates, holes in support, template concentration, human naturalness and topic fidelity. Evaluate before and after filters. Independent validation and delexicalized uniqueness reduce particular risks; they do not prove independence from the scoring models. Contrast sets can test whether small style edits change coordinates while intended readiness stays fixed.

The model-collapse paper [P039](2026-10-01_geodml-annotated-bibliography.md#p039) concerns recursive training on generated data. It is a relevant warning about feedback loops, but it does not establish that a fixed synthetic evaluation set suffers that same failure.

## 5. GEO and the closest competing work

Read first: [GEO, P044](2026-10-01_geodml-annotated-bibliography.md#p044), [AI-search source composition, P047](2026-10-01_geodml-annotated-bibliography.md#p047), [AgentGEO, P049](2026-10-01_geodml-annotated-bibliography.md#p049), [citation absorption, P051](2026-10-01_geodml-annotated-bibliography.md#p051), and [structural GEO, P052](2026-10-01_geodml-annotated-bibliography.md#p052).

| Work | Verified scope in inspected material | Overlap and implication |
|---|---|---|
| GEO, 2024 | Full-text §§3.1–3.4 describe a top-five-source engine, page edits and visibility metrics; results vary by strategy/domain | Page edits and answer visibility are established topics. Its reported “up to 40%” is setup-specific, not an expected gain here |
| How to Dominate AI Search, 2025 | Abstract reports engine comparisons, source ownership, languages and paraphrases | Ownership and prompt wording are already studied together; full-method inspection is needed before treating results as causal |
| AgentGEO, 2026 | Abstract describes citation-failure diagnosis, targeted repairs and held-out queries | Closely related to the distinction between contribution and citation; do not claim that distinction as new |
| Citation Selection to Citation Absorption, 2026 | Abstract reports descriptive analysis of 602 prompts and page features across AI search platforms | Direct overlap with going beyond citation counts and examining answer contribution |
| Structural Feature Engineering for GEO, 2026 | Abstract describes macro/meso/micro structure and six engines | Headings, chunking and structural page features are not untouched research territory |
| E-GEO | Bibliographic record only | E-commerce benchmark lead. Do not infer its design or findings from its title |

The 2026 results above remain author-reported abstract claims in this review. Full methods and benchmark artifacts need inspection before accepting their effect estimates. The risks position paper [P048](2026-10-01_geodml-annotated-bibliography.md#p048) also argues that much manipulation research assumes adversarial content has already entered the retrieved context. That assumption matters: winning retrieval access and changing the answer conditional on access are different problems.

A defensible framing is to investigate **how page-property associations and relevance/support gaps vary with measured prompt readiness across controlled search architectures**. This is a proposed positioning, not an established first-in-literature claim.

## 6. Comparing search architectures

Read first: [RAG, P055](2026-10-01_geodml-annotated-bibliography.md#p055), [ReAct, P057](2026-10-01_geodml-annotated-bibliography.md#p057), [IRCoT, P060](2026-10-01_geodml-annotated-bibliography.md#p060), [query rewriting, P062](2026-10-01_geodml-annotated-bibliography.md#p062), and [Self-RAG, P063](2026-10-01_geodml-annotated-bibliography.md#p063).

| Architecture | Decision that varies | Comparison requirement |
|---|---|---|
| Fixed-pool reranking | Ordering a supplied candidate set | Same candidates, query, evidence representation and output size |
| Fixed-retrieval RAG | Generating from selected external evidence | Separate retriever quality from generator evidence use |
| Parallel expansion | Multiple queries before evidence is observed | Record query/search budget, candidate pool and compaction losses |
| Reactive search | Later searches depend on earlier observations | Record trajectory, early stopping, actual calls and retained evidence |
| Adaptive trained RAG | Model learns when/how to retrieve or critique | Training differences make it more than a prompt-only baseline |

The project's three-query parallel method and up-to-three-search reactive method are not evidence-budget matched merely because both permit three queries. They differ in timing, selection, up to seven versus up to nine retained observations, and possible early stopping. Compare realized calls, unique documents, tokens, latency and cost, as well as answer outcomes. Analyze search-free completions explicitly.

Self-RAG §3 describes training and reflection tokens; it is not interchangeable with a generic reactive prompt. IRCoT's abstract directly motivates dependency between what has been derived and what must be retrieved next. Query rewriting provides a competing explanation for gains attributed too broadly to “reasoning.” BGE M3 [P065](2026-10-01_geodml-annotated-bibliography.md#p065) is an embedding paper, not direct validation of the project's cross-encoder reranker.

## 7. Order interventions and ablation estimands

Read first: [Lost in the Middle, P069](2026-10-01_geodml-annotated-bibliography.md#p069), [ContextCite, P068](2026-10-01_geodml-annotated-bibliography.md#p068), [irrelevant context, P067](2026-10-01_geodml-annotated-bibliography.md#p067), [The Power of Noise, P070](2026-10-01_geodml-annotated-bibliography.md#p070), and [few-shot order sensitivity, P066](2026-10-01_geodml-annotated-bibliography.md#p066). The last paper studies demonstration order, an adjacent intervention rather than document order.

Lost in the Middle §2.1 manipulates where answer-bearing evidence appears. Its results support testing evidence order, not predicting that every seven-snippet case will exhibit the same positional curve. The Power of Noise examines relevance, position and document count; the reported role of random versus distracting documents cautions against assuming that more ostensibly relevant context always improves the answer.

| Contrast | What it can identify under a valid paired design | Main ambiguity to control |
|---|---|---|
| Shuffle the final retained evidence | Effect of the chosen order distribution conditional on membership | Tokenization, labels and membership must stay fixed |
| Shuffle before compaction | Effect of upstream order through the selection pipeline | Reranking may erase the shuffle or change membership |
| Remove a source from fixed context | Response change under that deletion rule | Length, redundant evidence and replacement policy |
| Remove a source during reactive retrieval | Total change in the resulting search trajectory | Later queries and documents are post-intervention outcomes |
| Attempt removal when target is absent | No exposed-source removal contrast | Report ineligible/undefined rather than coding zero reliance |

Specify paired randomness and repeated-run policy before interpreting a difference. ContextCite §3 fits a surrogate to context ablations; Appendix C.4 warns that redundancy can break the linear approximation and recommends held-out ablations to assess faithfulness. One-source-at-a-time deletion is therefore not automatically a measure of unique information contribution.

## 8. Relevance, support, citations and reliance

Read first: [AIS, P072](2026-10-01_geodml-annotated-bibliography.md#p072), [ALCE, P074](2026-10-01_geodml-annotated-bibliography.md#p074), [search verifiability, P040](2026-10-01_geodml-annotated-bibliography.md#p040), [FActScore, P075](2026-10-01_geodml-annotated-bibliography.md#p075), and [ContextCite, P068](2026-10-01_geodml-annotated-bibliography.md#p068).

| Quantity | Evidence needed | What it does not establish |
|---|---|---|
| Request relevance | Source assessed against the request and task | That the final answer used the source |
| Textual answer support | Source passages that support answer claims | That generation depended on that source |
| Citation correctness/completeness | Entailment and coverage of cited statements | Unique source contribution or causal reliance |
| Importance within the answer | A validated rule for central versus peripheral supported content | The counterfactual answer without the source |
| Context contribution | Controlled changes to available evidence and an explicit outcome | A universal or order-free allocation of redundant information |

ALCE §3.3 separates citation recall and precision. Its limitations explicitly discuss incomplete claim coverage and an NLI evaluator that cannot detect partial support in its citation-precision calculation. This is directly relevant to SI-v4's partial/full/uncertain distinctions. FActScore §3.1 counts supported atomic facts; its limitations focus on biographies, Wikipedia and challenges with nuanced or conflicting claims. Atomic factual precision is not a substitute for recommendation centrality.

The source-blind SI-v4 map can reduce source-conditioned segmentation, but a bad map can contaminate every dependent source grade. Validate map completeness separately and quarantine failures. Citation masking removes one cue; it does not prove complete blinding. RARR [P073](2026-10-01_geodml-annotated-bibliography.md#p073) is an especially useful counterexample: evidence can be found after an answer is generated, so a convincing supporting source need not have caused the answer.

For ideal-versus-realized rankings, rank-biased overlap [P071](2026-10-01_geodml-annotated-bibliography.md#p071) addresses incomplete and nonidentical lists and discusses ties. Choose its top-weighting parameter explicitly. Also report absolute relevance and support grades: a ranking gap alone cannot tell whether all sources were good, poor or indistinguishable.

## 9. Validating the judges

Read first: [MT-Bench judge study, P081](2026-10-01_geodml-annotated-bibliography.md#p081), [G-Eval, P080](2026-10-01_geodml-annotated-bibliography.md#p080), [fair-evaluator critique, P085](2026-10-01_geodml-annotated-bibliography.md#p085), [ARES, P083](2026-10-01_geodml-annotated-bibliography.md#p083), and [RAGAs, P086](2026-10-01_geodml-annotated-bibliography.md#p086).

The MT-Bench paper §3.3 documents position, verbosity and self-enhancement biases, and §4 evaluates agreement on its own human-preference tasks. Its agreement figures do not validate Nemotron or Gemma on SI-v4. G-Eval's abstract studies a structured evaluation approach; ARES explicitly uses human annotations to mitigate evaluator error through prediction-powered inference. These are precedents for calibration, not reasons to skip it.

A proposed validation sample should span readiness, method, engine, source rank and answer length. Include constructed cases where a source is relevant but unused, supports only a peripheral fact, partly supports a recommendation, contradicts the answer, or duplicates another source. Include maps with known omissions to measure error propagation. Blind annotators to generator and treatment, and keep ideal-relevance judgments separate from answer-dependent judgments.

Report ordinal confusion matrices, weighted agreement, disagreement by condition and abstention/failure rates. Weighted kappa and ICC are relevant methodological leads [P078–P079](2026-10-01_geodml-annotated-bibliography.md#p078), but choose the metric to match the scale and sampling design. Do not turn N/A or failed parsing into zero importance. Human calibration is proposed work here; this review does not claim it has been completed.

## 10. Statistical interpretation and DML

Read first: [DML, P090](2026-10-01_geodml-annotated-bibliography.md#p090), [multiway clustered DML, P092](2026-10-01_geodml-annotated-bibliography.md#p092), [causal inference using texts, P093](2026-10-01_geodml-annotated-bibliography.md#p093), [cluster-robust inference, P088](2026-10-01_geodml-annotated-bibliography.md#p088), and [text/confounding review, P091](2026-10-01_geodml-annotated-bibliography.md#p091). The cluster-robust inference guide currently has metadata-only inspection; the text/confounding review was checked on ACL Anthology.

DML §§2–3 supply orthogonal-score and cross-fitting methods. Its treatment-effect applications in §5 require identifying assumptions such as unconfoundedness. Flexible nuisance fitting is not a substitute for those assumptions. The multiway-cluster paper develops cross-fitting and standard errors for its sampling structure; keyword and domain dependence must be mapped to that structure before adopting the estimator.

| Project question | Defensible estimand | Analysis requirements |
|---|---|---|
| How does source support vary with measured readiness? | Association in the selected prompt population | Nonlinear curves, uncertainty, measurement error and support diagnostics |
| What changes when randomized B changes preference strength? | Prompt-policy treatment effect under the historical randomization | Preserve fixed task/evidence and S restrictions; use the actual randomization unit |
| How do page features relate to rank/support? | Observational association; a causal effect only under justified identification assumptions | Define eligible sources, controls, overlap and dependence before DML |
| What changes under natural/ablated/shuffled evidence? | Effect of the implemented intervention | Eligibility, intervention timing, paired comparisons and actual trajectories |
| Does a feature association differ along readiness? | Moderation of the specified association/effect | Avoid interpreting an interaction as mediation or a readiness treatment effect |

The text-causal-inference paper [P093](2026-10-01_geodml-annotated-bibliography.md#p093) warns that discovering a representation on the same data can create identification and overfitting problems. This supports freezing the measurement before confirmation analyses. Cross-fit and resample at units that preserve relevant dependencies, rather than treating every source grade as independent. Rankings create competition within a candidate set; top-k inclusion changes which outcomes are observed. Report failed cells, unknown support and rejected high-readiness regions instead of silently analyzing survivors as a representative sample.

Measurement error, informative missingness, continuous-exposure DML and multiple-testing methods remain thinner parts of this search. They need a targeted follow-up once the exact final estimand and dependence structure are fixed. No source in this review licenses calling observed readiness a randomized treatment or embeddings “confounders” by definition.

## Priorities and novelty boundaries

1. Read the 2026 citation-absorption, structural-GEO and AgentGEO papers in full before writing a novelty paragraph. Their abstracts show enough overlap to rule out broad priority claims.
2. Validate readiness as a text measurement: holdouts, matched linguistic controls, one-versus-two dimensions, and external judgments. Preserve accepted and rejected prompt distributions.
3. Validate SI-v4's map and source grades separately. Benchmark textual support against intervention-based attribution on a small diagnostic set without equating the two outcomes.
4. Write the architecture and intervention estimands before pooling results. Track realized evidence budgets and distinguish fixed-context from adaptive-trajectory ablations.
5. Choose the statistical model after the estimand and dependence structure. Keep V2 readiness associations, historical randomized B effects and observational page-feature analyses separate.

The search supports this focused positioning; it does not establish that the exact combined design is unique. The attached log records the public sources queried, rate-limit failures, rejected mismatches and the limits of citation chasing. The saved plan remains reusable for extending the thinner themes without losing the distinctions above.
