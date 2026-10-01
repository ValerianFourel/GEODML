# Astra review of the old manuscript and current experiment

Valerian, the best next retrieval set is eight new papers, plus renewed public-access attempts for three foundational papers already catalogued. The existing library already contains most of the material needed to explain readiness measurement, RAG attribution, judge validation and clustered DML. Its main gaps are direct GEO competitors cited in the old manuscript, an explicit treatment of citation faithfulness, and methods for omitted-variable sensitivity and inference with machine-generated labels.

Review completed 2026-10-02 by GPT-6 Astra at high reasoning effort. The review began on 2026-10-01. Selection priorities refer to the initial 93-record, 77-PDF library. During this review the parent retrieved all eight new PDFs and recovered P026. Those retrievals do not change which papers were missing at selection time. P001 has public full-text HTML; its PDF and P078's PDF remained unavailable at the parent's latest report.

The accompanying `astra_download_selection.json` contains 56 decisions: all 36 old-manuscript references, four additional methodological candidates, and the initial 16 PDF gaps. `not_needed` means no new download is warranted now. It does not mean an already-held paper is unimportant or should be deleted. Years in arXiv-derived new records identify the first preprint date; publication and revision dates must remain distinct in final citations.

## What I read and what remains unverified

I read all extracted text of the 12-page `oldPaper/9568_What_Drives_LLM_Re_Rankin.pdf`, including its bibliography, figure captions, prompts and appendices. I inspected the active checkout's readiness map fitting, prompt generation, agentic search and judging, SI-v4, historical features and DML implementation. I also read the required three most recent handoffs, the relevant experiment contracts, and the library's titles, retrieval states and selected abstracts. Primary metadata and abstracts for proposed new sources were checked online, including the parent's verification of all 31 arXiv identifiers in the old bibliography.

This is a literature-selection and scientific-framing review, not a reproduction of the manuscript's estimates or a full-text audit of every cited paper. I did not run inference, inspect live cluster state, validate the complete 26,009-prompt population, or verify that every saved scientific artifact matches today's code. Historical results below are claims made by the old manuscript. Current source code establishes implemented behavior, not what every previously pinned job executed.

Paths below are relative to `.worktrees/threehour-relaunch-fix/` unless stated otherwise.

## The current study needs a different organizing question

The old paper asks which observed page properties causally affect admission and rank in LLM reranking. It reports six page features, 28 stated covariates, 1,011 B2B SaaS queries, two search engines, two 70B-class models and four prompt-by-evidence conditions. Admission, rank displacement and final rank are its outcomes. Probing, saliency and token edits supplement the DML analysis. These are the paper's reported design and results, not findings independently reproduced here.

Current V2 instead follows a frozen population reported as 26,009 prompts across measured information-seeking to action-readiness coordinates. The export contract targets 12 cells per prompt per generator, crossing two search methods, two engines and three evidence conditions. For Qwen and Llama this is 312,108 generation slots each and 624,216 bulk-judge slots. These are targets, not verified completion counts. See `analysis/docs/agentic_paper_dataset_contract.md:137` and `:146`. The earlier retrieval protocol's proposed four-model, three-method matrix is a different design and cannot replace these counts.

The scientific question is now how source relevance and source-supported answer content vary along a measured prompt-text construct, under explicitly identified search and evidence protocols. That is a coherent study, but it does not inherit the old paper's causal identification claim. A suitable descriptive framing is "How prompt action readiness relates to source relevance and answer support in bounded agentic search." A causal title would require a separately justified intervention and estimand.

The older `P = G(B, S)` policy experiment must remain separately named. Randomizing preference strength B can identify that prompt-policy contrast under its design; S changes expression only. It does not make measured readiness a randomized treatment, and it does not identify effects of observed page properties. Some older prose still calls B an "assigned semantic axis" in `agentic_search_retrieval_protocol.md:363`. The clearer current instruction is in `axis_permutation_report.md:60`: do not rename a readiness coordinate to B. The literature review should use the latter distinction consistently.

## What the implementation actually supports

The readiness map is supervised. `readiness_embedding_map.py:107` fits ridge coefficients from normalized embeddings to consensus readiness labels. At `:116` it constructs four rubric dimensions, then at `:145` obtains two supervised axes from the rubric coefficient matrix's SVD. These are not an unsupervised discovery of universal stages of human intention. `readiness_hf_subspace.py:123` separates development and confirmation items and fits on development at `:139`; confirmation evaluates the frozen map. The robustness battery compares both embedding views, dimensionality, nonlinear alternatives and source-held-out behavior. Implementing these checks does not establish that the realized population passed them.

Prompt construction also matters to interpretation. `readiness_prompt_population.py:1333` renders requested semantic destinations, with optional single-axis and two-axis designs, and separates surface instructions. The versioned search contract changes from standalone, exact-keyword questions to 4-60-word search triggers that can depend on topic metadata. High-axis recovery targets coordinates at or above 0.700. See `readiness_search_contracts_and_high_axis_harness.md:8`, `:24` and `:40`. Therefore an apparent high-readiness change may also reflect generation regime, syntax, explicitness or selection. Preserve their recorded provenance; do not infer a generation regime just from a score. Two embedding models agreeing is useful replication, but both can track the same linguistic shortcut.

The search agents operate on title/snippet records, not fetched full webpages. Parallel Expansion generates three queries and retains seven snippets; Reactive Loop can search up to three times and retain up to nine snippets. A cross-encoder compacts evidence. This limits what page-level features or source support can mean. Attribution is to the exact supplied record unless webpage provenance is separately established.

There is an important implementation qualification for the evidence conditions. The production runner does remove an exact frozen target URL; it is not limited to the arbitrary index-subset smoke transformation. `run_agentic_generation_worker.sh:32` enables production conditions, and the Qwen/Llama launchers pass `--production-conditions` at `:74`. `run_agentic_search_integration_smoke.py:1677` then instantiates `TargetUrlConditionHook`; its `apply` at `:319` removes the target for Ablated and shuffles records for Shuffled.

However, both methods still apply that hook before compaction. See `agentic_search.py:663` for Parallel Expansion and `:754` for Reactive Loop. `ContextCompactor.score_and_compact` at `:185` then sorts by descending score, with source index breaking ties. With distinct deterministic scores, this restores score order and erases the initial shuffle. The production flag does not add a post-compaction presentation hook. Score ties, numerical differences or changed model-generated query paths can complicate observed comparisons, so one must inspect the pinned configuration and actual traces before characterizing historical runs. The protocol's intended post-compaction shuffle is not evidence that it ran. The old protocol paragraph describing only smoke ablation is partly stale; exact-target production ablation exists, while its shuffle-timing concern remains supported by current code.

SI-v4 has a narrower and more defensible target than "which sources the model used." Its source-blind Stage A maps the completed answer into exact spans and importance roles. Stage B compares one supplied source against the fixed map and full answer. Grades 0-5 reflect the importance of supported answer content; redundant sources may receive identical high grades. It does not estimate causal reliance, exclusive contribution, factual truth outside the supplied record, or the ideal answer. See `source_importance_v4.py:27` and `:68`, and `source-importance-v4.md:3`. Ideal relevance remains a separate request-plus-evidence task in `agentic_judging.py:1379`; request fulfillment is separate again. Those distinctions should appear in the paper's outcome definitions before any gap is computed.

SI-v4 is implemented development work. Its map fidelity, witness agreement, ordinal grading and human validity are not established by schema checks or constructed examples. The current documentation explicitly says so at `source-importance-v4.md:9`, `:209` and `:214`. A rank difference between ideal relevance and supported content is interpretable only after defining eligible sources, ties, zero support, missing grades and incomplete answers. It is not automatically a measure of misallocation or harmful model behavior.

## Corrections needed before reusing the old paper's framing

1. **DML is an estimator, not an identification argument.** The abstract, section 3.1 and Figure 1 claim that DML provides causal identification, removes confounding bias or partials out all back-door paths. Orthogonalization and cross-fitting address nuisance-estimation error under assumptions. They do not establish exchangeability, absence of omitted confounding, valid control selection or a meaningful page-feature intervention. A proxy such as `has_llms_txt` does not demonstrably remove latent publisher sophistication. The stated limitations themselves acknowledge residual confounding. Current page-feature results should be described as adjusted observational associations unless identification is separately defended.

2. **The outcome scale and dependence need reconciliation with code.** The old table labels admission coefficients "log-odds." The inspected `analysis/scripts/dml_canonical.py:140` predicts admission probabilities and `:154` residualizes a binary Y before a linear coefficient calculation; it does not fit a logistic target parameter. The exact archived run must be checked before changing reported numbers. The script also lists 29 covariates, while manuscript prose says 28. Its random row-wise KFold at `:143` and row-level influence calculation do not by themselves handle repeated queries, pages, conditions and models. Existing P088 and P092 are more urgent reading than further generic DML papers.

3. **Admission-conditioned rank is a selected outcome.** The ranking outcomes are defined only for admitted pages. Contrasting their coefficients with full-pool admission coefficients does not automatically identify two independent causal stages. Conditioning on admission can induce selection bias, and rank displacement shares arithmetic with pre-rank. The old ceiling-artifact explanation is a hypothesis requiring matched samples and specifications, not something proven merely by a sign change. Retaining final rank does not cure selection.

4. **The recency citation is directionally mischaracterized.** Section 2 calls Fang et al. consistent with the paper's T6 result. The verified cited abstract describes artificial date injection and finds fresh passages promoted; the old paper reports observed freshness demotion. Different interventions, visibility of dates, domains and conditioning could explain the difference. It should be presented as a contrast, not supporting evidence. This is a citation/interpretation correction, not a new empirical result.

5. **Probes and saliency do not establish the claimed mechanism.** High probe AUC establishes decodability under that probe's controls, not causal use or the layer at which a decision becomes fixed. Low gradient saliency for schema tokens does not explain a negative association merely because the model apparently ignores those tokens. Deleting query words changes literal lexical overlap and potentially meaning; it is not a clean intervention on the SBERT cosine feature. P016 and P068 already provide better starting points for the availability-versus-influence distinction.

6. **Current novelty claims must be narrower.** The initial GEO paper actually studies content optimizations, and the current CC-GSEO-Bench abstract explicitly separates exposure, faithful credit and causal impact. Existing P051 separates citation selection from answer absorption; P049 diagnoses citation failure stages; P052 studies structure and citation behavior. These sources defeat broad current claims that all prior GEO is merely associative, that nobody separates stages, or that source support beyond citation is new. This comparison uses the versions currently available. I have not established what every competitor's earlier version said when the old manuscript was written, so it is not a retrospective priority accusation.

## The eight new downloads, in reading order

| Order | Paper and verified identifier | Why read it now |
| --- | --- | --- |
| 1 | Wallat, Heuss, de Rijke and Anand. *Correctness is not Faithfulness in RAG Attributions*. [arXiv:2412.18004](https://arxiv.org/abs/2412.18004). Published title expands RAG to Retrieval Augmented Generation, DOI 10.1145/3731120.3744592. | Directly constrains what the SI-v4 support score can claim about source reliance. |
| 2 | Qiyuan Chen and colleagues. *CC-GSEO-Bench: A Content-Centric Benchmark for Measuring Source Influence in Generative Search Engines*. [arXiv:2509.05607](https://arxiv.org/abs/2509.05607). | Closest missing comparator for intent-sensitive source influence, faithful credit and causal impact. Inspect exact metrics and interventions. |
| 3 | Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis. *Long Story Short: Omitted Variable Bias in Causal Machine Learning*. [arXiv:2112.13398](https://arxiv.org/abs/2112.13398). | Supplies sensitivity methods matched to flexible DML and partial linearity. Primary metadata now lists DOI 10.1162/REST.a.1705; preserve the actual version. |
| 4 | Angelopoulos, Bates, Fannjiang, Jordan and Zrnic. *Prediction-Powered Inference*. [arXiv:2301.09633](https://arxiv.org/abs/2301.09633), Science DOI 10.1126/science.adi6000. | Connects independent human calibration to inference on large machine-labeled populations. ARES P083 is its directly relevant RAG application. |
| 5 | Wu, Zhong, Kim and Xiong. *What Generative Search Engines Like and How to Optimize Web Content Cooperatively*. [arXiv:2510.11438](https://arxiv.org/abs/2510.11438). | AutoGEO is a missing direct competitor for learned engine preferences and page rewriting. |
| 6 | Fang, Tao, Chen, Chang and Sakai. *Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking*. [arXiv:2509.11353](https://arxiv.org/abs/2509.11353). | Corrects the old freshness comparison and demonstrates a concrete manipulated-feature design. |
| 7 | Feder and colleagues. *Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond*. [arXiv:2109.00725](https://arxiv.org/abs/2109.00725). | Makes the text-treatment, text-measurement and causal-identification boundaries explicit. |
| 8 | Bach, Chernozhukov, Kurz and Spindler. *DoubleML: An Object-Oriented Implementation of Double Machine Learning in Python*. [arXiv:2104.03220](https://arxiv.org/abs/2104.03220). | Supports an accurate estimator description and separation of PLR, treatment effects, binary outcomes and implementation details. |

Prediction-powered inference is a recommendation to study, not an instruction to replace the analysis. A suitable human sample, sampling weights, dependence handling and a compatible estimand are required. It does not turn ordinal grades into interval measurements or make a model-reference panel human gold. Likewise, sensitivity bounds quantify robustness under explicit assumptions; they do not retrospectively establish causality.

## Read existing papers before expanding the download list

For the scientific argument, first read P051, P049, P052 and P044 together with the new CC-GSEO-Bench and AutoGEO. Build a comparison by measured construct, unit of analysis, intervention, source visibility metric, answer-support metric and search architecture. The defensible contribution may be a validated association across a controlled prompt population and multiple bounded search protocols. It is not the invention of source attribution or staged source visibility.

For SI-v4, prioritize P072 AIS, P074 ALCE, P075 FActScore, P076 Attribute First, P068 ContextCite and P083 ARES. AIS and claim-level factuality inform annotation units; ContextCite and the new faithfulness paper clarify influence versus support; ARES and prediction-powered inference inform human calibration. P081, P084 and P085 cover judge validity, self-preference and position effects. A second model's agreement is useful evidence, not a substitute for independent human validation.

For readiness, prioritize P024 LLM2Vec, P006 Broder, P013 search-intent classification, P004 deliberative/implemental mindsets, P029 measurement invariance and P093 text-derived measures. Search intent and a person's action readiness are different constructs. The current measure concerns expressed prompt text, not the user's eventual behavior. P003/P005/P012 already provide ample implementation-intention background; downloading every related mindset experiment would add little.

For statistics, P090, P092, P088 and P093 are already present. They cover DML foundations, multiway dependence and cross-fitting, cluster-robust uncertainty, and the risks of discovering text measures on analysis data. Add the new omitted-variable paper. P089's E-value is not a universal sensitivity measure for the rank and ordinal outcomes here.

The three renewed foundational retrievals are P001 Cronbach and Meehl's construct validity paper, P026 Campbell and Fiske's convergent/discriminant validation paper, and P078 Cohen's weighted kappa paper. P026 has now been recovered and verified by the parent. P001's public text at <https://psychclassics.yorku.ca/Cronbach/construct.htm> is usable reading even without a PDF. P078 remains an explicit access gap. Weighted kappa needs declared weights and marginal distributions; it should supplement, not replace, grade-confusion tables, stratified errors, witness review and map-fidelity assessment.

Useful optional retrievals are P010 on exploratory search, P031 on measurement invariance, P079 if an ICC model is actually justified, and Cinelli and Hazlett's regression-sensitivity exposition, DOI 10.1111/rssb.12348. Historical RankVicuna, setwise ranking and Rank-R1 references are optional if the final paper retains a detailed reranking taxonomy. MS MARCO/TREC dataset papers, reward-model alignment, generic BERT layer narratives and multimodal DML should wait unless those methods return to the actual analysis.

## Old bibliography coverage

Nine of the old manuscript's 36 references already have PDFs in the original library. Five missing old references are in the immediate eight-paper supplement. The other 22 should remain optional or deferred according to their actual role. The full decisions and author lists are in the JSON. The following table records the mapping without duplicating files.

| Reference | Existing library | Retrieval decision |
| --- | --- | --- |
| A Causal Information-Flow Framework for Unbiased Learning-to-Rank | Missing initially | Optional |
| A setwise approach for effective and highly efficient zero-shot ranking with large language models | Missing initially | Optional |
| Aligning Language Models with Observational Data: Opportunities and Risks from a Causal Perspective | Missing initially | Defer |
| Applied Causal Inference Powered by ML and AI | Missing initially | Optional |
| BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models | P056 | Already held |
| BERT Rediscovers the Classical NLP Pipeline | Missing initially | Defer |
| Causal alignment: Augmenting language models with A/B tests | Missing initially | Defer |
| Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond | Missing initially | Download now |
| CC-GSEO-Bench: A Content-Centric Benchmark for Measuring Source Influence in Generative Search Engines | Missing initially | Download now |
| Debiasing Reward Models via Causally Motivated Inference-Time Intervention | Missing initially | Defer |
| Deep neural networks for estimation and inference | Missing initially | Optional |
| Designing and Interpreting Probes with Control Tasks | P016 | Already held |
| Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking | Missing initially | Download now |
| Double/Debiased Machine Learning for Treatment and Causal Parameters | P090 | Already held |
| DoubleML -- An Object-Oriented Implementation of Double Machine Learning in Python | Missing initially | Download now |
| DoubleMLDeep: Estimation of Causal Effects with Multimodal Data | Missing initially | Defer |
| E-GEO: A Testbed for Generative Engine Optimization in E-Commerce | P050 | Already held |
| Estimating Causal Effects with Double Machine Learning -- A Method Evaluation | Missing initially | Optional |
| Generative Engine Optimization: How to Dominate AI Search | P047 | Already held |
| GEO: Generative Engine Optimization | P044 | Already held |
| Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | P041 | Already held |
| Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting | P045 | Already held |
| MS MARCO: A Human Generated MAchine Reading COmprehension Dataset | Missing initially | Defer |
| Qwen2.5 Technical Report | Missing initially | Optional |
| Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning | Missing initially | Optional |
| RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models | Missing initially | Optional |
| RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! | P043 | Already held |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Missing initially | Optional |
| Stepwise multiple testing as formalized data snooping | Missing initially | Optional |
| The Llama 3 Herd of Models | Missing initially | Optional |
| The Rise of AI Search: Implications for Information Markets and Human Judgement at Scale | Missing initially | Optional |
| Transformer Feed-Forward Layers Are Key-Value Memories | Missing initially | Defer |
| TREC Deep Learning Track: Reusable Test Collections in the Large Data Regime | Missing initially | Defer |
| Understanding intermediate layers using linear classifier probes | Missing initially | Optional |
| What Generative Search Engines Like and How to Optimize Web Content Cooperatively | Missing initially | Download now |
| Zero-Shot Listwise Document Reranking with a Large Language Model | Missing initially | Optional |

## Decision after this retrieval round

The next useful work is full-method comparison of the direct competitors and a written estimand/measurement specification for V2. That specification should distinguish prompt-text readiness, request-relative relevance, answer-relative support, citation behavior and intervention-based reliance. It should preserve generation regimes and judge versions, define the sampling and clustering units, and state which condition manipulations actually reached the final evidence. No further broad downloading is necessary before those decisions. None of these recommendations authorizes new inference or changes to historical artifacts.
