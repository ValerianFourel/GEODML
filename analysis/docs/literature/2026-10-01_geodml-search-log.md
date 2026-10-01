# GEODML literature search log

Prepared for Valerian. Search date and publication cutoff: 1 October 2026.

## What was executed

The search used live public OpenAlex and Crossref APIs, direct arXiv abstract/full-text pages and ACL Anthology abstract pages. Broad discovery was followed by title matching, primary-source inspection and limited citation chasing. No general web-search connector was available in this session. No cluster commands or inference jobs were needed.

The result is 93 deduplicated works: 12 with selected full-text sections inspected, 63 with abstracts inspected, and 18 bibliographic leads with metadata only. Each entry names its inspection level. No claim of exhaustive coverage is made.

## Search coverage and limits

The saved plan aims for 15–25 strong papers per theme. This pass did not meet that numerical target in every theme; counts are reported below rather than padded with marginal matches. Most depth is in retrieval, GEO and attribution. Measurement error, continuous exposures, multiple testing and adaptive-intervention identification need further targeted work. Citation chasing was limited, not an exhaustive graph traversal. There was no independent Scholar, Semantic Scholar, ACM DL or OpenReview search in this pass; DOI records may point to those publishers.

OpenAlex and Crossref returned HTTP 429 on some requests. Failed requests remain in the evidence files and tables. Primary-source retrieval and exact-DOI requests recovered many core papers; a failed search is not evidence that a paper does not exist. Dates were filtered through the requested cutoff where supported. Preprint revisions and later indexing may have different dates.

| Theme | Included works |
|---|---|
| 1 | 14 |
| 2 | 11 |
| 3 | 11 |
| 4 | 9 |
| 5 | 13 |
| 6 | 16 |
| 7 | 7 |
| 8 | 12 |
| 9 | 11 |
| 10 | 8 |

## Screening and version checks

- Removed an escaped-newline duplicate of Plug and Play Language Models.
- Merged the “for” and “in” title variants of Query Rewriting after checking identical author lists and matching abstracts.
- Reconciled the DML preprint title with the published title and DOI 10.1111/ectj.12097. The bibliography uses 2018 for the published record.
- Rejected low-similarity title matches; a search-result score was only a screening aid, not a scientific-quality score.
- Rejected unrelated records returned for search-as-learning, GEO and text-causal-inference queries.
- Rejected the remembered DOI 10.1002/asi.23507 after it resolved to a crowdsourcing paper.
- Rejected arXiv identifiers 2106.00725, 1906.04732, 2406.04692, 2403.02113 and 2106.06599 after the fetched titles showed unrelated papers. They remain in raw lookup evidence, not in the bibliography.
- Excluded the Garden of Forking Paths title match because title identity did not establish the intended methodological work. Also excluded several application-specific or irrelevant probe queries.
- OpenAlex occasionally supplied a citation string or the word “published” as an abstract. Those were not treated as abstracts. Six ACL records were then enriched from their primary abstract pages.
- Recent GEO abstracts are labeled as author claims; no peer-review or causal-identification claim is inferred from repository presence. E-GEO remains metadata only.

## Backward and forward citation work

Backward checking used the references and related-work context of the full-text core papers. GEO and the verifiability paper connect optimization to citation auditing; ALCE, AIS, FActScore and RARR provide attribution comparisons; probe and representation papers supply the measurement critiques. This was a targeted check, not a complete reference-list audit.

Forward-citation API queries were executed for GEO, OpenAlex W4401864200, and ALCE, W4389520670. Automatic Document Editing for Improved Ranking was retained as an adjacent optimization comparator. Irrelevant application papers were excluded. Ranking by citation count can miss very recent work, so separate title searches supplied the 2025–2026 GEO leads.

## Query records

### OpenAlex thematic discovery

45 requests; 21 failed responses. [Saved response evidence](2026-10-01_search-evidence.json).

| Query | Outcome | Endpoint |
|---|---|---|
| taxonomy web search informational transactional Broder | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=taxonomy+web+search+informational+transactional+Broder&per-page=8&filter=to_publication_date%3A2026-10-01) |
| exploratory search lookup learning investigation | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=exploratory+search+lookup+learning+investigation&per-page=8&filter=to_publication_date%3A2026-10-01) |
| deliberative implemental mindsets implementation intentions | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=deliberative+implemental+mindsets+implementation+intentions&per-page=8&filter=to_publication_date%3A2026-10-01) |
| information search process uncertainty Kuhlthau | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=information+search+process+uncertainty+Kuhlthau&per-page=8&filter=to_publication_date%3A2026-10-01) |
| LLM2Vec | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=LLM2Vec&per-page=8&filter=to_publication_date%3A2026-10-01) |
| SimCSE contrastive sentence embeddings | Response received; screened separately | [request](https://api.openalex.org/works?search=SimCSE+contrastive+sentence+embeddings&per-page=8&filter=to_publication_date%3A2026-10-01) |
| linear representation hypothesis concept directions | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=linear+representation+hypothesis+concept+directions&per-page=8&filter=to_publication_date%3A2026-10-01) |
| probing classifiers control tasks selectivity | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=probing+classifiers+control+tasks+selectivity&per-page=8&filter=to_publication_date%3A2026-10-01) |
| interpretability beyond feature attribution TCAV | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=interpretability+beyond+feature+attribution+TCAV&per-page=8&filter=to_publication_date%3A2026-10-01) |
| construct validity measurement invariance language models | Response received; screened separately | [request](https://api.openalex.org/works?search=construct+validity+measurement+invariance+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| multitrait multimethod matrix convergent discriminant validity | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=multitrait+multimethod+matrix+convergent+discriminant+validity&per-page=8&filter=to_publication_date%3A2026-10-01) |
| similarity neural network representations CKA | Response received; screened separately | [request](https://api.openalex.org/works?search=similarity+neural+network+representations+CKA&per-page=8&filter=to_publication_date%3A2026-10-01) |
| pitfalls measuring emergent properties language models probes | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=pitfalls+measuring+emergent+properties+language+models+probes&per-page=8&filter=to_publication_date%3A2026-10-01) |
| Self Instruct aligning language models self generated instructions | Response received; screened separately | [request](https://api.openalex.org/works?search=Self+Instruct+aligning+language+models+self+generated+instructions&per-page=8&filter=to_publication_date%3A2026-10-01) |
| controllable text generation plug play language models | Response received; screened separately | [request](https://api.openalex.org/works?search=controllable+text+generation+plug+play+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| contrast sets evaluating local decision boundaries | Response received; screened separately | [request](https://api.openalex.org/works?search=contrast+sets+evaluating+local+decision+boundaries&per-page=8&filter=to_publication_date%3A2026-10-01) |
| text embeddings reveal almost as much as text | Response received; screened separately | [request](https://api.openalex.org/works?search=text+embeddings+reveal+almost+as+much+as+text&per-page=8&filter=to_publication_date%3A2026-10-01) |
| synthetic data model collapse language models | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=synthetic+data+model+collapse+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| GEO generative engine optimization | Response received; screened separately | [request](https://api.openalex.org/works?search=GEO+generative+engine+optimization&per-page=8&filter=to_publication_date%3A2026-10-01) |
| generative search citation source bias | Response received; screened separately | [request](https://api.openalex.org/works?search=generative+search+citation+source+bias&per-page=8&filter=to_publication_date%3A2026-10-01) |
| generative engine optimization benchmark robustness | Response received; screened separately | [request](https://api.openalex.org/works?search=generative+engine+optimization+benchmark+robustness&per-page=8&filter=to_publication_date%3A2026-10-01) |
| adversarial search engine optimization large language models | Response received; screened separately | [request](https://api.openalex.org/works?search=adversarial+search+engine+optimization+large+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| ReAct synergizing reasoning acting language models | Response received; screened separately | [request](https://api.openalex.org/works?search=ReAct+synergizing+reasoning+acting+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| retrieval augmented generation knowledge intensive NLP | Response received; screened separately | [request](https://api.openalex.org/works?search=retrieval+augmented+generation+knowledge+intensive+NLP&per-page=8&filter=to_publication_date%3A2026-10-01) |
| IRCoT interleaving retrieval chain thought | Response received; screened separately | [request](https://api.openalex.org/works?search=IRCoT+interleaving+retrieval+chain+thought&per-page=8&filter=to_publication_date%3A2026-10-01) |
| query rewriting retrieval augmented large language models | Response received; screened separately | [request](https://api.openalex.org/works?search=query+rewriting+retrieval+augmented+large+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| Self RAG adaptive retrieval | Response received; screened separately | [request](https://api.openalex.org/works?search=Self+RAG+adaptive+retrieval&per-page=8&filter=to_publication_date%3A2026-10-01) |
| lost in the middle long contexts | Response received; screened separately | [request](https://api.openalex.org/works?search=lost+in+the+middle+long+contexts&per-page=8&filter=to_publication_date%3A2026-10-01) |
| large language models sensitivity order retrieved documents | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=large+language+models+sensitivity+order+retrieved+documents&per-page=8&filter=to_publication_date%3A2026-10-01) |
| counterfactual retrieval source attribution language models | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=counterfactual+retrieval+source+attribution+language+models&per-page=8&filter=to_publication_date%3A2026-10-01) |
| retrieval augmented generation irrelevant context robustness | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=retrieval+augmented+generation+irrelevant+context+robustness&per-page=8&filter=to_publication_date%3A2026-10-01) |
| attributable grounded generation citation ALCE | Response received; screened separately | [request](https://api.openalex.org/works?search=attributable+grounded+generation+citation+ALCE&per-page=8&filter=to_publication_date%3A2026-10-01) |
| evaluating verifiability generative search engines | Response received; screened separately | [request](https://api.openalex.org/works?search=evaluating+verifiability+generative+search+engines&per-page=8&filter=to_publication_date%3A2026-10-01) |
| RARR researching revising attribution | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=RARR+researching+revising+attribution&per-page=8&filter=to_publication_date%3A2026-10-01) |
| context citing source attribution language models Shapley | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=context+citing+source+attribution+language+models+Shapley&per-page=8&filter=to_publication_date%3A2026-10-01) |
| judging LLM as a judge MT Bench chatbot arena | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=judging+LLM+as+a+judge+MT+Bench+chatbot+arena&per-page=8&filter=to_publication_date%3A2026-10-01) |
| LLM evaluators bias position verbosity self preference | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=LLM+evaluators+bias+position+verbosity+self+preference&per-page=8&filter=to_publication_date%3A2026-10-01) |
| G Eval NLG evaluation GPT4 human alignment | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=G+Eval+NLG+evaluation+GPT4+human+alignment&per-page=8&filter=to_publication_date%3A2026-10-01) |
| FActScore fine grained atomic evaluation | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=FActScore+fine+grained+atomic+evaluation&per-page=8&filter=to_publication_date%3A2026-10-01) |
| RAGAS automated evaluation retrieval augmented generation | Response received; screened separately | [request](https://api.openalex.org/works?search=RAGAS+automated+evaluation+retrieval+augmented+generation&per-page=8&filter=to_publication_date%3A2026-10-01) |
| double debiased machine learning treatment structural parameters | Response received; screened separately | [request](https://api.openalex.org/works?search=double+debiased+machine+learning+treatment+structural+parameters&per-page=8&filter=to_publication_date%3A2026-10-01) |
| multiway cluster robust double machine learning | Response received; screened separately | [request](https://api.openalex.org/works?search=multiway+cluster+robust+double+machine+learning&per-page=8&filter=to_publication_date%3A2026-10-01) |
| text causal inference assumptions challenges | Response received; screened separately | [request](https://api.openalex.org/works?search=text+causal+inference+assumptions+challenges&per-page=8&filter=to_publication_date%3A2026-10-01) |
| measurement error causal inference text | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?search=measurement+error+causal+inference+text&per-page=8&filter=to_publication_date%3A2026-10-01) |
| rank biased overlap incomplete rankings | Response received; screened separately | [request](https://api.openalex.org/works?search=rank+biased+overlap+incomplete+rankings&per-page=8&filter=to_publication_date%3A2026-10-01) |

### Crossref title searches

102 requests; 48 failed responses. [Saved response evidence](2026-10-01_targeted-search-evidence.json).

| Query | Outcome | Endpoint |
|---|---|---|
| A taxonomy of web search | Response received; screened separately | [request](https://api.crossref.org/works?query.title=A+taxonomy+of+web+search&rows=3&filter=until-pub-date%3A2026-10-01) |
| Exploratory search: from finding to understanding | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Exploratory+search%3A+from+finding+to+understanding&rows=3&filter=until-pub-date%3A2026-10-01) |
| Inside the search process: Information seeking from the user's perspective | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Inside+the+search+process%3A+Information+seeking+from+the+user%27s+perspective&rows=3&filter=until-pub-date%3A2026-10-01) |
| A Model of the Information Search Process | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=A+Model+of+the+Information+Search+Process&rows=3&filter=until-pub-date%3A2026-10-01) |
| Implementation intentions: Strong effects of simple plans | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Implementation+intentions%3A+Strong+effects+of+simple+plans&rows=3&filter=until-pub-date%3A2026-10-01) |
| Implementation intentions and goal achievement: A meta-analysis of effects and processes | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Implementation+intentions+and+goal+achievement%3A+A+meta-analysis+of+effects+and+processes&rows=3&filter=until-pub-date%3A2026-10-01) |
| Crossing the Rubicon: Decisional effectiveness through commitment and planning | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Crossing+the+Rubicon%3A+Decisional+effectiveness+through+commitment+and+planning&rows=3&filter=until-pub-date%3A2026-10-01) |
| Determining the informational, navigational, and transactional intent of Web queries | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Determining+the+informational%2C+navigational%2C+and+transactional+intent+of+Web+queries&rows=3&filter=until-pub-date%3A2026-10-01) |
| Search as learning: Exploring, understanding, and evaluating | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Search+as+learning%3A+Exploring%2C+understanding%2C+and+evaluating&rows=3&filter=until-pub-date%3A2026-10-01) |
| Construct validity in psychological tests | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Construct+validity+in+psychological+tests&rows=3&filter=until-pub-date%3A2026-10-01) |
| LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=LLM2Vec%3A+Large+Language+Models+Are+Secretly+Powerful+Text+Encoders&rows=3&filter=until-pub-date%3A2026-10-01) |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=SimCSE%3A+Simple+Contrastive+Learning+of+Sentence+Embeddings&rows=3&filter=until-pub-date%3A2026-10-01) |
| Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Interpretability+Beyond+Feature+Attribution%3A+Quantitative+Testing+with+Concept+Activation+Vectors+%28TCAV%29&rows=3&filter=until-pub-date%3A2026-10-01) |
| The Linear Representation Hypothesis and the Geometry of Large Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=The+Linear+Representation+Hypothesis+and+the+Geometry+of+Large+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Representation Engineering: A Top-Down Approach to AI Transparency | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Representation+Engineering%3A+A+Top-Down+Approach+to+AI+Transparency&rows=3&filter=until-pub-date%3A2026-10-01) |
| Designing and Interpreting Probes with Control Tasks | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Designing+and+Interpreting+Probes+with+Control+Tasks&rows=3&filter=until-pub-date%3A2026-10-01) |
| Probing Classifiers: Promises, Shortcomings, and Advances | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Probing+Classifiers%3A+Promises%2C+Shortcomings%2C+and+Advances&rows=3&filter=until-pub-date%3A2026-10-01) |
| A Structural Probe for Finding Syntax in Word Representations | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=A+Structural+Probe+for+Finding+Syntax+in+Word+Representations&rows=3&filter=until-pub-date%3A2026-10-01) |
| Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Null+It+Out%3A+Guarding+Protected+Attributes+by+Iterative+Nullspace+Projection&rows=3&filter=until-pub-date%3A2026-10-01) |
| LEACE: Perfect linear concept erasure in closed form | Response received; screened separately | [request](https://api.crossref.org/works?query.title=LEACE%3A+Perfect+linear+concept+erasure+in+closed+form&rows=3&filter=until-pub-date%3A2026-10-01) |
| Similarity of Neural Network Representations Revisited | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Similarity+of+Neural+Network+Representations+Revisited&rows=3&filter=until-pub-date%3A2026-10-01) |
| Convergent and discriminant validation by the multitrait-multimethod matrix | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Convergent+and+discriminant+validation+by+the+multitrait-multimethod+matrix&rows=3&filter=until-pub-date%3A2026-10-01) |
| Measurement invariance, factor analysis and factorial invariance | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Measurement+invariance%2C+factor+analysis+and+factorial+invariance&rows=3&filter=until-pub-date%3A2026-10-01) |
| Measurement invariance conventions and reporting: The state of the art and future directions for psychological research | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Measurement+invariance+conventions+and+reporting%3A+The+state+of+the+art+and+future+directions+for+psychological+research&rows=3&filter=until-pub-date%3A2026-10-01) |
| A generalized solution of the orthogonal procrustes problem | Response received; screened separately | [request](https://api.crossref.org/works?query.title=A+generalized+solution+of+the+orthogonal+procrustes+problem&rows=3&filter=until-pub-date%3A2026-10-01) |
| A Unified Framework for Measuring Preferences for Schools and Neighborhoods | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=A+Unified+Framework+for+Measuring+Preferences+for+Schools+and+Neighborhoods&rows=3&filter=until-pub-date%3A2026-10-01) |
| Disentangling construct validity and measurement invariance in psychological research | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Disentangling+construct+validity+and+measurement+invariance+in+psychological+research&rows=3&filter=until-pub-date%3A2026-10-01) |
| The garden of forking paths | Response received; screened separately | [request](https://api.crossref.org/works?query.title=The+garden+of+forking+paths&rows=3&filter=until-pub-date%3A2026-10-01) |
| Statistical Modeling: The Two Cultures | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Statistical+Modeling%3A+The+Two+Cultures&rows=3&filter=until-pub-date%3A2026-10-01) |
| Self-Instruct: Aligning Language Models with Self-Generated Instructions | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Self-Instruct%3A+Aligning+Language+Models+with+Self-Generated+Instructions&rows=3&filter=until-pub-date%3A2026-10-01) |
| Self-Instruct: Aligning Language Models with Self-Generated Instructions | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Self-Instruct%3A+Aligning+Language+Models+with+Self-Generated+Instructions&rows=3&filter=until-pub-date%3A2026-10-01) |
| Plug and Play Language Models: A Simple Approach to Controlled Text Generation | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Plug+and+Play+Language+Models%3A+A+Simple+Approach+to+Controlled+Text+Generation&rows=3&filter=until-pub-date%3A2026-10-01) |
| DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=DExperts%3A+Decoding-Time+Controlled+Text+Generation+with+Experts+and+Anti-Experts&rows=3&filter=until-pub-date%3A2026-10-01) |
| FUDGE: Controlled Text Generation With Future Discriminators | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=FUDGE%3A+Controlled+Text+Generation+With+Future+Discriminators&rows=3&filter=until-pub-date%3A2026-10-01) |
| Evaluating Models' Local Decision Boundaries via Contrast Sets | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Evaluating+Models%27+Local+Decision+Boundaries+via+Contrast+Sets&rows=3&filter=until-pub-date%3A2026-10-01) |
| Beyond Accuracy: Behavioral Testing of NLP Models with CheckList | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Beyond+Accuracy%3A+Behavioral+Testing+of+NLP+Models+with+CheckList&rows=3&filter=until-pub-date%3A2026-10-01) |
| Text Embeddings Reveal (Almost) As Much As Text | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Text+Embeddings+Reveal+%28Almost%29+As+Much+As+Text&rows=3&filter=until-pub-date%3A2026-10-01) |
| Large Language Models for Controllable Text Generation: A Survey | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Large+Language+Models+for+Controllable+Text+Generation%3A+A+Survey&rows=3&filter=until-pub-date%3A2026-10-01) |
| AI models collapse when trained on recursively generated data | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=AI+models+collapse+when+trained+on+recursively+generated+data&rows=3&filter=until-pub-date%3A2026-10-01) |
| Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Synthetic+Data+%28Almost%29+from+Scratch%3A+Generalized+Instruction+Tuning+for+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| WizardLM: Empowering Large Language Models to Follow Complex Instructions | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=WizardLM%3A+Empowering+Large+Language+Models+to+Follow+Complex+Instructions&rows=3&filter=until-pub-date%3A2026-10-01) |
| GEO: Generative Engine Optimization | Response received; screened separately | [request](https://api.crossref.org/works?query.title=GEO%3A+Generative+Engine+Optimization&rows=3&filter=until-pub-date%3A2026-10-01) |
| E-GEO: A Testbed for Generative Engine Optimization in E-Commerce | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=E-GEO%3A+A+Testbed+for+Generative+Engine+Optimization+in+E-Commerce&rows=3&filter=until-pub-date%3A2026-10-01) |
| Generative Engine Optimization: How to Dominate AI Search | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Generative+Engine+Optimization%3A+How+to+Dominate+AI+Search&rows=3&filter=until-pub-date%3A2026-10-01) |
| Evaluating Verifiability in Generative Search Engines | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Evaluating+Verifiability+in+Generative+Search+Engines&rows=3&filter=until-pub-date%3A2026-10-01) |
| Manipulating Large Language Models to Increase Product Visibility | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Manipulating+Large+Language+Models+to+Increase+Product+Visibility&rows=3&filter=until-pub-date%3A2026-10-01) |
| PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=PoisonedRAG%3A+Knowledge+Corruption+Attacks+to+Retrieval-Augmented+Generation+of+Large+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Not+what+you%27ve+signed+up+for%3A+Compromising+Real-World+LLM-Integrated+Applications+with+Indirect+Prompt+Injection&rows=3&filter=until-pub-date%3A2026-10-01) |
| Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Is+ChatGPT+Good+at+Search%3F+Investigating+Large+Language+Models+as+Re-Ranking+Agents&rows=3&filter=until-pub-date%3A2026-10-01) |
| Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Large+Language+Models+are+Effective+Text+Rankers+with+Pairwise+Ranking+Prompting&rows=3&filter=until-pub-date%3A2026-10-01) |
| RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=RankZephyr%3A+Effective+and+Robust+Zero-Shot+Listwise+Reranking+is+a+Breeze%21&rows=3&filter=until-pub-date%3A2026-10-01) |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Retrieval-Augmented+Generation+for+Knowledge-Intensive+NLP+Tasks&rows=3&filter=until-pub-date%3A2026-10-01) |
| ReAct: Synergizing Reasoning and Acting in Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=ReAct%3A+Synergizing+Reasoning+and+Acting+in+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Interleaving+Retrieval+with+Chain-of-Thought+Reasoning+for+Knowledge-Intensive+Multi-Step+Questions&rows=3&filter=until-pub-date%3A2026-10-01) |
| Query Rewriting in Retrieval-Augmented Large Language Models | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Query+Rewriting+in+Retrieval-Augmented+Large+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Self-RAG%3A+Learning+to+Retrieve%2C+Generate%2C+and+Critique+through+Self-Reflection&rows=3&filter=until-pub-date%3A2026-10-01) |
| Active Retrieval Augmented Generation | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Active+Retrieval+Augmented+Generation&rows=3&filter=until-pub-date%3A2026-10-01) |
| Precise Zero-Shot Dense Retrieval without Relevance Labels | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Precise+Zero-Shot+Dense+Retrieval+without+Relevance+Labels&rows=3&filter=until-pub-date%3A2026-10-01) |
| Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Adaptive-RAG%3A+Learning+to+Adapt+Retrieval-Augmented+Large+Language+Models+through+Question+Complexity&rows=3&filter=until-pub-date%3A2026-10-01) |
| Dense Passage Retrieval for Open-Domain Question Answering | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Dense+Passage+Retrieval+for+Open-Domain+Question+Answering&rows=3&filter=until-pub-date%3A2026-10-01) |
| Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Leveraging+Passage+Retrieval+with+Generative+Models+for+Open+Domain+Question+Answering&rows=3&filter=until-pub-date%3A2026-10-01) |
| BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=BEIR%3A+A+Heterogeneous+Benchmark+for+Zero-shot+Evaluation+of+Information+Retrieval+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=BGE+M3-Embedding%3A+Multi-Linguality%2C+Multi-Functionality%2C+Multi-Granularity+Text+Embeddings+Through+Self-Knowledge+Distillation&rows=3&filter=until-pub-date%3A2026-10-01) |
| Lost in the Middle: How Language Models Use Long Contexts | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Lost+in+the+Middle%3A+How+Language+Models+Use+Long+Contexts&rows=3&filter=until-pub-date%3A2026-10-01) |
| Lost in the Middle, and In-Between: Enhancing Language Models' Ability to Reason Over Long Contexts in Multi-Hop QA | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Lost+in+the+Middle%2C+and+In-Between%3A+Enhancing+Language+Models%27+Ability+to+Reason+Over+Long+Contexts+in+Multi-Hop+QA&rows=3&filter=until-pub-date%3A2026-10-01) |
| Making Retrieval-Augmented Language Models Robust to Irrelevant Context | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Making+Retrieval-Augmented+Language+Models+Robust+to+Irrelevant+Context&rows=3&filter=until-pub-date%3A2026-10-01) |
| The Power of Noise: Redefining Retrieval for RAG Systems | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=The+Power+of+Noise%3A+Redefining+Retrieval+for+RAG+Systems&rows=3&filter=until-pub-date%3A2026-10-01) |
| Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Fantastically+Ordered+Prompts+and+Where+to+Find+Them%3A+Overcoming+Few-Shot+Prompt+Order+Sensitivity&rows=3&filter=until-pub-date%3A2026-10-01) |
| ContextCite: Attributing Model Generation to Context | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=ContextCite%3A+Attributing+Model+Generation+to+Context&rows=3&filter=until-pub-date%3A2026-10-01) |
| Rethinking the Role of Proxy Rewards in Language Model Alignment | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Rethinking+the+Role+of+Proxy+Rewards+in+Language+Model+Alignment&rows=3&filter=until-pub-date%3A2026-10-01) |
| Enabling Large Language Models to Generate Text with Citations | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Enabling+Large+Language+Models+to+Generate+Text+with+Citations&rows=3&filter=until-pub-date%3A2026-10-01) |
| RARR: Researching and Revising What Language Models Say, Using Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=RARR%3A+Researching+and+Revising+What+Language+Models+Say%2C+Using+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Attribution and Verifiability in Language Models | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Attribution+and+Verifiability+in+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Measuring Attribution in Natural Language Generation Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Measuring+Attribution+in+Natural+Language+Generation+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| Attribute First, then Generate: Locally-attributable Grounded Text Generation | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Attribute+First%2C+then+Generate%3A+Locally-attributable+Grounded+Text+Generation&rows=3&filter=until-pub-date%3A2026-10-01) |
| FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation | Response received; screened separately | [request](https://api.crossref.org/works?query.title=FActScore%3A+Fine-grained+Atomic+Evaluation+of+Factual+Precision+in+Long+Form+Text+Generation&rows=3&filter=until-pub-date%3A2026-10-01) |
| LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-Context QA | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=LongCite%3A+Enabling+LLMs+to+Generate+Fine-grained+Citations+in+Long-Context+QA&rows=3&filter=until-pub-date%3A2026-10-01) |
| A similarity measure for indefinite rankings | Response received; screened separately | [request](https://api.crossref.org/works?query.title=A+similarity+measure+for+indefinite+rankings&rows=3&filter=until-pub-date%3A2026-10-01) |
| Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Judging+LLM-as-a-Judge+with+MT-Bench+and+Chatbot+Arena&rows=3&filter=until-pub-date%3A2026-10-01) |
| G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=G-Eval%3A+NLG+Evaluation+using+GPT-4+with+Better+Human+Alignment&rows=3&filter=until-pub-date%3A2026-10-01) |
| Large Language Models are not Fair Evaluators | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Large+Language+Models+are+not+Fair+Evaluators&rows=3&filter=until-pub-date%3A2026-10-01) |
| Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Judging+the+Judges%3A+Evaluating+Alignment+and+Vulnerabilities+in+LLMs-as-Judges&rows=3&filter=until-pub-date%3A2026-10-01) |
| LLM Evaluators Recognize and Favor Their Own Generations | Response received; screened separately | [request](https://api.crossref.org/works?query.title=LLM+Evaluators+Recognize+and+Favor+Their+Own+Generations&rows=3&filter=until-pub-date%3A2026-10-01) |
| RAGAs: Automated Evaluation of Retrieval Augmented Generation | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=RAGAs%3A+Automated+Evaluation+of+Retrieval+Augmented+Generation&rows=3&filter=until-pub-date%3A2026-10-01) |
| ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | Response received; screened separately | [request](https://api.crossref.org/works?query.title=ARES%3A+An+Automated+Evaluation+Framework+for+Retrieval-Augmented+Generation+Systems&rows=3&filter=until-pub-date%3A2026-10-01) |
| Prometheus: Inducing Fine-grained Evaluation Capability in Language Models | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Prometheus%3A+Inducing+Fine-grained+Evaluation+Capability+in+Language+Models&rows=3&filter=until-pub-date%3A2026-10-01) |
| A coefficient of agreement for nominal scales | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=A+coefficient+of+agreement+for+nominal+scales&rows=3&filter=until-pub-date%3A2026-10-01) |
| Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Weighted+kappa%3A+Nominal+scale+agreement+provision+for+scaled+disagreement+or+partial+credit&rows=3&filter=until-pub-date%3A2026-10-01) |
| Intraclass correlations: Uses in assessing rater reliability | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Intraclass+correlations%3A+Uses+in+assessing+rater+reliability&rows=3&filter=until-pub-date%3A2026-10-01) |
| Double/debiased machine learning for treatment and structural parameters | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Double%2Fdebiased+machine+learning+for+treatment+and+structural+parameters&rows=3&filter=until-pub-date%3A2026-10-01) |
| Multiway Cluster Robust Double/Debiased Machine Learning | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Multiway+Cluster+Robust+Double%2FDebiased+Machine+Learning&rows=3&filter=until-pub-date%3A2026-10-01) |
| Text and Causal Inference: A Review of Using Text to Remove Confounding from Causal Estimates | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Text+and+Causal+Inference%3A+A+Review+of+Using+Text+to+Remove+Confounding+from+Causal+Estimates&rows=3&filter=until-pub-date%3A2026-10-01) |
| Using Machine Learning to Estimate the Effect of Racial Segregation on COVID-19 Mortality in the United States | Response received; screened separately | [request](https://api.crossref.org/works?query.title=Using+Machine+Learning+to+Estimate+the+Effect+of+Racial+Segregation+on+COVID-19+Mortality+in+the+United+States&rows=3&filter=until-pub-date%3A2026-10-01) |
| Causal Inference with Text Data | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=Causal+Inference+with+Text+Data&rows=3&filter=until-pub-date%3A2026-10-01) |
| How to Make Causal Inferences Using Texts | Response received; screened separately | [request](https://api.crossref.org/works?query.title=How+to+Make+Causal+Inferences+Using+Texts&rows=3&filter=until-pub-date%3A2026-10-01) |
| A Practical Guide to Weighting and Aggregating Causal Effects with Double Machine Learning | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=A+Practical+Guide+to+Weighting+and+Aggregating+Causal+Effects+with+Double+Machine+Learning&rows=3&filter=until-pub-date%3A2026-10-01) |
| A Practitioner's Guide to Cluster-Robust Inference | Response received; screened separately | [request](https://api.crossref.org/works?query.title=A+Practitioner%27s+Guide+to+Cluster-Robust+Inference&rows=3&filter=until-pub-date%3A2026-10-01) |
| Sensitivity Analysis in Observational Research: Introducing the E-Value | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Sensitivity+Analysis+in+Observational+Research%3A+Introducing+the+E-Value&rows=3&filter=until-pub-date%3A2026-10-01) |
| An Introduction to Double/Debiased Machine Learning | Response received; no close title match accepted automatically | [request](https://api.crossref.org/works?query.title=An+Introduction+to+Double%2FDebiased+Machine+Learning&rows=3&filter=until-pub-date%3A2026-10-01) |
| The central role of the propensity score in observational studies for causal effects | Response received; screened separately | [request](https://api.crossref.org/works?query.title=The+central+role+of+the+propensity+score+in+observational+studies+for+causal+effects&rows=3&filter=until-pub-date%3A2026-10-01) |
| Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing | HTTP Error 429: Too Many Requests | [request](https://api.crossref.org/works?query.title=Controlling+the+False+Discovery+Rate%3A+A+Practical+and+Powerful+Approach+to+Multiple+Testing&rows=3&filter=until-pub-date%3A2026-10-01) |
| A new look at the statistical model identification | Response received; screened separately | [request](https://api.crossref.org/works?query.title=A+new+look+at+the+statistical+model+identification&rows=3&filter=until-pub-date%3A2026-10-01) |

### OpenAlex/DOI supplements

18 requests; 1 failed responses. [Saved response evidence](2026-10-01_supplemental-evidence.json).

| Query | Outcome | Endpoint |
|---|---|---|
| generative engine optimization | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Agenerative+engine+optimization%2Cto_publication_date%3A2026-10-01&per-page=25&sort=relevance_score%3Adesc) |
| source attribution language models | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Asource+attribution+language+models%2Cto_publication_date%3A2026-10-01&per-page=25&sort=relevance_score%3Adesc) |
| action readiness search | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Aaction+readiness+search%2Cto_publication_date%3A2026-10-01&per-page=25&sort=relevance_score%3Adesc) |
| measurement language models construct validity | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Ameasurement+language+models+construct+validity%2Cto_publication_date%3A2026-10-01&per-page=25&sort=relevance_score%3Adesc) |
| LLM2Vec-Gen | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3ALLM2Vec-Gen%2Cto_publication_date%3A2026-10-01&per-page=25&sort=relevance_score%3Adesc) |
| 10.1145/1121949.1121979 | Response received; screened separately | [request](https://api.crossref.org/works/10.1145%2F1121949.1121979) |
| 10.1037/0003-066X.54.7.493 | Response received; screened separately | [request](https://api.crossref.org/works/10.1037%2F0003-066X.54.7.493) |
| 10.1016/S0065-2601(06)38002-1 | Response received; screened separately | [request](https://api.crossref.org/works/10.1016%2FS0065-2601%2806%2938002-1) |
| 10.1002/asi.23507 | Response received; screened separately | [request](https://api.crossref.org/works/10.1002%2Fasi.23507) |
| 10.1037/h0046016 | Response received; screened separately | [request](https://api.crossref.org/works/10.1037%2Fh0046016) |
| 10.1007/BF02294825 | Response received; screened separately | [request](https://api.crossref.org/works/10.1007%2FBF02294825) |
| 10.1007/BF02289447 | Response received; screened separately | [request](https://api.crossref.org/works/10.1007%2FBF02289447) |
| 10.1214/ss/1009213726 | Response received; screened separately | [request](https://api.crossref.org/works/10.1214%2Fss%2F1009213726) |
| 10.1038/s41586-024-07566-y | Response received; screened separately | [request](https://api.crossref.org/works/10.1038%2Fs41586-024-07566-y) |
| 10.1111/ectj.12097 | Response received; screened separately | [request](https://api.crossref.org/works/10.1111%2Fectj.12097) |
| 10.2307/2346101 | HTTP Error 404: Not Found | [request](https://api.crossref.org/works/10.2307%2F2346101) |
| 10.1037/0033-2909.86.2.420 | Response received; screened separately | [request](https://api.crossref.org/works/10.1037%2F0033-2909.86.2.420) |
| 10.7326/M16-2607 | Response received; screened separately | [request](https://api.crossref.org/works/10.7326%2FM16-2607) |

### Follow-up and citation chasing

7 requests; 1 failed responses. [Saved response evidence](2026-10-01_followup-search-evidence.json).

| Query | Outcome | Endpoint |
|---|---|---|
| search as learning | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Asearch+as+learning%2Cto_publication_date%3A2026-10-01&per-page=5) |
| deliberative implemental mindsets | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Adeliberative+implemental+mindsets%2Cto_publication_date%3A2026-10-01&per-page=5) |
| goal intentions implementation intentions | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Agoal+intentions+implementation+intentions%2Cto_publication_date%3A2026-10-01&per-page=5) |
| measurement error text causal inference | Response received; screened separately | [request](https://api.openalex.org/works?filter=title.search%3Ameasurement+error+text+causal+inference%2Cto_publication_date%3A2026-10-01&per-page=5) |
| double machine learning continuous treatments | HTTP Error 429: Too Many Requests | [request](https://api.openalex.org/works?filter=title.search%3Adouble+machine+learning+continuous+treatments%2Cto_publication_date%3A2026-10-01&per-page=5) |
| Forward citations W4401864200 | Response received; screened separately | [request](https://api.openalex.org/works?filter=cites%3AW4401864200%2Cto_publication_date%3A2026-10-01&per-page=8&sort=cited_by_count%3Adesc) |
| Forward citations W4389520670 | Response received; screened separately | [request](https://api.openalex.org/works?filter=cites%3AW4389520670%2Cto_publication_date%3A2026-10-01&per-page=8&sort=cited_by_count%3Adesc) |

### 2026-10-01_primary-source-evidence.json

| Lookup | Retrieved title | Full text available |
|---|---|---|
| [2404.05961](https://arxiv.org/abs/2404.05961) | LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders | Selected sections inspected |
| [2311.09735](https://arxiv.org/abs/2311.09735) | GEO: Generative Engine Optimization | Selected sections inspected |
| [2104.08821](https://arxiv.org/abs/2104.08821) | SimCSE: Simple Contrastive Learning of Sentence Embeddings | Abstract lookup only |
| [2311.03658](https://arxiv.org/abs/2311.03658) | The Linear Representation Hypothesis and the Geometry of Large Language Models | Selected sections inspected |
| [2310.01405](https://arxiv.org/abs/2310.01405) | Representation Engineering: A Top-Down Approach to AI Transparency | Abstract lookup only |
| [2210.03629](https://arxiv.org/abs/2210.03629) | ReAct: Synergizing Reasoning and Acting in Language Models | Abstract lookup only |
| [2005.11401](https://arxiv.org/abs/2005.11401) | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Abstract lookup only |
| [2212.10509](https://arxiv.org/abs/2212.10509) | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | Abstract lookup only |
| [2310.11511](https://arxiv.org/abs/2310.11511) | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Selected sections inspected |
| [2305.14627](https://arxiv.org/abs/2305.14627) | Enabling Large Language Models to Generate Text with Citations | Selected sections inspected |
| [2210.08726](https://arxiv.org/abs/2210.08726) | RARR: Researching and Revising What Language Models Say, Using Language Models | Abstract lookup only |
| [2304.09848](https://arxiv.org/abs/2304.09848) | Evaluating Verifiability in Generative Search Engines | Selected sections inspected |
| [2307.03172](https://arxiv.org/abs/2307.03172) | Lost in the Middle: How Language Models Use Long Contexts | Selected sections inspected |
| [2409.00729](https://arxiv.org/abs/2409.00729) | ContextCite: Attributing Model Generation to Context | Selected sections inspected |
| [2305.14251](https://arxiv.org/abs/2305.14251) | FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation | Selected sections inspected |
| [2306.05685](https://arxiv.org/abs/2306.05685) | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Selected sections inspected |
| [2303.16634](https://arxiv.org/abs/2303.16634) | G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment | Abstract lookup only |
| [2212.10560](https://arxiv.org/abs/2212.10560) | Self-Instruct: Aligning Language Models with Self-Generated Instructions | Abstract lookup only |
| [2310.06816](https://arxiv.org/abs/2310.06816) | Text Embeddings Reveal (Almost) As Much As Text | Abstract lookup only |
| [1608.00060](https://arxiv.org/abs/1608.00060) | Double/Debiased Machine Learning for Treatment and Causal Parameters | Selected sections inspected |
| [1711.11279](https://arxiv.org/abs/1711.11279) | Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) | Abstract lookup only |
| [1905.00414](https://arxiv.org/abs/1905.00414) | Similarity of Neural Network Representations Revisited | Abstract lookup only |
| [2306.03819](https://arxiv.org/abs/2306.03819) | LEACE: Perfect linear concept erasure in closed form | Abstract lookup only |
| [1909.03368](https://arxiv.org/abs/1909.03368) | Designing and Interpreting Probes with Control Tasks | Abstract lookup only |
| [2005.04118](https://arxiv.org/abs/2005.04118) | Beyond Accuracy: Behavioral Testing of NLP models with CheckList | Abstract lookup only |
| [2305.17926](https://arxiv.org/abs/2305.17926) | Large Language Models are not Fair Evaluators | Abstract lookup only |
| [2401.14887](https://arxiv.org/abs/2401.14887) | The Power of Noise: Redefining Retrieval for RAG Systems | Selected sections inspected |
| [2310.01558](https://arxiv.org/abs/2310.01558) | Making Retrieval-Augmented Language Models Robust to Irrelevant Context | Abstract lookup only |

### 2026-10-01_additional-primary-evidence.json

| Lookup | Retrieved title | Full text available |
|---|---|---|
| [1912.02164](https://arxiv.org/abs/1912.02164) | Plug and Play Language Models: A Simple Approach to Controlled Text Generation | Abstract lookup only |
| [2106.00725](https://arxiv.org/abs/2106.00725) | Coupler-Assisted Controlled-Phase Gate with Enhanced Adiabaticity | Abstract lookup only |
| [2104.05218](https://arxiv.org/abs/2104.05218) | FUDGE: Controlled Text Generation With Future Discriminators | Abstract lookup only |
| [2304.12244](https://arxiv.org/abs/2304.12244) | WizardLM: Empowering large pre-trained language models to follow complex instructions | Abstract lookup only |
| [2304.09542](https://arxiv.org/abs/2304.09542) | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | Abstract lookup only |
| [2312.02724](https://arxiv.org/abs/2312.02724) | RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! | Abstract lookup only |
| [2306.17563](https://arxiv.org/abs/2306.17563) | Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting | Abstract lookup only |
| [2104.08663](https://arxiv.org/abs/2104.08663) | BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models | Abstract lookup only |
| [2007.01282](https://arxiv.org/abs/2007.01282) | Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering | Abstract lookup only |
| [2402.03216](https://arxiv.org/abs/2402.03216) | M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | Abstract lookup only |
| [2004.07667](https://arxiv.org/abs/2004.07667) | Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection | Abstract lookup only |
| [1906.04732](https://arxiv.org/abs/1906.04732) | Convergence analysis of a Crank-Nicolson Galerkin method for an inverse source problem for parabolic equations with boundary observations | Abstract lookup only |
| [2112.12870](https://arxiv.org/abs/2112.12870) | Measuring Attribution in Natural Language Generation Models | Abstract lookup only |
| [2309.15217](https://arxiv.org/abs/2309.15217) | Ragas: Automated Evaluation of Retrieval Augmented Generation | Abstract lookup only |
| [2311.09476](https://arxiv.org/abs/2311.09476) | ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | Abstract lookup only |
| [2310.08491](https://arxiv.org/abs/2310.08491) | Prometheus: Inducing Fine-grained Evaluation Capability in Language Models | Abstract lookup only |
| [2406.04692](https://arxiv.org/abs/2406.04692) | Mixture-of-Agents Enhances Large Language Model Capabilities | Abstract lookup only |
| [2403.02113](https://arxiv.org/abs/2403.02113) | Evolving disorder and chaos enhances the wave speed of elastic waves | Abstract lookup only |

### 2026-10-01_recent-primary-evidence.json

| Lookup | Retrieved title | Full text available |
|---|---|---|
| [2509.08919](https://arxiv.org/abs/2509.08919) | Generative Engine Optimization: How to Dominate AI Search | Abstract lookup only |
| [2603.09296](https://arxiv.org/abs/2603.09296) | Diagnosing and Repairing Citation Failures in Generative Engine Optimization | Abstract lookup only |
| [2604.25707](https://arxiv.org/abs/2604.25707) | From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms | Abstract lookup only |
| [2603.29979](https://arxiv.org/abs/2603.29979) | Structural Feature Engineering for Generative Engine Optimization: How Content Structure Shapes Citation Behavior | Abstract lookup only |
| [2603.10913](https://arxiv.org/abs/2603.10913) | LLM2Vec-Gen: Generative Embeddings from Large Language Models | Abstract lookup only |
| [2106.06599](https://arxiv.org/abs/2106.06599) | The value of travel speed | Abstract lookup only |
| [2306.17322](https://arxiv.org/abs/2306.17322) | Citations as Queries: Source Attribution Using Language Models as Rerankers | Abstract lookup only |

## Reproducibility

The bibliography JSON retains record URLs and theme tags. Evidence JSON files retain query URLs, returned candidates and failures. The report contains targeted full-text section locators. A future update should preserve this dated snapshot, rerun the recorded URLs, add new searches for unresolved themes, and record any revised publication/version matches. Do not interpret a retrieved abstract or successful API call as validation of a paper's results.

