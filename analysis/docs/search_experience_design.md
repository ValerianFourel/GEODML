# A captured-evidence search experience

This pilot adds a new answer and judge contract beside the ACL ARR experiment.
It does not replace the frozen Natural, Ablated, or Shuffled measurements.

The later agentic extension has its own
[retrieval protocol](agentic_search_retrieval_protocol.md). That protocol
defines `Parallel-Expansion-v1`, `Reactive-Snippet-Loop-v1`, local
cross-encoder compaction, exact execution bounds, retrieval freezing, counting,
and go or no-go gates. It does not change the captured-evidence pilot described
here.

## What public documentation supports

OpenAI describes ChatGPT Search as answering questions with web sources and
citations. It may rewrite a request into targeted queries and issue further
queries after inspecting results. These are publicly described behaviors, not
a specification of its retrieval or ranking implementation.
Source: [Searching the web with ChatGPT](https://help.openai.com/en/articles/9237897-chatgpt-search),
consulted 2026-09-07.

GEODML approximates an observable part of that experience: a user request,
recorded search evidence, a cited answer and a separate assessment of support.
Its prompts, fixed evidence budget, ranking task, judge rubric and scheduling
are engineering choices. They are not OpenAI's internal prompts or algorithms.
The pilot does not implement personalization, conversational memory, location,
adaptive retrieval, browser interaction or proprietary source ranking. The
agentic extension approximates public query-expansion and iterative-search
behavior. Neither protocol claims to reproduce a provider's private internals.

## Existing path and its limits

`collect_readiness_search_snapshot.py` searches distinct metadata keywords.
The complete generated question enters `render_primary_prompt` later. Several
different user intentions can therefore share one captured search.

`prepare_acl_arr_pilot_inputs.py` reads previously cached HTML. The old cache
selector matches engine and pool size, not the precise search capture. Search
snippets and extracted page text are different representations. A hash proves
which bytes were used, not that the bytes were fetched at search time.

The legacy answer contract requires citations and an ordered citation index.
The legacy judge requires a nonempty document-contribution ranking. Neither
contract cleanly represents a justified answer with no supporting source.
Citation syntax and resolvable IDs do not establish factual support.

## Chosen boundary

An offline adapter joins recorded captures to existing frozen documents and
assignments. It must reject mismatched evidence rather than rebuild a treatment.
The complete generated question remains unchanged. Acquisition records retain
duplicates, errors and unavailable metadata even when only usable frozen page
text enters the model request.

Ranking and answering remain siblings. Ranking uses the legacy request and
validator. The new answer contract contains claims, stable citation IDs and
expressed uncertainty. The displayed answer derives from those claims, so a
second prose field cannot silently diverge from the evaluated content.

The judge receives the user request, answer and permitted source text. It never
receives explicit generator identity, condition labels or measured rankings.
Its order is assigned independently. Retaining existing stable document IDs
does not hide every possible source-order cue; this is explicit-label blinding.

Exact quote and offset checks establish that cited evidence exists in the
frozen text. Whether that evidence supports a claim remains a fallible judge
assessment. Source contribution means observable textual support, not access
to the generator's internal reasoning.

## Alternatives and review

Two independent architecture sketches compared a captured-bundle adapter with
a broader experiment compiler. The independent reviewer selected the adapter,
23/25 versus 19/25 across invariant safety, interface depth, focus, evidence
fidelity and testability. The implementation adopts the compiler sketch's
claim-evidence checks and shared request executor, but not its broader framework.
Review agents used inherited configurations; model diversity was not verified.

The choice follows Poteto's domain-first and small-verifiable-unit principles.
Separate outputs make protocol differences inspectable. Existing transports
and bounded scheduling remain reusable without changing legacy validators.

## Scientific boundary

The answer instructions, uncertainty contract and judge rubric constitute a
new protocol. Pilot outputs are not eligible for the existing production
analysis. A full-request search that changes the evidence pool needs a separate
search-policy experiment and newly frozen assignments. The agentic protocol is
that separate proposed experiment. Target ablation is applied before
compaction. Natural or Shuffled presentation order is applied after top-K
selection so compaction cannot erase the order treatment. A reactive path can
still diverge after it observes a treatment. This divergence must be frozen and
analyzed as part of the method, not hidden by reusing an incompatible evidence
pool.

Human calibration packets are inputs for independent review, not calibration
results. A model must not supply its own sole primary evaluation. Human
agreement, answer quality, all-model compatibility and GPU throughput remain
unmeasured until the real pilot runs.

## Available evidence

No actual audited prompt files, captured SERPs, frozen page records or model
snapshots were available in this checkout during implementation. Historical
terminal logs are not current cluster state. Synthetic tests exercise contracts
and restart handling only. They are not search hits or measured LLM outputs.
