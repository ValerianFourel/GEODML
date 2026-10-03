# SI-v4 r2 review of actual questions and judgments

The current v4 r2 judgments do not meet the intended semantic standard, even
when inference failures are set aside. The two-stage design is implemented,
but successful calls can discard substantive answer content, give zero to a
source that directly supports it, accept a claim against the wrong source,
and assign grades inconsistent with their own support findings and fixed roles.
Retain this as a development result. No inference repair or new run was made
for this review.

## Evidence and review scope

The uploaded evidence is pinned to private Hugging Face dataset
`ValerianFourel/geodml-experiment-v2-paper-private`, revision
`bf9872076b185ead4c3a266319adde7ab449c328`, under
`reviews/si-v4-r2-cycle-20261002/development/assessment-all-82tGtA3n`.
The downloaded `all-inference.jsonl` matches Valerian's SHA-256 exactly:
`af1d4cf0016b92cce9a1b887636d944f19916fd25456271d30eddc211d83e5d0`.
Scientific inference used commit `caf8a7d3fbe69736789767efcc4551d0e0347c11`.

Three GPT-6 Astra subagents were requested at maximum reasoning effort. Two
reviewers covered disjoint halves of the 40 real cells, 110 and 104 sources,
and saved their initial grade ranges before opening the candidate maps,
findings and grades. The third reviewed all 48 constructed rows, five list
rotations and four fixed-map role controls. The parent checked the decisive
textual examples and independently recomputed the repeat metrics. Final record
checks confirm coverage of all 214 sources, 198 retained numeric grades and
185 retained findings, with no omitted or duplicated cell/source identities.
Both initial reference files remain byte-identical to their pre-comparison
freezes. Artifact hashes and coverage checks are preserved in
[the validation receipt](si-v4-r2-semantic-review-20261004.validation.json).

The 15 returned frozen data files match their archived manifests. Their decoded
JSON records match the original local release, although compressed bytes and
the corresponding manifest hashes differ. All 294 first-pass task input records
match the returned frozen task inventory. These checks do not establish the
original generator-to-webpage provenance. Some supplied records combine several
snippets; the review judges the complete supplied record without fetching or
substituting a webpage.

## What v4 was intended to achieve

V4 separates a source-blind map of the completed answer from each source's
support judgment. It measures support and importance within that answer,
including an imperfect or off-target answer. It does not measure causal
reliance, unique contribution, ideal relevance or request fulfilment.

The map must preserve substantive facts, recommendations, entities and
qualifications. Stage B must credit supported portions without inventing
support for the rest, and must report `map_issue` when a substantive map defect
prevents valid scoring. The unchanged scale runs from no substantive support
at 0 through substantial explanation at 3 and a central conclusion or essential
step at 4, to most essential answer content at 5.

R2 specifically strengthened instructions about list position, qualifications,
background support and consistency with fixed map roles. The current absolute
policy is [si_v4_absolute_evaluation.json](../config/si_v4_absolute_evaluation.json).
Its v3 comparison is cost-only. The historical guide's extra v3 superiority
condition is not a requirement of this authorized r2 cycle.

## Decisive failures in real questions

### Pharmacy features disappear before source scoring

Cell `166ffa4bb37c4f872e9a` asks which factors matter when implementing pharmacy
management software. The answer first discusses system components, then says:

> consider features like inventory management, regulatory compliance,
> dispensing, advance expiry alerts, automated sales operations, and purchase
> tracking to ensure seamless operations and efficiency.

Stage A excludes that substantive recommendation as `non_substantive`. The
Software Advice record, dependency `si-v4-dependency-ef9edd9c2e033ae5700c9355`,
explicitly describes inventory management, regulatory compliance, advance
expiry alerts, automated sales operations and purchase tracking. Stage B gives
it 0 because it does not support the earlier system-components claim.

The zero is incompatible with the complete answer and complete source. The
proper response to the supplied defective map is `map_issue`; changing the
grade while silently retaining that map would not satisfy v4 either.

### Wallet descriptions are excluded because they do not fulfil the request

Cell `0a04c9d1ba5852f0171a` asks about setting up and testing a blockchain wallet.
The completed answer includes explicit MetaMask currency, ownership and user
count claims, plus a Trust Wallet ownership/Web3 description. Stage A keeps
only the opening recommendation and excludes the other four sentences.

The MetaMask source, dependency `si-v4-dependency-3695b8db0a9a712df839b848`,
directly supports BTC/ETH/SOL, control over data/assets and the 100-million-user
description. The Trust Wallet source, dependency
`si-v4-dependency-60ce3fb071e8f43d8dd78590`, repeats its ownership and Web3
description. Both receive 0 because they do not establish the complete
pending-transaction-testing recommendation. Lack of support for that
recommendation does not erase support for the answer's other factual content.

### The CRM claim is supported by the wrong source

Cell `ebd04d863dcd705a9689` narrates what the supplied snippets contain. Its
mapped c7 is: "Snippet S2 discusses CRM and data integration broadly."
The CRM record, dependency `si-v4-dependency-97ee73dd05e5ad9689cca5ea`, describes
CRM and compiling data from websites, telephone, email, chat and social media.
V4 calls c7 unsupported and gives the record 0.

The social-scheduling record, dependency
`si-v4-dependency-73fb2a0976a4d6e75e4ac69b`, describes Buffer/Hootsuite and
connecting tools to an X account. V4 instead marks c7 fully supported by this
record, treating broad account connectivity as support for the CRM narrative.
This is a wrong-source/full-support error as well as a missed direct match.

### A missing qualification is filled in from the answer

Cell `ec1dff51454a2d283c6e` recommends recruitment strategies. The mdgroup source,
dependency `si-v4-dependency-31433f601e86e12279ce4cfa`, ends:

> partnerships with community organisations, can help boost recruitment from ...

V4's finding says that this source "explicitly mentions" recruitment from
underrepresented populations and marks the complete qualified recommendation
as fully supported. The provided title/text never names that population.
The source supports outreach and community partnerships, but it does not
establish the missing population qualification. The source must be judged as
supplied, without filling in the truncated webpage text.

## Controls locate the same failures

| V4 goal | Actual saved behavior | Assessment |
|---|---|---|
| Preserve factual content alongside global absence | Mixed-absence controls discard "Acme/Birch exports CSV offline" because it does not explain the requested migration | Two false N/A outcomes |
| Assign importance from meaning, not list position | Four of five identical proctoring factors change between central and major when moved first | Role invariance fails on this list |
| Respect fixed roles and ordinal anchors | A fully supported major migration procedure receives grade 2 | Under-credit relative to the substantial-explanation anchor |
| Give 5 when the source supports most essential content | Four one-sentence narration controls are fully supported but receive 4 | Grade-5 calibration fails |
| Distinguish partial support and wrong entities | Qualification and entity-list controls retain unsupported portions; all ten zero-target controls receive 0 | Narrow control successes |

Original constructed expectations match in 37/48 rows. Among the 44 scored
rows, 35 grades fall in their frozen ranges and nine do not. Two pure-absence
cases correctly remain N/A; two mixed-absence cases incorrectly do so. These
rows reuse 13 maps and 28 source dependencies, so they are not 48 independent
model decisions. Four advice-control rows have in-range grades despite a
substantive factual exclusion, illustrating why grade agreement alone is
insufficient. The four role controls return three in-range outcomes.

There are also sound natural-case distinctions, including rejecting a source
about a different named product and acknowledging that an example reward
requires a qualifying purchase rather than every order. The failures above
do not imply that every saved judgment is wrong.

## Stability, completion and cost

The following values were reproduced from the saved records. Agreement retains
missing attempts in its planned denominator. Top-group agreement uses complete
comparisons with a nonempty positive top group.

| Measure | Observed | Required |
|---|---:|---:|
| Model-declared eligible-source completion | 198/207 = 95.65% | At least 99% |
| Fixed-map exact repeat agreement | 330/372 = 88.71% | At least 95% |
| Fixed-map agreement within one grade | 346/372 = 93.01% | At least 99% |
| Fixed-map positive top-group agreement | 43/51 = 84.31% | At least 95% |
| End-to-end exact repeat agreement | 345/372 = 92.74% | At least 95% |
| End-to-end agreement within one grade | 355/372 = 95.43% | At least 99% |
| End-to-end positive top-group agreement | 44/52 = 84.62% | At least 95% |
| Warm GPU cost per completed cell against matched v3 | 7.94 times | At most 3 times |

Even conditional on both fixed-map results being scored, exact agreement is
330/348 = 94.83%. End-to-end conditional exact agreement is 345/355 = 97.18%.
These conditional figures do not replace the agreed all-attempt measures.
Structural validity passes for all 198 retained source outputs. It does not
test whether a valid span selection preserves meaning or whether its source
actually entails the finding.

Saved Slurm accounting covers jobs 5175818 and 5177064: 3,614 and 1,478 seconds
on one four-GPU node each. Total occupancy is 5,092 seconds, 84.87 minutes,
1.414 node-hours or 5.658 GPU-hours. Distinct warm workloads total 42.46 minutes,
0.708 node-hours or 2.831 GPU-hours. The development resource budget was not
exceeded. These are returned accounting records, not a claim about live jobs.

## Interpretation and review artifacts

The initial independent Astra ranges match 94/198 scored outcomes; the 104
out-of-range grades are all below those reviewers' ranges. This is exploratory
disagreement evidence, not a measured 47.5% scientific accuracy result. There
is one Astra reviewer per real source, all sources for a cell were visible in
the review packet, and these are development cases. The frozen formal protocol
requires independently preserved Astra/Sol reference evidence and separate
map/witness assessment. That acceptance process has not been completed here.
No existing `assessment.json` reference flags were relabeled or overwritten.

The direct omissions, false zeros and unsupported full findings are sufficient
to reject readiness without treating reviewer ranges as ground truth or
interpreting absent reference labels as zero semantic quality. Repairing only
the known structural inference errors would leave these successful-call errors.
The architecture's usefulness and the current judge's reliability are separate
questions. This review supports retaining the architecture for development;
it does not support trusting the current scores for the paper.

The distinction matters for Experiment V2. Penalizing a source because the
answer fails the request mixes request fulfilment into source importance.
That could distort comparisons along the observational readiness axis when
answers differ in their ability to fulfil increasingly specific requests.
This review establishes the scoring defect in concrete cases; it does not
estimate the resulting association or claim a causal axis effect.

The complete downloaded evidence and per-question reviewer records are in
`/Users/valerianfourel/Downloads/si-v4-r2-review-20261004/`:

- `goals-controls-review.md` and `.json`: every constructed/control case.
- `real-a-blind.jsonl` and `real-b-blind.jsonl`: initial ranges saved before
  candidate judgments were opened.
- `real-a-review.md` / `.jsonl` and `real-b-review.md` / `.jsonl`: case-by-case
  comparison of the 40 real questions.
- `real-omission-crosscheck.md`: additional independent check of key real cases.
- `independent-metric-check.json`, `frozen-input-check.json` and
  `diagnostic-blind-range-comparison.json`: recomputed accounting and joins.
- `download-receipt.json`: exact uploaded revision and verified dump hash.
