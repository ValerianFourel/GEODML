# Nemotron v4 reviewed by three maximum-effort Astra agents

Valerian supplied the private HF export and explicitly requested agents to judge
Nemotron against v4's intent, as for Gemma. Downloaded the exact revision
`99d32ee66bc9fd35c4d2b8f099c996c13958babf`, path
`reviews/nemotron-si-v4-20cells-20261004-job5177346/assessment-nemotron-wwd7dud2/all-inference.jsonl`.
SHA-256 matches `9d940509b83c40e19959e1ab122ca96f523746fb61719caa20e6201b60b50170`.
Evidence and review artifacts are in
`/Users/valerianfourel/Downloads/nemotron-v4-review-20261004`.

Three gpt-6-astra agents used requested reasoning effort max. Two reviewed ten
cells each, with 56 and 51 complete supplied source records. They froze initial
map feasibility/source expectations before viewing Nemotron or Gemma outputs.
Blind hashes remain `18d3fe896a763561df7bfdc42dd656784943825ad045b41e5fac747f01dffc21`
and `33660ffb9020b275e6128e0290dc30482d6daab9467b96c90bd4933613ba44c7`.
The third audited all twenty maps, all seven executed source calls, protocol,
validators and timing. All agents completed. This is exploratory diagnosis,
not formal independent Astra/Sol certification, human gold or a full holdout.

All twenty frozen cell objects and 147 input records equal matched Gemma first
pass records. The random twenty-cell sample reproduces seed 20261004 from the
forty-cell development pool. The model/tokenizer settings change as declared;
the v4 protocol, coordinator and request client are byte-identical between
scientific commits caf8a7d3fbe69736789767efcc4551d0e0347c11 and
f57435f425bd8bbffc8263f1f2af2da3c078e686. Source tasks include each model's own
map, so this is an end-to-end comparison, not a fixed-map source-judge test.

Nemotron returns eighteen unjustified unusable maps, one duplicate-span failure
and one ready map. All twenty original answers are mappable under the reviewers'
blind expectations. Ninety-three sources are blocked by unusable maps, seven by
the failed map; seven source calls return uncertain/null. There are zero numeric
grades and zero complete cells. Twenty fulfilment tasks are not requested.
All fourteen source findings are partial; seven procedural findings lack source
support, and four tool-choice notes invent absent named tools. ProProfs describes
SolarWinds, while the model attributes Zendesk/Freshdesk/DevRev to it. Generic,
descriptive and procedural answer content is repeatedly treated as non-substantive
or ineligible because it does not fulfil the request. Partial support is wrongly
treated as inherently ungradable.

Matched Gemma has twenty ready maps, 99 numeric grades, one source failure and
seven global-absence-only exclusions, with eighteen complete numeric cell vectors.
These counts are not semantic accuracy. Gemma still omits substantive content,
including the planning sentence that Nemotron preserves in its sole ready map.
Both models fail the intended semantic standard; the current Nemotron setup
does not provide a useful speed/cost replacement.

All 27 request fingerprints and 29 validation-attempt prompt hashes reproduce.
Validators reproduce 26 accepted parses and three rejected attempts. All attempts
finish with stop, maximum 2,811 completion tokens versus the 4,096 cap. No deadline
admission stop. This evidence does not support blaming token truncation or GPU
failure. Parent verified all 20 cells / 107 sources, all source content hashes,
254 compared outcome records and 147 non-null raw-output hashes. Seventeen artifact
hashes are preserved in the validation receipt. Whitespace checks passed; no
executable code changed and no inference tests or cluster jobs were run.

Saved accounting: job5177346 FAILED after 946 seconds, one exclusive node/four
GPUs, 0.262778 node-hours / 1.051111 GPU-hours. Warm SI phase 187.018 seconds,
0.051950 node-hours / 0.207798 GPU-hours. Serving/trial wrapper 610.089 seconds
starts after preparation, while execution.started_at precedes preparation;
do not combine those into an inferred end timestamp. Slurm timestamps have no
timezone; their offset is consistent with HoreKa UTC+2. With no completed cell,
valid throughput and full-corpus runtime cannot be estimated from this run.
These are returned records, not a live cluster query.

Report: `analysis/docs/si-v4-nemotron-semantic-review-20261004.md`.
Receipt: `analysis/docs/si-v4-nemotron-semantic-review-20261004.validation.json`.
Original evidence and grades remain unchanged. No repair, model rerun, allocation,
or production launch occurred. Next work, only if requested, is a bounded repair
of mapping/support behavior followed by another matched diagnostic, preserving
the abstention and validation contracts rather than forcing scores.
