# Actual v4 r2 evidence reviewed against its goals

## Request and evidence

Valerian authorized GPT-6 Astra subagents at maximum effort to judge the new
v4 workflow against its goals. Inference repairs remain deferred. He uploaded
the completed saved-output export and supplied its hash and pinned HF URL.

Dataset: `ValerianFourel/geodml-experiment-v2-paper-private`.
Revision: `bf9872076b185ead4c3a266319adde7ab449c328`.
Prefix: `reviews/si-v4-r2-cycle-20261002/development/assessment-all-82tGtA3n`.
Dump SHA-256: `af1d4cf0016b92cce9a1b887636d944f19916fd25456271d30eddc211d83e5d0`.
Scientific commit: `caf8a7d3fbe69736789767efcc4551d0e0347c11`.

Downloaded four files with local cached HF authentication and verified the
user's dump hash. Safely extracted the evidence locally. All 15 frozen data
files match their archived manifests, and decoded records equal the local
original release. Compressed bytes differ from the original release; their
manifest hashes reflect this, without a content discrepancy. All 294 first-pass
task records match the returned frozen input inventory.

## Review

Report: `analysis/docs/si-v4-r2-semantic-review-20261004.md`.
Evidence/reviewer directory:
`/Users/valerianfourel/Downloads/si-v4-r2-review-20261004`.

Three `gpt-6-astra` agents used requested reasoning effort `max`. Two covered
20 real cells each, 110 and 104 sources. They saved initial grade ranges before
opening candidate maps/findings/grades. The third reviewed 48 constructed rows,
five list rotations and four fixed-map role controls, then independently
cross-checked the wallet/pharmacy/loyalty omissions. Parent independently
checked headline examples and recomputed repeat metrics.

All reviewer files are complete. Final joins verified 40 cells, 214 sources,
198 retained numeric grades and 185 retained findings against original output.
Both blind files retained their frozen hashes. The receipt is
`analysis/docs/si-v4-r2-semantic-review-20261004.validation.json` and includes
16 artifact hashes. Documentation and evidence review changed no executable
behavior; code tests were unnecessary. Git whitespace checks passed.

This is a development diagnostic, not the formally frozen dual-model Astra/Sol
acceptance assessment or human gold. Each natural source has one initial Astra
reference. All sources for a cell were visible in its input packet. Initial
range comparison is 94/198 scored outcomes in range, 104 below the reviewer
ranges, 16 not scored. Do not present that fraction as measured scientific
accuracy or silently rewrite the preserved `assessment.json` reference gates.

The current results fail the intended construct even apart from inference
errors. Strong evidence:

- Wallet `0a04c9d1ba5852f0171a`: substantive wallet descriptions excluded;
  MetaMask/Trust Wallet sources matching them score zero.
- Pharmacy `166ffa4bb37c4f872e9a`: software features excluded; Software Advice
  supports five of six excluded features but scores zero. Support is partial,
  not complete. A defective map requires `map_issue`, not a guessed regrade.
- Loyalty `326525bc3e7b07f069b7`: tiered-membership/referral recommendation
  excluded; Popupsmart repeats those ideas but scores zero.
- Social API `ebd04d863dcd705a9689`: CRM record scores zero for matching
  narration; unrelated social-scheduling record receives full support for it.
- Recruitment `ec1dff51454a2d283c6e`: a source ending in an ellipsis is claimed
  to explicitly name underrepresented populations that are absent from it.

Constructed expectations match 37/48 logical rows, including two correct pure
absence N/A outcomes. Among 44 scored rows, 35 match and nine miss their frozen
ranges. Two mixed-absence rows wrongly become N/A. The 48 rows reuse 13 maps
and 28 dependencies. Four of five list factors change roles when reordered;
the major-role source returns grade 2 despite full support for a major step.
All four narration controls under-credit complete support with 4 instead of 5.
Simple entity/contradiction controls include valid successes. Missing independent
references are not zero semantic accuracy.

## Timing and stability

Saved Slurm accounting is complete for jobs 5175818 and 5177064, with elapsed
3,614 and 1,478 seconds on one four-GPU node each. Total occupancy: 1.414444
node-hours, 5.657778 GPU-hours. Warm processing: 42.46024 minutes, 0.707671
node-hours, 2.830683 GPU-hours. Development budget is not exceeded. No live
cluster state was inferred or queried this turn.

Recomputed fixed-map exact/within-one/top-group agreement: 330/372, 346/372,
43/51. End-to-end: 345/372, 355/372, 44/52. All miss the agreed all-attempt or
top-group thresholds. Eligible completion is 198/207. Warm normalized cost is
7.93758 times the matched v3 bridge, above the 3-times ceiling. V3 is cost-only
under the current absolute policy; historical superiority requirements do not
apply to this r2 cycle.

## Disposition

Keep v4 r2 in development. Its architecture is implemented, but the current
judge does not consistently preserve the answer or apply its own support and
importance rules. Repairing only rejected calls cannot establish acceptance.
No cluster allocation, new scientific inference, model substitution, result
rewrite or inference repair was performed. The requested result is the
evidence-based judgment and preserved per-question review.
