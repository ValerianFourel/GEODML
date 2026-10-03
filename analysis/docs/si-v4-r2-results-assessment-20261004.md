# SI-v4 r2 assessment against the agreed standards

Valerian's immediate priority is to review the actual questions and v4 judgments
here, with inference repairs deferred. This report supplies quantitative context;
it does not replace that content review. The new per-question outputs have not
yet been returned. The saved-output collector is
`analysis/docs/horeka-v4-r2-assess-all.sh`.

The returned development results do not meet the agreed acceptance standard.
Reported source completion fails its threshold. Fixed-map repeat coverage makes
both agreement thresholds unreachable. Reported warm compute cost exceeds its
ceiling. The constructed run also excludes more cases from grading than the
frozen control design permits. These conclusions do not require another
inference run. Semantic correctness of retained grades remains unestablished.

This is an assessment of Valerian's pasted summaries, preserved in
[the evidence file](si-v4-r2-pasted-results-20261004.json). It is not a substitute
for replaying validation against the complete remote artifacts. The policy is
[si_v4_absolute_evaluation.json](../config/si_v4_absolute_evaluation.json), and
the denominator/cost/repeat rules were checked in
`analysis/scripts/compare_source_importance_runs.py`.

## Findings

| Criterion | Required | Evidence | Assessment |
|---|---|---|---|
| Eligible-source completion | At least 99% | Main pass scored 198 of 207 model-declared eligible sources, 95.65% | Fails on reported counts |
| Fixed-map exact agreement, all planned comparisons | At least 95% | Each execution scored 116 of 124 sources; agreement can be at most 93.55% | Cannot pass, even if every available grade agrees |
| Fixed-map agreement within one grade, all planned comparisons | At least 99% | Same 93.55% upper bound | Cannot pass |
| Warm GPU-hours per completed eligible cell | At most three times matched SI-v3 cost bridge | Approximately 7.94 times from reported timing/completion counts | Fails on reported figures, subject to normal artifact/configuration matching |
| Constructed controls | All 48 assessed according to frozen expectations; zero unsupported grade 4/5 | Only two local frozen expectations permit a null grade, but the run reports four global-absence exclusions | At least two cases requiring grades were excluded, assuming the matching frozen cohort; control acceptance cannot pass |
| Map fidelity and essential content | At least 95% faithful; zero essential omissions or reversals | No independent map assessments in the supplied results | Unestablished |
| Claim-witness support | At least 95% sufficient among resolved assessments; at most 5% unresolved | No complete findings, sources or independent support assessments supplied | Unestablished |
| Agreement with independently frozen grade ranges | At least 90% | No per-source grades and independent grade ranges supplied | Unestablished |
| Nonempty top-group agreement | At least 95% | Summary counts contain no grade vectors | Unestablished |
| Retained structural validity / input integrity | 100% retained validity; zero unexplained mismatches | Rejected outputs were retained as failures; successful raw outputs and input joins were not supplied | Requires artifact revalidation; rejected calls alone do not prove invalid retained grades |
| List-order and role controls | Pass the frozen supplementary checks | Five maps and four source grades completed, but their roles/grades were not shown | Execution completed; quality pass unestablished |
| Finite resource budget | At most 16 GPU-hours development and 32 total | Warm timing supplied, actual Slurm allocation accounting absent | Unestablished |
| Fresh evaluation | Passing development, then 120 paired fresh/stress cells | This is the 40-cell development queue | Not established by this run |

The 99% source-completion threshold is not a cell-completion threshold. The
main pass's separate cell count is 36/40, or 90%. Of 214 supplied sources,
198 were scored, seven were declared global-absence-only, seven were blocked
by a failed map, one failed source validation and one remained uncertain.
Even excluding the seven global-absence cases, nine eligible sources lack
grades. At least 205 of 207 would need grades to reach 99%.

The fixed-map cohort is the predetermined 24 cells with 124 source pairs.
Three executions create three pairwise comparisons, or 372 planned source
comparisons. Each pair can contain at most 116 jointly scored sources, so at
most 348/372 comparisons can agree. This is a ceiling, not measured agreement.
Identical counts across executions do not establish identical grades. Agreement
conditional on both results being scored is a different quantity and cannot
replace the agreed all-attempt denominator.

The source-blind map failure is copied into the fixed-map executions by design.
The repeated `a3w6` error is not evidence of four independent fresh map failures.
Preserving failed maps in these runs exposes the cost of depending on that map
instead of silently regenerating it and changing the experiment.

## Timing and cost

| Workload | Complete cells | Scored / declared eligible sources | Warm minutes | Warm GPU-hours |
|---|---:|---:|---:|---:|
| Main r2 | 36/40 | 198/207 | 12.0423 | 0.802822 |
| Constructed | 44/48 | 44/44 | 0.7840 | 0.052264 |
| List order | 5/5 | No source grading | 0.5165 | 0.034435 |
| Role controls | 4/4 | 4/4 | 0.1013 | 0.006756 |
| Matched SI-v3 cost bridge | 40/40 | 214/214 | 1.6857 | 0.112380 |
| End-to-end repeat 2 | 24/24 | 124/124 | 6.9799 | 0.465327 |
| End-to-end repeat 3 | 23/24 | 123/124 | 7.8195 | 0.521300 |
| Fixed-map repeat 1 | 22/24 | 116/124 | 4.1470 | 0.276466 |
| Fixed-map repeat 2 | 22/24 | 116/124 | 4.4370 | 0.295799 |
| Fixed-map repeat 3 | 22/24 | 116/124 | 3.9470 | 0.263133 |

The reported normalized cost is
`(0.8028217954 / 36) / (0.1123798545 / 40) = 7.93758`.
This is warm GPU cost per completed cell, not summed overlapping request
durations and not model startup or billed allocation cost. The formal evaluator
must still confirm matched code/settings, original input identities, token
budgets and completed eligible-cell denominators. The local frozen development
plan uses a matched 4096-token SI-v3 bridge. It is a cost comparison, not a
requirement that v4 beat v3's grade quality.

The ten separate workload timings sum to 2.83068 warm GPU-hours. Reused queue
entries were counted once. This does not establish total resource use or budget
compliance; allocation startup, idle time and cleanup are additional.

## What the errors establish

`exclusion overlaps claim or exclusion at word IDs: a3w6` means the map gave the
same original-answer word to a claim and an exclusion, or to overlapping
exclusions. It blocked seven source judgments. The original text at a3w6 and
the raw rejected map are not present in the pasted summaries. It cannot be
assumed to be the earlier bare list marker or automatically repaired.

`full finding must cover the mapped claim` means a finding marked `full`
selected fewer answer words than its mapped claim contains. The validator
compares word coverage sets; it already permits different span segmentation
with identical coverage. This is a support-output contract violation. It does
not, by itself, prove that the complete source fails to support the claim,
that a retained grade is wrong, or that the validator should be relaxed.
The exact excluded words, source content and attempted outputs are needed to
distinguish incomplete output selection from substantive support overclaiming.

The 48-case frozen constructed design has two null acceptable-grade ranges,
both global-absence-only. Four observed exclusions therefore cannot all match
the design. The raw case identities will locate the extra exclusions and allow
checking every remaining control grade, including unsupported high scores.
"No inference failures" for these controls only means no calls failed the
execution checks; it does not establish correct judgments.

All summaries report zero `cells_provenance_verified`. The local frozen real
development cells carry `provenance_unresolved`; constructed cells carry
`constructed`. These counters are not evidence of newly corrupted inputs and
must not be turned into verified provenance. Hash-based input matching and
original generator provenance are separate checks.

## Disposition

Keep r2 in development. Do not launch fresh evaluation or production on these
results. The supplied evidence is enough to reject readiness, but not enough
to estimate semantic accuracy among the 198 retained first-pass grades.

The next analysis uses saved artifacts: the existing `run_si_v4_cycle.py assess`
command revalidates raw outputs, computes actual repeat agreement, and checks
constructed/list-order/role expectations. Independent source-blind map reviews,
blind grade ranges, and subsequent full-source support reviews remain required
for semantic acceptance. Missing references must not be interpreted as measured
zero semantic accuracy. Preserve complete failure denominators and unchanged
thresholds. No new model inference, model substitution, or code repair was
performed for this assessment.
