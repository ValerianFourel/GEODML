# V4 saved judgments for immediate content review

## User intent

Valerian wants to judge the new v4 workflow on the actual questions, answers,
claim maps, source-support decisions and grades now. Defer inference repairs.
He requested a quick command to print all saved inference and show processing
time and node-hours. No new allocation or model substitution was requested for
this inspection.

## Evidence and limits

The latest pasted r2 queue has all ten workloads terminal. The attempted
interactive continuation stopped with `evaluation queue already exhausted; no
allocation needed`. This proves neither semantic acceptance nor successful
execution of all individual calls. No live cluster access occurred in this turn.

The pasted summaries are preserved in
`analysis/docs/si-v4-r2-pasted-results-20261004.json`. The assessment in
`analysis/docs/si-v4-r2-results-assessment-20261004.md` distinguishes observed
counts, derived bounds and missing semantic evidence. Main completion is
198/207 eligible sources. The repeated fixed-map failure is inherited by design.
The source finding and map overlap errors remain unfixed, per user priority.

Distinct warm workloads total 42.46024 minutes, 0.707670655 node-hours and
2.830682619 GPU-hours. These figures exclude allocation startup, idle time and
cleanup. Actual Slurm occupancy remains to be returned. The optional projection
in the collector reports cost per 1,000 completed cells unless an explicit
remaining count is provided; this is not a corpus-size assumption or permission
to launch work.

## Published saved-output command

`analysis/docs/horeka-v4-r2-assess-all.sh` was committed locally as `5be318c` and
published on `codex/v4-interactive-20261003` as
`9a50de8dcef8ea3137e320ddb844962fd0226e95`. The remote SHA was verified using
`git ls-remote`.

The script uses the already-installed operator checkout
`1b34bebaf0007ed1010b4a4065a880cb4e10d73c`. Scientific inference remains pinned
to `caf8a7d3fbe69736789767efcc4551d0e0347c11`.

The collector reads the saved development run under
`$W/reviews/si-v4-r2-cycle-20261002/development`, checks its registered
configuration, reads task inputs/results through read-only SQLite, and prints
every queue workload. It creates a unique `assessment-all-XXXXXXXX` directory
containing `all-inference.jsonl`, `console.txt`, an assessment JSON/HTML and a full
evidence archive. It calls the existing saved-output evaluator and Slurm
accounting helpers. It requests no allocation and invokes no model. Missing
independent reference reviews must not be interpreted as zero semantic accuracy.

Bash and Zsh syntax, embedded Python parsing and Git whitespace checks passed.
Cluster execution is not yet observed.

## Next step

Return the per-question dump or full evidence archive, then assess claim-map
fidelity, source entailment, grade calibration and rationale consistency here.
Full frozen questions, answers and sources are available locally under
`/Users/valerianfourel/Downloads/si-v4-r2-release-20261002T121912Z`.
Do not substitute the older job 5175229 diagnostic for these r2 judgments.
Do not change thresholds, reopen the completed queue, repair inference or launch
fresh evaluation while doing this review.
