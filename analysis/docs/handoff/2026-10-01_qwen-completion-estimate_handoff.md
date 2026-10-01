# Qwen completion estimate, 1 October 2026

Valerian supplied a job-name-filtered snapshot since 25 September: 811 jobs,
499 COMPLETED, 20 FAILED, 35 RUNNING, 257 PENDING, all other displayed states zero.
This is pasted accounting/live-queue evidence, not a fresh independent query.

Conditional on one allocation per bout and complete accounting coverage of the
893-bout plan, 82 bouts are absent from the snapshot and likely still unsent.
There are 374 first-pass bouts remaining: 35 running, 257 pending, 82 unsent.
At sustained concurrency 35 and historical completed-bout durations 2.3–4.7 h,
374 * duration / 35 gives about 25–50 h, midpoint scenario 37 h. Already-submitted
work alone gives about 19–39 h. Running jobs are conservatively costed as full
bouts; their elapsed progress is unknown. This is a capacity scenario, not a
measured current completion rate or guaranteed finish date.

Approximate first-pass finish is 2–3 October if capacity and sender replenishment
hold. The 20 failed allocations and any unfinished cells in successful jobs
still require artifact/ledger reconciliation; the estimate does not promise
scientific completion or authorize retry allocations. No runtime edits, cluster
commands, submissions or cancellations occurred. Arithmetic checked locally.
