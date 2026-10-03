# Gemma v4 full-corpus runtime estimate

Valerian asked how long Gemma would take to judge all cells with the current v4
workflow. This is an estimate only; no allocation or inference was requested or
started. Read the last three indexed handoffs and used the actual returned r2
timings already verified locally.

Read current HF repository metadata at revision
`bf9872076b185ead4c3a266319adde7ab449c328`. The saved
`coordination/progress.json` records 624,192 tasks across the registered Qwen and
Llama inventory. This progress file's completion states are historical, not live
cluster status. The paper's checked final prompt population has 26,009 strings,
giving 312,108 cells per generator and 624,216 across both, versus 26,008 prompts
in the registered inventory. Round the forecast population to approximately
624,000 cells. Neither target is the current valid/eligible-answer census.

Only the three full map-plus-source candidate runs are used for throughput:

| Run | Warm node-hours | Complete cells | Seconds per completed cell |
|---|---:|---:|---:|
| candidate-e2e-1 | 0.200705449 | 36 | 20.0705 |
| candidate-e2e-2 | 0.116331726 | 24 | 17.4498 |
| candidate-e2e-3 | 0.130324957 | 23 | 20.3987 |

These imply 176-206 completed cells per warm four-A100 node-hour. Do not divide
the whole validation cycle's time by 40: it includes controls, fixed-map repeats
and a v3 bridge. Warm cost for 624,192 cells is about 3,026-3,537 node-hours.

For the current one-hour allocation policy, allow measured historical startup
8.9-10.9 minutes and five minutes for drain/cleanup. Usable warm time is therefore
44.1-46.1 minutes per allocation. Rounded projection for both generators:

- 3,938-4,813 allocated node-hours, or 15,752-19,252 GPU-hours.
- One node continuously: 164-201 days.
- Five simultaneous nodes continuously: 32.8-40.1 days, excluding scheduler wait.
- One generator, about 312,000 cells: 1,969-2,407 node-hours; 16.4-20.1 days
  with five nodes.

This assumes all five concurrency slots are available for judging and each node
uses four A100s. Start-spacing/admission policy and other workloads can increase
calendar time. Do not imply permission to consume this forecast budget.

The observed complete pilot had only 50.03% warm time relative to allocated
occupancy, including its finite queue and idle/tail overhead. If that utilization
persists at corpus scale, the conservative alternative is 6,048-7,070 node-hours,
or 50.4-58.9 days on five nodes. This is a scenario, not a second measured
production throughput or a guarantee that the backlog achieves better use.

The estimate is one SI-v4 answer-map plus source-judging pass, excluding J1,
generation, full-corpus repetitions, v3 comparison, queue waits and semantic
repair/revalidation. Similar answer lengths/source counts and observed completed
throughput are assumed. Failed/ineligible cells remain unresolved; this is not
a promise to produce valid grades for every target cell. A corrected workflow
can have different costs.

Machine-readable calculations:
`analysis/docs/si-v4-r2-full-corpus-cost-20261004.json`, also saved as
`/Users/valerianfourel/Downloads/si-v4-r2-review-20261004/full-corpus-gemma-cost-estimate.json`.
Arithmetic was recomputed from saved reports. No executable code changed.
