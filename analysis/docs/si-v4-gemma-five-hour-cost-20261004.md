Valerian, the saved Gemma measurements imply **about 8.9–10.4 days for the 624,192-cell proxy at an average of 15 running nodes**, excluding initial preparation and queue/admission delays. Under the new default partition, the finite ceiling is **800 five-hour inference bouts, 16,000 GPU-hours**, plus one preparation allocation of at most one hour and four GPU-hours. The actual eligible saved-cell census and frozen shard plan remain authoritative.

The estimate uses three end-to-end runs in `si-v4-r2-full-corpus-cost-20261004.json`: 176.48–206.31 pipeline-reported complete cells per warm four-A100 node-hour. Completion does not establish valid maps, witnesses or grades. The earlier Gemma semantic findings remain applicable. The range is the extremes of small development measurements, not a confidence interval or a guaranteed five-hour rate.

A five-hour bout with 8.9–10.9 minutes startup and five minutes drain has **284.1–286.1 warm minutes**. That models 835.64–983.74 reported-complete cells, or **835–983 whole cells per well-filled bout**. The profile assumes the existing one-node/four-A100 serving setup, 32 requested CPUs and `--mem=0` on the exclusive node, requesting all its allocatable memory. Each inference bout has a ceiling of five node-hours and 20 GPU-hours.

The partitioner caps the plan at 200 shards and keeps shared answer maps together. Each shard resumes sequentially with its own finite allocation ceiling. The following illustration assumes balanced shards and enough distinct map components; actual grouping can change the ceiling.

| Registered-count proxy | Default shards | Inference bout ceiling | Inference node-hour ceiling | Inference GPU-hour ceiling | Modeled days at average 15 running, early shard completion allowed |
|---|---:|---:|---:|---:|---:|
| 1,000 | 1 | 2 | 10 | 40 | Not applicable: one sequential shard |
| 312,096 | 200 | 400 | 2,000 | 8,000 | 4.46–5.21 |
| 624,192 | 200 | 800 | 4,000 | 16,000 | 8.92–10.41 |

The half proxy has about 1,560–1,561 cells per shard and two allocations each. The full proxy has about 3,120–3,121 cells per shard and four allocations each. These hard ceilings are larger than optimally packed aggregate estimates of 318–374 bouts for the half and 635–748 for the full. The JSON retains both calculations. Per-shard ceilings, not aggregate packing, determine the implemented finite budget.

If every allowed inference bout consumes its full wall time, the half/full resource ceilings correspond to about **5.56 / 11.11 days** at an average of 15 running. The shorter modeled ranges allow final partial bouts to finish normally when their frozen work runs out. Actual charged occupancy must come from Slurm accounting. Initial preparation adds up to one hour of serial elapsed time and four GPU-hours, giving illustrative total ceilings of **8,004 / 16,004 GPU-hours**.

For 1,000 cells, the default partition makes one shard and two sequential bouts. Modeled inference occupancy is about **5.31–6.20 hours**, plus preparation, queue waits and the sender's resume delay. Dividing ten requested node-hours by 15 would misleadingly imply 40 minutes. Its full requested ceiling including preparation is 11 node-hours and 44 GPU-hours.

**Two hundred outstanding jobs is a queue window, not 200 running nodes.** Pending and running jobs share that window. Fifteen running is a historical scheduling assumption, equivalent to about 60 active GPUs. Actual scheduler concurrency can differ. A tmux sender polling every 600 seconds can refill the window only from the fixed plan. Already-pending jobs can start between polls; a sequential resume can wait up to a polling interval plus reconciliation/submission time after its prior allocation ends.

If 200 inference bouts were the entire budget, the ceiling would instead be 1,000 node-hours / 4,000 GPU-hours. Well-filled bouts would cover about **167,000–196,600** reported-complete cells, 26.8–31.5% of the full proxy or 53.5–63.0% of the half; short shard tails can reduce that coverage. At 15 average running, that finite wave consumes about 66.7 clock hours, or 70 hours in ideal synchronized five-hour waves. It cannot cover either registered proxy. Add the preparation allowance if it belongs to the same total budget.

The 624,192 count is historical registered task inventory, not current saved, eligible, unjudged answers. The half count is arithmetic, not a verified per-generator subtotal. Freeze the actual census, immutable inputs and exact finite bout/resource budget before submission. Preserve completed results and task identities; record failed, blocked, uncertain and N/A outcomes separately. No unlimited retry or queue-expansion loop is authorized by this estimate.

This calculation covers one map-plus-source pass. It excludes J1, generation, full-pass repeats, semantic reference review and the v3 cost bridge. Lower concurrency, storage stops, source-length changes and runtime failures can increase elapsed time or leave work unfinished at the hard cap. No inference, submission or live cluster check was performed for this estimate.
