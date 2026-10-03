# V4 replacement stopped because the saved queue is finished

Valerian ran the published interactive launcher
`1b34bebaf0007ed1010b4a4065a880cb4e10d73c` on hkn1991. The checkout was created
and the command reached `submit`, which raised:

```text
ValueError: evaluation queue already exhausted; no allocation needed
```

Controller log:
`$W/reviews/si-v4-r2-cycle-20261002/development/interactive-20261004-7xV36a.log`.

This guard reads `evaluation-results/queue-summary.json` and fires only when
its `complete` field is true. It runs before scheduler admission, quota checks,
new allocation intent, salloc and srun. No new allocation or inference was
started by this invocation. The earlier manual allocation may have been
registered/reused in the budget before this guard; the returned trace does
not distinguish those paths. Do not assume current scheduler state or new
resource consumption from the exception.

The queue writer sets `complete` when every planned workload has a terminal
report, including `finished_with_failures`. It explicitly keeps semantic
acceptance unestablished. Therefore this is evidence of a saved finished queue,
not a clean quality pass or proof that every cell/source succeeded. Do not
reset the completion marker, discard results, or bypass it to repeat work.

Read the latest three handoffs and checked the queue writer and submit guard.
Prepared `analysis/docs/horeka-v4-r2-results.sh` for Valerian's existing HoreKa
login shell. It reads the pinned configuration, queue summary, recorded job IDs,
each planned workload's latest report, completion/failure counts, warm elapsed
timing and grouped exact map/source errors from saved cell exports. Missing
reports remain visible. It requests no allocation, launches no model and
changes no scientific data. Bash/Zsh syntax and embedded Python parsing pass.
No scientific code or tests were changed. No cluster connection was executed.

Next: obtain this new r2 report output from the cluster and inspect the results.
Then use the existing bounded exporter and assessment commands to evaluate
full evidence. The latest raw inference results read locally remain the older
job 5175229 diagnostic; this new guard establishes that an r2 queue summary
exists remotely but does not reveal its scores, failures, timing or executing
job IDs. Keep Gemma/Nemotron decisions pending the measured r2 results.
