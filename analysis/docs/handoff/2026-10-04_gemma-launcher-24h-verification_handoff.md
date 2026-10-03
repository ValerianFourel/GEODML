# Gemma launcher verification and 24-hour progress check

## Task for the next session

Check whether the Gemma launcher implements and delivers Valerian's requested
dataset-wide v4 pass: five-hour GPU bouts, up to 200 pending/running jobs from
this plan submitted ASAP, and a tmux sender checking/refilling every 600 seconds.
Valerian added "like 1 day or so". Treat one day as a desired completion horizon
and a useful progress checkpoint, not evidence of achievable runtime or approval
to expand the finite budget. This handoff does not schedule an automatic check.

Review the implementation and returned cluster artifacts before changing code.
Preserve unrelated allocations, task identities, completed results and the
unchanged scientific configuration. Do not relaunch into a new output directory
to bypass errors. Do not infer live state from earlier pasted logs.

## Exact implementation and evidence

- Active checkout: `.worktrees/threehour-relaunch-fix`.
- Launcher commit: `d9560dd21ee19d756f1a12b77f1b50ba65a59242`.
- Published branch: `codex/gemma-v4-bouts-20261004`.
- Entry point: `analysis/docs/horeka-gemma-v4-bouts.sh`.
- Preparation: `analysis/scripts/horeka_gemma_v4.py`.
- Partitioning: `analysis/scripts/partition_si_v4_tasks.py`.
- Sender: `analysis/scripts/horeka_gemma_v4_sender.py`.
- Configuration: `analysis/config/si_v4_gemma_full_pass.template.json`.
- Runtime evidence: `analysis/docs/si-v4-gemma-five-hour-cost-20261004.{json,md}`.
- Run root: `$W/reviews/gemma-si-v4-dataset-5h-20261004`.
- tmux session: `gemma-v4-bouts`.

The preceding handoff records 205 passing tests plus shell syntax and whitespace
checks. These are local/synthetic evidence, not proof of Slurm execution, storage
capacity, five-hour throughput or scientific validity. No launcher submission
output has been returned in this conversation. This session made no cluster query
or submission and did not rerun that test suite.

## Required checks and acceptance evidence

1. Trace the shell entry point through GPU preparation, input freeze, partition,
   HF ownership reservation, submission, worker startup, checkpoints and resume.
   Confirm the actual cluster checkout matches the launcher commit and is clean.
2. Verify the prepared census against the supplied local Qwen and Llama copies.
   Reconcile ready cells, excluded prior diagnostics, blocked inputs and
   `prior_map_requires_reconciliation`. Later generation completions are outside
   this frozen pass. Do not label a subset as the entire current HF dataset.
3. Verify one bounded one-hour GPU preparation allocation, then one exclusive
   four-A100 node per inference bout, 32 requested CPUs, `--mem=0`, five-hour
   walltime and no automatic requeue. Preparation duration at full scale remains
   unmeasured. Record its outcome before interpreting missing inference jobs.
4. Check submissions have no artificial start gap and count up to 200 jobs from
   this plan pending/running together. Fifteen nodes is not a throttle. Use live
   queue and accounting evidence to distinguish account/QOS limits, scheduler
   waiting, sender failure and lack of eligible work. Leave other jobs untouched.
5. Confirm the sender survives terminal disconnection in tmux and status/log
   timestamps advance at approximately 600-second intervals while active.
   Establish that refill actually occurs after terminal accounting and saved-result
   reconciliation. Inspect intents, receipts and worker bindings for duplicate
   submission, overlapping ownership or unresolved ambiguous acknowledgements.
6. Check that shared answer maps stay together, workers retain a loaded model
   while eligible tasks remain, each completion is saved, and replacement bouts
   reuse completed tasks. Terminal inference failures must remain reported rather
   than retried indefinitely. Startup/no-progress failures must block the shard.
7. Verify actual Slurm end-time draining, fresh bytes/inodes/quota checks and
   authenticated loopback serving on exclusive nodes. Storage failures must stop
   new admission while preserving artifacts and existing allocations.
8. Confirm the frozen per-shard allocation ceiling and thirty-day sender deadline.
   No automatic budget expansion. Results remain local; HF map reservation does
   not mean results were uploaded or scientifically accepted.
9. Compare executed configuration to the pinned Gemma template and v4 protocol.
   This pass runs mapping plus source importance, without generation, fulfilment,
   v3 bridge or repeated validation. Operational completion and semantic quality
   require separate conclusions. The earlier Gemma semantic review is unaccepted.

## One-day feasibility and measured ETA

The historical 624,192-cell proxy models 8.92–10.41 days at an average of fifteen
running nodes. Achieving that work in 24 hours needs approximately 134–157 nodes
running on average, before preparation and extra queue delays. Reserving the full
4,001-node-hour proxy ceiling in one day would require about 167 average nodes.
Two hundred queued jobs alone cannot establish either rate. For the arithmetic
half-population proxy, the throughput estimate requires about 67–79 average nodes.
Use the actual frozen eligible census instead of either proxy when available.

At approximately 24 hours after launch, collect Slurm elapsed/resource accounting,
including active allocations, verified complete cells, blocked/failed counts,
pending tasks, and completed/remaining bout budgets. Compute average running nodes
as allocated node-hours divided by elapsed wall-hours over the same observation
window. Calculate completed-cell throughput per allocated node-hour, including
startup and partial bouts. Separately report warm inference throughput if useful.
Never add overlapping request-seconds and call that node-hours.

Estimate remaining node-hours from remaining eligible cells and measured
throughput, stratified by shard where source counts or failures differ. Divide by
observed average running nodes for a conditional ETA, with queue/throughput range.
Flag insufficient remaining bout budgets; do not enlarge them. Report cells that
cannot complete under current inputs/protocol separately from the timed backlog.
If no valid cells complete, report that no useful completion ETA exists.

## First evidence to collect

Run in an already-open HoreKa login shell. This block only reads current state.

```bash
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  RUN="$W/reviews/gemma-si-v4-dataset-5h-20261004"
  tmux has-session -t gemma-v4-bouts 2>/dev/null && echo 'TMUX present' || echo 'TMUX absent'
  for FILE in preparation-submission.json plan.json sender/status.json sender/summary.json; do
    if [ -f "$RUN/$FILE" ]; then
      printf '\nFILE %s\n' "$FILE"
      cat "$RUN/$FILE"
    fi
  done
  if [ -f "$RUN/sender.log" ]; then tail -n 100 "$RUN/sender.log"; fi
  squeue --me -o '%.18i %.36j %.10T %.10M %.10l %.6D %R'
)
```

Then obtain accounting for the exact preparation and inference job IDs recorded
in the artifacts, plus selected worker logs/reports and saved-result receipts.
Do not use old diagnostic job IDs as evidence of this launch. Inspect failed,
resumed and completed shards as well as currently running shards.

Deliver a requirement-by-requirement verdict with artifact paths, measured
concurrency, throughput, remaining budget and ETA. Mark missing evidence as
unverified. If a defect is found, fix its owning layer locally, run focused tests,
commit/push and provide a pinned continuation that preserves existing work.

## Handoff validation

Read the previous three indexed handoffs and inspected the launcher guide and
sender status handling. This change adds review instructions only. No executable
code, runtime settings, budgets or scientific results changed. Validate the
handoff link and run `git diff --check` before committing.
