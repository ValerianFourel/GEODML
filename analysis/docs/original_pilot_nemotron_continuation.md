# Original 500-prompt Nemotron continuation

This is a generation-read-only continuation of the original 500-prompt study.
It does not rerun Llama or Qwen, judge overflow cohorts, or perform a concurrency
sweep. It preserves recorded-conversation judging, the pinned Nemotron model,
seed 20260915, 2048 output tokens, disabled thinking, and four concurrent requests.

## Approval and execution

Valerian approved two nodes for eight hours on 2026-09-22: eight GPUs total,
32 requested CPUs and 512G per node, maximum 64 GPU-hours. The estimate is
5–7 hours based on 696 judgments completed in 34m39s on one node. Conversation
lengths and startup can change the rate; completion is not promised.

The submission is a two-element Slurm array. Each element independently requests
one exclusive node for eight hours and can start at a different time. There is
no requeue, replacement, extension, cross-node serving, or new inference cohort.
Another allocation requires a fresh estimate and approval.

Use `analysis/scripts/manage_agentic_pilot_judging.py` from a clean pinned
checkout, with Python >=3.10 and the existing JUPITER inference environment:

- `prepare --adaptive-plan PATH --run-root PATH --snapshot PATH
  --execution-commit SHA --approved-walltime 08:00:00` verifies the frozen
  generation inputs and all 6000 cells for each model, verifies full traces,
  builds 12000 recorded-conversation tasks, and preserves the configured
  validation model and 2% sampling policy. Missing validation configuration stays
  explicitly unconfigured, never silently replaced by Nemotron.
- Preparation checks every context with the local tokenizer at 73728 tokens,
  reserving 2048 output tokens, and compiles every output schema. Oversized inputs
  fail without truncation or exclusion. Weight shard presence, nonzero sizes,
  pinned revision, and tokenizer/configuration hashes are checked; this is not
  a full weight-file checksum or a GPU model-startup test.
- `verify --run-root PATH` rechecks frozen queue and snapshot metadata. Existing
  preparations are never overwritten. A preparation interrupted before
  `launch.json` is written requires inspection, not a blind overwrite.
- `submit --run-root PATH --account ACCOUNT --approved-walltime 08:00:00`
  validates remaining task claims, then submits exactly once. The exclusive
  `submission-intent.json` prevents a second paste from submitting duplicates.
  If submission is uncertain, inspect that receipt, `submission-result.json`, and
  Slurm before any retry. Do not delete the receipt to bypass this protection.
- `status --run-root PATH` validates expected claims and reports completed,
  missing, busy, and terminal-failed tasks. It does not add up worker-local copies
  or equate successful HTTP calls with completed judgments.

## Ownership and startup

Both workers use the original adaptive plan's Nemotron claim root. Their stable
SHA-256 task-ID partitions are disjoint, and retained outcomes are validated
before reuse. Static ownership avoids concurrent attempts for the same task
between these two workers. The earlier generator commit-collision root cause is
not claimed fixed. Do not run the old adaptive launcher alongside this job.

The new wrapper checks the one-node allocation and four GPU UUIDs against the
controller, handles JUPITER's missing `SLURM_GPUS_ON_NODE`, and reads actual
start/end times. Deadlines include startup with 120-second admission and
45-second cleanup margins. Each node stops when its partition is exhausted;
it does not steal tasks or manufacture work to keep busy.

The existing standalone server has explicit HF/Transformers offline mode and
authenticated loopback serving on a controller-verified exclusive node. A
fully cached model is required. Real offline startup is confirmed only by
the cluster server log and completed judgments, not CPU test doubles.

The historical smoke-pilot launcher remains pilot-only by default. This new
queue sets `GEODML_JUDGE_PILOT_ONLY=0` in throughput mode to preserve the previous
adaptive bulk request/claim identity. Fixed smoke mode cannot disable this flag.

## Completion and remaining work

`bulk_complete=true` means all 12000 expected claims contain validated successful
Nemotron judgments. Terminal failures and missing tasks are not successes.
Validation and adjudication are separate: this command reports the frozen
validation task count but does not declare it executed or scientifically valid.
Generation results and all historical queues remain unchanged.
