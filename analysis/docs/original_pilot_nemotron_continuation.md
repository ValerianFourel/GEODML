# Original 500-prompt Nemotron continuation

This workflow resumes missing bulk Nemotron judgments for the frozen original
500-prompt Experiment V2 queue. It does not rerun generation, include overflow
cohorts, change the evidence, or alter the judging protocol.

## Current checkpoint

The allocation recorded under Slurm job 1950617 completed 1,298 of 12,000 tasks
without a written failure. It checkpointed at the allocation deadline with
10,702 tasks remaining. The worker ran for 47m56.8s, or about 1,624 judgments per
hour. The complete 58m37s allocation also included environment validation, server
startup, model loading, warmup, drain, and cleanup.

This is a valid operational checkpoint, not a scientific result. Analysis stays
disabled until all expected claims are present and validated.

## Frozen contract

Every continuation must preserve:

- the original adaptive plan and its 12,000 task bytes;
- the original generation artifacts and recorded conversations;
- the shared Nemotron claim root;
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at revision
  `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`;
- bulk judging with thinking disabled;
- 2,048 maximum output tokens, three attempts, and four concurrent requests;
- `request_timeout=120.0` as a floating-point identity value;
- the measured 81,920-token serving window and structured output schema.

The manager hashes and verifies the source manifest, task file, launch receipt,
context budget, model metadata, and continuation lineage. It rejects scientific
drift, a changed shared claim root, duplicate task identities, or a continuation
whose predecessor is not exactly one terminal allocation.

## Lifecycle and receipts

`prepare_continuation` reads its predecessor without changing it. It verifies
Slurm accounting, audits claims, and creates a fresh run root that links to the
frozen plan and context budget. The launch receipt records the exact approved
wall time, resource ceiling, estimate, execution commit, and predecessor
identity.

`submit` checks the clean pinned checkout again immediately before scheduling.
It writes these durable, exclusive receipts:

- `continuation-successor.json` in the predecessor;
- `submission-intent.json` in the new continuation;
- `submission-result.json` after the scheduler call returns.

An existing intent or result receipt blocks another automated submission. This
includes failed, interrupted, and orphaned receipts. Inspect Slurm and the stored
command manually; never delete a receipt to force a retry.

The worker validates the scheduler job identifier, allocation shape, four GPU
UUIDs, launch identity, predecessor link, and original ownership root. It admits
only missing eligible tasks until the controller-derived deadline. Completed
claims are reused and never repeated. Deadline checkpoints remain distinct from
terminal task failures.

## Status interpretation

The manager reports preparation and submission evidence separately:

- `preparation_status=prepared`: a frozen launch exists;
- `submission_status=accepted`: `sbatch` returned a parseable job identifier;
- `submission_status=failed`: `sbatch` returned a nonzero status;
- `submission_status=uncertain`: the durable receipts do not prove one outcome;
- `submission_status=not_submitted`: submission did not begin.

The status command does not query or guess live scheduler state. Use exact Slurm
accounting for that question. A completed allocation can still leave a
checkpointed queue. Queue completion requires all 12,000 expected successful
claim envelopes, not merely an empty failure file or successful HTTP traffic.

## Runtime budget for the next allocation

At the observed rate, the 10,702 remaining tasks need about 6h35m of serving
time. The prior allocation needed about 10m10s before worker inference. Including
admission, drain, cleanup, and task-length uncertainty gives a realistic range of
6h48m to 8h.

The recommended one-allocation request is one node, four GH200 GPUs, 32 CPUs,
512 GiB memory, and eight hours, for at most 32 GPU-hours. A four-hour staged run
uses at most 16 GPU-hours and should finish roughly 6,100 tasks after startup. A
two-hour run uses at most 8 GPU-hours and should finish roughly 2,900. These are
estimates, not guarantees.

No continuation may be prepared or submitted until Valerian approves the wall
time for that allocation. If work remains afterward, recompute the estimate from
the new checkpoint and obtain fresh approval.

## Scope after bulk completion

Bulk completion does not perform validation, adjudication, or scientific
analysis. Those remain separate, explicitly approved milestones. The final bulk
report should show exact counts and all experimental strata, then preserve the
frozen artifacts and receipts for reproducibility.
