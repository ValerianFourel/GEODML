# Gemma v4 finite five-hour dataset queue

Valerian requested Gemma v4 dataset judging, first estimating fifteen running
jobs, then explicitly authorizing five-hour bouts, up to 200 Gemma allocations
queued ASAP without start spacing, and tmux monitoring/refill every ten minutes.
This supersedes the usual one-hour/five-allocation/start-gap defaults for this
pass. The window counts this plan's pending and running jobs; other jobs remain
unchanged. Fifteen running nodes is an estimate assumption, not a throttle.

Prepared `analysis/scripts/horeka_gemma_v4.py`,
`horeka_gemma_v4_sender.py`, `partition_si_v4_tasks.py`, the unchanged-science
Gemma full-pass config template, and `analysis/docs/horeka-gemma-v4-bouts.{sh,md}`.
The shell script starts a tmux sender and sources the existing environment inside
its window, avoiding a stale tmux server environment. No environment redesign,
model download, model/prompt repair, cluster access or allocation occurred here.

One bounded one-hour four-A100 preparation allocation reads the supplied Qwen
and Llama dataset copies, verifies sealed completed generation references,
recovers full answers and freezes exact SI-v4 inputs. Its duration at full corpus
scale is unmeasured. The partitioner streams through SQLite, preserves raw
records, groups shared map tasks, orders keywords forward and emits up to 200
durable nonempty shards. Prior named diagnostic fingerprints are excluded with
receipts. Different cells sharing a prior diagnostic map are retained as
`prior_map_requires_reconciliation`, without duplicate inference or a claim that
earlier results were imported. Blocked inputs remain separately visible.

Inference keeps the existing Gemma revision, BF16, TP4, 73,728 context, four
concurrent requests, 4,096 tokens, temperature zero, disabled thinking and compact
xgrammar. It runs map-plus-source only; fulfilment, generation, v3 bridge and
full-corpus repeats are excluded. The existing HoreKa authenticated loopback and
exclusive-node boundary is reused. Each bout uses one exclusive four-A100 node,
32 requested CPUs, `--mem=0`, `accelerated`, five hours and no automatic requeue.
Slurm's actual end governs admission/draining. Fresh storage/quota evidence is
checked before submission/start and refreshed during inference.

A private HF compare-and-set SI registry reserves exact map identities, separate
from generator/shared-hour ownership. It never expires ownership or marks results
complete from accounting alone. Results remain in the cluster workspace; this
reservation is not a results upload. The existing cache/authentication is checked
before preparation allocation. The local source copies are not automatically
updated from HF; subsequent generator completions are outside the frozen pass.

The sender records fsynced intents before sbatch, unique scheduler comments and
job receipts. Worker startup binds the exact comment, plan/config hashes and
known job ID, including the fast-start-before-receipt case. Each intent permits
one worker binding. Ambiguous submissions are reconciled without duplicates;
live squeue state wins over old terminal accounting. Resumptions require confirmed
terminal owners, saved-result verification and measured remaining-work estimates.
Completed tasks and terminal inference failures are not retried. Startup/no-progress
failures stop that shard. A clear submit-count rejection is polled again; a generic
Slurm policy rejection stops with its reason. Preparation monitoring uses the
whole live queue so aged-out completed jobs reach the sacct fallback.

The total allocation budget is frozen per shard, ceil(cells * 20.398689 / 17046),
not an unlimited resubmission loop. The sender also has a thirty-day deadline.
The current code cannot complete arbitrary remaining work beyond these ceilings;
it reports it explicitly. Same-output restarts reject changed sources, exclusions,
repository, account or code. Status files include blocked inputs and exclusions,
and distinguish exhaustion of eligible frozen work from full-corpus or semantic
acceptance. Other live allocations are never cancelled or extended.

Cost evidence is `si-v4-gemma-five-hour-cost-20261004.{json,md}`. At 176.48–206.31
reported-complete cells per warm node-hour, 8.9–10.9 minutes startup and five
minutes drain, a filled five-hour bout models 835–983 complete cells. The historical
624,192 registered-cell proxy is not a live usable-answer count. With balanced
200-shard preparation it models 8.92–10.41 days at average fifteen running nodes,
with an 800-bout ceiling: 4,001 node-hours/16,004 GPU-hours including preparation.
The arithmetic half models 4.46–5.21 days, with 2,001 node-hours/8,004 GPU-hours.
Actual grouping/census determines the prepared budget. Queued jobs do not establish
running concurrency. Full-budget occupancy at average fifteen would be about
11.11 days for the full proxy, plus preparation/queue interruptions.

Three resumed maximum-effort Astra agents also finished the pending Nemotron
diagnostic. It remains unsuitable: zero numeric grades, zero complete cells,
eighteen false unusable maps and one validation failure; the seven executed source
calls abstain with unsupported findings. Gemma is better at producing output but
still fails the semantic standard. The original committed report is preserved;
the resumed review receipt verifies twenty cells/107 sources and unchanged raw
evidence/blind hashes under `output/nemotron-v4-review-20261004` in the workspace.

Validation: 205 focused and existing tests passed. Synthetic tests exercise the real freeze/partition/coordinator,
durable submission state and fake external Slurm/transport boundaries. They do
not establish GPU or scientific performance. The new policy-rejection regression
was first observed failing and then passed after narrowing the retry condition.
Shell syntax and diff whitespace checks passed. The final test result and pushed
SHA are reported to Valerian with the exact pinned launch command. Next milestone:
Valerian runs that command in a separate HoreKa login shell and returns the
prepared census/status. No jobs have been submitted by Codex.
