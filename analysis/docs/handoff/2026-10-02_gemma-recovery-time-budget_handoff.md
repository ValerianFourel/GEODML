# Gemma recovery time budget

New pasted Slurm snapshot: job 5175121 RUNNING on hkn0403, elapsed 49:40
of 01:00:00, time left 10:20. Interactive, extern and step .0 are running.
A running shell/step and accounting ExitCode=0:0 do not overturn the earlier
judge GEMMA_EXIT=2 or establish successful scientific completion. This snapshot
is historical immediately after capture; no live query was executed by Codex.

Read the last three indexed handoffs. Existing historical Gemma startup is
about 9–11 minutes. Ten minutes twenty seconds cannot defensibly cover startup,
remaining inference, checkpointing and cleanup. Do not weaken the 20-minute
startup floor, extend or replace this allocation, or terminate its owning shell.
No allocation or model work was started.

## Concrete recovery scope

Source run: reviews/gemma-selected-xsk1n4bx/run/attempts/job5175121/trial/v4-pass1.
Preserve all original frozen inputs, database, shards, receipts and attempts.
Verify sealed references before carrying forward the 84 successful logical tasks:
9 answer maps, 55 source judgments, 20 fulfilment judgments.
Remaining scope: 69 logical tasks, comprising 11 failed answer maps, six failed
source judgments, and 52 source dependencies blocked by failed maps. Successful
maps must be reused for the six source retries. New maps must feed only their
original dependent sources. Failed replacements must remain explicit failures.

Published correction b3cab6e enables compact JSON formatting for newly prepared
selected-cell runs and improves corrective feedback. Record the decoding and
feedback revision, old execution identity, inherited result provenance and new
execution identity separately; do not label mixed inherited/new outputs as an
unchanged run. Keep original semantic settings, model revision and task content.

A provisional planning range is 20–40 minutes including startup and cleanup,
based on previous SI wall time of about 16.83 minutes plus historical startup.
Corrected decoding throughput is unmeasured; this is not a completion guarantee.
A possible subsequent proposal is one 45-minute allocation on one exclusive
four-A100 node, 32 requested CPUs, whole-node memory: at most 0.75 node-hours,
3 GPU-hours and 24 requested CPU-hours. It is NOT approved and no allocation
command is provided. Cheaper alternative: reconcile saved outputs offline first;
this needs no model allocation. A smaller frozen retry segment would trade less
resource exposure for repeated model startup.

The failed-only execution/import path is not implemented or verified yet.
Do not use fresh or relaunch the existing run.sh as a substitute. Complete the
isolated recovery preparation, verify inherited sealed records and regression
proof, then present the concrete launch package and obtain fresh wall-time
approval. No permission to allocate is inferred from this status paste.

No production code changed in this status/budget turn. The operator HTML now
shows the returned time-left evidence and explicitly says it is not launch-ready.
