# Gemma SI-v3 HTML commands

Valerian replied "tes make a html for those commands on horeka" directly to the
one-hour approval question. Interpreted and communicated as approval of that
single allocation: 01:00:00, one exclusive node, four A100 40GB GPUs, 32 requested
CPUs, whole-node memory (~512GiB), max four GPU-hours / one node-hour. Exclusive
reservation holds all 152 CPUs. No resubmission approved. Estimate remains
15–45 minutes of work; Gemma throughput is unmeasured.

Created `analysis/docs/horeka-gemma-si-v3.html` and an identical convenient copy
at workspace-root `horeka-gemma-si-v3.html`. Five copyable blocks: pinned checkout
and download/removal, preparation, fresh admission + salloc, explicit-job-ID srun
of run.sh, read-only results/logs. The inference code pin stays
ddc93fe5342152d7c33f2abfc0658a48714d7dcc; this follow-up changes documentation only.
The operator must push that commit before HoreKa can fetch it.

Admission reuses the repository scheduler, quota, health and admission helpers;
includes all observed user's job IDs, not just geodml-named jobs. Refuses five or
more active allocations, released pending jobs, or a last observed start less
than ten minutes ago. Independent launchers must be coordinated: a snapshot
cannot guarantee future start spacing. Persists evidence and an exclusive
ALLOCATION_ATTEMPTED marker, so an ambiguous allocation cannot be resubmitted by
repeating the block. Keep live allocation shells open; no cancellation/exit
commands are provided.

Validation: all five extracted command blocks passed bash -n; both embedded
Python programs parsed; copy-button JavaScript passed node --check; five buttons
and matching local/tracked HTML verified; git diff --check passed. Browser visual
inspection was not performed. Existing page CSS/copy interaction reused. No
HoreKa access, model download, deletion, allocation or inference ran here. Cluster
facts remain historical; page performs live checks when the operator runs it.
