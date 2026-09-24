# First Llama allocation on JUPITER

Use `analysis/scripts/prepare_jupiter_llama.py` from a clean committed checkout.
This is one dataset-backed production allocation to obtain Llama throughput for
the frozen population. It does not invent GH200 timings or shared hour packages.

`prepare` runs before allocation. Supply the existing dataset, its keyword
priority file, Qwen's reference runtime for the shared population/search paths,
the pinned Llama serving profile, and the verified recovery publication.
It registers Llama separately, accepts only exact matching recovered results,
and freezes every eligible missing Llama cell in forward keyword order.
Qwen rows and historical outputs remain intact. Progress is printed with
`GEODML_AUDIT_PROGRESS=1`.

`submit` requires explicit approval for one hour, account, partition, history
start date and private HF repository. It checks current scheduler state, including
unnamed live allocations, and requires no other live allocations for this first
measurement. It checks the ten-minute observed-start gap, free bytes/inodes,
unacknowledged storage incidents and file creation. Filesystem availability is
not a verified user/group quota: inspect the site's quota report separately.
If HF already contains Llama hour packages, use the shared dispatcher instead.

The job requests one exclusive node, four GPUs, 32 CPUs and all node memory.
Submission initially holds the job so its ID is recorded before execution.
The helper rechecks admission, then releases it. A durable dataset-wide receipt
prevents duplicate submissions, including after an uncertain Slurm response.
Inspect `submission.json` and Slurm before taking any recovery action. A failed
submission or a held job is never automatically retried or cancelled.

The maintained worker verifies whole-node allocation on the compute host, uses
authenticated loopback serving and does not fall back to `unshare`. It keeps
Llama loaded and checkpoints useful work until the allocation's deadline. Task
admission ends five minutes before the actual Slurm end; cleanup starts two
minutes before it. Startup consumes this same budget. No automatic resubmission.

Preparation records file identities and local verification receipts. Before
submission and execution, unchanged prepared files reuse those receipts;
external search files are checked against their hashes. Task claims and
completion state remain live in the dataset ledger. This skips repeated full
population audits without caching mutable task status.

HF's existing Qwen plan is not replaced by this bootstrap. After the allocation
ends, reconcile its results, publish Llama inputs/results, and refresh shared
hour plans using measured Llama throughput. Do not launch Llama on another
cluster through an independent bootstrap meanwhile.
