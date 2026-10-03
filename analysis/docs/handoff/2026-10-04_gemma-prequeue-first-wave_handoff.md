# Gemma first wave before preparation

Valerian explicitly authorized killing the login tmux controller and submitting
200 jobs now, with ten-minute checks/refill thereafter. Returned scheduler output
shows preparation 5177483 PENDING Priority, estimated start 2026-10-05 01:30:00
cluster time, one hour, accelerated, one exclusive four-A100 node, 32 CPUs.
This is pasted evidence, not a live query. Qwen 5177268 remains unrelated.

Added `analysis/scripts/horeka_gemma_v4_prequeue.py` and
`analysis/docs/horeka-gemma-v4-prequeue.{sh,md}`. The helper is pinned independently
and imports runtime/sender from the original preparation checkout. Neither the
original scientific plan nor model/protocol code is changed. The shell derives
the original checkout from prepare.sh, checks its pin, replaces only the named
Gemma login tmux session, and starts the adapter with its own log.

The adapter submits at most 200 accepted held jobs with afterok on preparation,
without spacing; no GPU can run before release. Every job has a durable comment
and intent before sbatch. It recovers lost acknowledgements from Slurm, never
blindly repeats them. A known submit-count rejection is retried next 600-second
poll; generic/ambiguous errors stop. Preparation remains intact. After verified
preparation, checked_plan, fresh storage and private HF reservation, it writes
native sender intents/receipts and per-slot assignment scripts before release.
The original sender counts these as first bouts, avoiding duplicate jobs and
preserving the finite budget and resume rules. Fewer than 200 frozen shards
means surplus held jobs are cancelled; failed preparation also retires known
held jobs only. Unknown state is preserved for inspection. The helper obeys the
original preparation deadline and then the sender's original deadline.

Resource ceiling for the first 200 assigned bouts is 1,000 node-hours / 4,000 GPU
hours, included in original plan budgets. Each is five hours, one exclusive
four-A100 node, 32 CPUs, all memory. Historical estimated capacity 835–983 cells
per bout; no new timing or scientific result. These held jobs satisfy immediate
queue submission but do not start inference sooner than preparation, do not
bypass priority, and do not guarantee priority age or acceptance of 200 jobs.

Eight new lifecycle cases cover full first-wave adoption with real worker binding,
interrupted submission and release, surplus cancellation, failed preparation,
submit-count refusal followed by ordinary sender fill, storage at enqueue/release,
and unresolved acknowledgements. Existing Gemma runtime/sender tests also run.
No remote commands or jobs were executed by Codex. Cluster launch remains for
Valerian. Read prequeue.log and prequeue/status.json, then sender/status.json.
First-wave Slurm logs are prequeue/slot-*/slurm-*.out/err; results stay in shards.

Next: user runs pinned published helper and returns queue count plus log. Verify
actual accepted jobs, no duplicate ownership, preparation outcome and release.
