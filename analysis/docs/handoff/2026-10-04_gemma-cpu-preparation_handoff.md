# Gemma v4 llama preparation moved to one four-hour CPU allocation

Read the last three handoffs. Evidence below was pasted by Valerian; not live.

Login preparation of run `$W/reviews/gemma-si-v4-llama-login-5h-20261004`
(pin b918021) started 07:06 on hkn1991, scanned refs until 07:14 and was
stopped by its own one-hour timeout at 08:06 (`subprocess.TimeoutExpired`).
No OOM; the SSH drop was unrelated, since the sender kept logging. Partial
freeze: 94,418 cells, 550,296 unique tasks in 52 minutes, about 1,834 cells/min,
so ~3 hours for the estimated ~312k llama cells. No plan, nothing submitted.
That root keeps `LOGIN_PREPARATION_ATTEMPTED` and `frozen.partial/`; do not
delete or retry it.

Valerian explicitly approved one longer preparation allocation: `start
--prepare-on-cpu` submits a single `cpuonly` sbatch (1 node, 4 CPUs, 32G,
`--time=04:00:00`, no requeue, no GPU) and then hands the plan to the sender
as before. The freeze now accepts `--index-directory`; CPU preparation puts the
temporary SQLite deduplication index in node-local `$TMPDIR`. Frozen cells and
tasks are unchanged (tested). Mode is pinned in preparation.json, one attempt only.
`maximum_node_hours` still counts one preparation hour as before (GPU bound).

Tests: 142 pass across gemma v4, sender, prequeue and source-importance suites.
Risk: if the llama population is much larger than ~312k cells, four hours may
not suffice; the freeze is not resumable. Next: Valerian launches the fresh root
`$W/reviews/gemma-si-v4-llama-cpu-5h-20261004` and returns sender.log/squeue.
