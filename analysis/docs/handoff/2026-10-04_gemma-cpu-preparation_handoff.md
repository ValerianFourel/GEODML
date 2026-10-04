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

## Update: CPU preparation submitted (pasted evidence)

Sender started 08:51:02 CEST in tmux `gemma-v4-bouts` on hkn1993 (PID 635627)
from checkout fc9b47a. Startup checks wrote preparation.json at 08:54:06.
At 08:54:07 sbatch returned 0 for job 5179554 (`cpuonly`, 1 node, 4 CPUs, 32G,
04:00:00, no requeue), PENDING (Priority). Other live allocations, not touched:
Qwen 5177268 (running), 5178960 (pending), page-extract 5179545 (running) and
page-relocate 5179548 (pending). That makes five, the AGENTS.md limit.

## Update: CPU preparation timed out after a complete freeze (pasted, 13:15 CEST)

At 13:15:52 job 5179554 had used 3:58:48 of 4:00:00. The freeze had finished
and was promoted to `frozen/`: 312,052 cells_ok, 208,002 answer-map tasks,
1,104,124 source-dependency tasks, 208,132 fulfilment tasks, 907 trace-full
answers, 18 prior-map cells. No `shards/` or `plan.json`: partition and planning
cannot finish, and the pinned sender stops on a missing plan. No bouts submitted.

New `start --prepare-on-cpu --reuse-frozen <dir>` (fresh run only, pinned in
preparation.json) verifies the earlier freeze against this run's own inputs
(manifest hash at start, file hashes, protocol/budgets, sources, exclusion and
prior-map hashes), copies it, records `frozen-reuse.json` and `plan.reused_frozen`,
then partitions and plans in one one-hour cpuonly allocation (default limit).
Partition's SQLite index can live in node-local `$TMPDIR`; shards are
byte-identical (tested). Next root: `$W/reviews/gemma-si-v4-llama-reuse-5h-20261004`.
