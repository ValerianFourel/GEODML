# Reset Gemma while preserving Qwen 5177268

Latest explicit instruction supersedes the preceding request to cancel all jobs:
preserve Qwen job 5177268 and its sender, cancel the other jobs, and relaunch Gemma.
User last showed login preparation starting at 2026-10-03 19:17:35 UTC; no inference
results or allocations have been established. This is pasted evidence only.

Added horeka-gemma-v4-reset.sh. It stops only Gemma login controllers/preparation
on supplied login hosts (hkn1990 and hkn1991 for this run), using local execution
or noninteractive intra-cluster SSH. Failure to stop either host aborts before
cancellation. Qwen senders are not signalled. It snapshots the user's queue and
cancels the explicit IDs other than the preserved ID, never scancel --user. It
waits for other jobs to leave the queue and starts reset mode in named tmux.

Reset mode uses shared run/sender locks and exact-ID accounting to require all
recorded Gemma jobs were cancelled before starting. It preserves actual inference
and refuses a reset if saved inference exists or an allocation ran. It reuses a
verified completed plan; otherwise it renames interrupted input preparation into
cancelled-wave-archive, records the new helper pin and rebuilds from the same
source paths on login, nice 10, library threads one, one-hour timeout. This does
not preserve an unfinished snapshot's census: newly completed source answers can
enter the rebuilt snapshot. Source datasets and original diagnostic exclusions
are unchanged. No saved file is deleted or overwritten. Interrupted reset
preparation is not automatically repeated. Completed original inference plans
keep their pin/settings and resource budgets.

No prequeue adoption/hold path is used for the new jobs. After preparation,
original finite sender submits ordinary ready five-hour bouts up to 200 in
flight, polls/refills every 600 seconds, and keeps original allocation ceilings.
The 200 cancelled-before-start jobs used zero allocation time; their submission
records remain archived rather than counted as new inference work. Fresh log is
reset.log and reset.json records progress. Qwen remains independent.

Tests exercise actual reset shell with external scheduler/SSH stand-ins and
verify only non-preserved IDs reach scancel, both login hosts are addressed,
partial inputs are archived, ordinary sender submits fresh unheld jobs, and
remaining live jobs block reset. Existing runtime/sender/prequeue suites also run.
No cluster execution by Codex. Next: user runs pinned reset entry point and returns
reset.log if blocked. Internal SSH requires existing noninteractive access; if
unavailable the reset aborts before cancellation rather than silently leaving an
old sender running on the other host.
