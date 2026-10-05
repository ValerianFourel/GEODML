# Gemma v4 sender stopped on empty sacct comments; 57 bouts FAILED

Read the last three handoffs. Evidence below was pasted by Valerian; not live.
Run root `$W/reviews/gemma-si-v4-llama-reuse-5h-20261004`, plan pin 493d06e.

Pasted status: 57 bouts FAILED, 50 RUNNING, 93 PENDING (all 200 attempt-1
bouts). The sampled failed bouts ran 3:06–3:54 and ended with ExitCode 2:0.
Their .err files have no Python exception, only `srun: ... exit code 2`. The
judge returns 2 whenever its report status is not `finished`, i.e.
`incomplete` or `finished_with_failures`. The cause has not been established yet;
the per-shard report summaries were requested.

Sender stopped at 2026-10-04T19:11:56Z with "known allocation no longer matches
its submission comment". sacct shows an empty Comment for finished bouts:
HoreKa accounting does not store job comments. identify() compared the receipt's
job ID against that empty comment once a job left squeue. status.json still
says all 200 shards are `live`; it is stale. No resubmission happened since.

Fix aead85c (branch codex/gemma-v4-sender-empty-sacct-comment-20261005):
a known job ID with an empty accounting comment is accepted; a different
comment, or a live squeue mismatch, still stops. checked_plan now lets a newer
clean committed sender supervise a plan pinned to an older checkout: it verifies
the pinned checkout is clean and at the plan commit. Bouts still run the pinned
493d06e code. No scientific setting, budget (400 bouts), max_inflight (200) or
plan file changed. Tests: new regressions fail on the old code; 61 Gemma v4,
sender and prequeue tests pass.

Queue: the authorized ceiling is 200 in flight and 400 bouts total. The
restarted sender refills only eligible shards (terminal with pending work and
budget left). Shards whose report is `finished_with_failures` are exhausted and
are not retried. Going beyond 200 in flight needs an explicit plan change.

Next: Valerian runs the shard-report diagnostic, then restarts the sender from
a checkout of aead85c in tmux `gemma-v4-bouts`. Codex/Claude ran no cluster
commands.

## Update: the whole llama plan is done (pasted, 2026-10-05 ~14:50 UTC)

squeue is empty. The read-only count over all 200 shard databases shows every
cell closed: 312,052 of 312,052, 0 missing, all 200 shards finished. Tasks:
answer_map 208,002 done; source_dependency 1,004,308 done and 99,816 blocked;
fulfilment 208,132 not_requested (source-importance-only run). 27,074 cells
include a blocked task. A source task is blocked when its parent answer map is
not eligible, failed or quarantined. The per-status breakdown has not been
returned yet. The Slurm "FAILED" bouts were shards that finished early with
judge status other than `finished`, which exits 2. 200 of 400 bouts were used.

The fixed sender (aead85c) restarted at 14:50 UTC on hkn1990 and is reconciling
terminal bouts. With no pending work it should end as finished or
finished_with_failures without new submissions. Semantic acceptance of the
Gemma judgments remains not established. Next: return the blocked-status
breakdown and the sender's final status.
