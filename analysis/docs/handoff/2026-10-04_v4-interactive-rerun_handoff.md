# V4 pasted failure review and interactive continuation

Valerian asked again whether the last inference results and error had been read,
then requested a new salloc and rerun. The request authorizes one replacement
segment within the existing cycle budget. Use a separate HoreKa login shell in
tmux and an accelerated GPU allocation. Do not use cpuonly. Preserve existing
allocations. No cluster SSH, scheduler query, allocation, inference, model
download or model substitution was performed locally in this turn.

## Previously pasted inference evidence

Read the original response-item user messages at lines 4063 and 4128 of
`/Users/valerianfourel/.codex/sessions/2026/10/01/rollout-2026-10-01T22-27-37-01a0f7a6-2a31-7112-8218-64350455ba80.jsonl`.
The latter complete diagnostic was extracted to
`/private/tmp/v4-job5175229-pasted-diagnostic.txt`. These results were supplied
by Valerian; asking for them again was unnecessary.

Job 5175229 used historical v4 at
`bb76802ff5875af6bb777fb1d6361e1af6f97df8`, before r2. It finished with failures,
exit code 2: one of two cells complete, seven of ten source slots scored, one
failed answer map and three blocked sources. Fulfilment was not requested.
Semantic acceptance was not established and zero cells had verified provenance.
Warm SI time was about 78.24 seconds, not a sum of overlapping request times.

The EMR map first failed because `a4w2` to `a5w14` crossed answer units. Its
corrective retry succeeded by keeping the H.R. claim in two separate spans.
The proctoring map failed twice with identical rejected output hashes and
`JudgeOutputError: exclusion overlaps claim or exclusion at word IDs: a1w22.`
The word is the bare marker `1)`, both claimed and excluded. Both attempts
finished with `stop`, at 445 and 446 output tokens. This was not truncation.
That map failure blocked the three proctoring sources.

The EMR source explanations also conflict with fixed map roles: H.R.'s
secondary example was described as peripheral and c9's major recommendation
as secondary. Those are explanation inconsistencies. Grades and whole-source
support cannot be corrected mechanically from the labels or selected witnesses.
These findings explain the r2 repair and evaluation, already published at
`caf8a7d3fbe69736789767efcc4551d0e0347c11`. They do not prove that r2 passed.
The later job 5175818 was granted, but its inference output is still unavailable
locally. Live state must be obtained by the supplied launcher's checks.

## Interactive continuation

Extended the existing `run_si_v4_cycle.py submit` owner with `--interactive`.
It performs existing input/configuration, ownership, scheduler and storage
checks, captures actual prior GPU-hours, records saved workload progress and a
warm-time estimate, then makes one `salloc --no-shell` request and runs the
original pinned `run.sh` through an explicitly identified srun step.
The budget lock is released before the step can enter the cycle membership
gate. The allocation remains until its scheduled end, including after a failed
step. There is no cancellation, extension, retry loop or CPU allocation.

`--account-existing-job 5175818` includes the user's earlier manual allocation
even if it never reached registration. An existing matching receipt is reused
once. Import requires terminal accounting with the correct user, account,
job name, time limit and one-node/four-GPU resources. The scheduler check still
rejects a live allocation even when accounting has a stale terminal row.
Missing accounting or unresolved submission ownership prevents a new request.

The scheduler snapshot includes every user allocation for the interactive
path. The isolated release reuses the already published Qwen scheduler-snapshot
fix, including expanded array tasks and live-over-terminal precedence. It does
not change the Qwen sender, which was previously launched with cpuonly audits.

Salloc waits at most thirty seconds for resources. It has no process timeout
that could kill an allocated job during prolog. Success, failure and ambiguous
receipts are retained. Unknown ownership blocks another attempt. Actual cost
must remain within 32 GPU-hours total, 16 per phase, and four one-hour segments
per phase. The new segment uses one exclusive four-A100 node, 32 requested CPUs,
default memory and at most one hour: one node-hour and four GPU-hours.
Historical model startup is 8.9–10.9 minutes. The segment estimate is 30–60
minutes with drain/cleanup margins; saved warm throughput refreshes remaining
work estimates. A shorter allocation has lower reserved cost but the same
startup overhead. Actual exclusive CPU allocation comes from Slurm accounting.

## Release and verification

Active implementation commit: `b410dc2`.
Published launcher: `1b34bebaf0007ed1010b4a4065a880cb4e10d73c`, branch
`codex/v4-interactive-20261003`, isolated checkout
`.worktrees/v4-interactive-release`. Scientific execution remains at the
configuration's original r2 commit `caf8a7d3fbe69736789767efcc4551d0e0347c11`.
Neither frozen configuration nor scientific outputs are rewritten to adopt
the launcher revision. The new controller's path and source hash are recorded.

102 focused tests passed in the active and exact release checkouts:

```bash
/Users/valerianfourel/miniconda3/bin/python -m pytest -q \
  analysis/tests/test_si_v4_cycle.py \
  analysis/tests/test_si_v4_execution_exports.py \
  analysis/tests/test_horeka_si_v4.py \
  analysis/tests/test_capture_agentic_scheduler_snapshot.py
```

New coverage exercises durable allocation intent, membership before execution,
lock release, step failure without allocation cancellation, ambiguous receipts,
prior manual accounting, phase exhaustion from measured GPU-hours and warm-time
estimation without summing request durations. Existing queue tests verify saved
results resume without regenerating completed maps. These replace external
Slurm/model boundaries and establish no cluster execution or scientific result.
Shell syntax and `git diff --check` pass.

Operator template: `analysis/docs/horeka-v4-r2-interactive.sh`.
Rendered local copy:
`/Users/valerianfourel/Downloads/horeka-v4-r2-interactive-20261004.sh`.
It fetches the published launcher into a clean detached checkout and resumes
`$W/reviews/si-v4-r2-cycle-20261002/development`, with a new uniquely named
controller log there. Valerian receives two paste-ready blocks: create tmux,
then run the launcher. Next evidence is its admission output, allocation ID and
returned inference logs. Nemotron remains conditional on a matched measured
comparison; the current rerun retains Gemma.
