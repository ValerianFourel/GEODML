# SI-v4 HoreKa test page, 1 October 2026

Valerian requested pushing SI-v4 and an HTML guide for one one-hour interactive
salloc. This explicitly approves that wall-time for this test, not submission by
Codex, a scheduling-limit exception, or any retry allocation.

## Code and operator page

Execution pin `9aa5d90e0b310f3eea587a60787727787ab92757` includes the SI-v4
implementation `8b8991a` and the new bounded development helper. The subsequent
documentation commit contains this handoff and the pinned HTML.

Tracked page: [horeka-si-v4.html](../horeka-si-v4.html). Identical convenient copy
at workspace root `horeka-si-v4.html`.

New `analysis/scripts/horeka_si_v4.py` prepares matched inputs from the two saved
Gemma bundles, validates the pinned model receipt/runtime/template and native
JSON schemas before allocation, checks admission, and drives a finite queue.
The existing authenticated offline four-A100 stage runner owns model lifetime,
boundary verification and shutdown. Its new trial branch is unavailable through
the legacy Nemotron submission CLI.

Queue: v4 pass1 on 40 development cells, matched 4096-token v3 bridge, 48
constructed development pairs, then v4 passes2/3 with regenerated maps. Corpus
pass counts are 40 maps for v4, 214 sources and 40 J1 tasks. Constructed inputs
deduplicate to 13 maps, 28 sources and 13 J1 tasks. Total ceiling is 1,190 first
attempts before map eligibility filtering, plus bounded retries. No fresh
held-out cases or generator calls. No automatic continuation or submission.

Input migration verifies all saved task hashes and source/J1 agreement. It
preserves source text, request, masking, complete answer and source mappings.
Provenance is explicitly unresolved because these bundles do not contain the
original generator traces. Historical results remain untouched.

Quota refresh starts after model loading and runs every 120 seconds. Refresh
failure invalidates evidence and stops new admissions. Controller role deadline
is three minutes before Slurm end; its admission/cleanup margins precede the
outer server cleanup. All work stays within the original allocation deadline.

## Resource estimate and authorization

One exclusive accelerated node, four A100 40GB GPUs, 32 requested CPUs,
whole-node memory around 512 GiB. Maximum one node-hour/four GPU-hours and up
to 152 reserved CPU-hours. Estimate 20–50 minutes including startup, with a
60-minute wall-time and checkpoint margins. Historical Gemma wrappers took
8.9–10.9 minutes; v4 throughput remains unmeasured. A cheaper 30-minute run may
finish only the initial pass. Three repeats do not redefine the separate 3x
warm GPU-hours-per-completed-cell acceptance limit.

Official HoreKa batch, hardware and filesystem pages were read on this turn.
They list accelerated/A100, four GPUs, 152 hardware threads and 512 GiB memory.
Live scheduler/account availability is checked by the page on HoreKa, not
claimed from local inspection. No SSH or scheduler mutation was performed.

Admission retains five active allocations, no pending allocations, ten minutes
since observed starts, the 295-job account queue guard and fresh quota evidence.
The prior v3 run's scheduling exception is not reused. It refuses an existing
v4 allocation/attempt. Checks and salloc are separate; delayed or changed queue
conditions require another check, not cancellation of other jobs.

## Validation

56 focused tests passed across `test_horeka_si_v4.py`,
`test_horeka_gemma_si.py`, and `test_source_importance_v4.py`.
Coverage includes saved-input preservation, joins, duplicate rejection,
admission limits, routing to the existing boundary, finite queue execution,
and stopping when a run remains incomplete. No live inference is claimed.

The actual verified downloaded Gemma bundles were converted locally on CPU:
40 cells, 214 unique source tasks, 40 maps and 40 unchanged J1 tasks. Native
xgrammar and cached model/template checks will execute during cluster preparation;
local environment lacks those model/runtime artifacts.

All six HTML Bash blocks passed `bash -n`; embedded Python parsed, JavaScript
passed `node --check`, copies match, and `git diff --check` passed.

## Next action

Push execution and documentation commits to `origin/codex/pilot-continuation`,
then use the page from an already-open HoreKa login shell. Valerian allocates
once after admission and returns V4_EXIT, STAGE, QUEUE and RUN_SUMMARY output.
Interpretation, cost acceptance and production promotion remain later work.
Keep partial artifacts and live allocation shells intact.
