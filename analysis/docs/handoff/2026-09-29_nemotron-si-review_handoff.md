# Nemotron SI-v1 workflow review, 2026-09-29

## Scope and code

Read-only implementation review requested by Valerian, including the latest
Claude Code conversation, its approved SI plan, and the local operator page
`nemotron-si-smoke.html`. Reviewed code pin:
`55b7a4a36d090176a5129c50b564b23d6959b420`, worktree
`.worktrees/threehour-relaunch-fix`. This session changes only this handoff and
its index. No inference, allocation, remote command, publication, or source fix.

**Latest user instruction supersedes proposed follow-up development:** Valerian
likes Opus's implementation and explicitly says not to change it. Only explain
how to verify the running smoke. Findings below are review history, not an
implementation task or authorization to fix anything.

The last three indexed handoffs were read first. Their cluster snapshots are
historical; newer facts below come from Valerian's pasted output, not direct
cluster access. There is no configured HoreKa SSH alias on this Mac.

## Decisions recovered from the Claude conversation

- The revised primary construct is **judge-assessed source importance in the
  completed answer**. Answer-independent ideal relevance/J3 was explicitly
  removed. The earlier description of an actual-versus-ideal gap is obsolete
  for SI-v1. Compare source importance with the generator-emitted pre-answer
  ranking and presented evidence order, then study readiness associations.
- All production judging is to run on HoreKa, for both generators.
- No human annotators are planned. Valerian selected `deepseek-v4-flash-0731`
  and `glm-4.5-air` as reference models. Model agreement is not human ground
  truth; availability, execution and validation have not been verified here.
- Judge complete answers recovered from the final trace when stored answers
  were cut to 1,200 characters. Keep prefix-only cases identified. The builder
  adds stored-answer sensitivity tasks for a deterministic 10% of recovered
  cells; the five-cell smoke disables that sensitivity.
- The Claude plan mentions a 6,000-prompt sample-first option. Do not silently
  substitute that for the full corpus when preparing production.
- The latest conversation explicitly approved the final Qwen bouts 631–893.
  Older handoffs stopping approval at 430 are stale. This is no approval for
  Nemotron production allocations, retries, or new resource profiles.

## Current workflow and assessment

1. `agentic_cells.py` selects ledger-completed generator cells with verified
   generation/trace references.
2. `prepare_source_importance_tasks.py` recovers the judged answer, reconstructs
   observed evidence, and freezes one task per answer/source plus separate J1.
   Identical semantic tasks are deduplicated; cells keep factor metadata and
   links to task IDs. This writes task files, not registered production claims.
3. `source_importance.py` hides ranking and experimental metadata, masks source
   citations, splits answer units, builds the one-source rubric/schema, checks
   quote provenance, and computes tied ranks and alignment metrics. Valid zero
   and missing measurements are distinct. Quotes do not prove entailment.
4. `horeka_nemotron.py` starts the fixed four-A100 serving profile and invokes
   `try_source_importance_judge.py`. Model revision is
   `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`, BF16, TP4, 73,728 context,
   eager, concurrency 4, thinking disabled; SI output cap 640, J1 cap 64.
5. The smoke selects five Qwen cells, sends their tasks through the existing
   bounded retry path, and writes results, audit, per-cell metrics and summary.

The scientific direction fits the revised objective and avoids the old
all-claims/all-sources request. Scientific reliability is still unverified.
In particular, validate advice support and URL-dependent navigation cases across
readiness strata so measurement bias is not mistaken for an axis association.
Generator code and shuffle placement were not modified by the reviewed SI commits.

## Findings before production

1. **Production integration is absent.** The launcher allows only one-hour
   diagnostic trials and hardcodes Qwen. The smoke rejects existing output and
   has no resume, durable shared claim, or registered SI backlog. It schedules
   its entire small task set with `asyncio.gather`; increasing count/walltime is
   not a production solution. Reuse the existing claim ledger, sealed result
   writer and bounded `_iter_execute`/deadline machinery from the production
   runner. Preserve identities across bouts and reconcile abandoned claims only
   after confirming their allocation ended.
2. **Stored identity verification is incomplete.** At
   `source_importance.py:328`, SI task ID and seed are copied from the record
   and then compared to those same values. A local executable probe changed
   them to `wrong-identity` and `42`; `item_from_record` accepted both. Recompute
   identity/seed and validate version/configuration when registering/executing.
   Also, changing rubric text without a version bump leaves the short task ID
   unchanged, although prompt hashes change and old frozen prompts fail replay.
   Durable claims must bind the actual request and judge configuration, as the
   existing `agentic_judge_claim_identity` does. Do not cache by short ID alone.
3. **Mask provenance is dropped during freezing.** `prepare_source_task` creates
   reversible mask spans, but `task_record` omits them and per-cell entries keep
   only a boolean. Original answer/trace data remain intact, so this is not
   corpus loss. Persist the per-cell mapping needed to audit original passages.
   A probe also confirmed `Book at https://example.org/booking [S1].` becomes
   `Book at [ref] [ref].`; substantive URL cases need the documented sensitivity.
4. **Exit zero is not a passing smoke.** The smoke returns zero after reporting
   failed judgments or an empty selection. Inspect selected/complete cells,
   source and J1 success counts, finish reasons and actual request/response.
5. **Documentation/export migration is unfinished.** The paper dataset contract
   still requires legacy ideal-relevance and support fields. Add a versioned SI
   amendment/reader before export; do not pool old and new scores.

## Verification

Ran with the existing conda Python:

```text
/Users/valerianfourel/miniconda3/bin/python -m pytest -q \
  analysis/tests/test_source_importance.py \
  analysis/tests/test_source_importance_pipeline.py \
  analysis/tests/test_horeka_nemotron.py
45 passed in 2.78s
```

The initial system Python attempt could not run pytest because it is not
installed there. No dependencies were installed and no tests were changed.
The identity/mask findings above were additional in-memory CPU probes, not
scientific results. Live SI inference and semantic quality remain unverified.

## HoreKa evidence and next step

Pasted 2026-09-29 08:27 cluster snapshot: Qwen 142 COMPLETED, 4 FAILED,
282 PENDING, 12 RUNNING; account queue 294/295 before the interactive allocation.
These are scheduler counts, not verified cell completion. Start estimates are
provisional scheduling predictions, not completion guarantees.

Valerian reports the smoke running in allocation **5169925**, node **hkn0403**,
one hour, one exclusive node/four A100, 32 requested CPUs, all node memory.
Pasted setup confirms pin `55b7a4a`, weights verified, SI trial available,
295/295 account jobs. Run:
`/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/runs/nemotron-si-smoke-20260929-0829`.
Only the serving-profile fingerprint was shown after startup; that print comes
from `search_vllm_stage.py prepare`, not from an inference result. No server log,
CELL line, summary, or final state has yet been received. Preserve the live shell.
Absence of a tmux sender on the compute node says nothing about login-node senders.

Sent read-only commands for a second HoreKa login terminal: `squeue -j 5169925`,
`srun --jobid=5169925 --overlap nvidia-smi`, and tails of
`logs/interactive-5169925.log` and `attempts/job5169925/server.log` under that run.

Next: inspect returned logs and the final smoke summary/example, and report
whether the existing implementation works. Do not change Opus's implementation.
For future bout sizing, measure sustained validated task throughput on a frozen representative backlog,
including startup, retries, checkpointing and shutdown. Divide unique SI/J1 tasks
into finite packages while retaining cell links and keyword ordering. Count
actual sources and sensitivity tasks; Qwen seconds/cell do not size Nemotron.
The Claude plan's 24–94 five-hour bouts assume unmeasured concurrency 32–128;
current concurrency is 4. Those numbers are scenarios, not a measured budget.
One five-hour four-A100 bout has a ceiling of 5 node-hours/20 GPU-hours; actual
number of bouts, runtime margin, concurrency and admission exceptions need fresh
evidence and explicit approval. No allocating commands prepared in this review.


## Subsequent smoke result and runtime estimate

This update supersedes the awaiting-output status above. Valerian pasted the
finished smoke from job 5169925: 5 cells selected from 48,706 completed Qwen
cells; 138.4 seconds in the timed judging section; J1 5/5 valid; SI 10/29 valid,
19/29 terminal semantic failures after 38 semantic failure attempts. Every final
response ended with `stop`; maximum SI output 444 tokens, below the 640 cap.
First error: `JudgeOutputError: answer quote not found in a3`. No cell has all
source grades, so all five alignment results are undefined. Exit zero confirms
that the diagnostic ran to completion, not that the source judgments passed.
The example also retains compound citation markers such as `(S3, S5, S7)`;
previous blanket blinding claims do not cover that observed form. No code changed.

At the observed mix and concurrency 4: 138.4 / 5 = 27.68 node-seconds per cell,
about 130 cells/hour, including J1 and bounded retries but excluding model startup
and dataset selection. Linear processing-cost scenarios, not completion forecasts:

| Population | Cells | Timed node-hours | 5-hour bouts with assumed 15-minute overhead |
| --- | ---: | ---: | ---: |
| Qwen cells visible to smoke selector | 48,706 | 374.5 | about 79 |
| One full generator | 312,096 | 2,399.7 | about 506 |
| Both generators | 624,192 | 4,799.3 | about 1,011 |

The overhead is illustrative, not measured. Both-generator scenario is roughly
20,200 GPU-hours including that overhead, or 13.2 days at 16 continuously occupied
nodes / 6.6 days at 32, excluding queue waits. No such concurrency is authorized
or promised. These extrapolations omit sensitivity work and corpus-wide dedupe,
assume the same source-count/answer-length mix for Llama, and rely on only five
Qwen cells. With zero fully measured cells, time to a complete valid corpus is
unknown. Do not divide by the success rate and assume repeated retries will cure
systematic quote failures. User still forbids implementation changes. No new
allocation, benchmark, retry, or production launch was prepared or submitted.
