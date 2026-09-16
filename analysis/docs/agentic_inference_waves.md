# Restartable agentic inference waves

The inference pipeline has three independent task queues:

1. generator cells for Qwen3.8 and Llama 4;
2. blinded bulk judgments;
3. blinded validation and adjudication judgments.

Each queue can use the same wave mechanism. New CLI preparations default to a
version-2 `backlog` wave: freeze all approved missing tasks once, and give every
worker access to that same immutable backlog. Stable hash slots set the preferred
order, not exclusive ownership. After its preferred tasks, a worker tries missing
tasks in other slots. Per-task shared locks, not the number of submitted jobs,
prevent simultaneous inference on the same task.

Later allocations reuse the same backlog and claim directory. They discover
committed results and skip inference without rebuilding slices. Different worker
counts can overlap safely when all workers use these claims. No job submits its
successor or adds prompts beyond the frozen plan. Adding approved work requires a
new immutable plan; do not rewrite a live wave directory.

Historical version-1 waves and the Python builder's default `partition` mode keep
their fixed subsets for compatibility. They do not gain work stealing merely by
checking out newer code. The historical paired-trial submitter remains one such
fixed-plan path; prepare a version-2 wave to use the new dispatcher.

## Components

- `prepare_agentic_new_cohort.py` freezes a shared new-prompt cohort, excluding
  the original 500-prompt selection by both ID and normalized question text.
- `prepare_agentic_generation_tasks.py` freezes generator cell IDs.
- `prepare_agentic_judge_tasks.py` freezes blind bulk and validation queues.
- `prepare_agentic_adjudication.py` applies predefined disagreement rules and
  produces additional blind validation tasks.
- `prepare_inference_wave.py` freezes a shared backlog from one of those sources;
  `--dispatch-mode partition` retains the historical disjoint worker queues.
- `run_inference_wave_worker.sbatch` validates one array task and invokes a thin
  worker launcher.
- `run_agentic_generation_worker.sh` dispatches a generator worker to the
  pinned Qwen3.8 or Llama 4 launcher.
- `run_agentic_judge_worker.sh` serves the pinned judge profile and runs the
  bulk or validation queue through the common vLLM runner.

The Slurm worker intentionally has no `#SBATCH --time` or fixed accelerator
request. A specifically approved wall time and resource request must be supplied
at submission. The worker records the approved time and its supporting estimate
in `allocation_attempts.jsonl`.

## Fill the approved allocation with useful work

This is the default for new inference batch submissions. Freeze a backlog large
enough to exceed the estimated capacity of each approved allocation. Do not size
a one-hour throughput run as a 12/24-task smoke test. Workers keep one model server
loaded and refill the rolling scheduler with eligible missing tasks. They do not
sleep until expiry, repeat completed tasks, or retry failures indefinitely.

`inference_budget.AllocationBudget` uses `SLURM_JOB_END_TIME`, the projected job
end supplied by [Slurm](https://slurm.schedmd.com/sbatch.html). Model loading,
compilation, and earlier work consume this same budget. When an actual start time
and recorded approval are available, the deadline is the earlier of Slurm's end
and the approved duration from that start. It never starts a fresh hour after
model startup. The generic wave and Nemotron throughput wrappers require a valid
deadline before loading the model.

Defaults stop new admissions 120 seconds before the deadline. Existing calls can
finish until 45 seconds before the deadline; unfinished calls are then cancelled
without becoming fabricated successes or model failures. These margins are
configurable with `GEODML_START_MARGIN_SECONDS` and
`GEODML_CLEANUP_MARGIN_SECONDS`. A hard kill or slow filesystem can still prevent
clean shutdown. Slurm remains the hard limit.

Generator, primary/reranking and judge controllers record runtime
`allocation_budget` outside scientific resume identities. A deliberate cutoff
returns a successful process exit with `status=checkpointed`,
`stop_reason=allocation_deadline`, and the true remaining count. Exit zero alone
does not mean every cell completed. The batch allocation journal distinguishes
this from `complete`. Resume only the missing work in a separately approved run.

If a frozen queue is exhausted, the worker exits and records `queue_exhausted`.
If every remaining task is locked by another job, it reports shared work busy
instead of falsely reporting completion. It revisits busy tasks after making
useful progress, but does not spin or hold idle GPUs when all remaining work is
owned elsewhere. Terminal bounded failures are recorded separately and require
review rather than being retried in every allocation. Full allocation usage is
a throughput objective, not a guarantee of 100% instantaneous GPU utilization.

Historical fixed-scope pilots and finite CPU stages retain their task definitions.
They are explicit exceptions and are not the default for new throughput launches.
The legacy `chain_resubmit` helper no longer submits additional jobs: saved gaps
require a new estimate and fresh wall-time approval. This policy never extends a
running allocation, changes model/token settings, or silently updates pinned jobs.

## New prompts for a paired generator trial

The CPU-only `prepare_agentic_new_cohort.py` reads the original pilot's
`selection-manifest.json` to resolve the frozen full-population prompt and axis
files. It verifies all source and original-pilot artifact hashes and row counts.
Its defaults require the 26,009-row population and 500 excluded pilot prompts.
The original files are read-only. A new output directory is required.

The default cohort contains 120 prompts, six from each of the 20 existing axis
bins. It preserves the population's observed percentiles and bin boundaries.
It does not recompute an axis after removing the pilot. Question identity uses
the readiness population's whitespace-collapse and case-folding policy, so
different IDs with matching text are also excluded. Repeated eligible text is
deduplicated by stable candidate-ID order before selection. The manifest records
these counts and the exclusion-ID hash. Analysis weights refer only to this
eligible population, not the entire 26,009-row population.

Required CLI arguments are `--selection-root`, `--output-dir`, and
`--source-git-commit`. Optional `--prompt-count`, `--axis-bins`, and
`--master-seed` default to 120, 20, and 20260916. Prompt count must be a multiple
of axis-bin count to remain compatible with the existing inference loader's
balanced-selection contract. The preparer writes `pilot-prompts.jsonl`,
`pilot-axis.jsonl`, `selection-records.jsonl`, and `selection-manifest.json`.
No allocation, model load, or inference occurs during preparation.

Pass `--exclude-cohort-root` for each additional prior new-prompt cohort, including
the 120-prompt paired trial. The option is repeatable. Each cohort must refer to
the same frozen population, and its prompt, axis and selection-record hashes must
verify. Exclusions use both IDs and normalized text and retain provenance.
`--expected-excluded-count` still checks the original 500-prompt selection.

### Approved four-job generator launch

`submit_agentic_generator_backlog.py` prepares a separate backlog from a supplied
frozen cohort. It never extends an existing queue or reruns the historical trial.
The current approved layout is two Qwen3.8 jobs and two Llama4 jobs, each with one
node, four GH200 GPUs, 32 CPUs, 512G memory and `03:00:00` wall-time. Two arrays
with tasks `0-1` create four allocations in total, capped at 48 GPU-hours.
There is no automatic resubmission or requeue.

For this launch, freeze 1,200 prompts after excluding the original 500 and the
later 120-prompt cohort. This gives each model 14,400 cells, enough backlog to
exceed its estimated two-job capacity. At the earlier steady rates, each Qwen
job may save about 3,600–3,900 cells and each Llama job about 5,500–5,800, assuming
5–15 minutes of startup and a two-minute admission margin. Queueing, slower
prompts, retries and security failures can reduce that yield. Completion of the
entire queue is not promised.

Preparation is the default. Submission requires `--submit`, the approved budget
environment variables, and a successful CPU-only private-network check. Each
compute-node serving stage also isolates itself before starting the model.
No JSC document or manually asserted security flag substitutes for these checks.
See [endpoint security](inference_endpoint_security.md).

The helper validates both models before the first `sbatch`, records submission
intent durably before contacting Slurm, and rejects a repeated submission even
after an ambiguous timeout or partial acceptance. Inspect recorded job IDs and
Slurm state before requesting any separately approved recovery. Do not delete
submission records to make the command run again.

For the historical 120-prompt trial, both generator models use the same frozen
1,440-cell task queue. Keep their wave output directories and completed-result
roots separate. Generator cell IDs describe prompt and factorial identity, not
model identity. Completed Qwen cells must never be supplied as completed Llama
cells. A one-hour trial may finish only part of each queue; artifact completion
and model-to-model quality comparisons are separate checks. The existing runner
executes its selected cells in canonical prompt/factor order. A balanced planned
cohort therefore does not imply a balanced subset at the one-hour cutoff. Audit
the completed prompt and factor mix before making comparisons.

## Recovery guarantees

Generator workers write one immutable result and trace per completed cell and
atomically refresh their manifest. Judge workers append and `fsync` every model
attempt and outcome, and atomically refresh their manifest after every task.
Each worker has its own output directory, so parallel workers never share a
journal or manifest writer.

After interruption in backlog mode, reuse the frozen queue and shared claim
root with a new job output directory. Completed task identities remain unchanged;
only uncommitted tasks are executable. The generic wrapper creates per-job output
directories under `outputs/attempts/` so different allocations never share a
manifest writer. Use the durable registry or deduplicate compatible task
identities when reporting; summing per-job copied artifacts overcounts progress.
The old paired-trial report only describes its original fixed worker outputs.

Legacy outputs outside that registry still need explicit compatibility checks and
import/exclusion. The generic wave preparer's completed-ID scan is not a
scientific-provenance audit. Keep completed-result roots model-specific, and
never infer compatibility merely because cell IDs match. No existing datasets or
completed files are rewritten to migrate a queue automatically.

### Shared claims across overlapping runs

Version-2 generic wave workers require one absolute shared
`GEODML_INFERENCE_CLAIM_ROOT`. Generator workers pass it to
`--shared-claim-root`. Model identity, revision, cell content and scientific
configuration distinguish generator claims, while output paths, worker count and
Slurm job IDs do not. Success records contain the result, trace, diagnostics and
original producer provenance. An unfinished call interrupted by the deadline
leaves no terminal failure record, so a later job can take it.

New judge batch workers require `GEODML_JUDGE_CLAIM_ROOT`. Set it to the same
durable directory for every related judge queue and allocation, including later
waves. A different path creates an independent registry and cannot prevent
duplicate work. Do not put the registry in job-local temporary storage or delete
its lock files while jobs can run. Old pinned jobs do not gain this protection
when the source checkout is updated.

The judge runner accepts `--claim-root`, `--worker-index`, `--worker-count`, and
`--dispatch-mode backlog`. For a shared canonical task queue, the preferred slot is
`int(SHA256(judge_task_id), 16) % worker_count == worker_index`, with zero-based
indices. Every task has exactly one preferred slot for a given worker count, but
all tasks remain eligible in backlog mode. This assigns judgment cells, not whole
prompt groups: one prompt can have multiple factorial cells. The Nemotron queue
wrapper defaults to backlog mode; `GEODML_JUDGE_DISPATCH_MODE=partition` explicitly
retains the old fixed-slot behavior. Version-1 generic wave workers use the
default single runtime slot because their queues were already partitioned.

Immediately before an inference call, the worker acquires a nonblocking lock for
that exact task, judge model/revision, protocol, and request fingerprint. Worker
count, output directory, and queue path do not change this identity. A busy task
is left pending. A committed result is verified and reused without another model
call. A new result is flushed, synced, and atomically committed to the shared
registry before the worker journal is updated. Each worker still owns a separate
manifest and journal. Reused results retain the original producing job,
invocation, and code revision as well as the consuming invocation's metadata.
Count unique compatible judgment identities, not the sum
of copied outcomes from multiple worker journals.

Changing the worker count in a later allocation can redistribute the same queue;
the shared registry protects completed tasks even if old and new assignments
overlap. Use a new worker output directory when changing slot parameters. There
is no lock expiry or timeout-based stealing. Process exit releases the lock.
After a crash before durable commit, an unfinished call may need to be repeated:
this is not an exactly-once guarantee for remote HTTP execution. Committed
outcomes are not regenerated. Invalid committed records stop the worker rather
than silently triggering replacement inference.

After bounded attempts fail, backlog workers persist a separate immutable
`*.failed.json` record. Later allocations report the failure without repeating
the model calls. Failed tasks are never counted as completed. Cancellation and
allocation deadlines do not create these records. Do not delete failure records
or change scientific settings simply to consume the remaining allocation.

These guarantees require all writers to use the shared registry on a filesystem
with coherent cross-node `flock`, atomic rename, and `fsync`. Multiprocess local
tests do not verify JUPITER's filesystem semantics; verify those before expanding
to simultaneous nodes. Never mix uncoordinated legacy writers into the same
active queue. Existing legacy journals must still be verified and excluded by
the preparer; the registry does not automatically discover unrelated directories.

### Recovered generator answers

Generator repair policy `evidence-projection-and-malformed-prefix-v2` also covers
malformed reactive finish actions. Only the final bounded attempt can recover an
explicit `finish`, a complete ranking, and a nonempty model-written answer prefix.
Malformed search actions, incomplete rankings, and ambiguous extra fields remain
failures. The raw response and `controller_repair` event stay in the trace, with
`malformed_json_recovered`, `recovered_answer_quote_observed`, and
`answer_truncated` flags. A completed artifact can therefore contain a recovered
answer prefix; it is not evidence of answer quality or a complete original answer.
The existing token and answer-length limits are unchanged.

The runner accepts an exact matching v1 configuration from `0b8902f` or `f301dc5`
for a logged resume migration. It preserves completed results, traces, and
diagnostics and executes only missing cells. Other configuration changes still
fail the resume check.

For an existing legacy shard, `run_agentic_search_qwen38_resume.sbatch` resumes
in place through the original Qwen launcher. Keep its full prompt population,
shard assignment, and concurrency settings; the runner skips completed cells.
Do not add a cell-selection file to that legacy configuration. The wrapper
clears that override, binds the expected job ID to the new batch job, writes an
allocation record, and uses job-specific server and GPU logs in
`GEODML_RESUME_ATTEMPT_DIR`. Submission must supply the approved wall-time,
resources, `GEODML_APPROVED_WALLTIME`, and `GEODML_ALLOCATION_ESTIMATE`; the wrapper
does not allocate resources itself. An advisory output lock rejects concurrent
resume wrappers. Older launchers do not acquire this lock, so first check that
no older job is writing the shard. New parallel work should use the separate
wave worker directories.

## Judge roles

### Allocation-filling Nemotron queue

For new throughput runs, use `prepare_agentic_judge_pilot.py --all-available`
and `run_nemotron_judge_queue.sbatch`. The preparer freezes every complete prompt
group in the selected completed source shard instead of selecting two prompts.
Repeat `--exclude-outcomes PATH` for existing compatible Nemotron journals.
Exclusions validate the prior plan, seed, model revision, resume identity,
requests and saved outputs; overlapping or conflicting coverage is rejected.
The new queue records available, excluded and pending counts and retains the
exclusion provenance. Existing judgments are not rewritten or regenerated.

The queue wrapper uses the same pinned model and judgment schema as the earlier
24-case pilot. It accepts the separately approved wall-time, checks the actual
allocation deadline and handles intentional checkpoints. This is throughput
plumbing, not judge-quality validation; outputs remain `scientific_result=false`.
It never submits another job. A new queue/attempt must not reuse the old pilot
directory.

For parallel Nemotron jobs, point `GEODML_JUDGE_QUEUE_ROOT` at one frozen queue
containing `plan/`, and set the common `GEODML_JUDGE_CLAIM_ROOT`. Set
`GEODML_JUDGE_WORKER_COUNT` and each zero-based `GEODML_JUDGE_WORKER_INDEX`, or
use a zero-based contiguous Slurm array with the same number of slots. The
wrapper creates separate `attempts/job<job-id>-worker<index>/` directories for
serving profiles, logs, journals, and manifests. The canonical plan stays shared
and read-only. Submission stdout/stderr default to queue-level
`logs/slurm-<job-id>.out` and `.err`; matching submission paths must be supplied.
Empty or invalid slots fail before loading a model. Every additional allocation
still requires a runtime estimate and explicit wall-time approval.

### Bounded Nemotron plumbing pilot

`prepare_agentic_judge_pilot.py` prepares a separate Nemotron-only queue from one
completed Qwen shard, without regenerating or modifying its outputs. The default
pilot selects two prompts from the lowest and highest available axis bins, with
seeded tie-breaking, and includes all 12 factorial cells for each prompt. It
checks the selected result and trace hashes and retains the private source
mapping. This 24-case sample tests the pipeline, not representative judge quality.
The full-coverage production preparer is unchanged.

`run_nemotron_judge_pilot.sbatch` serves the cached BF16 model at revision
`bf77c3174f68ad409e1c2aa60daeb46e32d1c606` and runs only that queue. Submission
must supply the separately approved `01:00:00` wall-time and one node with four
GH200 GPUs, 32 CPUs, and 512G memory. The wrapper records the approval, estimate,
commit, environment versions, allocation resources, and terminal status. It
checks the cached shards and JSON schemas before loading weights. Eager serving
and reasoning-off are fixed for this short pilot; they are not a throughput
benchmark or a judge-accuracy recommendation.

Set `GEODML_JUDGE_PILOT_ROOT` to a new directory containing `plan/` from the
preparer. Also export `GEODML_EXECUTION_REPOSITORY`, `GEODML_EXECUTION_COMMIT`,
`ACL_ARR_VENV`, `GEODML_CACHE_ROOT`, `HF_HUB_CACHE`,
`GEODML_APPROVED_WALLTIME`, and `GEODML_ALLOCATION_ESTIMATE`.

The worker writes `nemotron/outcomes.jsonl`, `attempts.jsonl`, `failures.jsonl`,
and an atomically updated `run_manifest.json`. Each completed judgment is
flushed and synced. Outcomes include both an ideal evidence-relevance ranking
and a ranking of evidence supporting the existing answer, plus fulfillment and
grounding scores. Join evidence IDs to URLs using the matching task in
`plan/bulk_tasks.jsonl`. All pilot outputs remain `scientific_result=false`.

The wrapper rejects a reused attempt directory and takes an exclusive pilot
lock. Its underlying runner supports missing-task resume, but another allocation
or attempt requires fresh approval and a separately prepared launch. Do not
repeat the submission command to monitor it. Inspect `logs/slurm-<job>.out`,
`logs/slurm-<job>.err`, `logs/server.log`, and `logs/gpu.csv` instead.

No GLM model is invented, downloaded, or launched by this pilot. A later paired
comparison can reuse the frozen public tasks after its own model/runtime checks
and allocation approval.

### Bulk and validation

By default, the bulk judge sees only the request, independently ordered evidence,
and the answer. It does not see generator identity, search method, engine, condition, or
the generator's ranking. It independently produces an ideal relevance ranking
and a realized answer-support ranking.

The validation judge receives the same blind task representation on a frozen,
stratified subset. Predefined low-confidence and disagreement rules route
additional cases to validation. A judge never sees another judge's output.

Both judge preparers accept `--recorded-conversation` for a separate v2 protocol.
It includes recorded LLM prompts and responses, schema retries and repairs, and
the compacted tool observations shown to the generator. Each LLM input records
its visible URLs, original S IDs, one-based input positions, and corresponding
judge E IDs. The generator's final ranking is visible, and the workflow may be
inferred; this mode is not ranking-blinded. Discarded and pre-ablation retrieval
results are not supplied as evidence the generator saw. These are saved search
snippets, not full web pages: no live text is fetched, hidden reasoning is not
available, and lower-level transport retries may be absent. Recorded turns are
never truncated. The Nemotron wrapper checks the exact pinned local chat
tokenizer plus the 2,048-token output budget against its 16,384-token context
before loading the model. Oversized tasks stop the launch with their IDs rather
than silently dropping turns. Other serving profiles must perform the same
capacity check with their own tokenizer and context limit.
V2 binds both case and task IDs to its protocol and conversation hash. The old
24-case v1 pilot remains a different protocol and cannot satisfy v2 completion
or adjudication coverage. Prepare a separate output plan; retain the original
source-cell mapping when joining results across protocols.
