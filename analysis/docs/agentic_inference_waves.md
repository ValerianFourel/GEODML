# Restartable agentic inference waves

The inference pipeline has three independent task queues:

1. generator cells for Qwen3.8 and Llama 4;
2. blinded bulk judgments;
3. blinded validation and adjudication judgments.

Each queue uses the same wave mechanism. A wave freezes the tasks that are still
missing, orders their stable IDs with a seeded hash, and assigns position `i` to
worker `i % worker_count`. Workers therefore have disjoint queues and differ in
size by at most one task.

Changing `worker_count` in a later wave is safe. The planner first scans all
completed result roots, removes those IDs, and then applies the new modulo only
to the remaining IDs. Never reuse a wave directory for a different plan.

## Components

- `prepare_agentic_new_cohort.py` freezes a shared new-prompt cohort, excluding
  the original 500-prompt selection by both ID and normalized question text.
- `prepare_agentic_generation_tasks.py` freezes generator cell IDs.
- `prepare_agentic_judge_tasks.py` freezes blind bulk and validation queues.
- `prepare_agentic_adjudication.py` applies predefined disagreement rules and
  produces additional blind validation tasks.
- `prepare_inference_wave.py` creates any number of disjoint worker queues from
  one of those sources.
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
It cannot invent new scientific tasks to fill the remaining minutes. Static
modulo workers do not steal another worker's queue; prepare sufficiently large
queues and rebalance missing IDs in the next approved wave. Full allocation usage
is a throughput objective, not a guarantee of 100% instantaneous GPU utilization.

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

Both generator models should use these exact prompt files and the same frozen
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

After interruption, build a new wave from the original queue and pass every
prior wave output root as a completed-results root. The next wave contains only
missing task IDs. This works with a small number of long jobs or a larger number
of short jobs without changing the scientific task definition.

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
directory. The generic wave mechanism remains the route for parallel judging
and modulo redistribution of missing tasks.

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

The bulk judge sees only the request, independently ordered evidence, and the
answer. It does not see generator identity, search method, engine, condition, or
the generator's ranking. It independently produces an ideal relevance ranking
and a realized answer-support ranking.

The validation judge receives the same blind task representation on a frozen,
stratified subset. Predefined low-confidence and disagreement rules route
additional cases to validation. A judge never sees another judge's output.
