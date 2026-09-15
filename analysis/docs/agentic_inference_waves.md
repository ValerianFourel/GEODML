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
