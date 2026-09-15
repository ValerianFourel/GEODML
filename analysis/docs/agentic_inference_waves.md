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

## Judge roles

The bulk judge sees only the request, independently ordered evidence, and the
answer. It does not see generator identity, search method, engine, condition, or
the generator's ranking. It independently produces an ideal relevance ranking
and a realized answer-support ranking.

The validation judge receives the same blind task representation on a frozen,
stratified subset. Predefined low-confidence and disagreement rules route
additional cases to validation. A judge never sees another judge's output.
