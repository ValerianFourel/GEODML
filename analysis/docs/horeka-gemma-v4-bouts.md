# Gemma v4 dataset judging in five-hour bouts

Valerian authorized five-hour bouts, up to 200 Gemma jobs queued together, no
artificial start spacing, and a tmux sender that checks every ten minutes. This
overrides the usual one-hour and five-allocation limits for this finite pass.
Other allocations are preserved. Slurm still controls starts and account limits.
Fifteen running jobs is the runtime-estimate assumption, not a launcher throttle.

The existing Gemma v4 r2 instructions, BF16 model revision, 73,728-token context,
4,096 output-token cap, four-way tensor parallelism, four concurrent requests,
temperature zero and disabled thinking are retained. Each worker runs answer
mapping and source-importance judging. Fulfilment, generation, v3 comparisons and
repeated validation passes are not requested. Earlier diagnostic cohorts named
with `--exclude-inputs` are preserved separately, including their known failures.

The [measured cost calculation](si-v4-gemma-five-hour-cost-20261004.md) projects
8.8–10.4 days for a hypothetical 624,192 eligible cells at an average of fifteen
running four-A100 nodes. That is about 635–748 ideal pooled five-hour bouts,
3,175–3,740 node-hours and 12,700–14,960 GPU-hours. Counts in the historical task
registry are not a census of currently saved answers. Actual input counts and the
finite budget appear in the prepared `plan.json`.

Each inference allocation requests one exclusive A100 node, four GPUs, 32 CPUs,
all node memory through `--mem=0`, and `05:00:00` on `accelerated`. The first 200
full bouts can use at most 1,000 node-hours and 4,000 GPU-hours. Model startup is
estimated at 8.9–10.9 minutes and drain at five minutes. Five-hour throughput is
extrapolated from short runs and remains unmeasured. Longer bouts would reduce
startup further but are outside this authorization. Using fewer nodes lowers
concurrency without materially reducing the work's total GPU cost.

Start from a clean pinned checkout in an already-open HoreKa login shell after
sourcing the existing `geodml-nemotron-env.sh`. The runtime, pinned Gemma cache
receipt and private Hugging Face authentication must already work. This workflow
does not download a model or modify the environment.

```bash
bash "$CODE/analysis/docs/horeka-gemma-v4-bouts.sh" \
  "$W" "$DS" "$W/llama-hf/dataset" \
  "$W/reviews/gemma-si-v4-dataset-5h-20261004" \
  hk-project-p0026831 \
  "$W/reviews/si-v4-r2-cycle-20261002/development/candidate-inputs"
```

The first allocation is a bounded **one-hour GPU preparation job**, with the
same four-A100 node shape. Its additional ceiling is one node-hour/four GPU-hours.
Its census duration is not yet measured; a preparation timeout stops the sender
with partial artifacts intact. It verifies sealed generator references, recovers
complete saved answers, freezes their exact source records and divides ready
cells into at most 200 durable shards. It uses the currently supplied dataset
copies. It does not fetch later generator completions or silently add them to
the frozen pass. Input problems and unavailable generators remain in the census.

Cells sharing an answer-map task remain together. Each shard has one persistent
result directory and one active owner, so resumed bouts keep the map and source
judgments already saved. Known inference failures are terminal for this pass.
If a different cell shares a map from the named earlier diagnostics, its complete
input is retained as `prior_map_requires_reconciliation`. It receives no duplicate
inference and is reported separately for later result reuse.
An unfinished shard receives a replacement allocation only after Slurm confirms
the previous allocation ended and saved results are reconciled. An ambiguous
submission is reconciled by its unique scheduler comment before another submit.

The preparer computes a finite allocation ceiling for each shard from its cell
count, 20.399 measured seconds per completed cell, 654 startup seconds and 300
drain seconds. Near the full 624,192-cell target, balanced 200-shard preparation
typically permits four bouts per shard: at most 800 inference allocations,
4,000 node-hours and 16,000 GPU-hours, plus preparation. This per-shard ceiling
is greater than the ideal pooled forecast because final partial bouts also have
five-hour reservations. They finish early when their work is exhausted. There
is no automatic budget expansion if the estimate proves insufficient.

Before submitting inference, the sender reserves map identities through a
conflict-checked update in the existing private HF dataset. This SI reservation
is separate from generator/shared-hour ownership. Ownership never expires on a
network failure, and scheduler success never marks scientific work complete.
Compute jobs use the local frozen inputs and cached model. Results remain in the
workspace; the reservation does not publish results or establish acceptance.

Fresh quota, byte and inode evidence gates submissions and task admission.
Runtime quota checks refresh every two minutes and stop new tasks if verification
fails. The serving boundary remains authenticated and loopback-bound on a
verified exclusive node. Actual Slurm end time governs draining, including model
startup. Existing live allocations are neither cancelled nor extended.

```bash
(
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  RUN="$W/reviews/gemma-si-v4-dataset-5h-20261004"
  tail -n 50 "$RUN/sender.log"
  test ! -f "$RUN/sender/status.json" || cat "$RUN/sender/status.json"
  squeue --me --name=geodml-gemma-v4-bout,geodml-gemma-v4-prepare
)
```

The sender has a thirty-day deadline and the frozen allocation ceiling. Restart
the same command and output directory to resume its bookkeeping. Do not create
a fresh output to bypass an unresolved ownership or submission error. Inspect
the recorded reason and preserve all existing jobs and artifacts.

The Gemma semantic review remains unaccepted. Pipeline completion is an
operational count, not proof that the judgments meet v4's substantive standard.
The [Nemotron comparison](si-v4-nemotron-semantic-review-20261004.md) also rejects
the tested replacement configuration, which produced no numeric grades.
