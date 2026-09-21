# Adaptive inference runtime diagnostics

Use `analysis/scripts/slurm/jupiter/collect_agentic_runtime.sh` from a separate
JUPITER login shell. Its positional arguments are mode, existing job ID, experiment
run root, and a new report directory outside the experiment root. Modes are
`collect`, `profile`, and `summary`, in that order. The Python entry point also
accepts `--duration` between 0 and 600 seconds for testing or shortened collection.

No inference settings, claims, serving profiles, or running code are modified.
Every monitoring step explicitly specifies the existing job ID, overlap, one CPU
per node and 1 GiB memory per node. Monitoring introduces some CPU/filesystem
overhead. It does not reserve additional nodes or GPUs. Preserve the allocation's
owning shell. Do not use the older exclusive boundary diagnostic during inference.

## Evidence collected

- `allocation.json`, `provenance.json`: controller allocation fields, observed
  experiment revision, collector revision, and collector-environment package
  versions. These versions are not proof of packages imported by an existing client.
- `baseline.jsonl`: roughly minute-spaced unique claim counts, all-saved cell
  latency median/p95, worker materialization counts, selected config settings,
  and bounded log metrics. Logs are reduced to counts and throughput fields;
  prompts, tokens, full command lines and arbitrary error text are not exported.
- `node-*.jsonl`: five-second GPU readings, identified clients' CPU-time deltas,
  thread counts, RSS, affinity and cumulative I/O counters. PID start time prevents
  calculating deltas across PID reuse. Missing clients are not proof of inactivity.
- `summary.json`: measured completion rate, conditional generation-only ETA,
  per-worker materialization deltas, sampled resource means and node coverage.
- `profile-status.json`, optionally `cpu-profile.json`: a separate 30-second,
  25 Hz py-spy profile of one client, selected by lowest materialization delta.
  This proxy is used because current claim producer metadata lacks worker identity.

The collector parses each unchanged claim once and retains only a compact summary.
Counts are structural observations, not claim-digest or scientific validation.
Duplicate cell IDs count once within a role. Worker materialization is not a
reliable production count: a worker can materialize a claim another worker wrote.
Failed/malformed records are reported, not retried or repaired. Time is Unix epoch
in collector records; server log timestamps retain the cluster timezone implicitly.
Log reads are limited to 64 KiB per file per sample and mark truncation/rotation.
The first log sample is historical context, not activity newly observed in the window.

No step is launched when the job is not running and owned by the current user.
Collection is capped at the controller deadline minus 60 seconds; profiling needs
90 seconds remaining. Unknown deadlines disable live steps. Step failures still
leave saved-artifact diagnostics. A missing or restricted profiler is a recorded
limitation, not a reason to use sudo, modify ptrace permissions, install packages,
or restart a worker. CPU-stack samples include idle threads, so interpret frame
frequencies together with CPU deltas, not as exact wall-time stage percentages.

## Interpretation and follow-up

1. Confirm saved unique cells increase and failures do not increase. Compare at
   least ten minutes of steady-state data, not one GPU snapshot. Separate cold
   startup and any role transitions visible in worker configs/logs.
2. Correlate empty vLLM queues with CPU activity and sampled stacks. The current
   code uses a CPU cross-encoder and holds the memoization lock during scoring.
   This is a hypothesis to measure, not automatic proof of the limiting stage.
3. Distinguish scoring/lock waits, retrieval/tokenization, storage/claim waits and
   a genuinely saturated LLM server. Summing concurrent request durations does
   not yield an exact cell wall-time breakdown.
4. If evidence is insufficient, add opt-in stage timers in a separately tested
   revision. Never signal a running Python process to dump stacks without a
   confirmed signal handler. Never hot-patch live scientific workers.
5. Optimize only the measured limiting stage. Keep the existing PyTorch path;
   evaluate thread counts, scheduling or batching before a GPU port. GPU compaction
   needs a measured peak-memory budget and comparison with the current CPU scorer.
   The model already occupies most GPU memory. Do not change precision, evidence
   selection, model revision, ranking order, seeds or claim identity silently.
6. Benchmark one change at a time on an isolated frozen workload with matched
   methods/conditions/input sizes, warm-up separated, synchronized GPU timing and
   end-to-end successful-cell throughput. No inference benchmark or new allocation
   is started by these diagnostics. Obtain separate runtime/resource approval.

## Method reference

The requested [optimize-for-gpu skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/optimize-for-gpu/SKILL.md)
informed profiling before porting, numerical-contract preservation and end-to-end
measurement. No GPU optimization or speedup is claimed by this diagnostic release.

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
