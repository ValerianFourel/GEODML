# Queue the first Gemma wave before preparation finishes

Use `horeka-gemma-v4-prequeue.sh WORKSPACE EXISTING_RUN` from a HoreKa login
shell. Valerian explicitly authorized replacing the `gemma-v4-bouts` tmux
controller and submitting 200 jobs immediately. The script replaces only that
named login-side session. It preserves preparation job 5177483, Qwen jobs and
interactive allocations. Use a clean, committed helper checkout and keep the
original scientific checkout available.

The adapter submits 200 individual five-hour `accelerated` jobs without spacing.
Each requests one exclusive node, four A100 GPUs, 32 CPUs and all node memory.
They have an `afterok` dependency on the recorded preparation and a user hold.
They appear in the queue but take no nodes until preparation finishes, exact
inputs are verified, private HF ownership is reserved, fresh quota checks pass
and durable shard receipts are written. The adapter then releases assigned jobs.
It does not bypass scheduler priority or make preparation start sooner. Held jobs
do not establish useful priority age. Slurm may refuse the requested queue size.

Historical warm throughput gives 835–983 reported-complete cells per five-hour
bout, including an 8.9–10.9-minute startup and five-minute drain. This is a
capacity estimate, not a guarantee of semantic quality or five hours of useful
work in every shard. The first wave has a 1,000-node-hour / 4,000-GPU-hour maximum.
Only jobs assigned to verified nonempty shards are released; surplus held jobs
are cancelled. These first bouts consume the original frozen per-shard budgets,
not an additional wave on top of them. Preparation retains its original one-hour
ceiling, one node, four GPUs and 32 CPUs. The cheaper scheduling alternative is
the original controller, which waits for preparation before submitting anything.
The preparation workload duration remains unmeasured at full scale.

Every 600 seconds the controller checks preparation and retries explicit
submit-count rejections. It uses a seven-day preparation deadline inherited from
the existing run. Generic errors or unresolved submissions stop with evidence;
there is no blind resubmission. When preparation succeeds, the adapter adopts
queued jobs into native sender receipts and invokes the unchanged finite sender.
That sender continues ten-minute monitoring/refill under the original thirty-day
deadline and allocation limits. Failed preparation retires only known held jobs
from this wave. Errors that cannot safely establish job identity leave jobs
unchanged and require inspection.

Look at `prequeue.log` and `prequeue/status.json` before activation, then
`sender/status.json`. First-wave Slurm logs are under `prequeue/slot-*/slurm-*`;
scientific results remain under `shards/shard-*/results`. Subsequent bouts use the
original shard log paths. The original `sender.log` is historical after this
replacement. Re-running the same pinned helper recovers recorded submissions;
do not create a new run root to bypass an error.

Validation uses Slurm wire fixtures, real durable submission files, the real
worker binding validator, and the original sender. It does not establish live
HoreKa acceptance, timing, quota capacity or scientific validity.

## Explicit login preparation takeover

Valerian subsequently requested preparation on the login shell immediately.
The same shell entry point accepts `--prepare-on-login` as its third argument.
This is an explicit exception for saved-input preparation only; inference still
uses the original pinned four-A100 Slurm workflow.

The takeover holds the pending preparation job, verifies the accepted first-wave
jobs remain user-held, clears their dependencies, and only then cancels the
pending preparation job. It requires terminal cancellation accounting before
scanning inputs. If preparation has already started, it preserves that allocation
and stops. All scheduler changes have durable receipts. The first-wave jobs stay
held through preparation and are adopted/released using the same existing path.

Preparation runs as one nice-priority login process, with native-library thread
counts set to one and a one-hour execution timeout. Full-dataset duration and peak
RAM remain unmeasured; it loads no judge model. It retains storage/quota checks.
Partial preparation is preserved and never blindly overwritten or restarted.
An interrupted controller can continue from a fully prepared plan or from a
retired pending preparation with no partial outputs. It fails for unknown state.

The plan records the new preparation helper revision and login host separately
from the unchanged original inference checkout/revision. Scientific settings,
dataset sources and exclusions stay unchanged. The original conservative budget
still includes the old one-hour preparation ceiling; actual cancelled GPU use and
login preparation are recorded separately. No new GPU preparation job is sent.
The command replaces the existing named tmux controller and writes to prequeue.log.
