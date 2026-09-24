# Shared hours on JUPITER and HoreKa

The shared-hour path is optional. Existing segment plans and JUPITER launchers
keep their scientific behavior. Maintained inference entry points now require a
verified exclusive Slurm node on JUPITER. The entry point is `analysis/scripts/manage_agentic_hours.py`.
Its `--help` lists the commands. No command submits an allocation, automatically
resubmits a job, or releases an interactive shell.

## What an hour means

An hour is an immutable, homogeneous list of task fingerprints sized against one
JUPITER node with four GH200 GPUs. It is not an allocation or a completion claim.
An A100 allocation may finish part of an hour; another separately approved
allocation can finish the remainder on either cluster.

Each plan freezes keyword priorities, task definitions, scientific configuration,
input-bundle identity, and reference timing. Remaining cells of the same prompt
stay together. A group exceeding the reference budget is explicitly marked
`oversized_prompt_group`; the planner does not change its scientific settings.

Reference timing has three separate quantities: startup seconds, drain seconds,
and steady allocation-wall-clock seconds per validated task. The latter must
account for concurrency, rather than summing overlapping request durations.
Package cost is startup + drain + task count × seconds per task. Startup and
drain are counted once. These are estimates, never an allocation guarantee.

`inspect-inputs --dataset-root PATH` reports configuration identities and verified
completion counts. Build a calibration JSON object keyed by `qwen38`, `llama4`,
and `nemotron` for every model with eligible work. Each entry requires:

| Field | Meaning |
| --- | --- |
| `cluster`, `gpu_type`, `gpus` | `jupiter`, `GH200`, `4` |
| `scientific_config_sha256` | Configuration key reported by `inspect-inputs` |
| `reference_profile_sha256` | SHA-256 of the frozen JUPITER serving-profile file |
| `seconds_per_task` | Measured steady wall time divided by validated completions |
| `startup_seconds`, `drain_seconds` | Separately measured or explicitly justified overhead |
| `evidence` | Reference to the measurements, allocation IDs and estimation assumptions |

Do not copy historical throughput claims into this file without checking their
model, revision, workload and concurrency. Missing or mismatched calibration is
an error. Nemotron registrations lack the generator configuration field; inspection
derives a planning-only key from their pinned model, protocol and judge-plan ID.
The original registered rows and claim fingerprints remain unchanged.

## Host configuration and approval

Copy the templates under `analysis/config/shared_hours/` outside the source tree.
They intentionally contain no model-compatibility claims or allocation approval.
Fill account, partition, environment-file and scratch paths for the actual site.
The site environment script activates an architecture-compatible Python/vLLM
environment and configures already prepared model caches. It must not allocate
resources. `cache_root` can contain an allocation environment variable such as
`${SLURM_TMPDIR}`; execution rejects unresolved variables.

Add entries to `validated_models` only with compatibility evidence. Each entry
contains `evidence` and `reference_profile_sha256`. For example, a `qwen38` entry
refers to a tested serving configuration preserving the frozen reference profile.
Do not assume that every pinned model fits on four 40 GB A100s. An incompatible
model stays on JUPITER. No automatic quantization, CPU offload, context reduction
or multinode fallback is implemented.

The local serving profile is prepared with the existing `search_vllm_stage.py`
profile mechanism. Its hardware pattern is `GH200` or `A100` for the selected
cluster. Model identity, precision, context, topology, request concurrency and
serving features must match the frozen reference. Runtime task identity checks
also reject different prompts, seeds, evidence, token budgets or judge protocols.

The separate runtime-environment JSON supplies the existing leaf-worker variables.
Generator runs require the profile, compactor snapshot/revision, both search
snapshots, frozen prompts/selection records and prompt count under their existing
`SEARCH_AGENTIC_*` names. Preserve the approved request/cell concurrency values.
Judge runs use `GEODML_JUDGE_MANIFEST`, `GEODML_JUDGE_ROLE`,
`GEODML_JUDGE_PROFILE`, and the frozen concurrency, output-token and thinking
settings. All values are strings. The CLI rejects other keys so inherited output
paths or legacy queues cannot enlarge the reserved work.

The request template specifies a unique attempt ID, cluster, mode, ordered finite
`hour_ids`, external attempt directory, scheduler-history `since` date and full
committed Git SHA. The first hour runs first. Spillover follows keyword priority,
forward for batch and reverse for interactive. All packages use one model.
Choose enough approved packages to exceed the allocation's estimated capacity.

An approval records status, walltime, resources, exact GPU-hour cap, runtime
estimate and evidence of Valerian's approval. Supported attempts use one node,
four GPUs and at most 3600 seconds. The template remains `proposed` with null
time/budget until that specific allocation has been approved. Finishing or
checkpointing an attempt never approves another allocation.

## Preparation and execution sequence

1. Reconcile the existing JUPITER run before migration. Stop admitting independent
   legacy workers over the same task population. Preserve existing allocations
   until they end. `upload-inputs` requires a fresh JUPITER scheduler snapshot
   proving no legacy allocations remain. Use the existing scheduler capture tool
   and add the explicit `cluster: jupiter` field to its JSON.
2. Ensure the private HF dataset already exists and the login/transfer host has
   access through its saved HF login or `HF_TOKEN`. The tool does not create or
   change repository visibility. Use global `--repo-id` and a durable `--journal`
   outside the Git checkout for network commands.
3. `upload-inputs` publishes the sealed dataset and returns `input_bundle`.
   Include required frozen search evidence, prompts and configurations under the
   dataset's allowed artifact directories before this operation. Model weights,
   live shards, credentials, local claims and forensic-only files are excluded.
4. On JUPITER, `plan --cluster jupiter` accepts the dataset root, calibration, input bundle and source
   commit. It defaults to a proposal. `--output` saves an immutable local plan;
   `--publish --operation-id ID` publishes it with a conditional registry commit.
   Publishing requires `--output`. After a lost publication response, use
   `publish-plan --plan PATH --cluster jupiter --operation-id ID` with that saved
   file and the original operation ID. Do not regenerate the plan to retry it.
5. On either cluster, `reserve` accepts the request, cluster profile, runtime JSON,
   reference profile, dataset root and repository. It downloads verified inputs,
   imports prior published outcomes and reserves the listed hours atomically.
   It writes `attempt.json`, `tasks.json`, and `backlog.jsonl` to the request's
   attempt directory. Save and reuse the operation ID after an ambiguous response.
6. `admit --attempt PATH --quota-evidence PATH --operation-id ID` previews admission.
   `--apply` records the approved ticket. For a new allocation it returns a safely
   quoted allocating command. Execute that command once. If acceptance is
   ambiguous, reconcile the recorded comment in Slurm before doing anything else.
7. For an already open interactive allocation, use `admit --existing-job-id ID`
   with the same approval and `--apply`. Run the shared-hour shell wrapper as a
   child step with an explicit `srun --jobid`. The wrapper takes `attempt.json` as
   its sole argument. Keep the allocation-owning shell open.
8. Start `sync` on the networked host with the attempt, quota evidence and an
   explicit future `--until-epoch`. It polls at most once per configured interval,
   defaults to 30 seconds, and has no authority to submit jobs. `--once` performs
   one transfer/reconciliation pass. The deadline should cover the approved
   allocation and its final transfer; it does not extend the allocation.

Quota evidence is site-specific JSON with `cluster`, `captured_at_epoch`, and
`within_limits`. Refresh it using the site's actual quota tooling while the helper
runs; a snapshot older than five minutes is insufficient. Do not label a small
successful write probe as proof of quota health. The helper checks filesystem
bytes, inodes and incident records in addition to this evidence. It writes a
stop-admission marker when storage evidence is unsafe or unavailable. Review the
incident before preparing a new attempt; the helper never automatically clears it.

Admission counts all live `geodml-*` allocations and explicitly attached job IDs,
including other waves. Keep every GEODML allocation identifiable in Slurm. A
cluster admits at most five allocations and releases only one not-yet-observed
start at a time. The next start waits for the previous observed start plus ten
minutes. JUPITER and HoreKa maintain separate admission state.

## Ownership, recovery and progress

`coordination/hours.json` is the readable current registry. Each row records the
hour's tasks, owner, ownership generation, attempts, verified completed/failed
fingerprints and immutable checkpoint-bundle references. Full definitions live
under `coordination/plans/`. HF conditional commits prevent concurrent writers
from replacing each other's reservations. Operation receipts resolve lost replies.

Workers write one ordered union of their reserved packages using the existing
model worker, retaining its loaded server. Every claimed task must match the
reserved fingerprint. Data writers use cluster/attempt names, retain the compact
striped ledger, and seal shards at five-minute write boundaries and normal exit.
The helper uploads only sealed shards, checks remote bytes and all outcome
references, then commits progress. Repeated unchanged sync ticks transfer nothing.

Network failure retains ownership. The finite helper retries transient network
errors until its deadline, and workers finish only their already reserved work.
A stopped helper is safe to restart with the saved attempt and journal. No claim
expires from age or a missing heartbeat.

An hour can be `available`, `reserved`, `running`, `awaiting_sync`, `partial`,
`blocked`, `complete`, or `superseded`. Complete means every required task has
verified published output and the previous allocation is confirmed terminal.
Failed tasks remain failures. Exit zero is not scientific completion.

After Slurm confirms termination, the helper uses existing terminal-writer
reconciliation to recover sealable shards and release only uncommitted local
claims. Invalid saved references block release. Committed compatible results are
never inferred again. A crash before a durable result commit can repeat an
unfinished model request; this is not an exactly-once physical-inference promise.

Use `release-unstarted --apply` only for a reservation with no admission ticket,
execution receipt or scheduler history. An admitted attempt cannot use that
shortcut, even if no job ID was returned. Its allocation state must be reconciled.

JUPITER's next explicit `plan` synchronizes published checkpoints and repacks only
released unfinished work. It preserves owned assignments and historical plans.
A concurrent claim makes a stale plan publication fail. New plans do not grant
allocation approval. Other clusters can keep executing their owned old-plan hours.

## Storage and validation boundaries

Transfer objects are content-addressed. Per-attempt data files retain immutable
paths and manifests; importing the same bundle twice does not duplicate completion.
Conflicting or corrupt results stop import. Download only the inputs and history
needed by selected packages; planning is an explicit broader synchronization.

Put compile caches on job-local scratch where available. Execution creates a new,
marked cache directory for its attempt and removes only that directory after its
worker process group is gone. Unknown or still-live cache directories are retained.
The tool does not delete scientific data, unsynchronized results, model weights
or lock files. Historical shared-scratch cleanup remains explicit; storage pressure
stops admission rather than deleting evidence to make space. Five-way concurrency and staggered starts reduce
risk but do not establish the cause of historical `ENOSPC` failures.

Tests use synthetic datasets and an isolated revisioned HF store. They verify
ownership races, idempotent operations, interrupted transfers, cross-cluster
continuation, replan preservation, deadlines and storage admission. They do not
establish real HF permissions, cluster filesystem semantics, model fit or A100
throughput. Validate those on the real hosts before scientific execution, under
separately approved allocations where compute is required.


## Mandatory compute-node boundary

JUPITER wave batch workers, existing-allocation interactive launches and shared-hour
execution explicitly select `GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1` after site
setup. Saved wave and hour runtime configurations also record this value.
`verify_inference_allocation.py` calls the existing Slurm verifier, checks the
current host against the allocation, and fails without namespace fallback.
Authentication, loopback endpoints and restricted collective transports remain
required. Unrelated launchers retain their previous boundary policy.

For bootstrap preparation inside an allocation, the maintained entry point is
`analysis/scripts/slurm/jupiter/run_agentic_bootstrap.sh`. It activates the site
environment and verifies the compute-node boundary **before** invoking its
supplied preparation command. Use this wrapper instead of a remote-only bootstrap
script as the allocation's entry point. It does not request an allocation.

The built-in profiles in `analysis/config/cluster_execution/` explicitly select
`execution_boundary: exclusive-slurm-node` for both clusters. Shared-hour attempt
manifests retain that profile. JUPITER permits the existing psslurm full-node
representation; HoreKa requires explicit whole-node evidence (`Exclusive=NODE`
or `Shared=0`) and rejects that JUPITER-specific interpretation. Site compatibility
still requires real validation. Its allocation is separately
verified through Slurm on the A100 compute node, before loading a model.

Namespace execution remains available only by explicitly selecting
`private-network-namespace` in the boundary utility and supplying its `--exec`
command. It is never a fallback for the maintained exclusive-node launchers.

## Resume a prepared bootstrap backlog

`analysis/scripts/run_jupiter_prepared_backlog.py` consumes `--runtime-environment`
(the existing saved string mapping), `--scheduler-snapshot` (complete and at most
120 seconds old), and optional `--ledger-stripes`. Activate the site environment
or invoke it through the bootstrap wrapper. The snapshot must cover the dataset's
owners; unknown ownership remains blocked. The runtime mapping must identify the
existing dataset, wave, output paths, pinned checkout and the newly approved
allocation estimate and wall-time. The existing wave runner checks the immutable
backlog hash and skips verified completed claims during execution.

This entry point performs read-only terminal-owner reconciliation and reports
current registered, verified-completed and blocked counts. If repairs are needed,
it stops for the existing reconciliation workflow. It never registers tasks,
accepts recovery records, rebuilds the backlog or overwrites plans. Plan and
runtime files must be retained from the bootstrap; missing artifacts are errors.

Job 1994448 historically registered 312096 Qwen tasks, accepted 2172 recovered
completions and prepared 309924 remaining tasks. These numbers are not used as
current progress. Its `unshare: ... No space left on device` failure occurred
before inference while entering isolation; this alone establishes neither disk
exhaustion nor GPU OOM. The corrected path has local regression coverage only;
a real corrected JUPITER run is still required for cluster validation.

## Fast update workflow

The shared-hour model policy is Qwen on either cluster, preferably HoreKa;
Llama and Nemotron on JUPITER. Selection requires an explicit model. These are
routing preferences, not compatibility evidence or allocation approval.

Copy `analysis/config/shared_hours/site.template.json` outside the checkout.
Set its dataset root, journal, plan directory, quota evidence and cluster. Its
`attempts` list contains explicit paths to this host's saved `attempt.json` files.
Use the same local dataset root for attempts listed together. HoreKa can also set
`workspace`, `quota_project` and optional `quota_cluster`; update/pull then capture
fresh GPFS quota evidence automatically. JUPITER requires its actual site quota
report in the existing evidence format. Do not manufacture a quota pass.

Set `GEODML_SITE_CONFIG` to that JSON and `GEODML_PYTHON` to the prepared Python.
Call `bash analysis/scripts/slurm/update_agentic_hours.sh` with:

| Arguments | Effect |
|---|---|
| `status` | Read registry and local keyword/bin progress; no publication. |
| `update` | Sync listed attempts, retrieve published results and refresh progress; keep plans. |
| `update --scope plan` | JUPITER: retrieve published results and replan released work. |
| `update --scope both` | JUPITER: publish local results, retrieve remote results, then replan. |
| `pull --model qwen38` | Retrieve pinned plan/input/checkpoint bundles; no reservation. |
| `select --model qwen38 --count 3` | Propose HoreKa Qwen hours without claiming or allocating. |
| `select --model qwen38 --cluster jupiter --mode interactive --count 3` | Propose JUPITER Qwen hours in reverse keyword order. |

The Python CLI supports the same commands with `--site PATH`. Output is a compact
summary; `--details` includes all keyword/bin rows. Pull/update also save readable
`progress.json` and `progress.md` in the local plan directory. `--first HOUR_ID`
selects an explicit first hour. Selection never proves hardware compatibility.
Use `--configuration HASH` when selecting a specific scientific configuration;
the default uses the first eligible package's configuration and never mixes them.
All claims still go through the existing validated-profile and approval gates.

`coordination/progress.json` and `coordination/progress.md` summarize cells by
model, primary keyword and frozen axis bin. They include hour assignments,
owners and disjoint status counts. Missing bins are reported as unknown, never
inferred. Reports carry the source registry checksum/revision; `hours.json`
remains authoritative. A stale report cannot change ownership. Private HF write
access is required for publication and claiming; read-only tokens support pulls.

Repeated unchanged result updates upload no result payloads. A plan refresh with
unchanged assignments/settings preserves hour IDs. Changed plans never replace
owned packages, and completed or failed cells are never silently rescheduled.
Input/checkpoint downloads currently retrieve whole immutable bundles, including
other-model records needed for dependency verification. They do not download the
historical archive. Transfers reuse existing verified files and preserve conflicts.

Calibration accepts the original per-model shape, or
`{"qwen38":{"configurations":{"CONFIGURATION_SHA":{...measurement fields...}}}}`.
Each package contains one configuration and keyword. New update workflows defer
missing calibration as `awaiting_calibration`; invalid supplied measurements still
fail. The original `plan` command keeps its strict missing-calibration behavior.
A pending-calibration plan is a tracker, not a runnable one-hour workload.

## Initial publication from JUPITER

`bootstrap_shared_hours.py --source DATASET --output NEW_BOOTSTRAP --since DATE`
inspects Slurm, registrations and reconciliation needs without changing scientific
records. Add `--include-job-id 1995245` for the known resume attempt. Ledger reads
can initialize their normal advisory lock files.

For publication, add `--publish --model-inputs FILE --keyword-priority FILE
--quota-evidence FILE` and, when available, `--calibration FILE`.
`model-inputs.template.json` documents the input-map shape. Replace each model's
`runtime_environment` with its actual frozen runtime values; use no guessed
model revision, setting or file. Add a model entry for every registered model.
The helper reports missing registrations; it does not create them. Judgment
registration continues through the existing exact-generation mapping pipeline.

The helper reconciles terminal legacy writers, freezes a separate mirror,
publishes verified input objects, and creates `site.json`, calibration, publication
receipts and a plan. It checks for live legacy jobs before staging and again
before publication/plan installation. Without timing evidence it publishes only
deferred work. It never repeats registration or recovery acceptance and never
submits Slurm jobs. If a plan already exists, use the saved site and `update`.
The standard `publish-plan` command with the saved operation ID resolves an
ambiguous plan-publication response.

`prepare_shared_hour_inputs.py` also exposes staging independently for multiple
registered models. Shared prompts, memberships, task shards and verified outcomes
stay intact. Frozen input bindings and original runtime values are stored in
`artifacts/shared-preparations/<hash>.json`. Site-local paths must be rebound before
execution; the source paths are provenance, not portable runtime configuration.
Never overwrite an existing mirror after its source changes.

HoreKa's cluster profile may specify `reservation: "casualnet"`. Admission and
command generation verify the actual reservation's account/user, partition,
active state and remaining validity using Slurm. Null leaves reservation selection
unset. Neither reservation membership nor `--exclusive` replaces compute-node
execution-boundary verification.
# Tracker setup while legacy jobs are queued

After legacy jobs stop, `bootstrap_shared_hours.py --publish-dispatch` audits
the source without repairing it and publishes a complete, compressed task
inventory under `coordination/dispatch/<sha256>.json.gz`, with a summary at
`coordination/dispatch.json`. Each task retains its scientific identity,
keyword, prompt, axis bin, configuration, observed completion state and cluster
routing. This metadata-only step needs no bulk-transfer quota evidence. It
refuses unresolved reconciliation actions and an already active registry.
It leaves hour packages unset until timing and frozen inputs are available;
it neither transfers result payloads nor creates runnable claims. Identical
inventory publication is idempotent. Only currently registered models appear.

`bootstrap_shared_hours.py --publish-tracker` publishes `coordination/hours.json`
with no assignments and `coordination/progress.json` / `progress.md` with current
keyword/bin observations. It accepts pending legacy jobs and records their
scheduler snapshot. Unfinished cells await reconciliation; locally verified
completions await sync. These observations are not a frozen, runnable plan.

This mode neither reconciles the source nor uploads input bundles, creates hour
packages, or submits allocations. It requires fresh complete scheduler evidence
and refuses to overwrite a registry already in use. Concurrent publication is
protected by the HF parent revision. Repeating it refreshes the observations.
Normal bootstrap publication still requires legacy jobs to have stopped.

## Audit progress

Bootstrap commands enable progress messages on stderr by default. Set
`GEODML_AUDIT_PROGRESS=1` for standalone Python sync blocks too, or `0` to
silence them. Every ten seconds, each active audit stage reports elapsed time,
its current phase and observed counters. Inventory reports tasks checked,
verified completions and blocked tasks; reconciliation reports registry and
ledger counts. Bundle uploads report files and bytes confirmed remotely,
including files already present. These counters are audit/transfer progress,
not newly generated scientific results. Totals are shown only when known.
Final JSON stays on stdout. Updating a checkout does not change a running process.

## Repairing older bundles with omitted root files

The publisher previously traversed allowlisted files as directories, omitting
`README.md` while the frozen descriptor required it. New publications include
allowlisted regular files. Existing bundle identities and plans stay immutable.
`repair_shared_input_bundle.py --bundle ID --dataset-root ROOT --journal DIR`
repairs only files required by the checksum-verified frozen descriptor but absent
from that bundle's outer manifest. First run with `--publish` against the staged
JUPITER mirror to publish the exact content-addressed objects, then without that
flag on HoreKa. Missing, conflicting, escaping or credential-shaped content fails
closed. No ledger imports, registrations, plan changes or inference occur. Later
fresh downloads of an old affected bundle need this repair too; new bundles do not.
