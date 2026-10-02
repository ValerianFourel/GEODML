# AGENTS.md

## Project mission

GEODML studies how page properties and prompt policy influence LLM reranking.
The current ACL ARR experiment measures position bias, semantic displacement,
document ranking, answer generation, and citation use under natural, ablated,
and shuffled document conditions.

## Scientific invariants

- Treat assigned prompt-policy variables as experimental variables.
- Treat page-feature effects as observational unless page content is manipulated.
- Do not call prompt embeddings confounders.
- Do not infer scientific results from mocked data, smoke tests, or pilot plumbing.
- Keep ranking tasks, queries, candidates, evidence, output sizes, and schemas fixed
  when comparing prompt-policy conditions.
- Preserve deterministic seeds, stable identifiers, source hashes, and model revisions.
- Preserve the historical neutral and biased prompt pipelines.

## Session start and end

- At the start of every session, read the last three handoffs in the index of
  `analysis/docs/handoff/README.md` (newest last) before planning or acting,
  and recheck their cluster facts; they are pasted evidence, not live state.
- At the end of every session, write a new dated handoff in that folder
  (`YYYY-MM-DD_<topic>_handoff.md`) and append it to the index.

## Working rules

- Inspect relevant code before editing.
- Make one small, testable change at a time.
- Preserve existing datasets, outputs, schemas, and headline results.
- Do not download large datasets or model weights in Codex cloud.
- Do not run vLLM, GPU inference, or expensive experiments in Codex cloud.
- Keep JUPITER and HoreKa work in committed, reproducible Slurm scripts.
- For requested cluster work, prepare and submit necessary Slurm allocations,
  including finite retries/resumes and replacements, without asking Valerian
  for confirmation before each submission or separate per-job wall-time approval.
  Present and record a fresh runtime/resource estimate and chosen wall-time first.
  Reconcile saved work and ownership before retries. Stay within the requested
  scope, declared finite budget and resource limits. This policy replaces
  per-submission confirmation requirements in HPC skills for this project.
- Preserve every live `salloc` allocation unless Valerian explicitly asks to
  release it. Do not recommend or execute `exit`, Ctrl-D, `scancel`, termination
  of the allocation-owning process, or closure of its shell to switch context,
  update code, recover from an inference failure, or finish a task. Keep that
  shell open and use a separate login terminal for updates or monitoring.
- Keep strict-mode launch commands inside a child shell so a failed job step
  does not close the allocation-owning shell. Do not extend a live allocation
  without explicit instruction. After natural expiration, a finite replacement
  within the requested scope/budget may proceed after reconciliation and admission.
- Do not make cluster-only source changes. GitHub is the handoff boundary.
- Avoid hard-coded usernames and machine-specific paths in committed code.
- Add focused tests for every new behavioral contract.

## Default one-hour Slurm limit

- New batch jobs and interactive allocations default to `01:00:00` and must
  not exceed one hour unless Valerian explicitly approves a longer duration
  for that specific allocation.
- The one-hour limit includes environment setup, model loading, warmup,
  inference, retries, drain, and checkpointing.
- The requested cluster task authorizes in-scope submissions within the declared
  finite budget; the one-hour limit does not require a new confirmation each time.
- Ask before increasing resources, exceeding the finite budget or expanding scope.
  Do not create unlimited retry, requeue or chained-allocation loops. Carry
  unfinished identities into finite planned segments without repeating completed
  work. Preserve live allocations and their existing end times.

## Default homogeneous-wave scheduling

### Shared hours across clusters

- Prefer HoreKa for Qwen; keep Qwen eligible on JUPITER too. Llama and Nemotron
  run on JUPITER. Require an explicit model choice for each run. Routine result
  updates preserve plan IDs; replan only on explicit request. Historical search,
  HTML and paid SEO archives are separate from current scientific completion.
- JUPITER is the chief setup. HoreKa is the secondary setup. JUPITER publishes
  the frozen population, GH200 timing references, and replacement hour plans.
  Either cluster may independently claim compatible published hours through the
  private Hugging Face coordination registry.
- An hour is a fixed work package sized against one JUPITER node with four
  GH200 GPUs. It is not a Slurm allocation. A HoreKa package can span multiple
  allocations within the requested scope and finite budget. Record actual A100
  resource use separately from estimated JUPITER-equivalent work.
- Apply the five-allocation limit and minimum ten-minute observed-start gap
  separately on each cluster, counting batch and interactive allocations together.
  All existing wall-time, finite-wave, scope/budget, and live-allocation protections
  apply equally to JUPITER and HoreKa.
- Keep one model per package/allocation and keyword-first ordering. Batch selects
  from the front; interactive selects from the back. A specifically selected hour
  runs first. Spillover must belong to the explicitly approved finite hour list.
- Use conflict-checked HF reservations and unique cluster/attempt writer IDs.
  Filesystem locks coordinate only workers sharing the same cluster filesystem.
  Never run legacy workers over tasks assigned through independent HF claims.
- Network outages do not expire ownership. Finish already reserved work locally;
  claim no new work offline. Confirm the owning allocation is terminal and
  reconcile saved artifacts before releasing or reassigning its hours.
- HF completion requires verified task identities and published record references.
  Slurm success, elapsed time, and upload counts do not establish completion.
- Replanning replaces only released unfinished work. Preserve owned packages,
  historical plans, completed cells, frozen scientific settings, and task IDs.
- HoreKa runs only validated four-A100 serving profiles preserving model revision,
  precision, context and scientific settings. Leave incompatible models on JUPITER;
  do not silently quantize, offload, change models, or request multiple nodes.
- Transfer sealed shards incrementally using a finite login/transfer-host helper.
  Keep model caches and filesystem locks out of HF. Do not remove unsynchronized
  scientific data. Stop admissions on storage failures or stale quota evidence.
- Shared-hour code and configuration live in Git. Site paths and setup scripts
  are configured externally. See `analysis/docs/agentic_shared_hours.md`.

### Existing wave rules

- Every finite inference wave uses exactly one model role: Qwen, Llama, or
  Nemotron. Do not alternate model roles within a wave.
- Unless Valerian explicitly approves otherwise, allow at most five concurrent
  GEODML experiment allocations per cluster. Count batch and interactive allocations
  together, and count an allocation once regardless of its `srun` steps.
- Target a minimum ten-minute gap between observed allocation starts. Enforce
  the gap before releasing another allocation; do not reserve a GPU node merely
  to sleep. Submission spacing and array throttles do not prove start spacing.
- A wave contains a finite approved number of one-hour segments, for example
  twenty Qwen segments admitted five at a time. Do not silently add segments,
  retries, another model role, or a larger resource budget.
- Batch workers traverse the frozen keyword priority from the front.
  Interactive workers traverse it from the back. Both finish eligible work in
  the current keyword before advancing and use the same durable task ledger.
- Preserve historical plans and running assignments when an incremental audit
  replans unstarted segments. Planning and audit commands default to dry-run.

## Allocation-filling inference policy

- Default every new inference `sbatch` to useful throughput across its approved
  frozen backlog, not a small 12/24-case smoke cap. Apply this to generator,
  post-processing, bulk judge, validation and adjudication jobs on every cluster.
- Keep the model loaded and admit eligible missing tasks continuously within the
  approved allocation. Checkpoint each completed task. Never repeat completed
  tasks, change scientific settings, or retry known failures indefinitely just
  to consume GPU time.
- Use Slurm's actual allocation end time, including startup, with a short
  admission/drain/cleanup margin. Record deadline checkpoints separately from
  failures and full completion. A new allocation must not reset task identity.
- Size and freeze a sufficiently large backlog before submission. If eligible
  work runs out, report queue exhaustion; do not invent tasks or busy-wait.
  Fixed-size historical pilots remain explicit exceptions, not default launchers.
- Full utilization is a scheduling objective, not a promise of 100% GPU activity.
  Finite in-scope retries/resumes need refreshed estimates and admission checks,
  not a new per-submission confirmation. Do not extend live allocations, increase
  resources or expand the declared budget without explicit approval. Running
  pinned jobs are unchanged by local source edits.

## Cloud environment

Run the repository setup with:

```bash
bash .codex/setup.sh
```

Use the repository virtual environment explicitly because setup and agent commands
run in separate shells:

```bash
.venv/bin/python -m pytest -q analysis/tests/<focused_test>.py
```

For a full CPU analysis test pass:

```bash
.venv/bin/python -m pytest -q analysis/tests
```

Large datasets and the downloaded open-weight models are not present in Codex
cloud. Use synthetic fixtures for unit tests. Treat missing external artifacts as
an execution-boundary issue, not permission to fabricate scientific outputs.

## Completion report

Report files changed, behavior implemented, tests run, assumptions, unresolved
issues, and the smallest sensible next step.


### Maintained inference execution boundary

- JUPITER shared-hour, wave batch, interactive and bootstrap entry points must
  explicitly select `GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1` after environment
  setup and preserve it in saved runtime configurations.
- Verify whole-node exclusivity with the existing Slurm verifier on the allocated
  compute host before bootstrap preparation or model loading. Fail closed;
  these paths must never invoke `unshare` or fall back to namespace creation.
  Preserve authentication, loopback binding and transport restrictions.
- Validate HoreKa's explicitly configured boundary separately. Do not globally
  disable isolation or change unrelated launchers.
- Resume prepared bootstrap backlogs from existing artifacts after fresh
  reconciliation; do not repeat registration, recovery acceptance or overwrite
  plans. Job 1994448's counts are historical, not current progress. Its isolation
  failure alone is not evidence of disk exhaustion or GPU OOM.
