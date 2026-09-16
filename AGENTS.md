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

## Working rules

- Inspect relevant code before editing.
- Make one small, testable change at a time.
- Preserve existing datasets, outputs, schemas, and headline results.
- Do not download large datasets or model weights in Codex cloud.
- Do not run vLLM, GPU inference, or expensive experiments in Codex cloud.
- Keep JUPITER and HoreKa work in committed, reproducible Slurm scripts.
- Never submit or retry a Slurm allocation without a fresh runtime estimate and
  explicit wall-time approval from Valerian.
- Preserve every live `salloc` allocation unless Valerian explicitly asks to
  release it. Do not recommend or execute `exit`, Ctrl-D, `scancel`, termination
  of the allocation-owning process, or closure of its shell to switch context,
  update code, recover from an inference failure, or finish a task. Keep that
  shell open and use a separate login terminal for updates or monitoring.
- Keep strict-mode launch commands inside a child shell so a failed job step
  does not close the allocation-owning shell. Natural expiration of the approved
  wall-time does not authorize an extension or replacement.
- Do not make cluster-only source changes. GitHub is the handoff boundary.
- Avoid hard-coded usernames and machine-specific paths in committed code.
- Add focused tests for every new behavioral contract.

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
  Do not auto-resubmit, extend wall-time, or increase resources. Fresh allocations
  still need an estimate and Valerian's explicit approval. Running pinned jobs
  are unchanged by local source edits.

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
