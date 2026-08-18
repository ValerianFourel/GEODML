# 20k four-judge readiness annotations with explicit abstention

This run re-annotates the combined exact-unique corpus of at least 20,000
prompts using four canonical judge models and the versioned
`decision-readiness-ordinal-abstention-v2` rubric.

The v2 rubric separates three outcomes:

- `rating`: all readiness scores and an applicable category are present;
- `not_applicable`: the text is outside the decision/action-readiness construct;
- `dont_know`: the construct is relevant but the text cannot be rated
  defensibly without inventing information.

For `not_applicable` and `dont_know`, all five readiness scores and `category`
are JSON `null`. Ambiguity, confidence, and the short reason remain mandatory.
The original v1 rubric, tasks, and outputs are not changed.

## Persistence and short allocations

Every accepted model response is immediately written to an atomic per-task
JSON cache in project storage. Rejected attempts are also written atomically.
The slice runner has an internal deadline below the Slurm walltime. At that
deadline it terminates the active inference process, retains all completed task
caches, records the checkpoint, and exits successfully. A later slice uses
`--resume`, validates each cached response against the frozen v2 task, and only
generates missing tasks. Fully completed model stages are skipped before model
loading.

This design permits short repeated allocations. The final walltime is not
hard-coded in the SBATCH file and must be approved explicitly for every slice.

## 1. Audit the previous run before choosing a walltime

This operation is read-only and does not allocate resources:

```bash
source "$HOME/geodml_setup.sh"

bash analysis/scripts/slurm/jupiter/audit_readiness_incremental_four_judge.sh
```

Return the complete output before preparing an allocation. Runtime and
remaining GPU-hours must be recalculated from those observed durations and
cache counts.

## 2. Prepare the v2 task bank without allocating

Check out the exact committed revision first. Then run:

```bash
source "$HOME/geodml_setup.sh"

export GEODML_EXPECTED_COMMIT="REPLACE_WITH_EXACT_COMMIT"

bash analysis/scripts/slurm/jupiter/prepare_readiness_20k_abstention_four_judge.sh
```

The preparation must report at least 20,000 unique prompts, four identical
judge-slot item sets, and the explicit `dont_know` contract. It writes
`launch-environment.txt` but does not submit a Slurm job.

## 3. Approve each allocation separately

After the audit, record the proposed walltime, internal checkpoint deadline,
runtime assumptions, requested resources, and estimated GPU-hours. The
allocating command is intentionally omitted from this runbook until the user
approves the specific slice walltime.

The fixed resource shape is one JUPITER booster node with four GH200 GPUs and
16 CPUs. Each slice records the approved walltime and its estimate inside
`judge-queue/slices/<job-id>/`.
