# Prompt continuum: Milestone 3A policy-clause pilot

Milestone 2 exposed only five policy realizations in the deterministic template
scaffold. That generator remains the reproducible test and fallback backend,
but its finite phrase schedule is too coarse for the scientific prompt family.

Milestone 3A therefore uses an LLM **offline** to propose a small bank of policy
clauses. Each candidate is stored exactly with its raw output, model settings,
seed, specification version, and SHA-256 identity. The later reranking pipeline
will consume frozen, reviewed text; it will never generate experimental prompts
dynamically during reranking.

These candidates have only structural validation. Their semantic monotonicity,
axis purity, and equivalence across style realizations remain unestablished.
They must not be used for reranking or inference yet.

## Local dry run

From the repository root, create only the 64-row request/meta-prompt manifest:

```bash
python3 analysis/scripts/generate_policy_clause_pilot.py --mode dry-run --output-dir analysis/output/policy_clause_pilot_dry_run --master-seed 20260810
```

No provider, model, GPU, credential, query, candidate, or ranking data is used.

## Proposed HoreKa pilot command

After committing locally, pushing to GitHub, and checking out the exact SHA on
HoreKa, invoke the following **inside a site-approved Slurm wrapper**:

```bash
python3 analysis/scripts/generate_policy_clause_pilot.py --mode generate --provider local --model "$POLICY_GENERATOR_MODEL" --precision full --output-dir "$POLICY_PILOT_OUTPUT_DIR" --master-seed 20260810 --number-style-seeds 8 --number-b-values 8 --include-anchors
```

The existing repository Slurm scripts are JSC-specific, so this milestone does
not copy their partitions, accounts, modules, usernames, or paths into a HoreKa
script. The wrapper should record the checked-out Git SHA, model revision,
environment/container, resources, job ID, logs, and artifact directory.

After the job, return these files for local inspection:

- `policy_clause_requests.jsonl`;
- `policy_clause_candidates.jsonl`;
- `candidate_full_prompts.jsonl`;
- `policy_clause_pilot_report.md`;
- Slurm stdout and stderr;
- the run manifest containing Git SHA, model revision, environment, seeds, and
  allocated resources.

Do not edit generated clauses on the cluster. Any code change must be reproduced,
tested, committed, and pushed locally before another pilot run.
