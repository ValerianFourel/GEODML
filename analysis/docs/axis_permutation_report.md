# Update the axis-to-permutation report

Use this CPU-only report to track planned questions, saved rankings, axis fits,
and Nemotron agreement as inference artifacts arrive. It reads existing files;
it does not submit jobs, generate rankings, fetch linked pages, or edit results.

Every report is **exploratory**. Artifact validation does not validate the
judge's quality. The report always records `scientific_result=false`.

## 1. Name the inputs and axes

Create a JSON configuration with explicit paths. Relative paths resolve from the
configuration file's directory. Remove source sections that do not apply.

```json
{
  "format_version": "axis-permutation-study-config-v1",
  "study_id": "readiness-permutation-study",
  "axis_fields": {
    "target_normalized_axis_1": {
      "role": "assigned",
      "construct": "decision-readiness",
      "domain": [0, 1]
    },
    "consensus_normalized_axis_1": {
      "role": "observed",
      "construct": "decision-readiness",
      "domain": [0, 1]
    }
  },
  "fit": {
    "seed": 20260916,
    "holdout_modulus": 5,
    "ridge": 1.0,
    "min_train": 6,
    "min_test": 2
  },
  "sources": {
    "direct": [{
      "plan_manifest": "direct-plan/run_manifest.json",
      "outcome_files": ["direct-run/permutation_outcomes.jsonl"]
    }],
    "agentic": [{
      "prompts_jsonl": "cohort/pilot-prompts.jsonl",
      "task_manifest": "paired-run/run_manifest.json",
      "generator_roots": [
        "paired-run/models/qwen38",
        "paired-run/models/llama4"
      ]
    }],
    "judges": [{
      "plan_manifest": "judge-plan/run_manifest.json",
      "outcome_files": ["judge-run/outcomes.jsonl"],
      "role": "bulk"
    }]
  }
}
```

Use the names of coordinates actually recorded in your artifacts. Do not rename
a readiness coordinate to `B`. For the first-party-product-source study, use
`"B": {"role": "assigned", "construct": "first-party-source-preference", "domain": [0, 1]}`
only when `B` is the recorded assignment. Observed coordinates describe the
realized instruction; they do not define the treatment and are not confounders.

The supported direct plan is `readiness-to-permutation-plan-v1`. Its tasks must
contain the question, coordinates, frozen candidates, and model configuration.
The report reads that format without importing the experimental preparation
module. This does not add a new prompt generator or replace historical baselines.

For legacy agentic shards, omit `task_manifest` and list the actual shard output
directories under `generator_roots`. Their manifests must record the original
prompt sources and selection hash. For paired or backlog runs, use the outer
task manifest and its `models/<model>` directories, as above. This includes
not-yet-started tasks in the denominator.

If coordinates arrive in a separate agentic artifact, add `axis_jsonl`. Each
row must have `candidate_id`, the exact `question_sha256`, and finite coordinate
fields. Supported coordinate fields are `B`, names containing
`normalized_axis_`, and entries in a `readiness_coordinates` object. A late
coordinate can be added; an existing coordinate cannot silently change.

Judge sources can list multiple attempt outcome journals. `role` defaults to
`bulk`; use a separate entry with `"role": "validation"` for a validation
model. Full-conversation and legacy judgments remain separate protocols.
The adapter verifies frozen tasks, private mappings, source results, trace
hashes, model revisions, and parsed judgments before counting observations.

## 2. Generate or update the report

Run from the repository root in a Python environment with the project's CPU
dependencies, including NumPy and SciPy:

```bash
python3 analysis/scripts/report_axis_permutation_study.py \
  --config path/to/study.json \
  --output-dir path/to/axis-report
```

Rerun the same command after more results or coordinates arrive. No GPU is
needed. Use a dedicated report directory, not an inference output directory.
Keep analysis settings fixed during an incremental series. A different split,
ridge penalty, axis definition, or study ID requires a new report directory.

Each distinct update writes `snapshots/<content-hash>/` and atomically updates
`latest.json`. Identical inputs reuse the existing snapshot. The latest pointer
records the previous snapshot and the numbers of newly completed generator and
judge tasks. Concurrent report writers fail on a lock rather than overwrite
each other. This lock does not schedule or claim inference tasks.

Completed rankings, source provenance, and judgments cannot change or disappear
within the series. A truncated live journal tail is reported and not counted.
Malformed outcomes and conflicting duplicates are quarantined as input issues.
An invalid judge source stays in the planned denominator, without completion
credit. If an update loses a previously counted observation, publication stops
and the earlier snapshot remains available. Inspect the source instead of
deleting report history to suppress that check.

## 3. Read the outputs

| File in the snapshot | Contents |
| --- | --- |
| `report.md` | Coverage, fit status, held-out improvement, agreement, and limits |
| `report.json` | Full report, source hashes, issues, and immutable-observation receipts |
| `questions.jsonl` | One row per cohort/question: planned tasks, axes, rankings, fit diagnostics, predictions, and judge comparisons |
| `fits.jsonl` | Coefficients, support, training/holdout metrics, baseline, and predictions for each fit |
| `comparisons.jsonl` | Per-case direct/generator versus judge rankings and agreement metrics |
| `pending.jsonl` | Missing generator and materialized judge tasks; not a submission queue |

This short reader prints the latest coverage and an example question:

```python
import json
from pathlib import Path

root = Path("path/to/axis-report")
latest = json.loads((root / "latest.json").read_text())
snapshot = root / "snapshots" / latest["snapshot_id"]
report = json.loads((snapshot / "report.json").read_text())
print(report["generation"])
print(report["judging"])
print(report["fit_status_counts"])
print(report["fit_exclusions"])
print("New generator tasks:", latest["new_completed"])
print("New judge tasks:", latest["new_judgments"])
for question in report["questions"][:1]:
    print(question["question"])
    for task in question["tasks"]:
        print(task["model_id"], task["axes"], task["ranking"])
        print(task["fit_diagnostics"])
        print(task["judge_comparisons"])
```

An absent axis remains missing. An empty saved ranking counts as a saved
artifact but not a successful agreement. Fake-backend observations receive no
completion or fit credit. The denominator covers only the explicit source
plans, not the full 26,009-prompt rollout unless those plans are supplied.
Generator tasks without a materialized judge task are reported separately.

An absent direct plan is labeled `study_availability.direct=unavailable`, not
zero agreement. Do not substitute earlier answer-generation journals for a
fixed-candidate permutation plan.

## Prepare a cluster report

`analysis/scripts/prepare_axis_permutation_report.py` creates a configuration
from explicit existing run roots. It reads frozen input manifests, prompts,
coordinates, and judge plans, but does not scan inference result directories.
Use repeated `--legacy-generator-root` arguments for original shard directories
and repeated `--cohort-run-root` arguments for paired or backlog run directories.
Supply `--judge-plan-manifest` and `--judge-outcomes` together for a judge source.
`--output` and `--study-id` are required.

The helper verifies the selected prompt and axis artifact hashes. When an axis
map uses `text_sha256`, it checks that hash against the exact question before
writing a separate report-input sidecar with `question_sha256`. It never edits
the experiment's input files. It preserves missing observed coordinates and
refuses to overwrite a different prepared configuration. Its defaults describe
the assigned and observed first readiness coordinate, not first-party policy B.

The CPU batch entry point is
`analysis/scripts/slurm/jupiter/run_axis_permutation_report.sbatch`. Submission
must supply an approved wall-time and resources; the script has no allocation
defaults. It requires these exported values:

- `GEODML_EXECUTION_REPOSITORY` and `GEODML_EXECUTION_COMMIT`: a clean checkout
  at the published, full Git SHA.
- `GEODML_AXIS_REPORT_CONFIG` and `GEODML_AXIS_REPORT_OUTPUT`: absolute paths
  to the prepared configuration and dedicated report directory.
- `GEODML_AXIS_REPORT_CONFIG_SHA256`: the submitted configuration hash.
- `GEODML_REPORT_ATTEMPT_DIR`: a unique directory for this approved attempt.
- `GEODML_APPROVED_WALLTIME` and `GEODML_ALLOCATION_ESTIMATE`: the approved
  budget and its estimate.
- `ACL_ARR_VENV`: the cluster Python environment containing NumPy and SciPy.

The batch records `allocation.json` in the attempt directory, including resource
environment, versions, settings, input hash, start/end times and exit status.
A repeated or concurrent attempt is refused. An allocation killed by Slurm can
leave a nonterminal saved status; consult `sacct` for the scheduler outcome.

By default it also runs the server-free network-namespace diagnostic, with a
30-second timeout, and writes `network-isolation.log`. Failure does not block
CPU reporting and does not authorize inference without isolation. Set
`GEODML_AXIS_REPORT_CHECK_NETWORK=0` to omit this diagnostic. No model is loaded
and no inference endpoint is started by the report job.

Each later report allocation still requires fresh wall-time approval. Reuse the
same configuration and report output to update coverage, but use a new attempt
directory. Do not resubmit an inference run merely to refresh this report.

## Interpret an axis fit

Each fit uses one model/revision/configuration, cohort, method, engine,
condition, query, exact evidence presentation, and observed ranking length.
Direct fixed-candidate reranking is never pooled with agentic retrieval.
Mixed-resume or unknown agentic execution configurations are excluded from
fits and counted in `fit_exclusions`; their saved artifacts remain visible.

The model assigns each candidate a utility `alpha + beta * z`, where `z` is
one configured coordinate standardized using **training questions only**.
A regularized Bradley-Terry likelihood fits the observed ordered pairs. Each
task has equal total weight across its observed pairs. The objective is the
sum of task-mean pair losses plus `ridge / 2` times squared coefficients in
standardized units. Coefficients in the report are converted back to the
original coordinate units. An intercept-only model is the axis-free baseline.

The question-content hash and seed assign a fixed holdout group. Duplicate
question text cannot cross train/holdout even with different prompt IDs. The
minimum sample counts are counts of distinct eligible questions, not candidate
pairs. A fit stays `insufficient_data` until both groups meet their minimum.
A constant training coordinate is reported as `constant_axis`. Optimization
failure is not reported as a successful fit.

Read held-out log loss, Brier score, pairwise accuracy, and rank agreement.
Positive `heldout_improvement.log_loss_reduction` means the axis model predicts
observed pair order better than the baseline on that holdout. Missing candidates
are not assigned bottom ranks. Candidates without training pair support can
still appear in a regularized full-pool prediction; inspect
`training_candidate_pair_counts` before interpreting their coefficients.

These are separate, one-coordinate predictive fits, not a joint model of all
latent dimensions. They do not isolate semantic changes from surface seed `S`,
control other coordinates, provide uncertainty intervals, or identify a causal
effect. More observations can improve coverage without improving prediction.
Repeatedly inspecting holdout results is not a sequential significance test.

## Interpret Nemotron agreement

There are two comparison families:

- `source_generator_vs_judge`: the generator's saved ranking versus the judge
  of that same answer and evidence.
- `fixed_candidate_direct_vs_judge`: a separately recorded direct reranking
  versus the judge, only when question text hash, engine, generator model
  revision, URLs, titles, and evidence text match exactly. Different protocols
  and source configurations remain labeled. This is a descriptive comparison,
  not proof that the two tasks are equivalent.

No compatible direct run means **unavailable**, not disagreement. The report
counts these missing matches. It never matches URLs alone or fills missing
rank positions. Candidate presentation order is retained for fitting; cross-
protocol comparison matches unordered evidence content and labels both tasks.

Ideal relevance order and realized support order are reported separately.
Metrics include full-list exact match, first-choice match, top-k overlap
(`k = min(3, both lengths)`), and Kendall tau on jointly ranked candidates.
Each aggregate metric has its own denominator. Full-list exact match is zero
when lengths differ, even if the shared order agrees. With fewer than two
shared candidates, tau is undefined rather than zero.

The v2 full-conversation judge sees recorded model messages and ranking
positions. Its agreement is **not independent validation** of that ranking.
The v1 judge sees the frozen answer and snippets without the explicit generator
ranking, though answer wording can still reveal preferences. Neither protocol
fetches full linked web pages. A representative human check and a separate
validation/adjudication plan are still needed before treating scores as
scientific results.
