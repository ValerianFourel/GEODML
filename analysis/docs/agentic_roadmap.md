# Agentic experiment roadmap

Open `agentic_experiment_roadmap.html` in a browser. It is a self-contained,
offline dashboard: it never connects to Slurm, submits jobs, or downloads models.
Its initial values come from the pasted cluster reports dated 2026-09-15. They
are historical observations, not current scheduler state.

## Refresh the artifact snapshot

From the pinned repository in a cluster login shell, run the read-only exporter:

```bash
python3 analysis/scripts/report_agentic_roadmap_snapshot.py \
  --run-root "$GEODML_STUDY_ROOT" \
  --pilot-root "$GEODML_STUDY_ROOT/judge-pilots/nemotron30b-24-v1" \
  --job 1821028 --job 1821158
```

Set `GEODML_STUDY_ROOT` to the existing 500-prompt study directory. Paste the
JSON into the dashboard's snapshot import. The exporter reads existing files
and optional Slurm accounting; it does not perform inference. Its shard counts
describe that study only, not future wave directories. Include the actual job
IDs of interest when requesting accounting.

## Read the estimates

- The 500-prompt study has 6,000 generator cells per model: two methods, two
  engines, and three conditions for each prompt.
- The planned 26,009-prompt rollout has 312,108 generator cells per model.
- Existing pilot-study artifacts count toward the full rollout only after a
  provenance and compatibility audit. The dashboard makes reuse an explicit
  assumption rather than silently applying it.
- Qwen's historical throughput range is not a Llama throughput measurement.
  Llama and judge estimates require their own measured rates.
- One bulk judgment per generator cell is a planning assumption. Validation
  and adjudication workload is not yet frozen, so no whole-experiment finish
  time is established.
- Artifact completion does not establish answer quality, scientific validity,
  or the fraction of compute consumed. The separate 24-case Nemotron pilot is
  a plumbing test, not production judging completion.

## Next requested benchmark

The next requested experiment compares Qwen3.8 and Llama4 on the same new prompt
cohort, excluding the previous 500 prompts. Its results belong in a separate
run directory; they must not be counted as filling the older study's missing
Llama cells.

Keep completion scans model-specific. Generator cell IDs alone do not encode
the model identity; scanning Qwen's completed cells for a Llama wave would
incorrectly suppress Llama work. Freeze each wave before starting workers and
create a new missing-only wave when changing worker count.

The existing runner checkpoints completed cells individually. A Slurm hard
timeout can still interrupt active calls. A timeout is not proof that saved
cells were lost, and saved cells do not prove that the entire queue finished.

Allocation approval and the cluster submission are separate from this
dashboard. Never use a refresh command as a submission command.

## Paired one-hour trial

After approval, `submit_agentic_paired_trial.py` prepares the shared new cohort
and submits one Qwen job and one Llama job. Both jobs request four GPUs for one
hour, with requeue disabled. The helper requires an explicit `--submit`, the
recorded `GEODML_APPROVED_WALLTIME=01:00:00`, and an allocation estimate.

Use one fixed output directory. The helper refuses an existing directory,
including one left by a partial preparation or submission. It records each
accepted job ID immediately. If submission is uncertain or only one job is
accepted, inspect Slurm and the submission records before attempting anything
else. Do not delete the directory or change its name to repeat submission.

`report_agentic_paired_trial.py --run-root TRIAL_ROOT --json` reads the saved
cells, remaining queue, complete 12-cell prompt groups, factor counts, and
Slurm accounting. Import its JSON into the roadmap. It updates only the new
trial panel, not the earlier 500-prompt study. The collector performs no
inference or submission.

The one-hour trial retains the established model settings: Qwen final token
limit 2,048 with thinking disabled, and Llama final token limit 4,096. Both use
four concurrent requests and up to twelve active cells. This is a throughput
measurement of those settings, not a controlled comparison of equal token
budgets or answer quality.

The generic worker disables nounset during module setup, takes a per-output
writer lock, and records allocation start and terminal events. A hard kill
can prevent the terminal event; Slurm accounting remains authoritative.
