# Qwen shared-hour preparation

`analysis/scripts/prepare_jupiter_qwen_hours.py` divides registered Qwen work
with the same planner and conservative timing method used for the Llama wave.
It does not register tasks, accept recovery results, allocate GPUs, reserve
packages, or submit jobs.

## Required measurement

Use a saved, successful one-hour or explicitly approved three-hour JUPITER allocation on four GH200 GPUs for
the pinned Qwen model and registered scientific configuration. A failed
bootstrap or HoreKa compatibility probe is not reference calibration.
The helper requires the measurement directory to contain:

- `preparation.json`: original dataset root, `summary.configuration_sha256`,
  and frozen profile SHA-256 in `files`.
- `runtime.json`: `SEARCH_AGENTIC_PROFILE` pointing to that frozen profile.
- `boundary.json`: verified boundary and Slurm job ID.
- `allocation.json`: saved `slurm` fields with one node and the approved limit.
  Three-hour measurements require `preparation.json.approved_walltime` to match.
- Exactly one `results/attempts/*/run_manifest.json`: positive newly committed
  completions, no reused cells or failed cells, and matching dataset root.

`prepare_jupiter_qwen_measurement.py` prepares these artifacts from existing
Qwen registrations and the saved Qwen runtime. With `--submit` and explicit
approval evidence, it submits exactly one three-hour, exclusive four-GH200 job:
Qwen member 1 of the approved five-Qwen/five-Llama wave. It does not submit the
remaining nine members or automatically retry. Preparation reconciles the
source dataset once and freezes the missing Qwen backlog with one cached
inventory pass; it does not repeat registration or recovery acceptance.
Submission requires an idle scheduler and the observed start gap. It saves
intent before submitting a held job, saves its ID before release, and prevents
duplicate submission through a dataset-level receipt. Runtime checks the
exclusive Slurm boundary before loading the model; no namespace fallback is
used. The whole backlog supplies useful work until the deadline.

These are the same measurement artifacts used by the Llama first-hour workflow.
Existing Qwen runs need to supply equivalent evidence; do not manufacture a
successful measurement from a failed run. The hour-packaging helper itself
does not prepare or launch a new measurement allocation.

The helper checks the historical job through `sacct` for successful completion
on four GH200 GPUs on JUPITER. It uses `approved_allocation_seconds / newly_committed_cells` seconds
per cell, including startup, then adds a further 300-second reserve to each
package. Packages retain keyword ordering and complete prompt groups. An
indivisible group exceeding the estimate is explicitly reported as oversized.
These are estimates, not guarantees of completion within one hour, particularly
on HoreKa.

## Interface

Required arguments are `--site`, `--measurement`, `--output`, and `--repo-id`.
Use the current JUPITER shared-hour site, the measurement directory above, a
new output directory, and the private coordination repository respectively.
Default behavior prepares local `plan.json`, `calibration.json`, `site.json`,
and `summary.json`. Add `--publish` to publish using conflict-checked registry
updates. Publication does not approve or submit any allocation.

The current published plan supplies existing calibration and the immutable
input bundle. Qwen calibration is added for its measured configuration; Llama
calibration is retained. One inventory pass reuses cached verification.
Unsynchronized local completions block planning until the existing result-sync
workflow publishes them. No dataset bundle is uploaded or downloaded here.
Plan publication still transfers the plan and registry metadata.

Owned packages and historical completions are preserved by the existing
planner. Unowned unfinished work can be repacked. A concurrent registry update
invalidates a stale plan rather than overwriting claims. Preserve existing
output directories; the existing `manage_agentic_hours.py publish-plan` command
can publish a prepared plan without repeating preparation, provided its base
registry is still current. Use the newly written site configuration for later
replanning so that the Qwen calibration is retained.

Package IDs are assigned by the planner across all models. Use the published
Qwen IDs when reserving work; do not assume their suffixes start at 1.
