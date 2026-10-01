# Gemma 4 SI-v4 monitoring commands, 1 October 2026

Valerian requested completed/failed/running counts and an hourly forecast of new
job starts in the existing launch HTML. Updated `analysis/docs/horeka-si-v4.html`
and its identical workspace-root copy. Launch commands and execution pins stay
unchanged. Added navigation links and three copyable monitoring blocks for a
separate, already-open HoreKa login shell:

- Task counts per attempt and each of the five passes, from read-only SQLite
  indexes. Successful and failed sealed results are separated using `result.ok`;
  both use `state=done` in the existing runner. Saved/unsealed, running, pending,
  map-dependent waiting, blocked and unknown states remain distinct. Missing or
  corrupt indexes are explicit, with expected inventory counts where available.
- Slurm state counts for all user jobs and Gemma v4 separately. Live queue and
  accounting since 1 October are separate views; `sacct -X` excludes steps.
- Pending-job start estimates from `squeue --start`, grouped by hour in the login
  host timezone, with UTC offsets and job details. Unknown and stale estimates
  remain separate; these are changeable predictions, not launch guarantees.

The log block now includes console logs and bounds each log read to 16 KiB and
35 lines. Fixed the command-body grid column so long commands scroll inside
their code blocks without widening the mobile page.

Validation: extracted all eight HTML command blocks and checked Bash syntax,
embedded Python and page JavaScript. Executed the monitoring Python against
temporary SQLite and Slurm-output fixtures, covering outcomes, inventory
mismatch, unknown/corrupt/missing data, scheduler failures, hour grouping and
unknown/stale starts. Verified the reader leaves the fixture database unchanged
and does not create missing indexes. Checked bounded logs and confirmed launch
commands are unchanged. Playwright passed navigation and actual clipboard-text
checks for all four monitoring sections, mobile width and absence of page errors.
Desktop/mobile screenshots were inspected. Temporary proof scripts and images
are under `/private/tmp/verify-gemma-v4-status-*` and
`/private/tmp/gemma-v4-start-forecast-*`.

Read the last three indexed handoffs; cluster observations remain historical.
No SSH, Slurm query, allocation or inference ran here. No scientific result is
established by local fixtures. Next step: run the new blocks in the HoreKa login
shell and interpret the returned counts and forecast. No cluster code update is
needed to use these standalone monitoring commands.
