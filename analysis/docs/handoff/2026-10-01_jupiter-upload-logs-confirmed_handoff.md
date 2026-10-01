# JUPITER upload logs confirmed, 1 October 2026

Valerian returned the status/log commands from `jpbl-s02-01`. These are pasted
cluster observations, supplementing the independent fixed-revision HF audit in
`2026-10-01_jupiter-hf-verification_handoff.md`. No cluster operation was executed
by Codex on this turn.

All four saved uploader logs report success:

- General archive `run-20260930-042645.log`, last modified 30 September 07:13:45:
  `ARCHIVE_COMPLETE 12/12`, `RUN_EXIT=0 LOG_EXIT=0`. A warning concerns missing
  README YAML metadata; it did not fail the upload.
- Activations `activations-upload-20260929-1733.log`, last modified 29 September
  17:54:44: `REMOTE 8022 files; missing or different: 0`,
  `ACTIVATIONS_UPLOAD_COMPLETE 8020 files, 551.7 GB`, `UPLOAD_EXIT=0`.
  This supplies the uploader's local-to-remote filename/size comparison at that
  time. It is not a fresh local inventory or a full content restoration test.
- GeoAxis prompts `geoaxis-26k-upload-20260929-1621.log`, last modified
  29 September 16:22:36: `GEOAXIS_26K_COMPLETE 271 files, 9.66 GB verified`,
  `UPLOAD_EXIT=0`.
- Readiness axis `geoaxis-axis-upload-20260929-1650.log`, last modified
  29 September 16:53:51: `GEOAXIS_AXIS_COMPLETE 952 files, 1.28 GB verified`,
  `UPLOAD_EXIT=0`.

The timestamps above are as printed by the cluster log check, with no timezone
in the pasted output. Export counts differ from total HF file counts because
of repository metadata files. These results agree with the remote audit.
No repeat upload is indicated. Scope remains the selected export plans; general
archive exclusions prevent a claim that every arbitrary JUPITER file was saved.

tmux shows `geodml-agentic-resume` and `hf-archive` panes running Bash. No uploader
process is visible in this host's pasted process list. An approximately 17-day-old
`salloc` process, PID 1293803, remains. No squeue/sacct output was included, so
its allocation state and cluster-wide idleness are unresolved. Preserve it and
all live shells; an idle Bash or old process does not authorize termination.

The home-directory filename listing identifies pointer/env/log files but proves
neither their current contents nor inclusion in the archive. No new data/code
changes, inference, deletion, upload or allocation were made. Remaining checks,
if needed, are Slurm state, unpushed JUPITER code and unpublished/later-created
files outside the verified export scope.
