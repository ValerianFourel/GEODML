# Nemotron v4 comparison targeting about twenty minutes

Valerian requested the same v4 script with Nemotron on any twenty cells, then
allowed fewer cells and clarified a target of about twenty minutes. Prepared
one eight-cell diagnostic with a thirty-minute allocation cap: one exclusive
HoreKa accelerated node, four A100s, 32 requested CPUs, full-node memory,
at most 0.5 node-hours / 2 GPU-hours. No job was submitted by Codex.

The estimate is 12-25 minutes, not a Nemotron v4 measurement. Gemma previously
needed 8.9-10.9 minutes of startup and roughly 6-7 warm minutes per twenty cells.
Eight cells leave startup and cleanup margin. The existing runtime uses the
actual Slurm deadline to stop admissions and checkpoint; twenty-minute completion
is not guaranteed. A four-cell test is cheaper but supplies less diagnostic
evidence. Model-cache restoration, if necessary, occurs on the login host before
GPU allocation and is outside the inference runtime estimate.

Added `prepare-nemotron` to `analysis/scripts/horeka_si_v4.py`. It verifies the
saved Gemma evaluation configuration and candidate freeze, selects eight cells
with seed 20261004, and copies only the required tasks with the existing verified
subset writer. Questions, answers, sources, task identities and v4 settings are
preserved. Only the judge model/revision, tokenizer path and template fingerprint
change. It runs one map-plus-source pass, with fulfilment explicitly not requested,
and writes separate results. The existing deadline, authenticated serving,
exclusive-node verifier, quota refresh and failure reporting remain in use.

Nemotron is the historical HoreKa profile:
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, revision
`bf77c3174f68ad409e1c2aa60daeb46e32d1c606`, BF16, TP4, 73,728 context,
four concurrent requests, thinking disabled, temperature zero. Its verified
cache must exist. Runtime versions must equal the saved Gemma configuration.
No model fit, GPU runtime or throughput was established by local tests.

`analysis/docs/horeka-nemotron-v4-8cells.sh` is the paste-command helper. It
accepts workspace, saved Gemma run, new output path and account as arguments.
It restores the requested pinned model only if its verified cache is absent,
using the existing quota-aware downloader. It prepares eight cells and checks
fresh storage, account queue, all user allocations, five-allocation maximum
and ten-minute observed-start spacing. It requests one salloc with
`--no-shell --immediate=30`, records intent before submission and the scheduler
receipt afterward, and starts one srun only with an unambiguous successful job
ID. It has no retry loop. Ambiguous receipts require reconciliation. It prints
saved summaries and accounting and leaves the allocation to expire naturally.
Use tmux in a separate login shell; preserve other allocations. A prepared
output directory is never silently overwritten or automatically resubmitted.

Validation: 114 focused tests passed across test_horeka_si_v4.py,
test_horeka_nemotron.py and test_si_v4_cycle.py. New tests exercise unchanged
selected task content/settings, the pinned model's routing, missing cache,
changed input bytes, incorrect model revision, count limit, no overwrite and
the thirty-minute admission path. Bash syntax and both embedded Python blocks
compile. Git whitespace checks pass. The v4 protocol implementation and
run_source_importance_judge.py have no changes relative to scientific commit
caf8a7d3fbe69736789767efcc4551d0e0347c11.

HoreKa batch/hardware documentation was fetched; actual partition and quota
checks remain part of the command. No live cluster facts were inferred from
old handoffs. Next: Valerian executes the pinned helper and returns its saved
summary/logs for comparison with the matched Gemma cells.
