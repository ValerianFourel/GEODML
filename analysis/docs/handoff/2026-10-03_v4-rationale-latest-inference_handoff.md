# V4 revision rationale, latest evidence and interactive preference

Valerian asked why v4 changed, requested checking the latest inference, and
suggested Nemotron if Gemma is too slow. During the investigation Valerian
requested an interactive run and said not to use HoreKa's `cpuonly` queue,
preferring tmux. Use tmux for persistent control and a Slurm GPU allocation
for inference. Do not interpret tmux as permission to run model inference or
substantial computation on a login node. No new allocation was requested here.

Read the three latest indexed handoffs, then the v4 design trace, r2 repair
handoff, cycle guide and original Codex conversation. The explicit repair request
is in session `01a0f7a6-2a31-7112-8218-64350455ba80`, timestamp
`2026-10-02T11:09:21.724Z`, response-item line 4199. The user asked for absolute
validity against the intended task, with unchanged quality thresholds and a
bounded cycle, rather than requiring comparative superiority to v3.

The reasons and implemented changes are:

- The proctoring map assigned bare list marker `1)` / `a1w22` both to a claim
  and to an exclusion. R2 removes only the redundant marker exclusion, preserves
  original claims/roles and raw rejected outputs, and revalidates the whole map.
- Stage A must preserve entities, negation and qualifications, remain source-blind,
  and assign importance by the completed answer's substance. First list position
  and membership do not automatically imply centrality. Reordering controls test it.
- Stage B must distinguish full from partial support, read the complete supplied
  source and check background/off-target answer content before assigning zero.
  Weak selected witnesses alone do not establish that the whole source lacks support.
- Source justifications must respect fixed map roles. A fully supported secondary
  claim cannot silently become peripheral, or a major claim secondary. Substantive
  map defects return `map_issue`; labels never mechanically increase grades.
- Evaluation includes fixed-map and end-to-end repeats, constructed/list-order/role
  controls, blinded references, complete failure denominators and compute cost.

Original complete answers, sources, six ordinal anchors, masks and Gemma settings
are preserved. New task version is `source-importance-task-v4-r2`, published at
`caf8a7d3fbe69736789767efcc4551d0e0347c11`. Historical tasks still reproduce.
Original v4 introduced the source-blind answer-map stage after the v3 audit found
semantic errors despite 97.7% exact repeats. Repeatability was not correctness.

## What was actually verified

Recomputed the 10 release-artifact hashes and eight format-repair hashes in
`Downloads/si-v4-r2-release-20261002T121912Z`; all match. Re-ran the production
repair against the exact rejected output. The original fails for `a1w22` overlap;
the reproduced repaired bytes exactly match the saved repair; original claims
are unchanged and strict validation reports `eligible`. This is structural proof,
not new inference or semantic acceptance.

The last returned inference results in the original conversation remain the
earlier job 5175229 two-cell retry: one complete cell, seven scored sources of
ten, one failed map and three blocked sources. The preceding 20-cell execution
had 18 complete cells and 103 scored sources, with two failed maps. Keep these
separate diagnostic executions, not a merged acceptance dataset.

The original conversation ends its v4 execution trail with job 5175818 granted
on hkn0402 at `2026-10-02T16:36:47.086Z`, after the earlier submission was refused
for five active allocations. An existing-allocation r2 command and HTML followed.
No later Gemma execution output or report was returned. The current user's
reply preferred an interactive run but did not identify a later completed job.

Read-only HF query succeeded at current dataset revision
`73a0c123f66c7ad03c9d291c8fec9a56e247944b`. Recursively listing `reviews` found
only the old Gemma SI-v3 export
`reviews/gemma-si-v3/exports/31f3666c64e2f44d58cc4f06411f87f828ad454fee9c1b506c6c915dba77e62f.zip`.
No r2 review export was available there. No new inference evidence was downloaded.
No cluster SSH, scheduler query, model download, HF write or Slurm submission ran.

## Speed and next execution

Historical v4 warm timing was 5.87 minutes for twenty cells and 1.30 minutes for
the two-map retry; historical Gemma startup was 8.9–10.9 minutes. These do not
establish the latest r2 speed. Compare actual valid completed cells and warm
GPU-hours with allocation/startup occupancy separately before choosing a model.

The prior HoreKa Nemotron profile is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at
`bf77c3174f68ad409e1c2aa60daeb46e32d1c606`, with four A100s, BF16, TP4 and
73,728-token context. It previously ran SI-v3. The v4 preparation/check path is
currently Gemma-specific. Do not edit an existing immutable Gemma configuration
to swap models. Any Nemotron comparison must preserve the r2 task/cases/rubric,
use separately pinned judge outputs and check the installed model/runtime first.
Valerian previously authorized deleting Nemotron's HoreKa cache when switching
to Gemma; present weight availability is unknown. No new download is authorized
or performed just to make the conditional comparison possible.

The existing cycle remains bounded at 32 GPU-hours, split 16 development and
16 fresh, at most four one-hour four-A100 segments per phase. Read the current
budget and reconcile actual accounting before another interactive segment.
No budget expansion or model substitution has occurred in this turn.

Prepared `analysis/docs/horeka-si-v4-r2-inspect.sh`, a read-only block for the
already-open HoreKa shell, optionally inside tmux. It reads the registered
model/code, cycle submissions, queue status, latest report pointers, counts,
failures and timings; prints live scheduler/accounting and recent console logs.
It also reads the Qwen sender log, because the already-launched Qwen version
still targets `cpuonly` for its two audits. The user's new queue preference has
not changed that running pinned sender. Inspect its actual phase before replacing
an audit step; preserve all live allocations and submission receipts. No Qwen
code/configuration, sender process or allocation was altered here.

The inspection block passes Bash syntax and embedded Python AST checks. No
scientific code changed and no inference tests were needed. Next: Valerian
returns the inspection output; collect the complete raw r2 evidence using the
existing bounded cycle exporter, then assess maps/support/grades and choose the
next in-budget interactive segment. Latest semantic correctness remains unknown.
