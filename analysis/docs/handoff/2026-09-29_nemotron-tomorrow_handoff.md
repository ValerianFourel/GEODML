# Nemotron on HoreKa: handoff for 30 September 2026

Written 29 September. All cluster results below are user-pasted observations,
not live scheduler checks. Start tomorrow with status, not another launch.

## Where we stand

Nemotron SI-v3 passes the original five-cell structural test. Semantic quality
has not been established. A fresh 10 Qwen + 10 Llama diagnostic is prepared,
but no execution output or full judgments from that test have been received.
Do not claim it ran, failed or passed until checking saved artifacts.

| Protocol | Valid source judgments | Complete cells | Judging seconds |
| --- | ---: | ---: | ---: |
| SI-v1 | 10/29 | 0/5 | 138.4 |
| SI-v2 | 21/29 | 2/5 | 76.2 |
| SI-v3 | 29/29 | 5/5 | 35.3 |

J1 passed 5/5 in all three runs. Times exclude model startup and are not
production-corpus estimates. SI-v3 returned status passed / EXIT=0, no recorded
validation failures. Three of the five generator answers were flagged truncated;
this is separate from judge output validity.

SI-v1/v2 asked the model to copy quotations and failed exact-text validation.
SI-v3 asks for supplied answer/source passage IDs; code resolves them to exact
text and offsets. Valid IDs prove provenance, not that the passage supports the
answer. The scientific rubric and serving configuration were retained.

Concrete review concerns: the inventory example selects the title "Inventory
control system" for a substantive recommendation; topical overlap may be mistaken
for support. The crypto cell grades all sources 3, so true top-source alignment
under ties is weak evidence of ranking agreement. Inspect entailment and grading,
not only the pass count. This is diagnostic review, not scientific ground truth.

## Code and files

- Active checkout: `/Users/valerianfourel/Hamburg/GEODML_Unified/.worktrees/threehour-relaunch-fix`.
- Published inference branch: `origin/codex/pilot-continuation`.
- Exact working SI-v3 pin: `961ad0b181c70b9355db5b2366c0602d9abae675`.
  Later documentation commits do not change this inference pin.
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Serving/judging settings: seed 20260929, temperature 0, thinking disabled,
  max_tokens 640, concurrency 4. Native xgrammar 0.2.7 accepted SI-v3.
- Successful five-cell job: 5170116 on hkn0403.
- Workspace W: `/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen`.
- Environment: `$W/geodml-nemotron-env.sh`.
- Successful run: `$W/runs/nemotron-si-v3-replay-job5170116-20260929-112750`.
- Original five identities: `$W/runs/nemotron-si-smoke-20260929-0829/attempts/job5169925/trial/cells.jsonl`.
- Operator page: `/Users/valerianfourel/Hamburg/GEODML_Unified/testnemotron20.html`.
- Copyable setup: `/Users/valerianfourel/Hamburg/GEODML_Unified/testnemotron20-setup.sh`.
- Page builder: `/private/tmp/build-nemotron20.py`. Temporary files may disappear;
  the saved HTML contains the complete commands. Operator pages are outside Git.

## The 20-cell test

Review root: `$W/reviews/nemotron-si-v3-review20-961ad0b-2026092910`.
Ten fresh cells per generator, fixed sample seed 2026092910, alternating available
methods, excluding the original five. This is seeded sampling within methods,
not an unstratified random draw of all cells. Datasets: Qwen `$DS` from the env,
Llama `$W/llama-hf/dataset`. No dataset download is part of this test.

The original login command was too slow because it verified all completed
generation/trace references before sampling. The corrected page reads metadata
and ledger states, samples candidates, and verifies only enough candidates to
obtain ten valid cells per model. It prints progress and replaces invalid
candidates within the same method. Metadata scanning still takes time. Existing
frozen selections are checked and reused, never silently resampled.

Local real sealed-data fixture demonstrated identical selected cells before and
after the fix, 94 -> 42 reference checks, invalid-candidate replacement, exclusions,
method balance and no re-verification on frozen reuse. Saved HTML shell blocks,
embedded Python and copy JavaScript passed syntax checks. No cluster performance
measurement of the corrected sampler has been received.

Node step runs Qwen then Llama sequentially and reloads Nemotron between them.
Both use --no-submit inside the existing allocation. Any run directory already
present blocks replay. Never delete it to bypass that guard. A partial run needs
reconciliation and a separately approved resume, not a repeat of all 20.

Full output: `full-review/header.json`, `cells.jsonl`, `all-judgments.txt`,
`cell-01.txt` through `cell-20.txt`. Reports retain original prompts, recoverable
full answers, source inputs, raw/resolved judgments, grades, retries and failures.
Page step 3 prints two cells per batch. Inspect both summaries as well as the
report; report generation alone is not proof both inference runs succeeded.

## Approval and tomorrow's next action

Valerian approved ONE 20-cell test in an existing one-hour allocation, conditional
on at least 40 minutes remaining initially and 20 minutes before each startup.
Estimate: 15–35 minutes with two startups, one exclusive node, four A100s,
32 requested CPUs, all memory, about 1–2.33 additional GPU-hours.
This does not authorize a new allocation tomorrow, an extension or a repeat.
The page's approval token is not a fresh allocation approval. Check actual job
status; do not assume yesterday's allocation remains available.

1. Run the read-only status block below in an already-open HoreKa login shell.
2. If results exist, review them first. Ask for page step 3 batches 1–2 through
   19–20 and assess substantive support, qualifications/negation, zero grades,
   grade calibration and ties. Do not rerun completed work.
3. If only a selection exists, preserve it. If nothing exists, corrected page
   step 1 can prepare it without GPU inference.
4. If inference remains necessary and the approved existing allocation is gone,
   prepare a concrete budget for the remaining work and ask for fresh wall-time
   approval before supplying allocating commands. Do not automatically resubmit.
5. If an attempt is partial, inspect logs and saved completions before preparing
   a resume. Current page deliberately blocks repeated runs; no automatic resume.
6. Only after semantic review decide whether another protocol change or a larger
   validation is justified. Do not scale to the corpus from structural success.

Preserve live salloc shells. Strict-mode launch code belongs in a child shell.
Ctrl-C can stop the specific slow login setup; Ctrl-Z suspends it and can block
later guards. Do not cancel or close the separate GPU allocation to update code.
Keep Qwen production division unchanged. No JUPITER changes are part of this task.

## Read-only status command

The companion `.sh` contains this same block. Paste it into the HoreKa login
terminal; no SSH prefix, allocation, model loading, full-corpus scan or writes.

```bash
#!/usr/bin/env bash
# Paste into an already-open HoreKa login shell. Read-only; no inference or submission.
(
  set -eo pipefail
  date -Is
  squeue --me -o '%.18i %.38j %.10T %.12M %.12L %.20R'
  python3 - <<'PY_STATUS'
import json
from pathlib import Path
w = Path('/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen')
r = w / 'reviews/nemotron-si-v3-review20-961ad0b-2026092910'
print('REVIEW_ROOT', r)
m = r / 'selection-manifest.json'
print('FROZEN_SELECTION', m.is_file())
if m.is_file():
    print(m.read_text())
for model in ('qwen38', 'llama4'):
    run = r / 'runs' / model
    print('MODEL', model, 'RUN_EXISTS', run.exists())
    summaries = sorted(run.glob('attempts/job*/trial/summary.json'))
    if not summaries:
        print('NO_SAVED_SUMMARY: execution may be absent, active, or interrupted')
    for p in summaries:
        print('SUMMARY_PATH', p)
        print(p.read_text())
report = r / 'full-review'
print('FULL_REPORT_EXISTS', (report / 'all-judgments.txt').is_file())
print('CELL_REPORT_FILES', len(list(report.glob('cell-*.txt'))))
print('Do not restart existing runs before reconciling summaries, logs and live jobs.')
PY_STATUS
)
```

## Prompt to resume with Codex

> Resume the Nemotron HoreKa diagnostic using this handoff. First inspect current
> scheduler/artifact status; do not assume the 20-cell test ran or launch it again.
> We need full semantic review of 10 Qwen + 10 Llama judgments. Preserve the
> SI-v3 pin and frozen sample unless a concrete defect requires a change. No new
> allocation is approved. Keep commands in the existing Safari operator page,
> separated by login and node shell. Do not alter production Qwen division.
